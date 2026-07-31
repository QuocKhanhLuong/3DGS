"""Context-only, hashable intensity preprocessing for T1-C."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Literal

import torch

from .io import DecodedObservation


@dataclass(frozen=True)
class NormalizationConfig:
    policy: Literal["zscore", "identity"] = "zscore"
    epsilon: float = 1e-6
    unseen_modality_policy: Literal["reject", "identity"] = "reject"

    def __post_init__(self) -> None:
        if self.policy not in ("zscore", "identity"):
            raise ValueError("normalization policy must be zscore or identity")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("normalization epsilon must be positive and finite")
        if self.unseen_modality_policy not in ("reject", "identity"):
            raise ValueError("unseen_modality_policy must be reject or identity")

    def to_dict(self) -> dict[str, object]:
        return {
            "epsilon": self.epsilon,
            "policy": self.policy,
            "unseen_modality_policy": self.unseen_modality_policy,
        }

    @property
    def config_hash(self) -> str:
        return _hash(self.to_dict())


@dataclass(frozen=True)
class ModalityNormalization:
    modality_id: str
    mean: float
    scale: float
    valid_pixel_count: int

    def __post_init__(self) -> None:
        if not self.modality_id or not math.isfinite(self.mean) or not math.isfinite(self.scale):
            raise ValueError("normalization parameters must be named and finite")
        if self.scale <= 0.0 or self.valid_pixel_count <= 0:
            raise ValueError("normalization scale and valid_pixel_count must be positive")


@dataclass(frozen=True)
class PreprocessingRecord:
    policy_id: str
    fitted_from_context_ids: tuple[str, ...]
    modality_parameters: tuple[ModalityNormalization, ...]
    parameters_hash: str
    config_hash: str
    unseen_modality_policy: str

    def __post_init__(self) -> None:
        if self.policy_id not in ("zscore", "identity"):
            raise ValueError("unknown preprocessing policy_id")
        if not self.fitted_from_context_ids or len(set(self.fitted_from_context_ids)) != len(self.fitted_from_context_ids):
            raise ValueError("preprocessing must bind unique context observations")
        if not self.modality_parameters:
            raise ValueError("preprocessing requires at least one fitted modality")
        if len({item.modality_id for item in self.modality_parameters}) != len(self.modality_parameters):
            raise ValueError("preprocessing modality parameters must be unique")


@dataclass(frozen=True)
class NormalizedObservation:
    source: DecodedObservation
    image: torch.Tensor
    valid_mask: torch.Tensor
    preprocessing_hash: str

    def __post_init__(self) -> None:
        if self.image.shape != self.source.image.shape or self.valid_mask.shape != self.source.valid_mask.shape:
            raise ValueError("normalized tensors must preserve decoded topology")
        if self.valid_mask.dtype is not torch.bool or self.image.dtype != self.source.image.dtype:
            raise ValueError("normalized tensors must preserve mask and image dtypes")
        if not bool(torch.isfinite(self.image).all()):
            raise ValueError("normalized image must be finite")


def _hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def fit_preprocessing(
    observations: Iterable[DecodedObservation],
    *,
    context_ids: Iterable[str],
    config: NormalizationConfig | None = None,
) -> PreprocessingRecord:
    """Fit statistics from exactly the declared context observations."""

    resolved = tuple(observations)
    declared = tuple(sorted(context_ids))
    if not resolved or tuple(sorted(item.observation_id for item in resolved)) != declared:
        raise ValueError("normalization inputs must equal the complete declared context set")
    if len(set(declared)) != len(declared):
        raise ValueError("context_ids must be unique")
    config = config or NormalizationConfig()
    parameters: list[ModalityNormalization] = []
    for modality_id in sorted({item.modality_id for item in resolved}):
        values = torch.cat(
            [item.image[item.valid_mask].detach().to(dtype=torch.float64) for item in resolved if item.modality_id == modality_id]
        )
        if values.numel() == 0 or not bool(torch.isfinite(values).all()):
            raise ValueError("context normalization received no finite legal values")
        if config.policy == "identity":
            mean, scale = 0.0, 1.0
        else:
            mean = float(values.mean().cpu())
            scale = max(float(values.std(unbiased=False).cpu()), config.epsilon)
        parameters.append(ModalityNormalization(modality_id, mean, scale, int(values.numel())))
    canonical = [item.__dict__ for item in parameters]
    return PreprocessingRecord(
        policy_id=config.policy,
        fitted_from_context_ids=declared,
        modality_parameters=tuple(parameters),
        parameters_hash=_hash(canonical),
        config_hash=config.config_hash,
        unseen_modality_policy=config.unseen_modality_policy,
    )


def apply_preprocessing(record: PreprocessingRecord, observation: DecodedObservation) -> NormalizedObservation:
    """Apply frozen context-derived statistics to context or revealed target."""

    if not isinstance(record, PreprocessingRecord) or not isinstance(observation, DecodedObservation):
        raise TypeError("record and observation must use T1-C data contracts")
    parameter = next((item for item in record.modality_parameters if item.modality_id == observation.modality_id), None)
    if parameter is None:
        if record.unseen_modality_policy == "reject":
            raise ValueError("target modality has no context-derived normalization record")
        mean, scale = 0.0, 1.0
    else:
        mean, scale = parameter.mean, parameter.scale
    image = (observation.image - observation.image.new_tensor(mean)) / observation.image.new_tensor(scale)
    image = torch.where(observation.valid_mask, image, torch.zeros_like(image))
    preprocessing_hash = _hash(
        {
            "config_hash": record.config_hash,
            "fitted_from_context_ids": record.fitted_from_context_ids,
            "parameters_hash": record.parameters_hash,
        }
    )
    return NormalizedObservation(observation, image, observation.valid_mask, preprocessing_hash)
