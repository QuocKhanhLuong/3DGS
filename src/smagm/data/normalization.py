"""Context-only, hashable intensity preprocessing for legal T1-C episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Iterable, Literal, Mapping

import torch

from .io import DecodedObservation


class DegenerateNormalizationError(ValueError):
    """Raised when a configured context-scale policy rejects an episode."""


def _hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _mask_hash(mask: torch.Tensor) -> str:
    return hashlib.sha256(mask.to(dtype=torch.uint8).detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class FrozenPopulationStatistic:
    """A declared per-modality scale that never consults a target image."""

    center: float
    scale: float
    source_statistic_hash: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.center) or not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("population center and scale must be finite with positive scale")
        if len(self.source_statistic_hash) != 64 or any(char not in "0123456789abcdef" for char in self.source_statistic_hash):
            raise ValueError("source_statistic_hash must be a SHA-256 digest")

    def to_dict(self) -> dict[str, object]:
        return {"center": self.center, "scale": self.scale, "source_statistic_hash": self.source_statistic_hash}


@dataclass(frozen=True)
class NormalizationConfig:
    policy: Literal["zscore", "identity"] = "zscore"
    epsilon: float = 1e-6
    minimum_context_scale: float = 1e-3
    degenerate_scale_policy: Literal["reject_episode", "identity_scale", "frozen_population_scale"] = "identity_scale"
    frozen_population_statistics: Mapping[str, FrozenPopulationStatistic] = field(default_factory=dict)
    unseen_modality_policy: Literal["reject"] = "reject"

    def __post_init__(self) -> None:
        if self.policy not in ("zscore", "identity"):
            raise ValueError("normalization policy must be zscore or identity")
        for name in ("epsilon", "minimum_context_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.degenerate_scale_policy not in ("reject_episode", "identity_scale", "frozen_population_scale"):
            raise ValueError("unknown degenerate_scale_policy")
        if self.unseen_modality_policy != "reject":
            raise ValueError("the T1-C reference rejects unseen target modalities")
        statistics = dict(self.frozen_population_statistics)
        for modality, statistic in statistics.items():
            if not isinstance(modality, str) or not modality or not isinstance(statistic, FrozenPopulationStatistic):
                raise TypeError("frozen_population_statistics must map modality IDs to FrozenPopulationStatistic")
        object.__setattr__(self, "frozen_population_statistics", MappingProxyType(dict(sorted(statistics.items()))))

    def to_dict(self) -> dict[str, object]:
        return {
            "degenerate_scale_policy": self.degenerate_scale_policy,
            "epsilon": self.epsilon,
            "frozen_population_statistics": {key: value.to_dict() for key, value in self.frozen_population_statistics.items()},
            "minimum_context_scale": self.minimum_context_scale,
            "policy": self.policy,
            "unseen_modality_policy": self.unseen_modality_policy,
        }

    @property
    def config_hash(self) -> str:
        return _hash(self.to_dict())


@dataclass(frozen=True)
class ModalityNormalization:
    modality_id: str
    center: float
    scale: float
    valid_pixel_count: int
    fallback_reason: str | None
    minimum_context_scale: float
    source_statistic_hash: str | None = None

    @property
    def mean(self) -> float:
        """Compatibility alias for callers that previously used ``mean``."""
        return self.center

    def __post_init__(self) -> None:
        if not self.modality_id or not math.isfinite(self.center) or not math.isfinite(self.scale):
            raise ValueError("normalization parameters must be named and finite")
        if self.scale <= 0.0 or self.valid_pixel_count <= 0 or self.minimum_context_scale <= 0.0:
            raise ValueError("normalization scale, threshold, and valid_pixel_count must be positive")
        if self.source_statistic_hash is not None and (
            len(self.source_statistic_hash) != 64 or any(char not in "0123456789abcdef" for char in self.source_statistic_hash)
        ):
            raise ValueError("source_statistic_hash must be a SHA-256 digest when present")

    def to_dict(self) -> dict[str, object]:
        return {
            "center": self.center,
            "fallback_reason": self.fallback_reason,
            "minimum_context_scale": self.minimum_context_scale,
            "modality_id": self.modality_id,
            "scale": self.scale,
            "source_statistic_hash": self.source_statistic_hash,
            "valid_pixel_count": self.valid_pixel_count,
        }


@dataclass(frozen=True)
class PreprocessingRecord:
    policy_id: str
    fitted_from_context_ids: tuple[str, ...]
    modality_parameters: tuple[ModalityNormalization, ...]
    parameters_hash: str
    config_hash: str
    unseen_modality_policy: Literal["reject"]
    degenerate_scale_policy: str
    minimum_context_scale: float

    def __post_init__(self) -> None:
        if self.policy_id not in ("zscore", "identity"):
            raise ValueError("unknown preprocessing policy_id")
        if not self.fitted_from_context_ids or len(set(self.fitted_from_context_ids)) != len(self.fitted_from_context_ids):
            raise ValueError("preprocessing must bind unique context observations")
        if not self.modality_parameters:
            raise ValueError("preprocessing requires at least one fitted modality")
        if len({item.modality_id for item in self.modality_parameters}) != len(self.modality_parameters):
            raise ValueError("preprocessing modality parameters must be unique")
        if self.unseen_modality_policy != "reject":
            raise ValueError("the T1-C preprocessing record rejects unseen target modalities")

    @property
    def record_hash(self) -> str:
        return _hash(
            {
                "config_hash": self.config_hash,
                "degenerate_scale_policy": self.degenerate_scale_policy,
                "fitted_from_context_ids": self.fitted_from_context_ids,
                "minimum_context_scale": self.minimum_context_scale,
                "modality_parameters": [item.to_dict() for item in self.modality_parameters],
                "parameters_hash": self.parameters_hash,
                "policy_id": self.policy_id,
                "unseen_modality_policy": self.unseen_modality_policy,
            }
        )


@dataclass(frozen=True)
class NormalizedObservation:
    source: DecodedObservation
    image: torch.Tensor
    valid_mask: torch.Tensor
    preprocessing_hash: str
    input_content_hash: str

    def __post_init__(self) -> None:
        if self.image.shape != self.source.image.shape or self.valid_mask.shape != self.source.valid_mask.shape:
            raise ValueError("normalized tensors must preserve decoded topology")
        if self.valid_mask.dtype is not torch.bool or self.image.dtype != self.source.image.dtype:
            raise ValueError("normalized tensors must preserve mask and image dtypes")
        if not bool(torch.isfinite(self.image).all()):
            raise ValueError("normalized image must be finite")
        for name in ("preprocessing_hash", "input_content_hash"):
            value = getattr(self, name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a SHA-256 digest")


def _resolve_degenerate_scale(
    *,
    modality_id: str,
    center: float,
    observed_scale: float,
    valid_pixel_count: int,
    config: NormalizationConfig,
) -> ModalityNormalization:
    if config.policy == "identity":
        return ModalityNormalization(modality_id, 0.0, 1.0, valid_pixel_count, None, config.minimum_context_scale)
    if observed_scale >= config.minimum_context_scale:
        return ModalityNormalization(modality_id, center, observed_scale, valid_pixel_count, None, config.minimum_context_scale)
    reason = "CONTEXT_SCALE_BELOW_MINIMUM"
    if config.degenerate_scale_policy == "reject_episode":
        raise DegenerateNormalizationError(f"{reason}: modality {modality_id!r} has scale {observed_scale:g}")
    if config.degenerate_scale_policy == "identity_scale":
        return ModalityNormalization(modality_id, center, 1.0, valid_pixel_count, reason, config.minimum_context_scale)
    statistic = config.frozen_population_statistics.get(modality_id)
    if statistic is None:
        raise DegenerateNormalizationError(
            f"{reason}: frozen_population_scale requires a hash-bound statistic for modality {modality_id!r}"
        )
    if statistic.scale < config.minimum_context_scale:
        raise DegenerateNormalizationError("frozen population scale is below minimum_context_scale")
    return ModalityNormalization(
        modality_id,
        statistic.center,
        statistic.scale,
        valid_pixel_count,
        "FROZEN_POPULATION_SCALE",
        config.minimum_context_scale,
        statistic.source_statistic_hash,
    )


def fit_preprocessing(
    observations: Iterable[DecodedObservation],
    *,
    context_ids: Iterable[str],
    config: NormalizationConfig | None = None,
) -> PreprocessingRecord:
    """Fit statistics from exactly the declared legal context observations."""

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
        center = float(values.mean().cpu())
        scale = float(values.std(unbiased=False).cpu())
        parameters.append(
            _resolve_degenerate_scale(
                modality_id=modality_id,
                center=center,
                observed_scale=scale,
                valid_pixel_count=int(values.numel()),
                config=config,
            )
        )
    canonical = [item.to_dict() for item in parameters]
    return PreprocessingRecord(
        policy_id=config.policy,
        fitted_from_context_ids=declared,
        modality_parameters=tuple(parameters),
        parameters_hash=_hash(canonical),
        config_hash=config.config_hash,
        unseen_modality_policy=config.unseen_modality_policy,
        degenerate_scale_policy=config.degenerate_scale_policy,
        minimum_context_scale=config.minimum_context_scale,
    )


def apply_preprocessing(record: PreprocessingRecord, observation: DecodedObservation) -> NormalizedObservation:
    """Apply frozen context-derived statistics to context or a revealed target."""

    if not isinstance(record, PreprocessingRecord) or not isinstance(observation, DecodedObservation):
        raise TypeError("record and observation must use T1-C data contracts")
    parameter = next((item for item in record.modality_parameters if item.modality_id == observation.modality_id), None)
    if parameter is None:
        raise ValueError("target modality has no context-derived normalization record")
    image = (observation.image - observation.image.new_tensor(parameter.center)) / observation.image.new_tensor(parameter.scale)
    image = torch.where(observation.valid_mask, image, torch.zeros_like(image))
    preprocessing_hash = record.record_hash
    input_content_hash = _hash(
        {
            "decoder_config_hash": observation.decoder_config_hash,
            "observation_id": observation.observation_id,
            "payload_sha256": observation.payload_sha256,
            "plane_hash": hashlib.sha256(observation.metadata.plane.canonical_json().encode("utf-8")).hexdigest(),
            "preprocessing_record_hash": preprocessing_hash,
            "tensor_dtype": str(image.dtype),
            "tensor_shape": tuple(image.shape),
            "valid_mask_hash": _mask_hash(observation.valid_mask),
        }
    )
    return NormalizedObservation(observation, image, observation.valid_mask, preprocessing_hash, input_content_hash)
