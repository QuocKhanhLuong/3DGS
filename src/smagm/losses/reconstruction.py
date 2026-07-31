"""Supported-mask-aware target-plane objectives for legal T1-C training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Literal, Mapping

import torch

from ..renderer import RenderResult


class EmptyReconstructionMaskError(ValueError):
    """Raised when no target-valid and renderer-supported pixel exists."""


@dataclass(frozen=True)
class ReconstructionLossConfig:
    intensity: Literal["l1", "mse"] = "l1"
    intensity_weight: float = 1.0
    gradient_weight: float = 0.0
    frequency_weight: float = 0.0
    empty_mask_policy: Literal["raise", "skip"] = "raise"

    def __post_init__(self) -> None:
        if self.intensity not in ("l1", "mse"):
            raise ValueError("intensity must be l1 or mse")
        for name in ("intensity_weight", "gradient_weight", "frequency_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.intensity_weight + self.gradient_weight + self.frequency_weight <= 0.0:
            raise ValueError("at least one reconstruction component must be enabled")
        if self.empty_mask_policy not in ("raise", "skip"):
            raise ValueError("empty_mask_policy must be raise or skip")


@dataclass(frozen=True)
class ReconstructionLossResult:
    total: torch.Tensor
    components: Mapping[str, torch.Tensor]
    legal_pixel_count: int
    target_valid_pixel_count: int
    supported_fraction: float
    status: Literal["OK", "SKIPPED_EMPTY_LEGAL_MASK"]

    def __post_init__(self) -> None:
        if not isinstance(self.total, torch.Tensor) or self.total.ndim != 0 or not bool(torch.isfinite(self.total)):
            raise ValueError("reconstruction total must be one finite scalar")
        if self.legal_pixel_count < 0 or self.target_valid_pixel_count < 0:
            raise ValueError("pixel counts must be non-negative")
        if not math.isfinite(self.supported_fraction) or not 0.0 <= self.supported_fraction <= 1.0:
            raise ValueError("supported_fraction must lie in [0, 1]")
        if self.status == "OK" and self.legal_pixel_count <= 0:
            raise ValueError("OK reconstruction loss requires legal pixels")
        if self.status != "OK" and self.legal_pixel_count != 0:
            raise ValueError("a skipped reconstruction result cannot report legal pixels")
        for value in self.components.values():
            if not isinstance(value, torch.Tensor) or value.ndim != 0 or not bool(torch.isfinite(value)):
                raise ValueError("every reconstruction component must be a finite scalar")
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))


def _target_plane(target: torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(target, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if target.ndim == 3 and target.shape[0] == 1:
        target = target[0]
    if target.shape != reference.shape:
        raise ValueError(f"{name} must match the rendered [H, W] plane")
    if target.device != reference.device:
        raise ValueError(f"{name} must share the prediction device")
    return target


def reconstruction_loss(
    prediction: RenderResult,
    target: torch.Tensor,
    target_valid_mask: torch.Tensor,
    *,
    config: ReconstructionLossConfig | None = None,
) -> ReconstructionLossResult:
    """Compare a live render with a target only on the legal supported mask."""

    if not isinstance(prediction, RenderResult):
        raise TypeError("prediction must be a RenderResult")
    config = config or ReconstructionLossConfig()
    target = _target_plane(target, prediction.intensity, "target")
    target_valid_mask = _target_plane(target_valid_mask, prediction.intensity, "target_valid_mask")
    if target_valid_mask.dtype is not torch.bool:
        raise TypeError("target_valid_mask must be bool")
    if target.dtype != prediction.intensity.dtype:
        raise ValueError("target and prediction must share dtype")
    if prediction.unsupported_mask.dtype is not torch.bool or prediction.unsupported_mask.shape != target.shape:
        raise ValueError("prediction unsupported mask must match the target plane")
    if not bool(torch.isfinite(target[target_valid_mask]).all()):
        raise ValueError("valid target pixels must be finite")
    legal = target_valid_mask & ~prediction.unsupported_mask
    legal_count = int(legal.sum().detach().cpu())
    target_valid_count = int(target_valid_mask.sum().detach().cpu())
    supported_fraction = float(legal_count / target_valid_count) if target_valid_count else 0.0
    if legal_count == 0:
        if config.empty_mask_policy == "raise":
            raise EmptyReconstructionMaskError("reconstruction has no target-valid supported pixels")
        zero = torch.nan_to_num(prediction.intensity).sum() * 0.0
        return ReconstructionLossResult(
            zero,
            {},
            0,
            target_valid_count,
            supported_fraction,
            "SKIPPED_EMPTY_LEGAL_MASK",
        )
    if not bool(torch.isfinite(prediction.intensity[legal]).all()):
        raise ValueError("legal prediction pixels must be finite")
    residual = prediction.intensity - target
    components: dict[str, torch.Tensor] = {}
    if config.intensity == "l1":
        components["intensity"] = residual[legal].abs().mean()
    else:
        components["intensity"] = residual[legal].square().mean()
    total = config.intensity_weight * components["intensity"]

    if config.gradient_weight > 0.0:
        gradient_terms: list[torch.Tensor] = []
        horizontal = legal[:, 1:] & legal[:, :-1]
        vertical = legal[1:, :] & legal[:-1, :]
        if bool(horizontal.any()):
            gradient_terms.append((residual[:, 1:] - residual[:, :-1])[horizontal].abs())
        if bool(vertical.any()):
            gradient_terms.append((residual[1:, :] - residual[:-1, :])[vertical].abs())
        if not gradient_terms:
            raise EmptyReconstructionMaskError("gradient loss has no adjacent legal pixel pairs")
        components["gradient"] = torch.cat(gradient_terms).mean()
        total = total + config.gradient_weight * components["gradient"]

    if config.frequency_weight > 0.0:
        masked = torch.where(legal, residual, torch.zeros_like(residual))
        spectrum = torch.fft.rfft2(masked, norm="ortho")
        components["frequency"] = spectrum.abs().square().mean() / masked.new_tensor(
            max(supported_fraction, torch.finfo(masked.dtype).eps)
        )
        total = total + config.frequency_weight * components["frequency"]

    if not bool(torch.isfinite(total)):
        raise ValueError("reconstruction objective is non-finite")
    return ReconstructionLossResult(
        total,
        components,
        legal_count,
        target_valid_count,
        supported_fraction,
        "OK",
    )
