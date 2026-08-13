"""Gate-E E1 target-only 3-D reconstruction objectives.

This module is deliberately independent of the legacy renderer/training loss
stack.  It receives a prediction that was already produced by the target-free
Gate-D path and a supervision target only at this boundary.  No frontend,
trajectory, decoder, or routing object is accepted here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


CHARBONNIER_EPSILON = 1e-3
"""Fixed numerical epsilon in ``sqrt(residual**2 + epsilon**2)``."""

SSIM_WINDOW_SIZE = 3
"""The explicit unpadded local 3-D SSIM window edge length."""


@dataclass(frozen=True)
class ReconstructionLossConfig:
    """Tunable Gate-E E1 weights and explicit local-SSIM data range.

    ``ssim_data_range`` is an implementation parameter rather than an input
    normalization claim.  It must be supplied consistently with the target
    representation used by the caller; the initial value is one.
    """

    lambda_ssim: float = 0.2
    lambda_grad: float = 0.1
    ssim_data_range: float = 1.0

    def __post_init__(self) -> None:
        for name in ("lambda_ssim", "lambda_grad"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        data_range = float(self.ssim_data_range)
        if not math.isfinite(data_range) or data_range <= 0.0:
            raise ValueError("ssim_data_range must be positive and finite")
        object.__setattr__(self, "ssim_data_range", data_range)


@dataclass(frozen=True)
class ReconstructionLossResult:
    """Finite scalar E1 components, retained separately for E8 composition."""

    total: Tensor
    charbonnier: Tensor
    ssim: Tensor
    gradient: Tensor
    valid_voxel_count: int
    components: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        for name in ("total", "charbonnier", "ssim", "gradient"):
            value = getattr(self, name)
            if not isinstance(value, Tensor) or value.ndim != 0 or not bool(torch.isfinite(value)):
                raise ValueError(f"{name} must be one finite scalar tensor")
        if not isinstance(self.valid_voxel_count, int) or self.valid_voxel_count <= 0:
            raise ValueError("valid_voxel_count must be positive")
        checked = dict(self.components)
        for name, value in checked.items():
            if not isinstance(name, str) or not isinstance(value, Tensor) or value.ndim != 0:
                raise ValueError("components must map names to scalar tensors")
        object.__setattr__(self, "components", MappingProxyType(checked))


def _validate_prediction_target_mask(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(prediction, Tensor) or prediction.ndim != 5 or prediction.shape[1] != 1 or not prediction.is_floating_point():
        raise ValueError("prediction must be a floating tensor [B, 1, D, H, W]")
    if prediction.shape[0] <= 0 or any(length <= 0 for length in prediction.shape[-3:]):
        raise ValueError("prediction must have positive batch and spatial dimensions")
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("prediction must be finite")
    if not isinstance(target, Tensor) or target.shape != prediction.shape or not target.is_floating_point():
        raise ValueError("target must be a floating tensor matching prediction [B, 1, D, H, W]")
    if target.dtype != prediction.dtype or target.device != prediction.device:
        raise ValueError("target must match prediction dtype and device")
    if valid_mask is None:
        valid_mask = torch.ones_like(prediction, dtype=torch.bool)
    if not isinstance(valid_mask, Tensor) or valid_mask.dtype is not torch.bool or valid_mask.shape != prediction.shape:
        raise ValueError("valid_mask must be a bool tensor matching prediction exactly")
    if valid_mask.device != prediction.device:
        raise ValueError("valid_mask must match prediction device")
    per_subject_count = valid_mask.flatten(1).sum(dim=1)
    if bool((per_subject_count <= 0).any()):
        raise ValueError("valid_mask must retain at least one output voxel per subject")
    if not bool(torch.isfinite(target[valid_mask]).all()):
        raise ValueError("target must be finite on valid_mask support")
    # Outside valid support target values are intentionally irrelevant and may
    # be unavailable/non-finite.  Replacing them prevents an invalid value from
    # contaminating a rejected local SSIM/gradient window.
    safe_target = torch.where(valid_mask, target.detach(), torch.zeros_like(target))
    return prediction, safe_target, valid_mask


def _masked_mean(values: Tensor, mask: Tensor, *, name: str) -> Tensor:
    if values.shape != mask.shape:
        raise ValueError(f"{name} values and mask must share shape")
    count = mask.sum()
    if int(count.detach().cpu()) <= 0:
        raise ValueError(f"{name} has no valid support")
    masked = torch.where(mask, values, torch.zeros_like(values))
    return masked.sum() / count.to(dtype=values.dtype)


def charbonnier_loss(prediction: Tensor, target: Tensor, valid_mask: Tensor | None = None) -> Tensor:
    """Masked mean of the locked numerical Charbonnier expression."""

    prediction, target, valid_mask = _validate_prediction_target_mask(prediction, target, valid_mask)
    residual = prediction - target
    return _masked_mean(
        torch.sqrt(residual.square() + prediction.new_tensor(CHARBONNIER_EPSILON).square()),
        valid_mask,
        name="Charbonnier loss",
    )


def _gradient_pair_loss(prediction: Tensor, target: Tensor, valid_mask: Tensor) -> Tensor:
    terms: list[Tensor] = []
    for axis in (-3, -2, -1):
        left = [slice(None)] * prediction.ndim
        right = [slice(None)] * prediction.ndim
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        pair_mask = valid_mask[tuple(left)] & valid_mask[tuple(right)]
        difference = (prediction[tuple(right)] - prediction[tuple(left)]) - (
            target[tuple(right)] - target[tuple(left)]
        )
        if bool(pair_mask.any()):
            terms.append(torch.where(pair_mask, difference.abs(), torch.zeros_like(difference)).sum())
            terms.append(pair_mask.sum().to(dtype=prediction.dtype))
    if not terms:
        raise ValueError("gradient loss has no valid adjacent D/H/W voxel pairs")
    value_sum = torch.stack(terms[0::2]).sum()
    count = torch.stack(terms[1::2]).sum()
    return value_sum / count


def gradient_agreement_loss(prediction: Tensor, target: Tensor, valid_mask: Tensor | None = None) -> Tensor:
    """Masked finite-difference agreement over all D, H, and W axes."""

    prediction, target, valid_mask = _validate_prediction_target_mask(prediction, target, valid_mask)
    return _gradient_pair_loss(prediction, target, valid_mask)


def _ssim_loss(prediction: Tensor, target: Tensor, valid_mask: Tensor, *, data_range: float) -> Tensor:
    if any(length < SSIM_WINDOW_SIZE for length in prediction.shape[-3:]):
        raise ValueError("3-D SSIM requires every spatial dimension to be at least 3")
    kernel = prediction.new_full(
        (1, 1, SSIM_WINDOW_SIZE, SSIM_WINDOW_SIZE, SSIM_WINDOW_SIZE),
        1.0 / float(SSIM_WINDOW_SIZE**3),
    )
    full_window = F.conv3d(valid_mask.to(dtype=prediction.dtype), torch.ones_like(kernel)) == float(SSIM_WINDOW_SIZE**3)
    if not bool(full_window.any()):
        raise ValueError("3-D SSIM has no wholly valid unpadded 3x3x3 windows")
    mean_prediction = F.conv3d(prediction, kernel)
    mean_target = F.conv3d(target, kernel)
    variance_prediction = (F.conv3d(prediction.square(), kernel) - mean_prediction.square()).clamp_min(0.0)
    variance_target = (F.conv3d(target.square(), kernel) - mean_target.square()).clamp_min(0.0)
    covariance = F.conv3d(prediction * target, kernel) - mean_prediction * mean_target
    range_tensor = prediction.new_tensor(data_range)
    c1 = (0.01 * range_tensor).square()
    c2 = (0.03 * range_tensor).square()
    ssim = ((2.0 * mean_prediction * mean_target + c1) * (2.0 * covariance + c2)) / (
        (mean_prediction.square() + mean_target.square() + c1) * (variance_prediction + variance_target + c2)
    )
    return _masked_mean(1.0 - ssim, full_window, name="3-D SSIM loss")


def structural_similarity_loss(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
    *,
    data_range: float = 1.0,
) -> Tensor:
    """Differentiable 3-D local SSIM on wholly valid unpadded 3x3x3 windows."""

    if not math.isfinite(float(data_range)) or float(data_range) <= 0.0:
        raise ValueError("data_range must be positive and finite")
    prediction, target, valid_mask = _validate_prediction_target_mask(prediction, target, valid_mask)
    return _ssim_loss(prediction, target, valid_mask, data_range=float(data_range))


def reconstruction_loss(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
    *,
    config: ReconstructionLossConfig | None = None,
) -> ReconstructionLossResult:
    """Compute ``L_charbonnier + lambda_ssim L_ssim + lambda_grad L_grad``."""

    config = config or ReconstructionLossConfig()
    if not isinstance(config, ReconstructionLossConfig):
        raise TypeError("config must be a ReconstructionLossConfig")
    prediction, target, valid_mask = _validate_prediction_target_mask(prediction, target, valid_mask)
    residual = prediction - target
    charbonnier = _masked_mean(
        torch.sqrt(residual.square() + prediction.new_tensor(CHARBONNIER_EPSILON).square()),
        valid_mask,
        name="Charbonnier loss",
    )
    zero = prediction.sum() * 0.0
    ssim = (
        _ssim_loss(prediction, target, valid_mask, data_range=config.ssim_data_range)
        if config.lambda_ssim > 0.0
        else zero
    )
    gradient = _gradient_pair_loss(prediction, target, valid_mask) if config.lambda_grad > 0.0 else zero
    total = charbonnier + config.lambda_ssim * ssim + config.lambda_grad * gradient
    if not bool(torch.isfinite(total)):
        raise ValueError("Gate-E reconstruction objective is non-finite")
    return ReconstructionLossResult(
        total=total,
        charbonnier=charbonnier,
        ssim=ssim,
        gradient=gradient,
        valid_voxel_count=int(valid_mask.sum().detach().cpu()),
        components={"charbonnier": charbonnier, "ssim": ssim, "gradient": gradient},
    )


def pointwise_charbonnier_by_subject(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    *,
    allow_empty: bool = False,
) -> Tensor:
    """Per-subject Charbonnier used for compact local/spill E2--E6 probes.

    Point sets do not carry a regular 3-D neighborhood, so their local metric
    is explicitly the same fixed Charbonnier component used by E1, not a
    second learned or target-conditioned objective.
    """

    if not isinstance(prediction, Tensor) or prediction.ndim != 3 or prediction.shape[-1] != 1 or not prediction.is_floating_point():
        raise ValueError("prediction must be a floating tensor [B, N, 1]")
    if not isinstance(target, Tensor) or target.shape != prediction.shape or not target.is_floating_point():
        raise ValueError("target must match pointwise prediction [B, N, 1]")
    if prediction.dtype != target.dtype or prediction.device != target.device:
        raise ValueError("pointwise prediction and target must share dtype and device")
    if not isinstance(valid_mask, Tensor) or valid_mask.dtype is not torch.bool or valid_mask.shape != prediction.shape:
        raise ValueError("valid_mask must be bool and match pointwise prediction")
    if valid_mask.device != prediction.device:
        raise ValueError("pointwise valid_mask must match prediction device")
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("pointwise prediction must be finite")
    if not bool(torch.isfinite(target[valid_mask]).all()):
        raise ValueError("pointwise target must be finite on valid support")
    safe_target = torch.where(valid_mask, target.detach(), torch.zeros_like(target))
    values = torch.sqrt((prediction - safe_target).square() + prediction.new_tensor(CHARBONNIER_EPSILON).square())
    counts = valid_mask.flatten(1).sum(dim=1)
    if not allow_empty and bool((counts <= 0).any()):
        raise ValueError("every pointwise subject must have valid supervision support")
    total = torch.where(valid_mask, values, torch.zeros_like(values)).flatten(1).sum(dim=1)
    result = total / counts.clamp_min(1).to(dtype=prediction.dtype)
    return torch.where(counts > 0, result, prediction.flatten(1).sum(dim=1) * 0.0)


__all__ = [
    "CHARBONNIER_EPSILON",
    "ReconstructionLossConfig",
    "ReconstructionLossResult",
    "SSIM_WINDOW_SIZE",
    "charbonnier_loss",
    "gradient_agreement_loss",
    "pointwise_charbonnier_by_subject",
    "reconstruction_loss",
    "structural_similarity_loss",
]
