"""Target-after-inference reconstruction and semantic diagnostics.

The helpers in this module are deliberately small and dependency-free.  They
operate only after a target-free prediction exists; neither reconstruction nor
segmentation targets are accepted by the point-guided model APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .losses import structural_similarity_loss


@dataclass(frozen=True)
class ReconstructionMetrics:
    """Metrics for one prediction/target pair in a declared intensity space."""

    mae: float
    psnr: float
    ssim: float
    voxel_count: int
    intensity_space: str = "masked_robust_01_[0,1]"


@dataclass(frozen=True)
class SemanticDiceMetrics:
    """Per-class Dice values for the three coarse semantic classes."""

    dice_normal: float
    dice_edema: float
    dice_core: float
    voxel_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "dice_normal": self.dice_normal,
            "dice_edema": self.dice_edema,
            "dice_core": self.dice_core,
            "voxel_count": self.voxel_count,
        }


def _validate_reconstruction_inputs(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor | None,
) -> Tensor:
    if (
        not isinstance(prediction, Tensor)
        or prediction.ndim != 5
        or prediction.shape[1] != 1
        or not prediction.is_floating_point()
    ):
        raise ValueError("prediction must be a floating [B,1,D,H,W] tensor")
    if not isinstance(target, Tensor) or target.shape != prediction.shape or not target.is_floating_point():
        raise ValueError("target must be a floating tensor matching prediction")
    if target.dtype != prediction.dtype or target.device != prediction.device:
        raise ValueError("prediction and target must share dtype and device")
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("prediction and target must be finite")
    if valid_mask is None:
        valid_mask = torch.ones_like(prediction, dtype=torch.bool)
    if valid_mask.dtype is not torch.bool or valid_mask.shape != prediction.shape:
        raise ValueError("valid_mask must be bool and match prediction")
    if valid_mask.device != prediction.device or not bool(valid_mask.flatten(1).any(dim=1).all()):
        raise ValueError("valid_mask must match prediction device and retain each subject")
    return valid_mask


def compute_reconstruction_metrics(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
    *,
    data_range: float = 1.0,
    intensity_space: str = "masked_robust_01_[0,1]",
) -> ReconstructionMetrics:
    """Compute MAE, PSNR, and 3-D SSIM after inference.

    SSIM uses the repository's unpadded 3-D window implementation when a
    wholly valid window exists.  For very small/debug volumes or fragmented
    masks, a finite global SSIM fallback keeps diagnostics observable without
    changing the Gate-E training objective.
    """

    valid_mask = _validate_reconstruction_inputs(prediction, target, valid_mask)
    data_range = float(data_range)
    if not math.isfinite(data_range) or data_range <= 0.0:
        raise ValueError("data_range must be positive and finite")
    if not isinstance(intensity_space, str) or not intensity_space:
        raise ValueError("intensity_space must be a non-empty string")
    count = int(valid_mask.sum().detach().cpu())
    safe_target = torch.where(valid_mask, target, torch.zeros_like(target))
    residual = prediction - safe_target
    mae = float(torch.where(valid_mask, residual.abs(), torch.zeros_like(residual)).sum().detach().cpu()) / count
    mse = float(torch.where(valid_mask, residual.square(), torch.zeros_like(residual)).sum().detach().cpu()) / count
    # Clamp only the diagnostic denominator so an exact match has a finite,
    # machine-readable PSNR instead of an unbounded value in CSV/JSON logs.
    psnr = 10.0 * math.log10((data_range * data_range) / max(mse, 1e-12))
    try:
        ssim_loss = structural_similarity_loss(
            prediction,
            safe_target,
            valid_mask,
            data_range=data_range,
        )
        ssim = float((1.0 - ssim_loss).detach().cpu())
    except ValueError:
        values_prediction = prediction[valid_mask]
        values_target = safe_target[valid_mask]
        mean_prediction = values_prediction.mean()
        mean_target = values_target.mean()
        var_prediction = values_prediction.var(unbiased=False)
        var_target = values_target.var(unbiased=False)
        covariance = ((values_prediction - mean_prediction) * (values_target - mean_target)).mean()
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        ssim = float(
            (((2.0 * mean_prediction * mean_target + c1) * (2.0 * covariance + c2))
             / ((mean_prediction.square() + mean_target.square() + c1) * (var_prediction + var_target + c2)))
            .detach()
            .cpu()
        )
    return ReconstructionMetrics(
        mae=mae,
        psnr=psnr,
        ssim=ssim,
        voxel_count=count,
        intensity_space=intensity_space,
    )


def semantic_dice(
    probabilities: Tensor,
    target: Tensor,
    *,
    ignore_index: int = 255,
) -> SemanticDiceMetrics:
    """Compute normal/edema/core Dice from post-inference semantic output."""

    if (
        not isinstance(probabilities, Tensor)
        or probabilities.ndim != 5
        or probabilities.shape[1] != 3
        or not probabilities.is_floating_point()
    ):
        raise ValueError("probabilities must have shape [B,3,D,H,W]")
    if not isinstance(target, Tensor) or target.shape != probabilities.shape[:1] + probabilities.shape[2:] or target.dtype != torch.long:
        raise ValueError("target must be long [B,D,H,W] aligned with probabilities")
    if target.device != probabilities.device:
        raise ValueError("semantic target must match probability device")
    valid = target != int(ignore_index)
    if not bool(valid.any()):
        raise ValueError("semantic target has no non-ignored voxels")
    prediction = probabilities.argmax(dim=1)
    values: list[float] = []
    for class_index in range(3):
        predicted = (prediction == class_index) & valid
        expected = (target == class_index) & valid
        intersection = int((predicted & expected).sum().detach().cpu())
        denominator = int(predicted.sum().detach().cpu()) + int(expected.sum().detach().cpu())
        values.append(1.0 if denominator == 0 else 2.0 * intersection / denominator)
    return SemanticDiceMetrics(
        dice_normal=values[0],
        dice_edema=values[1],
        dice_core=values[2],
        voxel_count=int(valid.sum().detach().cpu()),
    )


__all__ = [
    "ReconstructionMetrics",
    "SemanticDiceMetrics",
    "compute_reconstruction_metrics",
    "semantic_dice",
]
