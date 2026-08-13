"""Pure tensor metrics for post-inference point-guided evaluation.

The helper has no model, data-loader, checkpoint, or target-observation
coupling.  A target may be supplied only by the caller at evaluation time;
all reductions are over the explicitly provided valid mask.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class PointGuidedMetrics:
    """Scalar masked reconstruction metrics."""

    mae: float
    mse: float
    nmse: float
    psnr: float
    ssim: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mae": self.mae,
            "mse": self.mse,
            "nmse": self.nmse,
            "psnr": self.psnr,
            "ssim": self.ssim,
        }


def _validate_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("prediction and target must be torch.Tensor values")
    if prediction.shape != target.shape or prediction.numel() == 0:
        raise ValueError("prediction and target must have the same non-empty shape")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating-point tensors")
    if valid_mask is None:
        mask = torch.ones(prediction.shape, dtype=torch.bool, device=prediction.device)
    else:
        if not isinstance(valid_mask, torch.Tensor) or valid_mask.shape != target.shape or valid_mask.dtype is not torch.bool:
            raise ValueError("valid_mask must be bool and match prediction/target shape")
        mask = valid_mask.to(device=prediction.device)
    if not bool(mask.any()):
        raise ValueError("valid_mask must contain at least one element")
    selected_prediction = prediction[mask]
    selected_target = target.to(device=prediction.device)[mask]
    if not bool(torch.isfinite(selected_prediction).all()) or not bool(torch.isfinite(selected_target).all()):
        raise ValueError("evaluated prediction and target values must be finite")
    return selected_prediction, selected_target, mask


def _resolve_data_range(target: torch.Tensor, data_range: float | None) -> torch.Tensor:
    if data_range is not None:
        value = float(data_range)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("data_range must be finite and positive")
        return target.new_tensor(value)
    span = target.max() - target.min()
    # Constant targets have no empirical range.  A unit range keeps SSIM and
    # PSNR finite and deterministic while preserving the exact-match cases.
    return torch.where(span > 0.0, span, target.new_tensor(1.0))


def _global_ssim(prediction: torch.Tensor, target: torch.Tensor, data_range: torch.Tensor) -> torch.Tensor:
    """Compute the global population-statistic SSIM over one masked vector."""

    mean_prediction = prediction.mean()
    mean_target = target.mean()
    centered_prediction = prediction - mean_prediction
    centered_target = target - mean_target
    variance_prediction = centered_prediction.square().mean()
    variance_target = centered_target.square().mean()
    covariance = (centered_prediction * centered_target).mean()
    c1 = (0.01 * data_range).square()
    c2 = (0.03 * data_range).square()
    numerator = (2.0 * mean_prediction * mean_target + c1) * (2.0 * covariance + c2)
    denominator = (mean_prediction.square() + mean_target.square() + c1) * (
        variance_prediction + variance_target + c2
    )
    return numerator / denominator.clamp_min(torch.finfo(numerator.dtype).eps)


def compute_point_guided_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    data_range: float | None = None,
) -> PointGuidedMetrics:
    """Return masked MAE, MSE, NMSE, PSNR, and global SSIM.

    ``data_range`` defaults to the target span over valid voxels, with a
    deterministic unit fallback for a constant target.  Exact reconstruction
    reports ``psnr=inf``, ``nmse=0``, and ``ssim=1`` as expected.
    """

    prediction_values, target_values, _ = _validate_inputs(prediction, target, valid_mask)
    error = prediction_values - target_values
    mse_tensor = error.square().mean()
    mae_tensor = error.abs().mean()
    target_energy = target_values.square().mean()
    nmse_tensor = mse_tensor / target_energy.clamp_min(torch.finfo(mse_tensor.dtype).eps)
    resolved_range = _resolve_data_range(target_values, data_range)
    if bool(mse_tensor == 0.0):
        psnr = float("inf")
    else:
        psnr = float((10.0 * torch.log10(resolved_range.square() / mse_tensor)).detach())
    ssim_tensor = _global_ssim(prediction_values, target_values, resolved_range)
    return PointGuidedMetrics(
        mae=float(mae_tensor.detach()),
        mse=float(mse_tensor.detach()),
        nmse=float(nmse_tensor.detach()),
        psnr=psnr,
        ssim=float(ssim_tensor.detach()),
    )


compute_metrics = compute_point_guided_metrics


__all__ = [
    "PointGuidedMetrics",
    "compute_metrics",
    "compute_point_guided_metrics",
]
