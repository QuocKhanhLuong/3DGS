"""Optional ROI fidelity metrics without training feedback."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..contracts.outputs import VolumeReconstruction


@dataclass(frozen=True)
class ROIFidelity:
    roi_voxels: int
    supported_roi_fraction: float
    roi_mae: float
    contrast_error: float
    status: str = "COMPUTED"
    boundary_band_voxels: int = 0
    supported_boundary_band_fraction: float = float("nan")
    boundary_band_mae: float = float("nan")
    tumor_mae: float = float("nan")
    non_tumor_mae: float = float("nan")


def _boundary_band(roi_mask: torch.Tensor, background_mask: torch.Tensor) -> torch.Tensor:
    boundary = torch.zeros_like(roi_mask)
    for axis in range(roi_mask.ndim):
        left = [slice(None)] * roi_mask.ndim
        right = [slice(None)] * roi_mask.ndim
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        transition = (roi_mask[tuple(left)] & background_mask[tuple(right)]) | (
            roi_mask[tuple(right)] & background_mask[tuple(left)]
        )
        boundary[tuple(left)] |= transition
        boundary[tuple(right)] |= transition
    return boundary


def evaluate_roi_fidelity(
    prediction: VolumeReconstruction,
    target: torch.Tensor,
    roi_mask: torch.Tensor,
    background_mask: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> ROIFidelity:
    if any(mask.shape != target.shape or mask.dtype is not torch.bool for mask in (roi_mask, background_mask)):
        raise ValueError("ROI/background masks must be bool and match target")
    if prediction.unsupported_mask.shape != target.shape:
        raise ValueError("prediction support mask must match target")
    if bool((roi_mask & background_mask).any()):
        raise ValueError("ROI and background must be disjoint")
    if valid_mask is None:
        valid_mask = torch.ones_like(roi_mask)
    if valid_mask.shape != target.shape or valid_mask.dtype is not torch.bool:
        raise ValueError("ROI valid_mask must be bool and match target")
    if not bool(roi_mask.any()):
        return ROIFidelity(0, float("nan"), float("nan"), float("nan"), status="SKIPPED_EMPTY_ROI")
    if not bool(background_mask.any()):
        return ROIFidelity(int(roi_mask.sum()), float("nan"), float("nan"), float("nan"), status="SKIPPED_EMPTY_BACKGROUND")

    legal = valid_mask & ~prediction.unsupported_mask
    legal_roi = roi_mask & legal
    legal_background = background_mask & legal
    boundary = _boundary_band(roi_mask, background_mask)
    legal_boundary = boundary & legal
    roi_fraction = float(legal_roi.sum() / roi_mask.sum())
    boundary_fraction = float(legal_boundary.sum() / boundary.sum()) if bool(boundary.any()) else float("nan")
    roi_mae = float((prediction.intensity[legal_roi] - target[legal_roi]).abs().mean()) if bool(legal_roi.any()) else float("nan")
    boundary_mae = float((prediction.intensity[legal_boundary] - target[legal_boundary]).abs().mean()) if bool(legal_boundary.any()) else float("nan")
    tumor_mae = roi_mae
    non_tumor_mae = float((prediction.intensity[legal_background] - target[legal_background]).abs().mean()) if bool(legal_background.any()) else float("nan")
    if bool(legal_roi.any()) and bool(legal_background.any()):
        predicted_contrast = prediction.intensity[legal_roi].mean() - prediction.intensity[legal_background].mean()
        target_contrast = target[legal_roi].mean() - target[legal_background].mean()
        contrast_error = float((predicted_contrast - target_contrast).abs())
    else:
        contrast_error = float("nan")
    if not bool(legal_roi.any()):
        status = "SKIPPED_UNSUPPORTED_ROI"
    elif not bool(legal_background.any()):
        status = "SKIPPED_UNSUPPORTED_BACKGROUND"
    elif bool(boundary.any()) and not bool(legal_boundary.any()):
        status = "SKIPPED_UNSUPPORTED_BOUNDARY"
    else:
        status = "COMPUTED"
    return ROIFidelity(
        int(roi_mask.sum()), roi_fraction, roi_mae, contrast_error,
        status=status,
        boundary_band_voxels=int(boundary.sum()),
        supported_boundary_band_fraction=boundary_fraction,
        boundary_band_mae=boundary_mae,
        tumor_mae=tumor_mae,
        non_tumor_mae=non_tumor_mae,
    )
