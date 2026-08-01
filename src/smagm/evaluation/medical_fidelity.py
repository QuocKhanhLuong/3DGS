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


def evaluate_roi_fidelity(prediction: VolumeReconstruction, target: torch.Tensor, roi_mask: torch.Tensor, background_mask: torch.Tensor) -> ROIFidelity:
    if any(mask.shape != target.shape or mask.dtype is not torch.bool for mask in (roi_mask, background_mask)):
        raise ValueError("ROI/background masks must be bool and match target")
    if bool((roi_mask & background_mask).any()) or not bool(roi_mask.any()) or not bool(background_mask.any()):
        raise ValueError("ROI and background must be non-empty and disjoint")
    legal_roi = roi_mask & ~prediction.unsupported_mask
    legal_background = background_mask & ~prediction.unsupported_mask
    if not bool(legal_roi.any()) or not bool(legal_background.any()):
        raise ValueError("ROI fidelity requires supported ROI and background")
    predicted_contrast = prediction.intensity[legal_roi].mean() - prediction.intensity[legal_background].mean()
    target_contrast = target[legal_roi].mean() - target[legal_background].mean()
    return ROIFidelity(int(roi_mask.sum()), float(legal_roi.sum() / roi_mask.sum()), float((prediction.intensity[legal_roi] - target[legal_roi]).abs().mean()), float((predicted_contrast - target_contrast).abs()))
