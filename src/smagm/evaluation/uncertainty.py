"""Diagnostic uncertainty/error association without calibration claims."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..contracts.outputs import VolumeReconstruction


@dataclass(frozen=True)
class UncertaintyAssociation:
    sample_count: int
    error_uncertainty_correlation: float
    label: str = "uncalibrated_support_uncertainty_association"


def evaluate_uncertainty_association(prediction: VolumeReconstruction, target: torch.Tensor, target_valid_mask: torch.Tensor) -> UncertaintyAssociation:
    legal = target_valid_mask & ~prediction.unsupported_mask
    if int(legal.sum()) < 2:
        raise ValueError("uncertainty association requires at least two supported targets")
    error = (prediction.intensity[legal] - target[legal]).abs(); uncertainty = prediction.support_uncertainty[legal]
    error = error - error.mean(); uncertainty = uncertainty - uncertainty.mean()
    denominator = torch.linalg.vector_norm(error) * torch.linalg.vector_norm(uncertainty)
    correlation = 0.0 if float(denominator) <= torch.finfo(error.dtype).eps else float((error * uncertainty).sum() / denominator)
    return UncertaintyAssociation(int(legal.sum()), correlation)
