"""Uncalibrated support/reliability diagnostics for static reconstruction."""

from __future__ import annotations

import torch


def support_uncertainty(support_mass: torch.Tensor, unsupported_mask: torch.Tensor, *, propagation_uncertainty: float) -> torch.Tensor:
    if support_mass.shape != unsupported_mask.shape or unsupported_mask.dtype is not torch.bool or propagation_uncertainty < 0:
        raise ValueError("support uncertainty inputs are invalid")
    value = support_mass.clamp_min(torch.finfo(support_mass.dtype).eps).reciprocal() + propagation_uncertainty
    return torch.where(unsupported_mask, torch.full_like(value, float("nan")), value)
