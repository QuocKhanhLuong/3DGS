"""Explicit modality appearance slots without missing-modality hallucination."""

from __future__ import annotations

import torch


def validate_appearance_slots(appearance: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if appearance.ndim != 2 or valid.shape != appearance.shape or valid.dtype is not torch.bool:
        raise ValueError("appearance and validity must be [N,M] tensors")
    if not bool(torch.isfinite(appearance).all()):
        raise ValueError("appearance values must be finite")
    return torch.where(valid, appearance, torch.zeros_like(appearance)), valid
