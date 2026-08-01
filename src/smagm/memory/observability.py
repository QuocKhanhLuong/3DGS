"""Observability construction and monotone uncertainty updates."""

from __future__ import annotations

import torch

from .contracts import PrimitiveObservability


def initial_observability(anchor_observability: torch.Tensor, *, initial_uncertainty: float = 1.0) -> PrimitiveObservability:
    if anchor_observability.ndim != 2 or anchor_observability.shape[1] < 3 or initial_uncertainty < 0:
        raise ValueError("anchor observability requires count, weight, disagreement")
    count = anchor_observability.shape[0]
    dtype, device = anchor_observability.dtype, anchor_observability.device
    return PrimitiveObservability(
        evidence_count=anchor_observability[:, 0:1].clamp_min(0),
        coverage=anchor_observability[:, 1:2].clamp(0, 1),
        disagreement=anchor_observability[:, 2:3].clamp_min(0),
        uncertainty=torch.full((count, 1), initial_uncertainty, dtype=dtype, device=device),
        propagation_depth=torch.zeros((count, 1), dtype=torch.int64, device=device),
        update_round=torch.zeros((count, 1), dtype=torch.int64, device=device),
    )


def propagated_observability(parent: PrimitiveObservability, indices: torch.Tensor, *, uncertainty_growth: float, update_round: int) -> PrimitiveObservability:
    if uncertainty_growth < 0 or update_round < 0:
        raise ValueError("propagation uncertainty and round must be non-negative")
    return PrimitiveObservability(
        evidence_count=parent.evidence_count[indices], coverage=parent.coverage[indices],
        disagreement=parent.disagreement[indices], uncertainty=parent.uncertainty[indices] + uncertainty_growth,
        propagation_depth=parent.propagation_depth[indices] + 1,
        update_round=torch.full_like(parent.update_round[indices], update_round),
    )
