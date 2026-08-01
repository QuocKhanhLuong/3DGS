"""Bounded Gaussian-memory spatial culling."""

from __future__ import annotations

import torch

from .contracts import GaussianMemoryBank


def query_memory_radius(bank: GaussianMemoryBank, points_ras_mm: torch.Tensor, *, radius_mm: float, maximum_primitives: int) -> tuple[torch.Tensor, torch.Tensor]:
    if points_ras_mm.ndim != 2 or points_ras_mm.shape[1] != 3 or radius_mm <= 0 or maximum_primitives <= 0:
        raise ValueError("memory queries require [Q,3] points and positive bounds")
    distance = torch.cdist(points_ras_mm, bank.gaussians.centers_ras_mm)
    values, indices = torch.sort(distance, dim=1, stable=True)
    values, indices = values[:, :maximum_primitives], indices[:, :maximum_primitives]
    return indices, values <= radius_mm
