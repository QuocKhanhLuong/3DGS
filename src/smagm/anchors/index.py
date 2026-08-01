"""Bounded deterministic spatial queries over anchor centres."""

from __future__ import annotations

import torch

from .contracts import AnchorBatch


def query_anchor_neighbors(anchors: AnchorBatch, points_ras_mm: torch.Tensor, *, radius_mm: float, maximum_neighbors: int) -> tuple[torch.Tensor, torch.Tensor]:
    if points_ras_mm.ndim != 2 or points_ras_mm.shape[1] != 3 or radius_mm <= 0 or maximum_neighbors <= 0:
        raise ValueError("queries require [Q,3] points, positive radius, and positive bound")
    distances = torch.cdist(points_ras_mm, anchors.centers_ras_mm)
    values, indices = torch.sort(distances, dim=1, stable=True)
    values, indices = values[:, :maximum_neighbors], indices[:, :maximum_neighbors]
    valid = values <= radius_mm
    return indices, valid
