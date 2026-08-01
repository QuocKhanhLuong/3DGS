"""Physical-to-local query construction and shared field evaluation."""

from __future__ import annotations

import torch

from ..anchors import AnchorBatch, query_anchor_neighbors
from .blend import blend_local_fields
from .contracts import FieldQueryBatch, StructuralFieldOutput
from .local import SharedStructuralField


def build_field_queries(
    anchors: AnchorBatch, points_ras_mm: torch.Tensor, *, maximum_neighbors: int = 8,
) -> FieldQueryBatch:
    radius = float(anchors.support_scales_mm.max().detach())
    indices, valid = query_anchor_neighbors(anchors, points_ras_mm, radius_mm=radius, maximum_neighbors=maximum_neighbors)
    centers = anchors.centers_ras_mm[indices]
    frames = anchors.frame_axes_ras[indices]
    scales = anchors.support_scales_mm[indices]
    delta = points_ras_mm[:, None, :] - centers
    local = torch.einsum("qkji,qkj->qki", frames, delta) / scales
    valid = valid & (local.abs() <= 1).all(dim=-1)
    return FieldQueryBatch(points_ras_mm, indices, local, valid)


def query_structural_field(
    field: SharedStructuralField, anchors: AnchorBatch, points_ras_mm: torch.Tensor, *,
    maximum_neighbors: int = 8, minimum_total_weight: float = 1e-6,
) -> StructuralFieldOutput:
    queries = build_field_queries(anchors, points_ras_mm, maximum_neighbors=maximum_neighbors)
    evidence = anchors.evidence[queries.anchor_indices]
    local_values = field(queries.local_coordinates, evidence)
    return blend_local_fields(local_values, queries.local_coordinates, queries.neighbor_valid, minimum_total_weight=minimum_total_weight)
