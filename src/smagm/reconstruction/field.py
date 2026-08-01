"""Optional structural level-set volume queries from frozen field weights."""

from __future__ import annotations

import torch

from ..contracts.coordinates import TargetGrid
from ..fields import SharedStructuralField, query_structural_field
from ..state import PatientState


def reconstruct_structural_field(state: PatientState, field: SharedStructuralField, grid: TargetGrid, *, point_chunk_size: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    if point_chunk_size <= 0:
        raise ValueError("point_chunk_size must be positive")
    points = [grid.world_from_dhw(d, h, w) for d in range(grid.shape_dhw[0]) for h in range(grid.shape_dhw[1]) for w in range(grid.shape_dhw[2])]
    parameter = next(field.parameters())
    coordinates = torch.tensor(points, dtype=parameter.dtype, device=parameter.device)
    values, supported = [], []
    for start in range(0, coordinates.shape[0], point_chunk_size):
        output = query_structural_field(field, state.anchors, coordinates[start:start + point_chunk_size])
        values.append(output.value); supported.append(output.supported)
    return torch.cat(values).reshape(grid.shape_dhw), torch.cat(supported).reshape(grid.shape_dhw)
