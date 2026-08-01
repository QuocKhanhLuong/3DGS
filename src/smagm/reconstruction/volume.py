"""Chunked full physical-grid reconstruction with affine preservation."""

from __future__ import annotations

import hashlib
import math

import torch

from ..contracts.coordinates import PhysicalPlane, TargetGrid
from ..contracts.outputs import VolumeReconstruction, volume_output_hash
from ..renderer import RenderConfig
from ..state import PatientState
from .plane import reconstruct_plane


def _unit_step(matrix: tuple[tuple[float, ...], ...], column: int) -> tuple[tuple[float, float, float], float]:
    vector = tuple(matrix[row][column] for row in range(3))
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("TargetGrid affine axes must have positive finite spacing")
    return tuple(value / norm for value in vector), norm


def plane_from_target_grid(grid: TargetGrid, depth_index: int, *, observation_id: str | None = None) -> PhysicalPlane:
    if not 0 <= depth_index < grid.shape_dhw[0]:
        raise IndexError("depth index is outside TargetGrid")
    axis_u, spacing_u = _unit_step(grid.index_to_ras_mm, 0)
    axis_v, spacing_v = _unit_step(grid.index_to_ras_mm, 1)
    normal, spacing_d = _unit_step(grid.index_to_ras_mm, 2)
    return PhysicalPlane(
        pixel_center_origin_ras_mm=grid.world_from_dhw(depth_index, 0, 0),
        axis_u_ras=axis_u, axis_v_ras=axis_v, spacing_uv_mm=(spacing_u, spacing_v),
        thickness_mm=spacing_d, shape_hw=grid.shape_dhw[1:], signed_normal_ras=normal,
        observation_id=observation_id,
    )


def reconstruct_volume(
    state: PatientState, grid: TargetGrid, *, modality_id: str, depth_chunk_size: int = 8,
    render_config: RenderConfig | None = None,
) -> VolumeReconstruction:
    if depth_chunk_size <= 0:
        raise ValueError("depth_chunk_size must be positive")
    render_config = render_config or RenderConfig()
    intensities, supports, masks, uncertainties = [], [], [], []
    for start in range(0, grid.shape_dhw[0], depth_chunk_size):
        for depth in range(start, min(start + depth_chunk_size, grid.shape_dhw[0])):
            result = reconstruct_plane(state, plane_from_target_grid(grid, depth), modality_id=modality_id, render_config=render_config)
            intensities.append(result.intensity); supports.append(result.support_mass)
            masks.append(result.unsupported_mask); uncertainties.append(result.support_uncertainty)
    intensity = torch.stack(intensities); support = torch.stack(supports)
    unsupported = torch.stack(masks); uncertainty = torch.stack(uncertainties)
    renderer_config_hash = hashlib.sha256(render_config.renderer_version.encode()).hexdigest()
    artifact_hash = volume_output_hash(
        patient_id=state.patient_id, modality_id=modality_id, grid=grid,
        intensity=intensity, support_mass=support, unsupported_mask=unsupported,
        support_uncertainty=uncertainty, depth_chunk_size=depth_chunk_size,
        renderer_config_hash=renderer_config_hash, patient_state_version=state.state_version,
    )
    return VolumeReconstruction(
        patient_id=state.patient_id, modality_id=modality_id, grid=grid,
        intensity=intensity, support_mass=support, unsupported_mask=unsupported,
        support_uncertainty=uncertainty, depth_chunk_size=depth_chunk_size,
        renderer_config_hash=renderer_config_hash, patient_state_version=state.state_version,
        artifact_hash=artifact_hash,
    )
