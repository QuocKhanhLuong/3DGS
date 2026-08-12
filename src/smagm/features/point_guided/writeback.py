"""Gate-C C6 compact physical 4-mm writes into dynamic tri-plane state."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .spectral_query import FeatureGridGeometry
from .state_init import DynamicTriPlanes
from .trajectory_cost import LOCKED_SUPPORT_RADIUS_MM
from .updater import PlaneCorrections


def _minimum_plane_spacing_mm(feature_geometry: FeatureGridGeometry, axes_dhw: tuple[int, int]) -> float:
    """Return a conservative physical lower bound for a two-axis feature lattice."""

    affine = feature_geometry.feature_geometry.voxel_to_ras_mm
    # VolumeGeometry consumes WHD; tensor DHW axis i therefore uses affine
    # spatial column 2-i.  The smallest singular value protects shear cases.
    columns = [[affine[row][2 - axis] for axis in axes_dhw] for row in range(3)]
    singular_values = torch.linalg.svdvals(torch.tensor(columns, dtype=torch.float64))
    minimum = float(singular_values[-1].item())
    if not math.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("feature-grid plane axes must define a nonsingular physical metric")
    return minimum


def _local_window(coordinate: Tensor, length: int, *, radius_mm: float, minimum_spacing_mm: float) -> tuple[int, int]:
    radius_index = int(math.ceil(radius_mm / minimum_spacing_mm))
    centre = float(coordinate.detach().item())
    start = max(0, int(math.floor(centre)) - radius_index)
    stop = min(length, int(math.ceil(centre)) + radius_index + 1)
    if start >= stop:
        raise ValueError("compact write has no valid retained-plane support window")
    return start, stop


def _write_plane(
    plane: Tensor,
    point_ras_mm: Tensor,
    feature_dhw: Tensor,
    correction: Tensor,
    feature_geometry: FeatureGridGeometry,
    *,
    plane_name: str,
    radius_mm: float,
) -> Tensor:
    if plane_name == "xy":
        row_axis, column_axis, omitted_axis = 1, 2, 0
    elif plane_name == "xz":
        row_axis, column_axis, omitted_axis = 0, 2, 1
    elif plane_name == "yz":
        row_axis, column_axis, omitted_axis = 0, 1, 2
    else:
        raise ValueError(f"unknown dynamic plane {plane_name!r}")
    row_start, row_stop = _local_window(
        feature_dhw[row_axis],
        plane.shape[-2],
        radius_mm=radius_mm,
        minimum_spacing_mm=_minimum_plane_spacing_mm(feature_geometry, (row_axis, column_axis)),
    )
    column_start, column_stop = _local_window(
        feature_dhw[column_axis],
        plane.shape[-1],
        radius_mm=radius_mm,
        minimum_spacing_mm=_minimum_plane_spacing_mm(feature_geometry, (row_axis, column_axis)),
    )
    rows = torch.arange(row_start, row_stop, dtype=plane.dtype, device=plane.device)
    columns = torch.arange(column_start, column_stop, dtype=plane.dtype, device=plane.device)
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    coordinates = [row_grid * 0.0 + feature_dhw[index] for index in range(3)]
    coordinates[row_axis] = row_grid
    coordinates[column_axis] = column_grid
    coordinates[omitted_axis] = row_grid * 0.0 + feature_dhw[omitted_axis]
    locations_ras_mm = feature_geometry.feature_dhw_to_ras_mm(torch.stack(coordinates, dim=-1))
    distance_mm = torch.linalg.vector_norm(locations_ras_mm - point_ras_mm, dim=-1)
    weight = torch.square(torch.clamp(1.0 - distance_mm / radius_mm, min=0.0))
    result = plane.clone()
    result[:, row_start:row_stop, column_start:column_stop] = (
        result[:, row_start:row_stop, column_start:column_stop] + correction[:, None, None] * weight.unsqueeze(0)
    )
    return result


class CompactTriPlaneWriteback(nn.Module):
    """Parameter-free local quadratic write respecting full feature-grid affine geometry."""

    def __init__(self, *, support_radius_mm: float = LOCKED_SUPPORT_RADIUS_MM) -> None:
        super().__init__()
        if float(support_radius_mm) != LOCKED_SUPPORT_RADIUS_MM:
            raise ValueError("support_radius_mm must be exactly 4.0 mm in Gate-C MAIN")
        self.support_radius_mm = float(support_radius_mm)

    def forward(
        self,
        state: DynamicTriPlanes,
        selected_points_ras_mm: Tensor,
        corrections: PlaneCorrections,
        feature_geometry: FeatureGridGeometry,
    ) -> DynamicTriPlanes:
        if not isinstance(state, DynamicTriPlanes) or not isinstance(corrections, PlaneCorrections):
            raise TypeError("state and corrections must use the Gate-C typed contracts")
        if not isinstance(feature_geometry, FeatureGridGeometry):
            raise TypeError("feature_geometry must be a FeatureGridGeometry")
        if not isinstance(selected_points_ras_mm, Tensor) or selected_points_ras_mm.shape != (state.xy.shape[0], 3):
            raise ValueError("selected_points_ras_mm must have shape [B,3]")
        if not selected_points_ras_mm.is_floating_point() or selected_points_ras_mm.dtype != state.xy.dtype or selected_points_ras_mm.device != state.xy.device or not bool(torch.isfinite(selected_points_ras_mm).all()):
            raise ValueError("selected_points_ras_mm must be finite and match state dtype/device")
        if corrections.xy.shape[0] != state.xy.shape[0] or corrections.xy.dtype != state.xy.dtype or corrections.xy.device != state.xy.device:
            raise ValueError("corrections must match dynamic state batch, dtype, and device")
        outputs: dict[str, list[Tensor]] = {"xy": [], "xz": [], "yz": []}
        for batch in range(state.xy.shape[0]):
            feature_dhw = feature_geometry.ras_mm_to_feature_dhw(selected_points_ras_mm[batch : batch + 1])[0]
            for name in ("xy", "xz", "yz"):
                outputs[name].append(
                    _write_plane(
                        getattr(state, name)[batch],
                        selected_points_ras_mm[batch],
                        feature_dhw,
                        getattr(corrections, name)[batch],
                        feature_geometry,
                        plane_name=name,
                        radius_mm=self.support_radius_mm,
                    )
                )
        return DynamicTriPlanes(xy=torch.stack(outputs["xy"]), xz=torch.stack(outputs["xz"]), yz=torch.stack(outputs["yz"]))


__all__ = ["CompactTriPlaneWriteback"]
