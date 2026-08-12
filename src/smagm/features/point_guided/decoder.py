"""Gate-D D1: chunked, geometry-aware implicit decoding from final ``Z`` only.

The decoder intentionally has a narrow boundary.  It samples the final
``DynamicTriPlanes`` at full-resolution output voxel centres through the
already-derived ``FeatureGridGeometry`` and applies one shared MLP.  It does
not accept observations, static planes, spectral evidence, route diagnostics,
or target data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .contracts import VolumeGeometry
from .reward import DynamicStatePointQuery
from .sampling import voxel_dhw_to_ras_mm
from .spectral_query import FeatureGridGeometry
from .state_init import DYNAMIC_STATE_CHANNELS, DynamicTriPlanes


TRIPLANE_QUERY_CHANNELS = 3 * DYNAMIC_STATE_CHANNELS
DECODER_HIDDEN_CHANNELS = (64, 32)
DECODER_OUTPUT_CHANNELS = 1


def _require_positive_chunk_size(chunk_size: int) -> None:
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")


@dataclass(frozen=True)
class ReconstructionOutput:
    """Narrow Gate-D inference result: an absolute predicted volume only."""

    prediction: Tensor  # [B, 1, D, H, W]
    geometry: VolumeGeometry

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, VolumeGeometry):
            raise TypeError("geometry must be a VolumeGeometry")
        if not isinstance(self.prediction, Tensor) or self.prediction.ndim != 5 or not self.prediction.is_floating_point():
            raise ValueError("prediction must be a floating tensor [B, 1, D, H, W]")
        if self.prediction.shape[0] <= 0 or self.prediction.shape[1] != DECODER_OUTPUT_CHANNELS:
            raise ValueError("prediction must have positive batch size and exactly one output channel")
        if tuple(self.prediction.shape[-3:]) != self.geometry.shape_dhw:
            raise ValueError("prediction spatial shape must match geometry.shape_dhw")
        if not bool(torch.isfinite(self.prediction).all()):
            raise ValueError("prediction must be finite")


class DynamicTriPlaneVoxelQuery(nn.Module):
    """Parameter-free XY/XZ/YZ sampling of final dynamic state at RAS points."""

    def __init__(self) -> None:
        super().__init__()
        self._point_query = DynamicStatePointQuery()

    def forward(
        self,
        final_state: DynamicTriPlanes,
        points_ras_mm: Tensor,
        feature_geometry: FeatureGridGeometry,
    ) -> Tensor:
        """Return exact ``[Zxy | Zxz | Zyz]`` samples with width 96."""

        samples = self._point_query(final_state, points_ras_mm, feature_geometry)
        packed = samples.packed
        if packed.shape[-1] != TRIPLANE_QUERY_CHANNELS:
            raise RuntimeError("Gate-D tri-plane query must pack exactly 96 channels")
        return packed


class ImplicitTriPlaneDecoder(nn.Module):
    """The locked shared ``96 -> 64 -> 32 -> 1`` absolute voxel decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.voxel_query = DynamicTriPlaneVoxelQuery()
        self.mlp = nn.Sequential(
            nn.Linear(TRIPLANE_QUERY_CHANNELS, DECODER_HIDDEN_CHANNELS[0], bias=True),
            nn.SiLU(),
            nn.Linear(DECODER_HIDDEN_CHANNELS[0], DECODER_HIDDEN_CHANNELS[1], bias=True),
            nn.SiLU(),
            nn.Linear(DECODER_HIDDEN_CHANNELS[1], DECODER_OUTPUT_CHANNELS, bias=True),
        )

    @staticmethod
    def _validate_state_feature_geometry(
        final_state: DynamicTriPlanes,
        feature_geometry: FeatureGridGeometry,
    ) -> None:
        if not isinstance(final_state, DynamicTriPlanes):
            raise TypeError("final_state must be a DynamicTriPlanes instance")
        if not isinstance(feature_geometry, FeatureGridGeometry):
            raise TypeError("feature_geometry must be a FeatureGridGeometry instance")
        depth, height, width = feature_geometry.shape_dhw
        expected_planes = (
            (final_state.xy, (height, width)),
            (final_state.xz, (depth, width)),
            (final_state.yz, (depth, height)),
        )
        if any(tuple(plane.shape[-2:]) != shape for plane, shape in expected_planes):
            raise ValueError("final dynamic planes must retain the selected-feature grids")

    @classmethod
    def _validate_geometry(
        cls,
        final_state: DynamicTriPlanes,
        feature_geometry: FeatureGridGeometry,
        output_geometry: VolumeGeometry,
    ) -> None:
        cls._validate_state_feature_geometry(final_state, feature_geometry)
        if not isinstance(output_geometry, VolumeGeometry):
            raise TypeError("output_geometry must be a VolumeGeometry instance")
        if output_geometry != feature_geometry.source_geometry:
            raise ValueError("output_geometry must be the source geometry paired with feature_geometry")

    def decode_points(
        self,
        final_state: DynamicTriPlanes,
        points_ras_mm: Tensor,
        feature_geometry: FeatureGridGeometry,
    ) -> Tensor:
        """Decode target-free RAS-mm point queries as ``[B, N, 1]`` scalars.

        This is the same parameter-free tri-plane query and locked shared MLP
        used by dense decoding.  It deliberately accepts neither an
        observation nor a target, so Gate-E local and collateral supervision
        can query only the required physical points without creating a full
        output volume.
        """

        self._validate_state_feature_geometry(final_state, feature_geometry)
        parameter = self.mlp[0].weight
        if final_state.xy.device != parameter.device or final_state.xy.dtype != parameter.dtype:
            raise ValueError("final_state and decoder parameters must share dtype and device")
        z_features = self.voxel_query(final_state, points_ras_mm, feature_geometry)
        prediction = self.mlp(z_features)
        if prediction.shape != (*points_ras_mm.shape[:2], DECODER_OUTPUT_CHANNELS):
            raise RuntimeError("pointwise decoder must return [B, N, 1] scalars")
        return prediction

    @staticmethod
    def _chunk_voxel_dhw(
        start: int,
        stop: int,
        *,
        height: int,
        width: int,
        batch: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """Create one flat output-lattice chunk, never a dense feature volume."""

        flat = torch.arange(start, stop, dtype=dtype, device=device)
        plane_area = height * width
        d = torch.div(flat, plane_area, rounding_mode="floor")
        hw = flat - d * plane_area
        h = torch.div(hw, width, rounding_mode="floor")
        w = hw - h * width
        voxel_dhw = torch.stack((d, h, w), dim=-1)
        return voxel_dhw.unsqueeze(0).expand(batch, -1, -1)

    def forward(
        self,
        final_state: DynamicTriPlanes,
        feature_geometry: FeatureGridGeometry,
        output_geometry: VolumeGeometry,
        *,
        chunk_size: int,
    ) -> Tensor:
        """Decode final ``Z`` into an absolute ``[B,1,D,H,W]`` prediction.

        Every intermediate query is ``[B, chunk, 96]``.  The only whole-volume
        tensor created is the required scalar output.
        """

        _require_positive_chunk_size(chunk_size)
        self._validate_geometry(final_state, feature_geometry, output_geometry)
        batch = final_state.xy.shape[0]
        depth, height, width = output_geometry.shape_dhw
        voxel_count = depth * height * width
        scalar_chunks: list[Tensor] = []
        for start in range(0, voxel_count, chunk_size):
            stop = min(start + chunk_size, voxel_count)
            output_voxels = self._chunk_voxel_dhw(
                start,
                stop,
                height=height,
                width=width,
                batch=batch,
                dtype=final_state.xy.dtype,
                device=final_state.xy.device,
            )
            points_ras_mm = voxel_dhw_to_ras_mm(output_voxels, output_geometry)
            scalar_chunks.append(self.decode_points(final_state, points_ras_mm, feature_geometry).squeeze(-1))
        prediction = torch.cat(scalar_chunks, dim=1).reshape(batch, DECODER_OUTPUT_CHANNELS, depth, height, width)
        return prediction


__all__ = [
    "DECODER_HIDDEN_CHANNELS",
    "DECODER_OUTPUT_CHANNELS",
    "DynamicTriPlaneVoxelQuery",
    "ImplicitTriPlaneDecoder",
    "ReconstructionOutput",
    "TRIPLANE_QUERY_CHANNELS",
]
