"""Deterministic fixed support points for matched T1-A encoder comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import torch

from ..contracts.coordinates import PhysicalPlane
from ..features.contracts import EncoderFeatureMaps


@dataclass(frozen=True)
class FixedSupportConfig:
    """Row-major feature-grid support selection with no learned topology."""

    step_vu: Sequence[int] = (4, 4)
    border_vu: Sequence[int] = (0, 0)
    max_points: int | None = None
    # Kept only as an explicit incompatibility guard for callers of an early
    # T1-A draft.  Support topology must never depend on learned reliability.
    minimum_reliability: float = 0.0

    def __post_init__(self) -> None:
        step = tuple(self.step_vu)
        border = tuple(self.border_vu)
        if len(step) != 2 or any(not isinstance(value, int) or value <= 0 for value in step):
            raise ValueError("step_vu must contain two positive integers")
        if len(border) != 2 or any(not isinstance(value, int) or value < 0 for value in border):
            raise ValueError("border_vu must contain two non-negative integers")
        if self.max_points is not None and (not isinstance(self.max_points, int) or self.max_points <= 0):
            raise ValueError("max_points must be None or a positive integer")
        if not isinstance(self.minimum_reliability, (int, float)) or not 0.0 <= float(self.minimum_reliability) <= 1.0:
            raise ValueError("minimum_reliability must lie in [0, 1]")
        if float(self.minimum_reliability) != 0.0:
            raise ValueError("minimum_reliability is forbidden: support topology must be value-independent")
        object.__setattr__(self, "step_vu", step)
        object.__setattr__(self, "border_vu", border)
        object.__setattr__(self, "minimum_reliability", float(self.minimum_reliability))


@dataclass(frozen=True)
class FixedSupportBatch:
    """Physical support centres and sampled compact feature vectors."""

    centers_ras_mm: torch.Tensor  # [N, 3]
    feature_vectors: torch.Tensor  # [N, C]
    feature_indices_vu: torch.Tensor  # [N, 2], long
    reliability: torch.Tensor  # [N, 1]
    observation_ids: tuple[str, ...]
    source_plane_hashes: tuple[str, ...]
    batch_index: int
    support_basis_ras: torch.Tensor  # [N, 3, 3], row basis (u, v, signed normal)

    def __post_init__(self) -> None:
        if self.centers_ras_mm.ndim != 2 or self.centers_ras_mm.shape[1] != 3:
            raise ValueError("centers_ras_mm must have shape [N, 3]")
        count = self.centers_ras_mm.shape[0]
        if count <= 0:
            raise ValueError("fixed support selection must produce at least one point")
        if self.feature_vectors.ndim != 2 or self.feature_vectors.shape[0] != count:
            raise ValueError("feature_vectors must have shape [N, C]")
        if self.feature_indices_vu.shape != (count, 2) or self.feature_indices_vu.dtype not in (torch.int32, torch.int64):
            raise ValueError("feature_indices_vu must be integer with shape [N, 2]")
        if self.reliability.shape != (count, 1):
            raise ValueError("reliability must have shape [N, 1]")
        if len(self.observation_ids) != count:
            raise ValueError("observation_ids must contain one ID per support")
        if any(not isinstance(value, str) or not value for value in self.observation_ids):
            raise ValueError("observation_ids must be non-empty strings")
        if (
            len(self.source_plane_hashes) != count
            or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in self.source_plane_hashes)
        ):
            raise ValueError("source_plane_hashes must contain one canonical SHA-256 digest per support")
        if not isinstance(self.batch_index, int) or self.batch_index < 0:
            raise ValueError("batch_index must be a non-negative integer")
        device = self.centers_ras_mm.device
        dtype = self.centers_ras_mm.dtype
        if self.feature_vectors.device != device or self.reliability.device != device or self.feature_indices_vu.device != device:
            raise ValueError("all support tensors must share device")
        if self.feature_vectors.dtype != dtype or self.reliability.dtype != dtype:
            raise ValueError("floating support tensors must share dtype")
        if not bool(torch.isfinite(self.centers_ras_mm).all()) or not bool(torch.isfinite(self.feature_vectors).all()):
            raise ValueError("support tensors must be finite")
        basis = self.support_basis_ras
        if (
            not isinstance(basis, torch.Tensor)
            or basis.shape != (count, 3, 3)
            or basis.dtype != dtype
            or basis.device != device
            or not bool(torch.isfinite(basis).all())
        ):
            raise ValueError("support_basis_ras must be finite with shape [N, 3, 3] on the support device")
        axis_u, axis_v, signed_normal = basis.unbind(dim=1)
        tolerance = 1e-5 if dtype is torch.float32 else 1e-10
        if (
            not bool(torch.allclose(axis_u.norm(dim=-1), torch.ones(count, dtype=dtype, device=device), atol=tolerance, rtol=0.0))
            or not bool(torch.allclose(axis_v.norm(dim=-1), torch.ones(count, dtype=dtype, device=device), atol=tolerance, rtol=0.0))
            or not bool(torch.allclose(signed_normal.norm(dim=-1), torch.ones(count, dtype=dtype, device=device), atol=tolerance, rtol=0.0))
            or not bool(torch.all((axis_u * axis_v).sum(dim=-1).abs() <= tolerance))
            or not bool(torch.all((axis_u * signed_normal).sum(dim=-1).abs() <= tolerance))
            or not bool(torch.all((axis_v * signed_normal).sum(dim=-1).abs() <= tolerance))
            or not bool(torch.all(torch.cross(axis_u, axis_v, dim=-1).mul(signed_normal).sum(dim=-1).abs() >= 1.0 - tolerance))
        ):
            raise ValueError("support_basis_ras rows must be an orthonormal (u, v, signed-normal) RAS basis")
        object.__setattr__(self, "observation_ids", tuple(self.observation_ids))
        object.__setattr__(self, "source_plane_hashes", tuple(self.source_plane_hashes))
        object.__setattr__(self, "support_basis_ras", basis)

    @property
    def count(self) -> int:
        return self.centers_ras_mm.shape[0]


def _grid_indices(height: int, width: int, config: FixedSupportConfig) -> list[tuple[int, int]]:
    border_v, border_u = config.border_vu
    if border_v * 2 >= height or border_u * 2 >= width:
        raise ValueError("border_vu removes the complete feature grid")
    indices = [
        (v, u)
        for v in range(border_v, height - border_v, config.step_vu[0])
        for u in range(border_u, width - border_u, config.step_vu[1])
    ]
    if not indices:
        raise ValueError("fixed support configuration produced no points")
    return indices


def sample_fixed_supports(
    features: EncoderFeatureMaps,
    plane: PhysicalPlane,
    *,
    batch_index: int = 0,
    observation_id: str | None = None,
    config: FixedSupportConfig | None = None,
) -> FixedSupportBatch:
    """Sample one deterministic support set from compact feature maps.

    Selection is row-major and independent of pixel values.  The same config
    therefore yields identical support topology for E0, E1, and E2.
    """

    if not isinstance(features, EncoderFeatureMaps) or not isinstance(plane, PhysicalPlane):
        raise TypeError("features and plane must use T1-A contract types")
    if not isinstance(batch_index, int) or not 0 <= batch_index < features.batch_size:
        raise IndexError("batch_index is outside the feature batch")
    bound_plane = features.grid_to_plane.input_plane
    if bound_plane is None:
        raise ValueError("fixed support sampling requires a FeatureGridToPlaneTransform bound to its source PhysicalPlane")
    if plane.canonical_json() != bound_plane.canonical_json():
        raise ValueError("support sampling plane must exactly match the transform-bound canonical source PhysicalPlane")
    config = config or FixedSupportConfig()
    feature_height, feature_width = features.feature_shape_hw
    indices = _grid_indices(feature_height, feature_width, config)
    concatenated = features.concatenated()[batch_index]
    valid_indices = [
        (v, u)
        for v, u in indices
        if bool(features.valid_feature_mask[batch_index, 0, v, u].detach().cpu())
    ]
    if not valid_indices:
        raise ValueError("no deterministic support lies on the declared valid feature grid")
    if config.max_points is not None:
        valid_indices = valid_indices[: config.max_points]
    device = concatenated.device
    dtype = concatenated.dtype
    index_tensor = torch.tensor(valid_indices, dtype=torch.long, device=device)
    sampled = concatenated[:, index_tensor[:, 0], index_tensor[:, 1]].transpose(0, 1)
    reliability = features.reliability[batch_index, :, index_tensor[:, 0], index_tensor[:, 1]].transpose(0, 1)
    world = [
        features.grid_to_plane.world_from_feature_vu(plane, float(v), float(u))
        for v, u in valid_indices
    ]
    centers = torch.tensor(world, dtype=dtype, device=device)
    resolved_observation_id = bound_plane.observation_id
    if not resolved_observation_id:
        raise ValueError("the transform-bound source plane requires an observation_id")
    if observation_id is not None and observation_id != resolved_observation_id:
        raise ValueError("observation_id override does not match the transform-bound source plane")
    source_plane_hash = features.grid_to_plane.source_plane_hash
    return FixedSupportBatch(
        centers_ras_mm=centers,
        feature_vectors=sampled,
        feature_indices_vu=index_tensor,
        reliability=reliability,
        observation_ids=(resolved_observation_id,) * len(valid_indices),
        source_plane_hashes=(source_plane_hash,) * len(valid_indices),
        batch_index=batch_index,
        support_basis_ras=torch.tensor(
            (plane.axis_u_ras, plane.axis_v_ras, plane.signed_normal_ras),
            dtype=dtype,
            device=device,
        ).expand(len(valid_indices), -1, -1),
    )
