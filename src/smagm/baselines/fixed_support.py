"""Deterministic fixed support points for matched T1-A encoder comparisons."""

from __future__ import annotations

from dataclasses import dataclass
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
    batch_index: int

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
        object.__setattr__(self, "observation_ids", tuple(self.observation_ids))

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
    if config.max_points is not None:
        indices = indices[: config.max_points]
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
    config = config or FixedSupportConfig()
    feature_height, feature_width = features.feature_shape_hw
    indices = _grid_indices(feature_height, feature_width, config)
    concatenated = features.concatenated()[batch_index]
    reliable_indices = [
        (v, u)
        for v, u in indices
        if float(features.reliability[batch_index, 0, v, u].detach().cpu()) >= config.minimum_reliability
    ]
    if not reliable_indices:
        raise ValueError("no deterministic support satisfies minimum_reliability")
    device = concatenated.device
    dtype = concatenated.dtype
    index_tensor = torch.tensor(reliable_indices, dtype=torch.long, device=device)
    sampled = concatenated[:, index_tensor[:, 0], index_tensor[:, 1]].transpose(0, 1)
    reliability = features.reliability[batch_index, :, index_tensor[:, 0], index_tensor[:, 1]].transpose(0, 1)
    world = [
        features.grid_to_plane.world_from_feature_vu(plane, float(v), float(u))
        for v, u in reliable_indices
    ]
    centers = torch.tensor(world, dtype=dtype, device=device)
    resolved_observation_id = observation_id or plane.observation_id
    if not resolved_observation_id:
        raise ValueError("an observation_id is required on the plane or call")
    return FixedSupportBatch(
        centers_ras_mm=centers,
        feature_vectors=sampled,
        feature_indices_vu=index_tensor,
        reliability=reliability,
        observation_ids=(resolved_observation_id,) * len(reliable_indices),
        batch_index=batch_index,
    )
