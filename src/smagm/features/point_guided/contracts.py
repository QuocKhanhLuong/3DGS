"""Typed tensor, physical-coordinate, and sparse-output contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch


class PointGuidedGeometryError(ValueError):
    """Raised when tensor and physical-volume geometry disagree."""


class EmptySparseSupportError(RuntimeError):
    """Raised when no positive compact-support PoU contribution exists.

    When local spatial support exists but every semantic denominator is zero,
    the sparse unsupported voxel record is attached to the exception.  This
    preserves fail-closed execution while keeping the unsupported evidence
    inspectable without returning an invalid all-zero PoU object.
    """

    def __init__(
        self,
        message: str,
        *,
        unsupported_batch_indices: torch.Tensor | None = None,
        unsupported_voxel_indices_dhw: torch.Tensor | None = None,
    ) -> None:
        super().__init__(message)
        self.unsupported_batch_indices = unsupported_batch_indices
        self.unsupported_voxel_indices_dhw = unsupported_voxel_indices_dhw


def _shape_dhw(value: Sequence[int]) -> tuple[int, int, int]:
    raw = tuple(value)
    if len(raw) != 3 or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in raw):
        raise PointGuidedGeometryError("shape_dhw must contain three positive integers")
    return raw  # type: ignore[return-value]


def _matrix4(value: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    if len(value) != 4 or any(len(row) != 4 for row in value):
        raise PointGuidedGeometryError("voxel_to_ras_mm must be a 4 by 4 matrix")
    matrix = tuple(tuple(float(item) for item in row) for row in value)
    if not all(math.isfinite(item) for row in matrix for item in row):
        raise PointGuidedGeometryError("voxel_to_ras_mm must be finite")
    if matrix[3] != (0.0, 0.0, 0.0, 1.0):
        raise PointGuidedGeometryError("voxel_to_ras_mm must be homogeneous")
    spatial = torch.tensor([row[:3] for row in matrix[:3]], dtype=torch.float64)
    if abs(float(torch.linalg.det(spatial))) <= 1e-10:
        raise PointGuidedGeometryError("voxel_to_ras_mm has a singular spatial block")
    return matrix


@dataclass(frozen=True)
class VolumeGeometry:
    """Map tensor voxel centres ``[d, h, w]`` to canonical RAS ``XYZ`` mm.

    ``voxel_to_ras_mm`` consumes homogeneous ``[w, h, d, 1]`` column vectors.
    Its columns therefore preserve a source affine's physical axes while the
    frontend itself consistently stores volume tensors as ``[D, H, W]``.
    """

    shape_dhw: Sequence[int]
    voxel_to_ras_mm: Sequence[Sequence[float]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape_dhw", _shape_dhw(self.shape_dhw))
        object.__setattr__(self, "voxel_to_ras_mm", _matrix4(self.voxel_to_ras_mm))

    @classmethod
    def from_spacing(
        cls,
        shape_dhw: Sequence[int],
        spacing_xyz_mm: Sequence[float] = (1.0, 1.0, 1.0),
        origin_ras_mm: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> "VolumeGeometry":
        spacing = tuple(float(item) for item in spacing_xyz_mm)
        origin = tuple(float(item) for item in origin_ras_mm)
        if len(spacing) != 3 or len(origin) != 3 or any(item <= 0.0 or not math.isfinite(item) for item in spacing):
            raise PointGuidedGeometryError("spacing_xyz_mm must contain three positive finite values")
        if not all(math.isfinite(item) for item in origin):
            raise PointGuidedGeometryError("origin_ras_mm must contain three finite values")
        return cls(
            shape_dhw=shape_dhw,
            voxel_to_ras_mm=(
                (spacing[0], 0.0, 0.0, origin[0]),
                (0.0, spacing[1], 0.0, origin[1]),
                (0.0, 0.0, spacing[2], origin[2]),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )

    @property
    def spacing_xyz_mm(self) -> tuple[float, float, float]:
        return tuple(
            math.sqrt(sum(self.voxel_to_ras_mm[row][column] ** 2 for row in range(3)))
            for column in range(3)
        )


def _float_tensor(name: str, tensor: torch.Tensor, dimensions: int) -> None:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != dimensions:
        raise ValueError(f"{name} must be a rank-{dimensions} torch.Tensor")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")


def validate_probability_simplex(
    name: str,
    tensor: torch.Tensor,
    *,
    class_dimension: int,
) -> None:
    """Fail closed unless a public semantic tensor is a probability simplex."""

    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if tensor.shape[class_dimension] <= 1:
        raise ValueError(f"{name} must have more than one semantic class")
    tolerance = max(1e-5, 8.0 * torch.finfo(tensor.dtype).eps)
    if bool((tensor < -tolerance).any()) or bool((tensor > 1.0 + tolerance).any()):
        raise ValueError(f"{name} must lie in [0, 1]")
    sums = tensor.sum(dim=class_dimension)
    if not bool(torch.allclose(sums, torch.ones_like(sums), atol=tolerance, rtol=tolerance)):
        raise ValueError(f"{name} must sum to one along its semantic-class dimension")


@dataclass(frozen=True)
class PointField:
    """The frozen per-forward point state in RAS millimetres."""

    original_centers_ras_mm: torch.Tensor  # [B, N, 3]
    refined_centers_ras_mm: torch.Tensor  # [B, N, 3]
    displacement_ras_mm: torch.Tensor  # [B, N, 3]
    semantic_vectors: torch.Tensor  # [B, N, K]
    support_radius_mm: float

    def __post_init__(self) -> None:
        for name in ("original_centers_ras_mm", "refined_centers_ras_mm", "displacement_ras_mm", "semantic_vectors"):
            _float_tensor(name, getattr(self, name), 3)
        original = self.original_centers_ras_mm
        if original.shape[0] <= 0 or original.shape[1] <= 0 or original.shape[-1] != 3 or self.refined_centers_ras_mm.shape != original.shape or self.displacement_ras_mm.shape != original.shape:
            raise ValueError("all point centre and displacement tensors must have shape [B, N, 3]")
        if self.semantic_vectors.shape[:2] != original.shape[:2] or self.semantic_vectors.shape[-1] <= 1:
            raise ValueError("semantic_vectors must have shape [B, N, K] with K > 1")
        validate_probability_simplex(
            "semantic_vectors",
            self.semantic_vectors,
            class_dimension=-1,
        )
        if len({original.device, self.refined_centers_ras_mm.device, self.displacement_ras_mm.device, self.semantic_vectors.device}) != 1:
            raise ValueError("point-field tensors must share one device")
        if not math.isfinite(float(self.support_radius_mm)) or self.support_radius_mm <= 0.0:
            raise ValueError("support_radius_mm must be positive and finite")
        if not math.isclose(float(self.support_radius_mm), 4.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("PointField support_radius_mm is locked to exactly 4.0 mm")
        if not torch.allclose(
            self.refined_centers_ras_mm - self.original_centers_ras_mm,
            self.displacement_ras_mm,
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("displacement_ras_mm must be measured from original centres")
        if bool((torch.linalg.vector_norm(self.displacement_ras_mm, dim=-1) > 2.0 + 1e-6).any()):
            raise ValueError("PointField displacement_ras_mm is locked to at most 2.0 mm from its original centre")


@dataclass(frozen=True)
class SparsePoU:
    """Ragged compact-support edges plus explicitly unsupported local voxels.

    ``unsupported_*`` records voxel centres that lay in at least one spatial
    support but had a zero semantic-affinity denominator.  It is sparse like
    the positive edge list; it is not a dense point-by-volume allocation.
    """

    batch_indices: torch.Tensor  # [E], int64
    voxel_indices_dhw: torch.Tensor  # [E, 3], int64
    point_indices: torch.Tensor  # [E], int64
    raw_affinity: torch.Tensor  # [E], float
    normalized_weight: torch.Tensor  # [E], float
    unsupported_batch_indices: torch.Tensor  # [U], int64
    unsupported_voxel_indices_dhw: torch.Tensor  # [U, 3], int64
    volume_shape_dhw: Sequence[int]

    def __post_init__(self) -> None:
        edges = self.raw_affinity.numel()
        for name in ("batch_indices", "point_indices"):
            tensor = getattr(self, name)
            if tensor.ndim != 1 or tensor.numel() != edges or tensor.dtype != torch.long:
                raise ValueError(f"{name} must be int64 with shape [E]")
        if self.voxel_indices_dhw.shape != (edges, 3) or self.voxel_indices_dhw.dtype != torch.long:
            raise ValueError("voxel_indices_dhw must be int64 with shape [E, 3]")
        unsupported = self.unsupported_batch_indices.numel()
        if (
            self.unsupported_batch_indices.ndim != 1
            or self.unsupported_batch_indices.dtype != torch.long
        ):
            raise ValueError("unsupported_batch_indices must be int64 with shape [U]")
        if (
            self.unsupported_voxel_indices_dhw.shape != (unsupported, 3)
            or self.unsupported_voxel_indices_dhw.dtype != torch.long
        ):
            raise ValueError("unsupported_voxel_indices_dhw must be int64 with shape [U, 3]")
        if edges == 0:
            raise EmptySparseSupportError("sparse PoU has no positive compact-support contributions")
        for name in ("raw_affinity", "normalized_weight"):
            tensor = getattr(self, name)
            if tensor.ndim != 1 or tensor.numel() != edges or not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must be finite floating point with shape [E]")
            if bool((tensor < 0.0).any()):
                raise ValueError(f"{name} must be non-negative")
        if not bool((self.raw_affinity > 0.0).all()) or not bool((self.normalized_weight > 0.0).all()):
            raise ValueError("SparsePoU stores only positive affinity edges")
        if len({self.batch_indices.device, self.voxel_indices_dhw.device, self.point_indices.device, self.raw_affinity.device, self.normalized_weight.device, self.unsupported_batch_indices.device, self.unsupported_voxel_indices_dhw.device}) != 1:
            raise ValueError("SparsePoU tensors must share one device")
        object.__setattr__(self, "volume_shape_dhw", _shape_dhw(self.volume_shape_dhw))
        depth, height, width = self.volume_shape_dhw
        upper = torch.tensor((depth, height, width), dtype=torch.long, device=self.voxel_indices_dhw.device)
        if bool((self.batch_indices < 0).any()) or bool((self.point_indices < 0).any()) or bool((self.unsupported_batch_indices < 0).any()):
            raise ValueError("SparsePoU batch and point indices must be non-negative")
        if bool((self.voxel_indices_dhw < 0).any()) or bool((self.voxel_indices_dhw >= upper).any()):
            raise ValueError("SparsePoU voxel indices must lie within volume_shape_dhw")
        if bool((self.unsupported_voxel_indices_dhw < 0).any()) or bool((self.unsupported_voxel_indices_dhw >= upper).any()):
            raise ValueError("SparsePoU unsupported voxel indices must lie within volume_shape_dhw")
        linear_voxels = (
            self.batch_indices * (depth * height * width)
            + self.voxel_indices_dhw[:, 0] * (height * width)
            + self.voxel_indices_dhw[:, 1] * width
            + self.voxel_indices_dhw[:, 2]
        )
        unique_voxels, inverse = torch.unique(linear_voxels, sorted=True, return_inverse=True)
        denominators = torch.zeros(
            unique_voxels.numel(),
            dtype=self.normalized_weight.dtype,
            device=self.normalized_weight.device,
        )
        denominators.scatter_add_(0, inverse, self.normalized_weight)
        tolerance = max(1e-5, 8.0 * torch.finfo(self.normalized_weight.dtype).eps)
        if not bool(torch.allclose(denominators, torch.ones_like(denominators), atol=tolerance, rtol=tolerance)):
            raise ValueError("SparsePoU normalized weights must sum to one for every supported voxel")
        edge_keys = torch.cat(
            (
                self.batch_indices.unsqueeze(1),
                self.voxel_indices_dhw,
                self.point_indices.unsqueeze(1),
            ),
            dim=1,
        )
        if torch.unique(edge_keys, dim=0).shape[0] != edges:
            raise ValueError("SparsePoU must not contain duplicate batch/voxel/point edges")
        unsupported_linear = (
            self.unsupported_batch_indices * (depth * height * width)
            + self.unsupported_voxel_indices_dhw[:, 0] * (height * width)
            + self.unsupported_voxel_indices_dhw[:, 1] * width
            + self.unsupported_voxel_indices_dhw[:, 2]
        )
        if torch.unique(unsupported_linear).numel() != unsupported:
            raise ValueError("SparsePoU unsupported voxel records must be unique")
        if bool(torch.isin(unsupported_linear, unique_voxels).any()):
            raise ValueError("SparsePoU supported and unsupported voxel records must be disjoint")


@dataclass(frozen=True)
class FrontendOutput:
    """Typed output of the fully implemented frontend-only forward path."""

    s_coarse: torch.Tensor  # [B, K, D, H, W], soft probabilities
    initial_points_ras_mm: torch.Tensor  # [B, N, 3]
    refined_points_ras_mm: torch.Tensor  # [B, N, 3]
    displacement_ras_mm: torch.Tensor  # [B, N, 3]
    point_semantic: torch.Tensor  # [B, N, K]
    sparse_pou: SparsePoU
    geometry: VolumeGeometry

    def __post_init__(self) -> None:
        _float_tensor("s_coarse", self.s_coarse, 5)
        for name in ("initial_points_ras_mm", "refined_points_ras_mm", "displacement_ras_mm", "point_semantic"):
            _float_tensor(name, getattr(self, name), 3)
        batch, classes, depth, height, width = self.s_coarse.shape
        if classes <= 1 or tuple(self.geometry.shape_dhw) != (depth, height, width):
            raise ValueError("s_coarse and geometry must agree on [D, H, W] with K > 1")
        validate_probability_simplex("s_coarse", self.s_coarse, class_dimension=1)
        points = self.initial_points_ras_mm
        if points.shape != (batch, points.shape[1], 3) or self.refined_points_ras_mm.shape != points.shape or self.displacement_ras_mm.shape != points.shape:
            raise ValueError("point outputs must share shape [B, N, 3]")
        if self.point_semantic.shape != (batch, points.shape[1], classes):
            raise ValueError("point_semantic must align with points and s_coarse classes")
        validate_probability_simplex("point_semantic", self.point_semantic, class_dimension=-1)
        if not torch.allclose(
            self.refined_points_ras_mm - self.initial_points_ras_mm,
            self.displacement_ras_mm,
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("displacement_ras_mm must be measured from initial_points_ras_mm")
        if bool((torch.linalg.vector_norm(self.displacement_ras_mm, dim=-1) > 2.0 + 1e-6).any()):
            raise ValueError("FrontendOutput displacement_ras_mm is locked to at most 2.0 mm from initial points")

    @property
    def S_coarse(self) -> torch.Tensor:
        """Specification spelling retained for callers that name the prior ``S_coarse``."""

        return self.s_coarse

    @property
    def initial_points(self) -> torch.Tensor:
        return self.initial_points_ras_mm

    @property
    def refined_points(self) -> torch.Tensor:
        return self.refined_points_ras_mm

    @property
    def displacement(self) -> torch.Tensor:
        return self.displacement_ras_mm
