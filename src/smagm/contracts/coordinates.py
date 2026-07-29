"""Canonical RAS-mm geometry records used by the T0 physical operator.

All matrices use homogeneous *column* vectors.  Source affine columns map
``[u, v, slice, 1]`` plane indices to source physical millimetres; tensor
pixels remain indexed as ``[v, u]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Any, Iterable, Sequence


_GEOMETRY_TOL = 1e-6


class SourceConvention(str, Enum):
    """Physical convention of a source affine before RAS canonicalization."""

    DICOM_LPS = "DICOM_LPS"
    NIFTI_RAS = "NIFTI_RAS"
    CANONICAL_RAS = "CANONICAL_RAS"


def _numbers(values: Iterable[Any], count: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != count or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {count} finite values")
    return result


def _matrix4(value: Sequence[Sequence[Any]], name: str) -> tuple[tuple[float, ...], ...]:
    if len(value) != 4:
        raise ValueError(f"{name} must have four rows")
    matrix = tuple(_numbers(row, 4, name) for row in value)
    if matrix[3] != (0.0, 0.0, 0.0, 1.0):
        raise ValueError(f"{name} must be an affine homogeneous matrix")
    spatial = tuple(tuple(row[column] for column in range(3)) for row in matrix[:3])
    column_norms = tuple(
        math.hypot(*(spatial[row][column] for row in range(3)))
        for column in range(3)
    )
    if any(norm == 0.0 or not math.isfinite(norm) for norm in column_norms):
        raise ValueError(f"{name} has a singular spatial block")
    unit_spatial = tuple(
        tuple(spatial[row][column] / column_norms[column] for column in range(3))
        for row in range(3)
    )
    relative_volume = abs(_det3(unit_spatial))
    if not math.isfinite(relative_volume) or relative_volume <= 1e-8:
        raise ValueError(f"{name} has a singular or ill-conditioned spatial block")
    return matrix


def _det3(matrix: tuple[tuple[float, ...], ...]) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _matmul(left: tuple[tuple[float, ...], ...], right: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(sum(left[i][k] * right[k][j] for k in range(4)) for j in range(4)) for i in range(4))


def _column(matrix: tuple[tuple[float, ...], ...], index: int) -> tuple[float, float, float]:
    return (matrix[0][index], matrix[1][index], matrix[2][index])


def _norm(vector: tuple[float, float, float]) -> float:
    return math.hypot(*vector)


def _unit(vector: tuple[float, float, float], name: str) -> tuple[float, float, float]:
    magnitude = _norm(vector)
    if magnitude <= _GEOMETRY_TOL:
        raise ValueError(f"{name} must be nonzero")
    return tuple(component / magnitude for component in vector)  # type: ignore[return-value]


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _near(left: tuple[float, float, float], right: tuple[float, float, float], tolerance: float) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(left, right))


@dataclass(frozen=True)
class SourceAffineTransform:
    """A provenance-preserving plane-index-to-source-mm affine.

    The transform stores the source affine verbatim and derives its RAS affine
    from the source convention.  It intentionally retains the third affine
    column so plane handedness is independently checkable.
    """

    plane_index_to_source_mm: Sequence[Sequence[Any]]
    convention: SourceConvention

    def __post_init__(self) -> None:
        object.__setattr__(self, "plane_index_to_source_mm", _matrix4(self.plane_index_to_source_mm, "plane_index_to_source_mm"))
        object.__setattr__(self, "convention", SourceConvention(self.convention))

    @property
    def source_to_canonical(self) -> tuple[tuple[float, ...], ...]:
        if self.convention is SourceConvention.DICOM_LPS:
            return ((-1.0, 0.0, 0.0, 0.0), (0.0, -1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        return ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    @property
    def plane_index_to_ras_mm(self) -> tuple[tuple[float, ...], ...]:
        return _matmul(self.source_to_canonical, self.plane_index_to_source_mm)

    @property
    def origin_ras_mm(self) -> tuple[float, float, float]:
        return _column(self.plane_index_to_ras_mm, 3)

    @property
    def axis_u_step_ras_mm(self) -> tuple[float, float, float]:
        return _column(self.plane_index_to_ras_mm, 0)

    @property
    def axis_v_step_ras_mm(self) -> tuple[float, float, float]:
        return _column(self.plane_index_to_ras_mm, 1)

    @property
    def signed_slice_axis_ras(self) -> tuple[float, float, float]:
        return _unit(_column(self.plane_index_to_ras_mm, 2), "source affine slice axis")

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "convention": self.convention.value,
            "distance_unit": "mm",
            "index_order": ["u", "v", "slice"],
            "plane_index_to_source_mm": self.plane_index_to_source_mm,
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_canonical_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class PhysicalPlane:
    """An immutable pixel-centre plane in canonical RAS millimetres.

    ``signed_normal_ras`` is the direction of increasing source slice index.
    It is parallel to the geometric plane normal but may oppose
    ``axis_u × axis_v`` for a valid left-handed NIfTI index affine.
    """

    pixel_center_origin_ras_mm: Sequence[Any]
    axis_u_ras: Sequence[Any]
    axis_v_ras: Sequence[Any]
    spacing_uv_mm: Sequence[Any]
    thickness_mm: float
    shape_hw: Sequence[int]
    signed_normal_ras: Sequence[Any]
    source_transform: SourceAffineTransform | None = None
    observation_id: str | None = None

    def __post_init__(self) -> None:
        origin = _numbers(self.pixel_center_origin_ras_mm, 3, "pixel_center_origin_ras_mm")
        axis_u = _numbers(self.axis_u_ras, 3, "axis_u_ras")
        axis_v = _numbers(self.axis_v_ras, 3, "axis_v_ras")
        normal = _numbers(self.signed_normal_ras, 3, "signed_normal_ras")
        spacing = _numbers(self.spacing_uv_mm, 2, "spacing_uv_mm")
        shape = tuple(self.shape_hw)
        if any(value <= 0.0 for value in spacing) or not math.isfinite(float(self.thickness_mm)) or self.thickness_mm <= 0.0:
            raise ValueError("plane spacing and thickness must be positive and finite")
        if len(shape) != 2 or any(not isinstance(value, int) or value <= 0 for value in shape):
            raise ValueError("shape_hw must contain two positive integers")
        for value, name in ((axis_u, "axis_u_ras"), (axis_v, "axis_v_ras"), (normal, "signed_normal_ras")):
            if abs(_norm(value) - 1.0) > _GEOMETRY_TOL:
                raise ValueError(f"{name} must have unit length")
        if abs(_dot(axis_u, axis_v)) > _GEOMETRY_TOL:
            raise ValueError("axis_u_ras and axis_v_ras must be orthogonal")
        if abs(_dot(_unit(_cross(axis_u, axis_v), "plane axes"), normal)) < 1.0 - _GEOMETRY_TOL:
            raise ValueError("signed_normal_ras must be parallel to axis_u cross axis_v")
        if self.source_transform is not None:
            if not isinstance(self.source_transform, SourceAffineTransform):
                raise TypeError("source_transform must be a SourceAffineTransform or None")
            self._validate_source_agreement(origin, axis_u, axis_v, normal, spacing)
        object.__setattr__(self, "pixel_center_origin_ras_mm", origin)
        object.__setattr__(self, "axis_u_ras", axis_u)
        object.__setattr__(self, "axis_v_ras", axis_v)
        object.__setattr__(self, "signed_normal_ras", normal)
        object.__setattr__(self, "spacing_uv_mm", spacing)
        object.__setattr__(self, "shape_hw", shape)
        object.__setattr__(self, "thickness_mm", float(self.thickness_mm))

    def _validate_source_agreement(self, origin: tuple[float, ...], axis_u: tuple[float, ...], axis_v: tuple[float, ...], normal: tuple[float, ...], spacing: tuple[float, ...]) -> None:
        assert self.source_transform is not None
        source = self.source_transform
        if not _near(origin, source.origin_ras_mm, _GEOMETRY_TOL):
            raise ValueError("source affine origin disagrees with pixel-centre plane origin")
        if _dot(_unit(source.axis_u_step_ras_mm, "source u axis"), axis_u) < 1.0 - _GEOMETRY_TOL:
            raise ValueError("source affine u axis disagrees with plane axis_u")
        if _dot(_unit(source.axis_v_step_ras_mm, "source v axis"), axis_v) < 1.0 - _GEOMETRY_TOL:
            raise ValueError("source affine v axis disagrees with plane axis_v")
        if abs(_norm(source.axis_u_step_ras_mm) - spacing[0]) > _GEOMETRY_TOL or abs(_norm(source.axis_v_step_ras_mm) - spacing[1]) > _GEOMETRY_TOL:
            raise ValueError("source affine in-plane step lengths disagree with plane spacing")
        if _dot(source.signed_slice_axis_ras, normal) < 1.0 - _GEOMETRY_TOL:
            raise ValueError("source affine signed slice axis disagrees with plane signed normal")

    def world_from_vu(self, v: float, u: float) -> tuple[float, float, float]:
        """Map a tensor pixel index ``[v, u]`` to its RAS-mm centre."""
        return tuple(
            self.pixel_center_origin_ras_mm[index]
            + float(u) * self.spacing_uv_mm[0] * self.axis_u_ras[index]
            + float(v) * self.spacing_uv_mm[1] * self.axis_v_ras[index]
            for index in range(3)
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "axis_u_ras": self.axis_u_ras, "axis_v_ras": self.axis_v_ras,
            "coordinate_system": "RAS", "distance_unit": "mm",
            "observation_id": self.observation_id, "pixel_center_origin_ras_mm": self.pixel_center_origin_ras_mm,
            "pixel_index_order": ["v", "u"],
            "shape_hw": self.shape_hw, "signed_normal_ras": self.signed_normal_ras,
            "spacing_uv_mm": self.spacing_uv_mm, "thickness_mm": self.thickness_mm,
        }
        if self.source_transform is not None:
            result["source_transform"] = self.source_transform.to_canonical_dict()
        return result

    def canonical_json(self) -> str:
        return json.dumps(self.to_canonical_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class TargetGrid:
    """A canonical RAS grid with explicit tensor order ``[d, h, w]``.

    ``index_to_ras_mm`` consumes homogeneous ``[w, h, d, 1]`` column vectors.
    """

    index_to_ras_mm: Sequence[Sequence[Any]]
    shape_dhw: Sequence[int]
    modality_ids: Sequence[str] = ()
    normalization_records: Sequence[str] = ()

    def __post_init__(self) -> None:
        shape = tuple(self.shape_dhw)
        if len(shape) != 3 or any(not isinstance(value, int) or value <= 0 for value in shape):
            raise ValueError("shape_dhw must contain three positive integers")
        object.__setattr__(self, "index_to_ras_mm", _matrix4(self.index_to_ras_mm, "index_to_ras_mm"))
        object.__setattr__(self, "shape_dhw", shape)
        object.__setattr__(self, "modality_ids", tuple(str(item) for item in self.modality_ids))
        object.__setattr__(self, "normalization_records", tuple(str(item) for item in self.normalization_records))

    def world_from_dhw(self, d: float, h: float, w: float) -> tuple[float, float, float]:
        matrix = self.index_to_ras_mm
        return tuple(matrix[row][0] * w + matrix[row][1] * h + matrix[row][2] * d + matrix[row][3] for row in range(3))

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "coordinate_system": "RAS",
            "distance_unit": "mm",
            "homogeneous_index_order": ["w", "h", "d", "1"],
            "index_to_ras_mm": self.index_to_ras_mm,
            "modality_ids": self.modality_ids,
            "normalization_records": self.normalization_records,
            "shape_dhw": self.shape_dhw,
            "tensor_index_order": ["d", "h", "w"],
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_canonical_dict(), sort_keys=True, separators=(",", ":"))
