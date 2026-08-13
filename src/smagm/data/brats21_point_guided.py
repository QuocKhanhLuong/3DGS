"""Full-volume BraTS21 input boundary for the point-guided baseline.

This module is intentionally separate from :mod:`smagm.data.brats21`.  The
legacy adapter prepares sparse physical planes and has a different modality
contract.  This adapter owns one narrow dense-volume contract instead:

* NIfTI source arrays are ``[X, Y, Z]`` and are copied explicitly to tensor
  order ``[D, H, W] == [Z, Y, X]``;
* the observation tensor contains only normalized ``T1, T2, FLAIR``;
* ``T1ce`` and ``seg`` remain separate target/evaluator tensors;
* the brain mask is derived from raw observation values before any
  normalization, and never from target or segmentation data;
* geometry, finite values, and BraTS labels are validated fail-closed; and
* subject splits are deterministic and carry a content hash.

The optional ``nibabel`` dependency is imported lazily so that the package
and non-NIfTI unit tests remain usable in a minimal installation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


POINT_GUIDED_OBSERVATION_MODALITIES: tuple[str, str, str] = ("t1", "t2", "flair")
POINT_GUIDED_TARGET_MODALITY = "t1ce"
POINT_GUIDED_SEGMENTATION_MODALITY = "seg"
POINT_GUIDED_MODALITIES: tuple[str, ...] = (
    *POINT_GUIDED_OBSERVATION_MODALITIES,
    POINT_GUIDED_TARGET_MODALITY,
)
BRATS21_POINT_GUIDED_LABELS = frozenset({0, 1, 2, 4})
BRATS21_POINT_GUIDED_SUBJECT_PATTERN = re.compile(r"^BraTS2021_(?P<number>\d{5})$")
NORMALIZATION_VERSION = "masked_zscore_v1"
MASKED_ZSCORE_POLICY = "masked_zscore"
MASKED_ROBUST_01_POLICY = "masked_robust_01"
SUPPORTED_NORMALIZATION_POLICIES = frozenset({MASKED_ZSCORE_POLICY, MASKED_ROBUST_01_POLICY})
SPLIT_VERSION = "brats21_point_guided_split_v1"


class BraTS21PointGuidedDependencyError(RuntimeError):
    """Raised when the optional NIfTI reader is unavailable."""


class BraTS21PointGuidedValidationError(ValueError):
    """Raised for an invalid subject, volume, geometry, or data contract."""


# A shorter alias is useful to callers that do not need the verbose class
# name, while retaining the explicit public type for diagnostics.
BraTS21PointGuidedError = BraTS21PointGuidedValidationError


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_float(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise BraTS21PointGuidedValidationError(f"{name} must be finite")
    return result


def _normalization_policy(value: object) -> str:
    if not isinstance(value, str) or value not in SUPPORTED_NORMALIZATION_POLICIES:
        raise BraTS21PointGuidedValidationError(
            "normalization_policy must be one of "
            f"{tuple(sorted(SUPPORTED_NORMALIZATION_POLICIES))}, got {value!r}"
        )
    return value


def _percentile(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise BraTS21PointGuidedValidationError(f"{name} must be a finite percentile in [0, 100]")
    result = _finite_float(value, name)
    if not 0.0 <= result <= 100.0:
        raise BraTS21PointGuidedValidationError(f"{name} must be in [0, 100]")
    return result


def _validate_percentile_range(lower: object, upper: object) -> tuple[float, float]:
    lower_value = _percentile(lower, "lower_percentile")
    upper_value = _percentile(upper, "upper_percentile")
    if upper_value <= lower_value:
        raise BraTS21PointGuidedValidationError(
            "lower_percentile must be strictly less than upper_percentile; the percentile range is empty"
        )
    return lower_value, upper_value


def _validate_subject_id(subject_id: object) -> str:
    value = str(subject_id)
    if BRATS21_POINT_GUIDED_SUBJECT_PATTERN.fullmatch(value) is None:
        raise BraTS21PointGuidedValidationError(f"malformed BraTS21 subject ID: {value!r}")
    return value


def _nibabel() -> Any:
    try:
        import nibabel as nib  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - environment dependent
        raise BraTS21PointGuidedDependencyError(
            "the point-guided BraTS21 adapter requires optional dependency 'nibabel'"
        ) from error
    return nib


def xyz_shape_to_dhw(shape_xyz: Sequence[int]) -> tuple[int, int, int]:
    """Convert a NIfTI ``[X, Y, Z]`` shape to tensor ``[D, H, W]`` order."""

    shape = tuple(shape_xyz)
    if len(shape) != 3 or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape):
        raise BraTS21PointGuidedValidationError("shape_xyz must contain three positive integers")
    x, y, z = shape
    return (z, y, x)


def nifti_xyz_to_dhw(array_xyz: np.ndarray) -> np.ndarray:
    """Return a contiguous ``[D, H, W]`` view/copy of a ``[X, Y, Z]`` array.

    The transpose is deliberately written at this boundary rather than
    relying on a loader default.  The NIfTI affine still consumes
    ``[x, y, z] == [w, h, d]`` coordinates; only the in-memory tensor axes are
    reordered.
    """

    array = np.asarray(array_xyz)
    if array.ndim != 3:
        raise BraTS21PointGuidedValidationError(
            f"NIfTI source data must be 3-D [X,Y,Z], got ndim={array.ndim}"
        )
    return np.ascontiguousarray(np.transpose(array, (2, 1, 0)))


# This name reads naturally in callers that are already operating on NIfTI
# arrays.  Keep the explicit primary name above for provenance reviews.
xyz_to_dhw = nifti_xyz_to_dhw


def _resolved_affine(image: Any, path: Path) -> np.ndarray:
    """Resolve a NIfTI affine and reject ambiguous or singular geometry."""

    try:
        header = image.header
        qform, qcode = header.get_qform(coded=True)
        sform, scode = header.get_sform(coded=True)
        if qcode > 0 and scode > 0:
            qform_array = np.asarray(qform, dtype=np.float64)
            sform_array = np.asarray(sform, dtype=np.float64)
            if not np.allclose(qform_array, sform_array, atol=1e-4, rtol=0.0):
                raise BraTS21PointGuidedValidationError(f"{path.name}: NIfTI qform and sform disagree")
        selected = sform if scode > 0 else qform if qcode > 0 else image.affine
        affine = np.asarray(selected, dtype=np.float64)
    except BraTS21PointGuidedValidationError:
        raise
    except Exception as error:
        raise BraTS21PointGuidedValidationError(f"{path.name}: cannot read NIfTI affine: {error}") from error

    if (
        affine.shape != (4, 4)
        or not np.isfinite(affine).all()
        or not np.allclose(affine[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6, rtol=0.0)
    ):
        raise BraTS21PointGuidedValidationError(f"{path.name}: affine must be finite homogeneous 4x4")
    determinant = float(np.linalg.det(affine[:3, :3]))
    if not math.isfinite(determinant) or abs(determinant) <= 1e-8:
        raise BraTS21PointGuidedValidationError(f"{path.name}: affine spatial block is singular")
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0.0):
        raise BraTS21PointGuidedValidationError(f"{path.name}: affine-derived XYZ spacing is invalid")
    return affine


def _orientation(nib: Any, affine: np.ndarray, path: Path) -> tuple[str, str, str]:
    try:
        codes = tuple(nib.aff2axcodes(affine))
    except Exception as error:
        raise BraTS21PointGuidedValidationError(f"{path.name}: NIfTI orientation is unavailable") from error
    if len(codes) != 3 or any(code not in {"R", "L", "A", "P", "S", "I"} for code in codes):
        raise BraTS21PointGuidedValidationError(f"{path.name}: NIfTI orientation is invalid: {codes!r}")
    return codes  # type: ignore[return-value]


@dataclass(frozen=True)
class NiftiGeometryMetadata:
    """Validated geometry for a source NIfTI and its DHW tensor view."""

    shape_xyz: tuple[int, int, int]
    shape_dhw: tuple[int, int, int]
    affine_xyz_to_ras_mm: tuple[tuple[float, ...], ...]
    spacing_xyz_mm: tuple[float, float, float]
    orientation: tuple[str, str, str]

    def __post_init__(self) -> None:
        if xyz_shape_to_dhw(self.shape_xyz) != tuple(self.shape_dhw):
            raise BraTS21PointGuidedValidationError("shape_dhw must be the explicit XYZ-to-DHW reversal")
        matrix = tuple(tuple(float(value) for value in row) for row in self.affine_xyz_to_ras_mm)
        if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
            raise BraTS21PointGuidedValidationError("affine_xyz_to_ras_mm must be 4x4")
        if not all(math.isfinite(value) for row in matrix for value in row):
            raise BraTS21PointGuidedValidationError("affine_xyz_to_ras_mm must be finite")
        if matrix[3] != (0.0, 0.0, 0.0, 1.0):
            raise BraTS21PointGuidedValidationError("affine_xyz_to_ras_mm must be homogeneous")
        spacing = tuple(float(value) for value in self.spacing_xyz_mm)
        if len(spacing) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in spacing):
            raise BraTS21PointGuidedValidationError("spacing_xyz_mm must contain three positive finite values")
        expected_spacing = tuple(
            math.sqrt(sum(matrix[row][column] ** 2 for row in range(3)))
            for column in range(3)
        )
        if any(abs(actual - expected) > 1e-5 for actual, expected in zip(spacing, expected_spacing)):
            raise BraTS21PointGuidedValidationError("spacing_xyz_mm disagrees with affine columns")
        if len(self.orientation) != 3 or any(code not in {"R", "L", "A", "P", "S", "I"} for code in self.orientation):
            raise BraTS21PointGuidedValidationError("orientation must contain three NIfTI axis codes")
        object.__setattr__(self, "shape_xyz", tuple(int(value) for value in self.shape_xyz))
        object.__setattr__(self, "shape_dhw", tuple(int(value) for value in self.shape_dhw))
        object.__setattr__(self, "affine_xyz_to_ras_mm", matrix)
        object.__setattr__(self, "spacing_xyz_mm", spacing)
        object.__setattr__(self, "orientation", tuple(str(value) for value in self.orientation))

    @property
    def voxel_to_ras_mm(self) -> tuple[tuple[float, ...], ...]:
        """The same affine viewed as a ``[w, h, d]`` tensor-index map."""

        return self.affine_xyz_to_ras_mm

    def to_dict(self) -> dict[str, object]:
        return {
            "affine_xyz_to_ras_mm": self.affine_xyz_to_ras_mm,
            "coordinate_system": "RAS",
            "distance_unit": "mm",
            "orientation": self.orientation,
            "shape_dhw": self.shape_dhw,
            "shape_xyz": self.shape_xyz,
            "spacing_xyz_mm": self.spacing_xyz_mm,
            "tensor_index_order": ["d", "h", "w"],
            "xyz_index_order": ["x", "y", "z"],
        }


@dataclass(frozen=True)
class ModalityNormalizationMetadata:
    """Reproducible statistics for one modality's masked transform."""

    modality: str
    voxel_count: int
    mean: float
    std: float
    scale: float
    minimum: float
    maximum: float
    mask_source: str = "raw_observation_nonzero_union"
    version: str = NORMALIZATION_VERSION
    normalization_policy: str = MASKED_ZSCORE_POLICY
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    clip_lower: float | None = None
    clip_upper: float | None = None
    output_range: tuple[float, float] | None = None
    metadata_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.modality, str) or not self.modality:
            raise BraTS21PointGuidedValidationError("normalization modality must be a non-empty string")
        if isinstance(self.voxel_count, bool) or not isinstance(self.voxel_count, int) or self.voxel_count <= 0:
            raise BraTS21PointGuidedValidationError("normalization voxel_count must be a positive integer")
        values = {
            "mean": _finite_float(self.mean, "normalization mean"),
            "std": _finite_float(self.std, "normalization std"),
            "scale": _finite_float(self.scale, "normalization scale"),
            "minimum": _finite_float(self.minimum, "normalization minimum"),
            "maximum": _finite_float(self.maximum, "normalization maximum"),
        }
        if values["std"] < 0.0 or values["scale"] <= 0.0 or values["minimum"] > values["maximum"]:
            raise BraTS21PointGuidedValidationError("normalization statistics are inconsistent")
        if not isinstance(self.mask_source, str) or not self.mask_source:
            raise BraTS21PointGuidedValidationError("normalization mask_source must be non-empty")
        if self.version != NORMALIZATION_VERSION:
            raise BraTS21PointGuidedValidationError(f"unsupported normalization version: {self.version!r}")
        policy = _normalization_policy(self.normalization_policy)
        lower_percentile, upper_percentile = _validate_percentile_range(
            self.lower_percentile,
            self.upper_percentile,
        )
        clip_lower = None if self.clip_lower is None else _finite_float(self.clip_lower, "normalization clip_lower")
        clip_upper = None if self.clip_upper is None else _finite_float(self.clip_upper, "normalization clip_upper")
        if (clip_lower is None) != (clip_upper is None):
            raise BraTS21PointGuidedValidationError("normalization clip bounds must be both present or both absent")
        if clip_lower is not None and clip_upper is not None and clip_upper <= clip_lower:
            raise BraTS21PointGuidedValidationError("normalization clip bounds define an empty range")
        output_range: tuple[float, float] | None
        if self.output_range is None:
            output_range = None
        else:
            try:
                raw_output_range = tuple(self.output_range)
            except TypeError as error:
                raise BraTS21PointGuidedValidationError(
                    "normalization output_range must contain two finite values"
                ) from error
            if len(raw_output_range) != 2:
                raise BraTS21PointGuidedValidationError("normalization output_range must contain two values")
            output_range = (
                _finite_float(raw_output_range[0], "normalization output_range lower"),
                _finite_float(raw_output_range[1], "normalization output_range upper"),
            )
            if output_range[1] <= output_range[0]:
                raise BraTS21PointGuidedValidationError("normalization output_range defines an empty range")
        if policy == MASKED_ROBUST_01_POLICY:
            if clip_lower is None or clip_upper is None:
                raise BraTS21PointGuidedValidationError(
                    "masked_robust_01 metadata requires finite clip bounds"
                )
            if output_range != (0.0, 1.0):
                raise BraTS21PointGuidedValidationError(
                    "masked_robust_01 metadata must record output_range=(0.0, 1.0)"
                )
        object.__setattr__(self, "mean", values["mean"])
        object.__setattr__(self, "std", values["std"])
        object.__setattr__(self, "scale", values["scale"])
        object.__setattr__(self, "minimum", values["minimum"])
        object.__setattr__(self, "maximum", values["maximum"])
        object.__setattr__(self, "normalization_policy", policy)
        object.__setattr__(self, "lower_percentile", lower_percentile)
        object.__setattr__(self, "upper_percentile", upper_percentile)
        object.__setattr__(self, "clip_lower", clip_lower)
        object.__setattr__(self, "clip_upper", clip_upper)
        object.__setattr__(self, "output_range", output_range)
        object.__setattr__(self, "metadata_hash", _canonical_hash(self._unsigned_dict()))

    @property
    def center(self) -> float:
        return self.mean

    @property
    def record_hash(self) -> str:
        return self.metadata_hash

    @property
    def policy(self) -> str:
        """Short alias for the explicit normalization policy."""

        return self.normalization_policy

    @property
    def percentiles(self) -> tuple[float, float]:
        return (self.lower_percentile, self.upper_percentile)

    @property
    def clip_bounds(self) -> tuple[float, float] | None:
        if self.clip_lower is None or self.clip_upper is None:
            return None
        return (self.clip_lower, self.clip_upper)

    @property
    def clip_low(self) -> float | None:
        return self.clip_lower

    @property
    def clip_high(self) -> float | None:
        return self.clip_upper

    @property
    def output_min(self) -> float | None:
        return None if self.output_range is None else self.output_range[0]

    @property
    def output_max(self) -> float | None:
        return None if self.output_range is None else self.output_range[1]

    def _unsigned_dict(self) -> dict[str, object]:
        return {
            "clip_bounds": self.clip_bounds,
            "clip_lower": self.clip_lower,
            "clip_upper": self.clip_upper,
            "clip_high": self.clip_high,
            "clip_low": self.clip_low,
            "lower_percentile": self.lower_percentile,
            "mask_source": self.mask_source,
            "maximum": self.maximum,
            "mean": self.mean,
            "minimum": self.minimum,
            "modality": self.modality,
            "normalization_policy": self.normalization_policy,
            "output_max": self.output_max,
            "output_min": self.output_min,
            "output_range": self.output_range,
            "policy": self.policy,
            "percentiles": self.percentiles,
            "scale": self.scale,
            "std": self.std,
            "upper_percentile": self.upper_percentile,
            "version": self.version,
            "voxel_count": self.voxel_count,
        }

    def to_dict(self) -> dict[str, object]:
        return self._unsigned_dict() | {"metadata_hash": self.metadata_hash}


NormalizationMetadata = ModalityNormalizationMetadata


@dataclass(frozen=True)
class BraTS21PointGuidedSubject:
    """File bindings for one validated-by-name BraTS21 subject."""

    subject_id: str
    directory: Path
    observation_paths: Mapping[str, Path]
    target_path: Path | None
    segmentation_path: Path | None
    unknown_nifti_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        subject_id = _validate_subject_id(self.subject_id)
        if tuple(self.observation_paths) != POINT_GUIDED_OBSERVATION_MODALITIES:
            raise BraTS21PointGuidedValidationError(
                f"{subject_id}: required observations are exactly {POINT_GUIDED_OBSERVATION_MODALITIES}"
            )
        directory = Path(self.directory).resolve()
        if not directory.is_dir():
            raise BraTS21PointGuidedValidationError(f"{subject_id}: subject directory is not a directory")
        paths = {
            modality: Path(self.observation_paths[modality]).resolve()
            for modality in POINT_GUIDED_OBSERVATION_MODALITIES
        }
        if any(not path.is_file() for path in paths.values()):
            raise BraTS21PointGuidedValidationError(f"{subject_id}: one or more observation files are missing")
        target = None if self.target_path is None else Path(self.target_path).resolve()
        segmentation = None if self.segmentation_path is None else Path(self.segmentation_path).resolve()
        if target is not None and not target.is_file():
            raise BraTS21PointGuidedValidationError(f"{subject_id}: target file is missing")
        if segmentation is not None and not segmentation.is_file():
            raise BraTS21PointGuidedValidationError(f"{subject_id}: segmentation file is missing")
        object.__setattr__(self, "subject_id", subject_id)
        object.__setattr__(self, "directory", directory)
        object.__setattr__(self, "observation_paths", MappingProxyType(paths))
        object.__setattr__(self, "target_path", target)
        object.__setattr__(self, "segmentation_path", segmentation)
        object.__setattr__(self, "unknown_nifti_files", tuple(sorted(str(item) for item in self.unknown_nifti_files)))

    @property
    def t1ce_path(self) -> Path | None:
        return self.target_path

    def to_dict(self) -> dict[str, object]:
        return {
            "directory": str(self.directory),
            "observation_paths": {key: str(value) for key, value in self.observation_paths.items()},
            "segmentation_path": None if self.segmentation_path is None else str(self.segmentation_path),
            "subject_id": self.subject_id,
            "target_path": None if self.target_path is None else str(self.target_path),
            "unknown_nifti_files": self.unknown_nifti_files,
        }


def _nifti_files(directory: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.name.endswith((".nii", ".nii.gz"))
        )
    )


def _suffix(subject_id: str, path: Path) -> str | None:
    prefix = f"{subject_id}_"
    if not path.name.startswith(prefix):
        return None
    filename = path.name[len(prefix) :]
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return None


def discover_point_guided_subject(directory: str | Path) -> BraTS21PointGuidedSubject:
    """Discover exact point-guided bindings without opening NIfTI payloads."""

    try:
        path = Path(directory).resolve(strict=True)
    except FileNotFoundError as error:
        raise BraTS21PointGuidedValidationError(f"subject path does not exist: {directory}") from error
    if not path.is_dir():
        raise BraTS21PointGuidedValidationError(f"subject path is not a directory: {path}")
    subject_id = _validate_subject_id(path.name)
    discovered: dict[str, list[Path]] = {}
    unknown: list[str] = []
    known = (*POINT_GUIDED_MODALITIES, POINT_GUIDED_SEGMENTATION_MODALITY)
    for nifti_path in _nifti_files(path):
        suffix = _suffix(subject_id, nifti_path)
        if suffix is None or suffix not in known:
            unknown.append(nifti_path.name)
            continue
        discovered.setdefault(suffix, []).append(nifti_path)

    missing = [modality for modality in POINT_GUIDED_OBSERVATION_MODALITIES if len(discovered.get(modality, ())) != 1]
    if missing:
        details = {modality: [str(item.name) for item in discovered.get(modality, ())] for modality in missing}
        raise BraTS21PointGuidedValidationError(
            f"{subject_id}: missing or duplicate observation files: {details}"
        )
    for modality in (POINT_GUIDED_TARGET_MODALITY, POINT_GUIDED_SEGMENTATION_MODALITY):
        if len(discovered.get(modality, ())) > 1:
            raise BraTS21PointGuidedValidationError(
                f"{subject_id}: duplicate {modality} files: {[item.name for item in discovered[modality]]}"
            )
    return BraTS21PointGuidedSubject(
        subject_id=subject_id,
        directory=path,
        observation_paths={
            modality: discovered[modality][0] for modality in POINT_GUIDED_OBSERVATION_MODALITIES
        },
        target_path=discovered.get(POINT_GUIDED_TARGET_MODALITY, [None])[0],
        segmentation_path=discovered.get(POINT_GUIDED_SEGMENTATION_MODALITY, [None])[0],
        unknown_nifti_files=tuple(unknown),
    )


def discover_point_guided_subjects(root: str | Path) -> tuple[BraTS21PointGuidedSubject, ...]:
    """Discover subjects in deterministic ID order and reject malformed roots."""

    try:
        source_root = Path(root).resolve(strict=True)
    except FileNotFoundError as error:
        raise BraTS21PointGuidedValidationError(f"source root does not exist: {root}") from error
    if not source_root.is_dir():
        raise BraTS21PointGuidedValidationError(f"source root is not a directory: {source_root}")
    directories = sorted(item for item in source_root.iterdir() if item.is_dir())
    malformed = [item.name for item in directories if BRATS21_POINT_GUIDED_SUBJECT_PATTERN.fullmatch(item.name) is None]
    if malformed:
        raise BraTS21PointGuidedValidationError(f"malformed subject directory names: {malformed}")
    return tuple(discover_point_guided_subject(item) for item in directories)


discover_subject = discover_point_guided_subject
discover_subjects = discover_point_guided_subjects


def _read_nifti(path: Path, *, role: str) -> tuple[np.ndarray, NiftiGeometryMetadata]:
    nib = _nibabel()
    try:
        image = nib.load(str(path), mmap=True)
        header_shape = tuple(int(value) for value in image.header.get_data_shape())
        if len(header_shape) != 3 or any(value <= 0 for value in header_shape):
            raise BraTS21PointGuidedValidationError(f"{path.name}: {role} must be a non-empty 3-D NIfTI")
        affine = _resolved_affine(image, path)
        data = np.asanyarray(image.dataobj)
        if data.shape != header_shape or data.ndim != 3:
            raise BraTS21PointGuidedValidationError(
                f"{path.name}: {role} data shape disagrees with the 3-D NIfTI header"
            )
        if np.iscomplexobj(data) or not np.issubdtype(data.dtype, np.number):
            raise BraTS21PointGuidedValidationError(f"{path.name}: {role} must contain real numeric data")
        if not np.isfinite(data).all():
            raise BraTS21PointGuidedValidationError(f"{path.name}: {role} contains non-finite values")
        geometry = NiftiGeometryMetadata(
            shape_xyz=header_shape,
            shape_dhw=xyz_shape_to_dhw(header_shape),
            affine_xyz_to_ras_mm=tuple(tuple(float(value) for value in row) for row in affine),
            spacing_xyz_mm=tuple(float(value) for value in np.linalg.norm(affine[:3, :3], axis=0)),
            orientation=_orientation(nib, affine, path),
        )
        return np.asarray(data), geometry
    except BraTS21PointGuidedValidationError:
        raise
    except Exception as error:
        raise BraTS21PointGuidedValidationError(f"{path.name}: cannot read {role} NIfTI: {error}") from error


def _same_geometry(reference: NiftiGeometryMetadata, candidate: NiftiGeometryMetadata, *, role: str) -> None:
    if candidate.shape_xyz != reference.shape_xyz:
        raise BraTS21PointGuidedValidationError(
            f"{role}: shape mismatch, expected XYZ {reference.shape_xyz}, got {candidate.shape_xyz}"
        )
    if not np.allclose(candidate.affine_xyz_to_ras_mm, reference.affine_xyz_to_ras_mm, atol=1e-4, rtol=0.0):
        raise BraTS21PointGuidedValidationError(f"{role}: affine mismatch with the T1 reference")
    if not np.allclose(candidate.spacing_xyz_mm, reference.spacing_xyz_mm, atol=1e-5, rtol=0.0):
        raise BraTS21PointGuidedValidationError(f"{role}: spacing mismatch with the T1 reference")


def derive_input_brain_mask(
    observations_xyz: np.ndarray,
    *,
    threshold: float = 0.0,
) -> np.ndarray:
    """Derive a binary brain mask from raw T1/T2/FLAIR values only.

    The union is evaluated before any per-modality centering or scaling.  The
    default threshold matches BraTS background semantics (exact zero), while
    callers may select a deterministic non-negative absolute-value threshold
    for a known acquisition's noise floor.
    """

    observations = np.asarray(observations_xyz)
    if observations.ndim != 4 or observations.shape[0] != len(POINT_GUIDED_OBSERVATION_MODALITIES):
        raise BraTS21PointGuidedValidationError(
            "observations_xyz must have shape [3, X, Y, Z] in T1/T2/FLAIR order"
        )
    threshold_value = _finite_float(threshold, "brain-mask threshold")
    if threshold_value < 0.0:
        raise BraTS21PointGuidedValidationError("brain-mask threshold must be non-negative")
    if np.iscomplexobj(observations) or not np.issubdtype(observations.dtype, np.number):
        raise BraTS21PointGuidedValidationError("raw observations must be real numeric values")
    if not np.isfinite(observations).all():
        raise BraTS21PointGuidedValidationError("raw observations must be finite before mask derivation")
    mask = np.any(np.abs(observations) > threshold_value, axis=0)
    if not bool(mask.any()):
        raise BraTS21PointGuidedValidationError("input-derived brain mask is empty")
    return np.ascontiguousarray(mask, dtype=bool)


def _normalize_masked(
    values_dhw: np.ndarray,
    mask_dhw: np.ndarray,
    *,
    modality: str,
    epsilon: float,
    normalization_policy: str,
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[torch.Tensor, ModalityNormalizationMetadata]:
    values = np.asarray(values_dhw, dtype=np.float64)
    mask = np.asarray(mask_dhw, dtype=bool)
    if values.ndim != 3 or mask.shape != values.shape or not bool(mask.any()):
        raise BraTS21PointGuidedValidationError(f"{modality}: normalization topology is invalid")
    selected = values[mask]
    if not np.isfinite(selected).all():
        raise BraTS21PointGuidedValidationError(f"{modality}: normalization values are non-finite")
    policy = _normalization_policy(normalization_policy)
    lower_percentile_value, upper_percentile_value = _validate_percentile_range(
        lower_percentile,
        upper_percentile,
    )
    mean = float(np.mean(selected, dtype=np.float64))
    std = float(np.std(selected, dtype=np.float64))
    normalized = np.zeros(values.shape, dtype=np.float64)
    clip_lower: float | None = None
    clip_upper: float | None = None
    output_range: tuple[float, float] | None = None
    if policy == MASKED_ZSCORE_POLICY:
        scale = std if std >= epsilon else 1.0
        normalized[mask] = (selected - mean) / scale
    else:
        clip_lower = float(np.percentile(selected, lower_percentile_value))
        clip_upper = float(np.percentile(selected, upper_percentile_value))
        if not math.isfinite(clip_lower) or not math.isfinite(clip_upper) or clip_upper <= clip_lower:
            raise BraTS21PointGuidedValidationError(
                f"{modality}: masked_robust_01 percentile range is invalid or empty"
            )
        scale = clip_upper - clip_lower
        clipped = np.clip(selected, clip_lower, clip_upper)
        normalized[mask] = (clipped - clip_lower) / scale
        output_range = (0.0, 1.0)
    normalized_float32 = normalized.astype(np.float32)
    if not np.isfinite(normalized_float32).all():
        raise BraTS21PointGuidedValidationError(f"{modality}: normalized values overflowed float32")
    metadata = ModalityNormalizationMetadata(
        modality=modality,
        voxel_count=int(selected.size),
        mean=mean,
        std=std,
        scale=scale,
        minimum=float(np.min(selected)),
        maximum=float(np.max(selected)),
        normalization_policy=policy,
        lower_percentile=lower_percentile_value,
        upper_percentile=upper_percentile_value,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
        output_range=output_range,
    )
    return torch.from_numpy(np.ascontiguousarray(normalized_float32)), metadata


@dataclass(frozen=True)
class BraTS21PointGuidedSample:
    """One dense subject with separated observations, target, and labels."""

    subject_id: str
    observations: torch.Tensor
    target: torch.Tensor | None
    segmentation: torch.Tensor | None
    brain_mask: torch.Tensor
    geometry: NiftiGeometryMetadata
    normalization_metadata: Mapping[str, ModalityNormalizationMetadata]
    source_paths: Mapping[str, Path]

    def __post_init__(self) -> None:
        subject_id = _validate_subject_id(self.subject_id)
        shape = self.geometry.shape_dhw
        if self.observations.ndim != 4 or tuple(self.observations.shape) != (3, *shape):
            raise BraTS21PointGuidedValidationError("observations must have shape [3,D,H,W]")
        if self.observations.dtype != torch.float32 or not bool(torch.isfinite(self.observations).all()):
            raise BraTS21PointGuidedValidationError("observations must be finite float32")
        if self.brain_mask.dtype is not torch.bool or tuple(self.brain_mask.shape) != (1, *shape):
            raise BraTS21PointGuidedValidationError("brain_mask must be bool with shape [1,D,H,W]")
        if not bool(self.brain_mask.any()):
            raise BraTS21PointGuidedValidationError("brain_mask must contain at least one voxel")
        if self.target is not None:
            if self.target.dtype != torch.float32 or tuple(self.target.shape) != (1, *shape) or not bool(torch.isfinite(self.target).all()):
                raise BraTS21PointGuidedValidationError("target must be finite float32 with shape [1,D,H,W]")
        if self.segmentation is not None:
            if self.segmentation.dtype != torch.int64 or tuple(self.segmentation.shape) != shape:
                raise BraTS21PointGuidedValidationError("segmentation must be int64 with shape [D,H,W]")
            labels = set(int(value) for value in torch.unique(self.segmentation).tolist())
            if not labels.issubset(BRATS21_POINT_GUIDED_LABELS):
                raise BraTS21PointGuidedValidationError(f"segmentation contains unexpected labels: {sorted(labels)}")
        metadata = dict(self.normalization_metadata)
        expected_keys = set(POINT_GUIDED_OBSERVATION_MODALITIES)
        if self.target is not None:
            expected_keys.add(POINT_GUIDED_TARGET_MODALITY)
        if set(metadata) != expected_keys or any(not isinstance(value, ModalityNormalizationMetadata) for value in metadata.values()):
            raise BraTS21PointGuidedValidationError("normalization_metadata keys must match loaded MRI modalities")
        paths = {str(key): Path(value).resolve() for key, value in self.source_paths.items()}
        if set(paths) != expected_keys | ({POINT_GUIDED_SEGMENTATION_MODALITY} if self.segmentation is not None else set()):
            raise BraTS21PointGuidedValidationError("source_paths must identify each loaded MRI/segmentation volume")
        object.__setattr__(self, "subject_id", subject_id)
        object.__setattr__(self, "normalization_metadata", MappingProxyType(metadata))
        object.__setattr__(self, "source_paths", MappingProxyType(paths))

    @property
    def inputs(self) -> torch.Tensor:
        """Alias for the model-ready ``[3,D,H,W]`` observation tensor."""

        return self.observations

    @property
    def target_t1ce(self) -> torch.Tensor | None:
        return self.target

    @property
    def segmentation_labels(self) -> torch.Tensor | None:
        return self.segmentation

    @property
    def voxel_to_ras_mm(self) -> tuple[tuple[float, ...], ...]:
        return self.geometry.voxel_to_ras_mm

    @property
    def shape_dhw(self) -> tuple[int, int, int]:
        return self.geometry.shape_dhw

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        return self.geometry.shape_xyz

    @property
    def spacing_xyz_mm(self) -> tuple[float, float, float]:
        return self.geometry.spacing_xyz_mm

    def to_metadata(self) -> dict[str, object]:
        return {
            "brain_mask_source": "raw_observation_nonzero_union",
            "geometry": self.geometry.to_dict(),
            "normalization_metadata": {
                key: value.to_dict() for key, value in self.normalization_metadata.items()
            },
            "observations": list(POINT_GUIDED_OBSERVATION_MODALITIES),
            "segmentation_present": self.segmentation is not None,
            "subject_id": self.subject_id,
            "target_present": self.target is not None,
        }


PointGuidedVolume = BraTS21PointGuidedSample


@dataclass(frozen=True)
class PointGuidedNormalizationConfig:
    """Small loader-only configuration for deterministic normalization."""

    brain_mask_threshold: float = 0.0
    normalization_epsilon: float = 1e-6
    normalization_policy: str = MASKED_ZSCORE_POLICY
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0

    def __post_init__(self) -> None:
        threshold = _finite_float(self.brain_mask_threshold, "brain_mask_threshold")
        epsilon = _finite_float(self.normalization_epsilon, "normalization_epsilon")
        if threshold < 0.0 or epsilon <= 0.0:
            raise BraTS21PointGuidedValidationError(
                "brain_mask_threshold must be non-negative and normalization_epsilon positive"
            )
        policy = _normalization_policy(self.normalization_policy)
        lower_percentile, upper_percentile = _validate_percentile_range(
            self.lower_percentile,
            self.upper_percentile,
        )
        object.__setattr__(self, "brain_mask_threshold", threshold)
        object.__setattr__(self, "normalization_epsilon", epsilon)
        object.__setattr__(self, "normalization_policy", policy)
        object.__setattr__(self, "lower_percentile", lower_percentile)
        object.__setattr__(self, "upper_percentile", upper_percentile)


@dataclass(frozen=True)
class PointGuidedBatch:
    """Geometry-homogeneous model batch produced by the point-guided collator."""

    observations: torch.Tensor
    target_t1ce: torch.Tensor
    segmentation: torch.Tensor | None
    brain_mask: torch.Tensor
    spacing_xyz_mm: torch.Tensor
    voxel_to_ras_mm: torch.Tensor
    subject_ids: tuple[str, ...]
    normalization_metadata: tuple[Mapping[str, ModalityNormalizationMetadata], ...]

    def __post_init__(self) -> None:
        if self.observations.ndim != 5 or self.observations.shape[1] != 3:
            raise BraTS21PointGuidedValidationError("batch observations must have shape [B,3,D,H,W]")
        batch_size = int(self.observations.shape[0])
        shape_dhw = tuple(int(value) for value in self.observations.shape[-3:])
        if batch_size <= 0 or self.observations.dtype != torch.float32 or not bool(torch.isfinite(self.observations).all()):
            raise BraTS21PointGuidedValidationError("batch observations must be non-empty finite float32")
        if self.target_t1ce.shape != (batch_size, 1, *shape_dhw) or self.target_t1ce.dtype != torch.float32 or not bool(torch.isfinite(self.target_t1ce).all()):
            raise BraTS21PointGuidedValidationError("target_t1ce must have shape [B,1,D,H,W] and be finite float32")
        if self.brain_mask.shape != (batch_size, 1, *shape_dhw) or self.brain_mask.dtype is not torch.bool or not bool(self.brain_mask.any()):
            raise BraTS21PointGuidedValidationError("brain_mask must be bool with shape [B,1,D,H,W]")
        if self.segmentation is not None:
            if self.segmentation.shape != (batch_size, *shape_dhw) or self.segmentation.dtype is not torch.int64:
                raise BraTS21PointGuidedValidationError("segmentation must have shape [B,D,H,W] and be int64")
            labels = set(int(value) for value in torch.unique(self.segmentation).tolist())
            if not labels.issubset(BRATS21_POINT_GUIDED_LABELS):
                raise BraTS21PointGuidedValidationError(f"batch segmentation contains unexpected labels: {sorted(labels)}")
        if self.spacing_xyz_mm.shape != (batch_size, 3) or not self.spacing_xyz_mm.is_floating_point() or not bool(torch.isfinite(self.spacing_xyz_mm).all()) or bool((self.spacing_xyz_mm <= 0.0).any()):
            raise BraTS21PointGuidedValidationError("spacing_xyz_mm must have shape [B,3] with positive finite values")
        if self.voxel_to_ras_mm.shape != (batch_size, 4, 4) or not self.voxel_to_ras_mm.is_floating_point() or not bool(torch.isfinite(self.voxel_to_ras_mm).all()):
            raise BraTS21PointGuidedValidationError("voxel_to_ras_mm must have shape [B,4,4] with finite values")
        if not bool(torch.allclose(self.voxel_to_ras_mm[:, 3, :], self.voxel_to_ras_mm.new_tensor((0.0, 0.0, 0.0, 1.0)))):
            raise BraTS21PointGuidedValidationError("voxel_to_ras_mm must be homogeneous")
        derived_spacing = torch.linalg.vector_norm(self.voxel_to_ras_mm[:, :3, :3], dim=1)
        if not bool(torch.allclose(derived_spacing, self.spacing_xyz_mm, atol=1e-5, rtol=0.0)):
            raise BraTS21PointGuidedValidationError("spacing_xyz_mm disagrees with voxel_to_ras_mm")
        if len(self.subject_ids) != batch_size or len(set(self.subject_ids)) != batch_size:
            raise BraTS21PointGuidedValidationError("subject_ids must contain one unique ID per batch item")
        for subject_id in self.subject_ids:
            _validate_subject_id(subject_id)
        if len(self.normalization_metadata) != batch_size:
            raise BraTS21PointGuidedValidationError("normalization_metadata must contain one record per subject")
        object.__setattr__(self, "subject_ids", tuple(str(value) for value in self.subject_ids))
        object.__setattr__(self, "normalization_metadata", tuple(self.normalization_metadata))


def _normalization_config(value: object | None) -> PointGuidedNormalizationConfig:
    if value is None:
        return PointGuidedNormalizationConfig()
    if isinstance(value, PointGuidedNormalizationConfig):
        return value
    if isinstance(value, Mapping):
        threshold = value.get("brain_mask_threshold", value.get("mask_threshold", 0.0))
        epsilon = value.get("normalization_epsilon", value.get("epsilon", 1e-6))
        policy = value.get("normalization_policy", value.get("policy", MASKED_ZSCORE_POLICY))
        lower_percentile = value.get("lower_percentile", 1.0)
        upper_percentile = value.get("upper_percentile", 99.0)
        return PointGuidedNormalizationConfig(
            brain_mask_threshold=threshold,
            normalization_epsilon=epsilon,
            normalization_policy=policy,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )
    threshold = getattr(value, "brain_mask_threshold", getattr(value, "mask_threshold", 0.0))
    epsilon = getattr(value, "normalization_epsilon", getattr(value, "epsilon", 1e-6))
    policy = getattr(value, "normalization_policy", getattr(value, "policy", MASKED_ZSCORE_POLICY))
    lower_percentile = getattr(value, "lower_percentile", 1.0)
    upper_percentile = getattr(value, "upper_percentile", 99.0)
    return PointGuidedNormalizationConfig(
        brain_mask_threshold=threshold,
        normalization_epsilon=epsilon,
        normalization_policy=policy,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )


def load_point_guided_subject(
    subject: str | Path | BraTS21PointGuidedSubject,
    *,
    require_target: bool = True,
    require_segmentation: bool = False,
    load_target: bool = True,
    load_segmentation: bool = True,
    brain_mask_threshold: float = 0.0,
    normalization_epsilon: float = 1e-6,
    normalization_policy: str = MASKED_ZSCORE_POLICY,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> BraTS21PointGuidedSample:
    """Load and normalize one full volume under the point-guided contract."""

    if isinstance(subject, BraTS21PointGuidedSubject):
        record = subject
    else:
        record = discover_point_guided_subject(subject)
    if record.target_path is None and require_target:
        raise BraTS21PointGuidedValidationError(f"{record.subject_id}: T1ce target is required")
    if record.segmentation_path is None and require_segmentation:
        raise BraTS21PointGuidedValidationError(f"{record.subject_id}: segmentation is required")
    if not isinstance(load_target, bool) or not isinstance(load_segmentation, bool):
        raise TypeError("load_target and load_segmentation must be bool")
    if require_target and not load_target:
        raise BraTS21PointGuidedValidationError("require_target=True cannot be combined with load_target=False")
    if require_segmentation and not load_segmentation:
        raise BraTS21PointGuidedValidationError(
            "require_segmentation=True cannot be combined with load_segmentation=False"
        )
    normalization_config = PointGuidedNormalizationConfig(
        brain_mask_threshold=brain_mask_threshold,
        normalization_epsilon=normalization_epsilon,
        normalization_policy=normalization_policy,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )

    raw_observations: list[np.ndarray] = []
    reference_geometry: NiftiGeometryMetadata | None = None
    for modality in POINT_GUIDED_OBSERVATION_MODALITIES:
        values, geometry = _read_nifti(record.observation_paths[modality], role=modality)
        if reference_geometry is None:
            reference_geometry = geometry
        else:
            _same_geometry(reference_geometry, geometry, role=modality)
        raw_observations.append(values)
    assert reference_geometry is not None
    observations_xyz = np.stack(raw_observations, axis=0)

    # Derive this before reading or normalizing target data.  Target and
    # segmentation therefore cannot influence observation topology.
    brain_mask_xyz = derive_input_brain_mask(
        observations_xyz,
        threshold=normalization_config.brain_mask_threshold,
    )
    brain_mask_dhw = torch.from_numpy(nifti_xyz_to_dhw(brain_mask_xyz).copy())
    normalized_observations: list[torch.Tensor] = []
    normalization: dict[str, ModalityNormalizationMetadata] = {}
    for index, modality in enumerate(POINT_GUIDED_OBSERVATION_MODALITIES):
        normalized, metadata = _normalize_masked(
            nifti_xyz_to_dhw(observations_xyz[index]),
            brain_mask_dhw.numpy(),
            modality=modality,
            epsilon=normalization_config.normalization_epsilon,
            normalization_policy=normalization_config.normalization_policy,
            lower_percentile=normalization_config.lower_percentile,
            upper_percentile=normalization_config.upper_percentile,
        )
        normalized_observations.append(normalized)
        normalization[modality] = metadata

    target: torch.Tensor | None = None
    if load_target and record.target_path is not None:
        target_xyz, target_geometry = _read_nifti(record.target_path, role=POINT_GUIDED_TARGET_MODALITY)
        _same_geometry(reference_geometry, target_geometry, role=POINT_GUIDED_TARGET_MODALITY)
        target, target_metadata = _normalize_masked(
            nifti_xyz_to_dhw(target_xyz),
            brain_mask_dhw.numpy(),
            modality=POINT_GUIDED_TARGET_MODALITY,
            epsilon=normalization_config.normalization_epsilon,
            normalization_policy=normalization_config.normalization_policy,
            lower_percentile=normalization_config.lower_percentile,
            upper_percentile=normalization_config.upper_percentile,
        )
        normalization[POINT_GUIDED_TARGET_MODALITY] = target_metadata

    segmentation: torch.Tensor | None = None
    if load_segmentation and record.segmentation_path is not None:
        segmentation_xyz, segmentation_geometry = _read_nifti(record.segmentation_path, role="segmentation")
        _same_geometry(reference_geometry, segmentation_geometry, role="segmentation")
        if not np.allclose(segmentation_xyz, np.rint(segmentation_xyz), atol=0.0, rtol=0.0):
            raise BraTS21PointGuidedValidationError("segmentation labels must be integral")
        labels = set(int(value) for value in np.unique(segmentation_xyz))
        if not labels.issubset(BRATS21_POINT_GUIDED_LABELS):
            raise BraTS21PointGuidedValidationError(f"segmentation contains unexpected labels: {sorted(labels)}")
        segmentation = torch.from_numpy(nifti_xyz_to_dhw(segmentation_xyz).astype(np.int64, copy=False))

    source_paths: dict[str, Path] = {
        modality: record.observation_paths[modality] for modality in POINT_GUIDED_OBSERVATION_MODALITIES
    }
    if load_target and record.target_path is not None:
        source_paths[POINT_GUIDED_TARGET_MODALITY] = record.target_path
    if load_segmentation and record.segmentation_path is not None:
        source_paths[POINT_GUIDED_SEGMENTATION_MODALITY] = record.segmentation_path
    return BraTS21PointGuidedSample(
        subject_id=record.subject_id,
        observations=torch.stack(normalized_observations, dim=0),
        target=None if target is None else target.unsqueeze(0),
        segmentation=segmentation,
        brain_mask=brain_mask_dhw.unsqueeze(0),
        geometry=reference_geometry,
        normalization_metadata=normalization,
        source_paths=source_paths,
    )


load_subject = load_point_guided_subject


def collate_point_guided_samples(
    samples: Sequence[BraTS21PointGuidedSample],
) -> PointGuidedBatch:
    """Collate target-bearing samples only when their geometry is identical."""

    if not samples:
        raise ValueError("cannot collate an empty point-guided sample sequence")
    if any(not isinstance(sample, BraTS21PointGuidedSample) for sample in samples):
        raise TypeError("all collated values must be BraTS21PointGuidedSample instances")
    reference = samples[0]
    if reference.target is None:
        raise BraTS21PointGuidedValidationError("point-guided batches require a T1ce target")
    reference_geometry = reference.geometry
    segmentation_present = reference.segmentation is not None
    for sample in samples[1:]:
        if sample.target is None:
            raise BraTS21PointGuidedValidationError(f"{sample.subject_id}: point-guided batches require a T1ce target")
        if sample.geometry.shape_dhw != reference_geometry.shape_dhw:
            raise BraTS21PointGuidedValidationError("cannot collate mixed volume shapes")
        if not np.allclose(sample.geometry.affine_xyz_to_ras_mm, reference_geometry.affine_xyz_to_ras_mm, atol=1e-5, rtol=0.0):
            raise BraTS21PointGuidedValidationError("cannot collate mixed voxel_to_ras_mm geometry")
        if (sample.segmentation is not None) != segmentation_present:
            raise BraTS21PointGuidedValidationError("cannot collate mixed segmentation availability")
    observations = torch.stack([sample.observations for sample in samples], dim=0)
    target_t1ce = torch.stack([sample.target for sample in samples if sample.target is not None], dim=0)
    segmentation = None
    if segmentation_present:
        segmentation = torch.stack([sample.segmentation for sample in samples if sample.segmentation is not None], dim=0)
    brain_mask = torch.stack([sample.brain_mask for sample in samples], dim=0)
    spacing = torch.tensor(
        [sample.spacing_xyz_mm for sample in samples],
        dtype=torch.float32,
        device=observations.device,
    )
    voxel_to_ras = torch.tensor(
        [sample.voxel_to_ras_mm for sample in samples],
        dtype=torch.float32,
        device=observations.device,
    )
    return PointGuidedBatch(
        observations=observations,
        target_t1ce=target_t1ce,
        segmentation=segmentation,
        brain_mask=brain_mask,
        spacing_xyz_mm=spacing,
        voxel_to_ras_mm=voxel_to_ras,
        subject_ids=tuple(sample.subject_id for sample in samples),
        normalization_metadata=tuple(sample.normalization_metadata for sample in samples),
    )


class BraTS21PointGuidedDataset(Dataset):
    """Indexable target-bearing dataset for the point-guided baseline.

    ``subject_ids`` controls the exact index order.  Each item delegates to
    :func:`load_point_guided_subject`; T1ce is required while segmentation is
    loaded only when the subject has it.
    """

    def __init__(
        self,
        root: str | Path,
        subject_ids: Iterable[str | Path | BraTS21PointGuidedSubject],
        normalization_config: object | None = None,
        *,
        require_segmentation: bool = False,
    ) -> None:
        try:
            resolved_root = Path(root).resolve(strict=True)
        except FileNotFoundError as error:
            raise BraTS21PointGuidedValidationError(f"source root does not exist: {root}") from error
        if not resolved_root.is_dir():
            raise BraTS21PointGuidedValidationError(f"source root is not a directory: {resolved_root}")
        if isinstance(subject_ids, (str, Path, BraTS21PointGuidedSubject)):
            raw_subject_ids: tuple[object, ...] = (subject_ids,)
        else:
            raw_subject_ids = tuple(subject_ids)
        if not raw_subject_ids:
            raise BraTS21PointGuidedValidationError("point-guided dataset requires at least one subject ID")
        if not isinstance(require_segmentation, bool):
            raise TypeError("require_segmentation must be bool")
        normalized_ids = tuple(_subject_id_from_item(item) for item in raw_subject_ids)
        if len(set(normalized_ids)) != len(normalized_ids):
            raise BraTS21PointGuidedValidationError("subject_ids must be unique")
        self.root = resolved_root
        self.subject_ids = normalized_ids
        self.normalization_config = _normalization_config(normalization_config)
        self.require_segmentation = require_segmentation

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int) -> BraTS21PointGuidedSample:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("point-guided dataset index must be an integer")
        try:
            subject_id = self.subject_ids[index]
        except IndexError as error:
            raise IndexError(f"subject index out of range: {index}") from error
        return load_point_guided_subject(
            self.root / subject_id,
            require_target=True,
            require_segmentation=self.require_segmentation,
            brain_mask_threshold=self.normalization_config.brain_mask_threshold,
            normalization_epsilon=self.normalization_config.normalization_epsilon,
            normalization_policy=self.normalization_config.normalization_policy,
            lower_percentile=self.normalization_config.lower_percentile,
            upper_percentile=self.normalization_config.upper_percentile,
        )


def _subject_id_from_item(item: object) -> str:
    if isinstance(item, BraTS21PointGuidedSubject):
        return item.subject_id
    if isinstance(item, Path):
        return _validate_subject_id(item.name)
    return _validate_subject_id(item)


def _normalize_caps(
    max_subjects: Mapping[str, int | None] | int | None,
    *,
    max_train_subjects: int | None,
    max_val_subjects: int | None,
    max_test_subjects: int | None,
) -> dict[str, int | None]:
    caps: dict[str, int | None] = {"train": None, "val": None, "test": None}
    if isinstance(max_subjects, bool):
        raise BraTS21PointGuidedValidationError("max_subjects cannot be bool")
    if isinstance(max_subjects, int):
        if max_subjects < 0:
            raise BraTS21PointGuidedValidationError("max_subjects must be non-negative")
        caps = {name: max_subjects for name in caps}
    elif max_subjects is not None:
        aliases = {"validation": "val", "valid": "val", "development": "val"}
        for raw_name, raw_cap in max_subjects.items():
            name = aliases.get(str(raw_name), str(raw_name))
            if name not in caps:
                raise BraTS21PointGuidedValidationError(f"unknown split cap: {raw_name!r}")
            if raw_cap is not None and (isinstance(raw_cap, bool) or not isinstance(raw_cap, int) or raw_cap < 0):
                raise BraTS21PointGuidedValidationError(f"max_subjects[{raw_name!r}] must be a non-negative integer or None")
            caps[name] = raw_cap
    explicit = {
        "train": max_train_subjects,
        "val": max_val_subjects,
        "test": max_test_subjects,
    }
    for name, cap in explicit.items():
        if cap is not None:
            if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
                raise BraTS21PointGuidedValidationError(f"{name} subject cap must be a non-negative integer")
            caps[name] = cap
    return caps


def _validate_fractions(
    split_fractions: Sequence[float] | None,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[float, float, float]:
    values = tuple(float(value) for value in (split_fractions or (train_fraction, val_fraction, test_fraction)))
    if len(values) != 3 or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise BraTS21PointGuidedValidationError("split fractions must be three finite non-negative values")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise BraTS21PointGuidedValidationError("train/val/test split fractions must sum to one")
    if values[0] <= 0.0:
        raise BraTS21PointGuidedValidationError("train fraction must be positive")
    return values  # type: ignore[return-value]


@dataclass(frozen=True)
class BraTS21PointGuidedSplit:
    """A deterministic subject-level split and its provenance hash."""

    all_subject_ids: tuple[str, ...]
    train_subject_ids: tuple[str, ...]
    val_subject_ids: tuple[str, ...]
    test_subject_ids: tuple[str, ...]
    split_fractions: tuple[float, float, float]
    seed: int
    max_subjects: Mapping[str, int | None]
    excluded_subject_ids: tuple[str, ...]
    split_hash: str

    def __post_init__(self) -> None:
        all_ids = tuple(_validate_subject_id(value) for value in self.all_subject_ids)
        if len(set(all_ids)) != len(all_ids) or all_ids != tuple(sorted(all_ids)):
            raise BraTS21PointGuidedValidationError("all_subject_ids must be unique and sorted")
        groups = {
            "train": tuple(self.train_subject_ids),
            "val": tuple(self.val_subject_ids),
            "test": tuple(self.test_subject_ids),
        }
        selected = [subject_id for group in groups.values() for subject_id in group]
        if len(selected) != len(set(selected)) or any(subject_id not in all_ids for subject_id in selected):
            raise BraTS21PointGuidedValidationError("split subject groups must be disjoint members of all_subject_ids")
        excluded = tuple(_validate_subject_id(value) for value in self.excluded_subject_ids)
        if set(excluded) != set(all_ids) - set(selected):
            raise BraTS21PointGuidedValidationError("excluded_subject_ids do not match capped split membership")
        fractions = _validate_fractions(self.split_fractions, 0.8, 0.1, 0.1)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise BraTS21PointGuidedValidationError("split seed must be an integer")
        caps = {str(key): value for key, value in self.max_subjects.items()}
        if set(caps) != {"train", "val", "test"}:
            raise BraTS21PointGuidedValidationError("max_subjects must contain train, val, and test")
        if not isinstance(self.split_hash, str) or len(self.split_hash) != 64:
            raise BraTS21PointGuidedValidationError("split_hash must be a SHA-256 digest")
        object.__setattr__(self, "all_subject_ids", all_ids)
        object.__setattr__(self, "train_subject_ids", tuple(groups["train"]))
        object.__setattr__(self, "val_subject_ids", tuple(groups["val"]))
        object.__setattr__(self, "test_subject_ids", tuple(groups["test"]))
        object.__setattr__(self, "split_fractions", fractions)
        object.__setattr__(self, "max_subjects", MappingProxyType(caps))
        object.__setattr__(self, "excluded_subject_ids", tuple(sorted(excluded)))

    @property
    def validation_subject_ids(self) -> tuple[str, ...]:
        return self.val_subject_ids

    @property
    def assignments(self) -> Mapping[str, str]:
        result = {subject_id: "train" for subject_id in self.train_subject_ids}
        result.update({subject_id: "val" for subject_id in self.val_subject_ids})
        result.update({subject_id: "test" for subject_id in self.test_subject_ids})
        result.update({subject_id: "excluded" for subject_id in self.excluded_subject_ids})
        return MappingProxyType(result)

    def to_dict(self) -> dict[str, object]:
        return {
            "all_subject_ids": self.all_subject_ids,
            "assignments": dict(self.assignments),
            "excluded_subject_ids": self.excluded_subject_ids,
            "max_subjects": dict(self.max_subjects),
            "seed": self.seed,
            "split_fractions": self.split_fractions,
            "split_hash": self.split_hash,
            "test_subject_ids": self.test_subject_ids,
            "train_subject_ids": self.train_subject_ids,
            "val_subject_ids": self.val_subject_ids,
        }


def deterministic_subject_split(
    subjects: Iterable[str | Path | BraTS21PointGuidedSubject],
    *,
    split_fractions: Sequence[float] | None = None,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
    max_subjects: Mapping[str, int | None] | int | None = None,
    max_train_subjects: int | None = None,
    max_val_subjects: int | None = None,
    max_test_subjects: int | None = None,
) -> BraTS21PointGuidedSplit:
    """Assign subjects deterministically, with optional per-split caps.

    Subject IDs are first ranked by ``SHA256(seed + subject_id)``.  Largest
    remainder allocation gives deterministic split counts, then caps retain
    the earliest ranked members of each split.  The hash covers the complete
    input cohort, fractions, seed, caps, final assignments, and exclusions.
    """

    fractions = _validate_fractions(split_fractions, train_fraction, val_fraction, test_fraction)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise BraTS21PointGuidedValidationError("split seed must be an integer")
    ids = tuple(sorted({_subject_id_from_item(item) for item in subjects}))
    if not ids:
        raise BraTS21PointGuidedValidationError("subject split requires at least one subject")
    caps = _normalize_caps(
        max_subjects,
        max_train_subjects=max_train_subjects,
        max_val_subjects=max_val_subjects,
        max_test_subjects=max_test_subjects,
    )
    ranked = tuple(
        sorted(
            ids,
            key=lambda subject_id: (
                hashlib.sha256(f"{seed}:{subject_id}".encode("utf-8")).hexdigest(),
                subject_id,
            ),
        )
    )
    exact_counts = [len(ids) * fraction for fraction in fractions]
    counts = [int(math.floor(value)) for value in exact_counts]
    remainder = len(ids) - sum(counts)
    fractional_order = sorted(
        range(3),
        key=lambda index: (-(exact_counts[index] - counts[index]), index),
    )
    for index in fractional_order[:remainder]:
        counts[index] += 1
    raw_groups = (
        ranked[: counts[0]],
        ranked[counts[0] : counts[0] + counts[1]],
        ranked[counts[0] + counts[1] :],
    )
    names = ("train", "val", "test")
    groups = tuple(
        tuple(group[: caps[name]]) if caps[name] is not None else tuple(group)
        for name, group in zip(names, raw_groups)
    )
    selected = {subject_id for group in groups for subject_id in group}
    excluded = tuple(sorted(set(ids) - selected))
    # Build the assignment object separately to keep the hash payload clear
    # and avoid depending on set iteration order.
    assignments = {
        subject_id: split_name
        for split_name, group in zip(names, groups)
        for subject_id in group
    }
    assignments.update({subject_id: "excluded" for subject_id in excluded})
    payload = {
        "all_subject_ids": ids,
        "assignments": dict(sorted(assignments.items())),
        "caps": caps,
        "fractions": fractions,
        "seed": seed,
        "version": SPLIT_VERSION,
    }
    split_hash = _canonical_hash(payload)
    return BraTS21PointGuidedSplit(
        all_subject_ids=ids,
        train_subject_ids=groups[0],
        val_subject_ids=groups[1],
        test_subject_ids=groups[2],
        split_fractions=fractions,
        seed=seed,
        max_subjects=caps,
        excluded_subject_ids=excluded,
        split_hash=split_hash,
    )


build_subject_split = deterministic_subject_split
make_subject_split = deterministic_subject_split


class BraTS21PointGuidedAdapter:
    """Small indexable adapter around the additive full-volume contract."""

    def __init__(
        self,
        root: str | Path,
        *,
        require_target: bool = True,
        require_segmentation: bool = False,
        load_target: bool = True,
        load_segmentation: bool = True,
        brain_mask_threshold: float = 0.0,
        normalization_epsilon: float = 1e-6,
        normalization_policy: str = MASKED_ZSCORE_POLICY,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
    ) -> None:
        try:
            resolved_root = Path(root).resolve(strict=True)
        except FileNotFoundError as error:
            raise BraTS21PointGuidedValidationError(f"source root does not exist: {root}") from error
        if not resolved_root.is_dir():
            raise BraTS21PointGuidedValidationError(f"source root is not a directory: {resolved_root}")
        if (
            not isinstance(require_target, bool)
            or not isinstance(require_segmentation, bool)
            or not isinstance(load_target, bool)
            or not isinstance(load_segmentation, bool)
        ):
            raise TypeError("target and segmentation loading requirements must be bool")
        if require_target and not load_target:
            raise BraTS21PointGuidedValidationError("require_target=True cannot be combined with load_target=False")
        if require_segmentation and not load_segmentation:
            raise BraTS21PointGuidedValidationError(
                "require_segmentation=True cannot be combined with load_segmentation=False"
            )
        normalization_config = PointGuidedNormalizationConfig(
            brain_mask_threshold=brain_mask_threshold,
            normalization_epsilon=normalization_epsilon,
            normalization_policy=normalization_policy,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )
        self.root = resolved_root
        self.require_target = require_target
        self.require_segmentation = require_segmentation
        self.load_target = load_target
        self.load_segmentation = load_segmentation
        self.brain_mask_threshold = normalization_config.brain_mask_threshold
        self.normalization_epsilon = normalization_config.normalization_epsilon
        self.normalization_policy = normalization_config.normalization_policy
        self.lower_percentile = normalization_config.lower_percentile
        self.upper_percentile = normalization_config.upper_percentile

    def discover_subjects(self) -> tuple[BraTS21PointGuidedSubject, ...]:
        return discover_point_guided_subjects(self.root)

    def split(self, **kwargs: Any) -> BraTS21PointGuidedSplit:
        return deterministic_subject_split(
            (subject.subject_id for subject in self.discover_subjects()),
            **kwargs,
        )

    def load(
        self,
        subject: str | Path | BraTS21PointGuidedSubject,
        *,
        load_target: bool | None = None,
        load_segmentation: bool | None = None,
    ) -> BraTS21PointGuidedSample:
        if isinstance(subject, str) and BRATS21_POINT_GUIDED_SUBJECT_PATTERN.fullmatch(subject):
            subject = self.root / subject
        return load_point_guided_subject(
            subject,
            require_target=self.require_target,
            require_segmentation=self.require_segmentation,
            load_target=self.load_target if load_target is None else load_target,
            load_segmentation=self.load_segmentation if load_segmentation is None else load_segmentation,
            brain_mask_threshold=self.brain_mask_threshold,
            normalization_epsilon=self.normalization_epsilon,
            normalization_policy=self.normalization_policy,
            lower_percentile=self.lower_percentile,
            upper_percentile=self.upper_percentile,
        )

    def __len__(self) -> int:
        return len(self.discover_subjects())

    def __getitem__(self, index: int | str) -> BraTS21PointGuidedSample:
        subjects = self.discover_subjects()
        if isinstance(index, bool):
            raise TypeError("subject index must be an integer or subject ID")
        if isinstance(index, int):
            try:
                subject = subjects[index]
            except IndexError as error:
                raise IndexError(f"subject index out of range: {index}") from error
        else:
            subject_id = _validate_subject_id(index)
            matches = [subject for subject in subjects if subject.subject_id == subject_id]
            if not matches:
                raise KeyError(subject_id)
            subject = matches[0]
        return self.load(subject)


__all__ = [
    "BRATS21_POINT_GUIDED_LABELS",
    "BRATS21_POINT_GUIDED_SUBJECT_PATTERN",
    "MASKED_ROBUST_01_POLICY",
    "MASKED_ZSCORE_POLICY",
    "SUPPORTED_NORMALIZATION_POLICIES",
    "BraTS21PointGuidedAdapter",
    "BraTS21PointGuidedDataset",
    "BraTS21PointGuidedDependencyError",
    "BraTS21PointGuidedError",
    "BraTS21PointGuidedSample",
    "BraTS21PointGuidedSplit",
    "BraTS21PointGuidedSubject",
    "BraTS21PointGuidedValidationError",
    "ModalityNormalizationMetadata",
    "NormalizationMetadata",
    "NiftiGeometryMetadata",
    "NORMALIZATION_VERSION",
    "POINT_GUIDED_MODALITIES",
    "POINT_GUIDED_OBSERVATION_MODALITIES",
    "POINT_GUIDED_SEGMENTATION_MODALITY",
    "POINT_GUIDED_TARGET_MODALITY",
    "PointGuidedVolume",
    "PointGuidedBatch",
    "PointGuidedNormalizationConfig",
    "SPLIT_VERSION",
    "build_subject_split",
    "collate_point_guided_samples",
    "derive_input_brain_mask",
    "deterministic_subject_split",
    "discover_point_guided_subject",
    "discover_point_guided_subjects",
    "discover_subject",
    "discover_subjects",
    "load_point_guided_subject",
    "load_subject",
    "make_subject_split",
    "nifti_xyz_to_dhw",
    "xyz_shape_to_dhw",
    "xyz_to_dhw",
]
