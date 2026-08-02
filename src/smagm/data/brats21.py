"""Fail-closed BraTS21 discovery, geometry validation, and plane preparation.

The BraTS directory contains dense source volumes and evaluator-only labels.
This module is deliberately an offline preparation boundary: training receives
only manifest-bound rank-2 NumPy payloads produced from a deterministic,
value-independent plane schedule.  It never exposes a NIfTI path to the
maintained ledger decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np

from ..contracts.coordinates import PhysicalPlane, SourceAffineTransform, SourceConvention


BRATS21_MODALITIES = ("t1", "t1ce", "t2", "flair")
BRATS21_SEGMENTATION = "seg"
BRATS21_LABELS = frozenset({0, 1, 2, 4})
BRATS21_PATIENT_PATTERN = re.compile(r"^BraTS2021_(?P<number>\d{5})$")


class BraTS21DependencyError(RuntimeError):
    """Raised when the optional NIfTI reader is unavailable."""


class BraTS21ValidationError(ValueError):
    """A patient or source-root validation failure with a clear cause."""


@dataclass(frozen=True)
class BraTS21Patient:
    """A discovered patient with exact modality suffix bindings."""

    patient_id: str
    directory: Path
    modality_paths: Mapping[str, Path]
    segmentation_path: Path | None
    unknown_nifti_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if BRATS21_PATIENT_PATTERN.fullmatch(self.patient_id) is None:
            raise BraTS21ValidationError(f"malformed BraTS21 patient ID: {self.patient_id!r}")
        if tuple(sorted(self.modality_paths)) != tuple(sorted(BRATS21_MODALITIES)):
            raise BraTS21ValidationError(
                f"{self.patient_id}: required modalities are exactly {BRATS21_MODALITIES}"
            )
        object.__setattr__(self, "directory", Path(self.directory).resolve())
        object.__setattr__(self, "modality_paths", MappingProxyType(dict(sorted(self.modality_paths.items()))))
        object.__setattr__(self, "unknown_nifti_files", tuple(sorted(self.unknown_nifti_files)))


@dataclass(frozen=True)
class BraTS21VolumeSummary:
    """Header and bounded data diagnostics for one NIfTI volume."""

    suffix: str
    shape_xyz: tuple[int, int, int]
    spacing_xyz_mm: tuple[float, float, float]
    affine: tuple[tuple[float, ...], ...]
    orientation: tuple[str, ...]
    dtype: str
    numeric_range: tuple[float, float] | None
    segmentation_labels: tuple[int, ...] | None
    source_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "affine": self.affine,
            "dtype": self.dtype,
            "numeric_range": self.numeric_range,
            "orientation": self.orientation,
            "segmentation_labels": self.segmentation_labels,
            "shape_xyz": self.shape_xyz,
            "source_hash": self.source_hash,
            "spacing_xyz_mm": self.spacing_xyz_mm,
            "suffix": self.suffix,
        }


@dataclass(frozen=True)
class BraTS21PatientValidation:
    """Validation result for one patient; errors are retained for inventory."""

    patient_id: str
    valid: bool
    error: str | None
    summaries: tuple[BraTS21VolumeSummary, ...]
    has_segmentation: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.error,
            "has_segmentation": self.has_segmentation,
            "patient_id": self.patient_id,
            "summaries": [item.to_dict() for item in self.summaries],
            "valid": self.valid,
        }


def _nibabel() -> Any:
    try:
        import nibabel as nib  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - depends on environment
        raise BraTS21DependencyError(
            "BraTS21 support requires the optional 'nibabel' dependency; install smagm[real-data]"
        ) from error
    return nib


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _nifti_files(directory: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in directory.iterdir() if path.is_file() and path.name.endswith((".nii", ".nii.gz"))))


def _suffix(patient_id: str, path: Path) -> str | None:
    prefix = f"{patient_id}_"
    if not path.name.startswith(prefix):
        return None
    name = path.name[len(prefix):]
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return None


def discover_patient(directory: str | Path) -> BraTS21Patient:
    """Discover exact BraTS suffixes without opening any NIfTI payload."""

    path = Path(directory).resolve(strict=True)
    if not path.is_dir():
        raise BraTS21ValidationError(f"BraTS21 patient path is not a directory: {path}")
    match = BRATS21_PATIENT_PATTERN.fullmatch(path.name)
    if match is None:
        raise BraTS21ValidationError(f"malformed BraTS21 patient directory name: {path.name!r}")
    discovered: dict[str, list[Path]] = {}
    unknown: list[str] = []
    for nifti_path in _nifti_files(path):
        suffix = _suffix(path.name, nifti_path)
        if suffix is None or suffix not in (*BRATS21_MODALITIES, BRATS21_SEGMENTATION):
            unknown.append(nifti_path.name)
            continue
        discovered.setdefault(suffix, []).append(nifti_path)
    missing = [name for name in BRATS21_MODALITIES if len(discovered.get(name, ())) != 1]
    if missing:
        details = {name: [item.name for item in discovered.get(name, ())] for name in missing}
        raise BraTS21ValidationError(f"{path.name}: missing or duplicate required modalities: {details}")
    segmentation = discovered.get(BRATS21_SEGMENTATION, ())
    if len(segmentation) > 1:
        raise BraTS21ValidationError(f"{path.name}: duplicate segmentation files: {[item.name for item in segmentation]}")
    return BraTS21Patient(
        path.name,
        path,
        {name: discovered[name][0] for name in BRATS21_MODALITIES},
        segmentation[0] if segmentation else None,
        tuple(unknown),
    )


def discover_patients(root: str | Path) -> tuple[BraTS21Patient, ...]:
    """Return structurally complete patients in deterministic ID order.

    This strict helper raises on the first malformed patient.  Inventory
    commands that need a complete error count should iterate direct children
    and call :func:`discover_patient` individually so one bad patient cannot
    be mistaken for a valid sparse source.
    """

    source_root = Path(root).resolve(strict=True)
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    patients: list[BraTS21Patient] = []
    for path in sorted(item for item in source_root.iterdir() if item.is_dir()):
        if BRATS21_PATIENT_PATTERN.fullmatch(path.name) is None:
            raise BraTS21ValidationError(f"malformed patient directory name: {path.name!r}")
        patients.append(discover_patient(path))
    return tuple(sorted(patients, key=lambda item: item.patient_id))


def _resolved_affine(image: Any) -> np.ndarray:
    """Resolve qform/sform with conflict rejection and finite geometry checks."""

    header = image.header
    qform, qcode = header.get_qform(coded=True)
    sform, scode = header.get_sform(coded=True)
    if qcode > 0 and scode > 0 and not np.allclose(qform, sform, atol=1e-4, rtol=0.0):
        raise BraTS21ValidationError("NIfTI qform and sform disagree")
    affine = np.asarray(sform if scode > 0 else qform if qcode > 0 else image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all() or not np.allclose(affine[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6, rtol=0.0):
        raise BraTS21ValidationError("NIfTI affine is non-finite or not homogeneous")
    if abs(float(np.linalg.det(affine[:3, :3]))) <= 1e-8:
        raise BraTS21ValidationError("NIfTI affine has a singular spatial block")
    return affine


def _summary(
    path: Path,
    suffix: str,
    *,
    include_data: bool,
    is_segmentation: bool,
    include_source_hash: bool,
) -> BraTS21VolumeSummary:
    nib = _nibabel()
    try:
        image = nib.load(str(path), mmap=True)
        shape = tuple(int(value) for value in image.header.get_data_shape())
        if len(shape) != 3 or any(value <= 0 for value in shape):
            raise BraTS21ValidationError(f"{path.name}: NIfTI must be a non-empty 3D volume")
        affine = _resolved_affine(image)
        spacing = tuple(float(value) for value in np.linalg.norm(affine[:3, :3], axis=0))
        if not all(np.isfinite(spacing)) or any(value <= 0 for value in spacing):
            raise BraTS21ValidationError(f"{path.name}: voxel spacing is invalid")
        values = np.asanyarray(image.dataobj) if include_data else None
        numeric_range: tuple[float, float] | None = None
        labels: tuple[int, ...] | None = None
        if values is not None:
            if not np.isfinite(values).all():
                raise BraTS21ValidationError(f"{path.name}: volume contains non-finite values")
            numeric_range = (float(np.min(values)), float(np.max(values)))
            if is_segmentation:
                if values.dtype.kind not in "biu" and not np.allclose(values, np.rint(values), atol=0.0, rtol=0.0):
                    raise BraTS21ValidationError(f"{path.name}: segmentation labels are not integral")
                labels = tuple(sorted(int(value) for value in np.unique(values)))
                if not set(labels).issubset(BRATS21_LABELS):
                    raise BraTS21ValidationError(f"{path.name}: unexpected BraTS labels {labels}")
        nib_orientation = tuple(str(value) for value in nib.aff2axcodes(affine))
        if len(nib_orientation) != 3:
            raise BraTS21ValidationError(f"{path.name}: NIfTI orientation is unavailable")
        return BraTS21VolumeSummary(
            suffix=suffix,
            shape_xyz=shape,
            spacing_xyz_mm=spacing,
            affine=tuple(tuple(float(value) for value in row) for row in affine),
            orientation=nib_orientation,
            dtype=str(np.asanyarray(image.dataobj).dtype if include_data else image.get_data_dtype()),
            numeric_range=numeric_range,
            segmentation_labels=labels,
            source_hash=_sha256_file(path) if include_source_hash else "",
        )
    except BraTS21ValidationError:
        raise
    except Exception as error:
        raise BraTS21ValidationError(f"{path.name}: cannot read NIfTI: {error}") from error


def validate_patient(
    patient: BraTS21Patient,
    *,
    require_segmentation: bool = False,
    include_data: bool = False,
    include_source_hash: bool = True,
) -> BraTS21PatientValidation:
    """Validate dimensions, qform/sform-resolved affine, finite data, and labels."""

    if not patient.modality_paths:
        return BraTS21PatientValidation(patient.patient_id, False, "missing required modality files", (), False)
    try:
        summaries = [
            _summary(
                path,
                suffix,
                include_data=include_data,
                is_segmentation=False,
                include_source_hash=include_source_hash,
            )
            for suffix, path in patient.modality_paths.items()
        ]
        if patient.segmentation_path is not None:
            summaries.append(
                _summary(
                    patient.segmentation_path,
                    BRATS21_SEGMENTATION,
                    include_data=include_data,
                    is_segmentation=True,
                    include_source_hash=include_source_hash,
                )
            )
        if require_segmentation and patient.segmentation_path is None:
            raise BraTS21ValidationError("segmentation is required for this evaluator-bound preparation")
        reference = summaries[0]
        for item in summaries[1:]:
            if item.shape_xyz != reference.shape_xyz:
                raise BraTS21ValidationError(f"shape mismatch: {reference.suffix}={reference.shape_xyz}, {item.suffix}={item.shape_xyz}")
            if not np.allclose(item.affine, reference.affine, atol=1e-4, rtol=0.0):
                raise BraTS21ValidationError(f"affine mismatch between {reference.suffix} and {item.suffix}")
        return BraTS21PatientValidation(patient.patient_id, True, None, tuple(summaries), patient.segmentation_path is not None)
    except BraTS21ValidationError as error:
        return BraTS21PatientValidation(patient.patient_id, False, str(error), (), patient.segmentation_path is not None)


def _round_fraction(fraction: float, depth: int) -> int:
    if not 0.0 < fraction < 1.0 or depth < 3:
        raise ValueError("plane schedule fractions require 0 < fraction < 1 and depth >= 3")
    return int(round(float(fraction) * float(depth - 1)))


def deterministic_plane_schedule(
    shape_xyz: Iterable[int],
    *,
    fractions: Mapping[str, float] | None = None,
) -> tuple[tuple[str, str, int], ...]:
    """Select four context planes and one held-out FLAIR plane from dimensions only."""

    shape = tuple(int(value) for value in shape_xyz)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("shape_xyz must contain three positive dimensions")
    values = {
        "t1": 0.34,
        "t1ce": 0.40,
        "t2": 0.46,
        "flair_context": 0.48,
        "flair_target": 0.52,
    }
    if fractions is not None:
        values.update({str(key): float(value) for key, value in fractions.items()})
    schedule = (
        ("t1", "context", _round_fraction(values["t1"], shape[2])),
        ("t1ce", "context", _round_fraction(values["t1ce"], shape[2])),
        ("t2", "context", _round_fraction(values["t2"], shape[2])),
        ("flair", "context", _round_fraction(values["flair_context"], shape[2])),
        ("flair", "target", _round_fraction(values["flair_target"], shape[2])),
    )
    indices = [item[2] for item in schedule]
    if len(set(indices)) != len(indices):
        raise ValueError(f"plane schedule collided for shape {shape}: {schedule}")
    context = {index for _, role, index in schedule if role == "context"}
    if any(index in context for _, role, index in schedule if role == "target"):
        raise ValueError("target plane overlaps a context plane")
    return schedule


def plane_from_nifti(
    affine: Iterable[Iterable[float]],
    shape_xyz: Iterable[int],
    slice_index: int,
    *,
    observation_id: str,
    inplane_stride_vu: tuple[int, int] = (1, 1),
    slice_position_index: float | None = None,
) -> PhysicalPlane:
    """Build an axial physical plane while preserving the source affine.

    Source arrays are ``[x, y, z]`` and payloads are ``[v, u] == [x, y]``.
    The source-transform columns therefore use ``y, x, z`` order, matching the
    repository's explicit plane-index convention ``[u, v, slice]``.
    """

    matrix = np.asarray(tuple(tuple(float(value) for value in row) for row in affine), dtype=np.float64)
    shape = tuple(int(value) for value in shape_xyz)
    stride_v, stride_u = (int(inplane_stride_vu[0]), int(inplane_stride_vu[1]))
    sample_index = float(slice_index if slice_position_index is None else slice_position_index)
    if (
        matrix.shape != (4, 4)
        or len(shape) != 3
        or not np.isfinite(sample_index)
        or not 0.0 <= sample_index <= float(shape[2] - 1)
    ):
        raise ValueError("invalid NIfTI affine, shape, or axial slice index")
    if stride_v <= 0 or stride_u <= 0:
        raise ValueError("inplane_stride_vu must be positive")
    origin = matrix @ np.asarray((0.0, 0.0, sample_index, 1.0))
    source = matrix.copy()
    source[:3, 3] = origin[:3]
    source[:3, 0] = matrix[:3, 1] * stride_u
    source[:3, 1] = matrix[:3, 0] * stride_v
    transform = SourceAffineTransform(source, SourceConvention.NIFTI_RAS)
    axis_u = np.asarray(transform.axis_u_step_ras_mm, dtype=np.float64)
    axis_v = np.asarray(transform.axis_v_step_ras_mm, dtype=np.float64)
    normal = np.asarray(transform.signed_slice_axis_ras, dtype=np.float64)
    shape_hw = ((shape[0] + stride_v - 1) // stride_v, (shape[1] + stride_u - 1) // stride_u)
    return PhysicalPlane(
        transform.origin_ras_mm,
        axis_u / np.linalg.norm(axis_u),
        axis_v / np.linalg.norm(axis_v),
        (float(np.linalg.norm(axis_u)), float(np.linalg.norm(axis_v))),
        float(np.linalg.norm(matrix[:3, 2])),
        shape_hw,
        tuple(float(value) for value in normal),
        source_transform=transform,
        observation_id=observation_id,
    )


def extract_axial_plane(path: str | Path, slice_index: int, *, inplane_stride_vu: tuple[int, int] = (1, 1)) -> np.ndarray:
    """Read one bounded nearest-neighbor axial plane as finite float32 data."""

    return extract_axial_plane_at_position(
        path,
        float(slice_index),
        inplane_stride_vu=inplane_stride_vu,
        interpolation="nearest",
    )


def extract_axial_plane_at_position(
    path: str | Path,
    slice_position_index: float,
    *,
    inplane_stride_vu: tuple[int, int] = (1, 1),
    interpolation: str = "nearest",
) -> np.ndarray:
    """Read one bounded axial plane at an integer or fractional source index.

    ``linear`` interpolation is reserved for hidden intensity targets and is
    invoked only by the receipt-gated reader. Evaluator label planes use
    ``nearest`` so discrete segmentation values are never interpolated.
    """

    image = _nibabel().load(str(Path(path)), mmap=True)
    data = image.dataobj
    if (
        getattr(data, "ndim", None) != 3
        or not np.isfinite(float(slice_position_index))
        or not 0.0 <= float(slice_position_index) <= float(data.shape[2] - 1)
        or interpolation not in ("linear", "nearest")
    ):
        raise BraTS21ValidationError("requested axial slice is outside a valid 3D NIfTI volume")
    position = float(slice_position_index)
    lower = int(np.floor(position))
    upper = min(lower + 1, int(data.shape[2] - 1))
    if interpolation == "nearest":
        source = int(np.floor(position + 0.5))
        plane_source = np.asanyarray(data[:, :, source])
    else:
        lower_plane = np.asarray(data[:, :, lower], dtype=np.float32)
        if upper == lower:
            plane_source = lower_plane
        else:
            upper_plane = np.asarray(data[:, :, upper], dtype=np.float32)
            weight = np.float32(position - lower)
            plane_source = (np.float32(1.0) - weight) * lower_plane + weight * upper_plane
    stride_v, stride_u = inplane_stride_vu
    if stride_v <= 0 or stride_u <= 0:
        raise BraTS21ValidationError("inplane_stride_vu must be positive")
    plane = np.asarray(plane_source[::stride_v, ::stride_u], dtype=np.float32)
    if not np.isfinite(plane).all():
        raise BraTS21ValidationError("extracted plane contains non-finite values")
    return np.ascontiguousarray(plane)


def npy_bytes(array: np.ndarray) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def json_hash(value: object) -> str:
    return canonical_hash(value)


__all__ = [
    "BRATS21_LABELS",
    "BRATS21_MODALITIES",
    "BRATS21_PATIENT_PATTERN",
    "BRATS21_SEGMENTATION",
    "BraTS21DependencyError",
    "BraTS21Patient",
    "BraTS21PatientValidation",
    "BraTS21ValidationError",
    "BraTS21VolumeSummary",
    "canonical_hash",
    "deterministic_plane_schedule",
    "discover_patient",
    "discover_patients",
    "extract_axial_plane",
    "extract_axial_plane_at_position",
    "json_hash",
    "npy_bytes",
    "plane_from_nifti",
    "validate_patient",
]
