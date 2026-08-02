from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smagm.contracts.episode import EpisodeLedger
from smagm.contracts.observation import PatientSplitRegistry
from smagm.data.brats21 import (
    BraTS21ValidationError,
    deterministic_plane_schedule,
    discover_patient,
    extract_axial_plane_at_position,
    plane_from_nifti,
    validate_patient,
)
from smagm.data.brats21_prepare import load_prepared_bundle, prepare_brats21_smoke


nib = pytest.importorskip("nibabel")


def _write_patient(root: Path, *, shape: tuple[int, int, int] = (8, 9, 11), affine: np.ndarray | None = None) -> Path:
    patient = root / "BraTS2021_00000"
    patient.mkdir(parents=True)
    matrix = np.eye(4, dtype=np.float64) if affine is None else affine
    for index, suffix in enumerate(("t1", "t1ce", "t2", "flair")):
        values = np.arange(np.prod(shape), dtype=np.int16).reshape(shape) + index
        nib.save(nib.Nifti1Image(values, matrix), patient / f"BraTS2021_00000_{suffix}.nii.gz")
    segmentation = np.zeros(shape, dtype=np.uint8)
    segmentation[1:3, 1:3, 4] = 1
    nib.save(nib.Nifti1Image(segmentation, matrix), patient / "BraTS2021_00000_seg.nii.gz")
    return patient


def test_filename_discovery_and_complete_validation(tmp_path: Path) -> None:
    patient_dir = _write_patient(tmp_path)
    patient = discover_patient(patient_dir)
    assert tuple(patient.modality_paths) == ("flair", "t1", "t1ce", "t2")
    result = validate_patient(patient, require_segmentation=True, include_data=True, include_source_hash=False)
    assert result.valid is True
    assert result.has_segmentation is True
    assert {summary.suffix for summary in result.summaries} == {"t1", "t1ce", "t2", "flair", "seg"}
    assert result.summaries[-1].segmentation_labels == (0, 1)


def test_missing_modality_is_rejected(tmp_path: Path) -> None:
    patient_dir = _write_patient(tmp_path)
    (patient_dir / "BraTS2021_00000_t2.nii.gz").unlink()
    with pytest.raises(BraTS21ValidationError, match="missing or duplicate"):
        discover_patient(patient_dir)


def test_shape_mismatch_is_rejected(tmp_path: Path) -> None:
    patient_dir = _write_patient(tmp_path)
    nib.save(
        nib.Nifti1Image(np.zeros((8, 9, 10), dtype=np.int16), np.eye(4)),
        patient_dir / "BraTS2021_00000_t2.nii.gz",
    )
    result = validate_patient(discover_patient(patient_dir), include_source_hash=False)
    assert result.valid is False
    assert "shape mismatch" in str(result.error)


def test_affine_mismatch_is_rejected(tmp_path: Path) -> None:
    patient_dir = _write_patient(tmp_path)
    shifted = np.eye(4)
    shifted[0, 3] = 2.0
    nib.save(
        nib.Nifti1Image(np.zeros((8, 9, 11), dtype=np.int16), shifted),
        patient_dir / "BraTS2021_00000_t2.nii.gz",
    )
    result = validate_patient(discover_patient(patient_dir), include_source_hash=False)
    assert result.valid is False
    assert "affine mismatch" in str(result.error)


def test_plane_schedule_is_deterministic_and_disjoint() -> None:
    first = deterministic_plane_schedule((240, 240, 155))
    second = deterministic_plane_schedule((240, 240, 155))
    assert first == second
    context = {index for _, role, index in first if role == "context"}
    target = {index for _, role, index in first if role == "target"}
    assert context.isdisjoint(target)


def test_plane_geometry_preserves_source_affine_and_axis_order() -> None:
    affine = ((-1.0, 0.0, 0.0, 0.0), (0.0, -1.0, 0.0, 8.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    plane = plane_from_nifti(affine, (8, 9, 11), 4, observation_id="p:flair:target", inplane_stride_vu=(2, 3))
    assert plane.shape_hw == (4, 3)
    assert plane.source_transform is not None
    assert plane.source_transform.to_canonical_dict()["index_order"] == ["u", "v", "slice"]
    assert plane.pixel_center_origin_ras_mm == (0.0, 8.0, 4.0)


def test_fractional_target_plane_interpolates_intensities_but_nearest_preserves_labels(tmp_path: Path) -> None:
    volume = np.zeros((3, 4, 5), dtype=np.float32)
    for index in range(volume.shape[2]):
        volume[:, :, index] = float(index)
    path = tmp_path / "fractional.nii.gz"
    nib.save(nib.Nifti1Image(volume, np.eye(4)), path)

    linear = extract_axial_plane_at_position(path, 1.5, interpolation="linear")
    nearest = extract_axial_plane_at_position(path, 1.5, interpolation="nearest")
    np.testing.assert_allclose(linear, 1.5)
    np.testing.assert_allclose(nearest, 2.0)


def test_preparation_hashes_are_deterministic_and_segmentation_is_evaluator_only(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_patient(source, shape=(8, 9, 11))
    schedule = {"t1": 0.1, "t1ce": 0.2, "t2": 0.3, "flair_context": 0.4, "flair_target": 0.7}
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = prepare_brats21_smoke(
        source_root=source,
        output_dir=first_dir,
        inplane_stride_vu=(2, 2),
        schedule_fractions=schedule,
    )
    second = prepare_brats21_smoke(
        source_root=source,
        output_dir=second_dir,
        inplane_stride_vu=(2, 2),
        schedule_fractions=schedule,
    )
    assert first["hashes"] == second["hashes"]
    bundle = load_prepared_bundle(first_dir)
    assert bundle.target_id not in bundle.assignment.context_ids
    assert all(entry.modality_id != "seg" for entry in bundle.manifest.entries)
    assert bundle.segmentation_payload_path.is_file()
    with pytest.raises(PermissionError):
        EpisodeLedger(
            bundle.manifest,
            bundle.assignment,
            first_dir,
            split_registry=PatientSplitRegistry.create((bundle.manifest,)),
        ).open_context(bundle.target_id)
