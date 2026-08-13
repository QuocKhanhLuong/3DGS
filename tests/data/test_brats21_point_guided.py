from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from smagm.data.brats21_point_guided import (
    BRATS21_POINT_GUIDED_LABELS,
    BraTS21PointGuidedDataset,
    BraTS21PointGuidedValidationError,
    collate_point_guided_samples,
    derive_input_brain_mask,
    deterministic_subject_split,
    discover_point_guided_subject,
    load_point_guided_subject,
    nifti_xyz_to_dhw,
    xyz_shape_to_dhw,
)


def _nibabel():
    return pytest.importorskip("nibabel")


def _write_subject(
    root: Path,
    *,
    shape_xyz: tuple[int, int, int] = (4, 3, 2),
    affine: np.ndarray | None = None,
    target_affine: np.ndarray | None = None,
    segmentation_values: np.ndarray | None = None,
    nonfinite_modality: str | None = None,
    subject_id: str = "BraTS2021_00000",
) -> Path:
    nib = _nibabel()
    subject = root / subject_id
    subject.mkdir(parents=True)
    matrix = np.diag((2.0, 3.0, 4.0, 1.0)) if affine is None else affine
    grid = np.arange(np.prod(shape_xyz), dtype=np.float32).reshape(shape_xyz)
    grid[:2, :2, :1] = 0.0
    for offset, modality in enumerate(("t1", "t2", "flair"), start=1):
        values = grid + float(offset)
        values[:2, :2, :1] = 0.0
        if nonfinite_modality == modality:
            values[0, 0, 0] = np.nan
        nib.save(nib.Nifti1Image(values, matrix), subject / f"{subject_id}_{modality}.nii.gz")
    target = grid + 20.0
    nib.save(
        nib.Nifti1Image(target, matrix if target_affine is None else target_affine),
        subject / f"{subject_id}_t1ce.nii.gz",
    )
    labels = np.zeros(shape_xyz, dtype=np.uint8) if segmentation_values is None else segmentation_values
    nib.save(nib.Nifti1Image(labels, matrix), subject / f"{subject_id}_seg.nii.gz")
    return subject


def test_xyz_to_dhw_mapping_is_explicit_and_contiguous() -> None:
    source = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
    converted = nifti_xyz_to_dhw(source)
    assert xyz_shape_to_dhw(source.shape) == (2, 3, 4)
    assert converted.shape == (2, 3, 4)
    np.testing.assert_array_equal(converted, source.transpose(2, 1, 0))
    assert converted.flags.c_contiguous


def test_input_mask_is_raw_observation_union_and_rejects_empty_input() -> None:
    observations = np.zeros((3, 2, 2, 2), dtype=np.float32)
    observations[1, 0, 1, 1] = 4.0
    mask = derive_input_brain_mask(observations)
    expected = np.zeros((2, 2, 2), dtype=bool)
    expected[0, 1, 1] = True
    np.testing.assert_array_equal(mask, expected)
    with pytest.raises(BraTS21PointGuidedValidationError, match="empty"):
        derive_input_brain_mask(np.zeros_like(observations))


def test_full_volume_load_keeps_observations_target_and_segmentation_separate(tmp_path: Path) -> None:
    subject_dir = _write_subject(tmp_path)
    record = discover_point_guided_subject(subject_dir)
    assert tuple(record.observation_paths) == ("t1", "t2", "flair")
    assert record.target_path is not None
    assert record.segmentation_path is not None

    first = load_point_guided_subject(subject_dir, require_segmentation=True)
    second = load_point_guided_subject(subject_dir, require_segmentation=True)
    assert first.observations.shape == (3, 2, 3, 4)
    assert first.target is not None and first.target.shape == (1, 2, 3, 4)
    assert first.segmentation is not None and first.segmentation.shape == (2, 3, 4)
    assert first.segmentation.dtype is torch.int64
    assert set(int(value) for value in first.segmentation.unique().tolist()).issubset(BRATS21_POINT_GUIDED_LABELS)
    assert first.target_t1ce is not None and first.target_t1ce.shape == (1, 2, 3, 4)
    assert first.brain_mask.shape == (1, 2, 3, 4)
    assert first.geometry.shape_xyz == (4, 3, 2)
    assert first.geometry.shape_dhw == (2, 3, 4)
    assert first.spacing_xyz_mm == (2.0, 3.0, 4.0)
    assert first.voxel_to_ras_mm[0][0] == 2.0
    assert torch.equal(first.observations, second.observations)
    assert first.to_metadata() == second.to_metadata()

    raw_observations = []
    nib = _nibabel()
    for modality in ("t1", "t2", "flair"):
        raw_observations.append(np.asarray(nib.load(str(record.observation_paths[modality])).dataobj))
    expected_mask = np.any(np.stack(raw_observations, axis=0) != 0.0, axis=0).transpose(2, 1, 0)
    np.testing.assert_array_equal(first.brain_mask.numpy(), expected_mask[None, ...])
    background = first.observations[:, ~first.brain_mask[0]]
    assert torch.equal(background, torch.zeros_like(background))
    assert all(metadata.metadata_hash == metadata.record_hash for metadata in first.normalization_metadata.values())


def test_target_and_segmentation_changes_do_not_change_input_observations_or_mask(tmp_path: Path) -> None:
    subject_dir = _write_subject(tmp_path)
    first = load_point_guided_subject(subject_dir, require_segmentation=True)
    nib = _nibabel()
    affine = np.diag((2.0, 3.0, 4.0, 1.0))
    shape = (4, 3, 2)
    nib.save(
        nib.Nifti1Image(np.full(shape, 999.0, dtype=np.float32), affine),
        str(subject_dir / f"{subject_dir.name}_t1ce.nii.gz"),
    )
    changed_segmentation = np.full(shape, 4, dtype=np.uint8)
    nib.save(
        nib.Nifti1Image(changed_segmentation, affine),
        str(subject_dir / f"{subject_dir.name}_seg.nii.gz"),
    )
    second = load_point_guided_subject(subject_dir, require_segmentation=True)
    torch.testing.assert_close(first.observations, second.observations)
    assert torch.equal(first.brain_mask, second.brain_mask)


def test_optional_target_and_segmentation_loading_can_be_disabled(tmp_path: Path) -> None:
    subject_dir = _write_subject(tmp_path)
    sample = load_point_guided_subject(
        subject_dir,
        require_target=False,
        load_target=False,
        load_segmentation=False,
    )
    assert sample.target is None
    assert sample.segmentation is None
    assert set(sample.source_paths) == {"t1", "t2", "flair"}


def test_dataset_and_collator_expose_model_batch_contract(tmp_path: Path) -> None:
    subject_dir = _write_subject(tmp_path)
    dataset = BraTS21PointGuidedDataset(
        tmp_path,
        [subject_dir.name],
        {"brain_mask_threshold": 0.0, "epsilon": 1e-5},
    )
    sample = dataset[0]
    batch = collate_point_guided_samples([sample])
    assert len(dataset) == 1
    assert batch.observations.shape == (1, 3, 2, 3, 4)
    assert batch.target_t1ce.shape == (1, 1, 2, 3, 4)
    assert batch.segmentation is not None and batch.segmentation.shape == (1, 2, 3, 4)
    assert batch.brain_mask.shape == (1, 1, 2, 3, 4)
    assert batch.spacing_xyz_mm.shape == (1, 3)
    assert batch.voxel_to_ras_mm.shape == (1, 4, 4)
    assert batch.subject_ids == (subject_dir.name,)
    assert batch.normalization_metadata[0]["t1"].metadata_hash


def test_dataset_allows_missing_optional_segmentation(tmp_path: Path) -> None:
    subject_dir = _write_subject(tmp_path)
    (subject_dir / f"{subject_dir.name}_seg.nii.gz").unlink()
    dataset = BraTS21PointGuidedDataset(tmp_path, [subject_dir.name])
    sample = dataset[0]
    assert sample.segmentation is None
    assert collate_point_guided_samples([sample]).segmentation is None


def test_collator_rejects_mixed_geometry(tmp_path: Path) -> None:
    first_dir = _write_subject(tmp_path, subject_id="BraTS2021_00000")
    shifted = np.diag((2.0, 3.0, 4.0, 1.0))
    shifted[0, 3] = 5.0
    _write_subject(tmp_path, subject_id="BraTS2021_00001", affine=shifted)
    dataset = BraTS21PointGuidedDataset(tmp_path, [first_dir.name, "BraTS2021_00001"])
    with pytest.raises(BraTS21PointGuidedValidationError, match="mixed voxel_to_ras_mm"):
        collate_point_guided_samples([dataset[0], dataset[1]])

def test_affine_mismatch_is_rejected(tmp_path: Path) -> None:
    shifted = np.diag((2.0, 3.0, 4.0, 1.0))
    shifted[0, 3] = 1.0
    subject_dir = _write_subject(tmp_path, target_affine=shifted)
    with pytest.raises(BraTS21PointGuidedValidationError, match="affine mismatch"):
        load_point_guided_subject(subject_dir)


def test_nonfinite_observation_is_rejected_before_normalization(tmp_path: Path) -> None:
    subject_dir = _write_subject(tmp_path, nonfinite_modality="t2")
    with pytest.raises(BraTS21PointGuidedValidationError, match="non-finite"):
        load_point_guided_subject(subject_dir)


def test_invalid_segmentation_label_is_rejected(tmp_path: Path) -> None:
    labels = np.zeros((4, 3, 2), dtype=np.uint8)
    labels[0, 0, 0] = 3
    subject_dir = _write_subject(tmp_path, segmentation_values=labels)
    with pytest.raises(BraTS21PointGuidedValidationError, match="unexpected labels"):
        load_point_guided_subject(subject_dir, require_segmentation=True)


def test_subject_split_is_deterministic_hashed_and_capped() -> None:
    subjects = tuple(f"BraTS2021_{index:05d}" for index in range(12))
    first = deterministic_subject_split(
        subjects,
        seed=17,
        max_subjects={"train": 3, "val": 1, "test": 2},
    )
    second = deterministic_subject_split(
        reversed(subjects),
        seed=17,
        max_subjects={"train": 3, "val": 1, "test": 2},
    )
    assert first == second
    assert len(first.train_subject_ids) <= 3
    assert len(first.val_subject_ids) <= 1
    assert len(first.test_subject_ids) <= 2
    assert len(first.excluded_subject_ids) == 7
    assert len(first.split_hash) == 64
    changed = deterministic_subject_split(subjects, seed=18, max_subjects=6)
    assert changed.split_hash != first.split_hash


def test_malformed_subject_id_is_rejected(tmp_path: Path) -> None:
    malformed = tmp_path / "subject-1"
    malformed.mkdir()
    with pytest.raises(BraTS21PointGuidedValidationError, match="malformed"):
        discover_point_guided_subject(malformed)
