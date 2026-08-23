from __future__ import annotations

from pathlib import Path
import pickle

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
    discover_point_guided_subject_ids,
    inventory_point_guided_subjects,
    load_point_guided_subject,
    nifti_xyz_to_dhw,
    PointGuidedNormalizationConfig,
    structural_inventory_point_guided_subjects,
    xyz_shape_to_dhw,
)
from smagm.training.point_guided import preflight


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
    round_trip = pickle.loads(pickle.dumps(first))
    assert round_trip.to_metadata() == first.to_metadata()
    batch = collate_point_guided_samples([first])
    assert pickle.loads(pickle.dumps(batch)).subject_ids == (first.subject_id,)


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


def test_masked_robust_01_clips_and_maps_masked_values_to_unit_range(tmp_path: Path) -> None:
    subject_dir = _write_subject(tmp_path)
    config = PointGuidedNormalizationConfig(
        normalization_policy="masked_robust_01",
        lower_percentile=20.0,
        upper_percentile=80.0,
    )
    sample = BraTS21PointGuidedDataset(tmp_path, [subject_dir.name], config)[0]

    nib = _nibabel()
    raw_t1_xyz = np.asarray(nib.load(str(subject_dir / f"{subject_dir.name}_t1.nii.gz")).dataobj)
    raw_t1_dhw = nifti_xyz_to_dhw(raw_t1_xyz)
    mask = sample.brain_mask[0].numpy()
    selected = raw_t1_dhw[mask]
    clip_lower = float(np.percentile(selected, 20.0))
    clip_upper = float(np.percentile(selected, 80.0))
    expected = np.zeros(raw_t1_dhw.shape, dtype=np.float32)
    expected[mask] = np.clip(
        (selected - clip_lower) / (clip_upper - clip_lower),
        0.0,
        1.0,
    ).astype(np.float32)

    torch.testing.assert_close(sample.observations[0], torch.from_numpy(expected))
    assert torch.equal(sample.observations[:, ~sample.brain_mask[0]], torch.zeros_like(sample.observations[:, ~sample.brain_mask[0]]))
    assert float(sample.observations[:, sample.brain_mask[0]].min()) >= 0.0
    assert float(sample.observations[:, sample.brain_mask[0]].max()) <= 1.0

    metadata = sample.normalization_metadata["t1"]
    assert metadata.normalization_policy == "masked_robust_01"
    assert metadata.policy == metadata.normalization_policy
    assert metadata.percentiles == (20.0, 80.0)
    assert metadata.clip_bounds == pytest.approx((clip_lower, clip_upper))
    assert metadata.output_range == (0.0, 1.0)
    assert metadata.voxel_count == int(mask.sum())
    assert metadata.mask_source == "raw_observation_nonzero_union"
    assert len(metadata.metadata_hash) == 64
    assert metadata.metadata_hash == metadata.record_hash
    serialized = sample.to_metadata()["normalization_metadata"]["t1"]
    assert serialized["normalization_policy"] == "masked_robust_01"
    assert serialized["policy"] == "masked_robust_01"
    assert serialized["percentiles"] == (20.0, 80.0)
    assert serialized["clip_bounds"] == pytest.approx((clip_lower, clip_upper))
    assert serialized["clip_low"] == pytest.approx(clip_lower)
    assert serialized["clip_high"] == pytest.approx(clip_upper)
    assert serialized["output_min"] == 0.0
    assert serialized["output_max"] == 1.0
    assert serialized["output_range"] == (0.0, 1.0)
    assert serialized["voxel_count"] == int(mask.sum())
    assert serialized["mask_source"] == "raw_observation_nonzero_union"
    assert serialized["metadata_hash"] == metadata.metadata_hash


def test_default_masked_zscore_is_compatible_with_explicit_policy(tmp_path: Path) -> None:
    subject_dir = _write_subject(tmp_path)
    implicit = load_point_guided_subject(subject_dir)
    explicit = load_point_guided_subject(subject_dir, normalization_policy="masked_zscore")

    torch.testing.assert_close(implicit.observations, explicit.observations)
    torch.testing.assert_close(implicit.target, explicit.target)
    assert implicit.to_metadata() == explicit.to_metadata()
    metadata = explicit.normalization_metadata["t1"]
    assert metadata.normalization_policy == "masked_zscore"
    assert metadata.clip_bounds is None
    assert metadata.output_range is None


def test_normalization_config_rejects_invalid_or_empty_percentile_range() -> None:
    with pytest.raises(BraTS21PointGuidedValidationError, match="percentile range"):
        PointGuidedNormalizationConfig(
            normalization_policy="masked_robust_01",
            lower_percentile=50.0,
            upper_percentile=50.0,
        )
    with pytest.raises(BraTS21PointGuidedValidationError, match=r"\[0, 100\]"):
        PointGuidedNormalizationConfig(
            normalization_policy="masked_robust_01",
            lower_percentile=-1.0,
        )


def test_masked_robust_01_rejects_empty_data_percentile_range(tmp_path: Path) -> None:
    subject_dir = _write_subject(tmp_path)
    nib = _nibabel()
    affine = np.diag((2.0, 3.0, 4.0, 1.0))
    constant = np.full((4, 3, 2), 5.0, dtype=np.float32)
    for modality in ("t1", "t2", "flair"):
        nib.save(
            nib.Nifti1Image(constant, affine),
            str(subject_dir / f"{subject_dir.name}_{modality}.nii.gz"),
        )

    with pytest.raises(BraTS21PointGuidedValidationError, match="percentile range"):
        load_point_guided_subject(subject_dir, normalization_policy="masked_robust_01")


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


def test_partial_cohort_inventory_keeps_only_complete_payload_valid_subjects(tmp_path: Path) -> None:
    complete = _write_subject(tmp_path, subject_id="BraTS2021_00000")
    missing_flair = _write_subject(tmp_path, subject_id="BraTS2021_00001")
    (missing_flair / f"{missing_flair.name}_flair.nii.gz").unlink()
    shifted = np.diag((2.0, 3.0, 4.0, 1.0))
    shifted[0, 3] = 1.0
    _write_subject(tmp_path, subject_id="BraTS2021_00002", target_affine=shifted)

    inventory = inventory_point_guided_subjects(tmp_path, require_segmentation=True)

    assert [subject.subject_id for subject in inventory.complete_subjects] == [complete.name]
    exclusions = {subject.subject_id: subject for subject in inventory.excluded_subjects}
    assert exclusions[missing_flair.name].classification == "MISSING_FLAIR"
    assert exclusions["BraTS2021_00002"].classification == "INVALID_GEOMETRY"
    assert inventory.to_dict()["complete_subject_ids"] == [complete.name]


def test_scoped_inventory_only_validates_selected_subject_payloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = _write_subject(tmp_path, subject_id="BraTS2021_00000")
    second = _write_subject(tmp_path, subject_id="BraTS2021_00001")
    assert discover_point_guided_subject_ids(tmp_path) == (first.name, second.name)

    import smagm.data.brats21_point_guided as point_guided_data

    original = point_guided_data.load_point_guided_subject
    loaded: list[str] = []

    def tracked(subject, **kwargs):
        loaded.append(subject.subject_id)
        return original(subject, **kwargs)

    monkeypatch.setattr(point_guided_data, "load_point_guided_subject", tracked)
    inventory = inventory_point_guided_subjects(
        tmp_path,
        require_segmentation=True,
        subject_ids=(first.name,),
    )

    assert [subject.subject_id for subject in inventory.complete_subjects] == [first.name]
    assert loaded == [first.name]


def test_preflight_uses_partial_cohort_inventory_without_admitting_exclusions(tmp_path: Path) -> None:
    _write_subject(tmp_path, subject_id="BraTS2021_00000")
    incomplete = _write_subject(tmp_path, subject_id="BraTS2021_00001")
    (incomplete / f"{incomplete.name}_seg.nii.gz").unlink()

    result = preflight(data_root=tmp_path, require_segmentation=True)

    assert result["subject_count"] == 1
    assert result["structural_inventory"]["eligible_subject_ids"] == ["BraTS2021_00000"]
    exclusion = result["structural_inventory"]["excluded_subjects"][0]
    assert exclusion["modality"] == "seg"
    assert exclusion["reason"] == "missing_file"
    assert result["test_payload_validation"] == "not_performed"


def test_structural_inventory_rejects_required_missing_and_empty_files_without_nifti_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete = _write_subject(tmp_path, subject_id="BraTS2021_00000")
    cases = (
        ("BraTS2021_00001", "t1", "missing_file"),
        ("BraTS2021_00002", "t2", "empty_file"),
        ("BraTS2021_00003", "flair", "empty_file"),
        ("BraTS2021_00004", "t1ce", "empty_file"),
        ("BraTS2021_00005", "seg", "empty_file"),
    )
    for subject_id, modality, reason in cases:
        subject = _write_subject(tmp_path, subject_id=subject_id)
        path = subject / f"{subject_id}_{modality}.nii.gz"
        if reason == "missing_file":
            path.unlink()
        else:
            path.write_bytes(b"")

    import smagm.data.brats21_point_guided as point_guided_data

    def no_nifti_reader() -> object:
        raise AssertionError("structural eligibility must not request nibabel")

    monkeypatch.setattr(point_guided_data, "_nibabel", no_nifti_reader)
    inventory = structural_inventory_point_guided_subjects(tmp_path, require_segmentation=True)

    assert inventory.eligible_subject_ids == (complete.name,)
    exclusions = {(item.subject_id, item.modality): item.reason for item in inventory.excluded_subjects}
    assert exclusions == {(subject_id, modality): reason for subject_id, modality, reason in cases}
    record = inventory.to_dict()
    assert record["discovered_subject_count"] == 6
    assert record["eligible_subject_count"] == 1
    assert record["excluded_subject_count"] == 5
    assert len(record["inventory_hash"]) == 64


def test_structural_eligible_ids_are_the_only_split_inputs(tmp_path: Path) -> None:
    subjects = [_write_subject(tmp_path, subject_id=f"BraTS2021_{index:05d}") for index in range(10)]
    blocked = subjects[4]
    (blocked / f"{blocked.name}_t2.nii.gz").write_bytes(b"")

    inventory = structural_inventory_point_guided_subjects(tmp_path, require_segmentation=True)
    split = deterministic_subject_split(inventory.eligible_subject_ids, seed=20260813)

    assigned = set(split.train_subject_ids) | set(split.val_subject_ids) | set(split.test_subject_ids)
    assert blocked.name not in assigned
    assert assigned == set(inventory.eligible_subject_ids)


def test_preflight_never_decodes_test_payloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for index in range(10):
        _write_subject(tmp_path, subject_id=f"BraTS2021_{index:05d}")

    import smagm.data.brats21_point_guided as point_guided_data

    original = point_guided_data.load_point_guided_subject
    loaded: list[str] = []

    def tracked(subject, **kwargs):
        loaded.append(subject.subject_id)
        return original(subject, **kwargs)

    monkeypatch.setattr(point_guided_data, "load_point_guided_subject", tracked)
    result = preflight(data_root=tmp_path, require_segmentation=True, split_seed=20260813)

    test_ids = set(result["split"]["test"])
    active_ids = set(result["active_payload_inventory"]["requested_subject_ids"])
    assert test_ids
    assert not test_ids.intersection(loaded)
    assert set(loaded) == active_ids
    assert result["test_payload_validation"] == "not_performed"


def test_preflight_fails_closed_when_an_active_payload_is_invalid(tmp_path: Path) -> None:
    for index in range(10):
        _write_subject(tmp_path, subject_id=f"BraTS2021_{index:05d}")
    eligible = structural_inventory_point_guided_subjects(tmp_path, require_segmentation=True)
    split = deterministic_subject_split(eligible.eligible_subject_ids, seed=20260813)
    invalid_subject_id = split.train_subject_ids[0]
    invalid_path = tmp_path / invalid_subject_id / f"{invalid_subject_id}_t1.nii.gz"
    invalid_path.write_bytes(b"invalid-but-nonempty")

    with pytest.raises(ValueError, match=invalid_subject_id):
        preflight(data_root=tmp_path, require_segmentation=True, split_seed=20260813)


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
