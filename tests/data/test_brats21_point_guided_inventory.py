"""Focused MAIN-009 tests for complete immediate-source accounting."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

import smagm.data.brats21_point_guided as point_guided_data
from smagm.data.brats21_point_guided import structural_inventory_point_guided_subjects
from smagm.training.point_guided import preflight


def _write_structural_subject(
    root: Path,
    subject_id: str,
    *,
    missing: str | None = None,
    duplicate: str | None = None,
) -> Path:
    directory = root / subject_id
    directory.mkdir()
    for modality in ("t1", "t2", "flair", "t1ce"):
        if modality == missing:
            continue
        (directory / f"{subject_id}_{modality}.nii.gz").write_bytes(b"synthetic")
        if modality == duplicate:
            (directory / f"{subject_id}_{modality}.nii").write_bytes(b"synthetic")
    return directory


def test_immediate_source_ledger_accounts_each_directory_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Creation order is intentionally not canonical source order.
    valid = "BraTS2021_00002"
    missing = "BraTS2021_00001"
    duplicate = "BraTS2021_00003"
    malformed = "malformed-source"
    _write_structural_subject(tmp_path, valid)
    _write_structural_subject(tmp_path, duplicate, duplicate="t1")
    (tmp_path / malformed).mkdir()
    _write_structural_subject(tmp_path, missing, missing="t2")
    (tmp_path / "README.txt").write_text("not an immediate subject directory", encoding="utf-8")

    monkeypatch.setattr(
        point_guided_data,
        "_nibabel",
        lambda: pytest.fail("immediate structural inventory must not load NIfTI payloads"),
    )
    inventory = structural_inventory_point_guided_subjects(tmp_path, require_target=True)

    discovered = tuple(sorted((valid, missing, duplicate, malformed)))
    assert inventory.discovered_subject_ids == discovered
    assert inventory.eligible_subject_ids == (valid,)

    exclusions = {item.subject_id: item for item in inventory.excluded_subjects}
    assert tuple(item.subject_id for item in inventory.excluded_subjects) == tuple(
        subject_id for subject_id in discovered if subject_id != valid
    )
    assert exclusions[missing].reason == "missing_file"
    assert exclusions[duplicate].reason == "duplicate_file"
    assert exclusions[malformed].reason == "OTHER_INVALID"
    assert exclusions[malformed].modality == "subject_directory"

    ledger = list(inventory.eligible_subject_ids) + [item.subject_id for item in inventory.excluded_subjects]
    assert len(ledger) == len(discovered)
    assert Counter(ledger) == Counter(discovered)
    assert inventory.to_dict()["discovered_subject_count"] == 4
    assert inventory.to_dict()["eligible_subject_count"] == 1
    assert inventory.to_dict()["excluded_subject_count"] == 3


def test_all_malformed_immediate_directories_return_excluded_ledger_without_nifti_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed = ("malformed-z", "also-invalid")
    for subject_id in malformed:
        (tmp_path / subject_id).mkdir()

    monkeypatch.setattr(
        point_guided_data,
        "_nibabel",
        lambda: pytest.fail("structural inventory must not load NIfTI payloads"),
    )
    inventory = structural_inventory_point_guided_subjects(tmp_path, require_target=True)

    discovered = tuple(sorted(malformed))
    assert inventory.discovered_subject_ids == discovered
    assert inventory.eligible_subject_ids == ()
    assert tuple(item.subject_id for item in inventory.excluded_subjects) == discovered
    assert tuple(item.reason for item in inventory.excluded_subjects) == ("OTHER_INVALID",) * len(discovered)
    assert tuple(item.modality for item in inventory.excluded_subjects) == ("subject_directory",) * len(discovered)
    record = inventory.to_dict()
    assert record["discovered_subject_count"] == len(discovered)
    assert record["eligible_subject_count"] == 0
    assert record["excluded_subject_count"] == len(discovered)


def test_preflight_fails_closed_after_all_subjects_are_structurally_excluded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for subject_id in ("malformed-z", "also-invalid"):
        (tmp_path / subject_id).mkdir()

    monkeypatch.setattr(
        point_guided_data,
        "_nibabel",
        lambda: pytest.fail("preflight must not load NIfTI payloads after structural exclusion"),
    )
    inventory = structural_inventory_point_guided_subjects(tmp_path, require_target=True)
    assert inventory.eligible_subject_ids == ()

    with pytest.raises(ValueError, match="no structurally eligible BraTS21 subjects"):
        preflight(data_root=tmp_path, require_segmentation=True)
