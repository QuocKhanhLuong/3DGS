"""MAIN-002 focused provenance, canonical split digest, and checkpoint binding tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from smagm.cli.point_guided_eval import _load_split, resolve_split_file
from smagm.data.brats21_point_guided import (
    BraTS21PointGuidedSplit,
    BraTS21PointGuidedValidationError,
    SPLIT_HASH_PATTERN,
    SPLIT_VERSION,
    compute_canonical_split_digest,
    deterministic_subject_split,
    load_point_guided_split,
)
from smagm.features.point_guided.baseline_checkpoint import (
    save_clean_inference_checkpoint,
)
from smagm.features.point_guided.baseline_inference import (
    baseline_checkpoint_metadata,
    load_validated_baseline_checkpoint,
)
from smagm.features.point_guided.config import PointGuidedConfig
from smagm.features.point_guided.model import PointGuidedMRIModel
from smagm.features.point_guided.trajectory_cost import TrajectoryConfig
from smagm.training.point_guided import _resolve_split


def _model() -> PointGuidedMRIModel:
    return PointGuidedMRIModel(
        PointGuidedConfig(
            num_semantic_classes=3,
            num_points=3,
            point_candidate_multiplier=3,
            offset_hidden_channels=12,
        ),
        trajectory_config=TrajectoryConfig(
            lambda_travel=0.05,
            lambda_overlap=0.20,
            lambda_step=0.05,
            k_max=2,
            selection_temperature=0.7,
            write_scale=0.1,
        ),
    )


def test_canonical_split_digest_reproducibility_and_order_invariance() -> None:
    subjects = tuple(f"BraTS2021_{index:05d}" for index in range(10))
    first = deterministic_subject_split(subjects, seed=42)
    second = deterministic_subject_split(reversed(subjects), seed=42)
    shuffled = tuple([subjects[i] for i in (3, 7, 0, 9, 2, 5, 1, 8, 4, 6)])
    third = deterministic_subject_split(shuffled, seed=42)

    assert first.split_hash == second.split_hash == third.split_hash
    assert SPLIT_HASH_PATTERN.fullmatch(first.split_hash) is not None
    assert first.all_subject_ids == tuple(sorted(subjects))
    assert first.excluded_subject_ids == tuple(sorted(first.excluded_subject_ids))

    # Changing any defining field produces a distinct digest
    changed_seed = deterministic_subject_split(subjects, seed=43)
    assert changed_seed.split_hash != first.split_hash

    changed_fractions = deterministic_subject_split(subjects, seed=42, split_fractions=(0.7, 0.2, 0.1))
    assert changed_fractions.split_hash != first.split_hash

    changed_caps = deterministic_subject_split(subjects, seed=42, max_subjects={"train": 2, "val": 1, "test": 1})
    assert changed_caps.split_hash != first.split_hash


def test_compute_canonical_split_digest_field_validation() -> None:
    subjects = tuple(f"BraTS2021_{index:05d}" for index in range(4))
    assignments = {
        "BraTS2021_00000": "train",
        "BraTS2021_00001": "train",
        "BraTS2021_00002": "val",
        "BraTS2021_00003": "test",
    }
    digest = compute_canonical_split_digest(
        all_subject_ids=subjects,
        assignments=assignments,
        caps={"train": None, "val": None, "test": None},
        fractions=(0.5, 0.25, 0.25),
        seed=1,
    )
    assert isinstance(digest, str) and len(digest) == 64
    assert SPLIT_HASH_PATTERN.fullmatch(digest) is not None

    # Invalid version
    with pytest.raises(BraTS21PointGuidedValidationError, match="unsupported split version"):
        compute_canonical_split_digest(
            all_subject_ids=subjects,
            assignments=assignments,
            version="unsupported_v99",
        )

    # Boolean seed rejected
    with pytest.raises(BraTS21PointGuidedValidationError, match="split seed must be an integer"):
        compute_canonical_split_digest(
            all_subject_ids=subjects,
            assignments=assignments,
            seed=True,  # type: ignore[arg-type]
        )

    # Incomplete assignment coverage
    with pytest.raises(BraTS21PointGuidedValidationError, match="assignments must cover"):
        compute_canonical_split_digest(
            all_subject_ids=subjects,
            assignments={"BraTS2021_00000": "train"},
        )

    # Invalid assignment target
    with pytest.raises(BraTS21PointGuidedValidationError, match="invalid assignment"):
        compute_canonical_split_digest(
            all_subject_ids=subjects,
            assignments={**assignments, "BraTS2021_00000": "invalid_group"},
        )


def test_load_point_guided_split_recomputes_hash_and_rejects_fake_hash(tmp_path: Path) -> None:
    subjects = tuple(f"BraTS2021_{index:05d}" for index in range(6))
    split = deterministic_subject_split(subjects, seed=10)
    split_dict = split.to_dict()

    # Valid split load
    valid_file = tmp_path / "valid_split.json"
    valid_file.write_text(json.dumps(split_dict), encoding="utf-8")
    loaded = load_point_guided_split(valid_file, subjects=subjects)
    assert loaded.split_hash == split.split_hash

    # Fake 64-char string 'a'*64 is rejected even for valid partition
    fake_hash_dict = dict(split_dict)
    fake_hash_dict["split_hash"] = "a" * 64
    fake_file = tmp_path / "fake_hash_split.json"
    fake_file.write_text(json.dumps(fake_hash_dict), encoding="utf-8")
    with pytest.raises(BraTS21PointGuidedValidationError, match="split_hash mismatch"):
        load_point_guided_split(fake_file, subjects=subjects)

    # Fake 64-char string '0'*64 is rejected
    fake_zero_dict = dict(split_dict)
    fake_zero_dict["split_hash"] = "0" * 64
    fake_zero_file = tmp_path / "fake_zero_split.json"
    fake_zero_file.write_text(json.dumps(fake_zero_dict), encoding="utf-8")
    with pytest.raises(BraTS21PointGuidedValidationError, match="split_hash mismatch"):
        load_point_guided_split(fake_zero_file, subjects=subjects)


def test_load_point_guided_split_rejects_malformed_and_non_hex_hash(tmp_path: Path) -> None:
    subjects = tuple(f"BraTS2021_{index:05d}" for index in range(6))
    split = deterministic_subject_split(subjects, seed=10)
    split_dict = split.to_dict()

    cases = [
        ("non_hex_g", "g" * 64),
        ("non_hex_z", "z" * 64),
        ("uppercase_hex", ("0123456789ABCDEF" * 4)),
        ("short_63", "a" * 63),
        ("long_65", "a" * 65),
        ("non_string_int", 12345678901234567890),
        ("none_value", None),
    ]
    for label, invalid_hash in cases:
        bad_dict = dict(split_dict)
        bad_dict["split_hash"] = invalid_hash
        bad_file = tmp_path / f"bad_{label}.json"
        bad_file.write_text(json.dumps(bad_dict), encoding="utf-8")
        with pytest.raises(BraTS21PointGuidedValidationError, match="lowercase hexadecimal"):
            load_point_guided_split(bad_file, subjects=subjects)


def test_load_point_guided_split_rejects_tampered_membership(tmp_path: Path) -> None:
    subjects = tuple(f"BraTS2021_{index:05d}" for index in range(6))
    split = deterministic_subject_split(subjects, seed=10)
    split_dict = split.to_dict()

    # Case 1: Swap subjects between train and val keeping split_hash unchanged
    tampered_swap = dict(split_dict)
    train_list = list(tampered_swap["train_subject_ids"])
    val_list = list(tampered_swap["val_subject_ids"])
    # swap first train and first val
    train_list[0], val_list[0] = val_list[0], train_list[0]
    tampered_swap["train_subject_ids"] = train_list
    tampered_swap["val_subject_ids"] = val_list
    swap_file = tmp_path / "tampered_swap.json"
    swap_file.write_text(json.dumps(tampered_swap), encoding="utf-8")
    with pytest.raises(BraTS21PointGuidedValidationError, match="split_hash mismatch"):
        load_point_guided_split(swap_file, subjects=subjects)

    # Case 2: Move a subject from train to test
    tampered_move = dict(split_dict)
    train_list = list(tampered_move["train_subject_ids"])
    test_list = list(tampered_move["test_subject_ids"])
    moved = train_list.pop()
    test_list.append(moved)
    tampered_move["train_subject_ids"] = train_list
    tampered_move["test_subject_ids"] = test_list
    move_file = tmp_path / "tampered_move.json"
    move_file.write_text(json.dumps(tampered_move), encoding="utf-8")
    with pytest.raises(BraTS21PointGuidedValidationError, match="split_hash mismatch"):
        load_point_guided_split(move_file, subjects=subjects)

    # Case 3: Tamper seed field
    tampered_seed = dict(split_dict)
    tampered_seed["seed"] = 999
    seed_file = tmp_path / "tampered_seed.json"
    seed_file.write_text(json.dumps(tampered_seed), encoding="utf-8")
    with pytest.raises(BraTS21PointGuidedValidationError, match="split_hash mismatch"):
        load_point_guided_split(seed_file, subjects=subjects)


def test_bra_ts21_point_guided_split_dataclass_fails_closed_on_fake_or_mismatched_hash() -> None:
    subjects = tuple(f"BraTS2021_{index:05d}" for index in range(4))
    valid_split = deterministic_subject_split(subjects, seed=0)

    # Arbitrary 64-char hash in direct construction is rejected
    with pytest.raises(BraTS21PointGuidedValidationError, match="split_hash mismatch"):
        BraTS21PointGuidedSplit(
            all_subject_ids=valid_split.all_subject_ids,
            train_subject_ids=valid_split.train_subject_ids,
            val_subject_ids=valid_split.val_subject_ids,
            test_subject_ids=valid_split.test_subject_ids,
            split_fractions=valid_split.split_fractions,
            seed=valid_split.seed,
            max_subjects=valid_split.max_subjects,
            excluded_subject_ids=valid_split.excluded_subject_ids,
            split_hash="a" * 64,
        )

    # Non-hex hash in direct construction is rejected
    with pytest.raises(BraTS21PointGuidedValidationError, match="lowercase hexadecimal"):
        BraTS21PointGuidedSplit(
            all_subject_ids=valid_split.all_subject_ids,
            train_subject_ids=valid_split.train_subject_ids,
            val_subject_ids=valid_split.val_subject_ids,
            test_subject_ids=valid_split.test_subject_ids,
            split_fractions=valid_split.split_fractions,
            seed=valid_split.seed,
            max_subjects=valid_split.max_subjects,
            excluded_subject_ids=valid_split.excluded_subject_ids,
            split_hash="g" * 64,
        )


def test_checkpoint_metadata_binds_split_hash_and_rejects_malformed() -> None:
    model = _model()
    valid_hash = "0123456789abcdef" * 4
    metadata = baseline_checkpoint_metadata(model, split_hash=valid_hash)
    assert metadata["split_hash"] == valid_hash
    assert metadata["schema"] == "point-guided-gate-f-baseline-v1"

    # Malformed split_hash rejected
    with pytest.raises(ValueError, match="split_hash must be a 64-character lowercase hexadecimal"):
        baseline_checkpoint_metadata(model, split_hash="a" * 63)
    with pytest.raises(ValueError, match="split_hash must be a 64-character lowercase hexadecimal"):
        baseline_checkpoint_metadata(model, split_hash="g" * 64)
    with pytest.raises(ValueError, match="split_hash must be a 64-character lowercase hexadecimal"):
        baseline_checkpoint_metadata(model, split_hash=("ABCDEF0123456789" * 4))


def test_load_validated_baseline_checkpoint_checks_split_hash_compatibility(tmp_path: Path) -> None:
    model = _model()
    split_hash_1 = "1111111111111111" * 4
    split_hash_2 = "2222222222222222" * 4

    ckpt_path = save_clean_inference_checkpoint(tmp_path / "model.pt", model, split_hash=split_hash_1)
    clone = _model()

    # Compatible split hash succeeds
    loaded_meta = load_validated_baseline_checkpoint(clone, ckpt_path, expected_split_hash=split_hash_1)
    assert loaded_meta["split_hash"] == split_hash_1

    # Incompatible split hash is rejected
    with pytest.raises(ValueError, match="split hash mismatch"):
        load_validated_baseline_checkpoint(clone, ckpt_path, expected_split_hash=split_hash_2)


def test_historical_checkpoint_without_split_hash_is_rejected_fail_closed(tmp_path: Path) -> None:
    model = _model()
    # Historical checkpoint format where metadata lacked split_hash
    historical_metadata = {
        "schema": "point-guided-gate-f-baseline-v1",
        "model_config": json.loads(json.dumps(model.config.__dict__, default=str)),
        "trajectory_config": json.loads(json.dumps(model.trajectory.config.__dict__, default=str)),
        "decoder_architecture": "96->64->32->1",
        "gate_e_architecture": "target-after-inference objective",
        # missing split_hash
    }
    payload = {
        "metadata": historical_metadata,
        "state_dict": model.state_dict(),
    }
    historical_ckpt = tmp_path / "historical.pt"
    torch.save(payload, historical_ckpt)

    clone = _model()
    with pytest.raises(ValueError, match="split_hash"):
        load_validated_baseline_checkpoint(clone, historical_ckpt)


def test_eval_load_split_and_resolve_split_integration(tmp_path: Path) -> None:
    subjects = tuple(f"BraTS2021_{index:05d}" for index in range(6))
    split = deterministic_subject_split(subjects, seed=2026)
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps(split.to_dict()), encoding="utf-8")

    # _load_split verification
    groups, split_hash = _load_split(split_file, subjects)
    assert split_hash == split.split_hash
    assert groups["train"] == split.train_subject_ids

    # _resolve_split verification
    resolved, resolved_hash = _resolve_split(
        subjects,
        seed=2026,
        split_file=split_file,
        max_train_subjects=None,
        max_val_subjects=None,
        max_test_subjects=None,
    )
    assert resolved_hash == split.split_hash
    assert resolved["train"] == split.train_subject_ids

    # Tampered split.json rejection through _load_split and _resolve_split
    tampered_dict = split.to_dict()
    tampered_dict["split_hash"] = "b" * 64
    tampered_file = tmp_path / "tampered.json"
    tampered_file.write_text(json.dumps(tampered_dict), encoding="utf-8")

    with pytest.raises(ValueError, match="split_hash mismatch"):
        _load_split(tampered_file, subjects)

    with pytest.raises(ValueError, match="split_hash mismatch"):
        _resolve_split(
            subjects,
            seed=2026,
            split_file=tampered_file,
            max_train_subjects=None,
            max_val_subjects=None,
            max_test_subjects=None,
        )
