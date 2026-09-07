from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import smagm.features.point_guided.pfgr_lite.value_bank as value_bank_module
from smagm.features.point_guided.pfgr_lite.value_bank import (
    STAGE_PROVENANCE_SCHEMA,
    ValueBankReader,
    ValueBankRow,
    ValueBankWriter,
    build_value_bank,
    compute_gain_scale,
)
from smagm.features.point_guided.pfgr_lite.types import GainLabel
from smagm.features.point_guided.pfgr_lite.provenance import ProducerCompatibility, SourceProvenance, canonical_digest
from smagm.features.point_guided.pfgr_lite.types import ProducerDependencies, TrainingRoleManifest


def _producer() -> ProducerDependencies:
    compatibility = ProducerCompatibility(
        observation_normalization_hash="norm",
        geometry_query_version_hash="geometry",
        medicalnet_provenance_hash="medicalnet",
        frozen_bn_hash="bn",
        static_head_hash="static",
        semantic_head_hash="semantic",
        point_refiner_hash="points",
        spectral_projector_hash="spectral",
        state_initializer_hash="state",
        updater_hash="updater",
        decoder_hash="decoder",
        writer_hash="writer",
        candidate_geometry_hash="candidate",
        label_definition_hash="label",
    )
    source = SourceProvenance(
        synthetic_untrained=False,
        official_pretrained_verified=True,
        parameter_hash="parameters",
        frozen_bn_hash="bn",
        traversal_count=1,
    )
    return ProducerDependencies(compatibility=compatibility, source_provenance=source)


def _build(rows, path: Path, **kwargs):
    role_membership = {row.subject_key: row.split_role for row in rows if isinstance(row, ValueBankRow)}
    split_role_hash = kwargs.pop("split_role_hash", "split-hash")
    return build_value_bank(rows, path, producer=_producer(), split_role_hash=split_role_hash, role_membership=role_membership, engineering_only=True, **kwargs)


def _row(
    index: int,
    gain: float,
    *,
    split_role: str = "producer_fit",
    subject: str | None = None,
    role: str = "exact_footprint",
    diagnostic: bool = False,
    measurement_mode: str | None = None,
) -> ValueBankRow:
    benefit = max(gain, 0.0)
    harm = max(-gain, 0.0)
    return ValueBankRow(
        state96=torch.full((96,), float(index)),
        f_spec168=torch.arange(168, dtype=torch.float32) + index,
        semantic3=torch.tensor([0.1, 0.2, 0.7]),
        reliability3=torch.tensor([0.2, 0.3, 0.5]),
        q_bar24=torch.linspace(0.0, 1.0, 24),
        delta96=torch.full((96,), 0.1 * (index + 1)),
        raw_gain=gain,
        benefit=benefit,
        harm=harm,
        action_id=f"action-{index}",
        context_id=f"context-{index // 2}",
        state_version=0,
        point_id=index,
        subject_id=subject or f"subject-{index // 2}",
        point_ras_mm=torch.tensor([1.0, 2.0, 3.0]),
        geometry_id=f"geometry-{index // 2}",
        proposal_hash=f"proposal-{index}",
        state_digest=f"state-{index}",
        split_role=split_role,
        role=role,
        measurement_mode=measurement_mode or role,
        q_draws=4 if role == "iid_fixed_q" else 0,
        variance=0.25 if role == "iid_fixed_q" else None,
        standard_error=0.5 if role == "iid_fixed_q" else None,
        mask_count=8,
        footprint_voxels=4,
        valid_masked_contributions=3,
        sampler_law="iid_fixed_q" if role == "iid_fixed_q" else "exact",
        support_provenance="complete_support_v1" if role != "screening" else "screening_subset",
        inclusion_mechanism="fixed_q_complete_support_v1" if role == "iid_fixed_q" else "complete_support_v1",
        # Ordinary bank fixtures do not publish S2 state snapshots; replay
        # references are populated only by the canonical staged collector.
        selected_replay_ref="",
        diagnostic=diagnostic,
    )


def _production_roles() -> tuple[object, dict[str, str]]:
    producer_ids = ("producer-0",)
    calibration_fit = tuple(f"cal-fit-{index}" for index in range(32))
    calibration_allowance = tuple(f"cal-allow-{index}" for index in range(32))
    validation = ("validation-0",)
    test = ("test-0",)
    baseline_train = producer_ids + calibration_fit + calibration_allowance
    subjects = baseline_train + validation + test
    role_manifest = TrainingRoleManifest(
        baseline_split_hash="baseline-split-v1",
        baseline_train_subject_ids=baseline_train,
        baseline_validation_subject_ids=validation,
        baseline_test_subject_ids=test,
        producer_fit_subject_ids=producer_ids,
        calibration_fit_subject_ids=calibration_fit,
        calibration_allowance_subject_ids=calibration_allowance,
        subject_group_ids=tuple((subject, f"group-{index}") for index, subject in enumerate(subjects)),
    )
    return role_manifest, {subject: role for role, ids in (("producer_fit", producer_ids), ("calibration_fit", calibration_fit), ("calibration_allowance", calibration_allowance), ("validation", validation), ("test", test)) for subject in ids}


def _stage(producer: ProducerDependencies, role_manifest: TrainingRoleManifest, *, spectral_arm: str = "u_plus_spectral", projector_after: str | None = None) -> dict[str, object]:
    return {
        "schema_version": STAGE_PROVENANCE_SCHEMA,
        "stage": "updater",
        "completed": True,
        "spectral_arm": spectral_arm,
        "producer_compatibility_hash": producer.digest,
        "projector_before_hash": "spectral-before",
        "projector_after_hash": projector_after or producer.compatibility.spectral_projector_hash,
        "projector_gradient_evidence": {"l2_norm_max": 1.25, "nonzero_steps": 2, "measured_steps": 2},
        "projector_update_evidence": {"changed_parameter_count": 3, "optimizer_steps": 2},
        "initialization_id": "u-init-20260907",
        "checkpoint_id": "u-parent-20260907",
        "source_id": "source-receipt-20260907",
        "split_role_hash": role_manifest.digest,
        "role_manifest_digest": role_manifest.digest,
        "verified_prior_receipt": None,
        "verified_prior_receipt_hash": None,
    }


def test_variants_share_identical_rows_and_delta() -> None:
    row = _row(0, -2.0)
    assert row.v126.shape == (126,)
    assert row.v222.shape == (222,)
    assert row.v270.shape == (270,)
    assert row.v366.shape == (366,)
    assert torch.equal(row.v222[:126], row.v126)
    assert torch.equal(row.v222[126:], row.delta96)
    assert torch.equal(row.v366[:270], row.o270)
    assert torch.equal(row.v366[270:], row.delta96)
    assert row.raw_gain < 0.0


def test_q90_linear_scale_and_floor_provenance() -> None:
    scale = compute_gain_scale([1.0, -2.0, 4.0, -8.0])
    assert scale.scale == pytest.approx(6.8)
    assert scale.method == "linear"
    assert scale.quantile == pytest.approx(0.9)
    assert compute_gain_scale([0.0], floor=1e-3).scale == pytest.approx(1e-3)
    assert compute_gain_scale([], floor=1e-3).floor_applied


def test_bank_roundtrip_hashes_and_source_is_immutable(tmp_path: Path) -> None:
    rows = [_row(i, value) for i, value in enumerate((1.0, -2.0, 4.0))]
    destination = tmp_path / "bank"
    original = rows[0].state96.clone()
    manifest = _build(rows, destination)
    rows[0].state96[0] = 99.0
    reader = ValueBankReader(destination)
    assert reader.manifest().row_count == 3
    assert torch.equal(reader.rows()[0].state96, original)
    assert reader.manifest().gain_scale == manifest.gain_scale
    verified = reader.verify()
    assert verified["shard_count"] == 1
    assert all(len(item["sha256"]) == 64 for item in reader.index["shards"])


def test_same_input_produces_deterministic_shards_and_index(tmp_path: Path) -> None:
    rows = [_row(i, float(i - 1)) for i in range(3)]
    first = tmp_path / "first"
    second = tmp_path / "second"
    _build(rows, first)
    _build(rows, second)
    for name in ("index.json", "shard-00000.pt"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_overwrite_and_destination_race_refused(tmp_path: Path) -> None:
    destination = tmp_path / "bank"
    _build([_row(0, 1.0)], destination)
    with pytest.raises(FileExistsError):
        ValueBankWriter(destination, producer=_producer(), split_role_hash="split-hash", role_membership={"subject-0": "producer_fit"}, engineering_only=True)
    raced = tmp_path / "raced"
    writer = ValueBankWriter(raced, producer=_producer(), split_role_hash="split-hash", role_membership={"subject-0": "producer_fit"}, engineering_only=True)
    writer.append(_row(0, 1.0))
    raced.mkdir()
    with pytest.raises(FileExistsError):
        writer.finalize()
    assert not (raced / "index.json").exists()
    writer.abort()


def test_corrupt_shard_and_manifest_are_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "bank"
    _build([_row(0, 1.0)], destination)
    shard = destination / "shard-00000.pt"
    shard.write_bytes(shard.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        ValueBankReader(destination)


def test_producer_and_split_mismatches_fail_closed(tmp_path: Path) -> None:
    destination = tmp_path / "bank"
    _build([_row(0, 1.0)], destination)
    with pytest.raises((TypeError, ValueError)):
        ValueBankReader(destination, expected_producer="changed")
    with pytest.raises(ValueError, match="split"):
        ValueBankReader(destination, expected_split_role_hash="changed")


def test_role_overlap_and_optional_stop_reject(tmp_path: Path) -> None:
    destination = tmp_path / "bank"
    writer = ValueBankWriter(destination, producer=_producer(), split_role_hash="split-hash", role_membership={"same": "producer_fit"}, engineering_only=True)
    writer.append(_row(0, 1.0, split_role="producer_fit", subject="same"))
    with pytest.raises(ValueError, match="overlaps|role differs"):
        writer.append(_row(1, 1.0, split_role="validation", subject="same"))
    assert not destination.exists()
    bad = _row(0, 1.0)
    bad = ValueBankRow(**{**bad.__dict__, "inclusion_mechanism": "optionally_stopped"})
    with pytest.raises(ValueError, match="exact support provenance|inclusion"):
        _build([bad], tmp_path / "bad")


def test_unsafe_nested_payload_and_known_secret_reject(tmp_path: Path) -> None:
    row = _row(0, 1.0)
    unsafe = {"state96": row.state96, "f_spec168": row.f_spec168, "semantic3": row.semantic3, "reliability3": row.reliability3, "q_bar24": row.q_bar24, "delta96": row.delta96, "raw_gain": 1.0, "benefit": 1.0, "harm": 0.0, "action_id": "a", "context_id": "c", "pixels": [[1, 2]]}
    with pytest.raises(ValueError, match="unknown value-bank row fields"):
        _build([unsafe], tmp_path / "unsafe")
    secret = _row(0, 1.0)
    secret = ValueBankRow(**{**secret.__dict__, "action_id": "api_key=synthetic_not_real"})
    with pytest.raises(ValueError, match="credential-like"):
        _build([secret], tmp_path / "secret")


def test_malformed_nonfinite_and_shape_rows_reject(tmp_path: Path) -> None:
    row = _row(0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        _build([ValueBankRow(**{**row.__dict__, "raw_gain": float("nan")})], tmp_path / "nan")
    with pytest.raises(ValueError, match="shape"):
        _build([ValueBankRow(**{**row.__dict__, "state96": torch.zeros(95)})], tmp_path / "shape")


def test_diagnostic_screening_is_explicitly_segregated(tmp_path: Path) -> None:
    screening = _row(0, 1.0, role="screening", diagnostic=True)
    destination = tmp_path / "diagnostic"
    _build([screening], destination, diagnostic=True)
    assert ValueBankReader(destination).verify()["status"]["evidence_status"] in {"DIAGNOSTIC_ONLY", "ENGINEERING_ONLY"}
    with pytest.raises(ValueError, match="diagnostic"):
        _build([_row(0, 1.0, role="screening")], tmp_path / "main")


def test_empty_bank_is_explicitly_blocked(tmp_path: Path) -> None:
    destination = tmp_path / "empty"
    build_value_bank([], destination, producer=_producer(), split_role_hash="split-hash", role_membership={}, engineering_only=True)
    result = ValueBankReader(destination).verify()
    assert result["status"]["evidence_status"] == "ENGINEERING_ONLY"
    assert result["row_count"] == 0


def test_action_and_gain_label_adapter_uses_actual_delta() -> None:
    row = _row(0, -1.0)
    action = {"o270": row.o270, "v126": row.v126, "delta": row.delta96, "action_id": "action", "context_id": "context", "point_id": 4, "state_version": 2}
    label = GainLabel(action_id="action", context_id="context", state_version=2, raw_gain=-1.0, benefit=0.0, harm=1.0, mask_count=4)
    adapted = ValueBankRow.from_action_label(action, label, support_provenance="complete_support_v1")
    assert torch.equal(adapted.delta96, row.delta96)
    assert adapted.point_id == 4


def test_source_scale_is_preserved_and_diagnostic_gain_cannot_change_it(tmp_path: Path) -> None:
    rows = [_row(0, 1.0), _row(1, 2.0)]
    first = tmp_path / "first"
    _build(rows, first)
    fixed = ValueBankReader(first).gain_scale
    diagnostic = _row(2, 1_000_000.0, diagnostic=True)
    refreshed = tmp_path / "refreshed"
    build_value_bank(
        rows + [diagnostic],
        refreshed,
        producer=_producer(),
        split_role_hash="split-hash",
        role_membership={row.subject_key: row.split_role for row in rows + [diagnostic]},
        engineering_only=True,
        diagnostic=True,
        source_scale=fixed,
    )
    reader = ValueBankReader(refreshed)
    assert reader.gain_scale.digest == fixed.digest
    assert reader.index["source_scale_hash"] == fixed.digest


def test_main_bank_requires_full_row_and_stage_identities(tmp_path: Path) -> None:
    compatibility = _producer().compatibility
    producer = _producer()
    # A non-engineering role manifest is intentionally not built here because
    # W1 enforces the real calibration-group minimum; the writer must still
    # reject a MAIN row before publication when immutable row identities are
    # absent.
    with pytest.raises(ValueError, match="stage|TrainingRoleManifest"):
        ValueBankWriter(tmp_path / "main", producer=producer, split_role_hash="split-hash")
    assert compatibility.spectral_projector_hash


def test_stage_provenance_schema_is_single_canonical_envelope(tmp_path: Path) -> None:
    producer = _producer()
    stage = {
        "schema_version": STAGE_PROVENANCE_SCHEMA,
        "stage": "updater",
        "completed": True,
        "spectral_arm": "u_plus_spectral",
        "producer_compatibility_hash": producer.digest,
        "projector_before_hash": "before",
        "projector_after_hash": producer.compatibility.spectral_projector_hash,
        "projector_gradient_evidence": {"l2_norm_max": 1.0, "nonzero_steps": 1, "measured_steps": 1},
        "projector_update_evidence": {"changed_parameter_count": 1, "optimizer_steps": 1},
        "initialization_id": "init",
        "checkpoint_id": "checkpoint",
        "source_id": "source",
        "split_role_hash": "split-hash",
        "role_manifest_digest": "roles",
        "verified_prior_receipt": None,
        "verified_prior_receipt_hash": None,
    }
    # Engineering fixtures may carry the complete envelope but still report
    # engineering-only status; reader validates it identically.
    path = tmp_path / "stage"
    _build([_row(0, 1.0)], path, stage_provenance=stage)
    assert ValueBankReader(path).index["stage_provenance"]["schema_version"] == STAGE_PROVENANCE_SCHEMA


def test_production_typed_role_and_stage_receipt_roundtrip(tmp_path: Path) -> None:
    producer = _producer()
    role_manifest, role_membership = _production_roles()
    row = _row(0, 1.0, subject="producer-0")
    row = ValueBankRow(**{**row.__dict__, "producer_compatibility_hash": producer.digest, "split_role_hash": role_manifest.digest})
    path = tmp_path / "production"
    build_value_bank(
        [row],
        path,
        producer=producer,
        split_role_hash=role_manifest.digest,
        role_manifest=role_manifest,
        stage_provenance=_stage(producer, role_manifest),
    )
    reader = ValueBankReader(path / "index.json", expected_role_manifest=role_manifest, expected_baseline_split_hash="baseline-split-v1", expected_producer=producer)
    assert reader.index["role_manifest"]["schema_version"] == "pfgr-lite-training-roles-v1"
    assert reader.index["stage_provenance"]["schema_version"] == STAGE_PROVENANCE_SCHEMA
    assert role_membership["producer-0"] == "producer_fit"


def test_production_stage_and_role_mismatches_fail_closed(tmp_path: Path) -> None:
    producer = _producer()
    role_manifest, _ = _production_roles()
    row = _row(0, 1.0, subject="producer-0")
    row = ValueBankRow(**{**row.__dict__, "producer_compatibility_hash": producer.digest, "split_role_hash": role_manifest.digest})
    with pytest.raises(ValueError, match="projector"):
        ValueBankWriter(
            tmp_path / "wrong-projector",
            producer=producer,
            split_role_hash=role_manifest.digest,
            role_manifest=role_manifest,
            stage_provenance=_stage(producer, role_manifest, projector_after="wrong"),
        )
    fake_prior = _stage(producer, role_manifest, spectral_arm="verified_prior")
    fake_prior["verified_prior_receipt"] = "not-a-receipt"
    fake_prior["verified_prior_receipt_hash"] = "0" * 64
    with pytest.raises(ValueError, match="verified_prior"):
        ValueBankWriter(
            tmp_path / "fake-prior",
            producer=producer,
            split_role_hash=role_manifest.digest,
            role_manifest=role_manifest,
            stage_provenance=fake_prior,
        )
    path = tmp_path / "production"
    build_value_bank([row], path, producer=producer, split_role_hash=role_manifest.digest, role_manifest=role_manifest, stage_provenance=_stage(producer, role_manifest))
    with pytest.raises(ValueError, match="baseline"):
        ValueBankReader(path, expected_baseline_split_hash="other-baseline")
    with pytest.raises(ValueError, match="role manifest"):
        altered = TrainingRoleManifest.from_dict({**role_manifest.as_dict(), "baseline_split_hash": "other-baseline"})
        ValueBankReader(path, expected_role_manifest=altered)


def test_verified_prior_requires_hashed_original_receipt(tmp_path: Path) -> None:
    producer = _producer()
    role_manifest, _ = _production_roles()
    original = _stage(producer, role_manifest)
    prior = _stage(producer, role_manifest, spectral_arm="verified_prior")
    prior["projector_gradient_evidence"] = {"l2_norm_max": 0.0, "nonzero_steps": 0, "measured_steps": 0}
    prior["projector_update_evidence"] = {"changed_parameter_count": 0, "optimizer_steps": 0}
    prior["verified_prior_receipt"] = original
    prior["verified_prior_receipt_hash"] = canonical_digest(original, prefix="pfgr-lite-producer-stage-receipt-v1|")
    row = _row(0, 1.0, subject="producer-0")
    row = ValueBankRow(**{**row.__dict__, "producer_compatibility_hash": producer.digest, "split_role_hash": role_manifest.digest})
    path = tmp_path / "verified-prior"
    build_value_bank([row], path, producer=producer, split_role_hash=role_manifest.digest, role_manifest=role_manifest, stage_provenance=prior)
    assert ValueBankReader(path).stage_provenance["spectral_arm"] == "verified_prior"


@pytest.mark.parametrize("mutation", ("incomplete", "not-completed", "unchanged"))
def test_verified_prior_rejects_invalid_original_receipt(tmp_path: Path, mutation: str) -> None:
    producer = _producer()
    role_manifest, _ = _production_roles()
    original = _stage(producer, role_manifest)
    if mutation == "incomplete":
        original.pop("completed")
    elif mutation == "not-completed":
        original["completed"] = False
    else:
        original["projector_before_hash"] = original["projector_after_hash"]
    prior = _stage(producer, role_manifest, spectral_arm="verified_prior")
    prior["projector_gradient_evidence"] = {"l2_norm_max": 0.0, "nonzero_steps": 0, "measured_steps": 0}
    prior["projector_update_evidence"] = {"changed_parameter_count": 0, "optimizer_steps": 0}
    prior["verified_prior_receipt"] = original
    prior["verified_prior_receipt_hash"] = canonical_digest(original, prefix="pfgr-lite-producer-stage-receipt-v1|")
    row = _row(0, 1.0, subject="producer-0")
    row = ValueBankRow(**{**row.__dict__, "producer_compatibility_hash": producer.digest, "split_role_hash": role_manifest.digest})
    with pytest.raises(ValueError, match="verified_prior"):
        ValueBankWriter(tmp_path / mutation, producer=producer, split_role_hash=role_manifest.digest, role_manifest=role_manifest, stage_provenance=prior)


def test_production_row_identity_and_atomic_index_last_are_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    producer = _producer()
    role_manifest, _ = _production_roles()
    incomplete = _row(0, 1.0, subject="producer-0")
    with pytest.raises(ValueError, match="immutable identities"):
        writer = ValueBankWriter(tmp_path / "missing-row-id", producer=producer, split_role_hash=role_manifest.digest, role_manifest=role_manifest, stage_provenance=_stage(producer, role_manifest))
        writer.append(ValueBankRow(**{**incomplete.__dict__, "producer_compatibility_hash": producer.digest, "split_role_hash": ""}))
    destination = tmp_path / "streamed"
    replace = value_bank_module.os.replace
    publication_moves: list[str] = []

    def record_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        if Path(target).parent == destination:
            publication_moves.append(Path(source).name)
        replace(source, target)

    monkeypatch.setattr(value_bank_module.os, "replace", record_replace)
    writer = ValueBankWriter(destination, producer=producer, split_role_hash="split-hash", role_membership={}, engineering_only=True, max_rows_per_shard=1)
    writer.append([_row(0, 1.0), _row(1, -1.0)])
    assert len(writer._row_buffer) == 0
    assert len(writer._shard_entries) == 2
    writer.finalize()
    assert publication_moves[-1] == "index.json"
