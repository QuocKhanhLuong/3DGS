from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import smagm.features.point_guided.pfgr_lite.bank_audit as bank_audit_module

from smagm.features.point_guided.contracts import VolumeGeometry
from smagm.features.point_guided.pfgr_lite.bank_audit import audit_bank_replay, write_state_snapshot
from smagm.features.point_guided.pfgr_lite.provenance import canonical_digest
from smagm.features.point_guided.pfgr_lite.types import ProducerDependencies, ProducerCompatibility, TrainingRoleManifest
from smagm.features.point_guided.pfgr_lite.provenance import SourceProvenance
from smagm.features.point_guided.pfgr_lite.value_bank import ValueBankReader, ValueBankRow, ValueBankWriter


def _fixture(tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    state = SimpleNamespace(
        planes=SimpleNamespace(
            xy=torch.ones(1, 2, 3, 3),
            xz=torch.full((1, 2, 3, 3), 2.0),
            yz=torch.full((1, 2, 3, 3), 3.0),
        ),
        state_version=2,
        state_digest="state-digest",
    )
    context = SimpleNamespace(
        context_id="context-01",
        producer=SimpleNamespace(compatibility_hash="producer-digest"),
        geometry=VolumeGeometry.from_spacing((3, 3, 3)),
        feature_geometry=None,
    )
    binding = {
        "schema_version": "pfgr-lite-subject-context-binding-v1",
        "subject_id": "subject-01",
        "observation_record_id": "observation-01",
        "context_id": "context-01",
        "geometry_hash": "geometry-digest",
        "normalization_hash": "normalization-digest",
    }
    binding["binding_digest"] = canonical_digest(
        {key: binding[key] for key in ("schema_version", "subject_id", "observation_record_id", "context_id", "geometry_hash", "normalization_hash")},
        prefix="pfgr-lite-subject-context-binding-v1|",
    )
    action = {"action_id": "action-01", "state_version": 2, "action_digest": "action-digest", "point_id": 1}
    reference = write_state_snapshot(
        bank,
        state,
        context,
        subject_binding=binding,
        route_hash="route-digest",
        selected_actions=(action,),
        split_role_hash="role-digest",
    )
    row = SimpleNamespace(
        selected_replay_ref=reference,
        producer_compatibility_hash="producer-digest",
        split_role_hash="role-digest",
        subject_key="subject-01",
        context_id="context-01",
        state_version=2,
        state_digest="state-digest",
        action_id="action-01",
    )
    role = SimpleNamespace(digest="role-digest")
    reader = SimpleNamespace(root=bank, role_manifest=role, rows=lambda: (row,))
    return bank, reader, row, role, reference


def _canonical_producer() -> ProducerDependencies:
    compatibility = ProducerCompatibility(
        observation_normalization_hash="normalization-digest",
        geometry_query_version_hash="geometry-query",
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
    return ProducerDependencies(
        compatibility=compatibility,
        source_provenance=SourceProvenance(
            synthetic_untrained=True,
            official_pretrained_verified=False,
            parameter_hash="synthetic-parameters",
            frozen_bn_hash="bn",
            traversal_count=1,
        ),
    )


def _engineering_roles() -> TrainingRoleManifest:
    return TrainingRoleManifest(
        baseline_split_hash="engineering-split",
        baseline_train_subject_ids=("subject-01",),
        baseline_validation_subject_ids=(),
        baseline_test_subject_ids=(),
        producer_fit_subject_ids=("subject-01",),
        calibration_fit_subject_ids=(),
        calibration_allowance_subject_ids=(),
        subject_group_ids=(("subject-01", "group-01"),),
        engineering_only=True,
    )


def test_canonical_writer_reader_audit_replay_indexing(tmp_path):
    """The real W3 writer publishes replay, then reader and audit consume it."""

    producer = _canonical_producer()
    roles = _engineering_roles()
    bank = tmp_path / "canonical-bank"
    writer = ValueBankWriter(
        bank,
        producer=producer,
        split_role_hash=roles.digest,
        role_manifest=roles,
        engineering_only=True,
    )
    state = SimpleNamespace(
        planes=SimpleNamespace(
            xy=torch.ones(1, 2, 3, 3),
            xz=torch.full((1, 2, 3, 3), 2.0),
            yz=torch.full((1, 2, 3, 3), 3.0),
        ),
        state_version=0,
        state_digest="state-digest",
    )
    context = SimpleNamespace(
        context_id="context-01",
        producer=producer,
        geometry=VolumeGeometry.from_spacing((3, 3, 3)),
        feature_geometry=None,
    )
    binding = {
        "schema_version": "pfgr-lite-subject-context-binding-v1",
        "subject_id": "subject-01",
        "observation_record_id": "observation-01",
        "context_id": "context-01",
        "geometry_hash": "geometry-digest",
        "normalization_hash": "normalization-digest",
    }
    binding["binding_digest"] = canonical_digest(
        {key: binding[key] for key in ("schema_version", "subject_id", "observation_record_id", "context_id", "geometry_hash", "normalization_hash")},
        prefix="pfgr-lite-subject-context-binding-v1|",
    )
    action = {"action_id": "action-01", "state_version": 0, "action_digest": "action-digest", "point_id": 1}
    reference = write_state_snapshot(
        writer._stage,
        state,
        context,
        subject_binding=binding,
        route_hash="route-digest",
        selected_actions=(action,),
        split_role_hash=roles.digest,
    )
    row = ValueBankRow(
        state96=torch.zeros(96),
        f_spec168=torch.zeros(168),
        semantic3=torch.zeros(3),
        reliability3=torch.ones(3),
        q_bar24=torch.zeros(24),
        delta96=torch.zeros(96),
        raw_gain=0.5,
        benefit=0.5,
        harm=0.0,
        action_id="action-01",
        context_id="context-01",
        state_version=0,
        point_id=1,
        subject_id="subject-01",
        geometry_id="geometry-digest",
        proposal_hash="action-digest",
        state_digest="state-digest",
        split_role="producer_fit",
        selected_replay_ref=reference,
        sampler_law="exact",
    )
    writer.append(row)
    writer.finalize()

    reader = ValueBankReader(bank, expected_role_manifest=roles)
    assert reader.rows()[0].selected_replay_ref == reference
    receipt = audit_bank_replay(reader, 1, producer=producer, role_manifest=roles)
    assert receipt["snapshots_checked"] == 1
    assert receipt["rows_checked"] == 1

    # Reader-side indexing is strict: a content-addressed file not referenced
    # by any row is rejected before audit, and tampering the indexed bytes is
    # likewise rejected by the shared hash/schema helper.
    rogue = bank / "replay" / ("0" * 64 + ".pt")
    rogue.write_bytes(b"unindexed")
    with pytest.raises(ValueError, match="unindexed replay artifact"):
        ValueBankReader(bank)
    rogue.unlink()
    snapshot = bank / reference
    snapshot.write_bytes(snapshot.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="content hash"):
        ValueBankReader(bank)


def test_audit_checks_identity_for_duplicate_shared_snapshot_reference(tmp_path):
    _bank, reader, row, role, reference = _fixture(tmp_path)
    second = SimpleNamespace(
        selected_replay_ref=reference,
        producer_compatibility_hash="producer-digest",
        split_role_hash="role-digest",
        subject_key="subject-01",
        context_id="different-context",
        state_version=2,
        state_digest="state-digest",
        action_id="action-01",
    )
    reader.rows = lambda: (row, second)
    with pytest.raises(ValueError, match="context identity"):
        audit_bank_replay(reader, 2, producer="producer-digest", role_manifest=role)


def test_audit_loads_shared_snapshot_once_and_accounts_unique_bytes(tmp_path, monkeypatch):
    bank, reader, row, role, reference = _fixture(tmp_path)
    second = SimpleNamespace(**row.__dict__)
    reader.rows = lambda: (row, second)
    original_load = bank_audit_module.torch.load
    loads = 0

    def counted_load(*args, **kwargs):
        nonlocal loads
        loads += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(bank_audit_module.torch, "load", counted_load)
    receipt = audit_bank_replay(reader, 2, producer="producer-digest", role_manifest=role)
    assert loads == 1
    assert receipt["rows_checked"] == 2
    assert receipt["snapshots_checked"] == 1
    assert receipt["bytes_checked"] == (bank / reference).stat().st_size


def test_write_and_audit_selected_state_snapshot(tmp_path):
    _bank, reader, _row, role, _reference = _fixture(tmp_path)
    result = audit_bank_replay(reader, 1, producer="producer-digest", role_manifest=role)
    assert result["audit_kind"] == "state_snapshot_and_row_identity"
    assert result["rows_checked"] == 1
    assert result["snapshots_checked"] == 1
    assert result["bytes_checked"] > 0
    assert result["decoder_calls"] == result["teacher_calls"] == 0
    assert result["reconstruction_replay"] is False


def test_audit_rejects_tampered_snapshot(tmp_path):
    bank, reader, _row, role, reference = _fixture(tmp_path)
    path = bank / reference
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="content hash"):
        audit_bank_replay(reader, 1, producer="producer-digest", role_manifest=role)


def test_audit_rejects_missing_or_unsafe_reference(tmp_path):
    bank, reader, row, role, reference = _fixture(tmp_path)
    (bank / reference).unlink()
    with pytest.raises(ValueError, match="regular file"):
        audit_bank_replay(reader, 1, producer="producer-digest", role_manifest=role)
    row.selected_replay_ref = "../escape.pt"
    with pytest.raises(ValueError, match="replay/<sha256>"):
        audit_bank_replay(reader, 1, producer="producer-digest", role_manifest=role)


def test_audit_rejects_identity_mismatch_and_overrequest(tmp_path):
    _bank, reader, row, role, _reference = _fixture(tmp_path)
    row.split_role_hash = "wrong-role"
    with pytest.raises(ValueError, match="split/role"):
        audit_bank_replay(reader, 1, producer="producer-digest", role_manifest=role)
    row.split_role_hash = "role-digest"
    with pytest.raises(ValueError, match="only 1 selected"):
        audit_bank_replay(reader, 2, producer="producer-digest", role_manifest=role)
