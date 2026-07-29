"""Blocking immutable-availability and episodic-role contracts for T0.5."""

from __future__ import annotations

import hashlib
import inspect

import pytest

from smagm.contracts.coordinates import PhysicalPlane
from smagm.contracts.episode import EpisodeAssignment, EpisodeLedger
from smagm.contracts.observation import (
    AvailabilityObservationMeta,
    PatientSplitRegistry,
    SparseAvailabilityManifest,
    validate_patient_split_manifests,
)


def _plane(observation_id: str) -> PhysicalPlane:
    return PhysicalPlane(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0),
        1.0,
        (2, 2),
        (0.0, 0.0, 1.0),
        observation_id=observation_id,
    )


def _entry(
    observation_id: str,
    *,
    patient_id: str = "patient-a",
    split: str = "train",
    cost_key: str | None = "T2:axial",
) -> AvailabilityObservationMeta:
    return AvailabilityObservationMeta(
        observation_id=observation_id,
        patient_id=patient_id,
        split=split,
        relative_path=f"{observation_id}.bin",
        modality_id="T2",
        plane=_plane(observation_id),
        is_synthetic=True,
        acquisition_cost_key=cost_key,
    )


def _manifest(*entries: AvailabilityObservationMeta, digest_seed: bytes = b"payload") -> SparseAvailabilityManifest:
    return SparseAvailabilityManifest(
        entries,
        manifest_id="permanently-sparse-v1",
        integrity_digests={entry.observation_id: hashlib.sha256(digest_seed + entry.observation_id.encode()).hexdigest() for entry in entries},
    )


def _training_ledger(manifest: SparseAvailabilityManifest, assignment: EpisodeAssignment, root) -> EpisodeLedger:
    return EpisodeLedger(manifest, assignment, root, split_registry=PatientSplitRegistry.create((manifest,)))


def test_availability_is_role_free_canonical_immutable_and_binds_content() -> None:
    context, target = _entry("context"), _entry("target")
    original_digests = {entry.observation_id: hashlib.sha256(b"a" + entry.observation_id.encode()).hexdigest() for entry in (context, target)}
    manifest = SparseAvailabilityManifest((target, context), manifest_id="m", integrity_digests=original_digests)
    reordered = SparseAvailabilityManifest((context, target), manifest_id="m", integrity_digests=dict(original_digests))
    changed_content = SparseAvailabilityManifest(
        (context, target),
        manifest_id="m",
        integrity_digests={**original_digests, "target": hashlib.sha256(b"replaced target").hexdigest()},
    )

    assert not hasattr(context, "access_level")
    assert manifest.entries == (context, target)
    assert manifest.manifest_hash == reordered.manifest_hash
    # Public availability identity is deliberately non-sensitive and stable
    # across payload replacement.  Its sealed content binding is not public.
    assert manifest.manifest_hash == changed_content.manifest_hash
    assert not hasattr(manifest, "content_binding_hash")
    assert "content_binding" not in manifest.canonical_json()
    original_digests["target"] = hashlib.sha256(b"mutable caller value").hexdigest()
    assert manifest.manifest_hash == changed_content.manifest_hash
    with pytest.raises((AttributeError, TypeError)):
        manifest.entries += (context,)  # type: ignore[misc]
    with pytest.raises(TypeError):
        manifest._integrity_digests["target"] = "0" * 64  # type: ignore[index]


def test_assignments_are_manifest_bound_immutable_and_roles_do_not_change_availability(tmp_path) -> None:
    entries = (_entry("a"), _entry("b"), _entry("c"))
    payloads = {"a": b"context-a", "b": b"target-b", "c": b"unused-c"}
    manifest = SparseAvailabilityManifest(
        entries,
        manifest_id="permanently-sparse-v1",
        integrity_digests={key: hashlib.sha256(value).hexdigest() for key, value in payloads.items()},
    )
    before = manifest.manifest_hash
    first = EpisodeAssignment.create(manifest, episode_id="episode-1", patient_id="patient-a", context_ids=("a",), target_ids=("b",))
    second = EpisodeAssignment.create(manifest, episode_id="episode-2", patient_id="patient-a", context_ids=("b",), target_ids=("a",))

    assert first.manifest_hash == second.manifest_hash == before
    assert first.assignment_hash != second.assignment_hash
    assert not hasattr(first, "content_binding_hash")
    assert "content_binding" not in repr(first)
    assert first.context_ids == ("a",)
    assert first.target_ids == ("b",)
    with pytest.raises((AttributeError, TypeError)):
        first.context_ids += ("c",)  # type: ignore[misc]

    # Episode roles are offline training metadata: opening context must not
    # instantiate or charge deployment acquisition state.
    for observation_id, payload in payloads.items():
        (tmp_path / f"{observation_id}.bin").write_bytes(payload)
    ledger = _training_ledger(manifest, EpisodeAssignment.create(manifest, episode_id="episode-1", patient_id="patient-a", context_ids=("a",), target_ids=("b",)), tmp_path)
    assert ledger.open_context("a") == payloads["a"]
    assert manifest.manifest_hash == before


@pytest.mark.parametrize(
    ("context_ids", "target_ids", "patient_id", "error"),
    [
        (("a", "a"), ("b",), "patient-a", "duplicate"),
        (("a",), ("a",), "patient-a", "disjoint"),
        (("unknown",), ("b",), "patient-a", "unknown"),
        (("a",), ("b",), "patient-b", "patient"),
        (("",), ("b",), "patient-a", "non-empty"),
    ],
)
def test_assignment_rejects_illegal_ids_and_patient_binding(context_ids, target_ids, patient_id, error) -> None:
    manifest = _manifest(_entry("a"), _entry("b"), _entry("foreign", patient_id="patient-b"))
    with pytest.raises((KeyError, ValueError), match=error):
        EpisodeAssignment.create(manifest, episode_id="episode", patient_id=patient_id, context_ids=context_ids, target_ids=target_ids)


def test_assignment_and_ledger_reject_wrong_manifest_and_unselected_payload_access(tmp_path) -> None:
    manifest = _manifest(_entry("a"), _entry("b"))
    other = _manifest(_entry("a"), _entry("b"), digest_seed=b"different")
    assignment = EpisodeAssignment.create(manifest, episode_id="episode", patient_id="patient-a", context_ids=("a",), target_ids=("b",))
    with pytest.raises(ValueError, match="bound to this .*manifest"):
        _training_ledger(other, assignment, tmp_path)

    (tmp_path / "a.bin").write_bytes(b"a")
    (tmp_path / "b.bin").write_bytes(b"b")
    legal = SparseAvailabilityManifest(
        (_entry("a"), _entry("b")),
        integrity_digests={"a": hashlib.sha256(b"a").hexdigest(), "b": hashlib.sha256(b"b").hexdigest()},
    )
    ledger = _training_ledger(legal, EpisodeAssignment.create(legal, episode_id="episode", patient_id="patient-a", context_ids=("a",), target_ids=("b",)), tmp_path)
    assert ledger.expose_target_metadata("b").relative_path == "b.bin"
    assert ledger.audit_records == ()
    with pytest.raises(PermissionError):
        ledger.open_context("b")
    with pytest.raises(PermissionError):
        ledger.expose_target_metadata("a")
    with pytest.raises(PermissionError):
        ledger.open_context("not-in-manifest")
    with pytest.raises(KeyError):
        legal.metadata("not-in-manifest")


def test_direct_assignment_constructor_cannot_bypass_manifest_and_patient_validation() -> None:
    manifest = _manifest(_entry("a"), _entry("b"))
    # Only the factory can bind actual manifest membership and patient identity.
    with pytest.raises(TypeError):
        EpisodeAssignment(
            episode_id="forged",
            manifest_hash=manifest.manifest_hash,
            patient_id="patient-b",
            context_ids=("a",),
            target_ids=("b",),
        )


def test_availability_patient_split_validator_is_joint_canonical_and_rejects_illegal_or_mixed_split() -> None:
    train = _manifest(_entry("train", patient_id="patient-train", split="train"))
    validation = _manifest(_entry("validation", patient_id="patient-validation", split="validation"))
    assert validate_patient_split_manifests((train, validation)) == validate_patient_split_manifests((validation, train))

    same_patient_train = _manifest(_entry("same-train", patient_id="patient-shared", split="train"))
    same_patient_validation = _manifest(_entry("same-validation", patient_id="patient-shared", split="validation"))
    with pytest.raises(ValueError, match="different splits"):
        validate_patient_split_manifests((same_patient_train, same_patient_validation))
    with pytest.raises(ValueError, match="split"):
        SparseAvailabilityManifest(
            (_entry("illegal", patient_id="patient-illegal", split="not-a-legal-cohort"),),
            integrity_digests={"illegal": hashlib.sha256(b"illegal").hexdigest()},
        )


def test_patient_split_registry_is_required_order_invariant_and_has_exactly_four_labels() -> None:
    manifests = tuple(
        _manifest(_entry(f"{split}-obs", patient_id=f"patient-{split}", split=split))
        for split in ("train", "validation", "t1_lesion_validation", "t5_final_audit")
    )
    registry = PatientSplitRegistry.create(manifests)
    reordered = PatientSplitRegistry.create(tuple(reversed(manifests)))
    assert registry.registry_hash == reordered.registry_hash
    assert registry.manifest_hashes == reordered.manifest_hashes
    assert dict(registry.patient_splits) == {
        "patient-train": "train",
        "patient-validation": "validation",
        "patient-t1_lesion_validation": "t1_lesion_validation",
        "patient-t5_final_audit": "t5_final_audit",
    }
    assert set(dict(registry.patient_splits).values()) == {
        "train", "validation", "t1_lesion_validation", "t5_final_audit"
    }


def test_training_ledger_rejects_unregistered_manifest_before_provider_or_audit(tmp_path) -> None:
    registered = _manifest(_entry("registered", patient_id="patient-registered", split="train"))
    unregistered = _manifest(_entry("context", patient_id="patient-unregistered", split="train"), _entry("target", patient_id="patient-unregistered", split="train"))
    (tmp_path / "context.bin").write_bytes(b"unregistered-context")
    (tmp_path / "target.bin").write_bytes(b"unregistered-target")
    assignment = EpisodeAssignment.create(
        unregistered,
        episode_id="unregistered",
        patient_id="patient-unregistered",
        context_ids=("context",),
        target_ids=("target",),
    )
    with pytest.raises(PermissionError, match="unregistered|registry"):
        EpisodeLedger(unregistered, assignment, tmp_path, split_registry=PatientSplitRegistry.create((registered,)))
    assert (tmp_path / "target.bin").read_bytes() == b"unregistered-target"


@pytest.mark.parametrize("sealed_split", ["t1_lesion_validation", "t5_final_audit"])
def test_development_episode_rejects_sealed_lesion_and_final_audit_roles(tmp_path, sealed_split: str) -> None:
    context = _entry("context", patient_id="sealed-patient", split=sealed_split)
    sealed = _entry("sealed", patient_id="sealed-patient", split=sealed_split)
    payloads = {"context": b"SEALED-CONTEXT-SENTINEL", "sealed": b"SEALED-AUDIT-SENTINEL"}
    for observation_id, payload in payloads.items():
        (tmp_path / f"{observation_id}.bin").write_bytes(payload)
    manifest = SparseAvailabilityManifest(
        (context, sealed),
        integrity_digests={key: hashlib.sha256(value).hexdigest() for key, value in payloads.items()},
    )
    assignment = EpisodeAssignment.create(
        manifest,
        episode_id="development-episode",
        patient_id="sealed-patient",
        context_ids=("context",),
        target_ids=("sealed",),
    )
    with pytest.raises(PermissionError, match="sealed|audit|development"):
        _training_ledger(manifest, assignment, tmp_path)
    # Rejection happens before any open/audit event is possible.
    assert (tmp_path / "sealed.bin").read_bytes() == b"SEALED-AUDIT-SENTINEL"


def test_stage_two_does_not_add_t1_or_later_source_modules() -> None:
    # T0.5 is an event/legality correction only.  This source-level contract
    # prevents quietly starting T1-A or later phases under the same gate.
    root = __import__("pathlib").Path(__file__).resolve().parents[2] / "src" / "smagm"
    prohibited = ("features", "baselines", "losses", "training", "anchors", "propagation", "routing")
    assert not [path for name in prohibited for path in root.rglob(f"*{name}*") if path.is_file()]
    assert "prediction_digest" not in inspect.signature(EpisodeAssignment.create).parameters


def test_t05_public_import_paths_are_explicit_and_legacy_access_level_stays_available() -> None:
    from smagm import (  # noqa: PLC0415
        AccessLevel,
        AvailabilityObservationMeta as RootAvailabilityObservationMeta,
        EpisodeAssignment as RootEpisodeAssignment,
        EpisodeLedger as RootEpisodeLedger,
    )
    from smagm.contracts.observations import AvailabilityObservationMeta as CompatibilityAvailabilityObservationMeta  # noqa: PLC0415
    from smagm.data.manifest import SparseAvailabilityManifest as DataSparseAvailabilityManifest  # noqa: PLC0415

    assert AccessLevel.CONTEXT.value == "CONTEXT"
    assert RootAvailabilityObservationMeta is AvailabilityObservationMeta
    assert RootEpisodeAssignment is EpisodeAssignment
    assert RootEpisodeLedger is EpisodeLedger
    assert CompatibilityAvailabilityObservationMeta is AvailabilityObservationMeta
    assert DataSparseAvailabilityManifest is SparseAvailabilityManifest
