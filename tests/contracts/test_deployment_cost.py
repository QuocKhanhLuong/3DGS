"""Exact-Decimal deployment accounting must stay separate from episode roles."""

from __future__ import annotations

from decimal import Decimal
import hashlib

import pytest

from smagm.contracts.coordinates import PhysicalPlane
from smagm.contracts.episode import AcquisitionCostEntry, AcquisitionCostSchedule, DeploymentAcquisitionLedger, EpisodeAssignment, EpisodeLedger
from smagm.contracts.observation import AvailabilityObservationMeta, SparseAvailabilityManifest


def _entry(observation_id: str, key: str | None) -> AvailabilityObservationMeta:
    return AvailabilityObservationMeta(
        observation_id=observation_id,
        patient_id="patient-a",
        split="train",
        relative_path=f"{observation_id}.bin",
        modality_id="T2" if "t2" in observation_id else "T1",
        plane=PhysicalPlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0), 1.0, (1, 1), (0.0, 0.0, 1.0), observation_id=observation_id),
        is_synthetic=True,
        acquisition_cost_key=key,
    )


def _manifest() -> SparseAvailabilityManifest:
    entries = (_entry("bootstrap-t1", "T1:axial"), _entry("later-t2", "T2:coronal"), _entry("uncosted", None))
    return SparseAvailabilityManifest(entries, integrity_digests={entry.observation_id: hashlib.sha256(entry.observation_id.encode()).hexdigest() for entry in entries})


def test_episode_roles_and_context_opens_consume_no_deployment_budget(tmp_path) -> None:
    manifest = _manifest()
    schedule = AcquisitionCostSchedule.create(schedule_id="scanner-v1", amounts={"T1:axial": "0.10", "T2:coronal": Decimal("1.25")})
    deployment = DeploymentAcquisitionLedger(manifest=manifest, budget=Decimal("1.35"), schedule=schedule)
    for observation_id in ("bootstrap-t1", "later-t2", "uncosted"):
        (tmp_path / f"{observation_id}.bin").write_bytes(observation_id.encode())
    legal_manifest = SparseAvailabilityManifest(
        manifest.entries,
        integrity_digests={entry.observation_id: hashlib.sha256(entry.observation_id.encode()).hexdigest() for entry in manifest.entries},
    )
    assignment = EpisodeAssignment.create(legal_manifest, episode_id="offline-training", patient_id="patient-a", context_ids=("bootstrap-t1",), target_ids=("later-t2",))
    episode = EpisodeLedger(legal_manifest, assignment, tmp_path)
    assert episode.open_context("bootstrap-t1") == b"bootstrap-t1"
    assert deployment.spent == Decimal("0")
    assert deployment.event_records == ()

    deployment.commit_bootstrap("bootstrap-t1")
    deployment.commit_observation("later-t2")
    assert deployment.spent == Decimal("1.35")
    assert deployment.remaining_budget == Decimal("0.00")
    assert [(event.event, event.cost_key, event.amount) for event in deployment.event_records] == [
        ("COMMIT_BOOTSTRAP", "T1:axial", "0.1"),
        ("COMMIT_OBSERVATION", "T2:coronal", "1.25"),
    ]


def test_schedule_is_canonical_immutable_and_rejects_float_or_hash_injection() -> None:
    schedule = AcquisitionCostSchedule.create(schedule_id="s", amounts={"T2:axial": "1.20", "T1:coronal": Decimal("0")})
    equivalent = AcquisitionCostSchedule.create(schedule_id="s", amounts={"T1:coronal": "0", "T2:axial": Decimal("1.2")})
    assert schedule.schedule_hash == equivalent.schedule_hash
    assert schedule.amount("T2:axial") == Decimal("1.2")
    with pytest.raises((AttributeError, TypeError)):
        schedule.entries += ()  # type: ignore[misc]
    with pytest.raises((TypeError, ValueError)):
        AcquisitionCostSchedule.create(schedule_id="s", amounts={"T2:axial": 1.2})
    with pytest.raises(ValueError):
        AcquisitionCostEntry("T2:axial", "01.20")
    with pytest.raises((TypeError, ValueError)):
        AcquisitionCostSchedule("s", (AcquisitionCostEntry("T2:axial", "1"),), "forged")  # type: ignore[call-arg]


def test_deployment_rejects_unknown_uncosted_duplicate_and_overspend() -> None:
    manifest = _manifest()
    schedule = AcquisitionCostSchedule.create(schedule_id="s", amounts={"T1:axial": "1", "T2:coronal": "2"})
    ledger = DeploymentAcquisitionLedger(manifest=manifest, budget=Decimal("1"), schedule=schedule)
    with pytest.raises(KeyError):
        ledger.commit_observation("unknown")
    with pytest.raises(ValueError, match="no deployment acquisition_cost_key"):
        ledger.commit_observation("uncosted")
    ledger.commit_bootstrap("bootstrap-t1")
    with pytest.raises(RuntimeError, match="already charged"):
        ledger.commit_observation("bootstrap-t1")
    with pytest.raises(RuntimeError, match="exceed"):
        ledger.commit_observation("later-t2")
    assert ledger.spent == Decimal("1")
    assert len(ledger.event_records) == 1


@pytest.mark.parametrize("budget", [1, 1.0, "1", Decimal("NaN"), Decimal("-1")])
def test_deployment_budget_requires_finite_decimal(budget) -> None:
    manifest = _manifest()
    schedule = AcquisitionCostSchedule.create(schedule_id="s", amounts={"T1:axial": "0"})
    with pytest.raises((TypeError, ValueError)):
        DeploymentAcquisitionLedger(manifest=manifest, budget=budget, schedule=schedule)  # type: ignore[arg-type]
