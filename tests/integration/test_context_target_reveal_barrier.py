"""End-to-end legal observation flow for a synthetic sparse patient."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from smagm.contracts.coordinates import PhysicalPlane
from smagm.contracts.observation import AccessLevel, ObservationLedger, ObservationMeta, SparseManifest


def _meta(identifier: str, level: AccessLevel) -> ObservationMeta:
    return ObservationMeta(
        observation_id=identifier,
        patient_id="synthetic-patient",
        split="train",
        relative_path=f"{identifier}.dat",
        access_level=level,
        modality_id="T2",
        plane=PhysicalPlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0), 1.0, (2, 2), (0.0, 0.0, 1.0), observation_id=identifier),
        is_synthetic=True,
    )


class ContextTargetRevealBarrierTests(unittest.TestCase):
    def test_target_never_enters_context_state_before_its_committed_reveal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "context.dat").write_bytes(b"context pixels")
            (root / "target.dat").write_bytes(b"target pixels")
            entries = (
                _meta("context", AccessLevel.CONTEXT),
                _meta("target", AccessLevel.TARGET),
            )
            manifest = SparseManifest(
                entries,
                integrity_digests={
                    entry.observation_id: hashlib.sha256(
                        f"{entry.observation_id} pixels".encode("utf-8")
                    ).hexdigest()
                    for entry in entries
                },
            )
            ledger = ObservationLedger(manifest, root)

            context_state = ledger.open_context("context")
            self.assertEqual(context_state, b"context pixels")
            self.assertNotIn(b"target pixels", context_state)
            self.assertEqual([row.observation_id for row in ledger.audit_records], ["context"])

            target_metadata = ledger.metadata("target")
            self.assertEqual(target_metadata.plane.shape_hw, (2, 2))
            self.assertEqual([row.observation_id for row in ledger.audit_records], ["context"])

            reveal = ledger.commit_target("target")
            target_loss_payload = ledger.reveal(reveal)
            self.assertEqual(target_loss_payload, b"target pixels")
            self.assertEqual([row.observation_id for row in ledger.audit_records], ["context", "target"])
