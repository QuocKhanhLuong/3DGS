"""Blocking leakage-boundary tests using only temporary synthetic payloads."""

from __future__ import annotations

import hashlib
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import smagm
from smagm.contracts.coordinates import (
    PhysicalPlane,
    SourceAffineTransform,
    SourceConvention,
)
from smagm.contracts.observation import (
    AccessLevel,
    ObservationLedger,
    ObservationMeta,
    SparseManifest,
    validate_patient_split_manifests,
)


def _plane(observation_id: str) -> PhysicalPlane:
    return PhysicalPlane(
        pixel_center_origin_ras_mm=(0.0, 0.0, 0.0),
        axis_u_ras=(1.0, 0.0, 0.0),
        axis_v_ras=(0.0, 1.0, 0.0),
        spacing_uv_mm=(1.0, 1.0),
        thickness_mm=1.0,
        shape_hw=(2, 2),
        signed_normal_ras=(0.0, 0.0, 1.0),
        observation_id=observation_id,
    )


def _entry(identifier: str, *, level: AccessLevel, patient: str = "patient-1", path: str | None = None) -> ObservationMeta:
    return ObservationMeta(
        observation_id=identifier,
        patient_id=patient,
        split="train",
        relative_path=path if path is not None else f"{identifier}.bin",
        access_level=level,
        modality_id="synthetic-T1",
        plane=_plane(identifier),
        is_synthetic=True,
    )


def _payload(identifier: str) -> bytes:
    return {
        "context": b"legal-context",
        "target": b"hidden-target",
    }.get(identifier, identifier.encode("utf-8"))


def _manifest(
    entries: tuple[ObservationMeta, ...],
    *,
    manifest_id: str = "",
    integrity_digests: dict[str, str] | None = None,
) -> SparseManifest:
    digests = integrity_digests or {
        entry.observation_id: hashlib.sha256(_payload(entry.observation_id)).hexdigest()
        for entry in entries
    }
    return SparseManifest(
        entries,
        manifest_id=manifest_id,
        integrity_digests=digests,
    )


class ManifestLegalityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        (self.root / "context.bin").write_bytes(b"legal-context")
        (self.root / "target.bin").write_bytes(b"hidden-target")
        (self.root / "forbidden-audit-volume.bin").write_bytes(b"forbidden-full-volume")
        self.manifest = _manifest(
            (_entry("context", level=AccessLevel.CONTEXT), _entry("target", level=AccessLevel.TARGET)),
            manifest_id="synthetic-manifest",
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _ledger(self) -> ObservationLedger:
        return ObservationLedger(self.manifest, self.root)

    def test_manifest_rejects_duplicate_ids_duplicate_paths_and_slice_level_patient_splits(self) -> None:
        with self.assertRaises(ValueError):
            _manifest((_entry("same", level=AccessLevel.CONTEXT), _entry("same", level=AccessLevel.TARGET)))
        with self.assertRaises(ValueError):
            _manifest((_entry("a", level=AccessLevel.CONTEXT, path="same.bin"), _entry("b", level=AccessLevel.TARGET, path="same.bin")))
        val_entry = ObservationMeta(
            observation_id="same-patient-validation",
            patient_id="patient-1",
            split="validation",
            relative_path="target.bin",
            access_level=AccessLevel.TARGET,
            modality_id="synthetic-T1",
            plane=_plane("same-patient-validation"),
            is_synthetic=True,
        )
        with self.assertRaises(ValueError):
            _manifest((_entry("context", level=AccessLevel.CONTEXT), val_entry))

    def test_manifest_relative_paths_are_confined_and_no_public_raw_provider_exists(self) -> None:
        for unsafe in ("", "../target.bin", "/absolute.bin", "nested/../../target.bin"):
            with self.subTest(unsafe=unsafe):
                with self.assertRaises(ValueError):
                    _entry("unsafe", level=AccessLevel.CONTEXT, path=unsafe)
        ledger = self._ledger()
        with self.assertRaises(KeyError):
            ledger.open_context("forbidden-audit-volume")
        self.assertFalse(hasattr(smagm, "ManifestFileProvider"))

    def test_symlink_escape_and_payload_mutation_fail_before_audit(self) -> None:
        with tempfile.TemporaryDirectory() as outside_directory:
            outside = Path(outside_directory) / "outside.bin"
            outside.write_bytes(b"outside")
            try:
                (self.root / "escape.bin").symlink_to(outside)
            except OSError as error:
                if getattr(error, "winerror", None) != 1314:
                    raise
            else:
                escape_manifest = _manifest(
                    (_entry("escape", level=AccessLevel.CONTEXT, path="escape.bin"),)
                )
                with self.assertRaises(PermissionError):
                    ObservationLedger(escape_manifest, self.root).open_context("escape")

        (self.root / "context.bin").write_bytes(b"mutated-after-manifest")
        ledger = self._ledger()
        with self.assertRaises(OSError):
            ledger.open_context("context")
        self.assertEqual(ledger.audit_records, ())
        self.assertEqual(ledger.event_records, ())

    def test_metadata_is_visible_without_opening_any_payload(self) -> None:
        ledger = self._ledger()
        metadata = ledger.metadata("target")

        self.assertEqual(metadata.observation_id, "target")
        self.assertEqual(metadata.access_level, AccessLevel.TARGET)
        self.assertFalse(hasattr(metadata, "content_sha256"))
        self.assertEqual(ledger.audit_records, ())

    def test_target_digest_is_absent_from_public_metadata_and_manifest_hash(self) -> None:
        entries = (
            _entry("context", level=AccessLevel.CONTEXT),
            _entry("target", level=AccessLevel.TARGET),
        )
        first = _manifest(
            entries,
            integrity_digests={"context": "a" * 64, "target": "b" * 64},
        )
        second = _manifest(
            entries,
            integrity_digests={"context": "c" * 64, "target": "d" * 64},
        )

        self.assertEqual(first.canonical_hash, second.canonical_hash)
        self.assertEqual(first.to_canonical_dict(), second.to_canonical_dict())
        self.assertNotIn("sha256", first.canonical_json())
        self.assertNotIn("integrity", repr(first))
        self.assertFalse(hasattr(first.metadata("target"), "content_sha256"))

    def test_context_is_readable_but_target_is_not_before_commit_and_reveal(self) -> None:
        ledger = self._ledger()
        self.assertEqual(ledger.open_context("context"), b"legal-context")
        with self.assertRaises(PermissionError):
            ledger.open_context("target")
        self.assertEqual(tuple(row.observation_id for row in ledger.audit_records), ("context",))

        token = ledger.commit_target("target")
        self.assertEqual(ledger.reveal(token), b"hidden-target")
        with self.assertRaises(PermissionError):
            ledger.reveal(token)
        self.assertEqual(tuple(row.observation_id for row in ledger.audit_records), ("context", "target"))

    def test_declared_target_budget_is_enforced_at_commit(self) -> None:
        ledger = ObservationLedger(self.manifest, self.root, target_budget=0.5)
        with self.assertRaisesRegex(RuntimeError, "budget"):
            ledger.commit_target("target")
        self.assertEqual(ledger.committed_target_cost, 0.0)
        self.assertEqual(ledger.remaining_target_budget, 0.5)

    def test_decimal_target_costs_do_not_false_reject_an_exact_budget(self) -> None:
        target_a = replace(
            _entry("target-a", level=AccessLevel.TARGET, path="target-a.bin"),
            cost=0.1,
        )
        target_b = replace(
            _entry("target-b", level=AccessLevel.TARGET, path="target-b.bin"),
            cost=0.2,
        )
        ledger = ObservationLedger(
            _manifest((target_a, target_b)),
            self.root,
            target_budget=0.3,
        )

        ledger.commit_target("target-a")
        ledger.commit_target("target-b")
        self.assertEqual(ledger.committed_target_cost, 0.3)
        self.assertEqual(ledger.remaining_target_budget, 0.0)

    def test_leakage_positive_control_rejects_target_bytes_before_commit(self) -> None:
        """A deliberate forbidden target access must fail, or T0 is scientifically invalid."""
        ledger = self._ledger()

        with self.assertRaises(PermissionError):
            ledger.open_context("target")
        self.assertEqual(ledger.audit_records, ())

    def test_wrong_ledger_invalid_objects_and_duplicate_commits_are_rejected(self) -> None:
        ledger = self._ledger()
        other_ledger = self._ledger()
        token = ledger.commit_target("target")

        with self.assertRaises(PermissionError):
            other_ledger.reveal(token)
        with self.assertRaises(PermissionError):
            ledger.reveal(object())
        with self.assertRaises(RuntimeError):
            ledger.commit_target("target")

    def test_open_audit_and_hash_are_deterministic_for_equal_legal_sequences(self) -> None:
        first = self._ledger()
        first.open_context("context")
        first.reveal(first.commit_target("target"))
        second = self._ledger()
        second.open_context("context")
        second.reveal(second.commit_target("target"))

        self.assertEqual(first.audit_records, second.audit_records)
        self.assertEqual(first.audit_hash, second.audit_hash)
        self.assertEqual(first.manifest_hash, second.manifest_hash)
        self.assertEqual([row.sequence for row in first.audit_records], [0, 1])
        self.assertEqual(
            [(event.sequence, event.event, event.observation_id) for event in first.event_records],
            [
                (0, "OPEN_CONTEXT", "context"),
                (1, "COMMIT_TARGET", "target"),
                (2, "REVEAL_TARGET", "target"),
            ],
        )
        self.assertEqual(
            first.audit_records[0].content_sha256,
            "8b786342acc141755cf386c0c29c533932e8e71923797e8ffcd087302b06a2a9",
        )

    def test_patient_split_validation_is_cohort_wide_and_hashable(self) -> None:
        train = _manifest((_entry("train-a", level=AccessLevel.CONTEXT),))
        validation_entry = ObservationMeta(
            observation_id="validation-a",
            patient_id="patient-1",
            split="validation",
            relative_path="validation-a.bin",
            access_level=AccessLevel.CONTEXT,
            modality_id="synthetic-T1",
            plane=_plane("validation-a"),
            is_synthetic=True,
        )
        validation = _manifest((validation_entry,))
        with self.assertRaises(ValueError):
            validate_patient_split_manifests((train, validation))
        self.assertEqual(
            validate_patient_split_manifests((train,)),
            validate_patient_split_manifests((train,)),
        )

    def test_observation_cost_and_real_source_provenance_are_blocking(self) -> None:
        with self.assertRaises(ValueError):
            replace(_entry("bad-cost", level=AccessLevel.CONTEXT), cost=math.inf)
        with self.assertRaisesRegex(ValueError, "source-affine provenance"):
            ObservationMeta(
                observation_id="real",
                patient_id="patient-real",
                split="train",
                relative_path="real.bin",
                access_level=AccessLevel.CONTEXT,
                modality_id="T1",
                plane=_plane("real"),
            )

        source = SourceAffineTransform(
            (
                (-1.0, 0.0, 0.0, 0.0),
                (0.0, -1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            SourceConvention.DICOM_LPS,
        )
        real_plane = PhysicalPlane(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0),
            1.0,
            (2, 2),
            (0.0, 0.0, 1.0),
            source_transform=source,
            observation_id="real",
        )
        (self.root / "real.bin").write_bytes(b"real")
        real = ObservationMeta(
            observation_id="real",
            patient_id="patient-real",
            split="train",
            relative_path="real.bin",
            access_level=AccessLevel.CONTEXT,
            modality_id="T1",
            plane=real_plane,
        )
        self.assertEqual(
            ObservationLedger(_manifest((real,)), self.root).open_context("real"),
            b"real",
        )
