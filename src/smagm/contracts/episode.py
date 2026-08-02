"""T0.5 immutable episode, receipt, and deployment-cost contracts.

The reference renderer remains a pure tensor function.  This module instead
records the scientific ordering around it: a target is committed against a
frozen pre-reveal state, a registrar digests the actual rendered result, and a
single-use receipt unlocks the target payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Context, Decimal, InvalidOperation, localcontext
import hashlib
import json
from pathlib import Path, PurePosixPath
import secrets
import threading
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch

from .coordinates import PhysicalPlane
from .observation import (
    TRAINING_LEDGER_SPLITS,
    AvailabilityObservationMeta,
    PatientSplitRegistry,
    SparseAvailabilityManifest,
)
from ..gaussians import AmplitudeGaugePolicy, GaussianBatch
from ..renderer import RENDERER_OUTPUT_SCHEMA_VERSION, RenderConfig, RenderResult, render_plane


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_hex(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 hexadecimal digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a 64-character SHA-256 hexadecimal digest") from error
    return value.lower()


_ASSIGNMENT_TOKEN = object()


@dataclass(frozen=True, init=False)
class EpisodeAssignment:
    """Canonical immutable context/target roles for one availability manifest."""

    episode_id: str
    manifest_hash: str
    patient_id: str
    context_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    assignment_hash: str
    _content_binding_hash: str = field(repr=False, compare=False)

    @classmethod
    def create(
        cls,
        manifest: SparseAvailabilityManifest,
        *,
        episode_id: str,
        patient_id: str,
        context_ids: Iterable[str],
        target_ids: Iterable[str],
    ) -> "EpisodeAssignment":
        if not isinstance(manifest, SparseAvailabilityManifest):
            raise TypeError("manifest must be a SparseAvailabilityManifest")
        context = cls._normalise_ids(context_ids, "context_ids")
        target = cls._normalise_ids(target_ids, "target_ids")
        if set(context) & set(target):
            raise ValueError("context_ids and target_ids must be disjoint")
        for observation_id in context + target:
            entry = manifest.metadata(observation_id)
            if entry.patient_id != patient_id:
                raise ValueError("every assigned observation must belong to patient_id")
        return cls(
            _token=_ASSIGNMENT_TOKEN,
            episode_id=episode_id,
            manifest_hash=manifest.manifest_hash,
            patient_id=patient_id,
            context_ids=context,
            target_ids=target,
            content_binding_hash=manifest._content_binding_hash,
        )

    def __init__(
        self,
        *,
        _token: object | None = None,
        episode_id: str,
        manifest_hash: str,
        patient_id: str,
        context_ids: Iterable[str],
        target_ids: Iterable[str],
        content_binding_hash: str,
    ) -> None:
        if _token is not _ASSIGNMENT_TOKEN:
            raise TypeError("EpisodeAssignment must be created with EpisodeAssignment.create(manifest, ...)")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("episode_id must be a non-empty string")
        if not isinstance(patient_id, str) or not patient_id:
            raise ValueError("patient_id must be a non-empty string")
        manifest_hash = _sha256_hex(manifest_hash, "manifest_hash")
        content_binding_hash = _sha256_hex(content_binding_hash, "content_binding_hash")
        context = self._normalise_ids(context_ids, "context_ids")
        target = self._normalise_ids(target_ids, "target_ids")
        if set(context) & set(target):
            raise ValueError("context_ids and target_ids must be disjoint")
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "manifest_hash", manifest_hash)
        object.__setattr__(self, "patient_id", patient_id)
        object.__setattr__(self, "context_ids", context)
        object.__setattr__(self, "target_ids", target)
        object.__setattr__(self, "_content_binding_hash", content_binding_hash)
        object.__setattr__(self, "assignment_hash", _hash(self.to_canonical_dict()))

    @staticmethod
    def _normalise_ids(values: Iterable[str], name: str) -> tuple[str, ...]:
        result = tuple(values)
        if any(not isinstance(value, str) or not value for value in result):
            raise ValueError(f"{name} must contain non-empty strings")
        if len(set(result)) != len(result):
            raise ValueError(f"{name} must not contain duplicate IDs")
        return tuple(sorted(result))

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "context_ids": list(self.context_ids),
            "episode_id": self.episode_id,
            "manifest_hash": self.manifest_hash,
            "patient_id": self.patient_id,
            "target_ids": list(self.target_ids),
        }


class _AvailabilityFileProvider:
    """Private bound provider; episode roles never affect payload legality."""

    def __init__(self, root: str | Path, manifest: SparseAvailabilityManifest) -> None:
        self._root = Path(root).resolve(strict=True)
        if not self._root.is_dir():
            raise NotADirectoryError(self._root)
        self._entries = {entry.observation_id: entry for entry in manifest.entries}
        self._digests = {entry.observation_id: manifest._expected_sha256(entry.observation_id) for entry in manifest.entries}

    def read_bytes(self, entry: AvailabilityObservationMeta) -> bytes:
        if self._entries.get(entry.observation_id) != entry:
            raise PermissionError("observation is not present in the bound sparse availability manifest")
        relative = PurePosixPath(entry.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise PermissionError("manifest path is not relative")
        candidate = (self._root / entry.relative_path).resolve(strict=True)
        try:
            candidate.relative_to(self._root)
        except ValueError as error:
            raise PermissionError("manifest path escapes provider root") from error
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        payload = candidate.read_bytes()
        if hashlib.sha256(payload).hexdigest() != self._digests[entry.observation_id]:
            raise OSError("observation payload does not match its manifest content digest")
        return payload


class _TargetCommitCapability:
    __slots__ = ("_ledger_nonce", "_target_id", "_state_version", "_secret")

    def __init__(self, ledger_nonce: object, target_id: str, state_version: str, secret: str) -> None:
        self._ledger_nonce = ledger_nonce
        self._target_id = target_id
        self._state_version = state_version
        self._secret = secret

    def __repr__(self) -> str:
        return "<TargetCommitCapability opaque>"


class _PredictionReceiptCapability:
    __slots__ = ("_ledger_nonce", "_target_id", "_secret")

    def __init__(self, ledger_nonce: object, target_id: str, secret: str) -> None:
        self._ledger_nonce = ledger_nonce
        self._target_id = target_id
        self._secret = secret

    def __repr__(self) -> str:
        return "<PredictionReceiptCapability opaque>"


TargetCommitCapability = _TargetCommitCapability
PredictionReceiptCapability = _PredictionReceiptCapability
_REGISTRATION_TOKEN = object()
_FROZEN_STATE_TOKEN = object()
_RENDER_EVIDENCE_TOKEN = object()


@dataclass(frozen=True, init=False)
class FrozenPatientState:
    """Factory-created binding to the exact live Phase-1 Gaussian state."""

    state_version: str
    _gaussians: GaussianBatch = field(repr=False, compare=False)
    _gaussian_digest: str = field(repr=False, compare=False)
    _ledger_nonce: object = field(repr=False, compare=False)
    _assignment_hash: str = field(repr=False, compare=False)
    _context_audit_hash: str = field(repr=False, compare=False)

    @classmethod
    def create(
        cls,
        *,
        ledger: "EpisodeLedger",
        gaussians: GaussianBatch,
        upstream_state_hash: str,
    ) -> "FrozenPatientState":
        return cls(
            _token=_FROZEN_STATE_TOKEN,
            ledger=ledger,
            gaussians=gaussians,
            upstream_state_hash=upstream_state_hash,
        )

    def __init__(
        self,
        *,
        _token: object | None = None,
        ledger: "EpisodeLedger",
        gaussians: GaussianBatch,
        upstream_state_hash: str,
    ) -> None:
        if _token is not _FROZEN_STATE_TOKEN:
            raise TypeError("FrozenPatientState must be created with FrozenPatientState.create(...)")
        if not isinstance(ledger, EpisodeLedger) or not isinstance(gaussians, GaussianBatch):
            raise TypeError("ledger and gaussians must use T0.5 contracts")
        gaussians.validate()
        if gaussians.gauge_policy is AmplitudeGaugePolicy.LEGACY_RAW:
            raise ValueError("Phase-1 frozen patient state rejects LEGACY_RAW GaussianBatch provenance")
        upstream_state_hash = _sha256_hex(upstream_state_hash, "upstream_state_hash")
        context_audit_hash = ledger._capture_context_audit_hash()
        gaussian_digest = gaussian_batch_digest(gaussians)
        state_payload = {
            "assignment_hash": ledger.assignment_hash,
            "context_audit_hash": context_audit_hash,
            "context_ids": list(ledger._assignment.context_ids),
            "gaussian_digest": gaussian_digest,
            "gauge_config_hash": gaussians.gauge_config_hash,
            "gauge_policy": gaussians.gauge_policy.value,
            "patient_id": ledger._assignment.patient_id,
            "upstream_state_hash": upstream_state_hash,
        }
        object.__setattr__(self, "state_version", _hash(state_payload))
        object.__setattr__(self, "_gaussians", gaussians)
        object.__setattr__(self, "_gaussian_digest", gaussian_digest)
        object.__setattr__(self, "_ledger_nonce", ledger._nonce)
        object.__setattr__(self, "_assignment_hash", ledger.assignment_hash)
        object.__setattr__(self, "_context_audit_hash", context_audit_hash)

    @property
    def gaussians(self) -> GaussianBatch:
        """Live state retained for differentiable rendering; do not detach it."""
        return self._gaussians

    def verify_live_gaussians(self) -> str:
        """Fail before render if a mutable live state diverged after freezing."""
        self._gaussians.validate()
        if self._gaussians.gauge_policy is AmplitudeGaugePolicy.LEGACY_RAW:
            raise ValueError("Phase-1 frozen patient state rejects LEGACY_RAW GaussianBatch provenance")
        current = gaussian_batch_digest(self._gaussians)
        if current != self._gaussian_digest:
            raise RuntimeError("live GaussianBatch changed after FrozenPatientState was created")
        return current

    def validate_for_commit(self, ledger: "EpisodeLedger", commit_capability: TargetCommitCapability) -> str:
        if self._ledger_nonce is not ledger._nonce or self._assignment_hash != ledger.assignment_hash:
            raise PermissionError("frozen state is not bound to this episode ledger")
        ledger._validate_commit(commit_capability)
        if commit_capability._state_version != self.state_version:
            raise PermissionError("committed state version does not match frozen state")
        if ledger._context_audit_hash() != self._context_audit_hash:
            raise RuntimeError("context audit changed after FrozenPatientState was created")
        return self.verify_live_gaussians()


class _RenderEvidence:
    """Controller-minted, in-process proof of one exact pure render call."""

    __slots__ = (
        "_token", "_ledger_nonce", "_commit_secret", "_state_version",
        "_plane_hash", "_renderer_version", "_gaussian_digest", "result",
        "_renderer_output_schema_version",
    )

    def __init__(
        self,
        *,
        _token: object,
        ledger_nonce: object,
        commit_secret: str,
        state_version: str,
        plane_hash: str,
        renderer_version: str,
        renderer_output_schema_version: str,
        gaussian_digest: str,
        result: RenderResult,
    ) -> None:
        if _token is not _RENDER_EVIDENCE_TOKEN:
            raise TypeError("render evidence is minted only by EpisodeController")
        self._token = _token
        self._ledger_nonce = ledger_nonce
        self._commit_secret = commit_secret
        self._state_version = state_version
        self._plane_hash = plane_hash
        self._renderer_version = renderer_version
        self._renderer_output_schema_version = renderer_output_schema_version
        self._gaussian_digest = gaussian_digest
        self.result = result


@dataclass(frozen=True)
class PredictionReceiptRecord:
    ledger_id: str
    episode_id: str
    assignment_hash: str
    target_id: str
    state_version: str
    plane_hash: str
    renderer_version: str
    renderer_output_schema_version: str
    gaussian_state_digest: str
    prediction_digest: str
    commit_sequence: int
    receipt_sequence: int

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "assignment_hash": self.assignment_hash,
            "commit_sequence": self.commit_sequence,
            "episode_id": self.episode_id,
            "ledger_id": self.ledger_id,
            "plane_hash": self.plane_hash,
            "prediction_digest": self.prediction_digest,
            "receipt_sequence": self.receipt_sequence,
            "renderer_version": self.renderer_version,
            "renderer_output_schema_version": self.renderer_output_schema_version,
            "gaussian_state_digest": self.gaussian_state_digest,
            "state_version": self.state_version,
            "target_id": self.target_id,
        }


class PredictionRegistration:
    """Registrar-owned evidence; direct construction is intentionally blocked."""

    __slots__ = ("record", "_token", "_commit_secret")

    def __init__(self, *, _token: object, record: PredictionReceiptRecord, commit_secret: str) -> None:
        if _token is not _REGISTRATION_TOKEN:
            raise TypeError("PredictionRegistration is owned by PredictionRegistrar")
        self.record = record
        self._token = _token
        self._commit_secret = commit_secret


@dataclass(frozen=True)
class EpisodeEvent:
    sequence: int
    event: str
    observation_id: str
    details: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(sorted(self.details.items()))))

    def to_canonical_dict(self) -> dict[str, Any]:
        return {"details": dict(self.details), "event": self.event, "observation_id": self.observation_id, "sequence": self.sequence}


@dataclass(frozen=True)
class EpisodeOpenedFileAudit:
    sequence: int
    observation_id: str
    relative_path: str
    role: str
    content_sha256: str

    def to_canonical_dict(self) -> dict[str, Any]:
        return {"content_sha256": self.content_sha256, "observation_id": self.observation_id, "relative_path": self.relative_path, "role": self.role, "sequence": self.sequence}


class EpisodeLedger:
    """No-budget ledger for a manifest-bound immutable training episode."""

    def __init__(
        self,
        manifest: SparseAvailabilityManifest,
        assignment: EpisodeAssignment,
        root: str | Path,
        *,
        split_registry: PatientSplitRegistry,
        deferred_target_readers: Mapping[str, Callable[[], bytes]] | None = None,
    ) -> None:
        if not isinstance(manifest, SparseAvailabilityManifest) or not isinstance(assignment, EpisodeAssignment) or not isinstance(split_registry, PatientSplitRegistry):
            raise TypeError("manifest, assignment, and split_registry must use T0.5 contracts")
        split_registry.assert_development_manifest(manifest)
        if assignment.manifest_hash != manifest.manifest_hash or assignment._content_binding_hash != manifest._content_binding_hash:
            raise ValueError("assignment is not bound to this exact sealed manifest")
        selected = assignment.context_ids + assignment.target_ids
        if not selected:
            raise ValueError("episode assignment must select at least one observation")
        for observation_id in selected:
            entry = manifest.metadata(observation_id)
            if entry.patient_id != assignment.patient_id:
                raise ValueError("assignment patient does not match manifest entry")
            if entry.split not in TRAINING_LEDGER_SPLITS:
                raise PermissionError(
                    "development EpisodeLedger rejects sealed audit and isolated evaluation cohorts"
                )
        self._manifest, self._assignment, self._split_registry = manifest, assignment, split_registry
        self._provider = _AvailabilityFileProvider(root, manifest)
        readers = dict(deferred_target_readers or {})
        if set(readers) - set(assignment.target_ids):
            raise ValueError("deferred target readers may bind only assigned target observations")
        if any(not callable(reader) for reader in readers.values()):
            raise TypeError("deferred target readers must be callable")
        self._deferred_target_readers = readers
        self._nonce, self._lock = object(), threading.RLock()
        self._ledger_id = _hash({
            "assignment_hash": assignment.assignment_hash,
            "sealed_manifest_binding": manifest._content_binding_hash,
            "manifest_hash": manifest.manifest_hash,
            "split_registry_hash": split_registry.registry_hash,
        })
        self._commits: dict[str, tuple[str, str, int]] = {}
        self._receipts: dict[str, tuple[str, PredictionReceiptRecord]] = {}
        self._prediction_records: list[PredictionReceiptRecord] = []
        self._revealed: set[str] = set()
        self._events: list[EpisodeEvent] = []
        self._audit: list[EpisodeOpenedFileAudit] = []

    @property
    def ledger_id(self) -> str:
        return self._ledger_id

    @property
    def manifest_hash(self) -> str:
        return self._manifest.manifest_hash

    @property
    def assignment_hash(self) -> str:
        return self._assignment.assignment_hash

    @property
    def event_records(self) -> tuple[EpisodeEvent, ...]:
        return tuple(self._events)

    @property
    def audit_records(self) -> tuple[EpisodeOpenedFileAudit, ...]:
        return tuple(self._audit)

    @property
    def prediction_records(self) -> tuple[PredictionReceiptRecord, ...]:
        """Immutable scientific receipt records, retained after target reveal."""
        return tuple(self._prediction_records)

    @property
    def audit_hash(self) -> str:
        return _hash({
            "assignment_hash": self.assignment_hash,
            "sealed_manifest_binding": self._manifest._content_binding_hash,
            "events": [event.to_canonical_dict() for event in self._events],
            "manifest_hash": self.manifest_hash,
            "split_registry_hash": self._split_registry.registry_hash,
            "opened_files": [row.to_canonical_dict() for row in self._audit],
            "prediction_records": [record.to_canonical_dict() for record in self._prediction_records],
        })

    def metadata(self, observation_id: str) -> AvailabilityObservationMeta:
        if observation_id not in self._assignment.context_ids + self._assignment.target_ids:
            raise PermissionError("metadata is available only for observations assigned to this episode")
        return self._manifest.metadata(observation_id)

    def expose_target_metadata(self, target_id: str) -> AvailabilityObservationMeta:
        if target_id not in self._assignment.target_ids:
            raise PermissionError("target metadata requires an assigned target")
        return self._manifest.metadata(target_id)

    def open_context(self, observation_id: str) -> bytes:
        with self._lock:
            if observation_id not in self._assignment.context_ids:
                raise PermissionError("only assigned context payloads may be opened before reveal")
            return self._open(self._manifest.metadata(observation_id), "CONTEXT", "OPEN_CONTEXT")

    def commit_target(self, target_id: str, state_version: str) -> TargetCommitCapability:
        state_version = _sha256_hex(state_version, "state_version")
        with self._lock:
            if target_id not in self._assignment.target_ids:
                raise PermissionError("only assigned targets may be committed")
            if target_id in self._commits or target_id in self._revealed:
                raise RuntimeError("target was already committed or revealed")
            secret = secrets.token_urlsafe(32)
            sequence = self._record_event("COMMIT_TARGET", target_id, {"state_version": state_version})
            self._commits[target_id] = (secret, state_version, sequence)
            return TargetCommitCapability(self._nonce, target_id, state_version, secret)

    def register_prediction_receipt(self, commit_capability: TargetCommitCapability, *, registration: PredictionRegistration) -> PredictionReceiptCapability:
        """Atomically record registrar-owned evidence and mint one receipt."""
        with self._lock:
            self._validate_commit(commit_capability)
            if not isinstance(registration, PredictionRegistration) or registration._token is not _REGISTRATION_TOKEN:
                raise PermissionError("only PredictionRegistrar may register prediction evidence")
            secret, state_version, commit_sequence = self._commits[commit_capability._target_id]
            record = registration.record
            expected_plane_hash = hashlib.sha256(self._manifest.metadata(commit_capability._target_id).plane.canonical_json().encode("utf-8")).hexdigest()
            if (
                registration._commit_secret != secret
                or record.ledger_id != self.ledger_id
                or record.episode_id != self._assignment.episode_id
                or record.assignment_hash != self.assignment_hash
                or record.target_id != commit_capability._target_id
                or record.state_version != state_version
                or record.plane_hash != expected_plane_hash
                or record.commit_sequence != commit_sequence
                or record.receipt_sequence != len(self._events)
                or not record.renderer_version
                or record.renderer_output_schema_version != RENDERER_OUTPUT_SCHEMA_VERSION
                or not _sha256_hex(record.gaussian_state_digest, "gaussian_state_digest")
                or not _sha256_hex(record.prediction_digest, "prediction_digest")
                or commit_capability._target_id in self._receipts
            ):
                raise PermissionError("prediction registration does not match committed target")
            receipt_secret = secrets.token_urlsafe(32)
            self._receipts[commit_capability._target_id] = (receipt_secret, record)
            self._prediction_records.append(record)
            record_hash = _hash(record.to_canonical_dict())
            self._record_event("REGISTER_PREDICTION", commit_capability._target_id, {
                "gaussian_state_digest": record.gaussian_state_digest,
                "prediction_digest": record.prediction_digest,
                "prediction_record_hash": record_hash,
                "renderer_output_schema_version": record.renderer_output_schema_version,
                "renderer_version": record.renderer_version,
                "state_version": state_version,
            })
            return PredictionReceiptCapability(self._nonce, commit_capability._target_id, receipt_secret)

    def reveal_target(self, target_id: str, receipt_capability: PredictionReceiptCapability) -> bytes:
        with self._lock:
            if target_id in self._revealed:
                raise PermissionError("target was already revealed")
            if not isinstance(receipt_capability, PredictionReceiptCapability) or receipt_capability._ledger_nonce is not self._nonce or receipt_capability._target_id != target_id:
                raise PermissionError("a matching receipt capability from this ledger is required")
            receipt = self._receipts.get(target_id)
            if receipt is None or receipt[0] != receipt_capability._secret:
                raise PermissionError("missing, invalid, or consumed prediction receipt")
            payload = self._open(self._manifest.metadata(target_id), "TARGET", "REVEAL_TARGET")
            del self._receipts[target_id]
            del self._commits[target_id]
            self._revealed.add(target_id)
            return payload

    def _validate_commit(self, capability: object) -> None:
        if not isinstance(capability, TargetCommitCapability) or capability._ledger_nonce is not self._nonce:
            raise PermissionError("a matching commit capability from this ledger is required")
        existing = self._commits.get(capability._target_id)
        if existing is None or existing[0] != capability._secret or existing[1] != capability._state_version:
            raise PermissionError("invalid or consumed commit capability")

    def _context_audit_hash(self) -> str:
        context_events = [event.to_canonical_dict() for event in self._events if event.event == "OPEN_CONTEXT"]
        context_rows = [row.to_canonical_dict() for row in self._audit if row.role == "CONTEXT"]
        return _hash({
            "assignment_hash": self.assignment_hash,
            "context_events": context_events,
            "context_opened_files": context_rows,
            "manifest_hash": self.manifest_hash,
        })

    def _capture_context_audit_hash(self) -> str:
        if self._commits or self._receipts or self._revealed or any(event.event != "OPEN_CONTEXT" for event in self._events):
            raise RuntimeError("FrozenPatientState must capture context-only audit before target commit")
        opened = [row.observation_id for row in self._audit if row.role == "CONTEXT"]
        if tuple(sorted(set(opened))) != self._assignment.context_ids:
            raise RuntimeError("all assigned context observations must be opened before freezing state")
        return self._context_audit_hash()

    def _record_event(self, event: str, observation_id: str, details: Mapping[str, str]) -> int:
        sequence = len(self._events)
        self._events.append(EpisodeEvent(sequence, event, observation_id, details))
        return sequence

    def _open(self, entry: AvailabilityObservationMeta, role: str, event: str) -> bytes:
        if role == "TARGET" and entry.observation_id in self._deferred_target_readers:
            # Product data preparation may bind a target reference without
            # materializing target intensity.  The callback is reachable only
            # from reveal_target, after prediction receipt registration.
            payload = self._deferred_target_readers[entry.observation_id]()
            if not isinstance(payload, bytes):
                raise TypeError("deferred target reader must return bytes")
        else:
            payload = self._provider.read_bytes(entry)
        self._record_event(event, entry.observation_id, {})
        self._audit.append(EpisodeOpenedFileAudit(len(self._audit), entry.observation_id, entry.relative_path, role, hashlib.sha256(payload).hexdigest()))
        return payload


def _tensor_digest_part(digest: "hashlib._Hash", name: str, tensor: torch.Tensor, *, boolean: bool = False) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"RenderResult.{name} must be a torch.Tensor")
    copy = tensor.detach().contiguous().cpu()
    if boolean:
        if copy.dtype is not torch.bool:
            raise TypeError(f"RenderResult.{name} must be bool")
        array = copy.to(torch.uint8).numpy()
        dtype_name = "bool-u8"
    else:
        if copy.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"RenderResult.{name} must be float32 or float64")
        # A detached CPU audit copy is deliberately canonicalized at the bit
        # level.  ``torch.where(..., nan)`` does not promise one quiet-NaN
        # payload across devices, whereas the digest must not vary with a NaN
        # payload produced by an otherwise equal unsupported render.
        array = copy.numpy().copy()
        nan_mask = np.isnan(array)
        if copy.dtype is torch.float32:
            bits = array.view(np.uint32)
            bits[nan_mask] = np.uint32(0x7FC00000)
        else:
            bits = array.view(np.uint64)
            bits[nan_mask] = np.uint64(0x7FF8000000000000)
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
        dtype_name = str(copy.dtype).replace("torch.", "")
    header = _canonical_json({"dtype": dtype_name, "name": name, "shape": list(copy.shape)}).encode("utf-8")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(np.ascontiguousarray(array).tobytes())


def prediction_digest_from_render_result(render_result: RenderResult, *, plane_hash: str, renderer_version: str) -> str:
    """Digest a detached audit copy without mutating or detaching live outputs."""
    if not isinstance(render_result, RenderResult):
        raise TypeError("render_result must be an actual RenderResult")
    digest = hashlib.sha256()
    digest.update(_canonical_json({"plane_hash": _sha256_hex(plane_hash, "plane_hash"), "renderer_version": renderer_version, "schema": RENDERER_OUTPUT_SCHEMA_VERSION}).encode("utf-8"))
    _tensor_digest_part(digest, "intensity", render_result.intensity)
    _tensor_digest_part(digest, "support_mass", render_result.support_mass)
    _tensor_digest_part(digest, "supported_psf_mass", render_result.supported_psf_mass)
    _tensor_digest_part(digest, "unsupported_mask", render_result.unsupported_mask, boolean=True)
    return digest.hexdigest()


def gaussian_batch_digest(gaussians: GaussianBatch) -> str:
    """Detached audit digest for a live validated runtime state, never its graph."""
    if not isinstance(gaussians, GaussianBatch):
        raise TypeError("gaussians must be a GaussianBatch")
    gaussians.validate()
    digest = hashlib.sha256()
    digest.update(_canonical_json({
        "covariance_epsilon": repr(gaussians.covariance_epsilon),
        "gauge_config_hash": gaussians.gauge_config_hash,
        "gauge_policy": gaussians.gauge_policy.value,
        "primitive_id": list(gaussians.primitive_id) if gaussians.primitive_id is not None else None,
        "primitive_kind": list(gaussians.primitive_kind) if gaussians.primitive_kind is not None else None,
        "schema": "gaussian-batch-v1",
    }).encode("utf-8"))
    _tensor_digest_part(digest, "centers_ras_mm", gaussians.centers_ras_mm)
    _tensor_digest_part(digest, "covariance_factor", gaussians.covariance_factor)
    _tensor_digest_part(digest, "log_support_amplitude", gaussians.log_support_amplitude)
    _tensor_digest_part(digest, "appearance", gaussians.appearance)
    _tensor_digest_part(digest, "appearance_valid", gaussians.appearance_valid, boolean=True)
    return digest.hexdigest()


class PredictionRegistrar:
    """Receipt factory separate from the pure renderer and ledger state."""

    def register_prediction_receipt(
        self,
        *,
        ledger: EpisodeLedger,
        commit_capability: TargetCommitCapability,
        frozen_state: FrozenPatientState,
        render_evidence: _RenderEvidence,
    ) -> PredictionReceiptCapability:
        if not isinstance(ledger, EpisodeLedger) or not isinstance(frozen_state, FrozenPatientState):
            raise TypeError("ledger and frozen_state must use T0.5 contracts")
        ledger._validate_commit(commit_capability)
        frozen_digest = frozen_state.validate_for_commit(ledger, commit_capability)
        if not isinstance(render_evidence, _RenderEvidence) or render_evidence._token is not _RENDER_EVIDENCE_TOKEN:
            raise PermissionError("PredictionRegistrar rejects bare or hand-built RenderResult evidence")
        if (
            render_evidence._ledger_nonce is not ledger._nonce
            or render_evidence._commit_secret != commit_capability._secret
            or render_evidence._state_version != frozen_state.state_version
            or render_evidence._gaussian_digest != frozen_digest
        ):
            raise PermissionError("render evidence is not bound to this committed live state")
        target = ledger.expose_target_metadata(commit_capability._target_id)
        plane_hash = hashlib.sha256(target.plane.canonical_json().encode("utf-8")).hexdigest()
        if (
            render_evidence._plane_hash != plane_hash
            or not render_evidence._renderer_version
            or not render_evidence._renderer_output_schema_version
        ):
            raise PermissionError("render evidence plane or controlled renderer configuration mismatches target")
        renderer_version = render_evidence._renderer_version
        prediction_digest = prediction_digest_from_render_result(render_evidence.result, plane_hash=plane_hash, renderer_version=renderer_version)
        # Registration sequence is the next ledger event, before atomic insertion.
        commit_sequence = ledger._commits[commit_capability._target_id][2]
        record = PredictionReceiptRecord(
            ledger.ledger_id,
            ledger._assignment.episode_id,
            ledger.assignment_hash,
            commit_capability._target_id,
            frozen_state.state_version,
            plane_hash,
            renderer_version,
            render_evidence._renderer_output_schema_version,
            frozen_digest,
            prediction_digest,
            commit_sequence,
            len(ledger._events),
        )
        registration = PredictionRegistration(_token=_REGISTRATION_TOKEN, record=record, commit_secret=commit_capability._secret)
        return ledger.register_prediction_receipt(commit_capability, registration=registration)


class EpisodeController:
    """Only T0.5 path that calls pure rendering for a committed target."""

    def __init__(self, registrar: PredictionRegistrar | None = None) -> None:
        self._registrar = registrar or PredictionRegistrar()

    def render_and_register(
        self,
        *,
        ledger: EpisodeLedger,
        commit_capability: TargetCommitCapability,
        frozen_state: FrozenPatientState,
        appearance_channel: int = 0,
        render_config: RenderConfig | None = None,
    ) -> tuple[RenderResult, PredictionReceiptCapability]:
        config = render_config or RenderConfig()
        if not isinstance(config, RenderConfig):
            raise TypeError("render_config must be a RenderConfig before controller rendering begins")
        if not isinstance(ledger, EpisodeLedger) or not isinstance(frozen_state, FrozenPatientState):
            raise TypeError("ledger and frozen_state must use T0.5 contracts")
        # Validate capability and frozen-state ledger binding before exposing
        # even target geometry, then before the pure render is entered.
        gaussian_digest = frozen_state.validate_for_commit(ledger, commit_capability)
        plane = ledger.expose_target_metadata(commit_capability._target_id).plane
        # `render_plane` itself remains pure.  The controller records the
        # resulting evidence only after the tensor operation returns.
        result = render_plane(frozen_state.gaussians, plane, appearance_channel=appearance_channel, config=config)
        plane_hash = hashlib.sha256(plane.canonical_json().encode("utf-8")).hexdigest()
        evidence = _RenderEvidence(
            _token=_RENDER_EVIDENCE_TOKEN,
            ledger_nonce=ledger._nonce,
            commit_secret=commit_capability._secret,
            state_version=frozen_state.state_version,
            plane_hash=plane_hash,
            renderer_version=config.renderer_version,
            renderer_output_schema_version=config.renderer_output_schema_version,
            gaussian_digest=gaussian_digest,
            result=result,
        )
        receipt = self._registrar.register_prediction_receipt(
            ledger=ledger,
            commit_capability=commit_capability,
            frozen_state=frozen_state,
            render_evidence=evidence,
        )
        return result, receipt


def _canonical_decimal(value: str | Decimal) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float) or not isinstance(value, (str, Decimal)):
        raise TypeError("cost amounts must be Decimal or canonical decimal strings, never float")
    try:
        decimal = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        raise ValueError("cost amount must be a finite decimal") from error
    if not decimal.is_finite() or decimal < 0:
        raise ValueError("cost amount must be finite and non-negative")
    return decimal


def _decimal_string(value: Decimal) -> str:
    """Canonical finite decimal spelling without consulting ambient context."""
    if not value.is_finite():
        raise ValueError("cost amount must be finite")
    sign, digits_tuple, exponent = value.as_tuple()
    digits = "".join(str(digit) for digit in digits_tuple) or "0"
    if not any(digit != "0" for digit in digits):
        return "0"
    # Strip insignificant fractional trailing zeros by changing the exponent,
    # rather than Decimal.normalize(), whose behaviour depends on context.
    while exponent < 0 and digits.endswith("0"):
        digits = digits[:-1]
        exponent += 1
    if exponent >= 0:
        body = digits + ("0" * exponent)
    else:
        point = len(digits) + exponent
        body = ("0." + ("0" * (-point)) + digits) if point <= 0 else (digits[:point] + "." + digits[point:])
    return ("-" if sign else "") + body


def _decimal_precision(*values: Decimal) -> int:
    """Enough significant digits for exact add/subtract of all finite inputs."""
    # The lowest represented place can be far below the largest adjusted
    # exponent (for example ``1E+999 + 1``).  Cover that full span instead of
    # relying on the process-global Decimal context.
    highest = max(value.adjusted() for value in values)
    lowest = min(value.as_tuple().exponent for value in values)
    return max(1, highest - lowest + 2)


def _decimal_add(left: Decimal, right: Decimal) -> Decimal:
    with localcontext(Context(prec=_decimal_precision(left, right))):
        return left + right


def _decimal_subtract(left: Decimal, right: Decimal) -> Decimal:
    with localcontext(Context(prec=_decimal_precision(left, right))):
        return left - right


@dataclass(frozen=True)
class AcquisitionCostEntry:
    cost_key: str
    canonical_amount: str

    def __post_init__(self) -> None:
        if not isinstance(self.cost_key, str) or not self.cost_key:
            raise ValueError("cost_key must be a non-empty string")
        amount = _canonical_decimal(self.canonical_amount)
        canonical = _decimal_string(amount)
        if self.canonical_amount != canonical:
            raise ValueError("canonical_amount must use canonical Decimal spelling")


@dataclass(frozen=True)
class AcquisitionCostSchedule:
    schedule_id: str
    entries: tuple[AcquisitionCostEntry, ...]
    schedule_hash: str = field(init=False)

    @classmethod
    def create(cls, *, schedule_id: str, amounts: Mapping[str, str | Decimal]) -> "AcquisitionCostSchedule":
        if not isinstance(amounts, Mapping) or not amounts:
            raise ValueError("amounts must be a non-empty mapping")
        entries = tuple(
            AcquisitionCostEntry(key, _decimal_string(_canonical_decimal(amount)))
            for key, amount in sorted(amounts.items())
        )
        return cls(schedule_id, entries)

    def __post_init__(self) -> None:
        if not isinstance(self.schedule_id, str) or not self.schedule_id:
            raise ValueError("schedule_id must be a non-empty string")
        entries = tuple(self.entries)
        if not entries or any(not isinstance(entry, AcquisitionCostEntry) for entry in entries):
            raise ValueError("entries must be non-empty AcquisitionCostEntry records")
        if tuple(sorted(entry.cost_key for entry in entries)) != tuple(entry.cost_key for entry in entries) or len({entry.cost_key for entry in entries}) != len(entries):
            raise ValueError("schedule entries must have unique sorted cost keys")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "schedule_hash", _hash({"entries": [{"canonical_amount": entry.canonical_amount, "cost_key": entry.cost_key} for entry in entries], "schedule_id": self.schedule_id}))

    def amount(self, cost_key: str) -> Decimal:
        for entry in self.entries:
            if entry.cost_key == cost_key:
                return _canonical_decimal(entry.canonical_amount)
        raise KeyError(f"unknown acquisition cost key: {cost_key}")


class _AcquisitionCapability:
    __slots__ = ("observation_id", "_ledger_nonce", "_secret")
    def __init__(self, observation_id: str, ledger_nonce: object, secret: str) -> None:
        self.observation_id, self._ledger_nonce, self._secret = observation_id, ledger_nonce, secret
    def __repr__(self) -> str:
        return "<AcquisitionCapability opaque>"


AcquisitionCapability = _AcquisitionCapability


@dataclass(frozen=True)
class AcquisitionEvent:
    sequence: int
    event: str
    observation_id: str
    cost_key: str
    amount: str
    budget_before: str
    budget_after: str
    schedule_hash: str

    def to_canonical_dict(self) -> dict[str, str | int]:
        return {"amount": self.amount, "budget_after": self.budget_after, "budget_before": self.budget_before, "cost_key": self.cost_key, "event": self.event, "observation_id": self.observation_id, "schedule_hash": self.schedule_hash, "sequence": self.sequence}


class DeploymentAcquisitionLedger:
    """Deployment-only exact-Decimal acquisition accounting, separate from episodes."""

    def __init__(self, *, manifest: SparseAvailabilityManifest, budget: Decimal, schedule: AcquisitionCostSchedule) -> None:
        if not isinstance(manifest, SparseAvailabilityManifest) or not isinstance(schedule, AcquisitionCostSchedule):
            raise TypeError("manifest and schedule must use T0.5 contracts")
        if not isinstance(budget, Decimal):
            raise TypeError("budget must be Decimal; float budgets are prohibited")
        if not budget.is_finite() or budget < 0:
            raise ValueError("budget must be finite and non-negative")
        self._manifest, self._schedule, self._budget = manifest, schedule, budget
        self._spent = Decimal("0")
        self._committed: set[str] = set()
        self._events: list[AcquisitionEvent] = []
        self._nonce = object()
        self._lock = threading.RLock()

    @property
    def spent(self) -> Decimal:
        return self._spent

    @property
    def remaining_budget(self) -> Decimal:
        return _decimal_subtract(self._budget, self._spent)

    @property
    def event_records(self) -> tuple[AcquisitionEvent, ...]:
        return tuple(self._events)

    @property
    def ledger_hash(self) -> str:
        return _hash({"events": [event.to_canonical_dict() for event in self._events], "manifest_hash": self._manifest.manifest_hash, "schedule_hash": self._schedule.schedule_hash})

    def commit_bootstrap(self, observation_id: str) -> AcquisitionCapability:
        return self._commit(observation_id, "COMMIT_BOOTSTRAP")

    def commit_observation(self, observation_id: str) -> AcquisitionCapability:
        return self._commit(observation_id, "COMMIT_OBSERVATION")

    def _commit(self, observation_id: str, event: str) -> AcquisitionCapability:
        with self._lock:
            if observation_id in self._committed:
                raise RuntimeError("observation was already charged")
            entry = self._manifest.metadata(observation_id)
            if entry.acquisition_cost_key is None:
                raise ValueError("availability entry has no deployment acquisition_cost_key")
            amount = self._schedule.amount(entry.acquisition_cost_key)
            after = _decimal_add(self._spent, amount)
            if after > self._budget:
                raise RuntimeError("acquisition commitment would exceed deployment budget")
            before = self._spent
            self._spent = after
            self._committed.add(observation_id)
            self._events.append(AcquisitionEvent(len(self._events), event, observation_id, entry.acquisition_cost_key, _decimal_string(amount), _decimal_string(before), _decimal_string(after), self._schedule.schedule_hash))
            return AcquisitionCapability(observation_id, self._nonce, secrets.token_urlsafe(32))
