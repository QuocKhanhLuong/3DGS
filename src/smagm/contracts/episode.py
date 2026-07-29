"""T0.5 immutable episode, receipt, and deployment-cost contracts.

The reference renderer remains a pure tensor function.  This module instead
records the scientific ordering around it: a target is committed against a
frozen pre-reveal state, a registrar digests the actual rendered result, and a
single-use receipt unlocks the target payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path, PurePosixPath
import secrets
import threading
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .coordinates import PhysicalPlane
from .observation import AvailabilityObservationMeta, SparseAvailabilityManifest
from ..renderer import RenderConfig, RenderResult, render_plane


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
            content_binding_hash=manifest.content_binding_hash,
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

    @property
    def content_binding_hash(self) -> str:
        """Private-manifest aggregate bound at creation, excluded from public hash."""
        return self._content_binding_hash

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


@dataclass(frozen=True)
class FrozenPatientState:
    """Opaque state binding frozen before a target commit."""

    state_version: str
    state_binding: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_version", _sha256_hex(self.state_version, "state_version"))
        if not isinstance(self.state_binding, str):
            raise TypeError("state_binding must be a string")


@dataclass(frozen=True)
class PredictionReceiptRecord:
    ledger_id: str
    episode_id: str
    assignment_hash: str
    target_id: str
    state_version: str
    plane_hash: str
    renderer_version: str
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

    def __init__(self, manifest: SparseAvailabilityManifest, assignment: EpisodeAssignment, root: str | Path) -> None:
        if not isinstance(manifest, SparseAvailabilityManifest) or not isinstance(assignment, EpisodeAssignment):
            raise TypeError("manifest and assignment must use T0.5 contracts")
        if assignment.manifest_hash != manifest.manifest_hash or assignment.content_binding_hash != manifest.content_binding_hash:
            raise ValueError("assignment is not bound to this exact sealed manifest")
        selected = assignment.context_ids + assignment.target_ids
        if not selected:
            raise ValueError("episode assignment must select at least one observation")
        for observation_id in selected:
            if manifest.metadata(observation_id).patient_id != assignment.patient_id:
                raise ValueError("assignment patient does not match manifest entry")
        self._manifest, self._assignment = manifest, assignment
        self._provider = _AvailabilityFileProvider(root, manifest)
        self._nonce, self._lock = object(), threading.RLock()
        self._ledger_id = _hash({
            "assignment_hash": assignment.assignment_hash,
            "content_binding_hash": manifest.content_binding_hash,
            "manifest_hash": manifest.manifest_hash,
        })
        self._commits: dict[str, tuple[str, str, int]] = {}
        self._receipts: dict[str, tuple[str, PredictionReceiptRecord]] = {}
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
    def audit_hash(self) -> str:
        return _hash({
            "assignment_hash": self.assignment_hash,
            "content_binding_hash": self._manifest.content_binding_hash,
            "events": [event.to_canonical_dict() for event in self._events],
            "manifest_hash": self.manifest_hash,
            "opened_files": [row.to_canonical_dict() for row in self._audit],
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
                or not record.renderer_version
                or not _sha256_hex(record.prediction_digest, "prediction_digest")
                or commit_capability._target_id in self._receipts
            ):
                raise PermissionError("prediction registration does not match committed target")
            receipt_secret = secrets.token_urlsafe(32)
            self._receipts[commit_capability._target_id] = (receipt_secret, record)
            self._record_event("REGISTER_PREDICTION", commit_capability._target_id, {"prediction_digest": record.prediction_digest, "renderer_version": record.renderer_version, "state_version": state_version})
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

    def _record_event(self, event: str, observation_id: str, details: Mapping[str, str]) -> int:
        sequence = len(self._events)
        self._events.append(EpisodeEvent(sequence, event, observation_id, details))
        return sequence

    def _open(self, entry: AvailabilityObservationMeta, role: str, event: str) -> bytes:
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
    digest.update(_canonical_json({"plane_hash": _sha256_hex(plane_hash, "plane_hash"), "renderer_version": renderer_version, "schema": "render-result-v1"}).encode("utf-8"))
    _tensor_digest_part(digest, "intensity", render_result.intensity)
    _tensor_digest_part(digest, "support_mass", render_result.support_mass)
    _tensor_digest_part(digest, "supported_psf_mass", render_result.supported_psf_mass)
    _tensor_digest_part(digest, "unsupported_mask", render_result.unsupported_mask, boolean=True)
    return digest.hexdigest()


class PredictionRegistrar:
    """Receipt factory separate from the pure renderer and ledger state."""

    def register_prediction_receipt(
        self,
        *,
        ledger: EpisodeLedger,
        commit_capability: TargetCommitCapability,
        frozen_state: FrozenPatientState,
        render_result: RenderResult,
        render_config: RenderConfig,
    ) -> PredictionReceiptCapability:
        if not isinstance(ledger, EpisodeLedger) or not isinstance(frozen_state, FrozenPatientState) or not isinstance(render_config, RenderConfig):
            raise TypeError("ledger, frozen_state, and render_config must use T0.5 contracts")
        ledger._validate_commit(commit_capability)
        if frozen_state.state_version != commit_capability._state_version:
            raise PermissionError("frozen state version does not match committed state")
        target = ledger.expose_target_metadata(commit_capability._target_id)
        plane_hash = hashlib.sha256(target.plane.canonical_json().encode("utf-8")).hexdigest()
        renderer_version = render_config.renderer_version
        prediction_digest = prediction_digest_from_render_result(render_result, plane_hash=plane_hash, renderer_version=renderer_version)
        # Registration sequence is the next ledger event, before atomic insertion.
        commit_sequence = ledger._commits[commit_capability._target_id][2]
        record = PredictionReceiptRecord(ledger.ledger_id, ledger._assignment.episode_id, ledger.assignment_hash, commit_capability._target_id, frozen_state.state_version, plane_hash, renderer_version, prediction_digest, commit_sequence, len(ledger._events))
        registration = PredictionRegistration(_token=_REGISTRATION_TOKEN, record=record, commit_secret=commit_capability._secret)
        return ledger.register_prediction_receipt(commit_capability, registration=registration)


class EpisodeController:
    """Convenience controller that calls pure rendering then separately registers it."""

    def __init__(self, registrar: PredictionRegistrar | None = None) -> None:
        self._registrar = registrar or PredictionRegistrar()

    def render_and_register(
        self,
        *,
        ledger: EpisodeLedger,
        commit_capability: TargetCommitCapability,
        frozen_state: FrozenPatientState,
        gaussians: Any,
        appearance_channel: int = 0,
        render_config: RenderConfig | None = None,
    ) -> tuple[RenderResult, PredictionReceiptCapability]:
        config = render_config or RenderConfig()
        plane = ledger.expose_target_metadata(commit_capability._target_id).plane
        result = render_plane(gaussians, plane, appearance_channel=appearance_channel, config=config)
        receipt = self._registrar.register_prediction_receipt(ledger=ledger, commit_capability=commit_capability, frozen_state=frozen_state, render_result=result, render_config=config)
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
    value = value.normalize()
    return "0" if value == 0 else format(value, "f")


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

    @property
    def spent(self) -> Decimal:
        return self._spent

    @property
    def remaining_budget(self) -> Decimal:
        return self._budget - self._spent

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
        if observation_id in self._committed:
            raise RuntimeError("observation was already charged")
        entry = self._manifest.metadata(observation_id)
        if entry.acquisition_cost_key is None:
            raise ValueError("availability entry has no deployment acquisition_cost_key")
        amount = self._schedule.amount(entry.acquisition_cost_key)
        after = self._spent + amount
        if after > self._budget:
            raise RuntimeError("acquisition commitment would exceed deployment budget")
        before = self._spent
        self._spent = after
        self._committed.add(observation_id)
        self._events.append(AcquisitionEvent(len(self._events), event, observation_id, entry.acquisition_cost_key, _decimal_string(amount), _decimal_string(before), _decimal_string(after), self._schedule.schedule_hash))
        return AcquisitionCapability(observation_id, self._nonce, secrets.token_urlsafe(32))
