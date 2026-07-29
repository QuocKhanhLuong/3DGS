"""Immutable manifest and capability-based observation access contracts.

This is an in-process research boundary, not a security sandbox.  Its purpose
is to make legal access explicit, testable, and auditable: metadata is visible
up front, context payloads are readable, and target payloads require a prior
commit plus an opaque capability issued by this ledger instance.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from decimal import Decimal
from enum import Enum
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import secrets
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .coordinates import PhysicalPlane


class AccessLevel(str, Enum):
    CONTEXT = "CONTEXT"
    TARGET = "TARGET"


def _sha256(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("content_sha256 must be a 64-character hexadecimal digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError("content_sha256 must be a 64-character hexadecimal digest") from error
    return value.lower()


def _relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ValueError("relative_path must be a non-empty manifest-relative path")
    return path.as_posix()


@dataclass(frozen=True)
class ObservationMeta:
    """Immutable public metadata for one registered physical observation."""

    observation_id: str
    patient_id: str
    split: str
    relative_path: str
    access_level: AccessLevel
    modality_id: str
    plane: PhysicalPlane
    is_synthetic: bool = False
    cost: float = 1.0

    def __post_init__(self) -> None:
        if not all(isinstance(item, str) and item for item in (self.observation_id, self.patient_id, self.split, self.modality_id)):
            raise ValueError("observation_id, patient_id, split, and modality_id must be non-empty strings")
        if not isinstance(self.plane, PhysicalPlane):
            raise TypeError("plane must be a PhysicalPlane")
        if self.plane.observation_id not in (None, self.observation_id):
            raise ValueError("plane observation_id must agree with observation metadata")
        if not isinstance(self.is_synthetic, bool):
            raise TypeError("is_synthetic must be a bool")
        if self.plane.source_transform is None and not self.is_synthetic:
            raise ValueError("non-synthetic observations require source-affine provenance")
        if isinstance(self.cost, bool) or not isinstance(self.cost, (int, float)) or not math.isfinite(float(self.cost)) or not self.cost > 0:
            raise ValueError("cost must be positive and finite")
        object.__setattr__(self, "relative_path", _relative_path(self.relative_path))
        object.__setattr__(self, "access_level", AccessLevel(self.access_level))
        object.__setattr__(self, "cost", float(self.cost))

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "access_level": self.access_level.value, "cost": self.cost,
            "is_synthetic": self.is_synthetic,
            "modality_id": self.modality_id, "observation_id": self.observation_id,
            "patient_id": self.patient_id, "plane": self.plane.to_canonical_dict(),
            "relative_path": self.relative_path, "split": self.split,
        }


@dataclass(frozen=True)
class SparseManifest:
    """Canonical, immutable sparse-observation manifest with patient splits."""

    entries: tuple[ObservationMeta, ...] | Iterable[ObservationMeta]
    manifest_id: str = ""
    integrity_digests: InitVar[Mapping[str, str] | None] = None
    _integrity_digests: Mapping[str, str] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self, integrity_digests: Mapping[str, str] | None) -> None:
        entries = tuple(self.entries)
        if not entries or any(not isinstance(entry, ObservationMeta) for entry in entries):
            raise ValueError("entries must be a non-empty sequence of ObservationMeta")
        identifiers = [entry.observation_id for entry in entries]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("observation_id values must be unique")
        paths = [entry.relative_path for entry in entries]
        if len(set(paths)) != len(paths):
            raise ValueError("each manifest-relative path must be unique")
        patient_splits: dict[str, str] = {}
        for entry in entries:
            previous = patient_splits.setdefault(entry.patient_id, entry.split)
            if previous != entry.split:
                raise ValueError("all observations for a patient must share one split")
        object.__setattr__(self, "entries", tuple(sorted(entries, key=lambda entry: entry.observation_id)))
        if self.manifest_id and not isinstance(self.manifest_id, str):
            raise TypeError("manifest_id must be a string")
        if integrity_digests is None or set(integrity_digests) != set(identifiers):
            raise ValueError("integrity_digests must bind every observation_id exactly once")
        normalized_digests = {
            observation_id: _sha256(integrity_digests[observation_id])
            for observation_id in identifiers
        }
        object.__setattr__(
            self,
            "_integrity_digests",
            MappingProxyType(normalized_digests),
        )

    @property
    def canonical_hash(self) -> str:
        encoded = self.canonical_json().encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def metadata(self, observation_id: str) -> ObservationMeta:
        for entry in self.entries:
            if entry.observation_id == observation_id:
                return entry
        raise KeyError(f"unknown observation_id: {observation_id}")

    def to_canonical_dict(self) -> dict[str, Any]:
        return {"entries": [entry.to_canonical_dict() for entry in self.entries], "manifest_id": self.manifest_id}

    def canonical_json(self) -> str:
        return json.dumps(self.to_canonical_dict(), sort_keys=True, separators=(",", ":"))

    def _expected_sha256(self, observation_id: str) -> str:
        return self._integrity_digests[observation_id]


@dataclass(frozen=True)
class AvailabilityObservationMeta:
    """Permanent sparse availability metadata, deliberately without a role.

    ``AccessLevel`` remains on :class:`ObservationMeta` only for T0 migration.
    Phase-1 code must use this record together with an ``EpisodeAssignment``.
    """

    observation_id: str
    patient_id: str
    split: str
    relative_path: str
    modality_id: str
    plane: PhysicalPlane
    is_synthetic: bool = False
    acquisition_cost_key: str | None = None
    registration_record_id: str | None = None

    def __post_init__(self) -> None:
        if not all(
            isinstance(item, str) and item
            for item in (self.observation_id, self.patient_id, self.split, self.modality_id)
        ):
            raise ValueError("observation_id, patient_id, split, and modality_id must be non-empty strings")
        if not isinstance(self.plane, PhysicalPlane):
            raise TypeError("plane must be a PhysicalPlane")
        if self.plane.observation_id not in (None, self.observation_id):
            raise ValueError("plane observation_id must agree with observation metadata")
        if not isinstance(self.is_synthetic, bool):
            raise TypeError("is_synthetic must be a bool")
        if self.plane.source_transform is None and not self.is_synthetic:
            raise ValueError("non-synthetic observations require source-affine provenance")
        if self.acquisition_cost_key is not None and (
            not isinstance(self.acquisition_cost_key, str) or not self.acquisition_cost_key
        ):
            raise ValueError("acquisition_cost_key must be None or a non-empty string")
        if self.registration_record_id is not None and (
            not isinstance(self.registration_record_id, str) or not self.registration_record_id
        ):
            raise ValueError("registration_record_id must be None or a non-empty string")
        object.__setattr__(self, "relative_path", _relative_path(self.relative_path))

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "acquisition_cost_key": self.acquisition_cost_key,
            "is_synthetic": self.is_synthetic,
            "modality_id": self.modality_id,
            "observation_id": self.observation_id,
            "patient_id": self.patient_id,
            "plane": self.plane.to_canonical_dict(),
            "registration_record_id": self.registration_record_id,
            "relative_path": self.relative_path,
            "split": self.split,
        }


@dataclass(frozen=True)
class SparseAvailabilityManifest:
    """Immutable, manifest-bound sparse observations with no episode roles."""

    entries: tuple[AvailabilityObservationMeta, ...] | Iterable[AvailabilityObservationMeta]
    manifest_id: str = ""
    integrity_digests: InitVar[Mapping[str, str] | None] = None
    _integrity_digests: Mapping[str, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self, integrity_digests: Mapping[str, str] | None) -> None:
        entries = tuple(self.entries)
        if not entries or any(not isinstance(entry, AvailabilityObservationMeta) for entry in entries):
            raise ValueError("entries must be a non-empty sequence of AvailabilityObservationMeta")
        identifiers = [entry.observation_id for entry in entries]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("observation_id values must be unique")
        paths = [entry.relative_path for entry in entries]
        if len(set(paths)) != len(paths):
            raise ValueError("each manifest-relative path must be unique")
        patient_splits: dict[str, str] = {}
        for entry in entries:
            previous = patient_splits.setdefault(entry.patient_id, entry.split)
            if previous != entry.split:
                raise ValueError("all observations for a patient must share one split")
        if self.manifest_id and not isinstance(self.manifest_id, str):
            raise TypeError("manifest_id must be a string")
        if integrity_digests is None or set(integrity_digests) != set(identifiers):
            raise ValueError("integrity_digests must bind every observation_id exactly once")
        normalized = {observation_id: _sha256(integrity_digests[observation_id]) for observation_id in identifiers}
        object.__setattr__(self, "entries", tuple(sorted(entries, key=lambda entry: entry.observation_id)))
        object.__setattr__(self, "_integrity_digests", MappingProxyType(normalized))

    @property
    def canonical_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def manifest_hash(self) -> str:
        return self.canonical_hash

    @property
    def content_binding_hash(self) -> str:
        """Sealed payload binding, distinct from the public availability hash.

        The individual integrity digests remain private to the bound provider;
        this aggregate only lets an episode reject a same-metadata manifest
        whose sealed payloads differ.
        """
        payload = {
            "entries": [
                {"content_sha256": self._integrity_digests[observation_id], "observation_id": observation_id}
                for observation_id in sorted(self._integrity_digests)
            ],
            "manifest_hash": self.manifest_hash,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def metadata(self, observation_id: str) -> AvailabilityObservationMeta:
        for entry in self.entries:
            if entry.observation_id == observation_id:
                return entry
        raise KeyError(f"unknown observation_id: {observation_id}")

    def to_canonical_dict(self) -> dict[str, Any]:
        return {"entries": [entry.to_canonical_dict() for entry in self.entries], "manifest_id": self.manifest_id}

    def canonical_json(self) -> str:
        return json.dumps(self.to_canonical_dict(), sort_keys=True, separators=(",", ":"))

    def _expected_sha256(self, observation_id: str) -> str:
        return self._integrity_digests[observation_id]


def validate_patient_split_manifests(manifests: Iterable[SparseManifest]) -> str:
    """Validate patient grouping across manifests and hash the split map."""

    resolved = tuple(manifests)
    if not resolved or any(not isinstance(manifest, SparseManifest) for manifest in resolved):
        raise ValueError("manifests must contain at least one SparseManifest")
    patient_splits: dict[str, str] = {}
    for manifest in resolved:
        for entry in manifest.entries:
            previous = patient_splits.setdefault(entry.patient_id, entry.split)
            if previous != entry.split:
                raise ValueError("a patient cannot appear in different splits across manifests")
    payload = json.dumps(patient_splits, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _ManifestFileProvider:
    """Private payload provider bound to one manifest and data root."""

    def __init__(self, root: str | Path, manifest: SparseManifest) -> None:
        if not isinstance(manifest, SparseManifest):
            raise TypeError("manifest must be a SparseManifest")
        self._root = Path(root).resolve(strict=True)
        if not self._root.is_dir():
            raise NotADirectoryError(self._root)
        self._manifest_hash = manifest.canonical_hash
        self._entries = {entry.observation_id: entry for entry in manifest.entries}
        self._digests = {
            entry.observation_id: manifest._expected_sha256(entry.observation_id)
            for entry in manifest.entries
        }

    @property
    def manifest_hash(self) -> str:
        return self._manifest_hash

    def read_bytes(self, entry: ObservationMeta) -> bytes:
        bound = self._entries.get(entry.observation_id)
        if bound != entry:
            raise PermissionError("observation is not present in the bound sparse manifest")
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


class _RevealCapability:
    """Unforgeable-by-convention token; its secret is never exposed in repr."""

    __slots__ = ("_ledger_nonce", "_observation_id", "_secret")

    def __init__(self, ledger_nonce: object, observation_id: str, secret: str) -> None:
        self._ledger_nonce = ledger_nonce
        self._observation_id = observation_id
        self._secret = secret

    def __repr__(self) -> str:
        return "<RevealCapability opaque>"


@dataclass(frozen=True)
class OpenedFileAudit:
    sequence: int
    observation_id: str
    relative_path: str
    access_level: AccessLevel
    content_sha256: str

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "access_level": self.access_level.value,
            "content_sha256": self.content_sha256,
            "observation_id": self.observation_id,
            "relative_path": self.relative_path,
            "sequence": self.sequence,
        }


@dataclass(frozen=True)
class LedgerEvent:
    sequence: int
    event: str
    observation_id: str

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "observation_id": self.observation_id,
            "sequence": self.sequence,
        }


class ObservationLedger:
    """Manifest-governed context and target access with deterministic audit rows."""

    def __init__(
        self,
        manifest: SparseManifest,
        root: str | Path,
        *,
        target_budget: float | None = None,
    ) -> None:
        if not isinstance(manifest, SparseManifest):
            raise TypeError("manifest must be a SparseManifest")
        if target_budget is not None and (
            isinstance(target_budget, bool)
            or not isinstance(target_budget, (int, float))
            or not math.isfinite(float(target_budget))
            or target_budget <= 0.0
        ):
            raise ValueError("target_budget must be None or a positive finite value")
        self._manifest = manifest
        self._provider = _ManifestFileProvider(root, manifest)
        self._target_budget_decimal = (
            None if target_budget is None else Decimal(str(target_budget))
        )
        self._committed_cost = Decimal("0")
        self._nonce = object()
        self._commits: dict[str, str] = {}
        self._committed_ids: set[str] = set()
        self._audit: list[OpenedFileAudit] = []
        self._events: list[LedgerEvent] = []

    @property
    def manifest_hash(self) -> str:
        return self._manifest.canonical_hash

    @property
    def audit_records(self) -> tuple[OpenedFileAudit, ...]:
        return tuple(self._audit)

    @property
    def event_records(self) -> tuple[LedgerEvent, ...]:
        return tuple(self._events)

    @property
    def committed_target_cost(self) -> float:
        return float(self._committed_cost)

    @property
    def remaining_target_budget(self) -> float | None:
        if self._target_budget_decimal is None:
            return None
        return float(self._target_budget_decimal - self._committed_cost)

    @property
    def audit_hash(self) -> str:
        payload = {
            "events": [event.to_canonical_dict() for event in self._events],
            "manifest_hash": self.manifest_hash,
            "opened_files": [row.to_canonical_dict() for row in self._audit],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def metadata(self, observation_id: str) -> ObservationMeta:
        """Return public metadata without opening any observation payload."""
        return self._manifest.metadata(observation_id)

    def open_context(self, observation_id: str) -> bytes:
        entry = self._manifest.metadata(observation_id)
        if entry.access_level is not AccessLevel.CONTEXT:
            raise PermissionError("target payloads require commit_target then reveal")
        return self._open(entry, "OPEN_CONTEXT")

    def commit_target(self, observation_id: str) -> _RevealCapability:
        """Commit a target action and issue the only valid reveal capability."""
        entry = self._manifest.metadata(observation_id)
        if entry.access_level is not AccessLevel.TARGET:
            raise ValueError("only TARGET observations require a reveal capability")
        if observation_id in self._committed_ids:
            raise RuntimeError("target was already committed")
        entry_cost = Decimal(str(entry.cost))
        if (
            self._target_budget_decimal is not None
            and self._committed_cost + entry_cost > self._target_budget_decimal
        ):
            raise RuntimeError("target commitment would exceed the declared observation budget")
        secret = secrets.token_urlsafe(32)
        self._commits[observation_id] = secret
        self._committed_ids.add(observation_id)
        self._committed_cost += entry_cost
        self._record_event("COMMIT_TARGET", observation_id)
        return _RevealCapability(self._nonce, observation_id, secret)

    def reveal(self, capability: object) -> bytes:
        if not isinstance(capability, _RevealCapability) or capability._ledger_nonce is not self._nonce:
            raise PermissionError("a capability from this ledger is required")
        if self._commits.get(capability._observation_id) != capability._secret:
            raise PermissionError("invalid or revoked reveal capability")
        payload = self._open(self._manifest.metadata(capability._observation_id), "REVEAL_TARGET")
        del self._commits[capability._observation_id]
        return payload

    reveal_target = reveal

    def _record_event(self, event: str, observation_id: str) -> None:
        self._events.append(LedgerEvent(len(self._events), event, observation_id))

    def _open(self, entry: ObservationMeta, event: str) -> bytes:
        payload = self._provider.read_bytes(entry)
        self._record_event(event, entry.observation_id)
        self._audit.append(
            OpenedFileAudit(
                len(self._audit),
                entry.observation_id,
                entry.relative_path,
                entry.access_level,
                hashlib.sha256(payload).hexdigest(),
            )
        )
        return payload
