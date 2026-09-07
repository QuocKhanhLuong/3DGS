"""Bounded selected-state snapshot production and value-bank replay audits.

The audit deliberately checks identity and storage integrity only.  It never
reconstructs a prediction and never imports a decoder, teacher, MRI loader, or
target service.  Snapshot files are content-addressed ``weights_only`` PyTorch
payloads containing one detached state plane tuple per selected state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import torch
from torch import Tensor

from .provenance import canonical_digest, tensor_digest


SNAPSHOT_SCHEMA = "pfgr-lite-selected-state-snapshot-v1"
AUDIT_SCHEMA = "pfgr-lite-bank-audit-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_REPLAY_COUNT = 4096
_MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024
_SNAPSHOT_METADATA_KEYS = frozenset(
    {
        "context_id",
        "state_version",
        "state_digest",
        "planes_digest",
        "producer_compatibility_hash",
        "split_role_hash",
        "subject_binding",
        "geometry",
        "feature_geometry",
        "normalization_hash",
        "route_hash",
        "selected_actions",
    }
)
_ALLOWED_BINDING_KEYS = {
    "schema_version",
    "subject_id",
    "observation_record_id",
    "context_id",
    "geometry_hash",
    "normalization_hash",
    "binding_digest",
}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError("snapshot metadata cannot contain nonfinite values")
        return value
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _plain(value.as_dict())
    if hasattr(value, "to_metadata") and callable(value.to_metadata):
        return _plain(value.to_metadata())
    raise TypeError(f"unsupported snapshot metadata value: {type(value).__name__}")


def _geometry_payload(geometry: object | None) -> Mapping[str, Any] | None:
    if geometry is None:
        return None
    shape = getattr(geometry, "shape_dhw", None)
    affine = getattr(geometry, "voxel_to_ras_mm", None)
    if shape is None or affine is None:
        raise TypeError("snapshot geometry must expose shape_dhw and voxel_to_ras_mm")
    shape_tuple = tuple(int(item) for item in shape)
    affine_tuple = tuple(tuple(float(item) for item in row) for row in affine)
    if len(shape_tuple) != 3 or any(item <= 0 for item in shape_tuple):
        raise ValueError("snapshot geometry shape_dhw must contain positive dimensions")
    if len(affine_tuple) != 4 or any(len(row) != 4 for row in affine_tuple):
        raise ValueError("snapshot geometry affine must be 4x4")
    if not all(torch.isfinite(torch.tensor(item)) for row in affine_tuple for item in row):
        raise ValueError("snapshot geometry affine must be finite")
    return {"shape_dhw": list(shape_tuple), "voxel_to_ras_mm": [list(row) for row in affine_tuple]}


def _feature_geometry_payload(feature_geometry: object | None) -> Mapping[str, Any] | None:
    if feature_geometry is None:
        return None
    source = _geometry_payload(getattr(feature_geometry, "source_geometry", None))
    feature = _geometry_payload(getattr(feature_geometry, "feature_geometry", None))
    scale = tuple(float(item) for item in getattr(feature_geometry, "feature_to_source_scale_dhw", ()))
    offset = tuple(float(item) for item in getattr(feature_geometry, "feature_to_source_offset_dhw", ()))
    operator_chain = tuple(str(item) for item in getattr(feature_geometry, "operator_chain", ()))
    if source is None or feature is None or len(scale) != 3 or len(offset) != 3 or not operator_chain:
        raise ValueError("snapshot feature geometry is incomplete")
    return {
        "source_geometry": source,
        "feature_geometry": feature,
        "tap": str(getattr(feature_geometry, "tap", "")),
        "feature_to_source_scale_dhw": list(scale),
        "feature_to_source_offset_dhw": list(offset),
        "operator_chain": list(operator_chain),
    }


def _state_planes(state: object) -> dict[str, Tensor]:
    planes = getattr(state, "planes", state)
    values: dict[str, Tensor] = {}
    for name in ("xy", "xz", "yz"):
        value = getattr(planes, name, None)
        if not isinstance(value, Tensor):
            raise TypeError(f"state must expose finite tensor plane {name}")
        if value.ndim != 4 or not value.is_floating_point() or value.numel() == 0 or not bool(torch.isfinite(value).all()):
            raise ValueError(f"state plane {name} must be finite nonempty floating rank-4")
        values[name] = value.detach().to(device="cpu").contiguous().clone()
    return values


def _planes_digest(planes: Mapping[str, Tensor]) -> str:
    return canonical_digest(
        {name: tensor_digest(value, name=name) for name, value in planes.items()},
        prefix="pfgr-lite-selected-state-planes-v1|",
    )


def _producer_hash(value: object | None) -> str:
    if value is None:
        return ""
    candidate = getattr(value, "compatibility_hash", None)
    if isinstance(candidate, str):
        return candidate
    candidate = getattr(getattr(value, "compatibility", None), "digest", None)
    if isinstance(candidate, str):
        return candidate
    candidate = getattr(value, "digest", None)
    return candidate if isinstance(candidate, str) else ""


def _action_payload(actions: Sequence[object]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for action in actions:
        if isinstance(action, Mapping):
            action_id = action.get("action_id", action.get("id"))
            state_version = action.get("state_version", 0)
            action_digest = action.get("action_digest", action.get("proposal_hash", ""))
            point_id = action.get("point_id", -1)
        else:
            action_id = getattr(action, "action_id", None)
            state_version = getattr(action, "state_version", 0)
            action_digest = getattr(action, "action_digest", getattr(action, "proposal_hash", ""))
            point_id = getattr(action, "point_id", -1)
        if not isinstance(action_id, str) or not action_id:
            raise ValueError("selected actions require nonempty action_id")
        if not isinstance(action_digest, str):
            raise TypeError("selected action digest must be a string")
        if not isinstance(state_version, int) or isinstance(state_version, bool) or state_version < 0:
            raise ValueError("selected action state_version must be nonnegative")
        if not isinstance(point_id, int) or isinstance(point_id, bool) or point_id < -1:
            raise ValueError("selected action point_id must be an integer >= -1")
        result.append({"action_id": action_id, "state_version": int(state_version), "action_digest": action_digest, "point_id": int(point_id)})
    return result


def _safe_binding(subject_binding: object) -> dict[str, Any]:
    if hasattr(subject_binding, "as_dict") and callable(subject_binding.as_dict):
        subject_binding = subject_binding.as_dict()
    if not isinstance(subject_binding, Mapping):
        raise TypeError("subject_binding must be a mapping or as_dict object")
    unknown = set(subject_binding) - _ALLOWED_BINDING_KEYS
    if unknown:
        raise ValueError(f"subject_binding has unknown fields: {sorted(unknown)}")
    binding = {str(key): _plain(value) for key, value in subject_binding.items()}
    required = {
        "schema_version",
        "subject_id",
        "observation_record_id",
        "context_id",
        "geometry_hash",
        "normalization_hash",
        "binding_digest",
    }
    if set(binding) != required:
        missing = sorted(required - set(binding))
        extra = sorted(set(binding) - required)
        detail = f"missing={missing}" if missing else f"unknown={extra}"
        raise ValueError(f"subject_binding must be a complete canonical envelope ({detail})")
    if binding.get("schema_version") != "pfgr-lite-subject-context-binding-v1":
        raise ValueError("unknown subject-context binding schema")
    for key in ("subject_id", "observation_record_id", "context_id", "geometry_hash", "normalization_hash"):
        if key not in binding or not isinstance(binding[key], str) or not binding[key]:
            raise ValueError(f"subject_binding requires nonempty {key}")
    binding_digest = binding.get("binding_digest")
    if not isinstance(binding_digest, str) or not binding_digest:
        raise ValueError("subject_binding binding_digest must be a nonempty string")
    expected = canonical_digest(
        {
            "schema_version": binding["schema_version"],
            "subject_id": binding["subject_id"],
            "observation_record_id": binding["observation_record_id"],
            "context_id": binding["context_id"],
            "geometry_hash": binding["geometry_hash"],
            "normalization_hash": binding["normalization_hash"],
        },
        prefix="pfgr-lite-subject-context-binding-v1|",
    )
    if binding_digest != expected:
        raise ValueError("subject_binding binding_digest does not match its fields")
    return binding


def _atomic_publish_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(temporary_fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = path.read_bytes()
            if existing != payload:
                raise ValueError(f"existing snapshot differs at content-addressed path: {path.name}")
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_state_snapshot(
    bank_root: str | os.PathLike[str],
    state: object,
    context: object,
    *,
    subject_binding: Mapping[str, Any],
    route_hash: str,
    selected_actions: Sequence[object],
    split_role_hash: str,
) -> str:
    """Write one detached state snapshot and return ``replay/<sha>.pt``."""

    root = Path(bank_root)
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise ValueError("bank_root must be a real directory")
    if not isinstance(route_hash, str) or not route_hash:
        raise ValueError("route_hash must be a nonempty identity")
    if not isinstance(split_role_hash, str) or not split_role_hash:
        raise ValueError("split_role_hash must be a nonempty identity")
    binding = _safe_binding(subject_binding)
    planes = _state_planes(state)
    producer_hash = _producer_hash(getattr(context, "producer", None))
    context_id = getattr(context, "context_id", binding.get("context_id", ""))
    if not isinstance(context_id, str) or not context_id:
        raise ValueError("context must expose a nonempty context_id")
    state_version = getattr(state, "state_version", getattr(state, "version", 0))
    if not isinstance(state_version, int) or isinstance(state_version, bool) or state_version < 0:
        raise ValueError("state must expose a nonnegative state_version")
    state_digest = getattr(state, "state_digest", "")
    if not isinstance(state_digest, str) or not state_digest:
        state_digest = _planes_digest(planes)
    actions = _action_payload(selected_actions)
    metadata = {
        "context_id": context_id,
        "state_version": int(state_version),
        "state_digest": state_digest,
        "planes_digest": _planes_digest(planes),
        "producer_compatibility_hash": producer_hash,
        "split_role_hash": split_role_hash,
        "subject_binding": binding,
        "geometry": _geometry_payload(getattr(context, "geometry", None)),
        "feature_geometry": _feature_geometry_payload(getattr(context, "feature_geometry", None)),
        "normalization_hash": binding["normalization_hash"],
        "route_hash": route_hash,
        "selected_actions": actions,
    }
    payload = {
        "schema_version": SNAPSHOT_SCHEMA,
        "metadata": metadata,
        "state_planes": planes,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer, _use_new_zipfile_serialization=True)
    encoded = buffer.getvalue()
    if len(encoded) > _MAX_SNAPSHOT_BYTES:
        raise ValueError(f"state snapshot exceeds {_MAX_SNAPSHOT_BYTES} bytes")
    digest = hashlib.sha256(encoded).hexdigest()
    relative = Path("replay") / f"{digest}.pt"
    _atomic_publish_bytes(root / relative, encoded)
    return relative.as_posix()


def _resolve_reference(root: Path, reference: object) -> Path:
    if not isinstance(reference, str) or not reference:
        raise ValueError("selected replay reference must be a nonempty relative path")
    relative = Path(reference)
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 2
        or relative.parts[0] != "replay"
        or relative.suffix != ".pt"
    ):
        raise ValueError("selected replay reference must be replay/<sha256>.pt")
    if not _SHA256.fullmatch(relative.stem):
        raise ValueError("selected replay filename must be a SHA256 digest")
    candidate = root / relative
    try:
        resolved_root = root.resolve()
        resolved = candidate.resolve()
        if os.path.commonpath((str(resolved_root), str(resolved))) != str(resolved_root):
            raise ValueError("selected replay path escapes bank root")
    except FileNotFoundError as exc:
        raise ValueError("selected replay path does not exist") from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("selected replay path must be a regular file")
    return candidate


def _expected_hash(value: object) -> str:
    if isinstance(value, str):
        return value
    candidate = _producer_hash(value)
    if candidate:
        return candidate
    digest = getattr(value, "digest", None)
    if isinstance(digest, str):
        return digest
    raise TypeError("producer must expose compatibility_hash/digest")


def validate_snapshot_file(
    path: str | os.PathLike[str],
    *,
    expected_digest: str | None = None,
    expected_producer: str | None = None,
    expected_split_role_hash: str | None = None,
    expected_row: object | None = None,
    max_bytes: int = _MAX_SNAPSHOT_BYTES,
) -> Mapping[str, Any]:
    """Validate one content-addressed replay snapshot without importing a bank.

    The ValueBank reader calls this helper lazily so the replay contract stays
    in one module and no circular import is introduced.  ``expected_row`` is a
    duck-typed row identity used by both reader and audit; tensor payloads are
    only loaded with ``weights_only=True`` and are never decoded or supervised.
    """

    snapshot_path = Path(path)
    if snapshot_path.is_symlink() or not snapshot_path.is_file():
        raise ValueError("selected replay path must be a regular file")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    size = snapshot_path.stat().st_size
    if size > max_bytes:
        raise ValueError("selected replay snapshot exceeds bounded audit size")
    encoded = snapshot_path.read_bytes()
    if len(encoded) != size:
        raise ValueError("selected replay snapshot changed while being read")
    digest = hashlib.sha256(encoded).hexdigest()
    filename_digest = snapshot_path.stem
    if not _SHA256.fullmatch(filename_digest) or digest != filename_digest:
        raise ValueError("selected replay content hash does not match filename")
    if expected_digest is not None:
        if not isinstance(expected_digest, str) or not _SHA256.fullmatch(expected_digest) or digest != expected_digest:
            raise ValueError("selected replay content hash does not match indexed reference")
    try:
        payload = torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("selected replay snapshot is not a weights-only payload") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"schema_version", "metadata", "state_planes"} or payload.get("schema_version") != SNAPSHOT_SCHEMA:
        raise ValueError("unknown selected replay snapshot schema")
    metadata = payload.get("metadata")
    planes = payload.get("state_planes")
    if not isinstance(metadata, Mapping) or set(metadata) != _SNAPSHOT_METADATA_KEYS:
        raise ValueError("selected replay snapshot metadata schema is incomplete or unknown")
    if not isinstance(planes, Mapping) or set(planes) != {"xy", "xz", "yz"}:
        raise ValueError("selected replay snapshot state planes are incomplete")
    snapshot_planes = _state_planes(type("Snapshot", (), {"planes": type("Planes", (), planes)()})())
    if metadata.get("planes_digest") != _planes_digest(snapshot_planes):
        raise ValueError("selected replay plane digest mismatch")
    for key in ("context_id", "state_digest", "normalization_hash", "route_hash"):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError(f"selected replay metadata requires nonempty {key}")
    state_version = metadata.get("state_version")
    if not isinstance(state_version, int) or isinstance(state_version, bool) or state_version < 0:
        raise ValueError("selected replay state_version must be nonnegative")
    producer_hash = metadata.get("producer_compatibility_hash")
    split_hash = metadata.get("split_role_hash")
    if not isinstance(producer_hash, str) or not isinstance(split_hash, str) or not split_hash:
        raise ValueError("selected replay producer/split identities are incomplete")
    if expected_producer is not None and producer_hash != expected_producer:
        raise ValueError("selected replay producer identity mismatch")
    if expected_split_role_hash is not None and split_hash != expected_split_role_hash:
        raise ValueError("selected replay split/role identity mismatch")
    binding = _safe_binding(metadata.get("subject_binding"))
    if binding["context_id"] != metadata["context_id"]:
        raise ValueError("selected replay subject binding context mismatch")
    if binding["normalization_hash"] != metadata["normalization_hash"]:
        raise ValueError("selected replay subject binding normalization mismatch")
    actions = metadata.get("selected_actions")
    if not isinstance(actions, list):
        raise ValueError("selected replay selected_actions must be a list")
    for action in actions:
        if not isinstance(action, Mapping) or set(action) != {"action_id", "state_version", "action_digest", "point_id"}:
            raise ValueError("selected replay action metadata is incomplete")
        if not isinstance(action["action_id"], str) or not action["action_id"]:
            raise ValueError("selected replay action_id must be nonempty")
        if not isinstance(action["state_version"], int) or isinstance(action["state_version"], bool) or action["state_version"] < 0:
            raise ValueError("selected replay action state_version must be nonnegative")
        if not isinstance(action["action_digest"], str):
            raise ValueError("selected replay action_digest must be a string")
        if not isinstance(action["point_id"], int) or isinstance(action["point_id"], bool) or action["point_id"] < -1:
            raise ValueError("selected replay action point_id must be >= -1")
    metadata = dict(metadata)
    if expected_row is not None:
        validate_snapshot_row(metadata, expected_row)
    return metadata


def validate_snapshot_row(metadata: Mapping[str, Any], expected_row: object) -> None:
    """Join one bank row to already-validated snapshot metadata.

    This intentionally performs no tensor/file reads, so duplicate action rows
    sharing one selected state still receive complete identity validation.
    """

    if not isinstance(metadata, Mapping):
        raise TypeError("snapshot metadata must be a mapping")
    binding = metadata.get("subject_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("selected replay subject binding is missing")
    subject = getattr(expected_row, "subject_key", getattr(expected_row, "subject_id", None))
    if binding.get("subject_id") != subject:
        raise ValueError("selected replay subject identity mismatch")
    if metadata.get("context_id") != getattr(expected_row, "context_id", None):
        raise ValueError("selected replay context identity mismatch")
    if metadata.get("state_version") != getattr(expected_row, "state_version", None):
        raise ValueError("selected replay state identity mismatch")
    if metadata.get("state_digest") != getattr(expected_row, "state_digest", None):
        raise ValueError("selected replay state digest mismatch")
    actions = metadata.get("selected_actions")
    if not isinstance(actions, list) or not any(isinstance(item, Mapping) and item.get("action_id") == getattr(expected_row, "action_id", None) for item in actions):
        raise ValueError("selected replay action identity mismatch")
    for row_name, binding_name in (
        ("observation_record_id", "observation_record_id"),
        ("geometry_hash", "geometry_hash"),
        ("normalization_hash", "normalization_hash"),
    ):
        row_value = getattr(expected_row, row_name, None)
        if row_value and binding.get(binding_name) != row_value:
            raise ValueError(f"selected replay {row_name} mismatch")
    row_route = getattr(expected_row, "route_hash", None)
    if row_route and metadata.get("route_hash") != row_route:
        raise ValueError("selected replay route identity mismatch")
    row_action_digest = getattr(expected_row, "proposal_hash", "")
    if row_action_digest:
        action = next((item for item in actions if item.get("action_id") == getattr(expected_row, "action_id", None)), None)
        if action is None or action.get("action_digest") != row_action_digest:
            raise ValueError("selected replay action digest mismatch")


def audit_bank_replay(
    reader: object,
    replay_count: int,
    *,
    producer: object,
    role_manifest: object,
) -> Mapping[str, Any]:
    """Validate bounded selected rows and their content-addressed snapshots."""

    if not isinstance(replay_count, int) or isinstance(replay_count, bool) or replay_count < 0 or replay_count > _MAX_REPLAY_COUNT:
        raise ValueError(f"replay_count must be an integer in [0,{_MAX_REPLAY_COUNT}]")
    root = getattr(reader, "root", None)
    if not isinstance(root, Path):
        root = Path(root) if root is not None else None
    if root is None or not root.is_dir() or root.is_symlink():
        raise TypeError("reader must expose a real bank root directory")
    expected_producer = _expected_hash(producer)
    expected_role = getattr(role_manifest, "digest", None)
    if not isinstance(expected_role, str) or not expected_role:
        raise TypeError("role_manifest must expose a nonempty digest")
    reader_role = getattr(getattr(reader, "role_manifest", None), "digest", None)
    if reader_role != expected_role:
        raise ValueError("bank role manifest does not match replay audit role")
    rows_attr = getattr(reader, "rows", None)
    if not callable(rows_attr):
        raise TypeError("reader must expose rows()")
    rows = tuple(rows_attr())
    selected = [row for row in rows if getattr(row, "selected_replay_ref", "")]
    if replay_count > len(selected):
        raise ValueError(f"requested {replay_count} replay rows but only {len(selected)} selected rows exist")
    selected = selected[:replay_count]
    checked_refs: set[str] = set()
    metadata_by_ref: dict[str, Mapping[str, Any]] = {}
    bytes_checked = 0
    for row in selected:
        row_producer = getattr(row, "producer_compatibility_hash", "")
        if row_producer != expected_producer:
            raise ValueError("selected row producer identity does not match audit producer")
        row_split = getattr(row, "split_role_hash", "")
        if row_split != expected_role:
            raise ValueError("selected row split/role identity does not match audit role")
        reference = getattr(row, "selected_replay_ref", "")
        path = _resolve_reference(root, reference)
        if reference not in metadata_by_ref:
            checked_refs.add(reference)
            metadata_by_ref[reference] = validate_snapshot_file(
                path,
                expected_digest=path.stem,
                expected_producer=expected_producer,
                expected_split_role_hash=expected_role,
            )
            bytes_checked += path.stat().st_size
        validate_snapshot_row(metadata_by_ref[reference], row)
    return {
        "schema_version": AUDIT_SCHEMA,
        "audit_kind": "state_snapshot_and_row_identity",
        "requested_replay_count": replay_count,
        "rows_checked": len(selected),
        "snapshots_checked": len(checked_refs),
        "bytes_checked": bytes_checked,
        "decoder_calls": 0,
        "teacher_calls": 0,
        "reconstruction_replay": False,
        "status": "PASS",
    }


__all__ = ["AUDIT_SCHEMA", "SNAPSHOT_SCHEMA", "audit_bank_replay", "validate_snapshot_file", "validate_snapshot_row", "write_state_snapshot"]
