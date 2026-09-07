"""Immutable cached ValueBank storage for PFGR-Lite.

This module intentionally deals only with already measured, detached action
rows.  It does not know about MRI readers, targets, teachers, updater modules,
or decoders.  The row format is a small private seam around the authoritative
W1 metadata declarations; the on-disk manifest always uses
``ValueBankManifest`` and the exact ``VALUE_BANK_SCHEMA``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Iterator, Mapping

import torch
from torch import Tensor

from .config import ValueModelConfig
from .provenance import ProducerCompatibility, SourceProvenance, canonical_digest, tensor_digest
from .types import (
    DESCRIPTOR_SCHEMA,
    GainLabel,
    ProducerDependencies,
    TrainingRoleManifest,
    ValueBankManifest,
    VALUE_BANK_SCHEMA,
)


INDEX_SCHEMA = "point-guided-pfgr-lite-value-bank-index-v1"
SHARD_SCHEMA = "point-guided-pfgr-lite-value-bank-shard-v1"
SCALE_SCHEMA = "point-guided-pfgr-lite-gain-scale-v1"
BANK_INDEX_NAME = "index.json"
DEFAULT_MAX_ROWS_PER_SHARD = 1024
DEFAULT_MAX_FILE_SIZE = 64 * 1024 * 1024

# ``LABEL_PROVENANCE`` is part of the immutable label identity, not merely a
# human-readable note.  Any future change to the estimand, mask or numerical
# epsilon therefore creates a new bank compatibility identity.
STAGE_PROVENANCE_SCHEMA = "pfgr-lite-producer-stage-v1"
_STAGE_REQUIRED_FIELDS = frozenset({
    "schema_version",
    "stage",
    "spectral_arm",
    "completed",
    "producer_compatibility_hash",
    "projector_before_hash",
    "projector_after_hash",
    "projector_gradient_evidence",
    "projector_update_evidence",
    "initialization_id",
    "checkpoint_id",
    "source_id",
    "split_role_hash",
    "role_manifest_digest",
    "verified_prior_receipt",
    "verified_prior_receipt_hash",
})
_STAGE_ALLOWED_SPECTRAL_ARMS = frozenset({"u_plus_spectral", "verified_prior"})

_ROLE_NAMES = frozenset({
    "producer_fit",
    "validation",
    "calibration_fit",
    "calibration_allowance",
    "test",
    "diagnostic",
    "engineering",
})
_TRAIN_ROLES = frozenset({"producer_fit"})
SUPPORT_PROVENANCE = "complete_support_v1"
_ALLOWED_INCLUSION_MECHANISMS = frozenset({"complete_support_v1", "fixed_q_complete_support_v1"})
LABEL_PROVENANCE = {
    "estimand": "signed_conditional_mean_masked_global_charbonnier",
    "rho": "charbonnier",
    "epsilon": 1e-3,
    "mask": "observation_derived_binary",
}
_ALLOWED_SAMPLER_LAWS = frozenset({
    "exact",
    "exact_union_v1",
    "iid_fixed_q_plane_mixture_c_over_s_v1",
    "iid_fixed_q_plane_mixture_c_over_S_v1",
})
_BANNED_LABEL_TOKENS = frozenset({
    "oracle",
    "target_aware",
    "target-aware",
    "hard_mining",
    "hard-mining",
    "hard mining",
    "privileged",
    "topk_mining",
})
_BANNED_METADATA_KEYS = frozenset({
    "target",
    "target_tensor",
    "target_volume",
    "oracle",
    "oracle_state",
    "teacher",
    "teacher_output",
    "prediction",
    "pixels",
    "image",
    "volume",
    "checkpoint",
    "model_state",
})
_KNOWN_SECRET_RE = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|secret(?:[_-]?key)?|password|passwd|authorization|bearer)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{8,}"
)
_ALLOWED_ROW_KEYS = frozenset({
    "label",
    "gain_label",
    "state96",
    "z96",
    "state",
    "f_spec168",
    "f_spec",
    "semantic3",
    "semantic",
    "reliability3",
    "reliability",
    "q_bar24",
    "q_bar",
    "delta96",
    "delta",
    "actual_delta",
    "o270",
    "v270",
    "v126",
    "raw_gain",
    "benefit",
    "harm",
    "action_id",
    "context_id",
    "state_version",
    "point_id",
    "subject_id",
    "subject",
    "point_ras_mm",
    "point",
    "geometry_id",
    "geometry_hash",
    "proposal_hash",
    "action_digest",
    "state_digest",
    "producer_compatibility_hash",
    "producer_hash",
    "split_role_hash",
    "split_role",
    "role_split",
    "measurement_mode",
    "inclusion_mechanism",
    "support_provenance",
    "selected_replay_ref",
    "replay_ref",
    "engineering_only",
    "diagnostic",
})
_ALLOWED_METADATA_KEYS = frozenset({
    "action_id",
    "context_id",
    "subject_id",
    "state_version",
    "point_id",
    "geometry_id",
    "proposal_hash",
    "state_digest",
    "producer_compatibility_hash",
    "split_role_hash",
    "split_role",
    "role",
    "measurement_mode",
    "q_draws",
    "seed",
    "variance",
    "standard_error",
    "mask_count",
    "footprint_voxels",
    "valid_masked_contributions",
    "sampler_law",
    "label_definition",
    "inclusion_mechanism",
    "support_provenance",
    "selected_replay_ref",
    "engineering_only",
    "diagnostic",
    "raw_gain",
    "benefit",
    "harm",
    "point_ras_mm",
})


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_text(value: Any, name: str, *, required: bool = False) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if required and (not value or value.lower() in {"unknown", "unset", "none", "null"}):
        raise ValueError(f"{name} must be a complete non-sentinel string")
    if "\x00" in value:
        raise ValueError(f"{name} contains NUL")
    return value


def _finite_scalar(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _owned_vector(value: Any, name: str, width: int, *, optional: bool = False) -> Tensor | None:
    if value is None and optional:
        return None
    if not isinstance(value, Tensor):
        # Lists are convenient for tiny synthetic engineering fixtures, but
        # all persisted tensors are converted to FP32 below.
        try:
            value = torch.as_tensor(value)
        except Exception as exc:  # pragma: no cover - defensive error path
            raise TypeError(f"{name} must be a torch.Tensor or numeric sequence") from exc
    if value.ndim != 1 or value.shape[0] != width:
        raise ValueError(f"{name} must have shape [{width}]")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be a floating tensor")
    if value.numel() == 0 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be nonempty and finite")
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()


def _safe_relative_reference(value: Any, name: str) -> str:
    text = _safe_text(value, name)
    if not text:
        return text
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a safe relative reference")
    return path.as_posix()


def _contains_banned(value: Any, *, key: str = "") -> str | None:
    """Find known unsafe metadata patterns without claiming perfect secrecy.

    Arbitrary text is not scanned as a generic secret detector.  Only the
    explicit metadata keys and known credential/target markers are rejected.
    """

    lowered_key = key.lower()
    if lowered_key in _BANNED_METADATA_KEYS:
        return key
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            found = _contains_banned(child_value, key=str(child_key))
            if found:
                return found
        return None
    if isinstance(value, (tuple, list)):
        for child in value:
            found = _contains_banned(child, key=key)
            if found:
                return found
        return None
    return None


def _contains_known_secret(value: Any, *, path: str = "") -> str | None:
    """Reject only explicit credential-like strings in whitelisted metadata."""

    if isinstance(value, str):
        if _KNOWN_SECRET_RE.search(value):
            return path or "string"
        return None
    if isinstance(value, Mapping):
        for key, child in value.items():
            found = _contains_known_secret(child, path=f"{path}.{key}" if path else str(key))
            if found:
                return found
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            found = _contains_known_secret(child, path=f"{path}[{index}]")
            if found:
                return found
    return None


def _label_identity_hash(label_definition: str | None, producer_metadata: Mapping[str, Any]) -> str:
    """Return the canonical hash of the exact label contract.

    The row's label-definition string alone is insufficient: the fixed
    estimand/mask/epsilon declaration and the producer compatibility's label
    hash are all part of the identity.  This helper is shared by writer and
    reader so a self-edited index cannot make a mismatched label population
    appear compatible.
    """

    compatibility = producer_metadata.get("compatibility", {})
    producer_label_hash = ""
    if isinstance(compatibility, Mapping):
        producer_label_hash = str(compatibility.get("label_definition_hash", ""))
    return canonical_digest(
        {
            "definitions": [label_definition or "empty"],
            "producer_label_definition_hash": producer_label_hash,
            **LABEL_PROVENANCE,
        },
        prefix="pfgr-lite-label-definition-v1|",
    )


def _validate_stage_provenance(
    value: Mapping[str, Any] | None,
    *,
    producer_hash: str,
    current_projector_hash: str = "",
    split_role_hash: str,
    role_manifest_digest: str,
    engineering_only: bool,
) -> dict[str, Any] | None:
    """Validate the single producer-stage/spectral evidence envelope.

    The stage envelope is intentionally a plain mapping so W4 checkpoint
    code can round-trip it without introducing a competing public dataclass.
    Production banks require the complete set of frozen producer/projector,
    stage completion, source/checkpoint, and split/role identities.  An
    engineering-only bank may omit it, but any supplied envelope still gets
    strict validation.
    """

    if value is None:
        if engineering_only:
            return None
        raise ValueError("MAIN bank requires explicit producer stage/spectral provenance")
    if not isinstance(value, Mapping):
        raise TypeError("stage_provenance must be a mapping")
    payload = dict(value)
    unknown = set(payload) - _STAGE_REQUIRED_FIELDS
    if unknown:
        raise ValueError(f"unknown stage_provenance keys: {sorted(unknown)}")
    missing = _STAGE_REQUIRED_FIELDS - set(payload)
    if missing and not engineering_only:
        raise ValueError(f"stage_provenance missing required fields: {sorted(missing)}")
    unsafe = _contains_banned(payload)
    if unsafe:
        raise ValueError(f"unsafe stage provenance key: {unsafe}")
    secret = _contains_known_secret(payload)
    if secret:
        raise ValueError(f"credential-like string in stage provenance: {secret}")
    if payload.get("schema_version") != STAGE_PROVENANCE_SCHEMA:
        raise ValueError("unknown stage_provenance schema")
    if payload.get("stage") != "updater":
        raise ValueError("stage_provenance stage must be updater")
    if payload.get("spectral_arm") not in _STAGE_ALLOWED_SPECTRAL_ARMS:
        raise ValueError("stage_provenance spectral_arm must be u_plus_spectral or verified_prior")
    if payload.get("completed") is not True:
        raise ValueError("stage_provenance must record completed=true")
    if payload.get("producer_compatibility_hash") != producer_hash:
        raise ValueError("stage provenance producer hash does not match writer")
    if payload.get("split_role_hash") != split_role_hash:
        raise ValueError("stage provenance split/role hash does not match writer")
    if role_manifest_digest and payload.get("role_manifest_digest") != role_manifest_digest:
        raise ValueError("stage provenance role manifest digest does not match writer")
    for key in (
        "producer_compatibility_hash",
        "projector_before_hash",
        "projector_after_hash",
        "initialization_id",
        "checkpoint_id",
        "source_id",
        "split_role_hash",
        "role_manifest_digest",
    ):
        try:
            _safe_text(payload.get(key), f"stage_provenance {key}", required=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"stage_provenance {key} must be a complete identity") from exc
    for key in ("projector_gradient_evidence", "projector_update_evidence"):
        if not isinstance(payload.get(key), Mapping):
            raise ValueError(f"stage_provenance {key} must be a mapping")
    gradient = payload["projector_gradient_evidence"]
    update = payload["projector_update_evidence"]
    if set(gradient) != {"l2_norm_max", "nonzero_steps", "measured_steps"}:
        raise ValueError("projector_gradient_evidence has unknown fields")
    if set(update) != {"changed_parameter_count", "optimizer_steps"}:
        raise ValueError("projector_update_evidence has unknown fields")
    l2_norm_max = _finite_scalar(gradient["l2_norm_max"], "projector_gradient_evidence.l2_norm_max")
    if l2_norm_max < 0.0:
        raise ValueError("projector_gradient_evidence.l2_norm_max must be nonnegative")
    nonzero_steps = _nonnegative_int(gradient["nonzero_steps"], "projector_gradient_evidence.nonzero_steps")
    measured_steps = _nonnegative_int(gradient["measured_steps"], "projector_gradient_evidence.measured_steps")
    changed_count = _nonnegative_int(update["changed_parameter_count"], "projector_update_evidence.changed_parameter_count")
    optimizer_steps = _nonnegative_int(update["optimizer_steps"], "projector_update_evidence.optimizer_steps")
    if measured_steps < nonzero_steps:
        raise ValueError("projector gradient nonzero_steps cannot exceed measured_steps")
    if current_projector_hash and payload["projector_after_hash"] != current_projector_hash:
        raise ValueError("stage provenance projector_after_hash does not match current producer")
    verified_receipt = payload["verified_prior_receipt"]
    verified_receipt_hash = payload["verified_prior_receipt_hash"]
    if payload["spectral_arm"] == "u_plus_spectral":
        if verified_receipt is not None or verified_receipt_hash is not None:
            raise ValueError("u_plus_spectral stage provenance cannot carry a verified-prior receipt")
        if not (l2_norm_max > 0.0 and nonzero_steps > 0 and changed_count > 0 and optimizer_steps > 0):
            raise ValueError("u_plus_spectral stage provenance requires measured nonzero gradients and updates")
        if payload["projector_before_hash"] == payload["projector_after_hash"]:
            raise ValueError("u_plus_spectral projector before/after hashes must differ")
    else:
        if not isinstance(verified_receipt, Mapping) or not isinstance(verified_receipt_hash, str) or len(verified_receipt_hash) != 64:
            raise ValueError("verified_prior requires a complete original stage receipt and its hash")
        receipt = dict(verified_receipt)
        receipt_unknown = set(receipt) - _STAGE_REQUIRED_FIELDS
        if receipt_unknown or set(receipt) != _STAGE_REQUIRED_FIELDS or receipt.get("schema_version") != STAGE_PROVENANCE_SCHEMA or receipt.get("stage") != "updater" or receipt.get("spectral_arm") != "u_plus_spectral" or receipt.get("completed") is not True:
            raise ValueError("verified_prior receipt must be an original u_plus_spectral stage envelope")
        if receipt.get("verified_prior_receipt") is not None or receipt.get("verified_prior_receipt_hash") is not None:
            raise ValueError("u_plus_spectral verified_prior receipt must have null nested receipt fields")
        for key in (
            "producer_compatibility_hash",
            "projector_before_hash",
            "projector_after_hash",
            "initialization_id",
            "checkpoint_id",
            "source_id",
            "split_role_hash",
            "role_manifest_digest",
        ):
            try:
                _safe_text(receipt.get(key), f"verified_prior receipt {key}", required=True)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"verified_prior receipt {key} must be a complete identity") from exc
        if receipt["split_role_hash"] != payload["split_role_hash"] or receipt["role_manifest_digest"] != payload["role_manifest_digest"]:
            raise ValueError("verified_prior receipt split/role lineage does not match current stage")
        receipt_gradient = receipt.get("projector_gradient_evidence")
        receipt_update = receipt.get("projector_update_evidence")
        if not isinstance(receipt_gradient, Mapping) or not isinstance(receipt_update, Mapping) or set(receipt_gradient) != {"l2_norm_max", "nonzero_steps", "measured_steps"} or set(receipt_update) != {"changed_parameter_count", "optimizer_steps"}:
            raise ValueError("verified_prior receipt is missing strict gradient/update evidence")
        receipt_l2 = _finite_scalar(receipt_gradient["l2_norm_max"], "verified_prior receipt gradient")
        receipt_nonzero = _nonnegative_int(receipt_gradient["nonzero_steps"], "verified_prior receipt nonzero_steps")
        receipt_measured = _nonnegative_int(receipt_gradient["measured_steps"], "verified_prior receipt measured_steps")
        receipt_changed = _nonnegative_int(receipt_update["changed_parameter_count"], "verified_prior receipt changed_parameter_count")
        receipt_steps = _nonnegative_int(receipt_update["optimizer_steps"], "verified_prior receipt optimizer_steps")
        if receipt_l2 <= 0.0 or receipt_nonzero <= 0 or receipt_measured < receipt_nonzero or receipt_changed <= 0 or receipt_steps <= 0 or receipt["projector_before_hash"] == receipt["projector_after_hash"]:
            raise ValueError("verified_prior receipt must prove a completed changed spectral projector")
        receipt_hash = canonical_digest(receipt, prefix="pfgr-lite-producer-stage-receipt-v1|")
        if receipt_hash != verified_receipt_hash:
            raise ValueError("verified_prior receipt hash mismatch")
        if receipt.get("projector_after_hash") != payload["projector_after_hash"]:
            raise ValueError("verified_prior receipt projector hash does not match current producer")
    return _jsonable(payload)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Tensor):
        raise TypeError("tensor values are not JSON metadata")
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata cannot contain nonfinite values")
        return value
    raise TypeError(f"unsupported metadata value {type(value).__name__}")


def _label_from(value: Any, defaults: Mapping[str, Any]) -> GainLabel:
    if isinstance(value, GainLabel):
        return value
    if value is None:
        source = defaults
    elif isinstance(value, Mapping):
        source = value
    else:
        # W1 GainLabel is deliberately duck-typed here to keep this cached
        # module independent from a teacher implementation.
        names = {field.name for field in fields(GainLabel)}
        source = {name: getattr(value, name) for name in names if hasattr(value, name)}
    required = ("action_id", "context_id", "state_version", "raw_gain", "benefit", "harm", "mask_count")
    missing = [name for name in required if name not in source]
    if missing:
        raise ValueError(f"gain label missing fields: {missing}")
    data = {field.name: source[field.name] for field in fields(GainLabel) if field.name in source}
    try:
        return GainLabel(**data)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid GainLabel: {exc}") from exc


def _get(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


@dataclass(frozen=True)
class ValueBankRow:
    """One detached measured action row.

    The six component vectors are persisted rather than arbitrary descriptor
    payloads.  V126/V222/V270/V366 are derived from them in one locked order,
    guaranteeing that every V variant sees identical action rows.
    """

    state96: Tensor
    f_spec168: Tensor
    semantic3: Tensor
    reliability3: Tensor
    q_bar24: Tensor
    delta96: Tensor
    raw_gain: float
    benefit: float
    harm: float
    action_id: str
    context_id: str
    state_version: int = 0
    point_id: int = 0
    subject_id: str = ""
    point_ras_mm: Tensor | None = None
    geometry_id: str = ""
    proposal_hash: str = ""
    state_digest: str = ""
    producer_compatibility_hash: str = ""
    split_role_hash: str = ""
    split_role: str = "producer_fit"
    role: str = "exact_footprint"
    measurement_mode: str = "exact_footprint"
    q_draws: int = 0
    seed: int | None = None
    variance: float | None = None
    standard_error: float | None = None
    mask_count: int = 1
    footprint_voxels: int = 0
    valid_masked_contributions: int = 0
    sampler_law: str = "exact"
    label_definition: str = "signed-conditional-mean-masked-global-charbonnier-v1"
    inclusion_mechanism: str = SUPPORT_PROVENANCE
    support_provenance: str = SUPPORT_PROVENANCE
    selected_replay_ref: str = ""
    engineering_only: bool = False
    diagnostic: bool = False

    def __post_init__(self) -> None:
        vectors = (
            ("state96", self.state96, 96),
            ("f_spec168", self.f_spec168, 168),
            ("semantic3", self.semantic3, 3),
            ("reliability3", self.reliability3, 3),
            ("q_bar24", self.q_bar24, 24),
            ("delta96", self.delta96, 96),
        )
        for name, value, width in vectors:
            if not isinstance(value, Tensor) or value.ndim != 1 or value.shape[0] != width:
                raise ValueError(f"{name} must have shape [{width}]")
            if not value.is_floating_point() or value.numel() == 0 or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite floating data")
        if self.point_ras_mm is not None:
            object.__setattr__(self, "point_ras_mm", _owned_vector(self.point_ras_mm, "point_ras_mm", 3))
        for name in ("raw_gain", "benefit", "harm"):
            _finite_scalar(getattr(self, name), name)
        if not math.isclose(float(self.raw_gain), float(self.benefit) - float(self.harm), rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("raw_gain must equal benefit-harm")
        _safe_text(self.action_id, "action_id", required=True)
        _safe_text(self.context_id, "context_id", required=True)
        if self.subject_id:
            _safe_text(self.subject_id, "subject_id", required=True)
        _nonnegative_int(self.state_version, "state_version")
        _nonnegative_int(self.point_id, "point_id")
        _nonnegative_int(self.q_draws, "q_draws")
        _nonnegative_int(self.mask_count, "mask_count")
        if self.mask_count <= 0:
            raise ValueError("mask_count must be positive")
        _nonnegative_int(self.footprint_voxels, "footprint_voxels")
        _nonnegative_int(self.valid_masked_contributions, "valid_masked_contributions")
        if self.seed is not None:
            _nonnegative_int(self.seed, "seed")
        for name in ("variance", "standard_error"):
            value = getattr(self, name)
            if value is not None and (_finite_scalar(value, name) < 0.0):
                raise ValueError(f"{name} must be nonnegative")
        if self.split_role not in _ROLE_NAMES:
            raise ValueError(f"unknown split role {self.split_role!r}")
        if self.role not in {"exact_footprint", "iid_fixed_q", "screening"}:
            raise ValueError(f"unknown GainLabel role {self.role!r}")
        if self.measurement_mode not in {"exact_footprint", "iid_fixed_q", "screening"}:
            raise ValueError(f"unknown measurement mode {self.measurement_mode!r}")
        if self.role == "iid_fixed_q" and self.q_draws < 2:
            raise ValueError("iid_fixed_q rows require q_draws >= 2")
        if self.role == "exact_footprint" and self.variance not in (None, 0.0):
            raise ValueError("exact_footprint rows have zero sampling variance")
        for name in (
            "geometry_id",
            "proposal_hash",
            "state_digest",
            "producer_compatibility_hash",
            "split_role_hash",
            "sampler_law",
            "label_definition",
            "inclusion_mechanism",
            "support_provenance",
            "measurement_mode",
        ):
            _safe_text(getattr(self, name), name)
        _safe_relative_reference(self.selected_replay_ref, "selected_replay_ref")

    @property
    def subject_key(self) -> str:
        return self.subject_id or self.context_id

    @property
    def v126(self) -> Tensor:
        return torch.cat((self.state96, self.semantic3, self.q_bar24, self.reliability3))

    @property
    def v222(self) -> Tensor:
        return torch.cat((self.v126, self.delta96))

    @property
    def o270(self) -> Tensor:
        return torch.cat((self.state96, self.f_spec168, self.semantic3, self.reliability3))

    @property
    def v270(self) -> Tensor:
        return self.o270

    @property
    def v366(self) -> Tensor:
        return torch.cat((self.o270, self.delta96))

    @classmethod
    def from_action_label(
        cls,
        action: Any,
        label: Any,
        *,
        split_role: str = "producer_fit",
        subject_id: str = "",
        geometry_id: str = "",
        split_role_hash: str = "",
        producer_compatibility_hash: str = "",
        support_provenance: str = SUPPORT_PROVENANCE,
        inclusion_mechanism: str = SUPPORT_PROVENANCE,
        engineering_only: bool = False,
        diagnostic: bool = False,
    ) -> "ValueBankRow":
        """Adapt one W1 ``ActionProposal`` plus ``GainLabel`` without imports.

        The action's already computed ``o270``/``v126`` are split into the
        canonical components.  No updater, decoder, teacher, or target call is
        made here.
        """

        if isinstance(action, Mapping):
            source = action

            def getter(key: str, default: Any = None) -> Any:
                return _get(source, key, default=default)

        else:

            def getter(key: str, default: Any = None) -> Any:
                return getattr(action, key, default)
        o270 = getter("o270")
        v126 = getter("v126")
        delta = getter("delta", getter("actual_delta"))
        if not isinstance(o270, Tensor) or o270.ndim != 1 or o270.shape[0] != 270:
            raise ValueError("action o270 must have shape [270]")
        if not isinstance(v126, Tensor) or v126.ndim != 1 or v126.shape[0] != 126:
            raise ValueError("action v126 must have shape [126]")
        if not isinstance(delta, Tensor) or delta.ndim != 1 or delta.shape[0] != 96:
            raise ValueError("action delta must have shape [96]")
        state96 = o270[:96]
        f_spec168 = o270[96:264]
        semantic3 = o270[264:267]
        reliability3 = o270[267:270]
        v_state = v126[:96]
        v_semantic = v126[96:99]
        q_bar24 = v126[99:123]
        v_reliability = v126[123:126]
        for left, right, name in (
            (state96, v_state, "state96"),
            (semantic3, v_semantic, "semantic3"),
            (reliability3, v_reliability, "reliability3"),
        ):
            if not torch.equal(left, right):
                raise ValueError(f"action descriptors disagree for {name}")
        gain_label = _label_from(label, {})
        action_id = getter("action_id", gain_label.action_id)
        context_id = getter("context_id", gain_label.context_id)
        return cls(
            state96=state96,
            f_spec168=f_spec168,
            semantic3=semantic3,
            reliability3=reliability3,
            q_bar24=q_bar24,
            delta96=delta,
            raw_gain=gain_label.raw_gain,
            benefit=gain_label.benefit,
            harm=gain_label.harm,
            action_id=action_id,
            context_id=context_id,
            state_version=getter("state_version", gain_label.state_version),
            point_id=getter("point_id", 0),
            subject_id=subject_id,
            point_ras_mm=getter("point_ras_mm"),
            geometry_id=geometry_id or getter("geometry_hash", ""),
            proposal_hash=getter("action_digest", getter("proposal_digest", "")) or "",
            state_digest=getter("state_digest", "") or "",
            producer_compatibility_hash=producer_compatibility_hash or getter("producer_compatibility_hash", "") or "",
            split_role_hash=split_role_hash,
            split_role=split_role,
            role=gain_label.role,
            measurement_mode=gain_label.role,
            q_draws=gain_label.q_draws,
            seed=gain_label.seed,
            variance=gain_label.variance,
            standard_error=gain_label.standard_error,
            mask_count=gain_label.mask_count,
            footprint_voxels=gain_label.footprint_voxels,
            valid_masked_contributions=gain_label.valid_masked_contributions,
            sampler_law=gain_label.sampler_law,
            label_definition=gain_label.label_definition,
            inclusion_mechanism=inclusion_mechanism,
            support_provenance=support_provenance,
            engineering_only=engineering_only,
            diagnostic=diagnostic,
        )

    def detached(self) -> "ValueBankRow":
        """Return an owned FP32 CPU copy suitable for immutable storage."""

        return replace(
            self,
            state96=self.state96.detach().to(dtype=torch.float32, device="cpu").contiguous().clone(),
            f_spec168=self.f_spec168.detach().to(dtype=torch.float32, device="cpu").contiguous().clone(),
            semantic3=self.semantic3.detach().to(dtype=torch.float32, device="cpu").contiguous().clone(),
            reliability3=self.reliability3.detach().to(dtype=torch.float32, device="cpu").contiguous().clone(),
            q_bar24=self.q_bar24.detach().to(dtype=torch.float32, device="cpu").contiguous().clone(),
            delta96=self.delta96.detach().to(dtype=torch.float32, device="cpu").contiguous().clone(),
            point_ras_mm=None if self.point_ras_mm is None else self.point_ras_mm.detach().to(dtype=torch.float32, device="cpu").contiguous().clone(),
        )


@dataclass(frozen=True)
class GainScale:
    """Fixed training-bank scale used for signed MSE fitting."""

    scale: float
    quantile: float = 0.90
    method: str = "linear"
    floor: float = 1e-8
    floor_applied: bool = False
    training_role: str = "producer_fit"
    training_row_hash: str = ""
    training_row_count: int = 0
    schema_version: str = SCALE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SCALE_SCHEMA:
            raise ValueError("unknown gain scale schema")
        if not math.isfinite(float(self.scale)) or self.scale <= 0.0:
            raise ValueError("gain scale must be positive and finite")
        if not math.isfinite(float(self.floor)) or self.floor <= 0.0:
            raise ValueError("gain scale floor must be positive and finite")
        if not 0.0 < float(self.quantile) < 1.0 or self.method != "linear":
            raise ValueError("gain scale uses linear interpolation and q in (0,1)")
        _nonnegative_int(self.training_row_count, "training_row_count")
        _safe_text(self.training_role, "training_role", required=True)
        _safe_text(self.training_row_hash, "training_row_hash", required=self.training_row_count > 0)

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self), prefix="pfgr-lite-gain-scale-v1|")

    @property
    def value(self) -> float:
        return self.scale

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["digest"] = self.digest
        return payload


def compute_gain_scale(
    raw_gains: Iterable[float],
    *,
    quantile: float = 0.90,
    floor: float = 1e-8,
    training_role: str = "producer_fit",
    training_row_hash: str = "",
) -> GainScale:
    """Compute q90(abs(raw signed gain)) via linear interpolation.

    The function is pure and does not inspect validation/calibration labels.
    Empty populations receive the explicit floor but callers should report a
    blocked status rather than treating that floor as scientific readiness.
    """

    if not 0.0 < float(quantile) < 1.0 or not math.isfinite(float(quantile)):
        raise ValueError("quantile must lie strictly between zero and one")
    if not math.isfinite(float(floor)) or float(floor) <= 0.0:
        raise ValueError("floor must be positive and finite")
    values = [_finite_scalar(value, "raw_gain") for value in raw_gains]
    if not training_row_hash and values:
        training_row_hash = canonical_digest(values, prefix="pfgr-lite-training-gain-values-v1|")
    absolute = sorted(abs(value) for value in values)
    if not absolute:
        result = float(floor)
        floor_applied = True
    else:
        position = (len(absolute) - 1) * float(quantile)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            result = absolute[lower]
        else:
            fraction = position - lower
            result = absolute[lower] + fraction * (absolute[upper] - absolute[lower])
        floor_applied = result < float(floor)
        result = max(float(result), float(floor))
    return GainScale(
        scale=result,
        quantile=float(quantile),
        method="linear",
        floor=float(floor),
        floor_applied=floor_applied,
        training_role=training_role,
        training_row_hash=training_row_hash,
        training_row_count=len(values),
    )


def _row_metadata(row: ValueBankRow) -> dict[str, Any]:
    return {
        "action_id": row.action_id,
        "context_id": row.context_id,
        "subject_id": row.subject_key,
        "state_version": row.state_version,
        "point_id": row.point_id,
        "geometry_id": row.geometry_id,
        "proposal_hash": row.proposal_hash,
        "state_digest": row.state_digest,
        "producer_compatibility_hash": row.producer_compatibility_hash,
        "split_role_hash": row.split_role_hash,
        "split_role": row.split_role,
        "role": row.role,
        "measurement_mode": row.measurement_mode,
        "q_draws": row.q_draws,
        "seed": row.seed,
        "variance": row.variance,
        "standard_error": row.standard_error,
        "mask_count": row.mask_count,
        "footprint_voxels": row.footprint_voxels,
        "valid_masked_contributions": row.valid_masked_contributions,
        "sampler_law": row.sampler_law,
        "label_definition": row.label_definition,
        "inclusion_mechanism": row.inclusion_mechanism,
        "support_provenance": row.support_provenance,
        "selected_replay_ref": row.selected_replay_ref,
        "engineering_only": row.engineering_only,
        "diagnostic": row.diagnostic,
        "raw_gain": float(row.raw_gain),
        "benefit": float(row.benefit),
        "harm": float(row.harm),
        "point_ras_mm": None if row.point_ras_mm is None else tensor_digest(row.point_ras_mm, name="point_ras_mm"),
    }


def row_digest(row: ValueBankRow) -> str:
    row = row.detached()
    payload = {
        "metadata": _row_metadata(row),
        "state96": tensor_digest(row.state96, name="state96"),
        "f_spec168": tensor_digest(row.f_spec168, name="f_spec168"),
        "semantic3": tensor_digest(row.semantic3, name="semantic3"),
        "reliability3": tensor_digest(row.reliability3, name="reliability3"),
        "q_bar24": tensor_digest(row.q_bar24, name="q_bar24"),
        "delta96": tensor_digest(row.delta96, name="delta96"),
    }
    return canonical_digest(payload, prefix="pfgr-lite-value-bank-row-v1|")


def _row_payload(row: ValueBankRow) -> dict[str, Any]:
    row = row.detached()
    return {
        "metadata": _row_metadata(row),
        "state96": row.state96,
        "f_spec168": row.f_spec168,
        "semantic3": row.semantic3,
        "reliability3": row.reliability3,
        "q_bar24": row.q_bar24,
        "delta96": row.delta96,
        "point_ras_mm": row.point_ras_mm,
    }


def _row_from_payload(payload: Mapping[str, Any]) -> ValueBankRow:
    if set(payload) != {"metadata", "state96", "f_spec168", "semantic3", "reliability3", "q_bar24", "delta96", "point_ras_mm"}:
        raise ValueError("unknown or incomplete value-bank row payload keys")
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError("row metadata must be a mapping")
    unknown_metadata = set(metadata) - _ALLOWED_METADATA_KEYS
    if unknown_metadata:
        raise ValueError(f"unknown value-bank metadata keys: {sorted(unknown_metadata)}")
    required_metadata = {"raw_gain", "benefit", "harm", "action_id", "context_id"}
    if not required_metadata.issubset(metadata):
        raise ValueError("value-bank row metadata is missing required identity/label fields")
    banned = _contains_banned(metadata)
    if banned:
        raise ValueError(f"unsafe row metadata key: {banned}")
    secret = _contains_known_secret(metadata)
    if secret:
        raise ValueError(f"credential-like string in value-bank metadata: {secret}")
    row = ValueBankRow(
        state96=payload["state96"],
        f_spec168=payload["f_spec168"],
        semantic3=payload["semantic3"],
        reliability3=payload["reliability3"],
        q_bar24=payload["q_bar24"],
        delta96=payload["delta96"],
        raw_gain=metadata["raw_gain"],
        benefit=metadata["benefit"],
        harm=metadata["harm"],
        action_id=metadata["action_id"],
        context_id=metadata["context_id"],
        subject_id=metadata.get("subject_id", ""),
        state_version=metadata.get("state_version", 0),
        point_id=metadata.get("point_id", 0),
        point_ras_mm=payload["point_ras_mm"],
        geometry_id=metadata.get("geometry_id", ""),
        proposal_hash=metadata.get("proposal_hash", ""),
        state_digest=metadata.get("state_digest", ""),
        producer_compatibility_hash=metadata.get("producer_compatibility_hash", ""),
        split_role_hash=metadata.get("split_role_hash", ""),
        split_role=metadata.get("split_role", "producer_fit"),
        role=metadata.get("role", "exact_footprint"),
        measurement_mode=metadata.get("measurement_mode", metadata.get("role", "exact_footprint")),
        q_draws=metadata.get("q_draws", 0),
        seed=metadata.get("seed"),
        variance=metadata.get("variance"),
        standard_error=metadata.get("standard_error"),
        mask_count=metadata.get("mask_count", 1),
        footprint_voxels=metadata.get("footprint_voxels", 0),
        valid_masked_contributions=metadata.get("valid_masked_contributions", 0),
        sampler_law=metadata.get("sampler_law", "exact"),
        label_definition=metadata.get("label_definition", "signed-conditional-mean-masked-global-charbonnier-v1"),
        inclusion_mechanism=metadata.get("inclusion_mechanism", SUPPORT_PROVENANCE),
        support_provenance=metadata.get("support_provenance", SUPPORT_PROVENANCE),
        selected_replay_ref=metadata.get("selected_replay_ref", ""),
        engineering_only=_strict_bool(metadata.get("engineering_only", False), "engineering_only"),
        diagnostic=_strict_bool(metadata.get("diagnostic", False), "diagnostic"),
    )
    return row.detached()


def _coerce_row(item: Any, *, defaults: Mapping[str, Any]) -> ValueBankRow:
    if isinstance(item, ValueBankRow):
        row = item
    elif isinstance(item, tuple) and len(item) == 2:
        adapter_defaults = {
            key: value
            for key, value in defaults.items()
            if key in {"split_role", "subject_id", "geometry_id", "split_role_hash", "producer_compatibility_hash", "engineering_only", "diagnostic"}
        } if defaults.get("engineering_only") else {}
        row = ValueBankRow.from_action_label(item[0], item[1], **adapter_defaults)
    elif isinstance(item, Mapping):
        unknown_keys = set(item) - _ALLOWED_ROW_KEYS
        if unknown_keys:
            raise ValueError(f"unknown value-bank row fields: {sorted(unknown_keys)}")
        banned = _contains_banned(item)
        if banned:
            raise ValueError(f"unsafe value-bank field: {banned}")
        secret = _contains_known_secret(item)
        if secret:
            raise ValueError(f"credential-like string in value-bank field: {secret}")
        label = _label_from(_get(item, "label", "gain_label"), item)
        # Already split components are preferred; descriptor bundles and W1
        # action rows are accepted as explicit compatibility conveniences.
        state = _get(item, "state96", "z96", "state")
        f_spec = _get(item, "f_spec168", "f_spec")
        semantic = _get(item, "semantic3", "semantic")
        reliability = _get(item, "reliability3", "reliability")
        q_bar = _get(item, "q_bar24", "q_bar")
        delta = _get(item, "delta96", "delta", "actual_delta")
        o270 = _get(item, "o270", "v270")
        v126 = _get(item, "v126")
        if (state is None or f_spec is None or semantic is None or reliability is None) and o270 is not None:
            state, f_spec, semantic, reliability = o270[:96], o270[96:264], o270[264:267], o270[267:270]
        if (state is None or semantic is None or reliability is None or q_bar is None) and v126 is not None:
            state, semantic, q_bar, reliability = v126[:96], v126[96:99], v126[99:123], v126[123:126]
        if any(value is None for value in (state, f_spec, semantic, reliability, q_bar, delta)):
            raise ValueError("value-bank row requires state96/f_spec168/semantic3/reliability3/q_bar24/delta96")
        row = ValueBankRow(
            state96=state,
            f_spec168=f_spec,
            semantic3=semantic,
            reliability3=reliability,
            q_bar24=q_bar,
            delta96=delta,
            raw_gain=label.raw_gain,
            benefit=label.benefit,
            harm=label.harm,
            action_id=_get(item, "action_id", default=label.action_id),
            context_id=_get(item, "context_id", default=label.context_id),
            state_version=_get(item, "state_version", default=label.state_version),
            point_id=_get(item, "point_id", default=0),
            subject_id=_get(item, "subject_id", "subject", default=defaults.get("subject_id", "") if defaults.get("engineering_only") else ""),
            point_ras_mm=_get(item, "point_ras_mm", "point"),
            geometry_id=_get(item, "geometry_id", "geometry_hash", default=defaults.get("geometry_id", "") if defaults.get("engineering_only") else ""),
            proposal_hash=_get(item, "proposal_hash", "action_digest", default=""),
            state_digest=_get(item, "state_digest", default=""),
            producer_compatibility_hash=_get(item, "producer_compatibility_hash", "producer_hash", default=defaults.get("producer_compatibility_hash", "") if defaults.get("engineering_only") else ""),
            split_role_hash=_get(item, "split_role_hash", default=defaults.get("split_role_hash", "") if defaults.get("engineering_only") else ""),
            split_role=_get(item, "split_role", "role_split", default=defaults.get("split_role", "producer_fit") if defaults.get("engineering_only") else "producer_fit"),
            role=label.role,
            measurement_mode=_get(item, "measurement_mode", default=label.role),
            q_draws=label.q_draws,
            seed=label.seed,
            variance=label.variance,
            standard_error=label.standard_error,
            mask_count=label.mask_count,
            footprint_voxels=label.footprint_voxels,
            valid_masked_contributions=label.valid_masked_contributions,
            sampler_law=label.sampler_law,
            label_definition=label.label_definition,
            inclusion_mechanism=_get(item, "inclusion_mechanism", default=SUPPORT_PROVENANCE),
            support_provenance=_get(item, "support_provenance", default=SUPPORT_PROVENANCE),
            selected_replay_ref=_get(item, "selected_replay_ref", "replay_ref", default=""),
            engineering_only=_strict_bool(_get(item, "engineering_only", default=defaults.get("engineering_only", False)), "engineering_only"),
            diagnostic=_strict_bool(_get(item, "diagnostic", default=defaults.get("diagnostic", False)), "diagnostic"),
        )
    else:
        raise TypeError("value-bank rows must be ValueBankRow, mapping, or (action,label) pair")
    # Writer-supplied defaults fill only absent provenance; a row's explicit
    # nonempty identity is never silently reinterpreted.
    updates: dict[str, Any] = {}
    # Convenience defaults are deliberately restricted to explicitly
    # engineering-only banks.  MAIN rows must carry their own immutable
    # proposal/state/producer/split identities; stamping them from the writer
    # would turn an incomplete measurement into a falsely compatible row.
    if defaults.get("engineering_only"):
        for key in ("split_role", "subject_id", "geometry_id", "split_role_hash", "producer_compatibility_hash"):
            if not getattr(row, key) and defaults.get(key):
                updates[key] = defaults[key]
        if not row.engineering_only:
            updates["engineering_only"] = True
    if updates:
        row = replace(row, **updates)
    return row.detached()


def _producer_hash(producer: Any) -> str:
    if isinstance(producer, ProducerDependencies):
        return producer.compatibility.digest
    if isinstance(producer, ProducerCompatibility):
        return producer.digest
    raise TypeError("producer must be the authoritative W1 ProducerCompatibility or ProducerDependencies")


def _role_subject_map(role_manifest: TrainingRoleManifest) -> dict[str, str]:
    """Expand the authoritative W1 role manifest into strict row joins."""

    mapping: dict[str, str] = {}
    for role, subjects in (
        ("validation", role_manifest.baseline_validation_subject_ids),
        ("test", role_manifest.baseline_test_subject_ids),
        ("producer_fit", role_manifest.producer_fit_subject_ids),
        ("calibration_fit", role_manifest.calibration_fit_subject_ids),
        ("calibration_allowance", role_manifest.calibration_allowance_subject_ids),
    ):
        for subject in subjects:
            if subject in mapping:
                raise ValueError("TrainingRoleManifest subject appears in multiple roles")
            mapping[subject] = role
    return mapping


class ValueBankWriter:
    """Append detached rows and atomically publish a versioned bank."""

    def __init__(
        self,
        destination: str | os.PathLike[str],
        *,
        producer: ProducerCompatibility | ProducerDependencies,
        split_role_hash: str,
        role_manifest: TrainingRoleManifest | Mapping[str, Any] | None = None,
        # Deprecated engineering-only convenience; MAIN rows must consume the
        # authoritative TrainingRoleManifest instead.
        role_membership: Mapping[str, str] | None = None,
        config: ValueModelConfig | Mapping[str, Any] | None = None,
        max_rows_per_shard: int = DEFAULT_MAX_ROWS_PER_SHARD,
        training_role: str = "producer_fit",
        engineering_only: bool = False,
        diagnostic: bool = False,
        stage_provenance: Mapping[str, Any] | None = None,
        source_scale: GainScale | Mapping[str, Any] | None = None,
    ) -> None:
        self.destination = Path(destination)
        if self.destination.exists():
            raise FileExistsError(f"value-bank destination already exists: {self.destination}")
        if self.destination.name in {"", ".", ".."}:
            raise ValueError("destination must be a unique named directory")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(config, Mapping):
            config = ValueModelConfig.from_dict(config)
        self.config = config or ValueModelConfig()
        if not isinstance(max_rows_per_shard, int) or isinstance(max_rows_per_shard, bool) or max_rows_per_shard <= 0:
            raise ValueError("max_rows_per_shard must be positive")
        self.max_rows_per_shard = max_rows_per_shard
        if isinstance(role_manifest, Mapping):
            role_manifest = TrainingRoleManifest.from_dict(role_manifest)
        if role_manifest is not None and not isinstance(role_manifest, TrainingRoleManifest):
            raise TypeError("role_manifest must be the authoritative TrainingRoleManifest")
        if role_manifest is not None:
            engineering_only = bool(engineering_only or role_manifest.engineering_only)
        if isinstance(producer, ProducerCompatibility) and not engineering_only:
            raise TypeError("MAIN value banks require ProducerDependencies with source/training provenance")
        if not isinstance(producer, (ProducerCompatibility, ProducerDependencies)):
            raise TypeError("producer must be the authoritative W1 ProducerCompatibility or ProducerDependencies")
        if not isinstance(producer, ProducerDependencies) and not engineering_only:
            raise TypeError("MAIN value banks require typed ProducerDependencies")
        if isinstance(producer, ProducerDependencies) and not engineering_only and producer.source_provenance.synthetic_untrained:
            raise ValueError("synthetic/untrained source provenance requires engineering_only=True")
        if isinstance(producer, ProducerDependencies) and not engineering_only:
            source = producer.source_provenance
            if not source.parameter_hash or not source.frozen_bn_hash or source.traversal_count < 1:
                raise ValueError("MAIN bank producer requires verified source parameter/BN/traversal provenance")
        self.producer_hash = _producer_hash(producer)
        if isinstance(producer, ProducerDependencies):
            self.current_projector_hash = producer.compatibility.spectral_projector_hash
        elif isinstance(producer, ProducerCompatibility):
            self.current_projector_hash = producer.spectral_projector_hash
        else:  # pragma: no cover - constructor rejects this branch above
            self.current_projector_hash = ""
        if isinstance(producer, (ProducerCompatibility, ProducerDependencies)):
            compatibility = producer if isinstance(producer, ProducerCompatibility) else producer.compatibility
            self.producer_metadata = {
                "compatibility": compatibility.as_dict(),
                "source_provenance": None if isinstance(producer, ProducerCompatibility) else producer.source_provenance.as_dict(),
            }
        else:
            self.producer_metadata = {"compatibility_hash": self.producer_hash, "engineering_identity": True}
        self.split_role_hash = _safe_text(split_role_hash, "split_role_hash", required=True)
        self.training_role = _safe_text(training_role, "training_role", required=True)
        if self.training_role not in _TRAIN_ROLES:
            raise ValueError("training_role must be producer_fit")
        if role_manifest is None and not engineering_only:
            raise ValueError("MAIN value banks require the authoritative TrainingRoleManifest")
        if role_manifest is not None:
            self.role_manifest = role_manifest
            self.role_membership = _role_subject_map(role_manifest)
            self.role_membership_digest = role_manifest.digest
            if not role_manifest.engineering_only and self.split_role_hash != role_manifest.digest:
                raise ValueError("split_role_hash must equal TrainingRoleManifest.digest")
        else:
            self.role_manifest = None
            self.role_membership = {}
            if role_membership is not None:
                for subject, role in role_membership.items():
                    subject_text = _safe_text(subject, "role_membership subject", required=True)
                    role_text = _safe_text(role, "role_membership role", required=True)
                    if role_text not in _ROLE_NAMES:
                        raise ValueError(f"unknown role_membership role {role_text!r}")
                    self.role_membership[subject_text] = role_text
            self.role_membership_digest = canonical_digest(sorted(self.role_membership.items()), prefix="pfgr-lite-role-membership-v1|")
        self.engineering_only = bool(engineering_only)
        self.diagnostic = bool(diagnostic)
        role_digest_for_stage = self.role_membership_digest if self.role_manifest is not None else ""
        self.stage_provenance = _validate_stage_provenance(
            stage_provenance,
            producer_hash=self.producer_hash,
            current_projector_hash=self.current_projector_hash,
            split_role_hash=self.split_role_hash,
            role_manifest_digest=role_digest_for_stage,
            engineering_only=self.engineering_only,
        )
        if source_scale is not None:
            if isinstance(source_scale, Mapping):
                source_data = dict(source_scale)
                source_digest = source_data.pop("digest", None)
                source_scale = GainScale(**source_data)
                if source_digest is not None and source_digest != source_scale.digest:
                    raise ValueError("source_scale digest does not match its provenance")
            if not isinstance(source_scale, GainScale):
                raise TypeError("source_scale must be GainScale")
        self.source_scale = source_scale
        if self.source_scale is not None and self.source_scale.training_role != self.training_role:
            raise ValueError("source_scale training_role does not match writer training_role")
        self._row_buffer: list[ValueBankRow] = []
        self._shard_entries: list[dict[str, Any]] = []
        self._row_entries: list[dict[str, Any]] = []
        self._training_gains: list[float] = []
        self._training_row_hashes: list[str] = []
        self._subject_ids: set[str] = set()
        self._row_count = 0
        self._has_diagnostic = False
        self._has_engineering = False
        self._seen_actions: set[str] = set()
        self._subjects_by_role: dict[str, str] = {}
        self._label_definition: str | None = None
        self._finalized = False
        self._aborted = False
        self._stage = Path(tempfile.mkdtemp(prefix=f".{self.destination.name}.value-bank-", dir=str(self.destination.parent)))

    def _flush_buffer(self) -> None:
        if not self._row_buffer:
            return
        shard_index = len(self._shard_entries)
        shard_rows = list(self._row_buffer)
        shard_name = f"shard-{shard_index:05d}.pt"
        shard_path = self._stage / shard_name
        payload = {
            "schema_version": SHARD_SCHEMA,
            "descriptor_schema": DESCRIPTOR_SCHEMA,
            "rows": [_row_payload(row) for row in shard_rows],
        }
        temporary = shard_path.with_name(f".{shard_name}.tmp")
        torch.save(payload, temporary, _use_new_zipfile_serialization=True)
        os.replace(temporary, shard_path)
        shard_hash = _sha256_file(shard_path)
        self._shard_entries.append({"path": shard_name, "sha256": shard_hash, "size": shard_path.stat().st_size, "row_count": len(shard_rows)})
        for offset, row in enumerate(shard_rows):
            self._row_entries.append({
                "row_id": len(self._row_entries),
                "shard": shard_name,
                "offset": offset,
                "row_hash": row_digest(row),
                "action_id": row.action_id,
                "context_id": row.context_id,
                "subject_id": row.subject_key,
                "split_role": row.split_role,
            })
        self._row_buffer.clear()

    def _validate_row(self, row: ValueBankRow) -> ValueBankRow:
        if row.action_id in self._seen_actions:
            raise ValueError(f"duplicate action_id {row.action_id!r}")
        if not self.engineering_only:
            if row.engineering_only:
                raise ValueError("engineering_only rows cannot enter a MAIN bank")
            if row.diagnostic:
                raise ValueError("diagnostic rows require a segregated diagnostic bank")
            missing_identities: set[str] = set()
            for name, value in (
                ("proposal_hash", row.proposal_hash),
                ("state_digest", row.state_digest),
                ("geometry_id", row.geometry_id),
                ("producer_compatibility_hash", row.producer_compatibility_hash),
                ("split_role_hash", row.split_role_hash),
            ):
                try:
                    _safe_text(value, name, required=True)
                except (TypeError, ValueError):
                    missing_identities.add(name)
            if missing_identities:
                raise ValueError(f"MAIN row missing immutable identities: {sorted(missing_identities)}")
        if row.producer_compatibility_hash and row.producer_compatibility_hash != self.producer_hash:
            raise ValueError("row producer compatibility does not match writer")
        if row.split_role_hash and row.split_role_hash != self.split_role_hash:
            raise ValueError("row split/role hash does not match writer")
        if self.engineering_only and row.split_role_hash != self.split_role_hash:
            row = replace(row, split_role_hash=self.split_role_hash)
        if self.engineering_only and row.producer_compatibility_hash != self.producer_hash:
            row = replace(row, producer_compatibility_hash=self.producer_hash)
        if self._label_definition is None:
            self._label_definition = row.label_definition
        elif row.label_definition != self._label_definition:
            raise ValueError("all main bank rows must use one label definition")
        role_subject = self._subjects_by_role.get(row.subject_key)
        if role_subject is None:
            self._subjects_by_role[row.subject_key] = row.split_role
        elif role_subject != row.split_role:
            raise ValueError(f"subject {row.subject_key!r} overlaps split roles {role_subject!r}/{row.split_role!r}")
        if self.role_membership:
            declared_role = self.role_membership.get(row.subject_key)
            if declared_role is None:
                raise ValueError(f"subject {row.subject_key!r} missing from immutable role_membership")
            if declared_role != row.split_role:
                raise ValueError(f"subject {row.subject_key!r} role differs from immutable role_membership")
        if row.role == "screening" and not (self.diagnostic or row.diagnostic):
            raise ValueError("screening labels require an explicit diagnostic bank")
        if row.measurement_mode != row.role:
            raise ValueError("measurement_mode must match the declared GainLabel role")
        if row.role in {"exact_footprint", "iid_fixed_q"}:
            if row.support_provenance != SUPPORT_PROVENANCE:
                raise ValueError(f"main labels require exact support provenance {SUPPORT_PROVENANCE!r}")
            if row.inclusion_mechanism not in _ALLOWED_INCLUSION_MECHANISMS:
                raise ValueError("main labels require an exact declared inclusion mechanism")
            if row.role == "exact_footprint" and row.inclusion_mechanism != SUPPORT_PROVENANCE:
                raise ValueError("exact footprint rows require complete_support_v1 inclusion")
            if row.role == "iid_fixed_q" and row.inclusion_mechanism not in _ALLOWED_INCLUSION_MECHANISMS:
                raise ValueError("fixed-Q rows require a declared complete-support inclusion mechanism")
            if row.sampler_law not in _ALLOWED_SAMPLER_LAWS:
                raise ValueError("main labels require an exact declared sampler law")
            if any(token in row.inclusion_mechanism.lower() or token in row.sampler_law.lower() for token in ("stopped", "optional", "adaptive")):
                raise ValueError("optionally stopped samples cannot enter the main bank")
        marker = " ".join((row.label_definition, row.sampler_law, row.inclusion_mechanism, row.support_provenance)).lower()
        if any(token in marker for token in _BANNED_LABEL_TOKENS) and not (self.diagnostic or row.diagnostic):
            raise ValueError("target-aware/oracle/hard-mining labels are not valid main rows")
        # Metadata values are scalar/identifier-only; rejecting nested arrays
        # here prevents an alternate key from smuggling a volume payload.
        metadata = _row_metadata(row)
        banned = _contains_banned(metadata)
        if banned:
            raise ValueError(f"unsafe row metadata key: {banned}")
        secret = _contains_known_secret(metadata)
        if secret:
            raise ValueError(f"credential-like string in value-bank metadata: {secret}")
        self._seen_actions.add(row.action_id)
        self._subject_ids.add(row.subject_key)
        self._row_count += 1
        digest = row_digest(row)
        # Privileged screening/diagnostic rows are never part of the MAIN
        # producer-fit scale population, even when a diagnostic bank happens
        # to reuse the producer_fit split label.
        if row.split_role == self.training_role and not row.diagnostic and row.role != "screening":
            self._training_gains.append(float(row.raw_gain))
            self._training_row_hashes.append(digest)
        self._has_diagnostic = self._has_diagnostic or row.diagnostic or row.role == "screening"
        self._has_engineering = self._has_engineering or row.engineering_only
        return row

    def append(self, rows: ValueBankRow | Mapping[str, Any] | tuple[Any, Any] | Iterable[Any]) -> "ValueBankWriter":
        if self._finalized or self._aborted:
            raise RuntimeError("value-bank writer is closed")
        if isinstance(rows, (ValueBankRow, Mapping)) or (isinstance(rows, tuple) and len(rows) == 2):
            iterable: Iterable[Any] = (rows,)
        else:
            iterable = rows
        defaults = {
            "split_role_hash": self.split_role_hash,
            "producer_compatibility_hash": self.producer_hash,
            "engineering_only": self.engineering_only,
            "diagnostic": self.diagnostic,
        }
        try:
            for item in iterable:
                row = _coerce_row(item, defaults=defaults)
                self._row_buffer.append(self._validate_row(row))
                if len(self._row_buffer) >= self.max_rows_per_shard:
                    self._flush_buffer()
        except Exception:
            # Do not leave a partially accepted row in memory if this append
            # fails; finalization can never mint a success manifest afterward.
            self._row_buffer.clear()
            self._shard_entries.clear()
            self._row_entries.clear()
            self._training_gains.clear()
            self._training_row_hashes.clear()
            self._subject_ids.clear()
            self._row_count = 0
            self._has_diagnostic = False
            self._has_engineering = False
            self._seen_actions.clear()
            self._subjects_by_role.clear()
            self._label_definition = None
            self.abort()
            raise
        return self

    def _write_json(self, path: Path, value: Mapping[str, Any]) -> None:
        payload = (json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def finalize(self) -> ValueBankManifest:
        if self._finalized or self._aborted:
            raise RuntimeError("value-bank writer is closed")
        try:
            self._flush_buffer()
            ordered_training_hash = canonical_digest(self._training_row_hashes, prefix="pfgr-lite-training-row-set-v1|") if self._training_row_hashes else ""
            if self.source_scale is not None:
                # S4/refresh banks keep the original fixed scale exactly.  We
                # still validate every training gain while appending, but do
                # not recompute q90 from the new population.
                scale = self.source_scale
            else:
                scale = compute_gain_scale(
                    self._training_gains,
                    quantile=self.config.gain_scale_quantile,
                    floor=self.config.gain_scale_floor,
                    training_role=self.training_role,
                    training_row_hash=ordered_training_hash,
                )
            manifest = ValueBankManifest(
                producer_compatibility_hash=self.producer_hash,
                label_definition_hash=_label_identity_hash(self._label_definition, self.producer_metadata),
                split_role_hash=self.split_role_hash,
                gain_scale=scale.scale,
                gain_scale_hash=scale.digest,
                shard_hashes=tuple(entry["sha256"] for entry in self._shard_entries),
                row_count=self._row_count,
                subject_count=len(self._subject_ids),
            )
            status = "READY"
            reasons: list[str] = []
            if not self._row_count:
                status = "BLOCKED_EMPTY"
                reasons.append("no rows supplied")
            elif not self._training_gains:
                status = "BLOCKED_MISSING_TRAINING"
                reasons.append(f"no rows in training role {self.training_role!r}")
            if self.diagnostic or self._has_diagnostic:
                status = "DIAGNOSTIC_ONLY"
                reasons.append("diagnostic/screening labels are segregated from MAIN population")
            elif self.engineering_only or self._has_engineering:
                status = "ENGINEERING_ONLY"
                reasons.append("engineering-only provenance; no scientific claim")
            index = {
                "index_schema": INDEX_SCHEMA,
                "schema_version": VALUE_BANK_SCHEMA,
                "descriptor_schema": DESCRIPTOR_SCHEMA,
                "manifest": asdict(manifest),
                "gain_scale": scale.as_dict(),
                "shards": self._shard_entries,
                "rows": self._row_entries,
                "role_manifest": None if self.role_manifest is None else self.role_manifest.as_dict(),
                "role_manifest_digest": self.role_membership_digest,
                "label_definition_provenance": LABEL_PROVENANCE,
                "stage_provenance": self.stage_provenance,
                "source_scale_hash": None if self.source_scale is None else self.source_scale.digest,
                "status": {"evidence_status": status, "reasons": reasons},
                "producer": {"compatibility_hash": self.producer_hash, "training_role": self.training_role, **self.producer_metadata},
                "limits": {"max_rows_per_shard": self.max_rows_per_shard},
            }
            index_path = self._stage / BANK_INDEX_NAME
            self._write_json(index_path, index)
            # Publish only after every shard and the complete index have been
            # fsynced; destination races are handled by exclusive mkdir.
            try:
                self.destination.mkdir(mode=0o755)
            except FileExistsError as exc:
                raise FileExistsError(f"value-bank destination appeared during finalization: {self.destination}") from exc
            published: list[Path] = []
            try:
                children = sorted(self._stage.iterdir(), key=lambda p: p.name)
                # Publish immutable shards first and the index last.  A
                # reader that observes the destination can therefore never
                # mistake an index that references absent shards for a valid
                # bank; the destination directory itself was created
                # exclusively above to close the overwrite race.
                for child in children:
                    if child.name != BANK_INDEX_NAME:
                        target = self.destination / child.name
                        if target.exists() or target.is_symlink():
                            raise FileExistsError(f"value-bank destination artifact appeared: {target.name}")
                        os.replace(child, target)
                        published.append(target)
                index_source = self._stage / BANK_INDEX_NAME
                if index_source.exists():
                    target = self.destination / BANK_INDEX_NAME
                    if target.exists() or target.is_symlink():
                        raise FileExistsError(f"value-bank destination artifact appeared: {target.name}")
                    os.replace(index_source, target)
                    published.append(target)
            except Exception:
                # Roll back only artifacts this writer actually published;
                # never recursively delete a raced destination that may now
                # contain another process's files.
                for target in reversed(published):
                    try:
                        if target.is_file() or target.is_symlink():
                            target.unlink()
                    except OSError:
                        pass
                try:
                    self.destination.rmdir()
                except OSError:
                    pass
                raise
            shutil.rmtree(self._stage, ignore_errors=True)
            self._finalized = True
            return manifest
        except Exception:
            self.abort()
            raise

    def abort(self) -> None:
        if self._finalized or self._aborted:
            return
        shutil.rmtree(self._stage, ignore_errors=True)
        self._aborted = True

    def __enter__(self) -> "ValueBankWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is not None or not self._finalized:
            self.abort()


def build_value_bank(
    rows: Iterable[Any],
    destination: str | os.PathLike[str],
    *,
    producer: ProducerCompatibility | ProducerDependencies | str,
    split_role_hash: str,
    config: ValueModelConfig | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ValueBankManifest:
    """Convenience wrapper for the detached-row writer."""

    writer = ValueBankWriter(destination, producer=producer, split_role_hash=split_role_hash, config=config, **kwargs)
    try:
        writer.append(rows)
        return writer.finalize()
    except Exception:
        writer.abort()
        raise


class ValueBankReader:
    """Strict reader that validates index, shard checksums and every row."""

    def __init__(
        self,
        bank: str | os.PathLike[str],
        *,
        expected_producer: ProducerCompatibility | ProducerDependencies | str | None = None,
        expected_split_role_hash: str | None = None,
        expected_role_manifest: TrainingRoleManifest | Mapping[str, Any] | None = None,
        expected_baseline_split_hash: str | None = None,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    ) -> None:
        path = Path(bank)
        if path.is_symlink():
            raise ValueError("value-bank root may not be a symlink")
        if path.is_file():
            if path.name != BANK_INDEX_NAME:
                raise ValueError(f"bank file must be {BANK_INDEX_NAME}")
            self.root = path.parent
            self.index_path = path
        elif path.is_dir():
            self.root = path
            self.index_path = path / BANK_INDEX_NAME
        else:
            raise FileNotFoundError(path)
        if not self.index_path.is_file() or self.index_path.is_symlink():
            raise ValueError(f"missing or symlinked {BANK_INDEX_NAME}")
        if not isinstance(max_file_size, int) or isinstance(max_file_size, bool) or max_file_size <= 0:
            raise ValueError("max_file_size must be positive")
        self.max_file_size = max_file_size
        try:
            index = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"malformed value-bank {BANK_INDEX_NAME}") from exc
        if not isinstance(index, Mapping) or index.get("index_schema") != INDEX_SCHEMA or index.get("schema_version") != VALUE_BANK_SCHEMA:
            raise ValueError("unknown value-bank index schema")
        required = {
            "index_schema",
            "schema_version",
            "descriptor_schema",
            "manifest",
            "gain_scale",
            "shards",
            "rows",
            "role_manifest",
            "role_manifest_digest",
            "label_definition_provenance",
            "stage_provenance",
            "source_scale_hash",
            "status",
            "producer",
            "limits",
        }
        unknown = set(index) - required
        if unknown:
            raise ValueError(f"unknown value-bank index keys: {sorted(unknown)}")
        if index.get("descriptor_schema") != DESCRIPTOR_SCHEMA:
            raise ValueError("unknown descriptor schema")
        manifest_required = {field.name for field in fields(ValueBankManifest)}
        if not isinstance(index.get("manifest"), Mapping) or set(index["manifest"]) != manifest_required:
            raise ValueError("invalid ValueBankManifest envelope")
        manifest_data = dict(index["manifest"])
        manifest_data["shard_hashes"] = tuple(manifest_data.get("shard_hashes", ()))
        self._manifest = ValueBankManifest(**manifest_data)
        scale_required = {field.name for field in fields(GainScale)} | {"digest"}
        if not isinstance(index.get("gain_scale"), Mapping) or set(index["gain_scale"]) != scale_required:
            raise ValueError("invalid gain scale envelope")
        scale_data = dict(index["gain_scale"])
        scale_data.pop("digest", None)
        self._scale = GainScale(**scale_data)
        if self._scale.digest != self._manifest.gain_scale_hash or not math.isclose(self._scale.scale, self._manifest.gain_scale, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("gain scale provenance does not match manifest")
        if not isinstance(index["shards"], list) or not isinstance(index["rows"], list):
            raise TypeError("shards/rows index entries must be lists")
        indexed_shard_hashes = tuple(str(spec.get("sha256")) for spec in index["shards"] if isinstance(spec, Mapping))
        if indexed_shard_hashes != tuple(self._manifest.shard_hashes):
            raise ValueError("manifest shard hashes do not match shard index")
        if index.get("label_definition_provenance") != LABEL_PROVENANCE:
            raise ValueError("label definition provenance mismatch")
        source_scale_hash = index.get("source_scale_hash")
        if source_scale_hash is not None and (not isinstance(source_scale_hash, str) or len(source_scale_hash) != 64):
            raise ValueError("source_scale_hash must be null or a SHA256 digest")
        if source_scale_hash is not None and source_scale_hash != self._scale.digest:
            raise ValueError("source scale hash does not match persisted gain scale")
        status = index.get("status")
        if not isinstance(status, Mapping) or set(status) != {"evidence_status", "reasons"} or status.get("evidence_status") not in {"READY", "BLOCKED_EMPTY", "BLOCKED_MISSING_TRAINING", "ENGINEERING_ONLY", "DIAGNOSTIC_ONLY"} or not isinstance(status.get("reasons"), list) or any(not isinstance(reason, str) for reason in status["reasons"]):
            raise ValueError("invalid value-bank evidence status")
        limits = index.get("limits")
        if not isinstance(limits, Mapping) or set(limits) != {"max_rows_per_shard"}:
            raise ValueError("invalid value-bank limits envelope")
        _positive_limit = limits.get("max_rows_per_shard")
        if not isinstance(_positive_limit, int) or isinstance(_positive_limit, bool) or _positive_limit <= 0:
            raise ValueError("max_rows_per_shard limit must be a positive integer")
        expected_bank_files = {BANK_INDEX_NAME}
        for shard_spec in index["shards"]:
            if isinstance(shard_spec, Mapping) and isinstance(shard_spec.get("path"), str):
                expected_bank_files.add(shard_spec["path"])
        # S2 may publish one immutable content-addressed replay directory in
        # addition to shards.  Its contents are validated against the loaded
        # row references below; no other top-level artifacts are permitted.
        expected_bank_files.add("replay")
        unexpected_files = {
            child.name
            for child in self.root.iterdir()
            if child.name not in expected_bank_files
        }
        if unexpected_files:
            raise ValueError(f"unexpected value-bank artifacts: {sorted(unexpected_files)}")
        role_manifest_payload = index["role_manifest"]
        if role_manifest_payload is None:
            self._role_manifest = None
            self._role_membership = {}
        else:
            self._role_manifest = TrainingRoleManifest.from_dict(role_manifest_payload)
            self._role_membership = _role_subject_map(self._role_manifest)
            if self._role_manifest.digest != index["role_manifest_digest"]:
                raise ValueError("role manifest digest mismatch")
            if self._manifest.split_role_hash != self._role_manifest.digest and not self._role_manifest.engineering_only:
                raise ValueError("manifest split/role hash does not match role manifest")
        self._engineering_only = bool(self._role_manifest is None or self._role_manifest.engineering_only or status.get("evidence_status") == "ENGINEERING_ONLY")
        if not self._engineering_only and status.get("evidence_status") in {"ENGINEERING_ONLY", "DIAGNOSTIC_ONLY"}:
            raise ValueError("MAIN role manifest bank cannot carry engineering/diagnostic status")
        role_manifest_for_stage = self._role_manifest
        role_digest_for_stage = "" if role_manifest_for_stage is None else role_manifest_for_stage.digest
        producer_envelope = index.get("producer")
        compatibility_envelope = producer_envelope.get("compatibility", {}) if isinstance(producer_envelope, Mapping) else {}
        current_projector_hash = compatibility_envelope.get("spectral_projector_hash", "") if isinstance(compatibility_envelope, Mapping) else ""
        self._stage_provenance = _validate_stage_provenance(
            index.get("stage_provenance"),
            producer_hash=self._manifest.producer_compatibility_hash,
            current_projector_hash=str(current_projector_hash),
            split_role_hash=self._manifest.split_role_hash,
            role_manifest_digest=role_digest_for_stage,
            engineering_only=(role_manifest_for_stage is None or role_manifest_for_stage.engineering_only),
        )
        if self._stage_provenance != index.get("stage_provenance"):
            raise ValueError("stage provenance is not canonical")
        if expected_role_manifest is not None:
            if isinstance(expected_role_manifest, Mapping):
                expected_role_manifest = TrainingRoleManifest.from_dict(expected_role_manifest)
            if not isinstance(expected_role_manifest, TrainingRoleManifest):
                raise TypeError("expected_role_manifest must be TrainingRoleManifest or mapping")
            if self._role_manifest is None or self._role_manifest.digest != expected_role_manifest.digest:
                raise ValueError("value-bank role manifest mismatch")
        if expected_baseline_split_hash is not None:
            expected_baseline = _safe_text(expected_baseline_split_hash, "expected_baseline_split_hash", required=True)
            if self._role_manifest is None or self._role_manifest.baseline_split_hash != expected_baseline:
                raise ValueError("value-bank baseline split hash mismatch")
        producer_meta = index["producer"]
        if not isinstance(producer_meta, Mapping) or set(producer_meta) != {"compatibility_hash", "training_role", "compatibility", "source_provenance"}:
            raise ValueError("invalid producer provenance envelope")
        if producer_meta["compatibility_hash"] != self._manifest.producer_compatibility_hash or producer_meta["training_role"] != self._scale.training_role:
            raise ValueError("producer provenance does not match manifest/scale")
        compatibility_payload = producer_meta.get("compatibility")
        if not isinstance(compatibility_payload, Mapping):
            raise ValueError("producer compatibility envelope is missing")
        compatibility_allowed = {field.name for field in fields(ProducerCompatibility)}
        if set(compatibility_payload) != compatibility_allowed:
            raise ValueError("invalid ProducerCompatibility envelope")
        compatibility_data = dict(compatibility_payload)
        compatibility_data["component_versions"] = tuple(tuple(item) for item in compatibility_data.get("component_versions", ()))
        try:
            compatibility = ProducerCompatibility(**compatibility_data)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid ProducerCompatibility envelope") from exc
        if compatibility.digest != self._manifest.producer_compatibility_hash:
            raise ValueError("producer compatibility digest mismatch")
        source_payload = producer_meta.get("source_provenance")
        if source_payload is not None:
            if not isinstance(source_payload, Mapping):
                raise ValueError("source provenance must be a mapping or null")
            source_data = dict(source_payload)
            # SourceProvenance.as_dict intentionally exposes two compatibility
            # aliases (sha256/integrity_verified); remove them only after
            # requiring their values to agree with the canonical fields.
            if "sha256" in source_data and source_data.get("sha256") != source_data.get("checkpoint_sha256"):
                raise ValueError("source provenance sha256 alias mismatch")
            if "integrity_verified" in source_data and source_data.get("integrity_verified") != source_data.get("checkpoint_integrity_verified"):
                raise ValueError("source provenance integrity alias mismatch")
            source_data.pop("sha256", None)
            source_data.pop("integrity_verified", None)
            source_allowed = {field.name for field in fields(SourceProvenance)}
            if set(source_data) != source_allowed:
                raise ValueError("invalid SourceProvenance envelope")
            source_data["details"] = tuple(tuple(item) for item in source_data.get("details", ()))
            try:
                SourceProvenance(**source_data)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid SourceProvenance envelope") from exc
        if not self._engineering_only:
            source_provenance = producer_meta.get("source_provenance")
            if not isinstance(source_provenance, Mapping) or bool(source_provenance.get("synthetic_untrained", True)):
                raise ValueError("MAIN bank producer provenance is missing a trained source")
            if not source_provenance.get("parameter_hash") or not source_provenance.get("frozen_bn_hash") or int(source_provenance.get("traversal_count", 0)) < 1:
                raise ValueError("MAIN bank producer provenance lacks verified parameter/BN/traversal fields")
        self._index = index
        self._rows = self._load_rows()
        self._validate_replay_files()
        if expected_producer is not None:
            self.validate_producer(expected_producer)
        if expected_split_role_hash is not None:
            self.validate_split_role(expected_split_role_hash)

    @property
    def manifest_hash(self) -> str:
        return _sha256_file(self.index_path)

    @property
    def gain_scale(self) -> GainScale:
        return self._scale

    @property
    def index(self) -> Mapping[str, Any]:
        return self._index

    def manifest(self) -> ValueBankManifest:
        return self._manifest

    @property
    def role_manifest(self) -> TrainingRoleManifest | None:
        return self._role_manifest

    @property
    def stage_provenance(self) -> Mapping[str, Any] | None:
        return self._stage_provenance

    @property
    def source_scale_hash(self) -> str | None:
        return self._index["source_scale_hash"]

    def _resolve_child(self, relative: str) -> Path:
        candidate = (self.root / relative)
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("bank path traversal")
        try:
            resolved_root = self.root.resolve()
            resolved = candidate.resolve()
            if os.path.commonpath((str(resolved_root), str(resolved))) != str(resolved_root):
                raise ValueError("bank path escapes root")
        except FileNotFoundError as exc:
            raise ValueError("bank path does not exist") from exc
        if candidate.is_symlink():
            raise ValueError("bank shard symlink/path escape")
        return candidate

    def _validate_loaded_row(self, row: ValueBankRow) -> None:
        if not self._engineering_only:
            if row.engineering_only or row.diagnostic:
                raise ValueError("engineering/diagnostic row cannot enter a MAIN bank")
            missing: set[str] = set()
            for name, value in (
                ("proposal_hash", row.proposal_hash),
                ("state_digest", row.state_digest),
                ("geometry_id", row.geometry_id),
                ("producer_compatibility_hash", row.producer_compatibility_hash),
                ("split_role_hash", row.split_role_hash),
            ):
                try:
                    _safe_text(value, name, required=True)
                except (TypeError, ValueError):
                    missing.add(name)
            if missing:
                raise ValueError(f"MAIN row missing immutable identities: {sorted(missing)}")
        if row.measurement_mode != row.role:
            raise ValueError("row measurement_mode does not match label role")
        if row.role == "screening" and not (self._engineering_only and bool(self._index["status"]["evidence_status"] in {"DIAGNOSTIC_ONLY", "ENGINEERING_ONLY"})):
            raise ValueError("screening labels require a segregated diagnostic bank")
        if row.role in {"exact_footprint", "iid_fixed_q"}:
            if row.support_provenance != SUPPORT_PROVENANCE:
                raise ValueError("row support provenance is not exact complete_support_v1")
            if row.inclusion_mechanism not in _ALLOWED_INCLUSION_MECHANISMS:
                raise ValueError("row inclusion mechanism is not an allowed complete-support law")
            if row.role == "exact_footprint" and row.inclusion_mechanism != SUPPORT_PROVENANCE:
                raise ValueError("exact footprint rows require complete_support_v1 inclusion")
            if row.sampler_law not in _ALLOWED_SAMPLER_LAWS:
                raise ValueError("row sampler_law is not an allowed exact law")
            if any(token in row.inclusion_mechanism.lower() or token in row.sampler_law.lower() for token in ("stopped", "optional", "adaptive")):
                raise ValueError("optionally stopped samples cannot enter a value bank")
        marker = " ".join((row.label_definition, row.sampler_law, row.inclusion_mechanism, row.support_provenance)).lower()
        if any(token in marker for token in _BANNED_LABEL_TOKENS) and not (self._engineering_only and row.diagnostic):
            raise ValueError("target-aware/oracle/hard-mining labels are not valid main rows")
        if self._role_membership and self._role_membership.get(row.subject_key) != row.split_role:
            raise ValueError("row split role does not match role_membership")

    def _load_rows(self) -> tuple[ValueBankRow, ...]:
        shard_specs = self._index["shards"]
        row_specs = self._index["rows"]
        by_shard: dict[str, list[ValueBankRow]] = {}
        for spec in shard_specs:
            if not isinstance(spec, Mapping) or set(spec) != {"path", "sha256", "size", "row_count"}:
                raise ValueError("invalid shard index entry")
            path = self._resolve_child(str(spec["path"]))
            size = path.stat().st_size
            if size > self.max_file_size:
                raise ValueError(f"shard size mismatch or exceeds configured max_file_size: {path.name}")
            actual_hash = _sha256_file(path)
            if actual_hash != str(spec["sha256"]):
                raise ValueError(f"shard checksum mismatch: {path.name}")
            if size != int(spec["size"]):
                raise ValueError(f"shard size mismatch: {path.name}")
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except Exception as exc:
                raise ValueError(f"cannot read value-bank shard {path.name}") from exc
            if not isinstance(payload, Mapping) or set(payload) != {"schema_version", "descriptor_schema", "rows"}:
                raise ValueError("unknown value-bank shard payload")
            if payload["schema_version"] != SHARD_SCHEMA or payload["descriptor_schema"] != DESCRIPTOR_SCHEMA or not isinstance(payload["rows"], list):
                raise ValueError("invalid value-bank shard schema")
            if len(payload["rows"]) != int(spec["row_count"]):
                raise ValueError("shard row count mismatch")
            by_shard[str(spec["path"])] = [_row_from_payload(row) for row in payload["rows"]]
        rows: list[ValueBankRow] = []
        for expected_id, spec in enumerate(row_specs):
            if not isinstance(spec, Mapping) or set(spec) != {"row_id", "shard", "offset", "row_hash", "action_id", "context_id", "subject_id", "split_role"}:
                raise ValueError("invalid row index entry")
            if int(spec["row_id"]) != expected_id:
                raise ValueError("row IDs must be contiguous and ordered")
            shard_rows = by_shard.get(str(spec["shard"]))
            if shard_rows is None:
                raise ValueError("row references unknown shard")
            offset = int(spec["offset"])
            if offset < 0 or offset >= len(shard_rows):
                raise ValueError("row offset out of range")
            row = shard_rows[offset]
            self._validate_loaded_row(row)
            if row_digest(row) != str(spec["row_hash"]):
                raise ValueError(f"row checksum mismatch: {row.action_id}")
            if row.action_id != spec["action_id"] or row.context_id != spec["context_id"] or row.subject_key != spec["subject_id"] or row.split_role != spec["split_role"]:
                raise ValueError("row metadata/index mismatch")
            if row.producer_compatibility_hash != self._manifest.producer_compatibility_hash or row.split_role_hash != self._manifest.split_role_hash:
                raise ValueError("row producer/split identity mismatch")
            rows.append(row)
        if len(rows) != self._manifest.row_count:
            raise ValueError("manifest row count mismatch")
        if len({row.subject_key for row in rows}) != self._manifest.subject_count:
            raise ValueError("manifest subject count mismatch")
        labels = sorted({row.label_definition for row in rows}) or ["empty"]
        if len(labels) > 1:
            raise ValueError("bank rows contain multiple label definitions")
        if _label_identity_hash(labels[0], self._index["producer"]) != self._manifest.label_definition_hash:
            raise ValueError("label definition hash mismatch")
        roles: dict[str, str] = {}
        for row in rows:
            prior = roles.setdefault(row.subject_key, row.split_role)
            if prior != row.split_role:
                raise ValueError("subject overlaps split roles")
            if self._role_membership and self._role_membership.get(row.subject_key) != row.split_role:
                raise ValueError("row split role does not match role_membership")
        return tuple(rows)

    def _validate_replay_files(self) -> None:
        """Validate the optional S2 replay directory against row references.

        Replay snapshots are not a second arbitrary bank payload: every file
        must be a content-addressed ``replay/<sha256>.pt`` referenced by at
        least one immutable row.  The narrow audit helper owns snapshot schema
        and tensor checks; this reader only binds files to bank identities and
        rejects unexpected/unindexed/path-escape entries.
        """

        references: list[str] = []
        for row in self._rows:
            reference = row.selected_replay_ref
            if not reference:
                continue
            relative = Path(reference)
            if (
                relative.is_absolute()
                or relative.parts[:1] != ("replay",)
                or len(relative.parts) != 2
                or relative.suffix != ".pt"
                or re.fullmatch(r"[0-9a-f]{64}", relative.stem) is None
            ):
                raise ValueError("selected replay reference must be replay/<sha256>.pt")
            references.append(relative.as_posix())
        replay_root = self.root / "replay"
        if replay_root.exists() and (replay_root.is_symlink() or not replay_root.is_dir()):
            raise ValueError("value-bank replay path must be a real directory")
        if not references:
            if replay_root.exists():
                children = tuple(replay_root.iterdir())
                if children:
                    raise ValueError("unexpected or unindexed replay artifact")
                # An empty replay directory is itself an unnecessary artifact;
                # reject it to keep publication and evidence manifests exact.
                raise ValueError("unexpected empty replay directory")
            return
        if not replay_root.is_dir():
            raise ValueError("selected replay reference is missing replay directory")
        indexed = set(references)
        actual: set[str] = set()
        for child in replay_root.iterdir():
            if child.is_symlink() or not child.is_file():
                raise ValueError("unexpected or unsafe replay artifact")
            relative = Path("replay") / child.name
            relative_text = relative.as_posix()
            if relative_text not in indexed:
                raise ValueError(f"unindexed replay artifact: {relative_text}")
            if relative.suffix != ".pt" or re.fullmatch(r"[0-9a-f]{64}", relative.stem) is None:
                raise ValueError("selected replay filename must be a SHA256 digest")
            actual.add(relative_text)
        if actual != indexed:
            missing = sorted(indexed - actual)
            raise ValueError(f"selected replay snapshot is missing: {missing}")
        # Import lazily to keep value_bank independent from audit's snapshot
        # implementation at module import time and avoid any import cycle.
        from .bank_audit import validate_snapshot_file, validate_snapshot_row

        expected_producer = self._manifest.producer_compatibility_hash
        expected_split_role = self._manifest.split_role_hash
        rows_by_reference: dict[str, list[ValueBankRow]] = {}
        for row in self._rows:
            if row.selected_replay_ref:
                rows_by_reference.setdefault(row.selected_replay_ref, []).append(row)
        for reference in sorted(indexed):
            path = self.root / Path(reference)
            try:
                resolved_root = self.root.resolve()
                resolved_path = path.resolve()
                if os.path.commonpath((str(resolved_root), str(resolved_path))) != str(resolved_root):
                    raise ValueError("selected replay path escapes bank root")
            except FileNotFoundError as exc:
                raise ValueError("selected replay path does not exist") from exc
            if path.is_symlink() or not path.is_file():
                raise ValueError("selected replay path must be a regular file")
            # Validate/load one file once, then bind every indexed row to its
            # snapshot identity (a selected state may serve multiple actions).
            metadata = validate_snapshot_file(
                path,
                expected_digest=path.stem,
                expected_producer=expected_producer,
                expected_split_role_hash=expected_split_role,
                max_bytes=self.max_file_size,
            )
            for row in rows_by_reference.get(reference, ()):  # defensive; set equality above
                validate_snapshot_row(metadata, row)

    def validate_producer(self, expected: ProducerCompatibility | ProducerDependencies | str) -> None:
        if _producer_hash(expected) != self._manifest.producer_compatibility_hash:
            raise ValueError("value-bank producer compatibility mismatch")

    def validate_split_role(self, expected: str) -> None:
        if _safe_text(expected, "expected_split_role_hash", required=True) != self._manifest.split_role_hash:
            raise ValueError("value-bank split/role hash mismatch")

    def verify(self) -> dict[str, Any]:
        return {
            "status": self._index["status"],
            "row_count": self._manifest.row_count,
            "subject_count": self._manifest.subject_count,
            "manifest_hash": self.manifest_hash,
            "gain_scale_hash": self._manifest.gain_scale_hash,
            "shard_count": len(self._index["shards"]),
            "producer_compatibility_hash": self._manifest.producer_compatibility_hash,
            "split_role_hash": self._manifest.split_role_hash,
        }

    def rows(self, *, split_role: str | None = None, include_diagnostic: bool = True) -> tuple[ValueBankRow, ...]:
        result = self._rows
        if split_role is not None:
            result = tuple(row for row in result if row.split_role == split_role)
        if not include_diagnostic:
            result = tuple(row for row in result if not row.diagnostic)
        return tuple(row.detached() for row in result)

    def iter_rows(self, *, split_role: str | None = None, include_diagnostic: bool = True) -> Iterator[ValueBankRow]:
        yield from self.rows(split_role=split_role, include_diagnostic=include_diagnostic)

    def descriptors(self, variant: int, *, split_role: str | None = None, include_diagnostic: bool = True) -> Tensor:
        if variant not in (126, 222, 270, 366):
            raise ValueError("descriptor variant must be 126, 222, 270, or 366")
        rows = self.rows(split_role=split_role, include_diagnostic=include_diagnostic)
        if not rows:
            return torch.empty((0, variant), dtype=torch.float32)
        return torch.stack([getattr(row, f"v{variant}") for row in rows]).to(dtype=torch.float32)

    def labels(self, *, split_role: str | None = None, include_diagnostic: bool = True) -> Tensor:
        rows = self.rows(split_role=split_role, include_diagnostic=include_diagnostic)
        return torch.tensor([row.raw_gain for row in rows], dtype=torch.float32)

    def training_rows(self) -> tuple[ValueBankRow, ...]:
        return self.rows(split_role=self._scale.training_role, include_diagnostic=False)


ValueBank = ValueBankWriter


__all__ = [
    "BANK_INDEX_NAME",
    "DEFAULT_MAX_FILE_SIZE",
    "GainScale",
    "INDEX_SCHEMA",
    "SCALE_SCHEMA",
    "SHARD_SCHEMA",
    "STAGE_PROVENANCE_SCHEMA",
    "ValueBank",
    "ValueBankReader",
    "ValueBankRow",
    "ValueBankWriter",
    "build_value_bank",
    "compute_gain_scale",
    "row_digest",
]
