"""Training-only positive score calibration for PFGR-Lite adaptive policy."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields, replace
import math
from typing import Any, Literal
from types import MappingProxyType

import torch

from .config import PFGRLiteConfig
from .provenance import canonical_digest
from .types import CompletedBehaviorTrace, GainCalibration, ObservationContext, ProducerDependencies, PFGRRouteResult, TrainingRoleManifest


CALIBRATION_SCHEMA = "pfgr-lite-calibration-fit-v1"
CALIBRATION_EVIDENCE_SCHEMA = "pfgr-lite-calibration-evidence-v1"
MIN_ROLE_SUBJECTS = 32
MIN_ROLE_RECORDS = 64
ALLOWED_MEASUREMENT_ROLES = ("exact_footprint", "iid_fixed_q")


def _complete(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.lower() in {"unknown", "unset", "none", "null"}:
        raise ValueError(f"{name} must be a complete non-sentinel string")
    return value


@dataclass(frozen=True)
class CalibrationEvidence:
    """Sealed calibration receipt required before adaptive deployment.

    Winner rows alone cannot prove that they came from complete forced K4
    target-free routes.  This explicit envelope binds split membership,
    unique row/action identities, completed trace receipts, and the fixed-Q
    confirmation protocol.  ``synthetic`` fixtures are deliberately
    diagnostic-only and can never mint an adaptive deployment calibration.
    """

    baseline_split_hash: str
    producer_fit_subjects: tuple[str, ...]
    fit_subjects: tuple[str, ...]
    allowance_subjects: tuple[str, ...]
    completed_trace_hashes: tuple[str, ...]
    completed_trace_receipts: tuple["TraceReceipt", ...]
    winner_bindings: tuple[tuple[str, str, str, str, int], ...]
    producer_compatibility_hash: str
    value_fit_identity_hash: str
    gain_scale_hash: str
    policy_hash: str
    writer_hash: str
    query_hash: str
    proposal_generator_hash: str
    config_hash: str
    role_manifest: TrainingRoleManifest
    winner_confirmations: tuple[tuple[object, ...], ...] = ()
    # Production calibration distinguishes the adaptive policy identity from
    # the frozen forced-greedy collection policy used to generate traces.
    collection_policy_hash: str = ""
    collection_policy_receipt: Mapping[str, object] | None = None
    # Each sealed receipt is explicitly joined to the subject/context that
    # produced it; winner subject IDs cannot be supplied independently of the
    # observed target-free route.
    trace_subject_bindings: tuple[tuple[str, str, str], ...] = ()
    subject_context_bindings: tuple[Mapping[str, object], ...] = ()
    value_input_variant: int = 366
    trace_budget: int = 4
    confirmation_mode: Literal["exact", "iid_fixed_q"] = "iid_fixed_q"
    confirmation_seed: int | None = None
    confirmation_q_draws: int = 0
    confirmation_independence_hash: str = ""
    fit_role_hash: str = ""
    allowance_role_hash: str = ""
    synthetic: bool = False
    target_free: bool = True
    sealed: bool = True
    version: str = CALIBRATION_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.version != CALIBRATION_EVIDENCE_SCHEMA:
            raise ValueError("unknown calibration evidence schema")
        for name in (
            "baseline_split_hash",
            "producer_compatibility_hash",
            "value_fit_identity_hash",
            "gain_scale_hash",
            "policy_hash",
            "writer_hash",
            "query_hash",
            "proposal_generator_hash",
            "config_hash",
        ):
            _complete(getattr(self, name), name)
        if self.collection_policy_hash:
            _complete(self.collection_policy_hash, "collection_policy_hash")
        if self.collection_policy_receipt is not None:
            if not isinstance(self.collection_policy_receipt, Mapping):
                raise TypeError("collection_policy_receipt must be a mapping or None")
            if self.collection_policy_hash and self.collection_policy_receipt.get("policy_hash") != self.collection_policy_hash:
                raise ValueError("collection policy receipt hash does not match collection policy identity")
        if not isinstance(self.trace_subject_bindings, tuple):
            raise TypeError("trace_subject_bindings must be a tuple")
        seen_trace_subjects: set[str] = set()
        for row in self.trace_subject_bindings:
            if not isinstance(row, tuple) or len(row) != 3:
                raise ValueError("trace_subject_bindings rows must be (trace_hash, subject_id, context_id)")
            trace_hash, subject_id, context_id = row
            _complete(trace_hash, "trace subject trace_hash")
            _complete(subject_id, "trace subject subject_id")
            _complete(context_id, "trace subject context_id")
            if trace_hash in seen_trace_subjects:
                raise ValueError("trace_subject_bindings must contain unique trace hashes")
            seen_trace_subjects.add(trace_hash)
        if not isinstance(self.subject_context_bindings, tuple):
            raise TypeError("subject_context_bindings must be a tuple")
        for binding in self.subject_context_bindings:
            if not isinstance(binding, Mapping):
                raise TypeError("subject_context_bindings rows must be mappings")
            required_binding = {
                "schema_version",
                "subject_id",
                "observation_record_id",
                "context_id",
                "geometry_hash",
                "normalization_hash",
                "binding_digest",
            }
            if set(binding) != required_binding:
                raise ValueError("subject_context_bindings row keys are incomplete or unknown")
            expected_binding_digest = canonical_digest(
                {key: binding[key] for key in required_binding if key != "binding_digest"},
                prefix="pfgr-lite-subject-context-binding-v1|",
            )
            if binding["schema_version"] != "pfgr-lite-subject-context-binding-v1" or binding["binding_digest"] != expected_binding_digest:
                raise ValueError("subject_context_bindings row identity is malformed")
        if not self.synthetic:
            if self.collection_policy_hash and (not self.trace_subject_bindings or {row[0] for row in self.trace_subject_bindings} != set(self.completed_trace_hashes)):
                raise ValueError("trace_subject_bindings must cover every completed trace receipt")
            if self.collection_policy_hash and len(self.subject_context_bindings) != len(self.completed_trace_hashes):
                raise ValueError("subject_context_bindings must cover every completed trace receipt")
        for name in ("producer_fit_subjects", "fit_subjects", "allowance_subjects", "completed_trace_hashes"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values or any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"{name} must be a nonempty tuple of subject/trace identities")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        role_sets = [set(self.producer_fit_subjects), set(self.fit_subjects), set(self.allowance_subjects)]
        if any(role_sets[i] & role_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("producer_fit, calibration_fit, and calibration_allowance subjects must be disjoint")
        # Producer fitting is a disjoint producer-stage role; the contract
        # requires at least one related group, not an invented 32-subject
        # minimum.  Calibration roles retain the explicit row/subject minimum
        # below and additionally require 32 independent related groups for a
        # non-engineering release.
        if not self.producer_fit_subjects:
            raise ValueError("producer_fit must contain at least one subject")
        if len(self.fit_subjects) < MIN_ROLE_SUBJECTS or len(self.allowance_subjects) < MIN_ROLE_SUBJECTS:
            raise ValueError(f"calibration roles require at least {MIN_ROLE_SUBJECTS} distinct subjects")
        if self.value_input_variant not in (126, 222, 270, 366):
            raise ValueError("value_input_variant must be one of 126, 222, 270, 366")
        if self.trace_budget != 4:
            raise ValueError("calibration evidence requires a complete forced K4 trace")
        if self.confirmation_mode not in ("exact", "iid_fixed_q"):
            raise ValueError("confirmation mode must be exact or iid_fixed_q")
        if self.confirmation_mode == "iid_fixed_q":
            if self.confirmation_q_draws < 2 or self.confirmation_seed is None or not isinstance(self.confirmation_seed, int):
                raise ValueError("iid_fixed_q confirmation requires fixed seed and q_draws >= 2")
        elif self.confirmation_q_draws != 0:
            raise ValueError("exact confirmation cannot carry q-draw metadata")
        _complete(self.confirmation_independence_hash, "confirmation_independence_hash")
        if not isinstance(self.role_manifest, TrainingRoleManifest):
            raise TypeError("calibration evidence requires the authoritative TrainingRoleManifest")
        if self.role_manifest.baseline_split_hash != self.baseline_split_hash:
            raise ValueError("role manifest baseline split does not match calibration evidence")
        if set(self.role_manifest.producer_fit_subject_ids) != set(self.producer_fit_subjects) or set(self.role_manifest.calibration_fit_subject_ids) != set(self.fit_subjects) or set(self.role_manifest.calibration_allowance_subject_ids) != set(self.allowance_subjects):
            raise ValueError("role manifest membership does not match calibration evidence")
        if not self.role_manifest.engineering_only:
            groups = dict(self.role_manifest.subject_group_ids)
            for role, subjects in (("producer_fit", self.producer_fit_subjects), ("calibration_fit", self.fit_subjects), ("calibration_allowance", self.allowance_subjects)):
                count = len({groups[subject] for subject in subjects})
                if role == "producer_fit" and count < 1:
                    raise ValueError("production evidence requires at least one producer-fit group")
                if role != "producer_fit" and count < 32:
                    raise ValueError(f"production evidence requires at least 32 independent {role} groups")
        if not isinstance(self.completed_trace_receipts, tuple) or not self.completed_trace_receipts:
            raise ValueError("calibration evidence requires completed forced-trace receipts")
        receipts = {receipt.trace_hash: receipt for receipt in self.completed_trace_receipts}
        if set(receipts) != set(self.completed_trace_hashes):
            raise ValueError("completed trace hashes do not match their structural receipts")
        if not isinstance(self.winner_bindings, tuple) or not self.winner_bindings:
            raise ValueError("calibration evidence requires winner-to-trace bindings")
        keys: list[tuple[str, str]] = []
        for row in self.winner_bindings:
            if not isinstance(row, tuple) or len(row) != 5:
                raise ValueError("winner_bindings rows must be (subject, action, proposal, digest, state)")
            subject, action, proposal, digest, state_version = row
            for name, value in (("binding subject", subject), ("binding action", action), ("binding proposal", proposal), ("binding digest", digest)):
                _complete(value, name)
            if not isinstance(state_version, int) or isinstance(state_version, bool) or state_version < 0:
                raise ValueError("binding state_version must be a nonnegative integer")
            keys.append((subject, action))
        if len(set(keys)) != len(keys):
            raise ValueError("winner_bindings must be unique; duplicate winner rows are not calibration evidence")
        if not isinstance(self.winner_confirmations, tuple):
            raise TypeError("winner_confirmations must be a tuple")
        confirmation_keys: list[tuple[str, str, str, str]] = []
        for row in self.winner_confirmations:
            if not isinstance(row, tuple) or len(row) != 9:
                raise ValueError("winner_confirmations rows must contain winner identity and Q/seed/SE metadata")
            subject, action, proposal, digest, mode, q_draws, seed, standard_error, confirmation_hash = row
            for name, value in (("confirmation subject", subject), ("confirmation action", action), ("confirmation proposal", proposal), ("confirmation digest", digest), ("confirmation mode", mode), ("confirmation hash", confirmation_hash)):
                _complete(value, name)
            if mode not in ALLOWED_MEASUREMENT_ROLES:
                raise ValueError("winner confirmation mode must be exact_footprint or iid_fixed_q")
            if not isinstance(q_draws, int) or isinstance(q_draws, bool) or q_draws < 0:
                raise ValueError("winner confirmation q_draws must be a nonnegative integer")
            if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
                raise ValueError("winner confirmation seed must be an integer or None")
            if standard_error is not None and (not math.isfinite(float(standard_error)) or float(standard_error) < 0.0):
                raise ValueError("winner confirmation standard_error must be finite and nonnegative")
            confirmation_keys.append((str(subject), str(action), str(proposal), str(digest)))
        if len(set(confirmation_keys)) != len(confirmation_keys):
            raise ValueError("winner confirmation identities must be unique")
        binding_identity_keys = {(row[0], row[1], row[2], row[3]) for row in self.winner_bindings}
        if any(key not in binding_identity_keys for key in confirmation_keys):
            raise ValueError("winner confirmation is not bound to a winner action")
        if not self.role_manifest.engineering_only and set(confirmation_keys) != binding_identity_keys:
            raise ValueError("production evidence requires confirmation metadata for every winner")
        receipt_bindings = {
            (proposal, action): index
            for receipt in self.completed_trace_receipts
            for index, (proposal, action) in enumerate(zip(receipt.proposal_digests, receipt.action_digests))
        }
        expected_bindings = sum(len(receipt.proposal_digests) for receipt in self.completed_trace_receipts)
        if len(receipt_bindings) != expected_bindings:
            raise ValueError("completed trace receipts must not reuse proposal/action identities")
        for _, _, proposal, digest, state_version in self.winner_bindings:
            if (proposal, digest) not in receipt_bindings:
                raise ValueError("winner rows are not bound to a completed forced trace receipt")
            if state_version != receipt_bindings[(proposal, digest)]:
                raise ValueError("winner state_version does not match its receipt transition index")
        if not isinstance(self.synthetic, bool) or not isinstance(self.target_free, bool) or not isinstance(self.sealed, bool):
            raise TypeError("calibration evidence boolean flags must be bool")
        if not self.target_free or not self.sealed:
            raise ValueError("calibration evidence must be sealed and target-free")
        if self.fit_role_hash:
            _complete(self.fit_role_hash, "fit_role_hash")
        if self.allowance_role_hash:
            _complete(self.allowance_role_hash, "allowance_role_hash")

    @property
    def deployment_ready(self) -> bool:
        return (
            not self.synthetic
            and self.sealed
            and self.target_free
            and bool(self.fit_role_hash)
            and bool(self.allowance_role_hash)
            and bool(self.collection_policy_hash)
            and self.collection_policy_receipt is not None
            and len(self.trace_subject_bindings) == len(self.completed_trace_hashes)
            and len(self.subject_context_bindings) == len(self.completed_trace_hashes)
            and not self.role_manifest.engineering_only
            and bool(self.winner_confirmations)
            and len(self.winner_confirmations) == len(self.winner_bindings)
            and len(self.winner_bindings) >= 2 * MIN_ROLE_RECORDS
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            field.name: (
                list(getattr(self, field.name))
                if field.name in {"producer_fit_subjects", "fit_subjects", "allowance_subjects", "completed_trace_hashes", "winner_bindings", "winner_confirmations"}
                else getattr(self, field.name).as_dict()
                if field.name == "role_manifest"
                else [receipt.as_dict() for receipt in getattr(self, field.name)]
                if field.name == "completed_trace_receipts"
                else getattr(self, field.name)
            )
            for field in fields(self)
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "CalibrationEvidence":
        if not isinstance(values, Mapping):
            raise TypeError("calibration evidence must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown calibration evidence keys: {sorted(unknown)}")
        parsed = dict(values)
        for name in ("producer_fit_subjects", "fit_subjects", "allowance_subjects", "completed_trace_hashes"):
            if name in parsed:
                parsed[name] = tuple(parsed[name])
        if "winner_bindings" in parsed:
            parsed["winner_bindings"] = tuple(tuple(row) for row in parsed["winner_bindings"])
        if "winner_confirmations" in parsed:
            parsed["winner_confirmations"] = tuple(tuple(row) for row in parsed["winner_confirmations"])
        if "subject_context_bindings" in parsed:
            parsed["subject_context_bindings"] = tuple(dict(row) for row in parsed["subject_context_bindings"])
        if "completed_trace_receipts" in parsed:
            parsed["completed_trace_receipts"] = tuple(TraceReceipt.from_dict(row) for row in parsed["completed_trace_receipts"])
        if not isinstance(parsed.get("role_manifest"), TrainingRoleManifest):
            parsed["role_manifest"] = TrainingRoleManifest.from_dict(parsed["role_manifest"])
        return cls(**parsed)  # type: ignore[arg-type]


_CALIBRATION_EVIDENCE: dict[str, CalibrationEvidence] = {}


def attach_calibration_evidence(calibration: GainCalibration, evidence: CalibrationEvidence) -> GainCalibration:
    if not isinstance(calibration, GainCalibration) or not isinstance(evidence, CalibrationEvidence):
        raise TypeError("attach_calibration_evidence requires GainCalibration and CalibrationEvidence")
    if evidence.producer_compatibility_hash != calibration.producer_compatibility_hash:
        raise ValueError("evidence producer identity does not match calibration")
    if evidence.value_fit_identity_hash != calibration.value_fit_identity_hash or evidence.gain_scale_hash != calibration.gain_scale_hash:
        raise ValueError("evidence value-fit/scale identities do not match calibration")
    if evidence.fit_role_hash and evidence.fit_role_hash != calibration.fit_role_hash:
        raise ValueError("evidence fit role hash does not match calibration")
    if evidence.allowance_role_hash and evidence.allowance_role_hash != calibration.allowance_role_hash:
        raise ValueError("evidence allowance role hash does not match calibration")
    _CALIBRATION_EVIDENCE[canonical_digest(calibration, prefix="pfgr-lite-calibration-v1|")] = evidence
    object.__setattr__(calibration, "_pfgr_calibration_evidence", evidence)
    return calibration


def calibration_evidence(calibration: GainCalibration) -> CalibrationEvidence | None:
    if not isinstance(calibration, GainCalibration):
        raise TypeError("calibration must be GainCalibration")
    attached = getattr(calibration, "_pfgr_calibration_evidence", None)
    if isinstance(attached, CalibrationEvidence):
        return attached
    return _CALIBRATION_EVIDENCE.get(canonical_digest(calibration, prefix="pfgr-lite-calibration-v1|"))


@dataclass(frozen=True)
class TraceReceipt:
    """Compact structural receipt for one complete forced K4 target-free trace."""

    trace_hash: str
    context_id: str
    state_versions: tuple[int, ...]
    proposal_digests: tuple[str, ...]
    action_digests: tuple[str, ...]
    terminal_proposal_digest: str = ""
    terminal_state_digest: str = ""
    terminal_state_version: int | None = None
    terminal_stop_code: str = ""
    sealed: bool = True
    version: str = "pfgr-lite-trace-receipt-v1"

    def __post_init__(self) -> None:
        if self.version != "pfgr-lite-trace-receipt-v1":
            raise ValueError("unknown trace receipt schema")
        for name in ("trace_hash", "context_id"):
            _complete(getattr(self, name), name)
        if not isinstance(self.state_versions, tuple) or not (1 <= len(self.state_versions) <= 5) or self.state_versions != tuple(range(len(self.state_versions))):
            raise ValueError("trace receipt state chain must be contiguous and bounded by K4")
        transition_count = len(self.state_versions) - 1
        if not isinstance(self.proposal_digests, tuple) or len(self.proposal_digests) != transition_count or any(not isinstance(item, str) or not item for item in self.proposal_digests):
            raise ValueError("trace receipt proposal digests must match transition count")
        if not isinstance(self.action_digests, tuple) or len(self.action_digests) != transition_count or any(not isinstance(item, str) or not item for item in self.action_digests):
            raise ValueError("trace receipt action digests must match transition count")
        if len(set(self.proposal_digests)) != transition_count or len(set(self.action_digests)) != transition_count:
            raise ValueError("trace receipt proposal/action digests must be unique")
        terminal_fields = (self.terminal_proposal_digest, self.terminal_state_digest, self.terminal_state_version, self.terminal_stop_code)
        if len(self.state_versions) == 5:
            if any(value not in ("", None) for value in terminal_fields):
                raise ValueError("full K4 receipts may not carry a terminal stop assessment")
        else:
            if not isinstance(self.terminal_proposal_digest, str) or not self.terminal_proposal_digest:
                raise ValueError("short forced routes require a terminal proposal assessment")
            _complete(self.terminal_state_digest, "terminal_state_digest")
            if self.terminal_state_version != transition_count:
                raise ValueError("terminal state version must match the final completed state")
            if self.terminal_stop_code != "no_legal_action":
                raise ValueError("only an actual no-legal terminal assessment can complete a short forced route")
            if self.terminal_proposal_digest in self.proposal_digests:
                raise ValueError("terminal proposal must be distinct from transition proposals")
        if not isinstance(self.sealed, bool) or not self.sealed:
            raise ValueError("trace receipt must be sealed")

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_hash": self.trace_hash,
            "context_id": self.context_id,
            "state_versions": list(self.state_versions),
            "proposal_digests": list(self.proposal_digests),
            "action_digests": list(self.action_digests),
            "terminal_proposal_digest": self.terminal_proposal_digest,
            "terminal_state_digest": self.terminal_state_digest,
            "terminal_state_version": self.terminal_state_version,
            "terminal_stop_code": self.terminal_stop_code,
            "sealed": self.sealed,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "TraceReceipt":
        if not isinstance(values, Mapping):
            raise TypeError("trace receipt must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown trace receipt keys: {sorted(unknown)}")
        parsed = dict(values)
        for name in ("state_versions", "proposal_digests", "action_digests"):
            parsed[name] = tuple(parsed.get(name, ()))
        return cls(**parsed)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ForcedCalibrationTrace:
    """Actual target-free forced collection route bound to its context/policy.

    Production calibration accepts this wrapper rather than a bare route or
    trace.  The wrapper is sealed immediately after the target-free route and
    before any target provider is invoked, so subject/context/producer and the
    exact collection EffectivePolicy cannot be replaced by winner metadata.
    Bare ``CompletedBehaviorTrace``/``PFGRRouteResult`` values remain available
    only to explicit engineering fixtures.  ``subject_id`` and its binding
    receipt are supplied by the W3b data->encode service, never inferred from
    the content-derived context hash.
    """

    observation_context: ObservationContext
    route: PFGRRouteResult
    collection_policy: Any
    subject_id: str
    subject_context_binding: Mapping[str, object]
    receipt: TraceReceipt = field(init=False)

    def __post_init__(self) -> None:
        # Local import avoids the policy -> calibration evidence import cycle.
        from .policy import EffectivePolicy

        if not isinstance(self.observation_context, ObservationContext):
            raise TypeError("ForcedCalibrationTrace requires an ObservationContext")
        if not isinstance(self.route, PFGRRouteResult):
            raise TypeError("ForcedCalibrationTrace requires a PFGRRouteResult")
        if not isinstance(self.collection_policy, EffectivePolicy):
            raise TypeError("ForcedCalibrationTrace requires the resolved EffectivePolicy")
        _complete(self.subject_id, "subject_id")
        if not isinstance(self.subject_context_binding, Mapping):
            raise TypeError("subject_context_binding must be a mapping")
        required_binding = {
            "schema_version",
            "subject_id",
            "observation_record_id",
            "context_id",
            "geometry_hash",
            "normalization_hash",
            "binding_digest",
        }
        if set(self.subject_context_binding) != required_binding:
            raise ValueError("subject_context_binding keys are incomplete or unknown")
        binding = dict(self.subject_context_binding)
        if binding["schema_version"] != "pfgr-lite-subject-context-binding-v1":
            raise ValueError("unknown subject_context_binding schema")
        _complete(binding["subject_id"], "binding subject_id")
        _complete(binding["observation_record_id"], "binding observation_record_id")
        _complete(binding["context_id"], "binding context_id")
        _complete(binding["geometry_hash"], "binding geometry_hash")
        _complete(binding["normalization_hash"], "binding normalization_hash")
        _complete(binding["binding_digest"], "binding_digest")
        expected_binding_digest = canonical_digest(
            {key: binding[key] for key in required_binding if key != "binding_digest"},
            prefix="pfgr-lite-subject-context-binding-v1|",
        )
        if binding["binding_digest"] != expected_binding_digest:
            raise ValueError("subject_context_binding digest mismatch")
        if binding["subject_id"] != self.subject_id:
            raise ValueError("subject_context_binding subject does not match wrapper subject")
        context = self.observation_context
        context.validate_integrity()
        route = self.route
        if route.context_id != context.context_id or route.final_state.context_id != context.context_id:
            raise ValueError("forced calibration route/context IDs do not match")
        if route.final_state.producer is None or not route.final_state.producer.matches(context.producer.compatibility):
            raise ValueError("forced calibration route producer is stale or incompatible with context")
        policy = self.collection_policy
        if policy.mode not in {"forced_diagnostic", "fixed_learned"} or policy.capability != "forced_diagnostic" or policy.budget != 4:
            raise ValueError("forced calibration collection policy must be forced_diagnostic (or explicit fixed-learned equivalent) with budget 4")
        if policy.calibration is not None:
            raise ValueError("forced calibration collection policy must not carry a prior calibration")
        if not policy.engineering_only and (
            policy.value_fit_identity is None
            or not policy.gain_scale_hash
            or policy.gain_scale_provenance is None
        ):
            raise ValueError("production forced collection policy requires exact V identity and GainScale provenance")
        if route.policy_hash != policy.policy_hash:
            raise ValueError("route policy identity does not match the forced collection policy")
        if any(decision.policy_hash != policy.policy_hash for decision in route.decisions):
            raise ValueError("route decisions are not bound to the forced collection policy")
        if route.parallel_trace is not None:
            raise ValueError("parallel diagnostic routes cannot enter sequential calibration")
        receipt = trace_receipt_from_route(route)
        # Route receipts are always complete or a genuine no-legal terminal;
        # arbitrary prefixes cannot be wrapped as calibration evidence.
        if route.stop_reason not in {"budget", "no_legal_action"}:
            raise ValueError("forced collection route has an invalid completion reason")
        if route.stop_reason == "budget" and route.k != 4:
            raise ValueError("budget-complete forced collection route must execute K4")
        if binding["context_id"] != context.context_id:
            raise ValueError("subject_context_binding context does not match ObservationContext")
        expected_geometry = canonical_digest(
            {
                "shape_dhw": context.geometry.shape_dhw,
                "voxel_to_ras_mm": context.geometry.voxel_to_ras_mm,
            },
            prefix="pfgr-lite-subject-geometry-v1|",
        )
        if binding["geometry_hash"] != expected_geometry:
            raise ValueError("subject_context_binding geometry identity does not match context")
        if binding["normalization_hash"] != context.producer.compatibility.observation_normalization_hash:
            raise ValueError("subject_context_binding normalization identity does not match context")
        object.__setattr__(self, "subject_context_binding", MappingProxyType(binding))
        object.__setattr__(self, "receipt", receipt)


def trace_receipt_from_trace(trace: CompletedBehaviorTrace) -> TraceReceipt:
    if not isinstance(trace, CompletedBehaviorTrace):
        raise TypeError("trace must be CompletedBehaviorTrace")
    if len(trace.states) != 5 or len(trace.proposals) != 4 or len(trace.decisions) != 4:
        raise ValueError("calibration requires a completed forced K4 trace")
    if any(decision.stop_code != "continue" for decision in trace.decisions):
        raise ValueError("calibration trace cannot contain a stop decision")
    return TraceReceipt(
        trace_hash=trace.route_hash,
        context_id=trace.context_id,
        state_versions=tuple(state.state_version for state in trace.states),
        proposal_digests=tuple(proposal.proposal_digest for proposal in trace.proposals),
        action_digests=tuple(decision.action_digest for decision in trace.decisions),
    )


def trace_receipt_from_route(route: PFGRRouteResult) -> TraceReceipt:
    """Seal a full K4 route or an actual no-legal early terminal assessment."""

    if not isinstance(route, PFGRRouteResult):
        raise TypeError("route must be PFGRRouteResult")
    if route.parallel_trace is not None:
        raise ValueError("parallel diagnostic traces cannot calibrate sequential policy")
    trace = route.completed_trace
    if trace is None:
        raise ValueError("route lacks a sequential completed trace")
    terminal = route.terminal_proposals
    if terminal is None:
        if route.k != 4 or route.stop_reason != "budget":
            raise ValueError("full forced calibration routes must finish exactly at K4 budget")
        return trace_receipt_from_trace(trace)
    if route.stop_reason != "no_legal_action" or route.k >= 4:
        raise ValueError("short forced calibration routes must stop no_legal_action before K4")
    if not route.decisions or route.decisions[-1].stop_code != "no_legal_action":
        raise ValueError("short forced calibration route requires an actual no-legal terminal decision")
    if terminal.context_id != route.context_id or terminal.state_version != route.final_state.state_version or terminal.state_digest != route.final_state.state_digest:
        raise ValueError("terminal proposal is stale or not bound to the final route state")
    if route.final_state.state_version != len(trace.states) - 1:
        raise ValueError("terminal proposal must assess the state immediately after completed transitions")
    if route.decisions[-1].proposal_digest != terminal.proposal_digest:
        raise ValueError("terminal decision does not bind the stored terminal proposal")
    terminal_legal = terminal.legal.to(dtype=torch.bool) & (terminal.delta.abs().amax(dim=-1) > 0.0)
    if bool(terminal_legal.any()):
        raise ValueError("no-legal terminal assessment must contain no legal nonzero action")
    return TraceReceipt(
        trace_hash=canonical_digest(
            {
                "completed_trace_hash": trace.route_hash,
                "terminal_proposal_digest": terminal.proposal_digest,
                "terminal_state_digest": route.final_state.state_digest,
                "terminal_state_version": route.final_state.state_version,
                "terminal_stop_code": route.decisions[-1].stop_code,
            },
            prefix="pfgr-lite-trace-receipt-v1|",
        ),
        context_id=route.context_id,
        state_versions=tuple(state.state_version for state in trace.states),
        proposal_digests=tuple(proposal.proposal_digest for proposal in trace.proposals),
        action_digests=tuple(decision.action_digest for decision in trace.decisions),
        terminal_proposal_digest=terminal.proposal_digest,
        terminal_state_digest=route.final_state.state_digest,
        terminal_state_version=route.final_state.state_version,
        terminal_stop_code=route.decisions[-1].stop_code,
    )


@dataclass(frozen=True)
class CalibrationWinner:
    """One measured winner from a completed target-free forced route.

    The record carries only identities and a measured signed gain; no target
    tensor or teacher object is accepted by this W4 boundary.  ``role`` is
    fixed before labels are read and must be disjoint by subject.
    """

    subject_id: str
    action_id: str
    proposal_digest: str
    action_digest: str
    raw_score: float
    measured_gain: float
    producer_compatibility_hash: str
    value_fit_identity_hash: str
    gain_scale_hash: str
    policy_hash: str
    writer_hash: str
    query_hash: str
    proposal_generator_hash: str
    role: Literal["calibration_fit", "calibration_allowance"]
    measurement_role: Literal["exact_footprint", "iid_fixed_q"]
    state_version: int = 0
    # These optional fields keep engineering fixtures concise but are required
    # for a production adaptive release.  They link each measured winner to
    # an actual forced trace and its exact/fixed-Q confirmation receipt.
    trace_hash: str = ""
    measurement_mode: str = ""
    q_draws: int = 0
    seed: int | None = None
    standard_error: float | None = None
    confirmation_hash: str = ""
    version: str = CALIBRATION_SCHEMA

    def __post_init__(self) -> None:
        if self.version != CALIBRATION_SCHEMA:
            raise ValueError("unknown calibration winner schema")
        for name in (
            "subject_id",
            "action_id",
            "proposal_digest",
            "action_digest",
            "producer_compatibility_hash",
            "value_fit_identity_hash",
            "gain_scale_hash",
            "policy_hash",
            "writer_hash",
            "query_hash",
            "proposal_generator_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.lower() in {"unknown", "unset", "none", "null"}:
                raise ValueError(f"{name} must be a complete non-sentinel string")
        if self.role not in ("calibration_fit", "calibration_allowance"):
            raise ValueError("calibration winner role must be fit or allowance")
        if self.measurement_role not in ALLOWED_MEASUREMENT_ROLES:
            raise ValueError("calibration requires exact or independent fixed-Q measurements")
        if not isinstance(self.state_version, int) or isinstance(self.state_version, bool) or self.state_version < 0:
            raise ValueError("state_version must be a nonnegative integer")
        for name in ("raw_score", "measured_gain"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.trace_hash and (not isinstance(self.trace_hash, str) or self.trace_hash.lower() in {"unknown", "unset", "none", "null"}):
            raise ValueError("trace_hash must be complete when supplied")
        if self.measurement_mode and self.measurement_mode not in ALLOWED_MEASUREMENT_ROLES:
            raise ValueError("measurement_mode must be exact_footprint or iid_fixed_q")
        if not isinstance(self.q_draws, int) or isinstance(self.q_draws, bool) or self.q_draws < 0:
            raise ValueError("q_draws must be a nonnegative integer")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise ValueError("seed must be an integer or None")
        if self.standard_error is not None and (not math.isfinite(float(self.standard_error)) or float(self.standard_error) < 0.0):
            raise ValueError("standard_error must be finite and nonnegative")
        if self.confirmation_hash and (not isinstance(self.confirmation_hash, str) or self.confirmation_hash.lower() in {"unknown", "unset", "none", "null"}):
            raise ValueError("confirmation_hash must be complete when supplied")

    @property
    def identity(self) -> str:
        return canonical_digest(self, prefix="pfgr-lite-calibration-winner-v1|")


def _coerce_record(value: CalibrationWinner | Mapping[str, object]) -> CalibrationWinner:
    if isinstance(value, CalibrationWinner):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("calibration winners must be CalibrationWinner records or strict mappings")
    allowed = {field for field in CalibrationWinner.__dataclass_fields__}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown calibration winner keys: {sorted(unknown)}")
    return CalibrationWinner(**dict(value))  # type: ignore[arg-type]


def _records(values: Iterable[CalibrationWinner | Mapping[str, object]], *, role: str) -> list[CalibrationWinner]:
    result = [_coerce_record(value) for value in values]
    if not result:
        raise ValueError(f"{role} calibration winners cannot be empty")
    if any(record.role != role for record in result):
        raise ValueError(f"all records in {role} must declare that role before target measurement")
    if any(record.measurement_role not in ALLOWED_MEASUREMENT_ROLES for record in result):
        raise ValueError("screening or optionally stopped labels cannot calibrate adaptive policy")
    identities = [(record.subject_id, record.action_id, record.proposal_digest, record.action_digest) for record in result]
    if len(set(identities)) != len(identities):
        raise ValueError(f"{role} calibration winners must be unique completed actions")
    return result


def _common_identity(records: list[CalibrationWinner], *, config: PFGRLiteConfig) -> dict[str, object]:
    identity = {
        "producer_compatibility_hash": records[0].producer_compatibility_hash,
        "value_fit_identity_hash": records[0].value_fit_identity_hash,
        "gain_scale_hash": records[0].gain_scale_hash,
        "policy_hash": records[0].policy_hash,
        "writer_hash": records[0].writer_hash,
        "query_hash": records[0].query_hash,
        "proposal_generator_hash": records[0].proposal_generator_hash,
        "config_policy": config.policy.as_dict(),
    }
    for record in records[1:]:
        for key in (
            "producer_compatibility_hash",
            "value_fit_identity_hash",
            "gain_scale_hash",
            "policy_hash",
            "writer_hash",
            "query_hash",
            "proposal_generator_hash",
        ):
            if getattr(record, key) != identity[key]:
                raise ValueError(f"calibration identity mismatch in {key}")
    return identity


def _role_hash(role: str, records: list[CalibrationWinner], identity: Mapping[str, object]) -> str:
    return canonical_digest(
        {
            "schema": CALIBRATION_SCHEMA,
            "role": role,
            "identity": dict(identity),
            "subjects": sorted({record.subject_id for record in records}),
            "records": sorted(
                (
                    record.subject_id,
                    record.action_id,
                    record.proposal_digest,
                    record.action_digest,
                    record.state_version,
                    record.measurement_role,
                    record.trace_hash,
                    record.measurement_mode,
                    record.q_draws,
                    record.seed,
                    record.standard_error,
                    record.confirmation_hash,
                )
                for record in records
            ),
        },
        prefix="pfgr-lite-calibration-role-v1|",
    )


def _higher_quantile(values: torch.Tensor, quantile: float) -> float:
    ordered = torch.sort(values.to(dtype=torch.float64)).values
    # NumPy/PyTorch 'higher' interpolation selects the first order statistic
    # at or above q*(n-1), with no tolerance or smoothing.
    index = min(ordered.numel() - 1, max(0, int(math.ceil(quantile * (ordered.numel() - 1)))))
    return float(ordered[index].item())


def fit_calibration(
    completed_fit_winners: Iterable[CalibrationWinner | Mapping[str, object]],
    completed_allowance_winners: Iterable[CalibrationWinner | Mapping[str, object]],
    config: PFGRLiteConfig | Mapping[str, object],
    *,
    evidence: CalibrationEvidence | Mapping[str, object] | None = None,
    completed_traces: Iterable[ForcedCalibrationTrace | CompletedBehaviorTrace | PFGRRouteResult] | None = None,
    producer: ProducerDependencies | None = None,
    collection_policy: Any | None = None,
) -> GainCalibration:
    """Fit ``g ~= a*raw+b`` and a pooled nonnegative q90 allowance.

    Both role collections must be complete forced-route records and have at
    least 32 distinct subjects and 64 winner records.  The fit is float64 and
    clamps only the positive slope constraint ``a >= 1e-6``.
    """

    if isinstance(config, Mapping):
        config = PFGRLiteConfig.from_dict(config)
    if not isinstance(config, PFGRLiteConfig):
        raise TypeError("config must be PFGRLiteConfig or a strict mapping")
    if evidence is None:
        raise ValueError("adaptive calibration requires a sealed CalibrationEvidence envelope")
    if isinstance(evidence, Mapping):
        evidence = CalibrationEvidence.from_dict(evidence)
    if not isinstance(evidence, CalibrationEvidence):
        raise TypeError("evidence must be CalibrationEvidence or a strict mapping")
    actual_traces: tuple[ForcedCalibrationTrace | CompletedBehaviorTrace | PFGRRouteResult, ...] | None = None
    actual_receipts: tuple[TraceReceipt, ...] = ()
    if completed_traces is not None:
        actual_traces = tuple(completed_traces)
        actual_receipts = tuple(
            trace.receipt
            if isinstance(trace, ForcedCalibrationTrace)
            else trace_receipt_from_route(trace)
            if isinstance(trace, PFGRRouteResult)
            else trace_receipt_from_trace(trace)
            for trace in actual_traces
        )
        if actual_receipts != evidence.completed_trace_receipts:
            raise ValueError("supplied completed traces do not match calibration evidence receipts")
    elif not evidence.synthetic:
        raise ValueError("production adaptive calibration requires actual completed forced K4 traces")
    if not evidence.synthetic:
        if producer is None or not isinstance(producer, ProducerDependencies):
            raise ValueError("production adaptive calibration requires actual ProducerDependencies")
        if config.engineering_only or evidence.role_manifest.engineering_only:
            raise ValueError("engineering-only PFGR configuration/roles cannot mint adaptive calibration")
        if config.policy.mode != "adaptive":
            raise ValueError("production calibration config must declare the intended adaptive policy")
        source = producer.source_provenance
        if source.synthetic_untrained or not source.official_pretrained_verified or not source.checkpoint_integrity_verified:
            raise ValueError("production adaptive calibration requires verified official MedicalNet source provenance")
        if producer.compatibility_hash != evidence.producer_compatibility_hash:
            raise ValueError("calibration producer does not match the actual producer dependency envelope")
        from .policy import EffectivePolicy

        if actual_traces is None or not actual_traces or any(not isinstance(trace, ForcedCalibrationTrace) for trace in actual_traces):
            raise ValueError("production calibration requires ForcedCalibrationTrace wrappers sealed from actual routes")
        wrapped_policies = {trace.collection_policy.policy_hash for trace in actual_traces if isinstance(trace, ForcedCalibrationTrace)}
        if len(wrapped_policies) != 1:
            raise ValueError("completed calibration traces must share one forced collection policy")
        wrapped_policy = next(iter(trace.collection_policy for trace in actual_traces if isinstance(trace, ForcedCalibrationTrace)))
        if collection_policy is None:
            collection_policy = wrapped_policy
        if not isinstance(collection_policy, EffectivePolicy):
            raise ValueError("production calibration requires the actual forced collection EffectivePolicy")
        if collection_policy.policy_hash != wrapped_policy.policy_hash:
            raise ValueError("collection policy does not match the sealed trace wrappers")
        if collection_policy.mode not in {"forced_diagnostic", "fixed_learned"} or collection_policy.budget != 4 or collection_policy.capability != "forced_diagnostic":
            raise ValueError("production calibration collection policy must be forced_diagnostic (or explicit fixed-learned equivalent) with budget 4")
        # A production collection route must itself be a real, first-pass
        # forced rollout.  Matching an engineering policy's self-consistent
        # hash is not evidence of an approved V/scale or source envelope.
        if collection_policy.engineering_only:
            raise ValueError("engineering-only collection policy cannot mint production calibration")
        if collection_policy.calibration is not None:
            raise ValueError("forced calibration collection policy must not carry a prior calibration")
        if collection_policy.producer_compatibility_hash != evidence.producer_compatibility_hash:
            raise ValueError("forced collection producer identity does not match calibration evidence")
        if collection_policy.value_fit_identity is None or collection_policy.value_fit_identity.digest != evidence.value_fit_identity_hash:
            raise ValueError("forced collection ValueFitIdentity does not match calibration evidence")
        if not collection_policy.gain_scale_hash or collection_policy.gain_scale_hash != evidence.gain_scale_hash or collection_policy.gain_scale_provenance is None:
            raise ValueError("production forced collection policy requires exact V identity and GainScale provenance")
        # The intended adaptive config and its forced collection control share
        # all route/cost semantics except mode and STOP behavior.  Compare the
        # concrete scalar/enumerated fields rather than trusting a receipt hash
        # that could have been minted from a mismatched config.
        for name in ("revisit", "tie_break", "gain_units", "quality_margin", "compute_cost"):
            if getattr(collection_policy, name) != getattr(config.policy, name):
                raise ValueError(f"forced collection policy {name} does not match intended adaptive config")
        if evidence.collection_policy_hash != collection_policy.policy_hash:
            raise ValueError("calibration evidence collection policy identity does not match the actual forced policy")
        if evidence.collection_policy_receipt is None or dict(evidence.collection_policy_receipt) != collection_policy.as_dict():
            raise ValueError("calibration evidence must retain the complete forced collection policy receipt")
        for wrapped in actual_traces:
            assert isinstance(wrapped, ForcedCalibrationTrace)
            if wrapped.observation_context.producer.source_provenance.digest != producer.source_provenance.digest:
                raise ValueError("sealed route source provenance does not match the actual producer dependency envelope")
    fit = _records(completed_fit_winners, role="calibration_fit")
    allowance = _records(completed_allowance_winners, role="calibration_allowance")
    all_identities = [
        (record.subject_id, record.action_id, record.proposal_digest, record.action_digest)
        for record in fit + allowance
    ]
    if len(set(all_identities)) != len(all_identities):
        raise ValueError("calibration winner records must be globally unique")
    for name, records in (("calibration_fit", fit), ("calibration_allowance", allowance)):
        if len({record.subject_id for record in records}) < MIN_ROLE_SUBJECTS:
            raise ValueError(f"{name} requires at least {MIN_ROLE_SUBJECTS} distinct subjects")
        if len(records) < MIN_ROLE_RECORDS:
            raise ValueError(f"{name} requires at least {MIN_ROLE_RECORDS} winner records")
    fit_subjects = {record.subject_id for record in fit}
    allowance_subjects = {record.subject_id for record in allowance}
    if fit_subjects & allowance_subjects:
        raise ValueError("calibration_fit and calibration_allowance subjects must be disjoint")
    if fit_subjects != set(evidence.fit_subjects) or allowance_subjects != set(evidence.allowance_subjects):
        raise ValueError("calibration evidence split membership does not match winner rows")
    producer_subjects = set(evidence.producer_fit_subjects)
    if producer_subjects & (fit_subjects | allowance_subjects):
        raise ValueError("producer-fit subjects must be disjoint from calibration roles")
    if evidence.config_hash != canonical_digest(config.as_dict(), prefix="pfgr-lite-calibration-config-v1|"):
        raise ValueError("calibration evidence config identity does not match config")
    identity = _common_identity(fit, config=config)
    allowance_identity = _common_identity(allowance, config=config)
    if identity != allowance_identity:
        raise ValueError("fit and allowance records must share producer/value/scale/policy identities")
    for key in (
        "producer_compatibility_hash",
        "value_fit_identity_hash",
        "gain_scale_hash",
        "policy_hash",
        "writer_hash",
        "query_hash",
        "proposal_generator_hash",
    ):
        expected_identity = evidence.collection_policy_hash if key == "policy_hash" and evidence.collection_policy_hash else getattr(evidence, key)
        if identity[key] != expected_identity:
            raise ValueError(f"calibration evidence identity does not match {key}")
    expected_policy_hash = canonical_digest(config.policy, prefix="pfgr-lite-policy-config-v1|")
    if evidence.policy_hash != expected_policy_hash:
        raise ValueError("calibration evidence policy identity does not match config")
    if evidence.value_input_variant not in config.value.input_variants:
        raise ValueError("calibration V variant is not declared by ValueModelConfig")
    if evidence.confirmation_mode == "iid_fixed_q":
        if any(record.measurement_role != "iid_fixed_q" for record in fit + allowance):
            raise ValueError("iid_fixed_q calibration evidence requires fixed-Q winner records")
    elif any(record.measurement_role != "exact_footprint" for record in fit + allowance):
        raise ValueError("exact calibration evidence requires exact-footprint winner records")
    binding_keys = {
        (record.subject_id, record.action_id, record.proposal_digest, record.action_digest, record.state_version)
        for record in fit + allowance
    }
    if binding_keys != set(evidence.winner_bindings):
        raise ValueError("calibration winner rows are not exactly bound to completed traces")

    # A receipt is only a compact revalidation aid after fitting.  Production
    # fitting itself must inspect the immutable trace object and prove that
    # every winner was the selected action of a complete target-free route.
    if not evidence.synthetic:
        assert actual_traces is not None
        traces_by_hash = {receipt.trace_hash: trace for receipt, trace in zip(actual_receipts, actual_traces)}
        if len(traces_by_hash) != len(actual_traces):
            raise ValueError("completed forced traces must have unique route identities")
        subject_bindings = {trace_hash: (subject_id, context_id) for trace_hash, subject_id, context_id in evidence.trace_subject_bindings}
        if set(subject_bindings) != set(traces_by_hash):
            raise ValueError("production calibration traces require complete subject/context bindings")
        expected_trace_subjects = {
            (trace.receipt.trace_hash, trace.subject_id, trace.observation_context.context_id)
            for trace in actual_traces
            if isinstance(trace, ForcedCalibrationTrace)
        }
        if set(evidence.trace_subject_bindings) != expected_trace_subjects:
            raise ValueError("calibration evidence trace subject/context bindings do not match sealed wrappers")
        expected_subject_context_bindings = {
            str(trace.subject_context_binding["binding_digest"]): dict(trace.subject_context_binding)
            for trace in actual_traces
            if isinstance(trace, ForcedCalibrationTrace)
        }
        actual_subject_context_bindings = {
            str(binding["binding_digest"]): dict(binding)
            for binding in evidence.subject_context_bindings
        }
        if actual_subject_context_bindings != expected_subject_context_bindings:
            raise ValueError("calibration evidence subject-context receipts do not match sealed wrappers")
        selected: dict[tuple[str, str], tuple[str, str, int, str, float]] = {}
        for receipt, route_or_trace in zip(actual_receipts, actual_traces):
            if isinstance(route_or_trace, ForcedCalibrationTrace):
                trace = route_or_trace.route.completed_trace
                trace_subject = route_or_trace.subject_id
                trace_context = route_or_trace.observation_context.context_id
            else:
                trace = route_or_trace.completed_trace if isinstance(route_or_trace, PFGRRouteResult) else route_or_trace
                trace_subject = ""
                trace_context = trace.context_id if trace is not None else ""
            if trace is None:
                raise ValueError("calibration route lacks a sequential completed trace")
            if trace.states[0].producer.digest != evidence.producer_compatibility_hash:
                raise ValueError("completed trace producer identity does not match calibration evidence")
            if not trace_subject:
                trace_subject, trace_context = subject_bindings[receipt.trace_hash]
            if trace.context_id != trace_context:
                raise ValueError("trace subject binding does not match the actual route context")
            if any(decision.policy_hash != evidence.collection_policy_hash for decision in trace.decisions):
                raise ValueError("completed trace decisions are not bound to the forced collection policy")
            for proposal, decision in zip(trace.proposals, trace.decisions):
                if decision.stop_code != "continue":
                    raise ValueError("calibration traces must contain only forced continue decisions")
                locations = (proposal.point_ids == decision.selected_point_id).nonzero(as_tuple=False)
                if locations.shape[0] != 1:
                    raise ValueError("trace decision does not select one unique stored action")
                action = proposal.row(int(locations[0, 0]), int(locations[0, 1]))
                key = (proposal.proposal_digest, decision.action_digest)
                if key in selected:
                    raise ValueError("completed traces reuse one proposal/action transition")
                selected[key] = (
                    receipt.trace_hash,
                    action.action_id,
                    proposal.state_version,
                    trace_subject,
                    decision.raw_value,
                )
        winners = fit + allowance
        for record in winners:
            key = (record.proposal_digest, record.action_digest)
            if key not in selected:
                raise ValueError("winner is not the selected action of an actual completed trace")
            trace_hash, action_id, state_version, trace_subject, route_raw_score = selected[key]
            if record.subject_id != trace_subject:
                raise ValueError("winner subject does not match the bound completed trace")
            if record.action_id != action_id:
                raise ValueError("winner action_id does not match its stored trace action")
            if record.state_version != state_version:
                raise ValueError("winner state_version does not match its completed trace transition")
            if record.trace_hash and record.trace_hash != trace_hash:
                raise ValueError("winner trace_hash does not match its completed trace")
            # The fitted raw score is the exact scaled V score retained in the
            # forced route decision; caller-supplied labels may not rewrite it.
            if not math.isclose(record.raw_score, route_raw_score, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("winner raw_score does not match the actual selected route score")
            if record.measurement_mode != evidence.confirmation_mode:
                raise ValueError("winner confirmation mode does not match calibration evidence")
            if evidence.confirmation_mode == "iid_fixed_q":
                if record.q_draws != evidence.confirmation_q_draws or record.seed != evidence.confirmation_seed:
                    raise ValueError("winner fixed-Q seed/draw count does not match calibration evidence")
                if record.standard_error is None or not record.confirmation_hash:
                    raise ValueError("winner fixed-Q confirmation requires standard error and receipt hash")
            elif record.q_draws != 0 or record.seed is not None or (record.standard_error is not None and record.standard_error != 0.0):
                raise ValueError("exact confirmation cannot carry fixed-Q metadata")
        confirmations = sorted(
            (record.subject_id, record.action_id, record.proposal_digest, record.action_digest, record.measurement_mode, record.q_draws, record.seed, record.standard_error, record.confirmation_hash)
            for record in winners
        )
        expected_confirmation_hash = canonical_digest(confirmations, prefix="pfgr-lite-confirmation-set-v1|")
        if evidence.confirmation_independence_hash != expected_confirmation_hash:
            raise ValueError("confirmation independence identity is not bound to winner metadata")

    x = torch.tensor([record.raw_score for record in fit], dtype=torch.float64)
    y = torch.tensor([record.measured_gain for record in fit], dtype=torch.float64)
    centered = x - x.mean()
    denominator = torch.sum(centered.square())
    if not bool(torch.isfinite(denominator)) or float(denominator.item()) <= 0.0:
        raise ValueError("calibration raw scores are degenerate; constrained OLS is undefined")
    slope = torch.sum(centered * (y - y.mean())) / denominator
    slope = torch.clamp(slope, min=1e-6)
    intercept = y.mean() - slope * x.mean()
    if not bool(torch.isfinite(slope)) or not bool(torch.isfinite(intercept)):
        raise ValueError("calibration OLS produced nonfinite parameters")
    allowance_errors = torch.tensor(
        [float(slope.item()) * record.raw_score + float(intercept.item()) - record.measured_gain for record in allowance],
        dtype=torch.float64,
    )
    if not bool(torch.isfinite(allowance_errors).all()):
        raise ValueError("allowance residuals must be finite")
    empirical_allowance = max(0.0, _higher_quantile(allowance_errors, 0.90))
    fit_role_hash = _role_hash("calibration_fit", fit, identity)
    allowance_role_hash = _role_hash("calibration_allowance", allowance, allowance_identity)
    result = GainCalibration(
        a=float(slope.item()),
        b=float(intercept.item()),
        allowance=empirical_allowance,
        quantile=0.90,
        quantile_method="higher",
        fit_role_hash=fit_role_hash,
        allowance_role_hash=allowance_role_hash,
        fit_count=len(fit),
        allowance_count=len(allowance),
        producer_compatibility_hash=str(identity["producer_compatibility_hash"]),
        value_fit_identity_hash=str(identity["value_fit_identity_hash"]),
        gain_scale_hash=str(identity["gain_scale_hash"]),
        capability="diagnostic" if evidence.synthetic else "adaptive",
    )
    evidence = replace(evidence, fit_role_hash=fit_role_hash, allowance_role_hash=allowance_role_hash)
    attach_calibration_evidence(result, evidence)
    return result


__all__ = [
    "ALLOWED_MEASUREMENT_ROLES",
    "CALIBRATION_SCHEMA",
    "CALIBRATION_EVIDENCE_SCHEMA",
    "CalibrationEvidence",
    "CalibrationWinner",
    "ForcedCalibrationTrace",
    "MIN_ROLE_RECORDS",
    "MIN_ROLE_SUBJECTS",
    "attach_calibration_evidence",
    "calibration_evidence",
    "TraceReceipt",
    "trace_receipt_from_route",
    "trace_receipt_from_trace",
    "fit_calibration",
]
