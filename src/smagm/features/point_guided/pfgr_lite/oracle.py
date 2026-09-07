"""Privileged target-aware PFGR-Lite oracle diagnostics.

Oracle code is deliberately separate from deployment inference. It first
completes the target-free route/candidate generation, then joins one immutable
W2 target context and evaluates exact or fixed-Q effects. ``OracleContext``
and ``OracleResult`` are diagnostic-only types and are never accepted by W4
inference, checkpoint, or value-bank APIs.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal

from torch import Tensor

from .experiments import (
    ExperimentOptions,
    _build_lattice,
    _completed_trace,
    _context_for_sample,
    _counter_metadata,
    _invoke,
    _jsonable,
    _load_policy,
    _prediction_for,
    _record_dict,
    _route_for_sample,
    _sample_id,
    _service_execution,
    _target_join,
)
from .metrics import action_metric_row, paired_subject_metrics, scientific_decision
from .provenance import canonical_digest, tensor_digest

ORACLE_OPTIONS_SCHEMA = "pfgr-lite-oracle-options-v1"
ORACLE_MODES = ("sampled_one", "greedy", "all_exact_one")


def _positive_int(name: str, value: object, *, allow_zero: bool = False, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds bound {maximum}")
    return int(value)


def _finite_nonnegative(name: str, value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class OracleOptions:
    """Strict options for a bounded privileged diagnostic."""

    mode: Literal["sampled_one", "greedy", "all_exact_one"] = "sampled_one"
    budget: int = 1
    candidate_count: int = 32
    teacher_mode: Literal["exact_footprint", "iid_fixed_q"] = "exact_footprint"
    query_count: int = 1024
    max_subjects: int = 1
    seed: int = 20260907
    split_role: str = "validation"
    confirmation_mode: Literal["none", "exact_footprint", "iid_fixed_q"] = "none"
    confirmation_query_count: int = 1024
    numerical_tolerance: float = 1e-10
    practical_margin: float = 0.0
    engineering_only: bool = False
    oracle_mode: str | None = None
    schema_version: str = ORACLE_OPTIONS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ORACLE_OPTIONS_SCHEMA:
            raise ValueError("unknown OracleOptions schema")
        mode = self.mode if self.oracle_mode is None else self.oracle_mode
        if self.oracle_mode is not None and self.mode != "sampled_one" and self.oracle_mode != self.mode:
            raise ValueError("oracle_mode conflicts with explicit mode")
        if mode not in ORACLE_MODES:
            raise ValueError(f"mode must be one of {ORACLE_MODES}")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "oracle_mode", mode)
        if self.budget not in (0, 1, 2, 4):
            raise ValueError("budget must be one of 0, 1, 2, or 4")
        if mode in ("sampled_one", "all_exact_one") and self.budget not in (0, 1):
            raise ValueError(f"{mode} oracle budget must be 0 or 1")
        _positive_int("candidate_count", self.candidate_count, maximum=2048)
        if self.teacher_mode not in ("exact_footprint", "iid_fixed_q"):
            raise ValueError("teacher_mode must be exact_footprint or iid_fixed_q")
        _positive_int("query_count", self.query_count, maximum=10_000_000)
        if self.teacher_mode == "iid_fixed_q" and self.query_count < 2:
            raise ValueError("iid_fixed_q screening requires query_count >= 2")
        if mode == "all_exact_one" and self.teacher_mode != "exact_footprint":
            raise ValueError("all_exact_one requires exact_footprint teacher mode")
        _positive_int("max_subjects", self.max_subjects, maximum=1_000_000)
        _positive_int("seed", self.seed, allow_zero=True, maximum=2**63 - 1)
        if not isinstance(self.split_role, str) or not self.split_role.strip():
            raise ValueError("split_role must be nonempty")
        if self.confirmation_mode not in ("none", "exact_footprint", "iid_fixed_q"):
            raise ValueError("confirmation_mode must be none, exact_footprint, or iid_fixed_q")
        _positive_int("confirmation_query_count", self.confirmation_query_count, maximum=10_000_000)
        if self.confirmation_mode == "iid_fixed_q" and self.confirmation_query_count < 2:
            raise ValueError("iid_fixed_q confirmation requires at least two draws")
        if self.teacher_mode == "iid_fixed_q" and self.confirmation_mode == "none":
            raise ValueError("iid_fixed_q screening requires independent confirmation")
        _finite_nonnegative("numerical_tolerance", self.numerical_tolerance)
        _finite_nonnegative("practical_margin", self.practical_margin)
        if not isinstance(self.engineering_only, bool):
            raise TypeError("engineering_only must be bool")

    def as_dict(self) -> dict[str, Any]:
        return {field.name: _jsonable(getattr(self, field.name)) for field in fields(self)}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> OracleOptions:
        if not isinstance(values, Mapping):
            raise TypeError("OracleOptions must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown OracleOptions keys: {sorted(unknown)}")
        if (
            "mode" in values
            and "oracle_mode" in values
            and values["mode"] != values["oracle_mode"]
        ):
            raise ValueError("oracle_mode conflicts with explicit mode")
        return cls(**dict(values))


@dataclass(frozen=True)
class OracleContext:
    """Privileged target-aware context; deployment APIs must reject it."""

    observation_context: object
    target_context: object
    route: object
    scope: str
    target_aware: bool = True
    diagnostic_only: bool = True
    schema_version: str = "pfgr-lite-oracle-context-v1"
    context_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != "pfgr-lite-oracle-context-v1":
            raise ValueError("unknown OracleContext schema")
        if not isinstance(self.scope, str) or not self.scope.strip():
            raise ValueError("OracleContext scope must be nonempty")
        if not self.target_aware or not self.diagnostic_only:
            raise ValueError("OracleContext is always target-aware diagnostic-only")
        if self.observation_context is self.target_context:
            raise ValueError("OracleContext keeps target context separate from observation context")
        expected = canonical_digest(
            {
                "schema_version": self.schema_version,
                "context_id": getattr(self.observation_context, "context_id", None),
                "route_hash": getattr(self.route, "route_hash", getattr(self.route, "policy_hash", None)),
                "scope": self.scope,
            },
            prefix="pfgr-lite-oracle-context-v1|",
        )
        if self.context_digest and self.context_digest != expected:
            raise ValueError("OracleContext digest does not match diagnostic identities")
        object.__setattr__(self, "context_digest", expected)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "observation_context_id": getattr(self.observation_context, "context_id", None),
            "route_hash": getattr(self.route, "route_hash", getattr(self.route, "policy_hash", None)),
            "scope": self.scope,
            "target_aware": self.target_aware,
            "diagnostic_only": self.diagnostic_only,
            "context_digest": self.context_digest,
        }


@dataclass(frozen=True)
class OracleResult:
    """Privileged per-subject result with explicit candidate scope."""

    subject_id: str
    mode: str
    budget: int
    candidate_scope: str
    rows: tuple[Mapping[str, Any], ...] = ()
    selected_action_ids: tuple[str, ...] = ()
    stop_reason: str = "budget"
    confirmation: tuple[Mapping[str, Any], ...] = ()
    context_digest: str = ""
    target_aware: bool = True
    diagnostic_only: bool = True
    schema_version: str = "pfgr-lite-oracle-result-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "pfgr-lite-oracle-result-v1":
            raise ValueError("unknown OracleResult schema")
        if not self.subject_id:
            raise ValueError("subject_id must be nonempty")
        if self.mode not in ORACLE_MODES:
            raise ValueError("unknown oracle mode")
        if self.budget not in (0, 1, 2, 4):
            raise ValueError("oracle budget must be one of 0, 1, 2, or 4")
        if not self.candidate_scope:
            raise ValueError("candidate_scope must be explicit")
        if not self.target_aware or not self.diagnostic_only:
            raise ValueError("OracleResult is always target-aware diagnostic-only")
        if any(not isinstance(row, Mapping) for row in self.rows + self.confirmation):
            raise TypeError("oracle rows must be mappings")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "subject_id": self.subject_id,
            "mode": self.mode,
            "budget": self.budget,
            "candidate_scope": self.candidate_scope,
            "rows": [_jsonable(dict(row)) for row in self.rows],
            "selected_action_ids": list(self.selected_action_ids),
            "stop_reason": self.stop_reason,
            "confirmation": [_jsonable(dict(row)) for row in self.confirmation],
            "context_digest": self.context_digest,
            "target_aware": self.target_aware,
            "diagnostic_only": self.diagnostic_only,
        }


def _route_attr(route: object, name: str, default: Any = None) -> Any:
    if isinstance(route, Mapping):
        return route.get(name, default)
    return getattr(route, name, default)


def _candidate_rows(
    inputs: Any,
    route: object,
    context: object | None,
    state: object,
    state_index: int,
    options: OracleOptions,
    proposal: object | None = None,
) -> list[object]:
    if proposal is not None:
        if hasattr(proposal, "row") and hasattr(proposal, "point_ids"):
            return [proposal.row(0, index) for index in range(proposal.point_ids.shape[1])]
        if isinstance(proposal, Sequence) and not isinstance(proposal, (str, bytes)):
            return list(proposal)
    builder = getattr(inputs, "proposal_builder", None)
    if builder is not None:
        rows = _invoke(
            builder,
            route,
            state,
            state_index=state_index,
            context=context,
            observation_context=context,
            seed=options.seed + state_index,
            candidate_count=options.candidate_count,
            oracle=True,
        )
        if isinstance(rows, Mapping):
            rows = rows.get("candidates", rows.get("proposals", ()))
        return list(rows)
    trace = _completed_trace(route)
    proposals = tuple(_route_attr(trace, "proposals", ())) if trace is not None else ()
    if state_index < len(proposals):
        proposal = proposals[state_index]
        # Return the complete stored proposal batch.  ``sampled_one`` and
        # bounded greedy scopes apply their declared candidate_count sampling
        # after this function; ``all_exact_one`` consequently remains truly
        # all-N rather than silently auditing a truncated prefix.
        return [proposal.row(0, index) for index in range(proposal.point_ids.shape[1])]
    terminal = _route_attr(route, "terminal_proposals")
    if terminal is not None and state_index == 0:
        return [terminal.row(0, index) for index in range(terminal.point_ids.shape[1])]
    return []


def _route_initial_state(route: object) -> object | None:
    """Get the frozen initial state without treating a random route as oracle history."""

    state = _route_attr(route, "initial_state")
    if state is not None:
        return state
    trace = _completed_trace(route)
    states = tuple(_route_attr(trace, "states", ())) if trace is not None else tuple(_route_attr(route, "states", ()))
    if states:
        return states[0]
    return _route_attr(route, "final_state")


def _oracle_continue_decision(proposal: object, action: object, policy: object | None, *, step: int) -> object:
    from .types import Decision

    if policy is None:
        raise ValueError("typed oracle execution requires the effective W4 policy")
    return Decision(
        selected_point_id=int(action.point_id),
        proposal_digest=str(proposal.proposal_digest),
        action_digest=str(action.action_digest),
        active=True,
        raw_value=0.0,
        calibrated_value=0.0,
        conservative_value=0.0,
        allowance=float(getattr(getattr(policy, "calibration", None), "allowance", 0.0)),
        quality_margin=float(getattr(policy, "quality_margin", 0.0)),
        compute_cost=float(getattr(policy, "compute_cost", 0.0)),
        policy_hash=str(getattr(policy, "policy_hash", "")),
        stop_code="continue",
        step=int(step),
    )


def _oracle_proposals(
    inputs: Any,
    context: object | None,
    state: object,
    route: object,
    query: object | None,
    writer: object | None,
    lattice: object | None,
    options: OracleOptions,
    *,
    state_index: int,
    policy: object | None,
) -> object | list[object]:
    """Generate one fresh candidate bank for the current oracle state."""

    metadata = getattr(inputs, "metadata", {})
    builder = getattr(inputs, "proposal_builder", None)
    if builder is not None:
        result = _invoke(
            builder,
            route,
            state,
            state_index=state_index,
            context=context,
            observation_context=context,
            seed=options.seed + state_index,
            candidate_count=options.candidate_count,
            oracle=True,
        )
        if isinstance(result, Mapping):
            result = result.get("proposals", result.get("candidates", result))
        return result
    if context is None or query is None:
        raise ValueError("oracle proposal generation requires ObservationContext and canonical query")
    model = getattr(inputs, "model", None)
    updater = getattr(model, "updater", None)
    if updater is None:
        raise ValueError("oracle proposal generation requires the model updater")
    from .action_proposal import propose_actions

    legal_mask = metadata.get("legal_mask") if isinstance(metadata, Mapping) else None
    if legal_mask is None:
        from .sparse_write import make_support_legal_mask

        if policy is None:
            raise ValueError("oracle proposal generation requires effective policy or legal-mask injection")
        if lattice is None:
            raise ValueError("oracle proposal generation requires the canonical query lattice")
        legal_mask = make_support_legal_mask(lattice)
    if not callable(writer):
        raise TypeError("oracle proposal generation requires canonical ActionWriter")
    return propose_actions(
        updater,
        state,
        context,
        query=query,
        candidate_chunk_size=max(1, options.candidate_count),
        counters=_route_attr(route, "counters", None),
        legal_mask=legal_mask,
        query_version=getattr(query, "query_version", None),
        query_hash=getattr(query, "query_hash", None),
        writer_version=getattr(writer, "writer_version", None),
        writer_hash=getattr(writer, "writer_hash", None),
    )


def _oracle_advance(
    inputs: Any,
    state: object,
    context: object | None,
    proposal: object,
    action: object,
    decision: object,
    *,
    writer: object | None,
    state_index: int,
    route: object,
    target_context: object,
) -> object:
    """Apply exactly the winning stored proposal from the current state."""

    apply_callback = None
    metadata = getattr(inputs, "metadata", {})
    if isinstance(metadata, Mapping):
        apply_callback = metadata.get("oracle_apply")
    if apply_callback is None:
        apply_callback = getattr(inputs, "oracle_apply", None)
    if apply_callback is not None:
        return _invoke(
            apply_callback,
            state,
            action,
            state_index=state_index,
            context=context,
            observation_context=context,
            target_context=target_context,
            route=route,
        )
    from .action_proposal import apply_scored_action

    if context is None or writer is None:
        raise ValueError("oracle winner execution requires typed ObservationContext and ActionWriter")
    return apply_scored_action(
        state,
        context,
        proposal,
        decision,
        writer=writer,
        counters=_route_attr(route, "counters", None),
    )


def _action_id(action: object, index: int) -> str:
    value = getattr(action, "action_id", None)
    if isinstance(action, Mapping):
        value = action.get("action_id", action.get("id", value))
    return str(value) if value else f"oracle-action-{index}"


def _measure_candidates(
    inputs: Any,
    route: object,
    candidates: Sequence[object],
    target_context: object,
    context: object | None,
    lattice: object | None,
    options: OracleOptions,
    *,
    seed: int,
    diagnostic_state: object | None = None,
) -> list[Mapping[str, Any]]:
    if not candidates:
        return []
    effect = getattr(inputs, "effect_measure", None)
    if effect is not None:
        measured = _invoke(
            effect,
            route,
            candidates,
            target_context,
            trace=_completed_trace(route),
            proposals=candidates,
            target_context=target_context,
            observation_context=context,
            seed=seed,
            candidate_scope="oracle",
            state=diagnostic_state,
            config=getattr(getattr(inputs, "execution", None), "config", getattr(inputs, "config", None)),
        )
        if measured is None:
            return []
        values = list(measured) if isinstance(measured, Iterable) and not isinstance(measured, (str, bytes, Mapping)) else [measured]
        return [_record_dict(item) for item in values]
    decoder = getattr(getattr(inputs, "model", None), "decoder", None)
    config = getattr(getattr(inputs, "execution", None), "config", getattr(inputs, "config", None))
    if decoder is None or lattice is None:
        raise ValueError("oracle candidate measurement requires effect_measure or W2 decoder/lattice")
    if diagnostic_state is None:
        raise ValueError("oracle candidate measurement requires the actual current PFGRState")
    from .teacher import measure_diagnostic_actions
    from .types import PFGRState

    if not isinstance(diagnostic_state, PFGRState):
        raise TypeError("typed oracle candidate measurement requires PFGRState")

    teacher_config = getattr(config, "teacher", None)
    from .config import EffectTeacherConfig

    if teacher_config is None:
        teacher_config = EffectTeacherConfig(
            mode=options.teacher_mode, q_draws=options.query_count
        )
    elif teacher_config.mode != options.teacher_mode or (
        options.teacher_mode == "iid_fixed_q"
        and teacher_config.q_draws != options.query_count
    ):
        # The service option is the declared scientific query budget; do not
        # silently inherit a larger config Q or report a different draw count.
        teacher_config = replace(
            teacher_config,
            mode=options.teacher_mode,
            q_draws=max(2, options.query_count),
        )
    measured = measure_diagnostic_actions(
        diagnostic_state,
        candidates,
        target_context,
        decoder,
        teacher_config,
        lattice=lattice,
        chunk_size=getattr(config, "decode_chunk_size", 1024),
        candidate_chunk_size=1,
        seed=seed,
        observation_context=context,
        counters=_route_attr(route, "counters", None),
    )
    rows: list[dict[str, Any]] = []
    for result in measured:
        row = _record_dict(result.label)
        row.update(
            {
                "diagnostic_scope": result.scope,
                "diagnostic_privileged": result.privileged,
                "diagnostic_schema_version": result.schema_version,
                "state_version": result.state_version,
                "state_digest": result.state_digest,
                "proposal_digest": result.proposal_digest,
                "target_context_digest": result.target_context_digest,
                "action_digest": result.action.action_digest,
            }
        )
        rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(dict(row)), sort_keys=True) + "\n")


def _oracle_source_receipt(
    inputs: Any,
    context: object | None,
    options: OracleOptions,
    *,
    subject_id: str | None = None,
) -> dict[str, Any]:
    """Capture actual checkpoint/initialization and context provenance."""

    metadata = getattr(inputs, "metadata", {})
    provided = (
        metadata.get("source_receipt", metadata.get("provenance", {}))
        if isinstance(metadata, Mapping)
        else {}
    )
    role_manifest = getattr(inputs, "role_manifest", None)
    producer = getattr(context, "producer", None)
    compatibility = getattr(producer, "compatibility", producer)
    initial_planes = getattr(context, "initial_planes", None)
    actual_baseline_split_hash = (
        getattr(role_manifest, "baseline_split_hash", None)
        if role_manifest is not None
        else None
    )
    actual_training_role_manifest_hash = (
        getattr(role_manifest, "digest", None) if role_manifest is not None else None
    )
    baseline_split_hash = actual_baseline_split_hash
    if baseline_split_hash is None and isinstance(metadata, Mapping):
        baseline_split_hash = metadata.get("baseline_split_hash")
    if baseline_split_hash is None and isinstance(provided, Mapping):
        baseline_split_hash = provided.get("baseline_split_hash")
    training_role_manifest_hash = actual_training_role_manifest_hash
    if training_role_manifest_hash is None and isinstance(metadata, Mapping):
        training_role_manifest_hash = metadata.get(
            "training_role_manifest_hash", metadata.get("role_manifest_hash")
        )
    if training_role_manifest_hash is None and isinstance(provided, Mapping):
        training_role_manifest_hash = provided.get(
            "training_role_manifest_hash", provided.get("role_manifest_hash")
        )
    receipt: dict[str, Any] = {
        "schema_version": ORACLE_OPTIONS_SCHEMA,
        "producer_compatibility_hash": getattr(
            compatibility,
            "digest",
            getattr(producer, "compatibility_hash", None),
        ),
        "normalization_hash": getattr(
            compatibility,
            "observation_normalization_hash",
            getattr(producer, "observation_normalization_hash", None),
        ),
        "initialization_hash": (
            canonical_digest(
                {
                    "context_id": getattr(context, "context_id", None),
                    "planes": _jsonable(initial_planes),
                },
                prefix="pfgr-lite-initialization-v1|",
            )
            if initial_planes is not None
            else None
        ),
        "baseline_split_hash": baseline_split_hash,
        "training_role_manifest_hash": training_role_manifest_hash,
        "split_role": options.split_role,
        "mask_definition": "observation_derived_binary",
        "label_definition": "masked_charbonnier_global_v1",
        "loss_definition": "masked_charbonnier_global_v1",
        "data_range": 1.0,
        "context_id": getattr(context, "context_id", None),
        "engineering_only": options.engineering_only,
    }
    if isinstance(provided, Mapping):
        for key in (
            "source_hash",
            "dirty_hash",
            "checkpoint_hash",
            "initialization_hash",
            "subject_initialization_hashes",
            "baseline_split_hash",
            "training_role_manifest_hash",
            "split_role_hash",
            "normalization_hash",
            "producer_compatibility_hash",
            "mask_definition",
            "label_definition",
            "loss_definition",
            "data_range",
        ):
            if key in provided:
                supplied = _jsonable(provided[key])
                actual = receipt.get(key)
                if key == "subject_initialization_hashes":
                    expected = (
                        {subject_id: receipt["initialization_hash"]}
                        if subject_id is not None and receipt.get("initialization_hash") is not None
                        else {}
                    )
                    if supplied != expected:
                        raise ValueError(
                            "source receipt 'subject_initialization_hashes' conflicts with sealed context"
                        )
                    continue
                if actual is not None and supplied != actual:
                    raise ValueError(
                        f"source receipt {key!r} conflicts with sealed service identity"
                    )
                if actual is None:
                    receipt[key] = supplied
    if isinstance(metadata, Mapping):
        for key in (
            "producer_compatibility_hash",
            "normalization_hash",
            "initialization_hash",
            "subject_initialization_hashes",
            "baseline_split_hash",
            "training_role_manifest_hash",
            "role_manifest_hash",
            "split_role",
            "mask_definition",
            "label_definition",
            "loss_definition",
            "data_range",
        ):
            if key not in metadata:
                continue
            supplied = _jsonable(metadata[key])
            normalized_key = (
                "training_role_manifest_hash" if key == "role_manifest_hash" else key
            )
            if key == "subject_initialization_hashes":
                expected = (
                    {subject_id: receipt["initialization_hash"]}
                    if subject_id is not None and receipt.get("initialization_hash") is not None
                    else {}
                )
                if supplied != expected:
                    raise ValueError(
                        "metadata 'subject_initialization_hashes' conflicts with sealed context"
                    )
                continue
            actual = receipt.get(normalized_key)
            if actual is not None and supplied != actual:
                raise ValueError(
                    f"metadata {key!r} conflicts with sealed service identity"
                )
    return receipt


def run_oracle_evaluation(inputs: Any, options: OracleOptions, output_dir: Path) -> Mapping[str, Any]:
    """Execute privileged diagnostics under detached/eval service semantics."""

    with _service_execution(getattr(inputs, "model", None)):
        return _run_oracle_evaluation_impl(inputs, options, output_dir)


def _run_oracle_evaluation_impl(inputs: Any, options: OracleOptions, output_dir: Path) -> Mapping[str, Any]:
    """Execute bounded privileged diagnostics and write ``privileged_oracle.jsonl``."""

    from .stages import StageInputs

    if not isinstance(inputs, StageInputs):
        raise TypeError("inputs must be StageInputs")
    if not isinstance(options, OracleOptions):
        raise TypeError("options must be OracleOptions")
    destination = Path(output_dir)
    if destination.exists():
        if not destination.is_dir():
            raise ValueError("output_dir must be a directory")
        if any(destination.iterdir()):
            raise FileExistsError(f"output_dir must be empty and exclusive: {destination}")
    else:
        destination.mkdir(parents=True, exist_ok=False)
    path = destination / "privileged_oracle.jsonl"
    samples = tuple(getattr(inputs, "samples", ()))[: options.max_subjects]
    if not samples:
        raise ValueError("oracle evaluation requires at least one target-free sample")
    # The deployment route is used only to seal the initial target-free
    # context/state and resolve W2/W4 dependencies.  Oracle winners are then
    # generated, measured, and applied in a separate privileged loop; no
    # random route history is reused as oracle state.
    experiment_options = ExperimentOptions(
        scenario="static",
        budget=0,
        max_subjects=options.max_subjects,
        seed=options.seed,
        split_role=options.split_role,
        teacher_mode=options.teacher_mode,
        query_count=options.query_count,
        numerical_tolerance=options.numerical_tolerance,
        practical_margin=options.practical_margin,
        engineering_only=options.engineering_only,
    )
    rows_out: list[dict[str, Any]] = []
    all_label_rows: list[dict[str, Any]] = []
    pipeline_counter_rows: list[dict[str, Any]] = []
    source_receipts: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        subject_id = _sample_id(sample, sample_index)
        pipeline_counters: dict[str, int] = {
            "observation_encode_calls": 0,
            "initial_decode_calls": 0,
            "final_decode_calls": 0,
        }
        context = _context_for_sample(
            inputs, sample, pipeline_counters=pipeline_counters
        )
        config = getattr(getattr(inputs, "execution", None), "config", getattr(inputs, "config", None))
        lattice = _build_lattice(inputs, context, getattr(inputs, "model", None), config)
        policy = _load_policy(inputs, context, experiment_options, config)
        route, query, writer = _route_for_sample(inputs, sample, context, experiment_options, config, lattice, policy)
        current_state = _route_initial_state(route)
        if current_state is None:
            raise ValueError("oracle route must expose an initial target-free state")

        # The first bank is generated before the target callback.  This is the
        # exact target-free boundary; later banks are generated immediately
        # after applying the previous winner and before their measurements.
        first_proposal: object | None = None
        first_candidates: list[object] = []
        if options.budget > 0:
            first_proposal = _oracle_proposals(
                inputs,
                context,
                current_state,
                route,
                query,
                writer,
                lattice,
                options,
                state_index=0,
                policy=policy,
            )
            first_candidates = _candidate_rows(
                inputs,
                route,
                context,
                current_state,
                0,
                options,
                proposal=first_proposal,
            )
        try:
            initial_prediction = _prediction_for(
                getattr(inputs, "model", None),
                route,
                context,
                final=False,
                options=experiment_options,
                pipeline_counters=pipeline_counters,
            )
        except ValueError:
            # Engineering callback routes may provide only one diagnostic
            # prediction tensor; production typed routes must expose/decode an
            # explicit initial state and therefore never take this branch.
            if not options.engineering_only or not isinstance(_route_attr(route, "final_prediction"), Tensor):
                raise
            initial_prediction = _route_attr(route, "final_prediction")
        # Target access starts only after the initial target-free state,
        # initial prediction, and first complete candidate bank are sealed.
        target_context = _target_join(inputs, sample, context, route, initial_prediction, experiment_options)
        oracle_context = OracleContext(
            observation_context=context,
            target_context=target_context,
            route=route,
            scope=("all_candidates" if options.mode == "all_exact_one" else "candidate_subset"),
        )
        selected_ids: list[str] = []
        measured_rows: list[dict[str, Any]] = []
        confirmations: list[dict[str, Any]] = []
        stop_reason = "budget"
        rng = random.Random(options.seed + sample_index)
        current_proposal = first_proposal
        current_candidates = first_candidates
        for state_index in range(options.budget):
            if state_index > 0:
                current_proposal = _oracle_proposals(
                    inputs,
                    context,
                    current_state,
                    route,
                    query,
                    writer,
                    lattice,
                    options,
                    state_index=state_index,
                    policy=policy,
                )
                current_candidates = _candidate_rows(
                    inputs,
                    route,
                    context,
                    current_state,
                    state_index,
                    options,
                    proposal=current_proposal,
                )
            candidates = list(current_candidates)
            if not candidates:
                stop_reason = "no_candidates"
                break
            if options.mode in ("sampled_one", "greedy") and len(candidates) > options.candidate_count:
                candidates = rng.sample(candidates, options.candidate_count)
            if hasattr(current_proposal, "row") and hasattr(current_proposal, "point_ids"):
                from .types import ActionProposalBatch

                if not isinstance(current_proposal, ActionProposalBatch):
                    raise TypeError("oracle proposal builder must return ActionProposalBatch for typed teacher measurement")
            labels = _measure_candidates(
                inputs,
                route,
                candidates,
                target_context,
                context,
                lattice,
                options,
                seed=options.seed + state_index,
                diagnostic_state=current_state,
            )
            scope = "all" if options.mode == "all_exact_one" else ("subset" if options.mode == "sampled_one" else "greedy_candidates")
            state_rows: list[dict[str, Any]] = []
            for label in labels:
                row = dict(label)
                row.update(
                    {
                        "subject_id": subject_id,
                        "state_index": state_index,
                        "candidate_scope": scope,
                        "oracle_mode": options.mode,
                        "screening_seed": options.seed + state_index,
                        "screening_teacher_mode": options.teacher_mode,
                        "screening_q_draws": options.query_count if options.teacher_mode == "iid_fixed_q" else 0,
                        "screening_reused_for_confirmation": False,
                    }
                )
                state_rows.append(action_metric_row(row, numerical_tolerance=options.numerical_tolerance, practical_margin=options.practical_margin, scope=scope, selected=False))
            measured_rows.extend(state_rows)
            if not state_rows:
                stop_reason = "no_measurements"
                break
            gains = [(float(row["true_gain"]), index, row) for index, row in enumerate(state_rows) if row.get("true_gain") is not None and math.isfinite(float(row["true_gain"]))]
            if not gains:
                stop_reason = "no_finite_gain"
                break
            def _tie_key(
                item: tuple[float, int, Mapping[str, Any]],
                candidate_rows: Sequence[object] = candidates,
            ) -> tuple[float, int, int]:
                candidate = candidate_rows[item[1]]
                point_id = getattr(candidate, "point_id", None)
                if isinstance(candidate, Mapping):
                    point_id = candidate.get("point_id", point_id)
                try:
                    resolved_point_id = int(point_id)
                except (TypeError, ValueError):
                    resolved_point_id = item[1]
                # ``max`` therefore chooses the smallest point ID for equal
                # gains, with the candidate index only as a final deterministic
                # fallback for engineering mappings without point IDs.
                return item[0], -resolved_point_id, -item[1]

            _, winning_index, winner = max(gains, key=_tie_key)
            winner_gain = float(winner["true_gain"])
            if winner_gain <= options.practical_margin:
                stop_reason = "low_gain"
                break
            action = candidates[winning_index] if winning_index < len(candidates) else None
            selected_id = _action_id(action, winning_index)
            selected_ids.append(selected_id)
            winner["selected"] = True
            winner["oracle_selected"] = True
            winner["selection_scope"] = scope
            if options.confirmation_mode != "none":
                confirm_options = OracleOptions(
                    mode="sampled_one",
                    budget=1,
                    candidate_count=1,
                    teacher_mode=options.confirmation_mode,
                    query_count=options.confirmation_query_count,
                    max_subjects=1,
                    seed=options.seed + 10_000 + state_index,
                    split_role=options.split_role,
                    # This is a terminal independent measurement, not a
                    # nested screening stage.  Set the declaration to the
                    # same mode so iid_fixed_q confirmation does not recurse
                    # into another required confirmation.
                    confirmation_mode=options.confirmation_mode,
                    confirmation_query_count=options.confirmation_query_count,
                    engineering_only=options.engineering_only,
                )
                confirm_labels = _measure_candidates(
                    inputs,
                    route,
                    [action],
                    target_context,
                    context,
                    lattice,
                    confirm_options,
                    seed=confirm_options.seed,
                    diagnostic_state=current_state,
                )
                if confirm_labels:
                    confirm = dict(confirm_labels[0])
                    confirm["subject_id"] = subject_id
                    confirm["action_id"] = selected_id
                    confirm["confirmation_seed"] = confirm_options.seed
                    confirm["confirmation_mode"] = confirm_options.teacher_mode
                    confirm["confirmation_q_draws"] = (
                        confirm_options.query_count
                        if confirm_options.teacher_mode == "iid_fixed_q"
                        else 0
                    )
                    confirm["screening_gain"] = winner_gain
                    confirm["confirmation_discrepancy"] = float(confirm.get("raw_gain", confirm.get("true_gain", 0.0))) - winner_gain
                    confirmations.append(confirm)
            # Oracle state transition: execute exactly this stored winner from
            # the current state, then generate a fresh proposal bank for the
            # next greedy step.  sampled/all-exact are one-step scopes.
            typed_proposal = hasattr(current_proposal, "row")
            metadata = getattr(inputs, "metadata", {})
            callback_apply = isinstance(metadata, Mapping) and metadata.get("oracle_apply") is not None
            if typed_proposal or callback_apply:
                oracle_decision = (
                    _oracle_continue_decision(
                        current_proposal,
                        action,
                        policy,
                        step=int(getattr(current_state, "state_version", state_index)),
                    )
                    if typed_proposal
                    else None
                )
                current_state = _oracle_advance(
                    inputs,
                    current_state,
                    context,
                    current_proposal,
                    action,
                    oracle_decision,
                    writer=writer,
                    state_index=state_index,
                    route=route,
                    target_context=target_context,
                )
                if hasattr(current_state, "state_version") and int(current_state.state_version) != state_index + 1:
                    raise RuntimeError("oracle winner execution produced a noncontiguous state version")
            elif options.mode == "greedy":
                raise ValueError("greedy oracle requires a typed proposal bank or explicit oracle_apply callback")
            stop_reason = "budget"
            if options.mode == "greedy" and state_index + 1 < options.budget:
                continue
            break
        oracle_final_prediction: Tensor | None = None
        model = getattr(inputs, "model", None)
        if model is not None and context is not None and hasattr(model, "decode_final") and hasattr(current_state, "planes"):
            oracle_final_prediction = model.decode_final(current_state, context, chunk_size=experiment_options.decode_chunk_size)
        elif isinstance(getattr(inputs, "metadata", None), Mapping):
            candidate_prediction = inputs.metadata.get("oracle_final_prediction")
            if isinstance(candidate_prediction, Tensor):
                oracle_final_prediction = candidate_prediction
        paired = None
        if oracle_final_prediction is not None:
            target = getattr(target_context, "target", None)
            mask = getattr(target_context, "observation_mask", getattr(target_context, "target_mask", None))
            if isinstance(target, Tensor):
                paired = paired_subject_metrics(
                    initial_prediction,
                    oracle_final_prediction,
                    target,
                    mask if isinstance(mask, Tensor) else None,
                    subject_id=subject_id,
                    context_id=getattr(context, "context_id", None),
                    scenario="oracle",
                    budget=options.budget,
                )
            # The final decode is part of the same full-pipeline receipt.  A
            # callback-provided tensor remains explicitly uninstrumented.
            if model is not None and hasattr(model, "decode_final"):
                pipeline_counters["final_decode_calls"] = pipeline_counters.get(
                    "final_decode_calls", 0
                ) + 1
        result = OracleResult(
            subject_id=subject_id,
            mode=options.mode,
            budget=options.budget,
            candidate_scope=("all_candidates" if options.mode == "all_exact_one" else ("subset_candidates" if options.mode == "sampled_one" else "greedy_per_state")),
            rows=tuple(measured_rows),
            selected_action_ids=tuple(selected_ids),
            stop_reason=stop_reason,
            confirmation=tuple(confirmations),
            context_digest=oracle_context.context_digest,
        )
        result_payload = result.as_dict()
        result_payload["confirmation_configuration"] = {
            "configured_mode": options.confirmation_mode,
            "effective_mode": options.confirmation_mode,
            "configured_query_count": options.confirmation_query_count,
            "screening_mode": options.teacher_mode,
            "screening_query_count": options.query_count,
            "screening_seed": options.seed,
            "independent_from_screening": options.confirmation_mode != "none",
        }
        result_payload["paired_metrics"] = paired
        result_payload["oracle_final_prediction_decoded"] = oracle_final_prediction is not None
        result_payload["oracle_route_gain"] = None if paired is None else paired["improvement"].get("masked_charbonnier")
        # Preserve the same source/initial-state identities consumed by the
        # paired-artifact comparison service.  This receipt is metadata only;
        # target tensors and predictions are never serialized here.
        source_receipt = _oracle_source_receipt(
            inputs, context, options, subject_id=subject_id
        )
        source_receipt["subject_id"] = subject_id
        result_payload["source_receipt"] = source_receipt
        source_receipts.append(source_receipt)
        result_payload["z0_digest"] = tensor_digest(
            initial_prediction.detach(), name="z0_prediction"
        )
        result_payload["z0_state_digest"] = getattr(
            _route_attr(route, "initial_state", current_state), "state_digest", None
        )
        result_payload["pipeline_counters"] = dict(pipeline_counters)
        result_payload["route_counter_scope"] = "route_only; excludes outer encode/decode"
        result_payload["route_counters"] = _jsonable(_counter_metadata(route))
        result_payload["route_counter_scope_after_measurement"] = (
            "route_and_teacher_measurement; excludes outer encode/decode"
        )
        result_payload["pipeline_counter_scope"] = (
            "service_outer_calls_only; callback-built contexts/routes uninstrumented"
        )
        pipeline_counter_rows.append(
            {
                "subject_id": subject_id,
                "counters": dict(pipeline_counters),
                "scope": "service_outer_calls_only; route counters remain separately tagged",
            }
        )
        rows_out.append(result_payload)
        all_label_rows.extend(measured_rows)
    _write_jsonl(path, rows_out)
    # Scientific uncertainty is over independent subjects, never over the
    # correlated candidate rows within one subject.
    gains: list[float] = []
    for result in rows_out:
        route_gain = result.get("oracle_route_gain")
        if route_gain is not None and math.isfinite(float(route_gain)):
            gains.append(float(route_gain))
            continue
        selected = [float(row["true_gain"]) for row in result["rows"] if row.get("selected") and row.get("true_gain") is not None]
        if selected:
            gains.append(float(sum(selected)))
    decision = scientific_decision(gains, practical_margin=options.practical_margin, minimum_subjects=32)
    # A multi-subject oracle has one initialization identity per target-free
    # context.  Publish the exact map at the service boundary so paired
    # comparison can join it without inventing a scalar first-subject hash;
    # retain the full per-subject receipts in ``contexts`` for auditability.
    if len(source_receipts) == 1:
        aggregate_source_receipt: dict[str, Any] = source_receipts[0]
    else:
        def _consistent(field: str) -> object | None:
            values: dict[str, object] = {}
            for item in source_receipts:
                value = item.get(field)
                if value is None or (isinstance(value, str) and not value.strip()):
                    continue
                values[json.dumps(_jsonable(value), sort_keys=True)] = value
            return next(iter(values.values())) if len(values) == 1 else None

        subject_initialization_hashes = {
            str(item.get("subject_id", index)): str(item["initialization_hash"])
            for index, item in enumerate(source_receipts)
            if item.get("initialization_hash")
        }
        aggregate_source_receipt = {
            "schema_version": ORACLE_OPTIONS_SCHEMA,
            "producer_compatibility_hash": _consistent("producer_compatibility_hash"),
            "normalization_hash": _consistent("normalization_hash"),
            "initialization_hash": (
                next(iter(subject_initialization_hashes.values()))
                if len(subject_initialization_hashes) == 1
                else None
            ),
            "subject_initialization_hashes": subject_initialization_hashes,
            "baseline_split_hash": _consistent("baseline_split_hash"),
            "training_role_manifest_hash": _consistent("training_role_manifest_hash"),
            "split_role": _consistent("split_role"),
            "mask_definition": _consistent("mask_definition"),
            "label_definition": _consistent("label_definition"),
            "loss_definition": _consistent("loss_definition"),
            "data_range": _consistent("data_range"),
            "engineering_only": all(bool(item.get("engineering_only", False)) for item in source_receipts),
            "contexts": source_receipts,
        }
    return {
        "software_status": "SOFTWARE_PASS",
        "scientific_status": decision["decision"],
        "scientific_decision": decision,
        "subject_count": len(rows_out),
        "candidate_count": len(all_label_rows),
        "privileged": True,
        "output_path": path,
        "schema_version": ORACLE_OPTIONS_SCHEMA,
        "pipeline_counters": {
            "rows": pipeline_counter_rows,
            "scope": "service_outer_calls_only; route counters remain separately tagged",
        },
        "source_receipt": aggregate_source_receipt,
    }


__all__ = [
    "ORACLE_MODES",
    "ORACLE_OPTIONS_SCHEMA",
    "OracleContext",
    "OracleOptions",
    "OracleResult",
    "run_oracle_evaluation",
]
