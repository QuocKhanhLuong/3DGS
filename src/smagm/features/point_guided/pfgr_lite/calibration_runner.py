"""Concrete S5 target-free forced-route calibration collection.

The runner is the narrow W3b -> W4 seam for calibration.  It seals every
forced K4 route before invoking the deferred target provider, measures only
the stored selected actions, and then delegates the fit to W4's strict
``fit_calibration`` implementation.  Engineering fixtures are explicit and
produce diagnostic-only calibration; they can never mint an adaptive release.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
import inspect
import math
from pathlib import Path
from typing import Any, Literal

import torch

from .calibration import (
    CalibrationEvidence,
    CalibrationWinner,
    ForcedCalibrationTrace,
    TraceReceipt,
    fit_calibration,
)
from .config import PFGRLiteConfig
from .provenance import canonical_digest


CALIBRATION_RUN_SCHEMA = "pfgr-lite-calibration-run-v1"


def _positive_int(name: str, value: object, *, maximum: int = 1_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > maximum:
        raise ValueError(f"{name} must be a positive integer <= {maximum}")
    return int(value)


def _stage_cpu_value(value: Any) -> Any:
    """Detach one sealed route/context value onto CPU-owned storage.

    S5 can collect dozens of subjects.  Retaining every GPU feature lattice
    until labels are measured would turn a target-free collection into an
    unbounded VRAM cache.  Dataclass reconstruction preserves all typed
    identity fields while recomputing their integrity digests; tensors are
    detached, cloned and moved to CPU, never shared with the live model.
    """

    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: _stage_cpu_value(getattr(value, field.name))
            for field in fields(value)
            if field.init
        }
        return replace(value, **updates)
    if isinstance(value, Mapping):
        return {key: _stage_cpu_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_stage_cpu_value(item) for item in value)
    if isinstance(value, list):
        return [_stage_cpu_value(item) for item in value]
    return value


def _tensor_storage_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if is_dataclass(value) and not isinstance(value, type):
        return sum(_tensor_storage_bytes(getattr(value, field.name)) for field in fields(value) if field.init)
    if isinstance(value, Mapping):
        return sum(_tensor_storage_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_storage_bytes(item) for item in value)
    return 0


def _new_collection_stats() -> dict[str, int]:
    return {"staged_trace_count": 0, "staged_tensor_bytes": 0, "peak_staged_tensor_bytes": 0, "replay_count": 0}


@dataclass(frozen=True)
class CalibrationRunOptions:
    """Strict collection options shared by W3b S5 and the CLI."""

    # Exact footprint confirmation is the safe MAIN/default route.  The
    # iid-fixed-Q pilot is explicit and must opt in with a positive draw count.
    confirmation_mode: Literal["exact", "iid_fixed_q"] = "exact"
    confirmation_q_draws: int = 0
    confirmation_seed: int = 20260907
    collection_seed: int = 20260907
    value_input_variant: int = 366
    max_subjects: int | None = None
    engineering_only: bool = False
    schema_version: str = CALIBRATION_RUN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CALIBRATION_RUN_SCHEMA:
            raise ValueError("unknown CalibrationRunOptions schema")
        if self.confirmation_mode not in ("exact", "iid_fixed_q"):
            raise ValueError("confirmation_mode must be exact or iid_fixed_q")
        if isinstance(self.confirmation_q_draws, bool) or not isinstance(self.confirmation_q_draws, int):
            raise TypeError("confirmation_q_draws must be an integer")
        if self.confirmation_mode == "exact":
            if self.confirmation_q_draws != 0:
                raise ValueError("exact confirmation requires confirmation_q_draws=0")
        elif self.confirmation_q_draws < 2 or self.confirmation_q_draws > 10_000_000:
            raise ValueError("iid_fixed_q confirmation requires at least two Q draws")
        if isinstance(self.collection_seed, bool) or not isinstance(self.collection_seed, int) or self.collection_seed < 0:
            raise ValueError("collection_seed must be a nonnegative integer")
        if isinstance(self.confirmation_seed, bool) or not isinstance(self.confirmation_seed, int) or self.confirmation_seed < 0:
            raise ValueError("confirmation_seed must be a nonnegative integer")
        if self.value_input_variant not in (126, 222, 270, 366):
            raise ValueError("value_input_variant must be one of 126, 222, 270, or 366")
        if self.max_subjects is not None:
            _positive_int("max_subjects", self.max_subjects)
        if not isinstance(self.engineering_only, bool):
            raise TypeError("engineering_only must be bool")

    def as_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "CalibrationRunOptions":
        if not isinstance(values, Mapping):
            raise TypeError("CalibrationRunOptions must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown CalibrationRunOptions keys: {sorted(unknown)}")
        return cls(**dict(values))


def _invoke(callback: Any, *positional: Any, **keyword: Any) -> Any:
    if not callable(callback):
        raise TypeError("calibration callback must be callable")
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return callback(*positional, **keyword)
    parameters = signature.parameters
    positional_names = [
        name
        for name, parameter in parameters.items()
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    consumed = set(positional_names[: len(positional)])
    accepts_var_kw = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    if accepts_var_kw:
        accepted = {name: value for name, value in keyword.items() if name not in consumed}
    else:
        accepted = {
            name: value
            for name, value in keyword.items()
            if name in parameters and name not in consumed and parameters[name].kind != inspect.Parameter.POSITIONAL_ONLY
        }
    return callback(*positional[: len(positional_names)], **accepted)


def _config(inputs: Any) -> PFGRLiteConfig:
    execution = getattr(inputs, "execution", None)
    value = getattr(execution, "config", None) if execution is not None else getattr(inputs, "config", None)
    if not isinstance(value, PFGRLiteConfig):
        raise TypeError("calibration runner requires StageInputs execution.config")
    return value


def _role_manifest(inputs: Any) -> Any:
    manifest = getattr(inputs, "role_manifest", None)
    if manifest is None:
        raise ValueError("calibration runner requires the reviewed TrainingRoleManifest")
    return manifest


def _subject_role(manifest: Any, subject_id: str) -> str | None:
    if subject_id in set(manifest.producer_fit_subject_ids):
        return "producer_fit"
    if subject_id in set(manifest.calibration_fit_subject_ids):
        return "calibration_fit"
    if subject_id in set(manifest.calibration_allowance_subject_ids):
        return "calibration_allowance"
    return None


def _policy_and_producer(inputs: Any, traces: Sequence[ForcedCalibrationTrace]) -> tuple[Any, Any]:
    if not traces:
        raise ValueError("calibration collection produced no forced traces")
    policy = traces[0].collection_policy
    producer = traces[0].observation_context.producer
    for trace in traces:
        if trace.collection_policy.policy_hash != policy.policy_hash:
            raise ValueError("all forced calibration traces must share one collection policy")
        if trace.observation_context.producer.compatibility_hash != producer.compatibility_hash:
            raise ValueError("forced calibration traces must share one producer identity")
    return policy, producer


def _context_for_sample(inputs: Any, sample: Any, *, counters: Any | None = None) -> Any:
    """Encode one observation-only sample through the already-built model."""

    model = getattr(inputs, "model", None)
    encode = getattr(model, "encode_observations", None)
    if not callable(encode):
        raise ValueError("S5 default collection requires PFGRLiteModel.encode_observations")
    execution = getattr(inputs, "execution", None)
    options = getattr(execution, "stage_options", None)
    device = getattr(options, "device", "cpu")
    observations = sample.observations.unsqueeze(0).to(device=device)
    mask = sample.brain_mask.to(device=observations.device)
    with torch.no_grad():
        context = encode(observations, mask, sample.geometry)
    context.validate_integrity()
    # This is an actual completed shared MedicalNet traversal.  The model's
    # provenance traversal_count is immutable source evidence, while the
    # operation counter records this S5 collection/replay call itself.
    if counters is not None and hasattr(counters, "add"):
        counters.add(medicalnet_traversals=1)
    return context


def _lattice_for_context(inputs: Any, context: Any, *, dtype: torch.dtype) -> Any:
    """Resolve the canonical W2 lattice without creating a fallback query."""

    metadata = getattr(inputs, "metadata", {})
    factory = metadata.get("lattice_factory") if isinstance(metadata, Mapping) else None
    if factory is None:
        factory = getattr(getattr(inputs, "model", None), "_query_lattice_factory", None)
    if factory is None:
        from .footprint import PFGRQueryLattice

        factory = PFGRQueryLattice
    kwargs = {
        "output_geometry": context.geometry,
        "feature_geometry": context.feature_geometry,
        "query_dtype": dtype,
        "build_chunk_size": _config(inputs).build_chunk_size,
    }
    if hasattr(factory, "build"):
        return factory.build(**kwargs)
    return factory(**kwargs)


def _value_binding(inputs: Any) -> tuple[Any, Any, float, str, Mapping[str, object] | None]:
    """Return the exact loaded V model and its immutable identity envelope."""

    metadata = getattr(inputs, "metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("S5 default collection requires value-model metadata")
    value_model = metadata.get("value_model")
    identity = metadata.get("value_fit_identity")
    if value_model is None or identity is None:
        raise ValueError("S5 default collection requires the loaded V model and ValueFitIdentity")
    scale_hash = metadata.get("gain_scale_hash", getattr(identity, "gain_scale_hash", None))
    scale_value = metadata.get("gain_scale")
    scale_provenance = metadata.get("gain_scale_provenance")
    if scale_value is None and isinstance(scale_provenance, Mapping):
        scale_value = scale_provenance.get("scale")
    if not isinstance(scale_value, (int, float)) or not math.isfinite(float(scale_value)) or float(scale_value) <= 0.0:
        raise ValueError("S5 default collection requires a finite positive GainScale")
    if not isinstance(scale_hash, str) or not scale_hash:
        raise ValueError("S5 default collection requires a complete GainScale identity")
    if not hasattr(identity, "digest"):
        raise TypeError("S5 value_fit_identity must be the typed ValueFitIdentity")
    return value_model, identity, float(scale_value), scale_hash, scale_provenance if isinstance(scale_provenance, Mapping) else None


def _default_forced_trace(
    inputs: Any,
    sample: Any,
    options: CalibrationRunOptions,
    *,
    index: int,
    counters: Any | None = None,
) -> ForcedCalibrationTrace:
    """Run one actual target-free forced learned K4 route.

    This is the production default: U proposes bounded rows, W4's policy
    selects every legal transition under a forced diagnostic policy, and W2's
    injected query/writer remain the only geometry implementation.
    """

    from .data import bind_observation_context
    from .inference import run_pfgr_inference
    from .policy import load_effective_policy
    from .sparse_write import make_point_query, make_support_legal_mask

    context = _context_for_sample(inputs, sample, counters=counters)
    query = getattr(inputs, "query", None)
    writer = getattr(inputs, "writer", None)
    if not callable(query) or not callable(writer):
        raise ValueError("S5 default collection requires canonical W2 query and writer injections")
    value_model, value_identity, gain_scale, gain_scale_hash, gain_scale_provenance = _value_binding(inputs)
    config = _config(inputs)
    forced_config = replace(config, policy=replace(config.policy, mode="forced_diagnostic"))
    policy = load_effective_policy(
        forced_config,
        None,
        dependencies=context.producer,
        capability="forced_diagnostic",
        budget=4,
        candidate_chunk_size=config.build_chunk_size,
        # Every subject in one calibration bank shares the exact forced
        # collection policy identity.  Subject order/seed is tracked by the
        # collection options, not by mutating EffectivePolicy.policy_hash.
        random_seed=options.collection_seed,
        value_input_variant=options.value_input_variant,
        value_fit_identity=value_identity,
        gain_scale=gain_scale,
        gain_scale_hash=gain_scale_hash,
        gain_scale_provenance=gain_scale_provenance,
    )
    lattice = _lattice_for_context(inputs, context, dtype=context.initial_planes.xy.dtype)
    legal_mask = make_support_legal_mask(lattice)
    metadata = getattr(inputs, "metadata", {})
    if counters is None:
        counters = metadata.get("operation_counters") if isinstance(metadata, Mapping) else None
    if counters is not None and not hasattr(counters, "add"):
        counters = None
    route = run_pfgr_inference(
        getattr(inputs, "model", None),
        context,
        policy,
        query=query if query is not None else make_point_query(),
        writer=writer,
        value_model=value_model,
        counters=counters,
        legal_mask=legal_mask,
    )
    binding = bind_observation_context(sample, context).as_dict()
    # Calibration's subject-context envelope uses the W4 subject-geometry
    # identity (distinct from the data-loader geometry hash) so it can be
    # revalidated independently by a later bank/replay consumer.
    binding["geometry_hash"] = canonical_digest(
        {"shape_dhw": context.geometry.shape_dhw, "voxel_to_ras_mm": context.geometry.voxel_to_ras_mm},
        prefix="pfgr-lite-subject-geometry-v1|",
    )
    binding.pop("binding_digest", None)
    binding["binding_digest"] = canonical_digest(
        {key: binding[key] for key in ("schema_version", "subject_id", "observation_record_id", "context_id", "geometry_hash", "normalization_hash")},
        prefix="pfgr-lite-subject-context-binding-v1|",
    )
    return ForcedCalibrationTrace(
        observation_context=context,
        route=route,
        collection_policy=policy,
        subject_id=sample.subject_id,
        subject_context_binding=binding,
    )


def _selected_actions(trace: ForcedCalibrationTrace) -> tuple[Any, ...]:
    completed = trace.route.completed_trace
    if completed is None:
        return ()
    actions: list[Any] = []
    for proposal, decision in zip(completed.proposals, completed.decisions):
        if decision.stop_code != "continue":
            continue
        locations = (proposal.point_ids == int(decision.selected_point_id)).nonzero(as_tuple=False)
        if locations.shape[0] != 1:
            raise ValueError("forced route decision does not identify one stored action")
        actions.append(proposal.row(int(locations[0, 0]), int(locations[0, 1])))
    return tuple(actions)


def _default_measure_trace(
    inputs: Any,
    trace: ForcedCalibrationTrace,
    options: CalibrationRunOptions,
    *,
    stats: dict[str, int] | None = None,
    counters: Any | None = None,
) -> tuple[CalibrationWinner, ...]:
    """Measure actual stored selected rows with W2's target-after-trace teacher."""

    from dataclasses import replace as dataclass_replace
    from .data import defer_supervision
    from .teacher import measure_actions

    # Collection traces are staged on CPU immediately after route sealing.
    # Replay the exact target-free route on the live device only when labels
    # are about to be measured, and reject any identity drift.  This keeps the
    # full cohort target-free before the first target read without retaining a
    # GPU ObservationContext/route per subject.
    working_trace = trace
    metadata = getattr(inputs, "metadata", {})
    if stats is not None and stats.get("staged_trace_count", 0) > 0:
        samples = tuple(getattr(inputs, "samples", ()))
        sample_for_replay = next((item for item in samples if item.subject_id == trace.subject_id), None)
        if sample_for_replay is None:
            raise ValueError("staged forced trace subject is not present in StageInputs")
        subject_index = next((index for index, item in enumerate(samples) if item.subject_id == trace.subject_id), 0)
        replayed = _default_forced_trace(inputs, sample_for_replay, options, index=subject_index, counters=counters)
        if replayed.receipt.trace_hash != trace.receipt.trace_hash:
            raise ValueError("target-free calibration replay changed the sealed trace identity")
        working_trace = replayed
        stats["replay_count"] += 1

    trace = working_trace
    completed = trace.route.completed_trace
    if completed is None:
        return ()
    sample = next((item for item in getattr(inputs, "samples", ()) if item.subject_id == trace.subject_id), None)
    if sample is None:
        raise ValueError("forced trace subject is not present in StageInputs")
    provider = getattr(inputs, "target_provider", None)
    if not callable(provider):
        raise ValueError("S5 selected-action measurement requires the deferred target provider")
    data_counters = metadata.get("counters") if isinstance(metadata, Mapping) else None
    joined = defer_supervision(sample, provider, counters=data_counters, engineering_only=bool(getattr(_config(inputs), "engineering_only", False)))
    target_context = joined(completed_context=trace.observation_context, trace=completed)
    # ``defer_supervision`` already performs the one authoritative
    # target/mask validation and binds the target to the sealed
    # observation context and completed trace.  Do not call ``validate_target``
    # again here: that would duplicate detached target construction and make
    # ``target_validations`` report two validations for one target read.  Keep
    # the hot immutable guard plus explicit route/config checks local to the
    # calibration boundary instead.
    target_context.validate_integrity()
    if counters is not None and hasattr(counters, "add"):
        counters.add(target_validations=1)
    if target_context.context_id != trace.observation_context.context_id:
        raise ValueError("deferred target context does not match calibration trace")
    if target_context.output_geometry != trace.observation_context.geometry:
        raise ValueError("deferred target geometry does not match calibration trace")
    if target_context.feature_geometry != trace.observation_context.feature_geometry:
        raise ValueError("deferred target feature geometry does not match calibration trace")
    compatibility = trace.observation_context.producer.compatibility
    if target_context.producer_compatibility_hash != compatibility.digest:
        raise ValueError("deferred target producer identity does not match calibration trace")
    if target_context.normalization_hash != compatibility.observation_normalization_hash:
        raise ValueError("deferred target normalization identity does not match calibration trace")
    if target_context.trace_route_hash not in (None, completed.route_hash):
        raise ValueError("deferred target route identity does not match calibration trace")
    teacher_config = _config(inputs).teacher
    # The serialized PFGR config uses the stable underscore spelling while
    # the target-bound teacher context carries the versioned hyphen spelling
    # (``LABEL_DEFINITION``).  Both are the same locked estimand; accept only
    # these two declarations, never an arbitrary caller-provided alias.
    from .teacher import LABEL_DEFINITION

    if target_context.mask_definition != teacher_config.mask_definition:
        raise ValueError("deferred target mask definition does not match teacher config")
    if target_context.label_definition not in {
        LABEL_DEFINITION,
        teacher_config.label_definition,
    }:
        raise ValueError("deferred target label definition does not match teacher config")
    if options.confirmation_mode == "exact":
        teacher_config = dataclass_replace(teacher_config, mode="exact_footprint", q_draws=0)
    else:
        teacher_config = dataclass_replace(teacher_config, mode="iid_fixed_q", q_draws=options.confirmation_q_draws)
    lattice = _lattice_for_context(inputs, trace.observation_context, dtype=trace.observation_context.initial_planes.xy.dtype)
    decoder = getattr(getattr(inputs, "model", None), "decoder", None)
    if decoder is None:
        raise ValueError("S5 selected-action measurement requires the shared decoder")
    execution = getattr(inputs, "execution", None)
    stage_options = getattr(execution, "stage_options", None)
    labels = measure_actions(
        completed,
        _selected_actions(trace),
        target_context,
        decoder,
        teacher_config,
        lattice=lattice,
        chunk_size=getattr(stage_options, "query_chunk_size", 1024),
        candidate_chunk_size=getattr(stage_options, "candidate_chunk_size", 1),
        seed=options.confirmation_seed,
        observation_context=trace.observation_context,
        counters=counters if counters is not None else getattr(trace.route, "counters", None),
    )
    value_model, value_identity, _gain_scale, gain_scale_hash, _gain_scale_provenance = _value_binding(inputs)
    del value_model
    action_by_id = {action.action_id: action for action in _selected_actions(trace)}
    raw_by_action: dict[str, float] = {}
    for proposal, decision in zip(completed.proposals, completed.decisions):
        if decision.stop_code != "continue":
            continue
        locations = (proposal.point_ids == int(decision.selected_point_id)).nonzero(as_tuple=False)
        if locations.shape[0] != 1:
            raise ValueError("forced route decision does not identify one stored action")
        action = proposal.row(int(locations[0, 0]), int(locations[0, 1]))
        raw_by_action[action.action_id] = float(decision.raw_value)
    compatibility = trace.observation_context.producer.compatibility
    role = _subject_role(_role_manifest(inputs), trace.subject_id)
    if role not in {"calibration_fit", "calibration_allowance"}:
        raise ValueError("forced trace subject must belong to a calibration role")
    winners: list[CalibrationWinner] = []
    for label in labels:
        action = action_by_id.get(label.action_id)
        if action is None:
            raise ValueError("teacher label action is not one of the stored selected rows")
        confirmation_hash = canonical_digest(
            {
                "subject_id": trace.subject_id,
                "action_id": label.action_id,
                "context_id": label.context_id,
                "state_version": label.state_version,
                "measurement_role": label.role,
                "q_draws": label.q_draws,
                "seed": label.seed,
                "standard_error": label.standard_error,
            },
            prefix="pfgr-lite-calibration-confirmation-v1|",
        )
        winners.append(
            CalibrationWinner(
                subject_id=trace.subject_id,
                action_id=label.action_id,
                proposal_digest=next(proposal.proposal_digest for proposal in completed.proposals if proposal.state_version == label.state_version),
                action_digest=action.action_digest,
                # Decision.raw_value is already the signed raw gain after W4
                # applies the fixed GainScale.  Do not divide it again: the
                # fitted calibration must regress against the exact stored
                # policy value (including non-unit scales).
                raw_score=raw_by_action[action.action_id],
                measured_gain=label.raw_gain,
                producer_compatibility_hash=trace.observation_context.producer.compatibility_hash,
                value_fit_identity_hash=value_identity.digest,
                gain_scale_hash=gain_scale_hash,
                policy_hash=trace.collection_policy.policy_hash,
                writer_hash=compatibility.writer_hash,
                query_hash=compatibility.geometry_query_version_hash,
                proposal_generator_hash="pfgr-lite-action-generator-v1",
                role=role,
                measurement_role=label.role,
                state_version=label.state_version,
                trace_hash=trace.receipt.trace_hash,
                measurement_mode=label.role,
                q_draws=label.q_draws,
                seed=label.seed,
                standard_error=label.standard_error,
                confirmation_hash=confirmation_hash,
            )
        )
    return tuple(winners)


def _winner_records(value: Any) -> tuple[CalibrationWinner, ...]:
    if isinstance(value, Mapping):
        value = value.get("winners", value.get("records", ()))
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("calibration winner callback must return a sequence")
    result = tuple(item if isinstance(item, CalibrationWinner) else CalibrationWinner(**dict(item)) for item in value)
    if not result:
        raise ValueError("calibration winner callback returned no records")
    return result


def _validate_winners(
    winners: Sequence[CalibrationWinner],
    traces: Sequence[ForcedCalibrationTrace],
    *,
    role: str,
    policy: Any,
    producer: Any,
    options: CalibrationRunOptions,
) -> None:
    by_subject = {trace.subject_id: trace for trace in traces}
    for winner in winners:
        if winner.role != role:
            raise ValueError(f"winner role must be {role}")
        trace = by_subject.get(winner.subject_id)
        if trace is None:
            raise ValueError("winner subject is not present in sealed forced traces")
        if winner.trace_hash != trace.receipt.trace_hash:
            raise ValueError("winner trace_hash does not match sealed forced trace")
        if winner.producer_compatibility_hash != producer.compatibility_hash:
            raise ValueError("winner producer identity does not match forced trace")
        if winner.policy_hash != policy.policy_hash:
            raise ValueError("winner policy identity does not match forced collection policy")
        if winner.value_fit_identity_hash != (policy.value_fit_identity.digest if policy.value_fit_identity is not None else winner.value_fit_identity_hash):
            raise ValueError("winner ValueFitIdentity does not match collection policy")
        if winner.gain_scale_hash != policy.gain_scale_hash:
            raise ValueError("winner gain-scale identity does not match collection policy")
        if winner.measurement_role != ("iid_fixed_q" if options.confirmation_mode == "iid_fixed_q" else "exact_footprint"):
            raise ValueError("winner measurement role does not match fixed confirmation mode")
        if winner.measurement_mode != winner.measurement_role:
            raise ValueError("winner measurement_mode must match measurement_role")
        if options.confirmation_mode == "iid_fixed_q":
            if winner.q_draws != options.confirmation_q_draws:
                raise ValueError("winner fixed-Q draw count does not match collection options")
            # W2 may derive a deterministic per-action seed from the parent
            # confirmation seed; retain that actual row seed rather than
            # laundering every winner to the parent value.
            if winner.seed is None or not isinstance(winner.seed, int) or isinstance(winner.seed, bool):
                raise ValueError("winner fixed-Q metadata requires the actual per-action seed")
            if winner.standard_error is None or not winner.confirmation_hash:
                raise ValueError("winner fixed-Q metadata requires standard_error and confirmation_hash")
        elif winner.q_draws != 0 or winner.seed is not None:
            raise ValueError("exact winner metadata cannot carry fixed-Q fields")


def _evidence(
    *,
    config: PFGRLiteConfig,
    role_manifest: Any,
    policy: Any,
    producer: Any,
    traces: Sequence[ForcedCalibrationTrace],
    fit: Sequence[CalibrationWinner],
    allowance: Sequence[CalibrationWinner],
    options: CalibrationRunOptions,
) -> CalibrationEvidence:
    receipts = tuple(trace.receipt for trace in traces)
    trace_subject_bindings = tuple((trace.receipt.trace_hash, trace.subject_id, trace.observation_context.context_id) for trace in traces)
    subject_context_bindings = tuple(dict(trace.subject_context_binding) for trace in traces)
    all_winners = tuple(fit) + tuple(allowance)
    winner_bindings = tuple((row.subject_id, row.action_id, row.proposal_digest, row.action_digest, row.state_version) for row in all_winners)
    confirmations = tuple(
        (
            row.subject_id,
            row.action_id,
            row.proposal_digest,
            row.action_digest,
            row.measurement_mode,
            row.q_draws,
            row.seed,
            row.standard_error,
            row.confirmation_hash,
        )
        for row in all_winners
    )
    confirmation_hash = canonical_digest(sorted(confirmations), prefix="pfgr-lite-confirmation-set-v1|")
    compatibility = producer.compatibility
    value_identity = policy.value_fit_identity
    if value_identity is None and not options.engineering_only:
        raise ValueError("production calibration collection requires exact ValueFitIdentity")
    value_identity_hash = value_identity.digest if value_identity is not None else "synthetic-value-fit-v1"
    gain_scale_hash = policy.gain_scale_hash or "synthetic-gain-scale-v1"
    return CalibrationEvidence(
        baseline_split_hash=role_manifest.baseline_split_hash,
        producer_fit_subjects=tuple(role_manifest.producer_fit_subject_ids),
        fit_subjects=tuple(role_manifest.calibration_fit_subject_ids),
        allowance_subjects=tuple(role_manifest.calibration_allowance_subject_ids),
        completed_trace_hashes=tuple(receipt.trace_hash for receipt in receipts),
        completed_trace_receipts=receipts,
        winner_bindings=winner_bindings,
        producer_compatibility_hash=producer.compatibility_hash,
        value_fit_identity_hash=value_identity_hash,
        gain_scale_hash=gain_scale_hash,
        policy_hash=canonical_digest(config.policy, prefix="pfgr-lite-policy-config-v1|"),
        writer_hash=compatibility.writer_hash,
        query_hash=compatibility.geometry_query_version_hash,
        proposal_generator_hash="pfgr-lite-action-generator-v1",
        config_hash=canonical_digest(config.as_dict(), prefix="pfgr-lite-calibration-config-v1|"),
        role_manifest=role_manifest,
        winner_confirmations=confirmations,
        collection_policy_hash=policy.policy_hash,
        collection_policy_receipt=policy.as_dict(),
        trace_subject_bindings=trace_subject_bindings,
        subject_context_bindings=subject_context_bindings,
        value_input_variant=options.value_input_variant,
        trace_budget=4,
        confirmation_mode=options.confirmation_mode,
        confirmation_seed=options.confirmation_seed,
        confirmation_q_draws=options.confirmation_q_draws if options.confirmation_mode == "iid_fixed_q" else 0,
        confirmation_independence_hash=confirmation_hash,
        synthetic=bool(options.engineering_only or role_manifest.engineering_only),
        target_free=True,
        sealed=True,
    )


def _synthetic_fixture(inputs: Any, options: CalibrationRunOptions, output_dir: Path) -> Mapping[str, Any]:
    """Create a deterministic diagnostic-only fixture, never a release fit."""

    from .types import TrainingRoleManifest

    names = tuple(f"synthetic-cal-{index:03d}" for index in range(65))
    fit_subjects = names[:32]
    allowance_subjects = names[32:64]
    producer_subjects = names[64:]
    baseline = TrainingRoleManifest(
        baseline_split_hash="synthetic-calibration-split-v1",
        baseline_train_subject_ids=names,
        baseline_validation_subject_ids=("synthetic-validation",),
        baseline_test_subject_ids=("synthetic-test",),
        producer_fit_subject_ids=producer_subjects,
        calibration_fit_subject_ids=fit_subjects,
        calibration_allowance_subject_ids=allowance_subjects,
        subject_group_ids=tuple((subject, f"group-{index:03d}") for index, subject in enumerate(names + ("synthetic-validation", "synthetic-test"))),
        engineering_only=True,
    )
    config = _config(inputs)
    producer_hash = "synthetic-calibration-producer-v1"
    value_hash = "synthetic-calibration-value-v1"
    scale_hash = "synthetic-calibration-scale-v1"
    policy_hash = canonical_digest(config.policy, prefix="pfgr-lite-policy-config-v1|")
    writer_hash = "synthetic-writer-v1"
    query_hash = "synthetic-query-v1"
    proposal_hash = "pfgr-lite-action-generator-v1"
    receipt = TraceReceipt(
        trace_hash="synthetic-trace-v1",
        context_id="synthetic-context-v1",
        state_versions=(0, 1, 2, 3, 4),
        proposal_digests=("synthetic-proposal-0", "synthetic-proposal-1", "synthetic-proposal-2", "synthetic-proposal-3"),
        action_digests=("synthetic-action-0", "synthetic-action-1", "synthetic-action-2", "synthetic-action-3"),
    )
    fit_rows: list[CalibrationWinner] = []
    allowance_rows: list[CalibrationWinner] = []
    for role, subjects, target in (("calibration_fit", fit_subjects, fit_rows), ("calibration_allowance", allowance_subjects, allowance_rows)):
        for index, subject in enumerate(subjects):
            for repeat in range(2):
                state = (index + repeat) % 4
                raw = float(index + repeat + 1) / 10.0
                gain = 0.2 * raw + (0.01 if role == "calibration_fit" else 0.02)
                target.append(
                    CalibrationWinner(
                        subject_id=subject,
                        action_id=f"{role}-{index:03d}-{repeat}",
                        proposal_digest=receipt.proposal_digests[state],
                        action_digest=receipt.action_digests[state],
                        raw_score=raw,
                        measured_gain=gain,
                        producer_compatibility_hash=producer_hash,
                        value_fit_identity_hash=value_hash,
                        gain_scale_hash=scale_hash,
                        policy_hash=policy_hash,
                        writer_hash=writer_hash,
                        query_hash=query_hash,
                        proposal_generator_hash=proposal_hash,
                        role=role,
                        measurement_role="iid_fixed_q",
                        state_version=state,
                        trace_hash=receipt.trace_hash,
                        measurement_mode="iid_fixed_q",
                        q_draws=options.confirmation_q_draws,
                        seed=options.confirmation_seed,
                        standard_error=0.01,
                        confirmation_hash=canonical_digest((subject, repeat, raw), prefix="synthetic-confirmation-v1|"),
                    )
                )
    # Fixture evidence deliberately uses no collection-policy identity and no
    # completed route object; W4 therefore returns capability=diagnostic.
    evidence = CalibrationEvidence(
        baseline_split_hash=baseline.baseline_split_hash,
        producer_fit_subjects=tuple(producer_subjects),
        fit_subjects=tuple(fit_subjects),
        allowance_subjects=tuple(allowance_subjects),
        completed_trace_hashes=(receipt.trace_hash,),
        completed_trace_receipts=(receipt,),
        winner_bindings=tuple((row.subject_id, row.action_id, row.proposal_digest, row.action_digest, row.state_version) for row in fit_rows + allowance_rows),
        producer_compatibility_hash=producer_hash,
        value_fit_identity_hash=value_hash,
        gain_scale_hash=scale_hash,
        policy_hash=policy_hash,
        writer_hash=writer_hash,
        query_hash=query_hash,
        proposal_generator_hash=proposal_hash,
        config_hash=canonical_digest(config.as_dict(), prefix="pfgr-lite-calibration-config-v1|"),
        role_manifest=baseline,
        winner_confirmations=tuple((row.subject_id, row.action_id, row.proposal_digest, row.action_digest, row.measurement_mode, row.q_draws, row.seed, row.standard_error, row.confirmation_hash) for row in fit_rows + allowance_rows),
        value_input_variant=options.value_input_variant,
        trace_budget=4,
        confirmation_mode="iid_fixed_q",
        confirmation_seed=options.confirmation_seed,
        confirmation_q_draws=options.confirmation_q_draws,
        confirmation_independence_hash=canonical_digest(sorted((row.subject_id, row.action_id, row.proposal_digest, row.action_digest, row.measurement_mode, row.q_draws, row.seed, row.standard_error, row.confirmation_hash) for row in fit_rows + allowance_rows), prefix="pfgr-lite-confirmation-set-v1|"),
        synthetic=True,
        target_free=True,
        sealed=True,
    )
    calibration = fit_calibration(fit_rows, allowance_rows, config, evidence=evidence)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "calibration_evidence.json", evidence.as_dict())
    _write_json(output_dir / "fit_winners.json", {"rows": [asdict(row) for row in fit_rows]})
    _write_json(output_dir / "allowance_winners.json", {"rows": [asdict(row) for row in allowance_rows]})
    _write_json(output_dir / "calibration.json", {"calibration": asdict(calibration), "capability": calibration.capability})
    metrics = {"synthetic": True, "target_free": True, "trace_count": 1, "fit_records": len(fit_rows), "allowance_records": len(allowance_rows), "capability": calibration.capability}
    _write_json(output_dir / "metrics.json", metrics)
    return {
        "schema_version": CALIBRATION_RUN_SCHEMA,
        "calibration_evidence": evidence,
        "fit_winners": tuple(fit_rows),
        "allowance_winners": tuple(allowance_rows),
        "completed_traces": (),
        "collection_policy": None,
        "calibration": calibration,
        "artifacts": {name: str(output_dir / name) for name in ("calibration_evidence.json", "fit_winners.json", "allowance_winners.json", "calibration.json", "metrics.json")},
        "metrics": metrics,
    }


def _collect_traces(inputs: Any, options: CalibrationRunOptions) -> tuple[tuple[ForcedCalibrationTrace, ...], dict[str, int], Any]:
    metadata = getattr(inputs, "metadata", {})
    callback = metadata.get("calibration_route_builder") if isinstance(metadata, Mapping) else None
    if callback is None and isinstance(metadata, Mapping):
        callback = metadata.get("forced_route_builder")
    samples = tuple(getattr(inputs, "samples", ()))
    manifest = _role_manifest(inputs)
    selected = [sample for sample in samples if _subject_role(manifest, sample.subject_id) in {"calibration_fit", "calibration_allowance"}]
    if options.max_subjects is not None:
        selected = selected[: options.max_subjects]
    if not selected:
        raise ValueError("calibration collection has no calibration-fit/allowance samples")
    from .types import OperationCounters

    stats = _new_collection_stats()
    # One runner-owned sink spans collection, replay, and target-after-trace
    # measurement for this S5 invocation.  Sharing it across every route is
    # what makes receipt counts additive and prevents per-route counters from
    # being silently dropped.
    operation_counters = OperationCounters()
    traces: list[ForcedCalibrationTrace] = []
    for index, sample in enumerate(selected):
        if callback is None:
            value = _default_forced_trace(inputs, sample, options, index=index, counters=operation_counters)
        else:
            value = _invoke(callback, sample, sample=sample, seed=options.collection_seed + index, options=options, config=_config(inputs), inputs=inputs)
        if not isinstance(value, ForcedCalibrationTrace):
            raise TypeError("calibration_route_builder must return a sealed ForcedCalibrationTrace")
        staged = _stage_cpu_value(value)
        if not isinstance(staged, ForcedCalibrationTrace):
            raise TypeError("CPU staged calibration route lost its typed wrapper")
        stats["staged_trace_count"] += 1
        stats["staged_tensor_bytes"] += _tensor_storage_bytes(staged)
        # Traces remain CPU-owned until measurement so this is the observed
        # peak staged footprint, not a requested estimate.
        stats["peak_staged_tensor_bytes"] = max(stats["peak_staged_tensor_bytes"], stats["staged_tensor_bytes"])
        traces.append(staged)
    return tuple(traces), stats, operation_counters


def _measure_traces(
    inputs: Any,
    options: CalibrationRunOptions,
    traces: Sequence[ForcedCalibrationTrace],
    policy: Any,
    producer: Any,
    *,
    stats: dict[str, int] | None = None,
    counters: Any | None = None,
) -> tuple[tuple[CalibrationWinner, ...], tuple[CalibrationWinner, ...]]:
    metadata = getattr(inputs, "metadata", {})
    callback = metadata.get("calibration_winner_measure") if isinstance(metadata, Mapping) else None
    fit: list[CalibrationWinner] = []
    allowance: list[CalibrationWinner] = []
    manifest = _role_manifest(inputs)
    for trace in traces:
        role = _subject_role(manifest, trace.subject_id)
        if callback is None:
            rows = _default_measure_trace(inputs, trace, options, stats=stats, counters=counters)
        else:
            rows = _winner_records(_invoke(callback, trace, role, inputs=inputs, options=options, target_provider=getattr(inputs, "target_provider", None), config=_config(inputs)))
        if role == "calibration_fit":
            fit.extend(rows)
        elif role == "calibration_allowance":
            allowance.extend(rows)
        else:
            raise ValueError("forced trace subject is not a calibration role")
    _validate_winners(fit, traces, role="calibration_fit", policy=policy, producer=producer, options=options)
    _validate_winners(allowance, traces, role="calibration_allowance", policy=policy, producer=producer, options=options)
    return tuple(fit), tuple(allowance)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    import json

    path.write_text(json.dumps(value, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")


def _diagnostic_only_result(
    *,
    output_dir: Path,
    options: CalibrationRunOptions,
    traces: Sequence[ForcedCalibrationTrace],
    fit: Sequence[CalibrationWinner],
    allowance: Sequence[CalibrationWinner],
    reason: str,
    stats: Mapping[str, int] | None = None,
    operation_counters: Any | None = None,
) -> Mapping[str, Any]:
    """Publish actual collection measurements without minting a calibration.

    Small CPU/synthetic runs are useful for route and teacher smoke checks,
    but must remain explicitly insufficient for adaptive deployment.  The
    inconclusive result carries measurements and receipts, never a fitted
    score model or placeholder calibration.
    """

    # Underpowered real/engineering collection is not a calibration.  Return
    # no fitted GainCalibration at all so downstream policy/checkpoint paths
    # cannot accidentally treat a smoke run as adaptive evidence.
    calibration = None
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "trace_receipts.json", {"rows": [trace.receipt.as_dict() for trace in traces]})
    _write_json(output_dir / "fit_winners.json", {"rows": [asdict(row) for row in fit]})
    _write_json(output_dir / "allowance_winners.json", {"rows": [asdict(row) for row in allowance]})
    collection_stats = dict(stats or {})
    metrics = {
        "synthetic": bool(options.engineering_only),
        "target_free": True,
        "trace_count": len(traces),
        "fit_records": len(fit),
        "allowance_records": len(allowance),
        "capability": None,
        "insufficient_data": True,
        "adaptive_blocked_reason": reason,
        "actual_forced_route_collection": True,
        "actual_teacher_measurement": bool(fit or allowance),
        "staged_trace_count": int(collection_stats.get("staged_trace_count", len(traces))),
        "staged_tensor_bytes": int(collection_stats.get("staged_tensor_bytes", 0)),
        "replay_count": int(collection_stats.get("replay_count", 0)),
        "peak_staged_tensor_bytes": int(collection_stats.get("peak_staged_tensor_bytes", collection_stats.get("staged_tensor_bytes", 0))),
        "collection_seed": int(options.collection_seed),
        "confirmation_seed": int(options.confirmation_seed),
        "confirmation_mode": options.confirmation_mode,
        "confirmation_q_draws": int(options.confirmation_q_draws),
        "measurement_modes": sorted({str(row.measurement_mode) for row in tuple(fit) + tuple(allowance)}),
        "measurement_q_draws": sorted({int(row.q_draws) for row in tuple(fit) + tuple(allowance)}),
        "collection_policy_hash": traces[0].collection_policy.policy_hash if traces else None,
        "bounded_cpu_staging": True,
        "staging_resource_limit_bytes": None,
        "staging_resource_scope": "measured_peak_tensor_bytes; no fixed byte cap claimed",
        "operation_counters": operation_counters.as_dict() if operation_counters is not None and hasattr(operation_counters, "as_dict") else {},
    }
    _write_json(output_dir / "calibration.json", {"calibration": None, "capability": None, "insufficient_data": True, "status": "INCONCLUSIVE", "reason": reason})
    _write_json(output_dir / "metrics.json", metrics)
    return {
        "schema_version": CALIBRATION_RUN_SCHEMA,
        "calibration_evidence": None,
        "fit_winners": tuple(fit),
        "allowance_winners": tuple(allowance),
        "completed_traces": tuple(traces),
        "collection_policy": traces[0].collection_policy if traces else None,
        "calibration": calibration,
        "operation_counters": operation_counters,
        "artifacts": {name: str(output_dir / name) for name in ("trace_receipts.json", "fit_winners.json", "allowance_winners.json", "calibration.json", "metrics.json")},
        "metrics": metrics,
    }


def run_calibration(inputs: Any, options: CalibrationRunOptions, output_dir: Path) -> Mapping[str, Any]:
    """Collect sealed forced traces, measure winners, and fit W4 calibration."""

    if not isinstance(options, CalibrationRunOptions):
        raise TypeError("options must be CalibrationRunOptions")
    output_dir = Path(output_dir)
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(f"calibration output directory must be empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    # The hand-built numeric fixture is intentionally private to unit tests;
    # the public CLI path always traverses the actual model/teacher seam.
    if options.engineering_only and isinstance(getattr(inputs, "metadata", None), Mapping) and getattr(inputs, "metadata", {}).get("private_numeric_fixture"):
        # Remove the empty directory before the fixture reserves its own
        # artifact root; this keeps one atomic output ownership boundary.
        output_dir.rmdir()
        return _synthetic_fixture(inputs, options, output_dir)
    config = _config(inputs)
    traces, collection_stats, operation_counters = _collect_traces(inputs, options)
    policy, producer = _policy_and_producer(inputs, traces)
    fit, allowance = _measure_traces(inputs, options, traces, policy, producer, stats=collection_stats, counters=operation_counters)
    if (
        len({row.subject_id for row in fit}) < 32
        or len({row.subject_id for row in allowance}) < 32
        or len(fit) < 64
        or len(allowance) < 64
    ):
        # ``run_calibration`` reserved this empty directory before collection;
        # the helper owns the actual artifact publication boundary.
        output_dir.rmdir()
        return _diagnostic_only_result(
            output_dir=output_dir,
            options=options,
            traces=traces,
            fit=fit,
            allowance=allowance,
            reason="calibration roles require >=32 subjects and >=64 measured winners per role",
            stats=collection_stats,
            operation_counters=operation_counters,
        )
    evidence = _evidence(config=config, role_manifest=_role_manifest(inputs), policy=policy, producer=producer, traces=traces, fit=fit, allowance=allowance, options=options)
    calibration = fit_calibration(fit, allowance, config, evidence=evidence, completed_traces=traces, producer=producer, collection_policy=policy)
    _write_json(output_dir / "calibration_evidence.json", evidence.as_dict())
    _write_json(output_dir / "fit_winners.json", {"rows": [asdict(row) for row in fit]})
    _write_json(output_dir / "allowance_winners.json", {"rows": [asdict(row) for row in allowance]})
    _write_json(output_dir / "trace_receipts.json", {"rows": [receipt.as_dict() for receipt in (trace.receipt for trace in traces)]})
    _write_json(output_dir / "collection_policy.json", policy.as_dict())
    _write_json(output_dir / "calibration.json", {"calibration": asdict(calibration), "evidence": evidence.as_dict()})
    metrics = {
        "synthetic": evidence.synthetic,
        "target_free": evidence.target_free,
        "trace_count": len(traces),
        "fit_records": len(fit),
        "allowance_records": len(allowance),
        "fit_subjects": len(set(row.subject_id for row in fit)),
        "allowance_subjects": len(set(row.subject_id for row in allowance)),
        "capability": calibration.capability,
        "collection_seed": int(options.collection_seed),
        "confirmation_seed": int(options.confirmation_seed),
        "confirmation_mode": options.confirmation_mode,
        "confirmation_q_draws": int(options.confirmation_q_draws),
        "measurement_modes": sorted({str(row.measurement_mode) for row in tuple(fit) + tuple(allowance)}),
        "measurement_q_draws": sorted({int(row.q_draws) for row in tuple(fit) + tuple(allowance)}),
        "collection_policy_hash": policy.policy_hash,
        "bounded_cpu_staging": True,
        "staging_resource_limit_bytes": None,
        "staging_resource_scope": "measured_peak_tensor_bytes; no fixed byte cap claimed",
        "staged_trace_count": int(collection_stats.get("staged_trace_count", len(traces))),
        "staged_tensor_bytes": int(collection_stats.get("staged_tensor_bytes", 0)),
        "replay_count": int(collection_stats.get("replay_count", 0)),
        "peak_staged_tensor_bytes": int(collection_stats.get("peak_staged_tensor_bytes", collection_stats.get("staged_tensor_bytes", 0))),
        "operation_counters": operation_counters.as_dict(),
    }
    _write_json(output_dir / "metrics.json", metrics)
    return {
        "schema_version": CALIBRATION_RUN_SCHEMA,
        "calibration_evidence": evidence,
        "fit_winners": fit,
        "allowance_winners": allowance,
        "completed_traces": traces,
        "collection_policy": policy,
        "calibration": calibration,
        "operation_counters": operation_counters,
        "artifacts": {name: str(output_dir / name) for name in ("calibration_evidence.json", "fit_winners.json", "allowance_winners.json", "trace_receipts.json", "collection_policy.json", "calibration.json", "metrics.json")},
        "metrics": metrics,
    }


__all__ = ["CALIBRATION_RUN_SCHEMA", "CalibrationRunOptions", "run_calibration"]
