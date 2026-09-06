"""Target-free PFGR-Lite inference orchestration.

This module owns the single public route loop.  It uses one effective policy,
the stored proposal rows, and explicit W2 writer/query injections.  Teacher,
target, oracle, and reconstruction modules never enter this import graph.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import torch
from torch import Tensor, nn

from .action_proposal import ActionWriter, PointQuery, apply_scored_action, propose_actions
from .policy import EffectivePolicy, select_or_stop
from .provenance import module_state_digest, tensor_digest
from .types import (
    ActionProposal,
    ActionProposalBatch,
    CompletedBehaviorTrace,
    Decision,
    DynamicTriPlanes,
    ObservationContext,
    OperationCounters,
    ParallelBehaviorTrace,
    PFGRRouteResult,
)


class ValueScorer(Protocol):
    """Narrow W3 V interface: score one frozen V126/222/270/366 batch."""

    def __call__(self, descriptors: Tensor) -> Tensor: ...


class CompoundWriter(Protocol):
    """Explicit frozen-bank writer used by the parallel diagnostic mode.

    The writer is called once with the initial state and selected immutable
    rows.  It must return one intermediate plane tensor per selected row;
    returning only a final tensor would make a sealed behavior trace
    impossible and is rejected.
    """

    def __call__(
        self,
        state: object,
        context: ObservationContext,
        actions: tuple[ActionProposal, ...],
    ) -> Sequence[DynamicTriPlanes]: ...


def _score_candidates(
    value_model: ValueScorer | Callable[[Tensor], Tensor],
    descriptors: Tensor,
) -> Tensor:
    if not callable(value_model):
        raise TypeError("value_model must be an explicit W3 descriptor scorer")
    with torch.no_grad():
        scores = value_model(descriptors)
    if not isinstance(scores, Tensor):
        raise TypeError("value_model must return a torch.Tensor")
    if scores.ndim == 3 and scores.shape[-1] == 1:
        scores = scores[..., 0]
    if scores.ndim != 2 or scores.shape[:2] != descriptors.shape[:2]:
        raise ValueError("value_model must return [B,N] or [B,N,1] candidate scores")
    if not scores.is_floating_point() or not bool(torch.isfinite(scores).all()):
        raise ValueError("value_model candidate scores must be finite floating values")
    return scores


def _validate_value_model_identity(value_model: object, policy: EffectivePolicy) -> None:
    identity = policy.value_fit_identity
    if identity is None:
        return
    if isinstance(value_model, nn.Module):
        actual_weights = module_state_digest(value_model)
    else:
        actual_weights = getattr(value_model, "weights_hash", None)
    if not isinstance(actual_weights, str) or actual_weights != identity.weights_hash:
        raise ValueError("value model weights do not match the exact ValueFitIdentity")
    actual_architecture = getattr(value_model, "architecture_hash", None)
    if not isinstance(actual_architecture, str) or actual_architecture != identity.architecture_hash:
        raise ValueError("value model architecture does not match the exact ValueFitIdentity")


def _value_descriptors(proposals: ActionProposalBatch, variant: int) -> Tensor:
    """Select one canonical stored V descriptor without recomputing U."""

    if variant == 126:
        result = proposals.v126
    elif variant == 222:
        result = torch.cat((proposals.v126, proposals.delta), dim=-1)
    elif variant == 270:
        result = proposals.o270
    elif variant == 366:
        # MAIN V366 is o270 + the actual bounded correction emitted by U.
        result = torch.cat((proposals.o270, proposals.delta), dim=-1)
    else:
        raise ValueError(f"unsupported V descriptor variant: {variant}")
    if result.shape[-1] != variant:
        raise RuntimeError("stored V descriptor has an unexpected channel count")
    return result


def _finish_route(
    *,
    context: ObservationContext,
    states: list[object],
    proposals: list[ActionProposalBatch],
    decisions: list[Decision],
    executed_action_ids: list[str],
    final_state: object,
    stop_reason: str,
    counters: OperationCounters,
    policy: EffectivePolicy,
    parallel_trace: ParallelBehaviorTrace | None = None,
) -> PFGRRouteResult:
    """Seal and retain all target-free route material on the result object."""

    if not states:
        raise RuntimeError("route state chain is empty")
    trace_proposals = list(proposals)
    trace_decisions = list(decisions)
    terminal_proposals: ActionProposalBatch | None = None
    # W1's sealed trace represents transitions, so a scored stop/no-legal
    # batch has no corresponding next state.  Keep that batch/decision on the
    # public route result, but exclude it from the transition-only trace.
    if parallel_trace is None and trace_proposals and trace_decisions and trace_decisions[-1].stop_code != "continue":
        terminal_proposals = trace_proposals.pop()
        trace_decisions.pop()
    if parallel_trace is None and (len(states) != len(trace_proposals) + 1 or len(trace_proposals) != len(trace_decisions)):
        raise RuntimeError("route state/proposal chain is incomplete")
    trace = None if parallel_trace is not None else CompletedBehaviorTrace(
        context_id=context.context_id,
        states=tuple(states),
        proposals=tuple(trace_proposals),
        decisions=tuple(trace_decisions),
    )
    result = PFGRRouteResult(
        final_state=final_state,
        decisions=tuple(decisions),
        executed_action_ids=tuple(executed_action_ids),
        k=final_state.state_version,
        stop_reason=stop_reason,
        counters=counters,
        context_id=context.context_id,
        policy_hash=policy.policy_hash,
        completed_trace=trace,
        parallel_trace=parallel_trace,
        terminal_proposals=terminal_proposals,
    )
    return result


def _parallel_scores(policy: EffectivePolicy, scores: Tensor) -> tuple[Tensor, Tensor, Tensor, float]:
    raw_scores = scores * float(policy.gain_scale)
    if policy.calibration is None:
        calibrated = raw_scores
        allowance = 0.0
    else:
        calibrated = float(policy.calibration.a) * raw_scores + float(policy.calibration.b)
        allowance = float(policy.calibration.allowance)
    conservative = calibrated - allowance - policy.quality_margin - policy.compute_cost
    if not bool(torch.isfinite(calibrated).all()) or not bool(torch.isfinite(conservative).all()):
        raise ValueError("nonfinite calibrated or conservative scores are an explicit numerical failure")
    return calibrated, conservative, raw_scores, allowance


def run_pfgr_inference(
    model: object,
    observation_context: ObservationContext,
    effective_policy: EffectivePolicy,
    *,
    query: PointQuery | Callable[..., Tensor],
    writer: ActionWriter | Callable[..., object],
    value_model: ValueScorer | Callable[[Tensor], Tensor] | None = None,
    counters: OperationCounters | None = None,
    legal_mask: Tensor | Callable[..., Tensor] | None = None,
    compound_writer: CompoundWriter | Callable[..., Sequence[DynamicTriPlanes]] | None = None,
    query_version: str | None = None,
    query_hash: str | None = None,
    writer_version: str | None = None,
    writer_hash: str | None = None,
) -> PFGRRouteResult:
    """Run one bounded target-free route from an already sealed context.

    ``query`` and ``writer`` are mandatory injections.  This function does not
    fabricate a legacy query/writer, and it performs no target or teacher
    reads.  Subject batches are intentionally serialized at B=1 until a
    corresponding metadata-complete batched contract is authorized.
    """

    if not isinstance(observation_context, ObservationContext):
        raise TypeError("observation_context must be ObservationContext")
    if not isinstance(effective_policy, EffectivePolicy):
        raise TypeError("effective_policy must be EffectivePolicy")
    if not callable(query) or not callable(writer):
        raise TypeError("run_pfgr_inference requires explicit query and writer injections")
    if observation_context.frontend.s_coarse.shape[0] != 1:
        raise ValueError("PFGR public inference currently processes subjects serially with B=1")
    if observation_context.producer.compatibility_hash != effective_policy.producer_compatibility_hash:
        raise ValueError("effective policy producer does not match observation context")
    if not hasattr(model, "initialize_state") or not hasattr(model, "updater"):
        raise TypeError("model must expose initialize_state(context) and the shared updater")
    live_producer = getattr(model, "producer_dependencies", getattr(model, "producer", None))
    if live_producer is not None:
        live_hash = getattr(live_producer, "compatibility_hash", getattr(live_producer, "digest", None))
        if live_hash is not None and live_hash != observation_context.producer.compatibility_hash:
            raise ValueError("model producer dependencies are stale or incompatible with context")
    if counters is None:
        counters = OperationCounters()

    # Deployment inference is target-free and does not retain reconstruction
    # gradients through V/selection.  Direct propose/apply calls remain
    # differentiable for the W3 S1 training schedule.
    with torch.no_grad():
        state = model.initialize_state(observation_context, role="deployment")
        if state.planes.xy.shape[0] != 1:
            raise ValueError("model state producer must preserve the subject B=1 seam")
        counters.add(behavior_states=1)
        states: list[object] = [state]
        proposals_history: list[ActionProposalBatch] = []
        if effective_policy.budget == 0 or effective_policy.mode in ("noop", "static"):
            return _finish_route(
                context=observation_context,
                states=states,
                proposals=proposals_history,
                decisions=[],
                executed_action_ids=[],
                final_state=state,
                stop_reason="budget",
                counters=counters,
                policy=effective_policy,
            )
        if legal_mask is None:
            raise ValueError("nonzero-budget inference requires an explicit writer-support legal_mask injection")

        decisions: list[Decision] = []
        executed_action_ids: list[str] = []
        stop_reason = "budget"

        if effective_policy.mode == "parallel_topk":
            if not callable(compound_writer):
                raise ValueError("parallel_topk requires an explicit compound_writer injection")
            proposals = propose_actions(
                model.updater,
                state,
                observation_context,
                query=query,
                candidate_chunk_size=effective_policy.candidate_chunk_size,
                counters=counters,
                legal_mask=legal_mask,
                query_version=query_version,
                query_hash=query_hash,
                writer_version=writer_version,
                writer_hash=writer_hash,
            )
            proposals_history.append(proposals)
            if value_model is None:
                raise ValueError("parallel_topk requires an explicit W3 value scorer")
            _validate_value_model_identity(value_model, effective_policy)
            predicted = _score_candidates(value_model, _value_descriptors(proposals, effective_policy.value_input_variant))
            counters.add(value_evaluations=int(predicted.numel()))
            legal = proposals.legal.to(dtype=torch.bool) & (proposals.delta.abs().amax(dim=-1) > 0.0)
            if not bool(legal.any()):
                decisions.append(select_or_stop(proposals, predicted, effective_policy, step=0))
                return _finish_route(
                    context=observation_context,
                    states=states,
                    proposals=proposals_history,
                    decisions=decisions,
                    executed_action_ids=executed_action_ids,
                    final_state=state,
                    stop_reason="no_legal_action",
                    counters=counters,
                    policy=effective_policy,
                )
            calibrated, conservative, raw_scores, allowance = _parallel_scores(effective_policy, predicted)
            indices = [index for index in range(proposals.point_ids.shape[1]) if bool(legal[0, index])]
            indices.sort(key=lambda index: (-float(conservative[0, index].item()), int(proposals.point_ids[0, index].item())))
            indices = indices[: min(effective_policy.budget, len(indices))]
            actions = tuple(proposals.row(0, index) for index in indices)
            outputs = compound_writer(state, observation_context, actions)
            if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)) or len(outputs) != len(actions):
                raise ValueError("compound_writer must return one intermediate DynamicTriPlanes per selected action")
            current = state
            for step, (index, action, next_planes) in enumerate(zip(indices, actions, outputs)):
                if not isinstance(next_planes, DynamicTriPlanes):
                    raise TypeError("compound_writer outputs must be DynamicTriPlanes")
                # Every selected row remains bound to the original initial
                # state.  Parallel diagnostics intentionally do not fabricate
                # new proposal/action identities for intermediate states.
                selected = action
                decision = Decision(
                    selected_point_id=selected.point_id,
                    proposal_digest=proposals.proposal_digest,
                    action_digest=selected.action_digest,
                    active=True,
                    raw_value=float(raw_scores[0, index].item()),
                    calibrated_value=float(calibrated[0, index].item()),
                    conservative_value=float(conservative[0, index].item()),
                    allowance=allowance,
                    quality_margin=effective_policy.quality_margin,
                    compute_cost=effective_policy.compute_cost,
                    policy_hash=effective_policy.policy_hash,
                    stop_code="continue",
                    step=step,
                )
                decisions.append(decision)
                current = current.next(next_planes)
                states.append(current)
                executed_action_ids.append(selected.action_id)
            parallel_trace = ParallelBehaviorTrace(
                context_id=observation_context.context_id,
                initial_state=state,
                proposals=proposals,
                selected_action_ids=tuple(action.action_id for action in actions),
                selected_action_digests=tuple(action.action_digest for action in actions),
                selected_delta_digests=tuple(
                    tensor_digest(action.delta, name="delta")
                    for action in actions
                ),
                intermediate_states=tuple(states[1:]),
                policy_hash=effective_policy.policy_hash,
            )
            return _finish_route(
                context=observation_context,
                states=states,
                proposals=[proposals],
                decisions=decisions,
                executed_action_ids=executed_action_ids,
                final_state=current,
                stop_reason="budget",
                counters=counters,
                policy=effective_policy,
                parallel_trace=parallel_trace,
            )

        while state.state_version < effective_policy.budget:
            proposals = propose_actions(
                model.updater,
                state,
                observation_context,
                query=query,
                candidate_chunk_size=effective_policy.candidate_chunk_size,
                counters=counters,
                legal_mask=legal_mask,
                query_version=query_version,
                query_hash=query_hash,
                writer_version=writer_version,
                writer_hash=writer_hash,
            )
            proposals_history.append(proposals)
            # Legal support is checked before invoking V.  A genuine no-legal
            # terminal assessment does not require a value model, and must not
            # manufacture a V call merely to produce a stop decision.
            legal = proposals.legal.to(dtype=torch.bool) & (proposals.delta.abs().amax(dim=-1) > 0.0)
            if not bool(legal.any()):
                predicted = torch.zeros(
                    proposals.point_ids.shape,
                    dtype=proposals.delta.dtype,
                    device=proposals.delta.device,
                )
            elif effective_policy.mode == "random":
                # Scores are intentionally unused by random diagnostics.  A
                # finite zero tensor still satisfies the typed selector while
                # avoiding any V call.
                predicted = None
            else:
                if value_model is None:
                    raise ValueError("this policy mode requires an explicit W3 value scorer")
                _validate_value_model_identity(value_model, effective_policy)
                predicted = _score_candidates(value_model, _value_descriptors(proposals, effective_policy.value_input_variant))
                counters.add(value_evaluations=int(predicted.numel()))
            decision = select_or_stop(
                proposals,
                predicted,
                effective_policy,
                step=state.state_version,
            )
            decisions.append(decision)
            if decision.stop_code != "continue":
                stop_reason = decision.stop_code
                break
            state = apply_scored_action(
                state,
                observation_context,
                proposals,
                decision,
                writer=writer,
                counters=counters,
            )
            states.append(state)
            selected_rows = (proposals.point_ids == decision.selected_point_id).nonzero(as_tuple=False)
            if selected_rows.shape[0] != 1:
                raise RuntimeError("selected action disappeared from the scored proposal batch")
            executed_action_ids.append(
                proposals.row(int(selected_rows[0, 0]), int(selected_rows[0, 1])).action_id
            )
            # No fifth candidate/V assessment is made after the fourth write.
            if state.state_version >= effective_policy.budget:
                stop_reason = "budget"
                break
        return _finish_route(
            context=observation_context,
            states=states,
            proposals=proposals_history,
            decisions=decisions,
            executed_action_ids=executed_action_ids,
            final_state=state,
            stop_reason=stop_reason,
            counters=counters,
            policy=effective_policy,
        )


__all__ = ["CompoundWriter", "ValueScorer", "run_pfgr_inference"]
