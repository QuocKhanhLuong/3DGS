"""Immutable PFGR-Lite action proposals and same-action execution.

W4 owns the proposal identity and selection-independent update wrapper.  The
point query and compact writer are deliberately injected protocols: W4 never
reimplements W2's geometry algebra and never silently falls back to the
legacy query/write path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
from torch import Tensor, nn

from ..updater import PlaneCorrections
from ..state_init import DynamicTriPlanes
from .provenance import canonical_digest, module_state_digest, tensor_digest
from .types import (
    ACTION_SCHEMA,
    ActionProposal,
    ActionProposalBatch,
    ObservationContext,
    OperationCounters,
    PFGRState,
)


ACTION_GENERATOR_VERSION = "pfgr-lite-action-generator-v1"
"""Version of the ordered U input and proposal construction."""

WRITER_VERSION = "compact-writeback-4mm-v1"
QUERY_VERSION = "pfgr-lite-point-query-v1"
GEOMETRY_VERSION = "pfgr-lite-static-geometry-v1"
POINT_VERSION = "point-candidate-geometry-v1"


class PointQuery(Protocol):
    """Injected state-at-point query used by proposal generation.

    The callable receives one subject's dynamic planes, ordered refined RAS-mm
    points, and the context's final feature geometry, returning ``[B,N,96]``.
    W2 may adapt its canonical lattice or a parameter-free point sampler to
    this exact protocol; no query implementation is imported here.
    """

    def __call__(
        self,
        state: DynamicTriPlanes,
        points_ras_mm: Tensor,
        feature_geometry: object,
    ) -> Tensor: ...


class ActionWriter(Protocol):
    """Injected compact writer for one already-scored action.

    The writer must consume the immutable action row and return the next
    dynamic planes.  It must not call U again, detach the supplied delta, or
    mutate the input state in place.
    """

    def __call__(
        self,
        state: PFGRState,
        context: ObservationContext,
        action: ActionProposal,
    ) -> DynamicTriPlanes: ...


def _positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _strict_binary_legal(legal: Tensor, *, expected: tuple[int, int]) -> Tensor:
    if not isinstance(legal, Tensor) or legal.shape != expected:
        raise ValueError(f"legal_mask must have shape {expected}")
    if legal.dtype == torch.bool:
        return legal.clone()
    if legal.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("legal_mask must be bool or integer 0/1")
    if not bool(((legal == 0) | (legal == 1)).all()):
        raise ValueError("legal_mask integer values must be exact 0/1")
    return legal.to(dtype=torch.bool).clone()


def _identity_from_injection(
    injection: object,
    *,
    names: tuple[str, ...],
    fallback: str,
    label: str,
) -> str:
    """Resolve an injected algorithm identity without accepting sentinels.

    W2's query and writer objects are intentionally injected across the W4
    seam.  When they expose an explicit version/hash, those values are bound
    into every proposal; otherwise the context's already-validated producer
    envelope is the only permitted fallback.
    """

    for name in names:
        value = getattr(injection, name, None)
        if value is not None:
            if not isinstance(value, str) or not value or value.lower() in {"unknown", "unset", "none", "null"}:
                raise ValueError(f"{label} identity must be a complete non-sentinel string")
            return value
    if not isinstance(fallback, str) or not fallback or fallback.lower() in {"unknown", "unset", "none", "null"}:
        raise ValueError(f"{label} identity fallback must be complete")
    return fallback


def _explicit_identity(value: object, *, label: str) -> str:
    """Validate caller-supplied algorithm identity without fallback."""

    if not isinstance(value, str) or not value or value.lower() in {"unknown", "unset", "none", "null"}:
        raise ValueError(f"{label} identity must be a complete non-sentinel string")
    return value


def _query_points(
    query: PointQuery | Callable[..., Tensor],
    state: PFGRState,
    points_ras_mm: Tensor,
    context: ObservationContext,
) -> Tensor:
    if not callable(query):
        raise TypeError("query must be an explicit callable PointQuery injection")
    queried = query(state.planes, points_ras_mm, context.feature_geometry)
    if not isinstance(queried, Tensor):
        raise TypeError("injected point query must return a torch.Tensor")
    if queried.ndim == 2:
        queried = queried.unsqueeze(0)
    expected = (points_ras_mm.shape[0], points_ras_mm.shape[1], 96)
    if queried.shape != expected:
        raise ValueError(f"injected point query must return shape {expected}")
    if queried.device != points_ras_mm.device or queried.dtype != points_ras_mm.dtype:
        raise ValueError("point query output must match point device and dtype")
    if not queried.is_floating_point() or not bool(torch.isfinite(queried).all()):
        raise ValueError("point query output must be finite floating values")
    return queried


def _run_updater(updater: nn.Module, values: Tensor, *, write_scale: float) -> Tensor:
    if not isinstance(updater, nn.Module):
        raise TypeError("updater must be the shared torch.nn.Module UpdateNet")
    output = updater(values, write_scale=write_scale)
    if isinstance(output, PlaneCorrections):
        packed = output.packed
    elif isinstance(output, Tensor):
        packed = output
    else:
        raise TypeError("UpdateNet injection must return PlaneCorrections or [rows,96] tensor")
    if packed.shape != (values.shape[0], 96):
        raise ValueError("UpdateNet output must have shape [rows,96]")
    if packed.device != values.device or packed.dtype != values.dtype:
        raise ValueError("UpdateNet output must match input device and dtype")
    if not packed.is_floating_point() or not bool(torch.isfinite(packed).all()):
        raise ValueError("UpdateNet output must be finite floating values")
    # UpdateNet itself owns the locked tanh bound.  Recheck the result at the
    # seam so a test/integration module cannot quietly widen it.
    if bool((packed.abs() > float(write_scale) + 1e-7).any()):
        raise ValueError("UpdateNet delta exceeds the locked write_scale bound")
    return packed


def _geometry_identity(context: ObservationContext) -> str:
    geometry = context.geometry
    feature = context.feature_geometry
    return canonical_digest(
        {
            "source_shape_dhw": geometry.shape_dhw,
            "source_voxel_to_ras_mm": geometry.voxel_to_ras_mm,
            "feature_shape_dhw": feature.shape_dhw,
            "feature_geometry": feature.feature_geometry.voxel_to_ras_mm,
            "feature_to_source_scale_dhw": feature.feature_to_source_scale_dhw,
            "feature_to_source_offset_dhw": feature.feature_to_source_offset_dhw,
            "operator_chain": feature.operator_chain,
            "version": GEOMETRY_VERSION,
        },
        prefix="pfgr-lite-action-geometry-v1|",
    )


def _point_identity(context: ObservationContext, points: Tensor) -> str:
    compatibility = context.producer.compatibility
    return canonical_digest(
        {
            "generator_version": ACTION_GENERATOR_VERSION,
            "candidate_geometry_hash": compatibility.candidate_geometry_hash,
            "context_id": context.context_id,
            "context_version": context.version,
            "points": tensor_digest(points, name="refined_points_ras_mm"),
        },
        prefix="pfgr-lite-point-identity-v1|",
    )


def _validate_state_context(state: PFGRState, context: ObservationContext) -> None:
    if not isinstance(state, PFGRState) or not isinstance(context, ObservationContext):
        raise TypeError("state and context must be PFGRState and ObservationContext")
    context.validate_integrity()
    state.validate_integrity()
    if state.context_id != context.context_id:
        raise ValueError("state/context IDs do not match")
    if not state.producer.matches(context.producer.compatibility):
        raise ValueError("state producer is stale or incompatible with context")
    if state.planes.xy.shape[0] != 1 or context.frontend.s_coarse.shape[0] != 1:
        raise ValueError("W4 proposal and writer seams currently require subject batch B=1")


def propose_actions(
    updater: nn.Module,
    state: PFGRState,
    context: ObservationContext,
    *,
    query: PointQuery | Callable[..., Tensor],
    candidate_chunk_size: int,
    counters: OperationCounters | None = None,
    legal_mask: Tensor | Callable[..., Tensor] | None = None,
    query_version: str | None = None,
    query_hash: str | None = None,
    writer_version: str | None = None,
    writer_hash: str | None = None,
    write_scale: float = 0.1,
) -> ActionProposalBatch:
    """Generate one bounded, ordered batch of actual U corrections.

    All rows are generated from the same state version.  Candidate chunks
    bound both query and U working sets, while concatenation preserves the
    exact point ordering.  The returned ``delta`` is the tensor that later
    execution must gather; U is never rerun by :func:`apply_scored_action`.
    """

    _positive_int("candidate_chunk_size", candidate_chunk_size)
    if not isinstance(write_scale, (int, float)) or isinstance(write_scale, bool) or float(write_scale) != 0.1:
        raise ValueError("PFGR write_scale is locked to exactly 0.1")
    _validate_state_context(state, context)
    if state.planes.xy.dtype not in (torch.float32, torch.float64):
        raise TypeError("PFGR proposals support only FP32 production or FP64 tests")
    if state.planes.xy.dtype != context.frontend.refined_points_ras_mm.dtype:
        raise ValueError("state and frontend point tensors must share dtype")
    if state.planes.xy.device != context.frontend.refined_points_ras_mm.device:
        raise ValueError("state and frontend point tensors must share device")
    points = context.frontend.refined_points_ras_mm
    semantic = context.frontend.point_semantic
    reliability = context.frontend.spectral_evidence.reliability
    f_spec = context.frontend.spectral_evidence.f_spec
    q_bar = context.q_bar
    batch, count, _ = points.shape
    if batch != 1 or count <= 0:
        raise ValueError("PFGR proposals require nonempty subject B=1 points")
    if semantic.shape[:2] != (batch, count) or reliability.shape[:2] != (batch, count) or f_spec.shape[:2] != (batch, count) or q_bar.shape[:2] != (batch, count):
        raise ValueError("frontend descriptors must share [B,N] point ordering")
    expected_updater_hash = context.producer.compatibility.updater_hash
    actual_updater_hash = module_state_digest(updater)
    if actual_updater_hash != expected_updater_hash:
        raise ValueError("updater producer is stale or incompatible with context")

    write_scale = 0.1
    z_chunks: list[Tensor] = []
    delta_chunks: list[Tensor] = []
    for start in range(0, count, candidate_chunk_size):
        stop = min(start + candidate_chunk_size, count)
        point_chunk = points[:, start:stop]
        z_chunk = _query_points(query, state, point_chunk, context)
        updater_input = torch.cat(
            (
                z_chunk,
                f_spec[:, start:stop],
                semantic[:, start:stop],
                reliability[:, start:stop],
            ),
            dim=-1,
        )
        if updater_input.shape[-1] != 270:
            raise RuntimeError("PFGR U input must be exactly o270=[z96,f_spec168,semantic3,reliability3]")
        delta_chunk = _run_updater(
            updater,
            updater_input.reshape(-1, 270),
            write_scale=float(write_scale),
        ).reshape(batch, stop - start, 96)
        z_chunks.append(z_chunk)
        delta_chunks.append(delta_chunk)
    z96 = torch.cat(z_chunks, dim=1)
    delta = torch.cat(delta_chunks, dim=1)
    v126 = torch.cat((z96, semantic, q_bar, reliability), dim=-1)
    o270 = torch.cat((z96, f_spec, semantic, reliability), dim=-1)
    if v126.shape != (batch, count, 126) or o270.shape != (batch, count, 270):
        raise RuntimeError("PFGR descriptor packing produced an unexpected shape")
    if callable(legal_mask):
        legal_mask = legal_mask(state, context, points)
    if legal_mask is None:
        # A zero correction is retained as a diagnostic row but is never an
        # adaptive correction candidate when an affine calibration intercept
        # is positive.
        legal = delta.abs().amax(dim=-1) > 0.0
    else:
        legal = _strict_binary_legal(legal_mask, expected=(batch, count))
        legal = legal & (delta.abs().amax(dim=-1) > 0.0)
    point_ids = torch.arange(count, device=points.device, dtype=torch.long).unsqueeze(0)
    compatibility = context.producer.compatibility
    point_identity_hash = _point_identity(context, points)
    geometry_hash = _geometry_identity(context)
    resolved_query_version = (
        _explicit_identity(query_version, label="query version")
        if query_version is not None
        else _identity_from_injection(
            query,
            names=("query_version", "version"),
            fallback=QUERY_VERSION,
            label="query version",
        )
    )
    resolved_query_hash = (
        _explicit_identity(query_hash, label="query hash")
        if query_hash is not None
        else _identity_from_injection(
            query,
            names=("query_hash", "algorithm_hash", "lattice_hash"),
            fallback=compatibility.geometry_query_version_hash,
            label="query hash",
        )
    )
    resolved_writer_version = (
        _explicit_identity(writer_version, label="writer version")
        if writer_version is not None
        else WRITER_VERSION
    )
    resolved_writer_hash = (
        _explicit_identity(writer_hash, label="writer hash")
        if writer_hash is not None
        else compatibility.writer_hash
    )
    proposal = ActionProposalBatch(
        context_id=context.context_id,
        context_version=context.version,
        producer_compatibility_hash=compatibility.digest,
        state_version=state.state_version,
        state_digest=state.state_digest,
        point_ids=point_ids,
        points_ras_mm=points,
        o270=o270,
        v126=v126,
        delta=delta,
        legal=legal,
        updater_version="update-net-270-128-96-v1",
        updater_producer_hash=actual_updater_hash,
        writer_version=resolved_writer_version,
        writer_hash=resolved_writer_hash,
        query_version=resolved_query_version,
        query_hash=resolved_query_hash,
        geometry_version=GEOMETRY_VERSION,
        geometry_hash=geometry_hash,
        point_version=POINT_VERSION,
        point_identity_hash=point_identity_hash,
        version=ACTION_SCHEMA,
    )
    if counters is not None:
        counters.add(candidate_proposals=count)
    return proposal


def apply_scored_action(
    state: PFGRState,
    context: ObservationContext,
    proposals: ActionProposalBatch,
    decision: object,
    *,
    writer: ActionWriter | Callable[..., DynamicTriPlanes],
    counters: OperationCounters | None = None,
) -> PFGRState:
    """Apply exactly the stored selected delta through an injected writer."""

    from .types import Decision

    _validate_state_context(state, context)
    if not isinstance(proposals, ActionProposalBatch) or not isinstance(decision, Decision):
        raise TypeError("proposals and decision must use PFGR typed contracts")
    proposals.validate_integrity()
    if proposals.context_id != context.context_id:
        raise ValueError("proposal/context IDs do not match")
    if proposals.state_version != state.state_version or proposals.state_digest != state.state_digest:
        raise ValueError("proposal state identity is stale")
    if proposals.producer_compatibility_hash != context.producer.compatibility_hash:
        raise ValueError("proposal producer is stale")
    if proposals.updater_producer_hash != context.producer.compatibility.updater_hash:
        raise ValueError("proposal U identity is stale")
    if decision.stop_code != "continue" or decision.selected_point_id < 0:
        raise ValueError("only a continuing selected decision can be executed")
    if decision.proposal_digest != proposals.proposal_digest:
        raise ValueError("decision is not bound to this scored proposal batch")
    locations = (proposals.point_ids == decision.selected_point_id).nonzero(as_tuple=False)
    if locations.shape[0] != 1:
        raise ValueError("selected point ID must occur exactly once in proposals")
    action = proposals.row(int(locations[0, 0]), int(locations[0, 1]))
    if decision.action_digest != action.action_digest:
        raise ValueError("decision action is not the stored proposal action")
    if not action.legal:
        raise ValueError("selected action is not legal")
    if not callable(writer):
        raise TypeError("writer must be an explicit injected ActionWriter")
    writer_version = getattr(writer, "writer_version", None)
    writer_hash = getattr(writer, "writer_hash", None)
    if writer_version is not None and writer_version != action.writer_version:
        raise ValueError("writer version does not match the scored proposal")
    if writer_hash is not None and writer_hash != action.writer_hash:
        raise ValueError("writer identity does not match the scored proposal")
    next_planes = writer(state, context, action)
    if not isinstance(next_planes, DynamicTriPlanes):
        raise TypeError("injected writer must return DynamicTriPlanes")
    expected_shapes = {
        "xy": state.planes.xy.shape,
        "xz": state.planes.xz.shape,
        "yz": state.planes.yz.shape,
    }
    if any(getattr(next_planes, name).shape != shape for name, shape in expected_shapes.items()):
        raise ValueError("writer output must preserve the dynamic tri-plane lattice shapes")
    if next_planes.xy.shape[0] != 1 or next_planes.xy.dtype != state.planes.xy.dtype or next_planes.xy.device != state.planes.xy.device:
        raise ValueError("writer output must preserve subject, dtype, and device")
    if any(not bool(torch.isfinite(getattr(next_planes, name)).all()) for name in ("xy", "xz", "yz")):
        raise ValueError("writer output must be finite")
    result = state.next(next_planes)
    if counters is not None:
        counters.add(executed_writes=1, behavior_states=1)
    return result


__all__ = [
    "ACTION_GENERATOR_VERSION",
    "ActionWriter",
    "GEOMETRY_VERSION",
    "POINT_VERSION",
    "PointQuery",
    "QUERY_VERSION",
    "WRITER_VERSION",
    "apply_scored_action",
    "propose_actions",
]
