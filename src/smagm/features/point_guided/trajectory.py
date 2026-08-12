"""Gate-C C7 composition of the locked adaptive reward-cost trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn

from .contracts import FrontendOutput, POINT_SPECTRAL_EVIDENCE_CHANNELS, VolumeGeometry
from .reward import (
    DynamicStatePointQuery,
    GateBDescriptorContext,
    RewardNet,
    build_reward_descriptor,
)
from .spectral_query import FeatureGridGeometry
from .state_init import DynamicStateInitializer, DynamicTriPlanes
from .trajectory_cost import TrajectoryConfig, route_utility, travel_cost
from .trajectory_solver import AdaptiveRouteSolver
from .triplane_projection import BaseTriPlanes
from .updater import PlaneCorrections, UpdateNet
from .writeback import CompactTriPlaneWriteback


class RouteAvailabilityPolicy(Protocol):
    """Optional caller-owned candidate-availability overlay for one route.

    Gate C supplies no policy and therefore remains revisit-capable. Gate F
    owns an explicit policy at its call boundary instead of altering generic
    route-solver semantics.
    """

    def initial_available(self, *, batch: int, point_count: int, device: torch.device) -> Tensor:
        """Return initial legal candidate flags with shape ``[B,N]``."""

    def mask_utility(self, utility: Tensor, available: Tensor) -> Tensor:
        """Apply a policy-owned eligibility overlay to finite utilities."""

    def update_available(self, available: Tensor, selection_indices: Tensor, active: Tensor) -> Tensor:
        """Return the next legal candidate flags after a selection."""


def _point_tensor(name: str, value: Tensor, *, width: int) -> None:
    if not isinstance(value, Tensor) or value.ndim != 3 or value.shape[-1] != width or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor [B,N,{width}]")
    if value.shape[0] <= 0 or value.shape[1] <= 0 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must have positive batch/point dimensions and finite values")


def _select(weights: Tensor, values: Tensor) -> Tensor:
    if values.shape[:2] != weights.shape:
        raise ValueError("selection weights and values must share [B,N]")
    return torch.einsum("bn,bnc->bc", weights, values)


def _subset_state(state: DynamicTriPlanes, rows: Tensor) -> DynamicTriPlanes:
    """Select only still-running subjects without changing their plane grids."""

    return DynamicTriPlanes(xy=state.xy[rows], xz=state.xz[rows], yz=state.yz[rows])


def _replace_running_state(state: DynamicTriPlanes, rows: Tensor, replacement: DynamicTriPlanes) -> DynamicTriPlanes:
    """Write changed rows back while preserving stopped rows bitwise."""

    row_indices = rows.nonzero(as_tuple=False).squeeze(1)
    return DynamicTriPlanes(
        xy=state.xy.index_copy(0, row_indices, replacement.xy),
        xz=state.xz.index_copy(0, row_indices, replacement.xz),
        yz=state.yz.index_copy(0, row_indices, replacement.yz),
    )


@dataclass(frozen=True)
class TrajectoryStepDiagnostics:
    """Compact selected-point diagnostics; no per-step dense candidate state."""

    selected_indices: Tensor  # [B], -1 for stopped batch rows
    selected_reward: Tensor  # [B]
    selected_travel: Tensor  # [B]
    selected_overlap: Tensor  # [B]
    selected_utility: Tensor  # [B]
    selected_update_norm: Tensor  # [B]
    max_utility: Tensor  # [B]
    candidate_evaluations: Tensor  # [B], actual dense RewardNet scores this step
    eligible_candidate_evaluations: Tensor  # [B], candidates eligible before masking


@dataclass(frozen=True)
class TrajectoryResult:
    """Final dynamic state and compact adaptive-route diagnostics only."""

    final_state: DynamicTriPlanes
    steps: tuple[TrajectoryStepDiagnostics, ...]
    stop_reasons: tuple[str, ...]
    candidate_evaluations: Tensor  # [B], cumulative actual dense RewardNet scores
    eligible_candidate_evaluations: Tensor  # [B], cumulative candidates eligible before masking

    def __post_init__(self) -> None:
        batch = self.final_state.xy.shape[0]
        if len(self.stop_reasons) != batch or any(not isinstance(reason, str) for reason in self.stop_reasons):
            raise ValueError("stop_reasons must contain one string per batch subject")
        for name in ("candidate_evaluations", "eligible_candidate_evaluations"):
            value = getattr(self, name)
            if (
                not isinstance(value, Tensor)
                or value.shape != (batch,)
                or value.dtype != torch.long
                or value.device != self.final_state.xy.device
                or bool((value < 0).any())
            ):
                raise ValueError(f"{name} must be a nonnegative device-matched [B] long tensor")
        if bool((self.eligible_candidate_evaluations > self.candidate_evaluations).any()):
            raise ValueError("eligible candidate evaluations cannot exceed actual dense RewardNet scores")

    @property
    def selected_indices(self) -> Tensor:
        if not self.steps:
            return torch.empty(
                self.final_state.xy.shape[0],
                0,
                dtype=torch.long,
                device=self.final_state.xy.device,
            )
        return torch.stack(tuple(step.selected_indices for step in self.steps), dim=1)

    @property
    def route_lengths(self) -> Tensor:
        return (self.selected_indices >= 0).sum(dim=1)


@dataclass(frozen=True)
class _TrajectoryTrainingTrace:
    """Private, transient live ``Z`` states paired with one route result.

    This intentionally retains only the dynamic states actually reached by the
    adaptive route: ``(Z0, Z1, ..., ZK)``.  It contains no target data and no
    per-step dense candidate tensors.  Existing compact step diagnostics carry
    hard selected indices and differentiable selected-update norms.
    """

    result: TrajectoryResult
    states: tuple[DynamicTriPlanes, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.result, TrajectoryResult):
            raise TypeError("result must be a TrajectoryResult")
        if len(self.states) != len(self.result.steps) + 1:
            raise ValueError("private trace must contain Z0 plus one state per executed route step")
        reference = self.result.final_state
        for state in self.states:
            if not isinstance(state, DynamicTriPlanes):
                raise TypeError("private trace states must be DynamicTriPlanes instances")
            for name in ("xy", "xz", "yz"):
                value = getattr(state, name)
                expected = getattr(reference, name)
                if value.shape != expected.shape or value.dtype != expected.dtype or value.device != expected.device:
                    raise ValueError("private trace states must retain one dynamic-state grid contract")
        if self.states[-1] is not self.result.final_state:
            raise ValueError("private trace final state must be the exact route final_state")


@dataclass(frozen=True)
class FrontendTrajectoryOutput:
    """Diagnostic Gate-C result that preserves the unchanged public Gate-B output."""

    frontend: FrontendOutput
    trajectory: TrajectoryResult


class AdaptiveRewardCostTrajectory(nn.Module):
    """One shared C1-C7 operator; only ``Z`` changes across route steps."""

    def __init__(self, config: TrajectoryConfig) -> None:
        super().__init__()
        if not isinstance(config, TrajectoryConfig):
            raise TypeError("config must be a TrajectoryConfig")
        self.config = config
        self.state_initializer = DynamicStateInitializer()
        self.dynamic_state_query = DynamicStatePointQuery()
        self.reward_net = RewardNet()
        self.route_solver = AdaptiveRouteSolver()
        self.update_net = UpdateNet()
        self.writeback = CompactTriPlaneWriteback(support_radius_mm=config.support_radius_mm)

    def _run(
        self,
        base_planes: BaseTriPlanes,
        refined_points_ras_mm: Tensor,
        point_semantic: Tensor,
        f_spec: Tensor,
        reliability: Tensor,
        gate_b_descriptors: GateBDescriptorContext,
        feature_geometry: FeatureGridGeometry,
        geometry: VolumeGeometry,
        *,
        trace_states: list[DynamicTriPlanes] | None,
        availability_policy: RouteAvailabilityPolicy | None,
        route_config: TrajectoryConfig | None = None,
    ) -> TrajectoryResult:
        if not isinstance(base_planes, BaseTriPlanes):
            raise TypeError("base_planes must be a BaseTriPlanes")
        if not isinstance(feature_geometry, FeatureGridGeometry) or not isinstance(geometry, VolumeGeometry):
            raise TypeError("feature_geometry and geometry must use typed geometry contracts")
        _point_tensor("refined_points_ras_mm", refined_points_ras_mm, width=3)
        _point_tensor("point_semantic", point_semantic, width=3)
        _point_tensor("f_spec", f_spec, width=POINT_SPECTRAL_EVIDENCE_CHANNELS)
        _point_tensor("reliability", reliability, width=3)
        reference = refined_points_ras_mm
        for name, value in (("point_semantic", point_semantic), ("f_spec", f_spec), ("reliability", reliability), ("q_xy", gate_b_descriptors.q_xy)):
            if value.shape[:2] != reference.shape[:2] or value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError(f"{name} must align with refined point batch, count, dtype, and device")
        if geometry != feature_geometry.source_geometry:
            raise ValueError("feature_geometry must be derived from the supplied source geometry")
        if route_config is not None and not isinstance(route_config, TrajectoryConfig):
            raise TypeError("route_config must be a TrajectoryConfig when supplied")
        active_config = self.config if route_config is None else route_config

        state = self.state_initializer(base_planes)
        if trace_states is not None:
            trace_states.append(state)
        batch, point_count, _ = refined_points_ras_mm.shape
        previous_indices = torch.full((batch,), -1, dtype=torch.long, device=reference.device)
        # This compact map is the only Gate-C history: it accumulates the
        # explicit maximum overlap cost and never excludes candidates itself.
        overlap = torch.zeros(batch, point_count, dtype=reference.dtype, device=reference.device)
        records: list[TrajectoryStepDiagnostics] = []
        candidate_evaluations = torch.zeros(batch, dtype=torch.long, device=reference.device)
        eligible_candidate_evaluations = torch.zeros(batch, dtype=torch.long, device=reference.device)
        stop_reasons: list[str | None] = [None] * batch
        running = torch.ones(batch, dtype=torch.bool, device=reference.device)
        available = (
            availability_policy.initial_available(batch=batch, point_count=point_count, device=reference.device)
            if availability_policy is not None
            else None
        )
        if available is not None and (
            not isinstance(available, Tensor)
            or available.shape != (batch, point_count)
            or available.dtype != torch.bool
            or available.device != reference.device
        ):
            raise ValueError("availability policy must return a device-matched bool tensor [B,N]")

        for _ in range(active_config.k_max):
            if availability_policy is not None:
                assert available is not None
                exhausted = running & ~available.any(dim=1)
                if bool(exhausted.any()):
                    for index in exhausted.nonzero(as_tuple=False).squeeze(1).tolist():
                        stop_reasons[index] = "candidates_exhausted"
                    running = running & ~exhausted
            if not bool(running.any()):
                break
            running_before_selection = running
            running_rows = running.nonzero(as_tuple=False).squeeze(1)
            eligible_now = (
                available[running].sum(dim=1, dtype=torch.long)
                if available is not None
                else torch.full((running_rows.numel(),), point_count, dtype=torch.long, device=reference.device)
            )
            # Dynamic state sampling and RewardNet are dense over all N
            # candidates of each running subject; availability masks utility
            # only after those actual computations have occurred.
            evaluated_now = torch.full((running_rows.numel(),), point_count, dtype=torch.long, device=reference.device)
            candidate_evaluations = candidate_evaluations.index_add(0, running_rows, evaluated_now)
            eligible_candidate_evaluations = eligible_candidate_evaluations.index_add(0, running_rows, eligible_now)
            running_state = _subset_state(state, running)
            running_points = refined_points_ras_mm[running]
            dynamic_samples = self.dynamic_state_query(running_state, running_points, feature_geometry)
            descriptor = build_reward_descriptor(
                dynamic_samples,
                point_semantic[running],
                GateBDescriptorContext(
                    q_xy=gate_b_descriptors.q_xy[running],
                    q_xz=gate_b_descriptors.q_xz[running],
                    q_yz=gate_b_descriptors.q_yz[running],
                ),
                reliability[running],
            )
            running_reward = self.reward_net(descriptor)
            running_travel = travel_cost(
                running_points,
                previous_indices[running],
                support_radius_mm=active_config.support_radius_mm,
            )
            running_utility = route_utility(running_reward, running_travel, overlap[running], active_config)
            if availability_policy is not None:
                assert available is not None
                running_utility = availability_policy.mask_utility(running_utility, available[running])
                if (
                    not isinstance(running_utility, Tensor)
                    or running_utility.shape != running_reward.shape
                    or running_utility.dtype != reference.dtype
                    or running_utility.device != reference.device
                    or not bool(torch.isfinite(running_utility).all())
                ):
                    raise ValueError("availability policy must return finite utility aligned with running candidates")
            reward = torch.zeros(batch, point_count, dtype=reference.dtype, device=reference.device).index_copy(0, running.nonzero(as_tuple=False).squeeze(1), running_reward)
            travel = torch.zeros_like(reward).index_copy(0, running.nonzero(as_tuple=False).squeeze(1), running_travel)
            utility = torch.zeros_like(reward).index_copy(0, running.nonzero(as_tuple=False).squeeze(1), running_utility)
            selection = self.route_solver(
                utility,
                running,
                training=self.training,
                temperature=active_config.selection_temperature,
            )
            # This latch is irreversible for the current batch execution.
            # A nonpositive subject never receives another update or choice.
            nonpositive = running_before_selection & ~selection.active
            if bool(nonpositive.any()):
                for index in nonpositive.nonzero(as_tuple=False).squeeze(1).tolist():
                    stop_reasons[index] = "nonpositive_utility"
            running = selection.active
            if not bool(selection.active.any()):
                break
            selected_within_running = selection.active[running_before_selection]

            safe_indices = selection.indices.clamp_min(0)
            selected_reward = reward.gather(1, safe_indices.unsqueeze(1)).squeeze(1) * selection.active
            selected_travel = travel.gather(1, safe_indices.unsqueeze(1)).squeeze(1) * selection.active
            selected_overlap = overlap.gather(1, safe_indices.unsqueeze(1)).squeeze(1) * selection.active
            selected_utility = utility.gather(1, safe_indices.unsqueeze(1)).squeeze(1) * selection.active
            updater_input = torch.cat(
                (
                    _select(selection.weights[selection.active], dynamic_samples.packed[selected_within_running]),
                    _select(selection.weights[selection.active], f_spec[selection.active]),
                    _select(selection.weights[selection.active], point_semantic[selection.active]),
                    _select(selection.weights[selection.active], reliability[selection.active]),
                ),
                dim=-1,
            )
            corrections = self.update_net(updater_input, write_scale=active_config.write_scale)
            selected_corrections = PlaneCorrections(xy=corrections.xy, xz=corrections.xz, yz=corrections.yz)
            update_norm = torch.zeros(batch, dtype=reference.dtype, device=reference.device).index_copy(
                0,
                selection.active.nonzero(as_tuple=False).squeeze(1),
                torch.linalg.vector_norm(selected_corrections.packed, dim=-1),
            )
            records.append(
                TrajectoryStepDiagnostics(
                    selected_indices=selection.indices,
                    selected_reward=selected_reward,
                    selected_travel=selected_travel,
                    selected_overlap=selected_overlap,
                    selected_utility=selected_utility,
                    selected_update_norm=update_norm,
                    max_utility=selection.max_utility,
                    candidate_evaluations=torch.zeros(batch, dtype=torch.long, device=reference.device).index_copy(
                        0,
                        running_rows,
                        evaluated_now,
                    ),
                    eligible_candidate_evaluations=torch.zeros(
                        batch, dtype=torch.long, device=reference.device
                    ).index_copy(0, running_rows, eligible_now),
                )
            )
            selected_points = _select(selection.weights[selection.active], refined_points_ras_mm[selection.active])
            updated_running_state = self.writeback(
                _subset_state(state, selection.active),
                selected_points,
                selected_corrections,
                feature_geometry,
            )
            state = _replace_running_state(state, selection.active, updated_running_state)
            if trace_states is not None:
                trace_states.append(state)

            previous_indices = torch.where(selection.active, selection.indices, previous_indices)
            if availability_policy is not None:
                assert available is not None
                available = availability_policy.update_available(available, selection.indices, selection.active)
                if (
                    not isinstance(available, Tensor)
                    or available.shape != (batch, point_count)
                    or available.dtype != torch.bool
                    or available.device != reference.device
                ):
                    raise ValueError("availability policy must retain a device-matched bool tensor [B,N]")
            distance = torch.linalg.vector_norm(
                refined_points_ras_mm[selection.active] - selected_points.unsqueeze(1),
                dim=-1,
            )
            selected_overlap_map = torch.square(
                torch.clamp(1.0 - distance / (2.0 * active_config.support_radius_mm), min=0.0)
            )
            overlap = overlap.index_copy(
                0,
                selection.active.nonzero(as_tuple=False).squeeze(1),
                torch.maximum(overlap[selection.active], selected_overlap_map),
            )

        reasons = tuple(
            stop_reasons[index]
            if stop_reasons[index] is not None
            else ("k_max" if bool(running[index].item()) else "nonpositive_utility")
            for index in range(batch)
        )
        return TrajectoryResult(
            final_state=state,
            steps=tuple(records),
            stop_reasons=reasons,
            candidate_evaluations=candidate_evaluations,
            eligible_candidate_evaluations=eligible_candidate_evaluations,
        )

    def forward(
        self,
        base_planes: BaseTriPlanes,
        refined_points_ras_mm: Tensor,
        point_semantic: Tensor,
        f_spec: Tensor,
        reliability: Tensor,
        gate_b_descriptors: GateBDescriptorContext,
        feature_geometry: FeatureGridGeometry,
        geometry: VolumeGeometry,
    ) -> TrajectoryResult:
        """Run the unchanged compact Gate-C diagnostic trajectory."""

        return self._run(
            base_planes,
            refined_points_ras_mm,
            point_semantic,
            f_spec,
            reliability,
            gate_b_descriptors,
            feature_geometry,
            geometry,
            trace_states=None,
            availability_policy=None,
        )

    def _forward_with_training_trace(
        self,
        base_planes: BaseTriPlanes,
        refined_points_ras_mm: Tensor,
        point_semantic: Tensor,
        f_spec: Tensor,
        reliability: Tensor,
        gate_b_descriptors: GateBDescriptorContext,
        feature_geometry: FeatureGridGeometry,
        geometry: VolumeGeometry,
        *,
        availability_policy: RouteAvailabilityPolicy | None = None,
    ) -> _TrajectoryTrainingTrace:
        """Run Gate C once and retain its private, transient live-state trace.

        This is a narrow Gate-E supervision seam.  It deliberately accepts the
        same observation-only inputs as :meth:`forward` and never accepts a
        target, loss, candidate subset, or training policy.
        """

        states: list[DynamicTriPlanes] = []
        result = self._run(
            base_planes,
            refined_points_ras_mm,
            point_semantic,
            f_spec,
            reliability,
            gate_b_descriptors,
            feature_geometry,
            geometry,
            trace_states=states,
            availability_policy=availability_policy,
        )
        return _TrajectoryTrainingTrace(result=result, states=tuple(states))


__all__ = [
    "AdaptiveRewardCostTrajectory",
    "FrontendTrajectoryOutput",
    "RouteAvailabilityPolicy",
    "TrajectoryResult",
    "TrajectoryStepDiagnostics",
]
