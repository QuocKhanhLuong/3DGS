"""Gate-E E5--E9 target-after-inference trajectory supervision.

The target-free model creates :class:`GateESupervisionContext` first.  Only
then does ``compute_training_objective`` receive T1ce supervision.  Keeping
that boundary explicit makes prediction, route, RewardNet prediction, and
final dynamic state independently reproducible for different targets.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor

from .contracts import FrontendOutput, VolumeGeometry
from .decoder import ImplicitTriPlaneDecoder, ReconstructionOutput
from .losses import ReconstructionLossConfig, ReconstructionLossResult, pointwise_charbonnier_by_subject, reconstruction_loss
from .reward import GateBDescriptorContext
from .reward_supervision import (
    CounterfactualConfig,
    CounterfactualRewardResult,
    _sample_target_support,
    _validate_target_and_mask,
    build_local_support_samples,
    counterfactual_reward_supervision,
)
from .spectral_query import FeatureGridGeometry
from .state_init import DynamicTriPlanes
from .trajectory import AdaptiveRewardCostTrajectory, _TrajectoryTrainingTrace


def _finite_nonnegative(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class SupervisionConfig:
    """All Gate-E E1--E8 coefficients and bounded counterfactual controls.

    Values are tunable initial hyperparameters, not a claim of a final
    training recipe.  Optimizer, schedule, data orchestration, and evaluation
    settings intentionally do not belong to this configuration.
    """

    lambda_ssim: float = 0.2
    lambda_grad: float = 0.1
    ssim_data_range: float = 1.0
    lambda_local: float = 0.5
    lambda_reward: float = 1.0
    lambda_monotonic: float = 0.2
    lambda_delta: float = 1e-4
    counterfactual_candidates: int = 32
    high_candidate_count: int = 16
    random_candidate_count: int = 15
    spill_weight_beta: float = 1.0
    spill_sample_count: int = 12

    def __post_init__(self) -> None:
        for name in (
            "lambda_ssim",
            "lambda_grad",
            "lambda_local",
            "lambda_reward",
            "lambda_monotonic",
            "lambda_delta",
            "spill_weight_beta",
        ):
            object.__setattr__(self, name, _finite_nonnegative(name, getattr(self, name)))
        data_range = float(self.ssim_data_range)
        if not math.isfinite(data_range) or data_range <= 0.0:
            raise ValueError("ssim_data_range must be positive and finite")
        object.__setattr__(self, "ssim_data_range", data_range)
        # Delegate bounded candidate validation to the exact E2 config rather
        # than silently duplicating a second policy here.
        counterfactual = CounterfactualConfig(
            counterfactual_candidates=self.counterfactual_candidates,
            high_candidate_count=self.high_candidate_count,
            random_candidate_count=self.random_candidate_count,
            spill_weight_beta=self.spill_weight_beta,
            spill_sample_count=self.spill_sample_count,
        )
        for name in (
            "counterfactual_candidates",
            "high_candidate_count",
            "random_candidate_count",
            "spill_sample_count",
        ):
            object.__setattr__(self, name, getattr(counterfactual, name))

    @property
    def reconstruction_config(self) -> ReconstructionLossConfig:
        return ReconstructionLossConfig(
            lambda_ssim=self.lambda_ssim,
            lambda_grad=self.lambda_grad,
            ssim_data_range=self.ssim_data_range,
        )

    @property
    def counterfactual_config(self) -> CounterfactualConfig:
        return CounterfactualConfig(
            counterfactual_candidates=self.counterfactual_candidates,
            high_candidate_count=self.high_candidate_count,
            random_candidate_count=self.random_candidate_count,
            spill_weight_beta=self.spill_weight_beta,
            spill_sample_count=self.spill_sample_count,
        )


# Explicit alternate spelling for callers that want the phase name while the
# compact ``SupervisionConfig`` remains the primary public API.
GateESupervisionConfig = SupervisionConfig


@dataclass(frozen=True)
class GateESupervisionContext:
    """Target-free intermediate result consumed later by Gate-E losses only.

    This is deliberately not an inference ``FrontendOutput`` or
    ``TrajectoryResult`` replacement.  Its state trace is transient and
    bounded by the already-tunable route length, never a K-by-full-volume
    prediction history.
    """

    frontend: FrontendOutput
    _trace: _TrajectoryTrainingTrace
    _trajectory: AdaptiveRewardCostTrajectory
    _decoder: ImplicitTriPlaneDecoder
    reconstruction: ReconstructionOutput
    gate_b_descriptors: GateBDescriptorContext
    feature_geometry: FeatureGridGeometry

    def __post_init__(self) -> None:
        if not isinstance(self.frontend, FrontendOutput):
            raise TypeError("frontend must be a FrontendOutput")
        if not isinstance(self._trace, _TrajectoryTrainingTrace):
            raise TypeError("_trace must be the private Gate-E trajectory trace")
        if not isinstance(self._trajectory, AdaptiveRewardCostTrajectory):
            raise TypeError("_trajectory must be the private trajectory that produced _trace")
        if not isinstance(self._decoder, ImplicitTriPlaneDecoder):
            raise TypeError("_decoder must be the private decoder paired with _trace")
        if not isinstance(self.reconstruction, ReconstructionOutput):
            raise TypeError("reconstruction must be a ReconstructionOutput")
        if not isinstance(self.gate_b_descriptors, GateBDescriptorContext):
            raise TypeError("gate_b_descriptors must be a GateBDescriptorContext")
        if not isinstance(self.feature_geometry, FeatureGridGeometry):
            raise TypeError("feature_geometry must be a FeatureGridGeometry")
        if self.reconstruction.geometry != self.frontend.geometry or self.feature_geometry.source_geometry != self.frontend.geometry:
            raise ValueError("Gate-E context geometries must agree with the frontend source geometry")
        final_state = self._trace.result.final_state
        if self._trace.states[-1] is not final_state:
            raise ValueError("Gate-E context trace must end at the trajectory final state")
        batch = self.frontend.s_coarse.shape[0]
        if final_state.xy.shape[0] != batch or self.reconstruction.prediction.shape[0] != batch:
            raise ValueError("Gate-E context batch dimensions must agree")
        if self.gate_b_descriptors.q_xy.shape[:2] != self.frontend.refined_points_ras_mm.shape[:2]:
            raise ValueError("Gate-B descriptors must align with frontend refined points")

    @property
    def trajectory(self):
        """Compact inference-compatible route diagnostics retained by the trace."""

        return self._trace.result


@dataclass(frozen=True)
class TrainingObjectiveResult:
    """Typed E8 scalar composition plus compact component evidence."""

    total: Tensor
    reconstruction: ReconstructionLossResult
    reward: Tensor
    local: Tensor
    monotonic: Tensor
    delta: Tensor
    reward_supervision: tuple[CounterfactualRewardResult, ...]
    local_step_count: int
    monotonic_pair_count: int
    delta_step_count: int
    components: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        scalar_fields = ("total", "reward", "local", "monotonic", "delta")
        reference = self.total
        for name in scalar_fields:
            value = getattr(self, name)
            if not isinstance(value, Tensor) or value.ndim != 0 or not value.is_floating_point() or not bool(torch.isfinite(value)):
                raise ValueError(f"{name} must be one finite scalar tensor")
            if value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError("Gate-E scalar components must share total dtype and device")
        for name in ("local_step_count", "monotonic_pair_count", "delta_step_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        checked = dict(self.components)
        if set(checked) != {"reconstruction", "reward", "local", "monotonic", "delta"}:
            raise ValueError("components must retain exactly the five E8 terms")
        for value in checked.values():
            if not isinstance(value, Tensor) or value.ndim != 0:
                raise ValueError("components must map to scalar tensors")
        object.__setattr__(self, "components", MappingProxyType(checked))


def _subset_state(state: DynamicTriPlanes, rows: Tensor) -> DynamicTriPlanes:
    return DynamicTriPlanes(xy=state.xy[rows], xz=state.xz[rows], yz=state.yz[rows])


def _weighted_scalar_mean(values: list[Tensor], counts: list[int], zero: Tensor) -> Tensor:
    if len(values) != len(counts):
        raise ValueError("values and counts must align")
    total_count = sum(counts)
    if total_count == 0:
        return zero
    weighted = torch.stack(tuple(value * int(count) for value, count in zip(values, counts) if count > 0)).sum()
    return weighted / weighted.new_tensor(float(total_count))


def _actual_local_errors(
    decoder: ImplicitTriPlaneDecoder,
    state_before: DynamicTriPlanes,
    state_after: DynamicTriPlanes,
    selected_points_ras_mm: Tensor,
    feature_geometry: FeatureGridGeometry,
    output_geometry: VolumeGeometry,
    safe_target: Tensor,
    valid_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    samples = build_local_support_samples(selected_points_ras_mm, output_geometry)
    sampled_target, sampled_valid = _sample_target_support(safe_target, valid_mask, samples, output_geometry)
    before = pointwise_charbonnier_by_subject(
        decoder.decode_points(state_before, samples.points_ras_mm, feature_geometry),
        sampled_target,
        sampled_valid,
        allow_empty=True,
    )
    after = pointwise_charbonnier_by_subject(
        decoder.decode_points(state_after, samples.points_ras_mm, feature_geometry),
        sampled_target,
        sampled_valid,
        allow_empty=True,
    )
    contributing = sampled_valid.squeeze(-1).any(dim=1)
    return before, after, contributing


def _local_error_at_fixed_support(
    decoder: ImplicitTriPlaneDecoder,
    state: DynamicTriPlanes,
    support_centres_ras_mm: Tensor,
    feature_geometry: FeatureGridGeometry,
    output_geometry: VolumeGeometry,
    safe_target: Tensor,
    valid_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Measure one state on a fixed 4-mm physical support per subject.

    E6 compares post-update states on each subject's first executed support,
    rather than comparing losses from unrelated later selected regions.  The
    pointwise Charbonnier measurement is therefore the same compact local
    reconstruction metric as E5, with a stable physical domain across route
    steps and no decoded volume history.
    """

    samples = build_local_support_samples(support_centres_ras_mm, output_geometry)
    sampled_target, sampled_valid = _sample_target_support(safe_target, valid_mask, samples, output_geometry)
    error = pointwise_charbonnier_by_subject(
        decoder.decode_points(state, samples.points_ras_mm, feature_geometry),
        sampled_target,
        sampled_valid,
        allow_empty=True,
    )
    return error, sampled_valid.squeeze(-1).any(dim=1)


def _compute_training_objective(
    context: GateESupervisionContext,
    target: Tensor,
    *,
    config: SupervisionConfig | None = None,
    valid_mask: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> TrainingObjectiveResult:
    """Compute E1--E8 only after target-free context construction.

    ``target`` enters this function, never frontend, trajectory, RewardNet
    descriptor construction, selector, updater, or decoder input APIs.
    """

    if not isinstance(context, GateESupervisionContext):
        raise TypeError("context must be a GateESupervisionContext")
    trajectory = context._trajectory
    decoder = context._decoder
    config = config or SupervisionConfig()
    if not isinstance(config, SupervisionConfig):
        raise TypeError("config must be a SupervisionConfig")
    prediction = context.reconstruction.prediction
    reconstruction = reconstruction_loss(prediction, target, valid_mask, config=config.reconstruction_config)
    safe_target, support_mask = _validate_target_and_mask(
        target,
        valid_mask,
        batch=prediction.shape[0],
        dtype=prediction.dtype,
        device=prediction.device,
        geometry=context.reconstruction.geometry,
    )
    trace = context._trace
    frontend = context.frontend
    zero = prediction.sum() * 0.0
    reward_results: list[CounterfactualRewardResult] = []
    reward_values: list[Tensor] = []
    reward_counts: list[int] = []
    local_values: list[Tensor] = []
    local_counts: list[int] = []
    monotonic_values: list[Tensor] = []
    monotonic_counts: list[int] = []
    delta_values: list[Tensor] = []
    delta_counts: list[int] = []
    batch = prediction.shape[0]
    # E6 retains only one physical support centre and one scalar error per
    # subject.  This is deliberately not a public/all-state trace and avoids
    # comparing different selected regions across steps.
    has_monotonic_anchor = torch.zeros(batch, dtype=torch.bool, device=prediction.device)
    monotonic_anchor_points = torch.zeros(batch, 3, dtype=prediction.dtype, device=prediction.device)
    previous_monotonic_error = torch.zeros(batch, dtype=prediction.dtype, device=prediction.device)

    for step_index, step in enumerate(trace.result.steps):
        active = step.selected_indices >= 0
        if not bool(active.any()):
            continue
        selected = step.selected_indices[active]
        state_before = _subset_state(trace.states[step_index], active)
        state_after = _subset_state(trace.states[step_index + 1], active)
        points = frontend.refined_points_ras_mm[active]
        semantic = frontend.point_semantic[active]
        evidence = frontend.f_spec[active]
        reliability = frontend.reliability[active]
        descriptors = GateBDescriptorContext(
            q_xy=context.gate_b_descriptors.q_xy[active],
            q_xz=context.gate_b_descriptors.q_xz[active],
            q_yz=context.gate_b_descriptors.q_yz[active],
        )
        reward_result = counterfactual_reward_supervision(
            trajectory,
            decoder,
            state_before,
            points,
            semantic,
            evidence,
            reliability,
            descriptors,
            context.feature_geometry,
            context.reconstruction.geometry,
            safe_target[active],
            selected_indices=selected,
            valid_mask=support_mask[active],
            config=config.counterfactual_config,
            generator=generator,
        )
        reward_results.append(reward_result)
        if reward_result.valid_count:
            reward_values.append(reward_result.loss)
            reward_counts.append(reward_result.valid_count)

        selected_points = points[torch.arange(points.shape[0], device=points.device), selected]
        before, after, contributing = _actual_local_errors(
            decoder,
            state_before,
            state_after,
            selected_points,
            context.feature_geometry,
            context.reconstruction.geometry,
            safe_target[active],
            support_mask[active],
        )
        active_rows = active.nonzero(as_tuple=False).squeeze(1)
        if bool(contributing.any()):
            local_values.append(after[contributing].mean())
            local_counts.append(int(contributing.sum().detach().cpu()))
        # For E6, use each subject's first *executed* 4-mm support for every
        # later post-state comparison.  Thus one step yields no pair, whereas
        # each later active step contributes ``relu(ell_{t+1} - ell_t)`` on a
        # fixed physical domain.  Stopped rows never participate again.
        anchor_points = torch.where(
            has_monotonic_anchor[active_rows].unsqueeze(1),
            monotonic_anchor_points[active_rows],
            selected_points,
        )
        monotonic_error, monotonic_available = _local_error_at_fixed_support(
            decoder,
            state_after,
            anchor_points,
            context.feature_geometry,
            context.reconstruction.geometry,
            safe_target[active],
            support_mask[active],
        )
        had_previous = has_monotonic_anchor[active_rows] & monotonic_available
        if bool(had_previous.any()):
            monotonic_values.append(
                torch.relu(
                    monotonic_error[had_previous]
                    - previous_monotonic_error[active_rows][had_previous]
                ).mean()
            )
            monotonic_counts.append(int(had_previous.sum().detach().cpu()))
        newly_anchored = ~has_monotonic_anchor[active_rows] & monotonic_available
        if bool(newly_anchored.any()):
            monotonic_anchor_points = monotonic_anchor_points.index_copy(
                0,
                active_rows[newly_anchored],
                selected_points[newly_anchored],
            )
        if bool(monotonic_available.any()):
            previous_monotonic_error = previous_monotonic_error.index_copy(
                0,
                active_rows[monotonic_available],
                monotonic_error[monotonic_available],
            )
            has_monotonic_anchor = has_monotonic_anchor.index_copy(
                0,
                active_rows[monotonic_available],
                torch.ones_like(monotonic_available[monotonic_available]),
            )
        update_norm = step.selected_update_norm[active]
        delta_values.append(update_norm.square().mean())
        delta_counts.append(int(active.sum().detach().cpu()))

    reward = _weighted_scalar_mean(reward_values, reward_counts, zero)
    local = _weighted_scalar_mean(local_values, local_counts, zero)
    monotonic = _weighted_scalar_mean(monotonic_values, monotonic_counts, zero)
    delta = _weighted_scalar_mean(delta_values, delta_counts, zero)
    total = (
        reconstruction.total
        + config.lambda_local * local
        + config.lambda_reward * reward
        + config.lambda_monotonic * monotonic
        + config.lambda_delta * delta
    )
    return TrainingObjectiveResult(
        total=total,
        reconstruction=reconstruction,
        reward=reward,
        local=local,
        monotonic=monotonic,
        delta=delta,
        reward_supervision=tuple(reward_results),
        local_step_count=sum(local_counts),
        monotonic_pair_count=sum(monotonic_counts),
        delta_step_count=sum(delta_counts),
        components={
            "reconstruction": reconstruction.total,
            "reward": reward,
            "local": local,
            "monotonic": monotonic,
            "delta": delta,
        },
    )


__all__ = [
    "GateESupervisionConfig",
    "GateESupervisionContext",
    "SupervisionConfig",
    "TrainingObjectiveResult",
]
