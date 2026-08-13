"""Focused Gate-E E5--E9 tests for target-after-inference supervision."""

from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from smagm.features.point_guided import PointGuidedConfig, PointGuidedMRIModel
from smagm.features.point_guided.losses import pointwise_charbonnier_by_subject
from smagm.features.point_guided.reward_supervision import build_local_support_samples
from smagm.features.point_guided.sampling import sample_volume_ras_mm
from smagm.features.point_guided.state_init import DynamicTriPlanes
from smagm.features.point_guided.training_objective import SupervisionConfig
from smagm.features.point_guided.trajectory import AdaptiveRewardCostTrajectory
from smagm.features.point_guided.trajectory_cost import TrajectoryConfig


def _make_model(*, k_max: int, positive_utility: bool, training: bool = False) -> PointGuidedMRIModel:
    """Build a tiny real frontend whose route length is deterministic in tests."""

    torch.manual_seed(4300 + k_max + int(positive_utility) + 10 * int(training))
    model = PointGuidedMRIModel(
        PointGuidedConfig(
            num_semantic_classes=3,
            num_points=3,
            point_candidate_multiplier=3,
            offset_hidden_channels=12,
        ),
        trajectory_config=TrajectoryConfig(
            lambda_travel=0.01,
            lambda_overlap=0.01,
            lambda_step=0.01,
            k_max=k_max,
            selection_temperature=0.7,
            write_scale=0.15,
        ),
    )
    assert model.trajectory is not None and model.decoder is not None
    # The real RewardNet remains the only object used at runtime.  A constant
    # logit simply makes this CPU fixture demonstrate either an economic stop
    # or the configured number of executed transitions reproducibly.
    first = model.trajectory.reward_net.network[0]
    last = model.trajectory.reward_net.network[2]
    assert isinstance(first, nn.Linear) and isinstance(last, nn.Linear)
    with torch.no_grad():
        first.weight.zero_()
        first.bias.zero_()
        last.weight.zero_()
        last.bias.fill_(3.0 if positive_utility else -10.0)
    return model.train(training)


def _input_volume(*, batch: int = 1) -> torch.Tensor:
    if batch <= 0:
        raise ValueError("batch must be positive")
    volume = torch.linspace(-1.0, 1.0, steps=3 * 7 * 7 * 7).reshape(1, 3, 7, 7, 7)
    return volume.expand(batch, -1, -1, -1, -1).clone()


def _config(**overrides: float | int) -> SupervisionConfig:
    values: dict[str, float | int] = {
        "lambda_ssim": 0.13,
        "lambda_grad": 0.07,
        "ssim_data_range": 1.0,
        "lambda_local": 0.31,
        "lambda_reward": 0.41,
        "lambda_monotonic": 0.43,
        "lambda_delta": 0.47,
        "counterfactual_candidates": 3,
        "high_candidate_count": 1,
        "random_candidate_count": 1,
        "spill_weight_beta": 0.75,
        # E4 spill behavior is separately covered in test_reward_supervision;
        # zero keeps these E5--E9 trace assertions compact and deterministic.
        "spill_sample_count": 0,
    }
    values.update(overrides)
    return SupervisionConfig(**values)  # type: ignore[arg-type]


def _subset_state(state: DynamicTriPlanes, active: torch.Tensor) -> DynamicTriPlanes:
    return DynamicTriPlanes(xy=state.xy[active], xz=state.xz[active], yz=state.yz[active])


def _actual_trace_metrics(
    model: PointGuidedMRIModel,
    context: object,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """Independently evaluate E5--E7 from the stored hard selected trace."""

    # Keep the test coupled to public typed context attributes, rather than
    # using the implementation's private E5 helper.
    frontend = context.frontend  # type: ignore[union-attr]
    trace = context._trace  # type: ignore[union-attr]
    feature_geometry = context.feature_geometry  # type: ignore[union-attr]
    output_geometry = context.reconstruction.geometry  # type: ignore[union-attr]
    assert model.decoder is not None
    monotonic_after_by_subject: dict[int, list[torch.Tensor]] = {}
    monotonic_anchor_by_subject: dict[int, torch.Tensor] = {}
    all_after: list[torch.Tensor] = []
    update_norms: list[torch.Tensor] = []

    for step_index, step in enumerate(trace.result.steps):
        active = step.selected_indices >= 0
        if not bool(active.any()):
            continue
        selected = step.selected_indices[active]
        points = frontend.refined_points_ras_mm[active]
        state_before = _subset_state(trace.states[step_index], active)
        state_after = _subset_state(trace.states[step_index + 1], active)
        selected_points = points[torch.arange(points.shape[0], device=points.device), selected]
        samples = build_local_support_samples(selected_points, output_geometry)
        sampled_target = sample_volume_ras_mm(target[active], samples.points_ras_mm, output_geometry)
        before = pointwise_charbonnier_by_subject(
            model.decoder.decode_points(state_before, samples.points_ras_mm, feature_geometry),
            sampled_target,
            samples.valid_mask,
        )
        after = pointwise_charbonnier_by_subject(
            model.decoder.decode_points(state_after, samples.points_ras_mm, feature_geometry),
            sampled_target,
            samples.valid_mask,
        )
        del before
        active_rows = active.nonzero(as_tuple=False).squeeze(1)
        for local_index, global_index in enumerate(active_rows.tolist()):
            # E6 must compare states on one invariant physical support, not
            # after-losses from distinct later selected points.  The first
            # executed point becomes this subject's compact 4-mm probe.
            anchor = monotonic_anchor_by_subject.setdefault(int(global_index), selected_points[local_index].detach())
            fixed_samples = build_local_support_samples(anchor.unsqueeze(0), output_geometry)
            fixed_target = sample_volume_ras_mm(target[active][local_index : local_index + 1], fixed_samples.points_ras_mm, output_geometry)
            fixed_after = pointwise_charbonnier_by_subject(
                model.decoder.decode_points(
                    DynamicTriPlanes(
                        xy=state_after.xy[local_index : local_index + 1],
                        xz=state_after.xz[local_index : local_index + 1],
                        yz=state_after.yz[local_index : local_index + 1],
                    ),
                    fixed_samples.points_ras_mm,
                    feature_geometry,
                ),
                fixed_target,
                fixed_samples.valid_mask,
            )
            monotonic_after_by_subject.setdefault(int(global_index), []).append(fixed_after[0])
            all_after.append(after[local_index])
        update_norms.append(step.selected_update_norm[active])

    assert all_after and update_norms
    after = torch.stack(all_after)
    monotonic_terms = [
        torch.relu(subject_after[index] - subject_after[index - 1])
        for subject_after in monotonic_after_by_subject.values()
        for index in range(1, len(subject_after))
    ]
    monotonic_pairs = torch.stack(monotonic_terms) if monotonic_terms else after.sum().reshape(1) * 0.0
    update_norm = torch.cat(update_norms)
    return (
        after.mean(),
        monotonic_pairs.mean() if monotonic_terms else after.sum() * 0.0,
        update_norm.square().mean(),
        int(after.numel()),
        len(monotonic_terms),
        int(update_norm.numel()),
    )


def _full_target_free_snapshot(context: object) -> tuple[torch.Tensor, ...]:
    """Capture every Gate-E-relevant target-free tensor with exact equality."""

    frontend = context.frontend  # type: ignore[union-attr]
    trace = context._trace  # type: ignore[union-attr]
    reconstruction = context.reconstruction  # type: ignore[union-attr]
    descriptors = context.gate_b_descriptors  # type: ignore[union-attr]
    sparse = frontend.sparse_pou
    tensors: list[torch.Tensor] = [
        frontend.s_coarse,
        frontend.initial_points_ras_mm,
        frontend.refined_points_ras_mm,
        frontend.displacement_ras_mm,
        frontend.point_semantic,
        sparse.batch_indices,
        sparse.voxel_indices_dhw,
        sparse.point_indices,
        sparse.raw_affinity,
        sparse.normalized_weight,
        sparse.unsupported_batch_indices,
        sparse.unsupported_voxel_indices_dhw,
        frontend.base_planes.xy,
        frontend.base_planes.xz,
        frontend.base_planes.yz,
        frontend.spectral_anchor.xy,
        frontend.spectral_anchor.xz,
        frontend.spectral_anchor.yz,
        frontend.f_spec,
        frontend.reliability,
        descriptors.q_xy,
        descriptors.q_xz,
        descriptors.q_yz,
        trace.result.selected_indices,
        reconstruction.prediction,
    ]
    for step in trace.result.steps:
        tensors.extend(
            (
                step.selected_indices,
                step.selected_reward,
                step.selected_travel,
                step.selected_overlap,
                step.selected_utility,
                step.selected_update_norm,
                step.max_utility,
            )
        )
    # Retain Z0 through ZK, including the explicit final ZK identity held by
    # ``trace.result.final_state``; no target is available while these exist.
    for state in trace.states:
        tensors.extend((state.xy, state.xz, state.yz))
    return tuple(tensor.detach().clone() for tensor in tensors)


def test_zero_step_objective_has_only_e1_and_keeps_trace_compact() -> None:
    model = _make_model(k_max=3, positive_utility=False)
    context = model.forward_training_context(_input_volume(), chunk_size=17)
    target = torch.zeros_like(context.reconstruction.prediction)

    assert context.trajectory.steps == ()
    assert not hasattr(context, "trajectory_trace")
    assert not hasattr(AdaptiveRewardCostTrajectory, "forward_with_training_trace")
    assert len(context._trace.states) == 1
    assert context.trajectory.route_lengths.tolist() == [0]
    assert context.trajectory.stop_reasons == ("nonpositive_utility",)

    result = model.compute_training_objective(
        context,
        target,
        config=_config(),
        generator=torch.Generator().manual_seed(71),
    )

    zero = context.reconstruction.prediction.sum() * 0.0
    torch.testing.assert_close(result.reward, zero)
    torch.testing.assert_close(result.local, zero)
    torch.testing.assert_close(result.monotonic, zero)
    torch.testing.assert_close(result.delta, zero)
    torch.testing.assert_close(result.total, result.reconstruction.total)
    assert result.reward_supervision == ()
    assert (result.local_step_count, result.monotonic_pair_count, result.delta_step_count) == (0, 0, 0)


@pytest.mark.parametrize(("k_max", "expected_pairs"), ((1, 0), (3, 2)), ids=("one_step", "variable_three_step"))
def test_actual_trace_drives_e5_e6_e7_and_weighted_e8_composition(k_max: int, expected_pairs: int) -> None:
    model = _make_model(k_max=k_max, positive_utility=True)
    context = model.forward_training_context(_input_volume(), chunk_size=19)
    target = torch.zeros_like(context.reconstruction.prediction)
    config = _config()

    result = model.compute_training_objective(
        context,
        target,
        config=config,
        generator=torch.Generator().manual_seed(911),
    )
    expected_local, expected_monotonic, expected_delta, local_count, monotonic_count, delta_count = _actual_trace_metrics(
        model,
        context,
        target,
    )

    assert len(context.trajectory.steps) == k_max
    assert len(context._trace.states) == k_max + 1
    assert all(step.selected_indices.item() >= 0 for step in context.trajectory.steps)
    torch.testing.assert_close(result.local, expected_local)
    torch.testing.assert_close(result.monotonic, expected_monotonic)
    torch.testing.assert_close(result.delta, expected_delta)
    assert (result.local_step_count, result.monotonic_pair_count, result.delta_step_count) == (
        local_count,
        monotonic_count,
        delta_count,
    )
    assert monotonic_count == expected_pairs
    assert len(result.reward_supervision) == k_max

    expected_total = (
        result.reconstruction.total
        + config.lambda_local * result.local
        + config.lambda_reward * result.reward
        + config.lambda_monotonic * result.monotonic
        + config.lambda_delta * result.delta
    )
    torch.testing.assert_close(result.total, expected_total)
    assert set(result.components) == {"reconstruction", "reward", "local", "monotonic", "delta"}
    torch.testing.assert_close(result.components["reconstruction"], result.reconstruction.total)
    torch.testing.assert_close(result.components["reward"], result.reward)
    torch.testing.assert_close(result.components["local"], result.local)
    torch.testing.assert_close(result.components["monotonic"], result.monotonic)
    torch.testing.assert_close(result.components["delta"], result.delta)


def test_target_arrives_after_an_unchanged_context_and_objective_backpropagates_only_to_model() -> None:
    model = _make_model(k_max=1, positive_utility=True, training=True)
    context = model.forward_supervision_context(_input_volume(), chunk_size=23)
    target_a = torch.zeros_like(context.reconstruction.prediction, requires_grad=True)
    target_b = torch.full_like(context.reconstruction.prediction, 0.7)
    config = _config()
    frozen_context = tuple(
        tensor.detach().clone()
        for tensor in (
            context.frontend.refined_points_ras_mm,
            context.frontend.f_spec,
            context.trajectory.selected_indices,
            *(plane for state in context._trace.states for plane in (state.xy, state.xz, state.yz)),
        )
    )

    first = model.compute_training_objective(
        context,
        target_a,
        config=config,
        generator=torch.Generator().manual_seed(404),
    )
    second = model.compute_training_objective(
        context,
        target_b,
        config=config,
        generator=torch.Generator().manual_seed(404),
    )

    live_context = tuple(
        tensor.detach()
        for tensor in (
            context.frontend.refined_points_ras_mm,
            context.frontend.f_spec,
            context.trajectory.selected_indices,
            *(plane for state in context._trace.states for plane in (state.xy, state.xz, state.yz)),
        )
    )
    assert all(torch.equal(before, after) for before, after in zip(frozen_context, live_context))
    assert not torch.isclose(first.total, second.total)
    assert torch.equal(first.reward_supervision[0].candidates.indices, second.reward_supervision[0].candidates.indices)

    model.zero_grad(set_to_none=True)
    first.total.backward()
    assert target_a.grad is None

    def has_nonzero_gradient(parameters: object) -> bool:
        return any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and bool(parameter.grad.abs().sum() > 0.0)
            for parameter in parameters  # type: ignore[union-attr]
        )

    assert model.trajectory is not None and model.decoder is not None
    assert has_nonzero_gradient(model.decoder.parameters())
    assert has_nonzero_gradient(model.trajectory.state_initializer.parameters())
    assert has_nonzero_gradient(model.trajectory.update_net.parameters())
    assert has_nonzero_gradient(model.trajectory.reward_net.parameters())
    assert has_nonzero_gradient(model.base_plane_projector.parameters())
    assert has_nonzero_gradient(model.spectral_anchor_builder.band_projector.parameters())
    assert all(parameter.grad is None for parameter in model.semantic_prior.backbone.parameters())

    target_free_methods = (
        "forward_frontend",
        "forward_trajectory",
        "forward_reconstruction",
        "forward_training_context",
        "forward_supervision_context",
    )
    for method_name in target_free_methods:
        parameters = inspect.signature(getattr(PointGuidedMRIModel, method_name)).parameters
        assert not any("target" in name.lower() for name in parameters)


def test_independent_eval_contexts_are_identical_before_targets_and_targets_change_only_objectives() -> None:
    """T1ce cannot affect any target-free frontend, route, Z, or prediction."""

    model = _make_model(k_max=2, positive_utility=True)
    x = _input_volume()
    first_context = model.forward_supervision_context(x, chunk_size=31)
    second_context = model.forward_supervision_context(x, chunk_size=31)

    first_snapshot = _full_target_free_snapshot(first_context)
    second_snapshot = _full_target_free_snapshot(second_context)
    assert len(first_snapshot) == len(second_snapshot)
    assert all(torch.equal(first, second) for first, second in zip(first_snapshot, second_snapshot))
    assert first_context.trajectory.stop_reasons == second_context.trajectory.stop_reasons

    target_a = torch.zeros_like(first_context.reconstruction.prediction)
    target_b = torch.full_like(second_context.reconstruction.prediction, 0.7)
    config = _config()
    first_objective = model.compute_training_objective(
        first_context,
        target_a,
        config=config,
        generator=torch.Generator().manual_seed(991),
    )
    second_objective = model.compute_training_objective(
        second_context,
        target_b,
        config=config,
        generator=torch.Generator().manual_seed(991),
    )

    assert not torch.isclose(first_objective.reconstruction.total, second_objective.reconstruction.total)
    assert not torch.isclose(first_objective.total, second_objective.total)
    assert tuple(item.candidates.indices.tolist() for item in first_objective.reward_supervision) == tuple(
        item.candidates.indices.tolist() for item in second_objective.reward_supervision
    )


class _TwoSubjectScheduledReward(nn.Module):
    """Make subject 1 stop after step 1 while subject 0 reaches ``K_max``."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(descriptor.shape[0])
        if len(self.batch_sizes) == 1:
            return torch.full(descriptor.shape[:2], 0.9, dtype=descriptor.dtype, device=descriptor.device)
        if len(self.batch_sizes) == 2:
            assert descriptor.shape[0] == 2
            return torch.stack(
                (
                    torch.full((descriptor.shape[1],), 0.9, dtype=descriptor.dtype, device=descriptor.device),
                    torch.zeros(descriptor.shape[1], dtype=descriptor.dtype, device=descriptor.device),
                )
            )
        # Later calls include route step 3 and the E2 reward regressions.
        # They do not change the already-latched route state.
        return torch.full(descriptor.shape[:2], 0.9, dtype=descriptor.dtype, device=descriptor.device)


def test_batched_variable_route_excludes_a_stopped_subject_from_all_step_losses() -> None:
    model = _make_model(k_max=3, positive_utility=True)
    assert model.trajectory is not None
    scheduled_reward = _TwoSubjectScheduledReward()
    model.trajectory.reward_net = scheduled_reward
    context = model.forward_training_context(_input_volume(batch=2), chunk_size=29)
    target = torch.zeros_like(context.reconstruction.prediction)
    config = _config()

    result = model.compute_training_objective(
        context,
        target,
        config=config,
        generator=torch.Generator().manual_seed(123),
    )
    expected_local, expected_monotonic, expected_delta, local_count, monotonic_count, delta_count = _actual_trace_metrics(
        model,
        context,
        target,
    )

    assert scheduled_reward.batch_sizes[:3] == [2, 2, 1]
    assert context.trajectory.route_lengths.tolist() == [3, 1]
    assert context.trajectory.selected_indices[1].tolist() == [0, -1, -1]
    assert context.trajectory.stop_reasons == ("k_max", "nonpositive_utility")
    assert all(step.selected_indices[1].item() == -1 for step in context.trajectory.steps[1:])
    assert all(step.selected_update_norm[1].item() == 0.0 for step in context.trajectory.steps[1:])

    # E2, E5, E6, and E7 contain exactly the four real updates: both subjects
    # at t=0, then only subject 0 at t=1 and t=2.  In particular, later
    # inactive sentinel rows must not create a candidate target or a zero
    # local/monotonic/delta contribution for subject 1.
    assert tuple(item.reward_prediction.shape[0] for item in result.reward_supervision) == (2, 1, 1)
    assert (result.local_step_count, result.monotonic_pair_count, result.delta_step_count) == (
        local_count,
        monotonic_count,
        delta_count,
    ) == (4, 2, 4)
    torch.testing.assert_close(result.local, expected_local)
    torch.testing.assert_close(result.monotonic, expected_monotonic)
    torch.testing.assert_close(result.delta, expected_delta)
