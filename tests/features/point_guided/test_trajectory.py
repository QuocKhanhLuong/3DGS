"""Focused C7 integration tests for adaptive reward-cost trajectory composition."""

from __future__ import annotations

import torch
from torch import nn

from smagm.features.point_guided import PointGuidedConfig, PointGuidedMRIModel
from smagm.features.point_guided.contracts import VolumeGeometry
from smagm.features.point_guided.reward import GateBDescriptorContext
from smagm.features.point_guided.spectral_query import FeatureGridGeometry
from smagm.features.point_guided.trajectory import AdaptiveRewardCostTrajectory
from smagm.features.point_guided.trajectory_cost import TrajectoryConfig
from smagm.features.point_guided.triplane_projection import BaseTriPlanes


def _geometry() -> FeatureGridGeometry:
    source = VolumeGeometry.from_spacing((3, 5, 7), (1.0, 1.0, 1.0))
    return FeatureGridGeometry(
        source_geometry=source,
        feature_geometry=source,
        tap="conv1_pre_maxpool",
        feature_to_source_scale_dhw=(1.0, 1.0, 1.0),
        feature_to_source_offset_dhw=(0.0, 0.0, 0.0),
        operator_chain=("synthetic",),
    )


def _inputs() -> tuple[BaseTriPlanes, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, GateBDescriptorContext]:
    base = BaseTriPlanes(
        xy=torch.zeros(1, 64, 5, 7),
        xz=torch.zeros(1, 64, 3, 7),
        yz=torch.zeros(1, 64, 3, 5),
    )
    geometry = _geometry()
    points = geometry.feature_dhw_to_ras_mm(torch.tensor([[[1.0, 2.0, 3.0], [1.0, 3.0, 4.0], [2.0, 2.0, 5.0]]]))
    semantic = torch.tensor([[[0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.6, 0.2, 0.2]]])
    f_spec = torch.randn(1, 3, 168)
    reliability = torch.full((1, 3, 3), 1.0 / 3.0)
    descriptors = GateBDescriptorContext(
        q_xy=torch.randn(1, 3, 24),
        q_xz=torch.randn(1, 3, 24),
        q_yz=torch.randn(1, 3, 24),
    )
    return base, points, semantic, f_spec, reliability, descriptors


def _trajectory(k_max: int) -> AdaptiveRewardCostTrajectory:
    trajectory = AdaptiveRewardCostTrajectory(
        TrajectoryConfig(
            lambda_travel=0.01,
            lambda_overlap=0.01,
            lambda_step=0.01,
            k_max=k_max,
            selection_temperature=0.7,
            write_scale=0.2,
        )
    ).eval()
    with torch.no_grad():
        trajectory.reward_net.network[0].weight.zero_()
        trajectory.reward_net.network[0].bias.zero_()
        trajectory.reward_net.network[2].weight.zero_()
        trajectory.reward_net.network[2].bias.fill_(3.0)
    return trajectory


def test_trajectory_recomputes_reward_and_returns_only_final_dynamic_state_and_compact_diagnostics() -> None:
    base, points, semantic, f_spec, reliability, descriptors = _inputs()
    geometry = _geometry()
    before_base = tuple(getattr(base, name).clone() for name in ("xy", "xz", "yz"))
    before_f_spec = f_spec.clone()
    before_reliability = reliability.clone()
    before_q = tuple(getattr(descriptors, name).clone() for name in ("q_xy", "q_xz", "q_yz"))
    trajectory = _trajectory(k_max=2)
    reward_calls = [0]
    hook = trajectory.reward_net.register_forward_hook(lambda *_: reward_calls.__setitem__(0, reward_calls[0] + 1))
    try:
        result = trajectory(base, points, semantic, f_spec, reliability, descriptors, geometry, geometry.source_geometry)
    finally:
        hook.remove()

    assert result.final_state.xy.shape == (1, 32, 5, 7)
    assert result.final_state.xz.shape == (1, 32, 3, 7)
    assert result.final_state.yz.shape == (1, 32, 3, 5)
    assert result.selected_indices.shape == (1, 2)
    assert result.route_lengths.tolist() == [2]
    assert result.selected_indices[0, 0].item() == result.selected_indices[0, 1].item()
    assert reward_calls == [2]
    assert result.stop_reasons == ("k_max",)
    assert bool(result.final_state.xy.abs().sum() > 0.0)
    for actual, expected in zip((base.xy, base.xz, base.yz), before_base):
        assert torch.equal(actual, expected)
    assert torch.equal(f_spec, before_f_spec)
    assert torch.equal(reliability, before_reliability)
    for actual, expected in zip((descriptors.q_xy, descriptors.q_xz, descriptors.q_yz), before_q):
        assert torch.equal(actual, expected)


def test_trajectory_stops_before_update_when_all_utilities_are_nonpositive() -> None:
    base, points, semantic, f_spec, reliability, descriptors = _inputs()
    geometry = _geometry()
    trajectory = AdaptiveRewardCostTrajectory(
        TrajectoryConfig(
            lambda_travel=1.0,
            lambda_overlap=1.0,
            lambda_step=2.0,
            k_max=3,
            selection_temperature=0.7,
            write_scale=0.2,
        )
    ).eval()
    result = trajectory(base, points, semantic, f_spec, reliability, descriptors, geometry, geometry.source_geometry)

    assert result.steps == ()
    assert result.route_lengths.tolist() == [0]
    assert result.stop_reasons == ("nonpositive_utility",)


def test_model_trajectory_reuses_one_gate_b_pass_and_keeps_full_forward_closed() -> None:
    trajectory_config = TrajectoryConfig(
        lambda_travel=0.01,
        lambda_overlap=0.01,
        lambda_step=0.01,
        k_max=2,
        selection_temperature=0.7,
        write_scale=0.2,
    )
    model = PointGuidedMRIModel(
        PointGuidedConfig(
            num_semantic_classes=3,
            num_points=3,
            point_candidate_multiplier=3,
            offset_hidden_channels=12,
        ),
        trajectory_config=trajectory_config,
    ).eval()
    assert model.trajectory is not None
    with torch.no_grad():
        first = model.trajectory.reward_net.network[0]
        last = model.trajectory.reward_net.network[2]
        assert isinstance(first, torch.nn.Linear) and isinstance(last, torch.nn.Linear)
        first.weight.zero_()
        first.bias.zero_()
        last.weight.zero_()
        last.bias.fill_(3.0)

    backbone = model.semantic_prior.backbone
    backbone_names = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4")
    calls = {name: 0 for name in (*backbone_names, "projector", "anchor", "query", "consistency")}
    hooks = [
        getattr(backbone, name).register_forward_hook(
            lambda _module, _inputs, _output, name=name: calls.__setitem__(name, calls[name] + 1)
        )
        for name in backbone_names
    ]
    for name, module in (
        ("projector", model.base_plane_projector),
        ("anchor", model.spectral_anchor_builder),
        ("query", model.spectral_point_query),
        ("consistency", model.cross_plane_consistency),
    ):
        hooks.append(
            module.register_forward_hook(
                lambda _module, _inputs, _output, name=name: calls.__setitem__(name, calls[name] + 1)
            )
        )
    try:
        result = model.forward_trajectory(torch.randn(1, 3, 7, 7, 7))
    finally:
        for hook in hooks:
            hook.remove()

    assert all(count == 1 for count in calls.values())
    assert result.frontend.f_spec.shape == (1, 3, 168)
    assert result.frontend.reliability.shape == (1, 3, 3)
    assert result.trajectory.final_state.xy.shape == (1, 32, 4, 4)
    assert result.trajectory.route_lengths.tolist() == [2]
    assert result.trajectory.stop_reasons == ("k_max",)
    with torch.no_grad():
        try:
            model(torch.randn(1, 3, 7, 7, 7))
        except NotImplementedError:
            pass
        else:  # pragma: no cover - explicit public fail-closed boundary
            raise AssertionError("full model forward must remain unavailable")


def test_existing_phase1_to_phase7_model_requires_explicit_gate_c_runtime_config() -> None:
    model = PointGuidedMRIModel(PointGuidedConfig(num_semantic_classes=3, num_points=3, point_candidate_multiplier=3))
    try:
        model.forward_trajectory(torch.randn(1, 3, 7, 7, 7))
    except RuntimeError as error:
        assert "explicit TrajectoryConfig" in str(error)
    else:  # pragma: no cover - fail-closed public API contract
        raise AssertionError("trajectory execution must be opt-in for the preserved Phase-1--7 constructor")


def test_final_dynamic_state_has_the_required_trainable_gate_c_paths_but_not_frozen_medicalnet() -> None:
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
            k_max=1,
            selection_temperature=0.7,
            write_scale=0.2,
        ),
    ).train()
    assert model.trajectory is not None

    result = model.forward_trajectory(torch.randn(1, 3, 7, 7, 7))
    loss = sum(plane.square().mean() for plane in (result.trajectory.final_state.xy, result.trajectory.final_state.xz, result.trajectory.final_state.yz))
    loss.backward()

    def has_finite_nonzero_gradient(parameters: object) -> bool:
        return any(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) and bool(parameter.grad.abs().sum() > 0.0)
            for parameter in parameters  # type: ignore[union-attr]
        )

    assert has_finite_nonzero_gradient(model.trajectory.state_initializer.parameters())
    assert has_finite_nonzero_gradient(model.trajectory.update_net.parameters())
    assert has_finite_nonzero_gradient(model.trajectory.reward_net.parameters())
    assert has_finite_nonzero_gradient(model.spectral_anchor_builder.band_projector.parameters())
    assert has_finite_nonzero_gradient(model.base_plane_projector.parameters())
    assert all(parameter.grad is None for parameter in model.semantic_prior.backbone.parameters())


def test_trajectory_supports_per_subject_stopping_with_compact_inactive_diagnostics() -> None:
    base, points, semantic, f_spec, reliability, descriptors = _inputs()
    base = BaseTriPlanes(
        xy=base.xy.expand(2, -1, -1, -1).clone(),
        xz=base.xz.expand(2, -1, -1, -1).clone(),
        yz=base.yz.expand(2, -1, -1, -1).clone(),
    )
    points = points.expand(2, -1, -1).clone()
    semantic = semantic.expand(2, -1, -1).clone()
    semantic[0, :, 0] = 1.0
    semantic[1, :, 0] = 0.0
    f_spec = f_spec.expand(2, -1, -1).clone()
    reliability = reliability.expand(2, -1, -1).clone()
    descriptors = GateBDescriptorContext(
        q_xy=descriptors.q_xy.expand(2, -1, -1).clone(),
        q_xz=descriptors.q_xz.expand(2, -1, -1).clone(),
        q_yz=descriptors.q_yz.expand(2, -1, -1).clone(),
    )
    trajectory = AdaptiveRewardCostTrajectory(
        TrajectoryConfig(
            lambda_travel=0.01,
            lambda_overlap=0.01,
            lambda_step=0.5,
            k_max=1,
            selection_temperature=0.7,
            write_scale=0.2,
        )
    ).eval()
    with torch.no_grad():
        first = trajectory.reward_net.network[0]
        last = trajectory.reward_net.network[2]
        assert isinstance(first, torch.nn.Linear) and isinstance(last, torch.nn.Linear)
        first.weight.zero_()
        first.bias.zero_()
        first.weight[0, 96] = 10.0
        first.bias[0] = -5.0
        last.weight.zero_()
        last.weight[0, 0] = 10.0
        last.bias.fill_(-5.0)

    geometry = _geometry()
    result = trajectory(base, points, semantic, f_spec, reliability, descriptors, geometry, geometry.source_geometry)

    assert result.route_lengths.tolist() == [1, 0]
    assert result.selected_indices[0, 0].item() >= 0
    assert result.selected_indices[1, 0].item() == -1
    assert result.stop_reasons == ("k_max", "nonpositive_utility")


class _ScheduledReward(nn.Module):
    """Synthetic C7 probe: B becomes nonpositive after its first local write."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(descriptor.shape[0])
        if len(self.batch_sizes) == 1:
            return torch.full(descriptor.shape[:2], 0.9, dtype=descriptor.dtype, device=descriptor.device)
        if len(self.batch_sizes) == 2:
            return torch.stack(
                (
                    torch.full((descriptor.shape[1],), 0.9, dtype=descriptor.dtype, device=descriptor.device),
                    torch.zeros(descriptor.shape[1], dtype=descriptor.dtype, device=descriptor.device),
                )
            )
        assert descriptor.shape[0] == 1
        return torch.full(descriptor.shape[:2], 0.9, dtype=descriptor.dtype, device=descriptor.device)


def test_per_subject_nonpositive_stop_latches_across_later_steps_without_mutating_stopped_z() -> None:
    base, points, semantic, f_spec, reliability, descriptors = _inputs()
    base = BaseTriPlanes(
        xy=base.xy.expand(2, -1, -1, -1).clone(),
        xz=base.xz.expand(2, -1, -1, -1).clone(),
        yz=base.yz.expand(2, -1, -1, -1).clone(),
    )
    points = points.expand(2, -1, -1).clone()
    semantic = semantic.expand(2, -1, -1).clone()
    f_spec = f_spec.expand(2, -1, -1).clone()
    reliability = reliability.expand(2, -1, -1).clone()
    descriptors = GateBDescriptorContext(
        q_xy=descriptors.q_xy.expand(2, -1, -1).clone(),
        q_xz=descriptors.q_xz.expand(2, -1, -1).clone(),
        q_yz=descriptors.q_yz.expand(2, -1, -1).clone(),
    )
    trajectory = AdaptiveRewardCostTrajectory(
        TrajectoryConfig(
            lambda_travel=0.01,
            lambda_overlap=0.01,
            lambda_step=0.1,
            k_max=3,
            selection_temperature=0.7,
            write_scale=0.2,
        )
    ).eval()
    scheduled_reward = _ScheduledReward()
    trajectory.reward_net = scheduled_reward
    write_states: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    hook = trajectory.writeback.register_forward_hook(
        lambda _module, _inputs, output: write_states.append(
            (output.xy.detach().clone(), output.xz.detach().clone(), output.yz.detach().clone())
        )
    )
    try:
        geometry = _geometry()
        result = trajectory(base, points, semantic, f_spec, reliability, descriptors, geometry, geometry.source_geometry)
    finally:
        hook.remove()

    assert scheduled_reward.batch_sizes == [2, 2, 1]
    assert result.selected_indices.shape == (2, 3)
    assert result.route_lengths.tolist() == [3, 1]
    assert result.selected_indices[1].tolist() == [0, -1, -1]
    assert result.stop_reasons == ("k_max", "nonpositive_utility")
    assert all(step.selected_indices[1].item() == -1 for step in result.steps[1:])
    assert all(step.selected_update_norm[1].item() == 0.0 for step in result.steps[1:])
    assert len(write_states) == 3
    assert [state[0].shape[0] for state in write_states] == [2, 1, 1]
    for plane_index, final_plane in enumerate((result.final_state.xy, result.final_state.xz, result.final_state.yz)):
        assert torch.equal(write_states[0][plane_index][1], final_plane[1])
