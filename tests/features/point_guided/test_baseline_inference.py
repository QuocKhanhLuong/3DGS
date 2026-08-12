"""Gate-G G1--G4 deterministic software-policy tests on synthetic tensors."""

from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from smagm.features.point_guided.availability import ExactNoRevisitPolicy
from smagm.features.point_guided.baseline_inference import (
    GateGInferenceConfig,
    baseline_checkpoint_metadata,
    load_validated_baseline_checkpoint,
)
import smagm.features.point_guided.baseline_inference as baseline_inference_module
from smagm.features.point_guided.config import PointGuidedConfig
from smagm.features.point_guided.model import PointGuidedMRIModel
from smagm.features.point_guided.trajectory_cost import TrajectoryConfig, route_utility, travel_cost
from smagm.features.point_guided.trajectory_solver import AdaptiveRouteSolver


def _model(*, points: int = 3, k_max: int = 4) -> PointGuidedMRIModel:
    return PointGuidedMRIModel(
        PointGuidedConfig(
            num_semantic_classes=3,
            num_points=points,
            point_candidate_multiplier=3,
            offset_hidden_channels=12,
        ),
        trajectory_config=TrajectoryConfig(
            lambda_travel=0.05,
            lambda_overlap=0.20,
            lambda_step=0.05,
            k_max=k_max,
            selection_temperature=0.7,
            write_scale=0.1,
        ),
    )


def _constant_reward(model: PointGuidedMRIModel, value: float) -> None:
    assert model.trajectory is not None
    with torch.no_grad():
        first = model.trajectory.reward_net.network[0]
        last = model.trajectory.reward_net.network[2]
        assert isinstance(first, nn.Linear) and isinstance(last, nn.Linear)
        first.weight.zero_()
        first.bias.zero_()
        last.weight.zero_()
        last.bias.fill_(torch.logit(torch.tensor(value)).item())


def _run(model: PointGuidedMRIModel, *, k_max: int, x: torch.Tensor | None = None):
    return model.forward_baseline_inference(
        torch.randn(1, 3, 7, 7, 7) if x is None else x,
        inference_config=GateGInferenceConfig(k_max=k_max, decoder_chunk_size=29),
    )


def test_gate_g_config_is_hard_only_and_uses_plan_initial_values() -> None:
    config = GateGInferenceConfig()
    assert (config.k_max, config.lambda_travel, config.lambda_overlap, config.lambda_step) == (64, 0.05, 0.20, 0.05)
    assert "temperature" not in inspect.signature(GateGInferenceConfig).parameters
    with pytest.raises(ValueError, match="positive"):
        GateGInferenceConfig(k_max=0)
    with pytest.raises(ValueError, match="positive and finite"):
        GateGInferenceConfig(lambda_step=0.0)


def test_gate_c_solver_can_revisit_while_gate_g_policy_cannot() -> None:
    solver = AdaptiveRouteSolver()
    utility = torch.tensor([[0.9, 0.8]])
    running = torch.tensor([True])
    assert solver(utility, running, training=False, temperature=1.0).indices.tolist() == [0]
    assert solver(utility, running, training=False, temperature=1.0).indices.tolist() == [0]

    policy = ExactNoRevisitPolicy()
    available = policy.initial_available(batch=1, point_count=2, device=utility.device)
    first = solver(policy.mask_utility(utility, available), running, training=False, temperature=1.0)
    available = policy.update_available(available, first.indices, first.active)
    second = solver(policy.mask_utility(utility, available), running, training=False, temperature=1.0)
    assert first.indices.tolist() == [0]
    assert second.indices.tolist() == [1]


def test_gate_g_first_step_utility_is_raw_reward_minus_step_and_costs_change_order() -> None:
    points = torch.tensor([[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]])
    first_travel = travel_cost(points, torch.tensor([-1]))
    first_overlap = torch.zeros_like(first_travel)
    reward = torch.tensor([[0.90, 0.94]])
    low = TrajectoryConfig(0.01, 0.20, 0.05, 2, 1.0, 0.1)
    high = TrajectoryConfig(0.20, 0.20, 0.05, 2, 1.0, 0.1)
    torch.testing.assert_close(first_travel, torch.zeros_like(first_travel))
    torch.testing.assert_close(first_overlap, torch.zeros_like(first_overlap))
    low_utility = route_utility(reward, torch.tensor([[0.0, 1.0]]), first_overlap, low)
    high_utility = route_utility(reward, torch.tensor([[0.0, 1.0]]), first_overlap, high)
    torch.testing.assert_close(route_utility(reward, first_travel, first_overlap, low), reward - low.lambda_step)
    assert low_utility.argmax(dim=1).tolist() == [1]
    assert high_utility.argmax(dim=1).tolist() == [0]
    torch.testing.assert_close(reward, torch.tensor([[0.90, 0.94]]))


def test_gate_g_is_deterministic_target_free_parameter_immutable_and_decodes_once() -> None:
    torch.manual_seed(71)
    model = _model(k_max=2)
    _constant_reward(model, 0.95)
    x = torch.randn(1, 3, 7, 7, 7)
    parameter_before = {name: value.detach().clone() for name, value in model.named_parameters()}
    assert model.trajectory is not None and model.decoder is not None
    backbone = model.semantic_prior.backbone
    names = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4")
    calls = {name: 0 for name in (*names, "projector", "anchor", "query", "consistency", "decoder", "updater")}
    hooks = [
        getattr(backbone, name).register_forward_hook(
            lambda _module, _inputs, _output, name=name: calls.__setitem__(name, calls[name] + 1)
        )
        for name in names
    ]
    for name, module in (
        ("projector", model.base_plane_projector),
        ("anchor", model.spectral_anchor_builder),
        ("query", model.spectral_point_query),
        ("consistency", model.cross_plane_consistency),
        ("decoder", model.decoder),
        ("updater", model.trajectory.update_net),
    ):
        hooks.append(module.register_forward_hook(lambda _module, _inputs, _output, name=name: calls.__setitem__(name, calls[name] + 1)))
    update_widths: list[int] = []
    hooks.append(model.trajectory.update_net.register_forward_pre_hook(lambda _module, args: update_widths.append(args[0].shape[-1])))
    try:
        first = _run(model, k_max=2, x=x)
        second = _run(model, k_max=2, x=x)
    finally:
        for hook in hooks:
            hook.remove()
    assert first.prediction.shape == (1, 1, 7, 7, 7)
    assert first.selected_indices.tolist() == [[0, 1]]  # native argmax: first maximum index
    assert first.k_used.tolist() == [2]
    assert first.stop_reasons == ("k_max",)
    assert first.candidate_evaluations.tolist() == [6]
    assert first.eligible_candidate_evaluations.tolist() == [5]
    assert torch.unique(first.selected_indices[first.selected_indices >= 0]).numel() == 2
    torch.testing.assert_close(first.prediction, second.prediction)
    assert torch.equal(first.selected_indices, second.selected_indices)
    assert first.stop_reasons == second.stop_reasons
    assert all(torch.equal(before, dict(model.named_parameters())[name]) for name, before in parameter_before.items())
    assert calls["decoder"] == 2  # exactly once per complete Gate-G inference
    assert all(calls[name] == 2 for name in (*names, "projector", "anchor", "query", "consistency"))
    assert calls["updater"] == 4
    assert update_widths == [270, 270, 270, 270]
    assert model.trajectory.writeback.support_radius_mm == 4.0
    assert "target_t1ce" not in inspect.signature(model.forward_baseline_inference).parameters
    assert "baseline_training" not in inspect.getsource(baseline_inference_module)
    assert not tuple(parameter for name, parameter in model.named_parameters() if "baseline_inference" in name)


def test_gate_g_recomputes_current_state_then_stops_for_nonpositive_or_exhaustion() -> None:
    model = _model(points=3, k_max=4)
    assert model.trajectory is not None

    class ScheduledReward(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inputs: list[torch.Tensor] = []

        def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
            self.inputs.append(descriptor.detach().clone())
            values = ([0.95, 0.80, 0.70], [0.10, 0.95, 0.20])
            return torch.tensor(values[min(len(self.inputs) - 1, 1)], dtype=descriptor.dtype, device=descriptor.device).expand(descriptor.shape[0], -1)

    scheduled = ScheduledReward()
    model.trajectory.reward_net = scheduled
    result = _run(model, k_max=2)
    assert result.selected_indices.tolist() == [[0, 1]]
    assert len(scheduled.inputs) == 2
    assert not torch.equal(scheduled.inputs[0], scheduled.inputs[1])  # Z query was recomputed after write-back.

    nonpositive = _model(points=3, k_max=4)
    _constant_reward(nonpositive, 1e-4)
    stopped = _run(nonpositive, k_max=4)
    assert stopped.k_used.tolist() == [0]
    assert stopped.stop_reasons == ("nonpositive_utility",)
    assert stopped.candidate_evaluations.tolist() == [3]
    assert stopped.eligible_candidate_evaluations.tolist() == [3]

    exhausted = _model(points=2, k_max=4)
    _constant_reward(exhausted, 0.95)
    exhausted_result = _run(exhausted, k_max=4)
    assert exhausted_result.selected_indices.tolist() == [[0, 1]]
    assert exhausted_result.stop_reasons == ("candidates_exhausted",)
    assert exhausted_result.candidate_evaluations.tolist() == [4]
    assert exhausted_result.eligible_candidate_evaluations.tolist() == [3]


def test_gate_g_latches_each_batch_subject_and_strict_checkpoint_loader(tmp_path) -> None:
    model = _model(points=3, k_max=4)
    assert model.trajectory is not None

    class BatchScheduledReward(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes: list[int] = []

        def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
            self.batch_sizes.append(descriptor.shape[0])
            if len(self.batch_sizes) == 1:
                return torch.stack((
                    torch.full((descriptor.shape[1],), 0.95, dtype=descriptor.dtype, device=descriptor.device),
                    torch.zeros(descriptor.shape[1], dtype=descriptor.dtype, device=descriptor.device),
                ))
            return torch.full(descriptor.shape[:2], 0.95, dtype=descriptor.dtype, device=descriptor.device)

    scheduled = BatchScheduledReward()
    model.trajectory.reward_net = scheduled
    result = model.forward_baseline_inference(
        torch.randn(2, 3, 7, 7, 7),
        inference_config=GateGInferenceConfig(k_max=4, decoder_chunk_size=29),
    )
    assert scheduled.batch_sizes == [2, 1, 1]
    assert result.k_used.tolist() == [3, 0]
    assert result.selected_indices[1].tolist() == [-1, -1, -1]
    assert result.stop_reasons == ("candidates_exhausted", "nonpositive_utility")
    assert result.candidate_evaluations.tolist() == [9, 3]
    assert result.eligible_candidate_evaluations.tolist() == [6, 3]

    checkpoint_model = _model()
    path = tmp_path / "synthetic_gate_f_state.pt"
    torch.save({"metadata": baseline_checkpoint_metadata(checkpoint_model), "state_dict": checkpoint_model.state_dict()}, path)
    clone = _model()
    load_validated_baseline_checkpoint(clone, path)
    assert all(torch.equal(value, clone.state_dict()[name]) for name, value in checkpoint_model.state_dict().items())
    invalid = tmp_path / "mismatch.pt"
    metadata = dict(baseline_checkpoint_metadata(checkpoint_model))
    metadata["decoder_architecture"] = "wrong"
    torch.save({"metadata": metadata, "state_dict": checkpoint_model.state_dict()}, invalid)
    with pytest.raises(ValueError, match="metadata"):
        load_validated_baseline_checkpoint(clone, invalid)
