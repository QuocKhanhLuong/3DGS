"""Gate-F F1/F2 ownership, no-revisit overlay, and synthetic smoke tests."""

from __future__ import annotations

import pytest
import torch

from smagm.features.point_guided.baseline_training import (
    BaselineNoRevisitPolicy,
    BaselineTrainingConfig,
    build_baseline_optimizer,
    resolve_parameter_ownership,
    run_synthetic_smoke,
)
from smagm.features.point_guided.config import PointGuidedConfig
from smagm.features.point_guided.model import PointGuidedMRIModel
from smagm.features.point_guided.training_objective import SupervisionConfig
from smagm.features.point_guided.trajectory_cost import TrajectoryConfig
from smagm.features.point_guided.trajectory_solver import AdaptiveRouteSolver


def _model() -> PointGuidedMRIModel:
    return PointGuidedMRIModel(
        PointGuidedConfig(
            num_semantic_classes=3,
            num_points=3,
            point_candidate_multiplier=3,
            offset_hidden_channels=12,
        ),
        trajectory_config=TrajectoryConfig(
            lambda_travel=0.05,
            lambda_overlap=0.2,
            lambda_step=0.05,
            k_max=2,
            selection_temperature=0.7,
            write_scale=0.1,
        ),
    )


def _supervision() -> SupervisionConfig:
    return SupervisionConfig(
        counterfactual_candidates=3,
        high_candidate_count=1,
        random_candidate_count=1,
        spill_sample_count=3,
    )


def test_gate_f_resolves_the_exact_optimizer_set_including_offset_predictor() -> None:
    model = _model()
    optimizer, ownership = build_baseline_optimizer(model, BaselineTrainingConfig())
    rows = {row.module: row for row in ownership}

    expected_trainable = {
        "semantic_head": 1539,
        "point_refiner.offset_predictor": 1419,
        "base_plane_projector": 579,
        "spectral_anchor_builder.band_projector": 520,
        "trajectory.state_initializer": 2080,
        "trajectory.reward_net": 14337,
        "trajectory.update_net": 47072,
        "decoder": 8321,
    }
    assert {name: rows[name].parameter_count for name in expected_trainable} == expected_trainable
    assert all(rows[name].requires_grad and rows[name].optimizer_member for name in expected_trainable)
    assert rows["semantic_prior.backbone"].parameter_count == 14_399_424
    assert not rows["semantic_prior.backbone"].requires_grad
    assert not rows["semantic_prior.backbone"].optimizer_member
    assert all(not parameter.requires_grad for parameter in model.semantic_prior.backbone.parameters())
    assert not tuple(model.spectral_anchor_builder.swt.parameters())
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert len(optimizer_ids) == sum(1 for _ in model.parameters() if _.requires_grad)
    assert all(id(parameter) in optimizer_ids for parameter in model.point_refiner.offset_predictor.parameters())
    assert all(id(parameter) not in optimizer_ids for parameter in model.semantic_prior.backbone.parameters())
    assert resolve_parameter_ownership(model, optimizer) == ownership


def test_gate_f_no_revisit_overlay_is_separate_from_gate_c_solver() -> None:
    solver = AdaptiveRouteSolver()
    utility = torch.tensor([[0.9, 0.8]], dtype=torch.float32)
    running = torch.tensor([True])
    core_first = solver(utility, running, training=False, temperature=1.0)
    core_second = solver(utility, running, training=False, temperature=1.0)
    assert core_first.indices.tolist() == [0]
    assert core_second.indices.tolist() == [0]  # Gate-C primitive may revisit.

    policy = BaselineNoRevisitPolicy()
    availability = policy.initial_available(batch=1, point_count=2, device=utility.device)
    baseline_first = solver(policy.mask_utility(utility, availability), running, training=False, temperature=1.0)
    availability = policy.update_available(availability, baseline_first.indices, baseline_first.active)
    baseline_second = solver(policy.mask_utility(utility, availability), running, training=False, temperature=1.0)
    assert baseline_first.indices.tolist() == [0]
    assert baseline_second.indices.tolist() == [1]


def test_gate_f_synthetic_smoke_updates_all_authorized_components_and_preserves_bounds() -> None:
    torch.manual_seed(51)
    model = _model()
    observations = torch.randn(1, 3, 7, 7, 7)
    target = torch.sigmoid(torch.randn(1, 1, 7, 7, 7))
    result = run_synthetic_smoke(
        model,
        observations,
        target,
        training_config=BaselineTrainingConfig(decoder_chunk_size=29, seed=51),
        supervision_config=_supervision(),
    )
    expected = (
        "semantic_head",
        "point_refiner.offset_predictor",
        "base_plane_projector",
        "spectral_anchor_builder.band_projector",
        "trajectory.state_initializer",
        "trajectory.reward_net",
        "trajectory.update_net",
        "decoder",
    )
    assert result.prediction_shape == (1, 1, 7, 7, 7)
    assert result.total_loss >= 0.0
    assert result.gradient_modules == expected
    assert result.changed_modules == expected
    assert result.max_displacement_mm <= 2.0 + 1e-6
    for route in result.selected_indices:
        selected = route[route >= 0]
        assert torch.unique(selected).numel() == selected.numel()


def test_gate_f_engineering_config_rejects_an_implicit_optimizer_change() -> None:
    with pytest.raises(ValueError, match="optimizer_name"):
        BaselineTrainingConfig(optimizer_name="adamw")  # type: ignore[arg-type]
