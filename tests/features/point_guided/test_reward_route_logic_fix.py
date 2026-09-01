"""Regression tests for the 2026-09 reward/route logic correction."""

from __future__ import annotations

import torch
from torch import nn

from smagm.features.point_guided import PointGuidedConfig, PointGuidedMRIModel
from smagm.features.point_guided.training_objective import SupervisionConfig
from smagm.features.point_guided.trajectory_cost import TrajectoryConfig, route_utility, travel_cost
from smagm.features.point_guided.trajectory_solver import AdaptiveRouteSolver


def _corrected_config(*, threshold: float = 0.25) -> TrajectoryConfig:
    return TrajectoryConfig(
        lambda_travel=0.05,
        lambda_overlap=0.2,
        lambda_step=threshold,
        k_max=3,
        selection_temperature=1.0,
        write_scale=0.1,
        bounded_travel_cost=True,
        separate_halt_from_utility=True,
        training_exploration_steps=0,
    )


def test_corrected_travel_is_bounded_and_step_threshold_is_not_a_ranking_penalty() -> None:
    points = torch.tensor([[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]]])
    travel = travel_cost(points, torch.tensor([0]), bounded=True)
    reward = torch.tensor([[0.3, 0.3, 0.3]])
    overlap = torch.zeros_like(reward)
    config = _corrected_config(threshold=0.25)

    torch.testing.assert_close(travel, torch.tensor([[0.0, 0.5, 2.0 / 3.0]]))
    torch.testing.assert_close(
        route_utility(reward, travel, overlap, config),
        reward - config.lambda_travel * travel,
    )
    assert bool((travel >= 0.0).all()) and bool((travel < 1.0).all())


def test_corrected_solver_can_continue_when_rank_utility_is_negative_but_gain_is_sufficient() -> None:
    utility = torch.tensor([[-0.8, -0.2, -0.5]])
    reward = torch.tensor([[0.4, 0.3, 0.35]])
    result = AdaptiveRouteSolver()(
        utility,
        torch.tensor([True]),
        training=False,
        temperature=1.0,
        halt_score=reward,
        halt_threshold=0.25,
    )

    assert result.active.tolist() == [True]
    assert result.indices.tolist() == [1]

    stopped = AdaptiveRouteSolver()(
        utility,
        torch.tensor([True]),
        training=False,
        temperature=1.0,
        halt_score=torch.full_like(reward, 0.2),
        halt_threshold=0.25,
    )
    assert stopped.active.tolist() == [False]
    assert stopped.indices.tolist() == [-1]


def test_terminal_state_reward_supervision_survives_a_k0_route_stop() -> None:
    torch.manual_seed(991)
    model = PointGuidedMRIModel(
        PointGuidedConfig(
            num_semantic_classes=3,
            num_points=3,
            point_candidate_multiplier=3,
            offset_hidden_channels=12,
        ),
        trajectory_config=_corrected_config(threshold=0.9),
    ).train()
    assert model.trajectory is not None
    first = model.trajectory.reward_net.network[0]
    last = model.trajectory.reward_net.network[2]
    assert isinstance(first, nn.Linear) and isinstance(last, nn.Linear)
    with torch.no_grad():
        first.weight.zero_()
        first.bias.zero_()
        last.weight.zero_()
        last.bias.fill_(-10.0)

    volume = torch.linspace(-1.0, 1.0, steps=3 * 7 * 7 * 7).reshape(1, 3, 7, 7, 7)
    context = model.forward_training_context(volume, chunk_size=17)
    assert context.trajectory.route_lengths.tolist() == [0]

    result = model.compute_training_objective(
        context,
        torch.zeros_like(context.reconstruction.prediction),
        config=SupervisionConfig(
            counterfactual_candidates=3,
            high_candidate_count=1,
            random_candidate_count=1,
            spill_sample_count=0,
            supervise_terminal_state=True,
        ),
        generator=torch.Generator().manual_seed(17),
    )

    assert len(result.reward_supervision) == 1
    assert result.reward_supervision[0].valid_count > 0
    assert bool(torch.isfinite(result.reward))
    assert bool(result.reward > 0.0)
