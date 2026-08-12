"""Focused C3 tests for explicit physical routing costs and utility."""

from __future__ import annotations

import pytest
import torch

from smagm.features.point_guided.reward import REWARD_DESCRIPTOR_CHANNELS, RewardNet
from smagm.features.point_guided.trajectory_cost import (
    TrajectoryConfig,
    overlap_cost,
    route_utility,
    travel_cost,
)
from smagm.features.point_guided.trajectory_solver import AdaptiveRouteSolver


def _config() -> TrajectoryConfig:
    return TrajectoryConfig(
        lambda_travel=0.25,
        lambda_overlap=0.5,
        lambda_step=0.1,
        k_max=3,
        selection_temperature=0.7,
        write_scale=0.2,
    )


def test_travel_overlap_and_utility_follow_the_locked_physical_formulas() -> None:
    points = torch.tensor([[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]]])
    first = travel_cost(points, torch.tensor([-1]))
    travel = travel_cost(points, torch.tensor([0]))
    overlap = overlap_cost(points, points[:, :1])
    reward = torch.tensor([[0.8, 0.8, 0.8]])
    utility = route_utility(reward, travel, overlap, _config())

    torch.testing.assert_close(first, torch.zeros_like(first))
    torch.testing.assert_close(travel, torch.tensor([[0.0, 1.0, 2.0]]))
    torch.testing.assert_close(overlap, torch.tensor([[1.0, 0.25, 0.0]]))
    torch.testing.assert_close(utility, reward - 0.25 * travel - 0.5 * overlap - 0.1)
    torch.testing.assert_close(reward, torch.full_like(reward, 0.8))


def test_overlap_changes_gate_c_selection_economically_without_changing_raw_reward() -> None:
    reward_net = RewardNet()
    with torch.no_grad():
        for parameter in reward_net.parameters():
            parameter.zero_()
    descriptor = torch.randn(1, 2, REWARD_DESCRIPTOR_CHANNELS)
    reward = reward_net(descriptor)
    travel = torch.zeros_like(reward)
    low_overlap = torch.zeros_like(reward)
    high_overlap_at_previous_winner = torch.tensor([[1.0, 0.0]])
    solver = AdaptiveRouteSolver()

    low_cost_choice = solver(
        route_utility(reward, travel, low_overlap, _config()),
        torch.tensor([True]),
        training=False,
        temperature=1.0,
    )
    high_cost_choice = solver(
        route_utility(reward, travel, high_overlap_at_previous_winner, _config()),
        torch.tensor([True]),
        training=False,
        temperature=1.0,
    )

    assert low_cost_choice.indices.tolist() == [0]
    assert high_cost_choice.indices.tolist() == [1]
    torch.testing.assert_close(reward, reward_net(descriptor))


@pytest.mark.parametrize(
    "kwargs",
    (
        {"lambda_travel": 0.0},
        {"lambda_overlap": float("nan")},
        {"lambda_step": -1.0},
        {"k_max": 0},
        {"selection_temperature": float("inf")},
        {"write_scale": 0.0},
        {"support_radius_mm": 3.0},
    ),
)
def test_trajectory_config_fails_closed_for_invalid_or_non_main_values(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "lambda_travel": 0.25,
        "lambda_overlap": 0.5,
        "lambda_step": 0.1,
        "k_max": 3,
        "selection_temperature": 0.7,
        "write_scale": 0.2,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match="positive|exactly"):
        TrajectoryConfig(**values)  # type: ignore[arg-type]
