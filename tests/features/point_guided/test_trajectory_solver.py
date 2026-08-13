"""Focused C4 tests for adaptive hard and straight-through selection."""

from __future__ import annotations

import torch

from smagm.features.point_guided.trajectory_solver import AdaptiveRouteSolver


def test_inference_uses_hard_argmax_and_latches_only_nonrunning_or_nonpositive_rows() -> None:
    solver = AdaptiveRouteSolver()
    result = solver(
        torch.tensor([[0.1, 0.7, 0.3], [-0.1, 0.0, -0.3], [0.5, 0.2, 0.4]]),
        torch.tensor([True, True, False]),
        training=False,
        temperature=0.5,
    )

    assert result.indices.tolist() == [1, -1, -1]
    assert result.active.tolist() == [True, False, False]
    torch.testing.assert_close(result.weights, torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert sum(parameter.numel() for parameter in solver.parameters()) == 0


def test_training_straight_through_selection_is_hard_forward_and_soft_backward() -> None:
    utility = torch.tensor([[0.1, 0.7, 0.3]], requires_grad=True)
    result = AdaptiveRouteSolver()(utility, torch.tensor([True]), training=True, temperature=0.8)

    torch.testing.assert_close(result.weights.detach(), torch.tensor([[0.0, 1.0, 0.0]]))
    downstream = (result.weights * torch.tensor([[1.0, 2.0, 4.0]])).sum()
    downstream.backward()
    assert utility.grad is not None
    assert bool(utility.grad.abs().sum() > 0.0)


def test_gate_c_permits_a_previous_winner_to_repeat_when_it_remains_best() -> None:
    solver = AdaptiveRouteSolver()
    running = torch.tensor([True])
    first = solver(torch.tensor([[0.9, 0.8, 0.1]]), running, training=False, temperature=1.0)
    second = solver(torch.tensor([[0.95, 0.8, 0.1]]), running, training=False, temperature=1.0)

    assert first.indices.item() == 0
    assert second.indices.item() == 0
