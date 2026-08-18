"""Focused C5 tests for the shared 270-d local updater."""

from __future__ import annotations

import pytest
import torch

from smagm.features.point_guided.updater import (
    UPDATER_INPUT_CHANNELS,
    UPDATER_OUTPUT_CHANNELS,
    UpdateNet,
)


def test_updater_uses_full_270_d_input_and_returns_three_bounded_32_d_blocks() -> None:
    updater = UpdateNet()
    values = torch.randn(3, UPDATER_INPUT_CHANNELS, requires_grad=True)
    correction = updater(values, write_scale=0.25)

    assert correction.xy.shape == correction.xz.shape == correction.yz.shape == (3, 32)
    assert correction.packed.shape == (3, UPDATER_OUTPUT_CHANNELS)
    assert bool((correction.packed.abs() <= 0.25 + 1e-6).all())
    assert sum(parameter.numel() for parameter in updater.parameters()) == 47_072
    correction.packed.square().mean().backward()
    assert values.grad is not None and bool(values.grad.abs().sum() > 0.0)
    assert all(parameter.grad is not None for parameter in updater.parameters())


@pytest.mark.parametrize("shape", ((2, 269), (2, 270, 1)))
def test_updater_fails_closed_for_non_270_input(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match=r"\[B,270\]"):
        UpdateNet()(torch.randn(*shape), write_scale=0.1)
