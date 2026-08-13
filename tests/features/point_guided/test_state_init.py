"""Focused C1 invariants for dynamic tri-plane initialization."""

from __future__ import annotations

import pytest
import torch

from smagm.features.point_guided.state_init import DynamicStateInitializer
from smagm.features.point_guided.triplane_projection import BaseTriPlanes


def _base_planes(*, requires_grad: bool = False) -> BaseTriPlanes:
    return BaseTriPlanes(
        xy=torch.randn(2, 64, 5, 7, requires_grad=requires_grad),
        xz=torch.randn(2, 64, 3, 7, requires_grad=requires_grad),
        yz=torch.randn(2, 64, 3, 5, requires_grad=requires_grad),
    )


def test_shared_initializer_retains_every_static_plane_grid_without_mutating_b() -> None:
    initializer = DynamicStateInitializer()
    base_planes = _base_planes()
    before = tuple(getattr(base_planes, name).clone() for name in ("xy", "xz", "yz"))

    state = initializer(base_planes)

    assert state.xy.shape == (2, 32, 5, 7)
    assert state.xz.shape == (2, 32, 3, 7)
    assert state.yz.shape == (2, 32, 3, 5)
    assert initializer.shared_projection.in_channels == 64
    assert initializer.shared_projection.out_channels == 32
    assert initializer.shared_projection.kernel_size == (1, 1)
    assert initializer.shared_projection.bias is not None
    assert sum(parameter.numel() for parameter in initializer.parameters()) == 32 * 64 + 32
    for actual, expected in zip((base_planes.xy, base_planes.xz, base_planes.yz), before):
        assert torch.equal(actual, expected)


def test_state_loss_reaches_the_one_shared_initializer_and_base_input() -> None:
    initializer = DynamicStateInitializer()
    base_planes = _base_planes(requires_grad=True)
    state = initializer(base_planes)
    (state.xy.square().mean() + state.xz.square().mean() + state.yz.square().mean()).backward()

    assert initializer.shared_projection.weight.grad is not None
    assert initializer.shared_projection.bias.grad is not None
    assert base_planes.xy.grad is not None
    assert base_planes.xz.grad is not None
    assert base_planes.yz.grad is not None


@pytest.mark.parametrize("kwargs", ({"input_channels": 32}, {"state_channels": 16}))
def test_state_initializer_rejects_non_main_channel_contract(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="exactly"):
        DynamicStateInitializer(**kwargs)
