from __future__ import annotations

from dataclasses import replace

import torch

from smagm.features.point_guided.pfgr_lite.sparse_write import reference_full_write
from tests.features.point_guided.pfgr_lite.test_teacher import _fixture


def test_parallel_stored_actions_commute_in_linear_write_but_not_loss_claim() -> None:
    _, _, lattice, state, action, _, decoder = _fixture(torch.float64)
    second = replace(
        action,
        action_id="parallel-second",
        point_ras_mm=torch.tensor((1.0, 1.0, 1.0), dtype=torch.float64),
        delta=action.delta * 0.25,
        action_digest=None,
    )
    first_then_second = reference_full_write(lattice, state.planes, action)
    first_then_second = reference_full_write(lattice, first_then_second, second)
    second_then_first = reference_full_write(lattice, state.planes, second)
    second_then_first = reference_full_write(lattice, second_then_first, action)
    assert torch.allclose(first_then_second.xy, second_then_first.xy, atol=1e-12, rtol=1e-12)
    assert torch.allclose(first_then_second.xz, second_then_first.xz, atol=1e-12, rtol=1e-12)
    assert torch.allclose(first_then_second.yz, second_then_first.yz, atol=1e-12, rtol=1e-12)
    ids = lattice.output_geometry
    del ids
    # The decoder is nonlinear, so equal writes do not imply additive loss;
    # this test only establishes the frozen-write part of the diagnostic.
    assert decoder.mlp is not None
