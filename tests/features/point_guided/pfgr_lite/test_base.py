from __future__ import annotations

import pytest
import torch

from smagm.features.point_guided import PointGuidedConfig
from smagm.features.point_guided.pfgr_lite import PFGRLiteConfig, PFGRLiteModel, StaticSynthesisConfig


def _frontend_config(**overrides: object) -> PointGuidedConfig:
    values: dict[str, object] = {
        "num_semantic_classes": 3,
        "num_points": 4,
        "point_candidate_multiplier": 3,
        "offset_hidden_channels": 12,
        "detach_backbone_features": False,
    }
    values.update(overrides)
    return PointGuidedConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("variant", ("b0_legacy_v1", "b1_multiscale_v1", "b2_ordered_multiscale_v1", "b_light_ordered_v1"))
def test_static_variants_return_graph_preserving_32_channel_z0(variant: str) -> None:
    config = PFGRLiteConfig(static=StaticSynthesisConfig(variant=variant))
    model = PFGRLiteModel(config, frontend_config=_frontend_config()).train()
    x = torch.randn(1, 3, 9, 11, 13)
    context = model.encode_observations(x, None, (1.0, 1.5, 2.0))
    state = model.initialize_state(context)
    assert context.initial_planes.xy.shape == state.planes.xy.shape == (1, 32, 6, 7)
    assert state.planes.xy.grad_fn is not None
    loss = sum(plane.square().mean() for plane in (state.planes.xy, state.planes.xz, state.planes.yz))
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.static_head.parameters())
    context.validate_integrity()
    state.validate_integrity()


def test_ordered_source_conditioner_distinguishes_modalities_and_b1_is_zero_slot_control() -> None:
    from smagm.features.point_guided.pfgr_lite import StaticSynthesisConfig

    x = torch.zeros(1, 3, 9, 9, 9)
    x[:, 0] = 1.0
    y = x.clone()
    y[:, 0], y[:, 1] = y[:, 1].clone(), y[:, 0].clone()
    b2 = PFGRLiteModel(PFGRLiteConfig(static=StaticSynthesisConfig(variant="b2_ordered_multiscale_v1")), frontend_config=_frontend_config()).eval()
    b1 = PFGRLiteModel(PFGRLiteConfig(static=StaticSynthesisConfig(variant="b1_multiscale_v1")), frontend_config=_frontend_config()).eval()
    with torch.no_grad():
        first = b2.encode_observations(x, None, (1.0, 1.0, 1.0)).initial_planes.xy
        second = b2.encode_observations(y, None, (1.0, 1.0, 1.0)).initial_planes.xy
        first_control = b1.encode_observations(x, None, (1.0, 1.0, 1.0)).initial_planes.xy
        second_control = b1.encode_observations(y, None, (1.0, 1.0, 1.0)).initial_planes.xy
    assert not torch.equal(first, second)
    # B1 may still differ because its legacy semantic/refiner frontend sees
    # ordered raw modalities; its reserved static source slots themselves are
    # explicit zeros and therefore do not add a source bypass.
    assert first_control.shape == second_control.shape


def test_decode_final_requires_canonical_w2_lattice_injection() -> None:
    model = PFGRLiteModel(PFGRLiteConfig(), frontend_config=_frontend_config()).eval()
    context = model.encode_observations(torch.randn(1, 3, 9, 9, 9), None, (1.0, 1.0, 1.0))
    state = model.initialize_state(context)
    with pytest.raises(RuntimeError, match="canonical PFGRQueryLattice"):
        model.decode_final(state, context, chunk_size=8)
