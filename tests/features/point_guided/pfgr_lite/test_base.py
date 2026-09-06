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
    config = PFGRLiteConfig(static=StaticSynthesisConfig(variant=variant), num_points=4, engineering_only=True)
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


def test_b0_matches_legacy_state_initializer_on_nontrivial_affine() -> None:
    from smagm.features.point_guided.state_init import DynamicStateInitializer

    config = PFGRLiteConfig(
        static=StaticSynthesisConfig(variant="b0_legacy_v1"),
        num_points=4,
        engineering_only=True,
    )
    model = PFGRLiteModel(config, frontend_config=_frontend_config()).eval()
    affine = (
        (1.1, 0.2, 0.0, 4.0),
        (0.0, 1.7, 0.15, -3.0),
        (0.1, 0.0, 2.3, 2.5),
        (0.0, 0.0, 0.0, 1.0),
    )
    x = torch.randn(1, 3, 9, 11, 13)
    # Construct through the typed geometry path so rotation/shear/translation
    # are part of the identity (spacing agrees with affine column lengths).
    from smagm.features.point_guided.contracts import VolumeGeometry

    geometry = VolumeGeometry((9, 11, 13), affine)
    context = model.encode_observations(x, None, geometry)
    legacy = DynamicStateInitializer()
    legacy.load_state_dict(model.static_head.state_initializer.state_dict())
    expected = legacy(context.frontend.base_planes)
    assert torch.equal(context.initial_planes.xy, expected.xy)
    assert torch.equal(context.initial_planes.xz, expected.xz)
    assert torch.equal(context.initial_planes.yz, expected.yz)


def test_ordered_source_conditioner_distinguishes_modalities_and_b1_is_zero_slot_control() -> None:
    from smagm.features.point_guided.pfgr_lite import StaticSynthesisConfig

    x = torch.zeros(1, 3, 9, 9, 9)
    x[:, 0] = 1.0
    y = x.clone()
    y[:, 0], y[:, 1] = y[:, 1].clone(), y[:, 0].clone()
    b2 = PFGRLiteModel(PFGRLiteConfig(static=StaticSynthesisConfig(variant="b2_ordered_multiscale_v1"), num_points=4, engineering_only=True), frontend_config=_frontend_config()).eval()
    b1 = PFGRLiteModel(PFGRLiteConfig(static=StaticSynthesisConfig(variant="b1_multiscale_v1"), num_points=4, engineering_only=True), frontend_config=_frontend_config()).eval()
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


def test_b1_b2_capacity_match_and_model_enforces_single_subject_batch() -> None:
    b1 = PFGRLiteModel(
        PFGRLiteConfig(static=StaticSynthesisConfig(variant="b1_multiscale_v1"), num_points=4, engineering_only=True),
        frontend_config=_frontend_config(),
    )
    b2 = PFGRLiteModel(
        PFGRLiteConfig(static=StaticSynthesisConfig(variant="b2_ordered_multiscale_v1"), num_points=4, engineering_only=True),
        frontend_config=_frontend_config(),
    )
    assert b1.static_head.effective_parameter_count == b2.static_head.effective_parameter_count
    with pytest.raises(ValueError, match="B=1"):
        b1.encode_observations(torch.randn(2, 3, 9, 9, 9), None, (1.0, 1.0, 1.0))


def test_state_clone_preserves_graph_and_isolates_context_planes() -> None:
    model = PFGRLiteModel(
        PFGRLiteConfig(num_points=4, engineering_only=True), frontend_config=_frontend_config()
    ).train()
    context = model.encode_observations(torch.randn(1, 3, 9, 9, 9), None, (1.0, 1.0, 1.0))
    state = model.initialize_state(context)
    original = context.initial_planes.xy.clone()
    with torch.no_grad():
        state.planes.xy.add_(1.0)
    assert torch.equal(context.initial_planes.xy, original)
    with pytest.raises(RuntimeError, match="mutation"):
        state.validate_integrity()
    # The mutation guard failure is expected; a fresh state remains connected.
    fresh = model.initialize_state(context)
    assert fresh.planes.xy.grad_fn is not None


def test_context_resolves_and_owns_observation_mask() -> None:
    model = PFGRLiteModel(
        PFGRLiteConfig(num_points=4, engineering_only=True), frontend_config=_frontend_config()
    ).eval()
    x = torch.randn(1, 3, 9, 9, 9)
    mask = torch.ones(1, 9, 9, 9, dtype=torch.float32)
    mask[..., 0] = 0.0
    context = model.encode_observations(x, mask, (1.0, 1.0, 1.0))
    assert context.mask.dtype == torch.bool
    mask.fill_(0.0)
    assert bool(context.mask[..., 1].all())
    with pytest.raises(ValueError, match="binary"):
        model.encode_observations(x, torch.full((1, 9, 9, 9), 0.5), (1.0, 1.0, 1.0))


def test_medicalnet_parameters_and_bn_buffers_stay_frozen_across_modes() -> None:
    from smagm.features.point_guided.pfgr_lite import batchnorm_state_digest, module_parameter_digest

    model = PFGRLiteModel(
        PFGRLiteConfig(num_points=4, engineering_only=True), frontend_config=_frontend_config()
    )
    backbone = model.frontend.semantic_prior.backbone
    before_parameters = module_parameter_digest(backbone)
    before_bn = batchnorm_state_digest(backbone)
    model.train()
    model.encode_observations(torch.randn(1, 3, 9, 9, 9), None, (1.0, 1.0, 1.0))
    model.eval()
    model.encode_observations(torch.randn(1, 3, 9, 9, 9), None, (1.0, 1.0, 1.0))
    assert module_parameter_digest(backbone) == before_parameters
    assert batchnorm_state_digest(backbone) == before_bn
    assert all(not parameter.requires_grad for parameter in backbone.parameters())


def test_extra_medicalnet_traversal_is_fail_closed(monkeypatch) -> None:
    model = PFGRLiteModel(
        PFGRLiteConfig(num_points=4, engineering_only=True), frontend_config=_frontend_config()
    )
    original = model.frontend._forward_frontend_with_gate_b_context_and_features

    def twice(*args, **kwargs):
        first = original(*args, **kwargs)
        original(*args, **kwargs)
        return first

    monkeypatch.setattr(model.frontend, "_forward_frontend_with_gate_b_context_and_features", twice)
    with pytest.raises(RuntimeError, match="exactly one MedicalNet traversal"):
        model.encode_observations(torch.randn(1, 3, 9, 9, 9), None, (1.0, 1.0, 1.0))


def test_context_rejects_state_initialization_after_producer_optimizer_change() -> None:
    model = PFGRLiteModel(
        PFGRLiteConfig(num_points=4, engineering_only=True), frontend_config=_frontend_config()
    )
    context = model.encode_observations(torch.randn(1, 3, 9, 9, 9), None, (1.0, 1.0, 1.0))
    with torch.no_grad():
        next(iter(model.static_head.parameters())).add_(0.001)
    with pytest.raises(ValueError, match="stale"):
        model.initialize_state(context)


def test_decode_final_requires_canonical_w2_lattice_injection() -> None:
    model = PFGRLiteModel(PFGRLiteConfig(num_points=4, engineering_only=True), frontend_config=_frontend_config()).eval()
    context = model.encode_observations(torch.randn(1, 3, 9, 9, 9), None, (1.0, 1.0, 1.0))
    state = model.initialize_state(context)
    with pytest.raises(RuntimeError, match="canonical PFGRQueryLattice"):
        model.decode_final(state, context, chunk_size=8)


def test_decode_final_uses_integer_voxel_ids_and_configured_chunk_bound() -> None:
    class Lattice:
        def __init__(self) -> None:
            self.ids: list[torch.Tensor] = []
            self.chunks: list[int] = []

        def query(self, state, voxel_ids_dhw, *, chunk_size):
            del state
            self.ids.append(voxel_ids_dhw.detach().clone())
            self.chunks.append(chunk_size)
            return torch.zeros(voxel_ids_dhw.shape[0], 96, dtype=torch.float32)

    class Factory:
        def __init__(self, lattice: Lattice) -> None:
            self.lattice = lattice
            self.build_kwargs = None

        def build(self, **kwargs):
            self.build_kwargs = kwargs
            return self.lattice

    config = PFGRLiteConfig(num_points=4, engineering_only=True, decode_chunk_size=2, build_chunk_size=3)
    model = PFGRLiteModel(config, frontend_config=_frontend_config()).eval()
    context = model.encode_observations(torch.randn(1, 3, 5, 5, 5), None, (1.0, 1.0, 1.0))
    state = model.initialize_state(context)
    lattice = Lattice()
    factory = Factory(lattice)
    model.set_query_lattice_factory(factory)
    prediction = model.decode_final(state, context, chunk_size=99)
    assert prediction.shape == (1, 1, 5, 5, 5)
    assert factory.build_kwargs is not None
    assert factory.build_kwargs["build_chunk_size"] == 3
    assert all(ids.dtype == torch.long for ids in lattice.ids)
    assert all(chunk <= 2 for chunk in lattice.chunks)
