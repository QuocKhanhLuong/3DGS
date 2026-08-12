"""End-to-end CPU smoke tests for the locked frontend boundary."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from smagm.features.point_guided import PointGuidedConfig, PointGuidedMRIModel
from smagm.features.point_guided.medicalnet_resnet10 import MedicalNetResNet10
from smagm.features.point_guided.sampling import ras_mm_in_bounds
from smagm.features.point_guided.spectral_anchor import SpectralAnchor
from smagm.features.point_guided.triplane_projection import BaseTriPlanes


def _model(**overrides: object) -> PointGuidedMRIModel:
    values: dict[str, object] = {
        "num_semantic_classes": 3,
        "num_points": 4,
        "point_candidate_multiplier": 3,
    }
    values.update(overrides)
    return PointGuidedMRIModel(PointGuidedConfig(**values))  # type: ignore[arg-type]


def _assert_planes_close(actual: BaseTriPlanes, expected: BaseTriPlanes) -> None:
    for name in ("xy", "xz", "yz"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0.0, atol=0.0)


def _assert_anchors_close(actual: SpectralAnchor, expected: SpectralAnchor) -> None:
    for name in ("xy", "xz", "yz"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0.0, atol=0.0)


def _assert_spectral_evidence_close(actual: object, expected: object) -> None:
    torch.testing.assert_close(actual.f_spec, expected.f_spec, rtol=0.0, atol=0.0)  # type: ignore[attr-defined]
    torch.testing.assert_close(actual.reliability, expected.reliability, rtol=0.0, atol=0.0)  # type: ignore[attr-defined]


def _assert_core_outputs_equal(actual: object, expected: object) -> None:
    for name in (
        "s_coarse",
        "initial_points_ras_mm",
        "refined_points_ras_mm",
        "displacement_ras_mm",
        "point_semantic",
    ):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0.0, atol=0.0)
    assert actual.geometry == expected.geometry  # type: ignore[attr-defined]
    actual_pou = actual.sparse_pou  # type: ignore[attr-defined]
    expected_pou = expected.sparse_pou  # type: ignore[attr-defined]
    assert actual_pou.volume_shape_dhw == expected_pou.volume_shape_dhw
    for name in (
        "batch_indices",
        "voxel_indices_dhw",
        "point_indices",
        "raw_affinity",
        "normalized_weight",
        "unsupported_batch_indices",
        "unsupported_voxel_indices_dhw",
    ):
        torch.testing.assert_close(getattr(actual_pou, name), getattr(expected_pou, name), rtol=0.0, atol=0.0)


def test_frontend_returns_only_the_locked_point_field_and_sparse_pou() -> None:
    model = _model().eval()
    x = torch.randn(1, 3, 9, 9, 9)
    brain_mask = torch.ones(1, 1, 9, 9, 9, dtype=torch.bool)

    with torch.no_grad():
        output = model.forward_frontend(x, brain_mask, spacing_mm=(1.0, 1.5, 2.0))

    assert output.S_coarse.shape == (1, 3, 9, 9, 9)
    assert output.initial_points.shape == output.refined_points.shape == (1, 4, 3)
    assert output.displacement.shape == (1, 4, 3)
    assert output.point_semantic.shape == (1, 4, 3)
    assert torch.allclose(output.S_coarse.sum(dim=1), torch.ones_like(output.S_coarse[:, 0]), atol=1e-5)
    assert torch.linalg.vector_norm(output.displacement, dim=-1).amax() <= 2.0 + 1e-6
    assert bool(ras_mm_in_bounds(output.refined_points, output.geometry).all())
    assert output.sparse_pou.normalized_weight.ndim == 1
    assert output.sparse_pou.normalized_weight.numel() > 0
    assert isinstance(output.base_planes, BaseTriPlanes)
    assert output.base_planes.xy.shape == output.base_planes.xz.shape == output.base_planes.yz.shape == (1, 64, 5, 5)
    assert isinstance(output.spectral_anchor, SpectralAnchor)
    assert output.spectral_anchor.xy.shape == output.spectral_anchor.xz.shape == output.spectral_anchor.yz.shape == (1, 56, 5, 5)
    assert output.f_spec.shape == (1, 4, 168)
    assert output.reliability.shape == (1, 4, 3)
    assert bool(torch.isfinite(output.f_spec).all())
    assert bool(torch.isfinite(output.reliability).all())
    torch.testing.assert_close(output.reliability.sum(dim=-1), torch.ones_like(output.reliability[..., 0]))


def test_frontend_output_fails_closed_for_a_nonproduction_semantic_width() -> None:
    model = _model().eval()
    with torch.no_grad():
        output = model.forward_frontend(torch.randn(1, 3, 9, 9, 9))

    with pytest.raises(ValueError, match="exactly 3"):
        replace(output, s_coarse=output.s_coarse[:, :2])
    with pytest.raises(TypeError, match="BaseTriPlanes"):
        replace(output, base_planes=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="SpectralAnchor"):
        replace(output, spectral_anchor=None)  # type: ignore[arg-type]


def test_downstream_loss_reaches_the_offset_predictor_through_refined_points() -> None:
    model = _model().train()
    x = torch.randn(1, 3, 9, 9, 9)
    output = model.forward_frontend(x, spacing_mm=(1.0, 1.0, 1.0))
    loss = (
        output.refined_points.square().mean()
        + output.point_semantic.square().mean()
        + output.sparse_pou.raw_affinity.mean()
    )
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.point_refiner.offset_predictor.parameters()
        if parameter.requires_grad
    ]
    assert gradients and all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients)


def test_full_forward_refuses_to_synthesize_an_unresolved_t1ce_volume() -> None:
    with pytest.raises(NotImplementedError, match="Full T1ce synthesis is unresolved"):
        _model()(torch.randn(1, 3, 9, 9, 9))


@pytest.mark.parametrize("spectral_tap", ("conv1_pre_maxpool", "layer1"))
def test_integrated_frontend_uses_one_shared_backbone_pass_and_routes_the_selected_tap(
    spectral_tap: str,
) -> None:
    model = _model(spectral_tap=spectral_tap, projection_mode="mean").eval()
    final_offset_layer = model.point_refiner.offset_predictor.network[-1]
    assert isinstance(final_offset_layer, torch.nn.Linear)
    with torch.no_grad():
        final_offset_layer.weight.zero_()
        final_offset_layer.bias.fill_(0.25)
    x = torch.randn(1, 3, 9, 11, 13)
    backbone = model.semantic_prior.backbone
    module_names = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4")
    calls = {name: 0 for name in module_names}
    captured: dict[str, torch.Tensor] = {}

    def counter(name: str):
        def count_call(*_args: object) -> None:
            calls[name] += 1

        return count_call

    def capture_output(name: str):
        def record_output(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            captured[name] = output.detach().clone()

        return record_output

    def capture_projector_input(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        captured["projector"] = inputs[0].detach().clone()

    def capture_anchor_input(
        _module: torch.nn.Module,
        inputs: tuple[BaseTriPlanes, ...],
    ) -> None:
        planes = inputs[0]
        captured["anchor_input_xy"] = planes.xy.detach().clone()
        captured["anchor_input_xz"] = planes.xz.detach().clone()
        captured["anchor_input_yz"] = planes.yz.detach().clone()

    def capture_query_points(
        _module: torch.nn.Module,
        inputs: tuple[object, ...],
    ) -> None:
        points = inputs[1]
        assert isinstance(points, torch.Tensor)
        captured["query_points"] = points.detach().clone()
        anchor = inputs[0]
        assert isinstance(anchor, SpectralAnchor)
        captured["query_anchor_xy"] = anchor.xy.detach().clone()
        captured["query_anchor_xz"] = anchor.xz.detach().clone()
        captured["query_anchor_yz"] = anchor.yz.detach().clone()

    hooks = [getattr(backbone, name).register_forward_hook(counter(name)) for name in module_names]
    hooks.extend(
        (
            backbone.relu.register_forward_hook(capture_output("shallow")),
            backbone.layer1.register_forward_hook(capture_output("layer1")),
            model.base_plane_projector.register_forward_pre_hook(capture_projector_input),
            model.spectral_anchor_builder.register_forward_pre_hook(capture_anchor_input),
            model.spectral_point_query.register_forward_pre_hook(capture_query_points),
        )
    )
    projector_calls = [0]
    anchor_calls = [0]
    query_calls = [0]
    consistency_calls = [0]

    def count_base_projector(*_args: object) -> None:
        projector_calls[0] += 1

    def count_anchor_builder(*_args: object) -> None:
        anchor_calls[0] += 1

    def count_spectral_query(*_args: object) -> None:
        query_calls[0] += 1

    def count_cross_plane_consistency(*_args: object) -> None:
        consistency_calls[0] += 1

    hooks.extend(
        (
            model.base_plane_projector.register_forward_hook(count_base_projector),
            model.spectral_anchor_builder.register_forward_hook(count_anchor_builder),
            model.spectral_point_query.register_forward_hook(count_spectral_query),
            model.cross_plane_consistency.register_forward_hook(count_cross_plane_consistency),
        )
    )
    semantic_head_calls = [0]

    def count_semantic_head(*_args: object) -> None:
        semantic_head_calls[0] += 1

    hooks.append(model.semantic_prior.semantic_head.register_forward_hook(count_semantic_head))
    try:
        with torch.no_grad():
            output = model.forward_frontend(x)
    finally:
        for hook in hooks:
            hook.remove()

    assert calls == {name: 1 for name in module_names}
    assert semantic_head_calls == [1]
    assert projector_calls == [1]
    assert anchor_calls == [1]
    assert query_calls == [1]
    assert consistency_calls == [1]
    assert not torch.equal(output.initial_points, output.refined_points)
    torch.testing.assert_close(captured["query_points"], output.refined_points, rtol=0.0, atol=0.0)
    for name in ("xy", "xz", "yz"):
        torch.testing.assert_close(
            captured[f"anchor_input_{name}"],
            getattr(output.base_planes, name),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            captured[f"query_anchor_{name}"],
            getattr(output.spectral_anchor, name),
            rtol=0.0,
            atol=0.0,
        )
    selected = captured["shallow" if spectral_tap == "conv1_pre_maxpool" else "layer1"]
    torch.testing.assert_close(captured["projector"], selected, rtol=0.0, atol=0.0)
    batch, channels, depth, height, width = selected.shape
    assert output.base_planes.xy.shape == (batch, channels, height, width)
    assert output.base_planes.xz.shape == (batch, channels, depth, width)
    assert output.base_planes.yz.shape == (batch, channels, depth, height)


def test_integrated_semantics_and_existing_point_pou_outputs_are_isolated_from_base_planes() -> None:
    torch.manual_seed(41)
    model = _model().eval()
    x = torch.randn(1, 3, 9, 9, 9)

    with torch.no_grad():
        features = model.semantic_prior.extract_intermediate_features(x)
        expected_semantics = model.semantic_prior.forward_from_intermediate_features(
            features,
            output_spatial_shape=x.shape[-3:],
        )
        before = model.forward_frontend(x)
        assert model.base_plane_projector.xy_scorer is not None
        model.base_plane_projector.xy_scorer.weight.fill_(0.25)
        after = model.forward_frontend(x)

    torch.testing.assert_close(before.s_coarse, expected_semantics, rtol=0.0, atol=0.0)
    _assert_core_outputs_equal(after, before)
    assert not torch.allclose(after.base_planes.xy, before.base_planes.xy)
    assert not torch.allclose(after.spectral_anchor.xy, before.spectral_anchor.xy)
    assert not torch.allclose(after.f_spec, before.f_spec)


def test_phase7_query_and_fusion_are_read_only_to_earlier_base_and_anchor_outputs() -> None:
    model = _model().eval()
    captured_geometry: list[object] = []

    def capture_feature_geometry(
        _module: torch.nn.Module,
        inputs: tuple[object, ...],
    ) -> None:
        captured_geometry.append(inputs[2])

    hook = model.spectral_point_query.register_forward_pre_hook(capture_feature_geometry)
    try:
        with torch.no_grad():
            output = model.forward_frontend(torch.randn(1, 3, 9, 9, 9))
    finally:
        hook.remove()

    assert len(captured_geometry) == 1
    base_before = BaseTriPlanes(
        xy=output.base_planes.xy.clone(),
        xz=output.base_planes.xz.clone(),
        yz=output.base_planes.yz.clone(),
    )
    anchor_before = SpectralAnchor(
        xy=output.spectral_anchor.xy.clone(),
        xz=output.spectral_anchor.xz.clone(),
        yz=output.spectral_anchor.yz.clone(),
    )
    with torch.no_grad():
        samples = model.spectral_point_query(
            output.spectral_anchor,
            output.refined_points,
            captured_geometry[0],  # type: ignore[arg-type]
        )
        recomputed = model.cross_plane_consistency(samples.xy, samples.xz, samples.yz)

    _assert_planes_close(output.base_planes, base_before)
    _assert_anchors_close(output.spectral_anchor, anchor_before)
    torch.testing.assert_close(recomputed.f_spec, output.f_spec, rtol=0.0, atol=0.0)
    torch.testing.assert_close(recomputed.reliability, output.reliability, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "projection_mode",
    ("mean", "max", "pointwise_weighted", "axis_local_weighted"),
)
def test_all_projection_modes_are_wired_to_the_diagnostic_frontend_output(
    projection_mode: str,
) -> None:
    model = _model(projection_mode=projection_mode).eval()
    x = torch.randn(1, 3, 7, 7, 7)

    with torch.no_grad():
        first = model.forward_frontend(x)
        second = model.forward_frontend(x)

    assert isinstance(first.base_planes, BaseTriPlanes)
    assert first.base_planes.xy.shape == first.base_planes.xz.shape == first.base_planes.yz.shape == (1, 64, 4, 4)
    for plane in (first.base_planes.xy, first.base_planes.xz, first.base_planes.yz):
        assert bool(torch.isfinite(plane).all())
    _assert_planes_close(second.base_planes, first.base_planes)
    _assert_anchors_close(second.spectral_anchor, first.spectral_anchor)
    _assert_spectral_evidence_close(second, first)


def test_projection_mode_changes_only_the_static_diagnostic_base_planes() -> None:
    baseline = _model(projection_mode="axis_local_weighted").eval()
    variant = _model(projection_mode="max").eval()
    core_state = {
        name: value
        for name, value in baseline.state_dict().items()
        if not name.startswith("base_plane_projector.")
    }
    variant.load_state_dict(core_state)
    x = torch.randn(1, 3, 7, 7, 7)

    with torch.no_grad():
        baseline_output = baseline.forward_frontend(x)
        variant_output = variant.forward_frontend(x)

    _assert_core_outputs_equal(variant_output, baseline_output)
    assert not torch.allclose(variant_output.base_planes.xy, baseline_output.base_planes.xy)
    assert not torch.allclose(variant_output.spectral_anchor.xy, baseline_output.spectral_anchor.xy)
    assert not torch.allclose(variant_output.f_spec, baseline_output.f_spec)


@pytest.mark.parametrize(
    ("freeze_backbone", "detach_feature", "expects_backbone_grad", "expects_input_grad"),
    (
        (True, True, False, False),
        (True, False, False, True),
        (False, True, False, False),
        (False, False, True, True),
    ),
)
def test_spectral_anchor_loss_honors_the_integrated_detach_and_freeze_boundaries(
    freeze_backbone: bool,
    detach_feature: bool,
    expects_backbone_grad: bool,
    expects_input_grad: bool,
) -> None:
    model = _model(
        freeze_coarse_backbone=freeze_backbone,
        detach_backbone_features=detach_feature,
    ).eval()
    x = torch.randn(1, 3, 7, 7, 7, requires_grad=True)

    output = model.forward_frontend(x)
    loss = sum(plane.square().mean() for plane in (output.spectral_anchor.xy, output.spectral_anchor.xz, output.spectral_anchor.yz))
    loss.backward()

    for name, parameter in model.base_plane_projector.named_parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        # Softmax is invariant to a scorer's spatially constant bias, so its
        # bias gradient may be exactly zero.  Each learned scorer kernel must
        # still receive the B-only training signal.
        if name.endswith("weight"):
            assert bool(parameter.grad.abs().sum() > 0.0)
    for parameter in model.spectral_anchor_builder.band_projector.parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        assert bool(parameter.grad.abs().sum() > 0.0)
    assert all(parameter.grad is None for parameter in model.semantic_prior.semantic_head.parameters())
    assert (model.semantic_prior.backbone.conv1.weight.grad is not None) is expects_backbone_grad
    assert (x.grad is not None) is expects_input_grad
    if expects_backbone_grad:
        assert bool(model.semantic_prior.backbone.conv1.weight.grad.abs().sum() > 0.0)
    if expects_input_grad:
        assert bool(x.grad.abs().sum() > 0.0)


def test_phase7_spectral_evidence_loss_reaches_static_planes_anchor_and_refined_points() -> None:
    torch.manual_seed(61)
    model = _model(offset_hidden_channels=12).train()
    x = torch.randn(1, 3, 9, 9, 9)

    output = model.forward_frontend(x)
    output.f_spec.square().mean().backward()

    for name, parameter in model.base_plane_projector.named_parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        if name.endswith("weight"):
            assert bool(parameter.grad.abs().sum() > 0.0)
    for parameter in model.spectral_anchor_builder.band_projector.parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        assert bool(parameter.grad.abs().sum() > 0.0)
    assert all(parameter.grad is None for parameter in model.semantic_prior.backbone.parameters())
    gradients = [
        parameter.grad
        for parameter in model.point_refiner.offset_predictor.parameters()
        if parameter.requires_grad
    ]
    assert gradients and all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert any(bool(gradient.abs().sum() > 0.0) for gradient in gradients if gradient is not None)


def test_model_owns_persistent_phase6_projectors_with_round_trippable_default_state() -> None:
    torch.manual_seed(43)
    model = _model().eval()
    projector = model.base_plane_projector
    anchor_builder = model.spectral_anchor_builder
    spectral_point_query = model.spectral_point_query
    cross_plane_consistency = model.cross_plane_consistency
    assert projector is model.base_plane_projector
    assert anchor_builder is model.spectral_anchor_builder
    assert spectral_point_query is model.spectral_point_query
    assert cross_plane_consistency is model.cross_plane_consistency
    assert sum(isinstance(module, MedicalNetResNet10) for module in model.modules()) == 1
    assert tuple(name for name in model.state_dict() if name.startswith("base_plane_projector.")) == (
        "base_plane_projector.xy_scorer.weight",
        "base_plane_projector.xy_scorer.bias",
        "base_plane_projector.xz_scorer.weight",
        "base_plane_projector.xz_scorer.bias",
        "base_plane_projector.yz_scorer.weight",
        "base_plane_projector.yz_scorer.bias",
    )
    assert tuple(name for name in model.state_dict() if name.startswith("spectral_anchor_builder.")) == (
        "spectral_anchor_builder.swt.low_filter",
        "spectral_anchor_builder.swt.high_filter",
        "spectral_anchor_builder.swt.ll_filter",
        "spectral_anchor_builder.swt.lh_filter",
        "spectral_anchor_builder.swt.hl_filter",
        "spectral_anchor_builder.swt.hh_filter",
        "spectral_anchor_builder.band_projector.weight",
        "spectral_anchor_builder.band_projector.bias",
    )
    assert not tuple(
        name
        for name in model.state_dict()
        if name.startswith(("spectral_point_query.", "cross_plane_consistency."))
    )
    for module in (spectral_point_query, cross_plane_consistency):
        assert sum(parameter.numel() for parameter in module.parameters()) == 0
        assert tuple(module.state_dict()) == ()
    assert sum(parameter.numel() for parameter in projector.parameters()) == 579
    x = torch.randn(1, 3, 7, 7, 7)

    with torch.no_grad():
        before = model.forward_frontend(x)
        assert projector.xy_scorer is not None
        projector.xy_scorer.weight.fill_(0.2)
        changed = model.forward_frontend(x)

    assert model.base_plane_projector is projector
    assert model.spectral_anchor_builder is anchor_builder
    assert model.spectral_point_query is spectral_point_query
    assert model.cross_plane_consistency is cross_plane_consistency
    assert not torch.allclose(before.base_planes.xy, changed.base_planes.xy)
    assert not torch.allclose(before.spectral_anchor.xy, changed.spectral_anchor.xy)
    restored = _model().eval()
    restored.load_state_dict(model.state_dict())
    with torch.no_grad():
        replayed = restored.forward_frontend(x)
    _assert_core_outputs_equal(replayed, changed)
    _assert_planes_close(replayed.base_planes, changed.base_planes)
    _assert_anchors_close(replayed.spectral_anchor, changed.spectral_anchor)
    _assert_spectral_evidence_close(replayed, changed)


def test_changing_only_the_phase6_band_projector_changes_only_the_spectral_anchor() -> None:
    torch.manual_seed(59)
    model = _model().eval()
    x = torch.randn(1, 3, 9, 9, 9)
    with torch.no_grad():
        before = model.forward_frontend(x)
        model.spectral_anchor_builder.band_projector.weight.add_(0.2)
        after = model.forward_frontend(x)

    _assert_core_outputs_equal(after, before)
    _assert_planes_close(after.base_planes, before.base_planes)
    assert not torch.allclose(after.spectral_anchor.xy, before.spectral_anchor.xy)
    assert not torch.allclose(after.f_spec, before.f_spec)


@pytest.mark.parametrize("spectral_tap", ("conv1_pre_maxpool", "layer1"))
def test_phase6_anchor_shapes_follow_both_configured_shared_taps(spectral_tap: str) -> None:
    model = _model(spectral_tap=spectral_tap).eval()
    with torch.no_grad():
        output = model.forward_frontend(torch.randn(1, 3, 9, 11, 13))
    for anchor_plane, base_plane in zip(
        (output.spectral_anchor.xy, output.spectral_anchor.xz, output.spectral_anchor.yz),
        (output.base_planes.xy, output.base_planes.xz, output.base_planes.yz),
        strict=True,
    ):
        assert anchor_plane.shape == (1, 56, *base_plane.shape[-2:])


def test_projector_only_optimization_leaves_the_default_frozen_medicalnet_unchanged() -> None:
    torch.manual_seed(47)
    model = _model().train()
    assert not model.semantic_prior.backbone.training
    assert model.semantic_prior.semantic_head.training
    assert model.base_plane_projector.training
    assert all(not parameter.requires_grad for parameter in model.semantic_prior.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.base_plane_projector.parameters())
    backbone_before = {
        name: value.detach().clone() for name, value in model.semantic_prior.backbone.state_dict().items()
    }
    projector_before = {
        name: value.detach().clone() for name, value in model.base_plane_projector.state_dict().items()
    }
    optimizer = torch.optim.SGD(model.base_plane_projector.parameters(), lr=0.1)
    output = model.forward_frontend(torch.randn(1, 3, 7, 7, 7))
    loss = sum(plane.square().mean() for plane in (output.base_planes.xy, output.base_planes.xz, output.base_planes.yz))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert any(
        not torch.equal(value, model.base_plane_projector.state_dict()[name])
        for name, value in projector_before.items()
    )
    assert all(torch.equal(value, model.semantic_prior.backbone.state_dict()[name]) for name, value in backbone_before.items())


def test_phase7_evidence_scales_to_the_locked_2048_point_frontend_contract() -> None:
    model = _model(num_points=2048, point_candidate_multiplier=1, offset_hidden_channels=12).eval()
    with torch.no_grad():
        output = model.forward_frontend(torch.randn(1, 3, 7, 7, 7))

    assert output.refined_points.shape == (1, 2048, 3)
    assert output.f_spec.shape == (1, 2048, 168)
    assert output.reliability.shape == (1, 2048, 3)
    assert bool(torch.isfinite(output.f_spec).all())
    torch.testing.assert_close(output.reliability.sum(dim=-1), torch.ones_like(output.reliability[..., 0]))
