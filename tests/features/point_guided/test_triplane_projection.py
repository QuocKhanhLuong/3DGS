"""CPU invariants for the Phase 4 static base tri-plane projector."""

from __future__ import annotations

import pytest
import torch

from smagm.features.point_guided.config import PointGuidedConfig
from smagm.features.point_guided.semantic_prior import SemanticPrior
from smagm.features.point_guided.triplane_projection import (
    BaseTriPlaneProjector,
    BaseTriPlanes,
)


PROJECTION_MODES = ("mean", "max", "pointwise_weighted", "axis_local_weighted")


def _config(**overrides: object) -> PointGuidedConfig:
    return PointGuidedConfig(num_semantic_classes=3, **overrides)


def _feature(*, channels: int = 3) -> torch.Tensor:
    torch.manual_seed(23)
    return torch.randn(2, channels, 5, 6, 7)


def _assert_planes_close(actual: BaseTriPlanes, expected: BaseTriPlanes) -> None:
    torch.testing.assert_close(actual.xy, expected.xy, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual.xz, expected.xz, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual.yz, expected.yz, rtol=1e-6, atol=1e-6)


def _mean_reference(feature: torch.Tensor) -> BaseTriPlanes:
    return BaseTriPlanes(
        xy=feature.mean(dim=2),
        xz=feature.mean(dim=3),
        yz=feature.mean(dim=4),
    )


@pytest.mark.parametrize("projection_mode", PROJECTION_MODES)
def test_all_authorized_projection_modes_return_exact_non_cubic_plane_shapes(
    projection_mode: str,
) -> None:
    feature = _feature()
    projector = BaseTriPlaneProjector(_config(projection_mode=projection_mode), input_channels=3)

    planes = projector(feature)

    assert isinstance(planes, BaseTriPlanes)
    assert planes.xy.shape == (2, 3, 6, 7)
    assert planes.xz.shape == (2, 3, 5, 7)
    assert planes.yz.shape == (2, 3, 5, 6)
    assert planes.bxy is planes.xy
    assert planes.bxz is planes.xz
    assert planes.byz is planes.yz
    _assert_planes_close(projector(feature), planes)


def test_mean_and_max_modes_match_their_exact_axis_references() -> None:
    feature = _feature()
    mean_projector = BaseTriPlaneProjector(_config(projection_mode="mean"), input_channels=3)
    max_projector = BaseTriPlaneProjector(_config(projection_mode="max"), input_channels=3)

    _assert_planes_close(mean_projector(feature), _mean_reference(feature))
    _assert_planes_close(
        max_projector(feature),
        BaseTriPlanes(
            xy=feature.max(dim=2).values,
            xz=feature.max(dim=3).values,
            yz=feature.max(dim=4).values,
        ),
    )
    assert not tuple(mean_projector.parameters())
    assert not tuple(max_projector.parameters())


@pytest.mark.parametrize("projection_mode", ("pointwise_weighted", "axis_local_weighted"))
def test_weighted_modes_emit_scalar_logits_and_axis_normalized_weights(
    projection_mode: str,
) -> None:
    feature = _feature()
    projector = BaseTriPlaneProjector(_config(projection_mode=projection_mode), input_channels=3)

    logits = projector.scorer_logits(feature)
    weights = projector.normalized_weights(feature)

    for score in (logits.xy, logits.xz, logits.yz):
        assert score.shape == (2, 1, 5, 6, 7)
    for weight, axis in ((weights.xy, 2), (weights.xz, 3), (weights.yz, 4)):
        assert bool((weight >= 0.0).all())
        torch.testing.assert_close(
            weight.sum(dim=axis),
            torch.ones_like(weight.sum(dim=axis)),
            rtol=1e-6,
            atol=1e-6,
        )
    _assert_planes_close(projector(feature), _mean_reference(feature))


def test_axis_local_main_is_zero_initialized_and_equals_mean_projection() -> None:
    feature = _feature(channels=4)
    projector = BaseTriPlaneProjector(_config(), input_channels=4)

    assert projector.projection_mode == "axis_local_weighted"
    assert projector.xy_scorer is not None
    assert projector.xz_scorer is not None
    assert projector.yz_scorer is not None
    assert projector.xy_scorer.kernel_size == (3, 1, 1)
    assert projector.xy_scorer.padding == (1, 0, 0)
    assert projector.xz_scorer.kernel_size == (1, 3, 1)
    assert projector.xz_scorer.padding == (0, 1, 0)
    assert projector.yz_scorer.kernel_size == (1, 1, 3)
    assert projector.yz_scorer.padding == (0, 0, 1)
    for scorer in (projector.xy_scorer, projector.xz_scorer, projector.yz_scorer):
        assert torch.count_nonzero(scorer.weight) == 0
        assert scorer.bias is not None
        assert torch.count_nonzero(scorer.bias) == 0

    _assert_planes_close(projector(feature), _mean_reference(feature))


def test_coordinate_ramps_prove_the_locked_xy_xz_yz_orientation() -> None:
    depth, height, width = 5, 6, 7
    d, h, w = torch.meshgrid(
        torch.arange(depth, dtype=torch.float32),
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    feature = (100.0 * d + 10.0 * h + w).unsqueeze(0).unsqueeze(0)
    projector = BaseTriPlaneProjector(_config(), input_channels=1)

    planes = projector(feature)
    expected_xy = 100.0 * d.mean(dim=0) + 10.0 * h.mean(dim=0) + w.mean(dim=0)
    expected_xz = 100.0 * d.mean(dim=1) + 10.0 * h.mean(dim=1) + w.mean(dim=1)
    expected_yz = 100.0 * d.mean(dim=2) + 10.0 * h.mean(dim=2) + w.mean(dim=2)

    torch.testing.assert_close(planes.xy[0, 0], expected_xy)
    torch.testing.assert_close(planes.xz[0, 0], expected_xz)
    torch.testing.assert_close(planes.yz[0, 0], expected_yz)


def test_projection_mode_is_explicit_and_rejects_unknown_values() -> None:
    assert _config().projection_mode == "axis_local_weighted"
    with pytest.raises(ValueError, match="projection_mode"):
        _config(projection_mode="attention")


def test_axis_local_main_has_only_the_three_locked_scorers_and_their_gradients() -> None:
    torch.manual_seed(29)
    feature = torch.randn(2, 4, 5, 6, 7)
    projector = BaseTriPlaneProjector(_config(), input_channels=4)

    assert set(projector.state_dict()) == {
        "xy_scorer.weight",
        "xy_scorer.bias",
        "xz_scorer.weight",
        "xz_scorer.bias",
        "yz_scorer.weight",
        "yz_scorer.bias",
    }
    assert sum(parameter.numel() for parameter in projector.parameters()) == 3 * (3 * 4 + 1)
    assert sum(
        parameter.numel()
        for parameter in BaseTriPlaneProjector(_config(), input_channels=64).parameters()
    ) == 579

    planes = projector(feature)
    (planes.xy.square().mean() + planes.xz.square().mean() + planes.yz.square().mean()).backward()

    for parameter in projector.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert bool(parameter.grad.abs().sum() > 0.0)


def test_axis_local_scorers_do_not_couple_unrelated_orthogonal_locations() -> None:
    projector = BaseTriPlaneProjector(_config(), input_channels=1)
    assert projector.xy_scorer is not None
    assert projector.xz_scorer is not None
    assert projector.yz_scorer is not None
    with torch.no_grad():
        for scorer in (projector.xy_scorer, projector.xz_scorer, projector.yz_scorer):
            scorer.weight.fill_(1.0)
            assert scorer.bias is not None
            scorer.bias.zero_()

    feature = torch.zeros(1, 1, 5, 6, 7)
    baseline = projector.scorer_logits(feature)
    perturbed = feature.clone()
    perturbed[0, 0, 2, 3, 4] = 1.0
    changed = projector.scorer_logits(perturbed)

    # Each scorer may communicate only along its own collapsed axis. The
    # selected locations differ on both orthogonal axes and stay unchanged.
    torch.testing.assert_close(changed.xy[:, :, :, 0, 0], baseline.xy[:, :, :, 0, 0])
    torch.testing.assert_close(changed.xz[:, :, 0, :, 0], baseline.xz[:, :, 0, :, 0])
    torch.testing.assert_close(changed.yz[:, :, 0, 0, :], baseline.yz[:, :, 0, 0, :])
    assert not torch.equal(changed.xy[:, :, :, 3, 4], baseline.xy[:, :, :, 3, 4])
    assert not torch.equal(changed.xz[:, :, 2, :, 4], baseline.xz[:, :, 2, :, 4])
    assert not torch.equal(changed.yz[:, :, 2, 3, :], baseline.yz[:, :, 2, 3, :])


@pytest.mark.parametrize("detach_backbone_features", (True, False))
def test_projector_preserves_the_phase_two_detach_boundary(
    detach_backbone_features: bool,
) -> None:
    torch.manual_seed(31)
    config = _config(
        freeze_coarse_backbone=False,
        detach_backbone_features=detach_backbone_features,
    )
    prior = SemanticPrior(config).eval()
    volumes = torch.randn(1, 3, 9, 11, 13, requires_grad=True)

    features = prior.extract_intermediate_features(volumes)
    selected = prior.select_spectral_feature(features)
    projector = BaseTriPlaneProjector(config, input_channels=selected.shape[1])
    planes = projector(selected)
    (planes.xy.square().mean() + planes.xz.square().mean() + planes.yz.square().mean()).backward()

    assert all(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0.0)
        for parameter in projector.parameters()
    )
    if detach_backbone_features:
        assert not selected.requires_grad
        assert volumes.grad is None
        assert all(parameter.grad is None for parameter in prior.backbone.parameters())
    else:
        assert selected.requires_grad
        assert volumes.grad is not None
        assert prior.backbone.conv1.weight.grad is not None
        assert bool(prior.backbone.conv1.weight.grad.abs().sum() > 0.0)


def test_selected_feature_projection_uses_one_shared_medicalnet_traversal() -> None:
    config = _config()
    prior = SemanticPrior(config).eval()
    volumes = torch.randn(1, 3, 9, 11, 13)
    module_names = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4")
    calls = {name: 0 for name in module_names}

    def counter(name: str):
        def count_call(*_args: object) -> None:
            calls[name] += 1

        return count_call

    hooks = [getattr(prior.backbone, name).register_forward_hook(counter(name)) for name in module_names]
    try:
        with torch.no_grad():
            features = prior.extract_intermediate_features(volumes)
            selected = prior.select_spectral_feature(features)
            projector = BaseTriPlaneProjector(config, input_channels=selected.shape[1])
            planes = projector(selected)
    finally:
        for hook in hooks:
            hook.remove()

    assert planes.xy.shape == (1, 64, 6, 7)
    assert calls == {name: 1 for name in module_names}
