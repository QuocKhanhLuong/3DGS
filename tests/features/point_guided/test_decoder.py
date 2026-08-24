"""Gate-D D1 tests: final-Z-only, affine-aware chunked implicit decoding."""

from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from smagm.features.point_guided import PointGuidedConfig, PointGuidedMRIModel
from smagm.features.point_guided.contracts import VolumeGeometry
from smagm.features.point_guided.decoder import (
    DECODER_HIDDEN_CHANNELS,
    TRIPLANE_QUERY_CHANNELS,
    DynamicTriPlaneVoxelQuery,
    ImplicitTriPlaneDecoder,
)
from smagm.features.point_guided.reward import DynamicStatePointQuery
from smagm.features.point_guided.sampling import voxel_dhw_to_ras_mm
from smagm.features.point_guided.spectral_query import FeatureGridGeometry
from smagm.features.point_guided.state_init import DynamicTriPlanes
from smagm.features.point_guided.trajectory_cost import TrajectoryConfig


def _feature_geometry(
    shape_dhw: tuple[int, int, int] = (3, 5, 7),
    affine: tuple[tuple[float, ...], ...] | None = None,
) -> FeatureGridGeometry:
    source = (
        VolumeGeometry(shape_dhw, affine)
        if affine is not None
        else VolumeGeometry.from_spacing(shape_dhw, (1.0, 1.0, 1.0))
    )
    # Synthetic tests deliberately use the identity source-to-feature lattice
    # so a source voxel maps to the corresponding known Z-plane value.  The
    # affine can still be arbitrary and therefore exercises DHW -> RAS -> DHW.
    return FeatureGridGeometry(
        source_geometry=source,
        feature_geometry=source,
        tap="conv1_pre_maxpool",
        feature_to_source_scale_dhw=(1.0, 1.0, 1.0),
        feature_to_source_offset_dhw=(0.0, 0.0, 0.0),
        operator_chain=("synthetic_identity_feature_grid",),
    )


def _ramp_state(
    geometry: FeatureGridGeometry,
    *,
    batch: int = 1,
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> DynamicTriPlanes:
    depth, height, width = geometry.shape_dhw
    channels = torch.arange(32, dtype=dtype).view(1, 32, 1, 1) * 1000.0
    d = torch.arange(depth, dtype=dtype).view(1, 1, depth, 1)
    h = torch.arange(height, dtype=dtype).view(1, 1, height, 1)
    h_yz = torch.arange(height, dtype=dtype).view(1, 1, 1, height)
    w = torch.arange(width, dtype=dtype).view(1, 1, 1, width)
    xy = (channels + 10.0 * h + w).expand(batch, -1, height, width).clone()
    xz = (channels + 10.0 * d + w).expand(batch, -1, depth, width).clone()
    yz = (channels + 10.0 * d + h_yz).expand(batch, -1, depth, height).clone()
    if requires_grad:
        xy.requires_grad_()
        xz.requires_grad_()
        yz.requires_grad_()
    return DynamicTriPlanes(xy=xy, xz=xz, yz=yz)


def _random_state(
    geometry: FeatureGridGeometry,
    *,
    batch: int = 1,
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> DynamicTriPlanes:
    depth, height, width = geometry.shape_dhw
    values = (
        torch.randn(batch, 32, height, width, dtype=dtype),
        torch.randn(batch, 32, depth, width, dtype=dtype),
        torch.randn(batch, 32, depth, height, dtype=dtype),
    )
    if requires_grad:
        values = tuple(value.requires_grad_() for value in values)
    return DynamicTriPlanes(xy=values[0], xz=values[1], yz=values[2])


def test_voxel_query_packs_exact_xy_xz_yz_blocks_at_integer_centres_without_axis_swap() -> None:
    geometry = _feature_geometry()
    state = _ramp_state(geometry)
    feature_dhw = torch.tensor([[[1.0, 2.0, 3.0]]])
    ras_mm = geometry.feature_dhw_to_ras_mm(feature_dhw)

    packed = DynamicTriPlaneVoxelQuery()(state, ras_mm, geometry)

    assert packed.shape == (1, 1, 96)
    assert torch.equal(packed[..., :32], state.xy[:, :, 2, 3].unsqueeze(1))
    assert torch.equal(packed[..., 32:64], state.xz[:, :, 1, 3].unsqueeze(1))
    assert torch.equal(packed[..., 64:96], state.yz[:, :, 1, 2].unsqueeze(1))
    assert packed[0, 0, 0].item() == pytest.approx(23.0)
    assert packed[0, 0, 32].item() == pytest.approx(13.0)
    assert packed[0, 0, 64].item() == pytest.approx(12.0)


def test_voxel_query_matches_manual_fractional_bilinear_ramps_and_has_coordinate_gradients() -> None:
    geometry = _feature_geometry()
    state = _ramp_state(geometry)
    feature_dhw = torch.tensor([[[1.25, 2.5, 3.25]]], requires_grad=True)
    packed = DynamicTriPlaneVoxelQuery()(state, geometry.feature_dhw_to_ras_mm(feature_dhw), geometry)

    channels = torch.arange(32, dtype=packed.dtype).view(1, 1, 32) * 1000.0
    xy_expected = channels + 10.0 * 2.5 + 3.25
    xz_expected = channels + 10.0 * 1.25 + 3.25
    yz_expected = channels + 10.0 * 1.25 + 2.5
    assert torch.allclose(packed[..., :32], xy_expected)
    assert torch.allclose(packed[..., 32:64], xz_expected)
    assert torch.allclose(packed[..., 64:96], yz_expected)

    packed.square().mean().backward()
    assert feature_dhw.grad is not None
    assert bool(torch.isfinite(feature_dhw.grad).all())
    assert bool(feature_dhw.grad.abs().sum() > 0.0)


@pytest.mark.parametrize(
    "affine",
    (
        ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ((1.7, 0.0, 0.0, 0.0), (0.0, 2.3, 0.0, 0.0), (0.0, 0.0, 3.1, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ((1.0, 0.0, 0.0, 13.0), (0.0, 1.0, 0.0, -7.0), (0.0, 0.0, 1.0, 2.5), (0.0, 0.0, 0.0, 1.0)),
        ((0.0, -1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ((1.0, 0.25, 0.0, 0.0), (0.15, 1.0, 0.1, 0.0), (0.0, 0.2, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ((0.0, -2.0, 0.3, 5.0), (1.5, 0.2, 0.1, -4.0), (0.1, 0.4, 3.0, 2.0), (0.0, 0.0, 0.0, 1.0)),
    ),
    ids=("identity", "anisotropic", "translation", "rotation", "shear", "combined"),
)
def test_voxel_query_preserves_full_affine_through_source_ras_and_feature_geometry(
    affine: tuple[tuple[float, ...], ...],
) -> None:
    geometry = _feature_geometry(affine=affine)
    state = _ramp_state(geometry)
    output_voxel = torch.tensor([[[2.0, 3.0, 4.0]]])
    ras_mm = voxel_dhw_to_ras_mm(output_voxel, geometry.source_geometry)
    packed = DynamicTriPlaneVoxelQuery()(state, ras_mm, geometry)

    assert torch.allclose(geometry.ras_mm_to_feature_dhw(ras_mm), output_voxel, atol=1e-5, rtol=1e-5)
    assert torch.allclose(packed[..., :32], state.xy[:, :, 3, 4].unsqueeze(1), atol=1e-4, rtol=1e-6)
    assert torch.allclose(packed[..., 32:64], state.xz[:, :, 2, 4].unsqueeze(1), atol=1e-4, rtol=1e-6)
    assert torch.allclose(packed[..., 64:96], state.yz[:, :, 2, 3].unsqueeze(1), atol=1e-4, rtol=1e-6)


def test_decoder_has_exact_shared_mlp_parameter_count_and_z_only_public_interface() -> None:
    decoder = ImplicitTriPlaneDecoder()
    linears = tuple(module for module in decoder.mlp if isinstance(module, nn.Linear))
    assert tuple((layer.in_features, layer.out_features, layer.bias is not None) for layer in linears) == (
        (96, 64, True),
        (64, 32, True),
        (32, 1, True),
    )
    assert DECODER_HIDDEN_CHANNELS == (64, 32)
    assert sum(parameter.numel() for parameter in decoder.parameters()) == 96 * 64 + 64 + 64 * 32 + 32 + 32 + 1
    assert sum(parameter.numel() for parameter in decoder.voxel_query.parameters()) == 0
    assert tuple(inspect.signature(ImplicitTriPlaneDecoder.forward).parameters) == (
        "self",
        "final_state",
        "feature_geometry",
        "output_geometry",
        "chunk_size",
    )
    assert tuple(inspect.signature(ImplicitTriPlaneDecoder.decode_points).parameters) == (
        "self",
        "final_state",
        "points_ras_mm",
        "feature_geometry",
    )
    assert TRIPLANE_QUERY_CHANNELS == 96


def test_target_free_point_ras_decode_matches_the_exact_gate_d_query_and_mlp_with_gradients() -> None:
    torch.manual_seed(29)
    geometry = _feature_geometry()
    state = _random_state(geometry, requires_grad=True)
    decoder = ImplicitTriPlaneDecoder().train()
    feature_dhw = torch.tensor([[[0.5, 1.25, 2.75], [2.0, 4.0, 6.0]]])
    points_ras_mm = geometry.feature_dhw_to_ras_mm(feature_dhw)

    decoded = decoder.decode_points(state, points_ras_mm, geometry)
    expected = decoder.mlp(decoder.voxel_query(state, points_ras_mm, geometry))

    assert decoded.shape == (1, 2, 1)
    torch.testing.assert_close(decoded, expected)
    decoded.square().mean().backward()
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in decoder.mlp.parameters())
    for plane in (state.xy, state.xz, state.yz):
        assert plane.grad is not None
        assert bool(torch.isfinite(plane.grad).all())
        assert bool(plane.grad.abs().sum() > 0.0)


def test_point_state_dtype_mismatch_fails_before_sampling_and_decoder_mlp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _feature_geometry()
    state = _random_state(geometry, dtype=torch.float32)
    points = geometry.feature_dhw_to_ras_mm(
        torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float64)
    )
    sampled = False

    def unexpected_grid_sample(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal sampled
        sampled = True
        raise AssertionError("grid_sample must not run for a point/state dtype mismatch")

    monkeypatch.setattr("smagm.features.point_guided.reward.F.grid_sample", unexpected_grid_sample)
    with pytest.raises(ValueError, match="^unsupported dynamic-state/point dtype pair:"):
        DynamicStatePointQuery()(state, points, geometry)
    assert sampled is False

    decoder = ImplicitTriPlaneDecoder()
    mlp_calls: list[bool] = []
    hook = decoder.mlp.register_forward_hook(lambda *_args: mlp_calls.append(True))
    try:
        with pytest.raises(ValueError, match="^unsupported dynamic-state/point dtype pair:"):
            decoder.decode_points(state, points, geometry)
    finally:
        hook.remove()
    assert mlp_calls == []


def test_full_volume_decode_delegates_to_point_ras_api_and_preserves_pointwise_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(31)
    geometry = _feature_geometry((3, 5, 7))
    state = _random_state(geometry, batch=2)
    decoder = ImplicitTriPlaneDecoder().eval()
    original_decode_points = decoder.decode_points
    point_chunks: list[torch.Tensor] = []

    def record_decode_points(
        final_state: DynamicTriPlanes,
        points_ras_mm: torch.Tensor,
        feature_geometry: FeatureGridGeometry,
    ) -> torch.Tensor:
        point_chunks.append(points_ras_mm.detach().clone())
        return original_decode_points(final_state, points_ras_mm, feature_geometry)

    monkeypatch.setattr(decoder, "decode_points", record_decode_points)
    prediction = decoder(state, geometry, geometry.source_geometry, chunk_size=4)

    depth, height, width = geometry.source_geometry.shape_dhw
    d, h, w = torch.meshgrid(
        torch.arange(depth, dtype=state.xy.dtype),
        torch.arange(height, dtype=state.xy.dtype),
        torch.arange(width, dtype=state.xy.dtype),
        indexing="ij",
    )
    output_voxels = torch.stack((d, h, w), dim=-1).reshape(1, -1, 3).expand(2, -1, -1)
    all_points_ras_mm = voxel_dhw_to_ras_mm(output_voxels, geometry.source_geometry)
    pointwise = original_decode_points(state, all_points_ras_mm, geometry)

    assert len(point_chunks) == 27
    assert all(chunk.shape[0] == 2 and chunk.shape[1] <= 4 and chunk.shape[-1] == 3 for chunk in point_chunks)
    assert torch.equal(torch.cat(point_chunks, dim=1), all_points_ras_mm)
    torch.testing.assert_close(prediction, pointwise.transpose(1, 2).reshape(2, 1, depth, height, width))


def test_decoder_outputs_non_cubic_batched_volume_and_chunking_is_exactly_equivalent() -> None:
    torch.manual_seed(19)
    geometry = _feature_geometry((3, 5, 7))
    state = _random_state(geometry, batch=2)
    decoder = ImplicitTriPlaneDecoder().eval()

    all_at_once = decoder(state, geometry, geometry.source_geometry, chunk_size=3 * 5 * 7)
    chunked = decoder(state, geometry, geometry.source_geometry, chunk_size=4)
    singleton = decoder(state, geometry, geometry.source_geometry, chunk_size=1)

    assert all_at_once.shape == (2, 1, 3, 5, 7)
    assert torch.allclose(chunked, all_at_once, atol=1e-6, rtol=1e-6)
    assert torch.allclose(singleton, all_at_once, atol=1e-6, rtol=1e-6)


def test_decoder_only_materializes_chunkwise_96_features_and_rejects_invalid_chunk_sizes() -> None:
    geometry = _feature_geometry((3, 5, 7))
    state = _random_state(geometry, batch=2)
    decoder = ImplicitTriPlaneDecoder().eval()
    query_shapes: list[tuple[int, ...]] = []
    hook = decoder.voxel_query.register_forward_hook(
        lambda _module, _inputs, output: query_shapes.append(tuple(output.shape))
    )
    try:
        prediction = decoder(state, geometry, geometry.source_geometry, chunk_size=4)
    finally:
        hook.remove()

    assert prediction.shape == (2, 1, 3, 5, 7)
    assert query_shapes
    assert all(shape[0] == 2 and shape[-1] == 96 and shape[1] <= 4 for shape in query_shapes)
    assert all(len(shape) == 3 for shape in query_shapes)
    for bad_chunk_size in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            decoder(state, geometry, geometry.source_geometry, chunk_size=bad_chunk_size)  # type: ignore[arg-type]


def test_decoder_gradients_flow_to_mlp_and_final_z_without_observation_inputs() -> None:
    geometry = _feature_geometry()
    state = _random_state(geometry, requires_grad=True)
    decoder = ImplicitTriPlaneDecoder().train()
    prediction = decoder(state, geometry, geometry.source_geometry, chunk_size=5)
    prediction.square().mean().backward()

    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in decoder.mlp.parameters())
    for plane in (state.xy, state.xz, state.yz):
        assert plane.grad is not None
        assert bool(torch.isfinite(plane.grad).all())
        assert bool(plane.grad.abs().sum() > 0.0)


def test_decoder_preserves_float64_and_is_deterministic_across_repeated_calls() -> None:
    torch.manual_seed(23)
    geometry = _feature_geometry()
    state = _random_state(geometry, dtype=torch.float64)
    decoder = ImplicitTriPlaneDecoder().double().eval()

    first = decoder(state, geometry, geometry.source_geometry, chunk_size=3)
    second = decoder(state, geometry, geometry.source_geometry, chunk_size=3)
    assert first.dtype is torch.float64
    assert torch.equal(first, second)


def test_model_reconstruction_runs_frontend_trajectory_and_decoder_once_without_mutating_static_inputs() -> None:
    trajectory_config = TrajectoryConfig(
        lambda_travel=0.01,
        lambda_overlap=0.01,
        lambda_step=0.01,
        k_max=1,
        selection_temperature=0.7,
        write_scale=0.2,
    )
    model = PointGuidedMRIModel(
        PointGuidedConfig(num_semantic_classes=3, num_points=3, point_candidate_multiplier=3, offset_hidden_channels=12),
        trajectory_config=trajectory_config,
    ).eval()
    assert model.trajectory is not None and model.decoder is not None
    with torch.no_grad():
        first = model.trajectory.reward_net.network[0]
        last = model.trajectory.reward_net.network[2]
        assert isinstance(first, nn.Linear) and isinstance(last, nn.Linear)
        first.weight.zero_()
        first.bias.zero_()
        last.weight.zero_()
        last.bias.fill_(3.0)

    backbone = model.semantic_prior.backbone
    names = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4")
    calls = {name: 0 for name in (*names, "projector", "anchor", "query", "consistency", "trajectory", "decoder")}
    frozen_static: dict[str, tuple[torch.Tensor, ...] | torch.Tensor] = {}
    hooks = [
        getattr(backbone, name).register_forward_hook(
            lambda _module, _inputs, _output, name=name: calls.__setitem__(name, calls[name] + 1)
        )
        for name in names
    ]
    for name, module in (
        ("projector", model.base_plane_projector),
        ("anchor", model.spectral_anchor_builder),
        ("query", model.spectral_point_query),
        ("consistency", model.cross_plane_consistency),
        ("trajectory", model.trajectory),
        ("decoder", model.decoder),
    ):
        hooks.append(module.register_forward_hook(lambda _module, _inputs, _output, name=name: calls.__setitem__(name, calls[name] + 1)))

    def capture_static(_module: nn.Module, args: tuple[object, ...]) -> None:
        base = args[0]
        frozen_static["base_before"] = tuple(getattr(base, name).detach().clone() for name in ("xy", "xz", "yz"))
        frozen_static["base_live"] = tuple(getattr(base, name) for name in ("xy", "xz", "yz"))
        frozen_static["f_spec_before"] = args[3].detach().clone()  # type: ignore[union-attr]
        frozen_static["f_spec_live"] = args[3]  # type: ignore[assignment]

    hooks.append(model.trajectory.register_forward_pre_hook(capture_static))
    try:
        result = model.forward_reconstruction(torch.randn(1, 3, 7, 7, 7), chunk_size=9)
    finally:
        for hook in hooks:
            hook.remove()

    assert result.prediction.shape == (1, 1, 7, 7, 7)
    assert result.geometry.shape_dhw == (7, 7, 7)
    assert all(count == 1 for count in calls.values())
    base_before = frozen_static["base_before"]
    base_live = frozen_static["base_live"]
    assert isinstance(base_before, tuple) and isinstance(base_live, tuple)
    assert all(torch.equal(before, after) for before, after in zip(base_before, base_live))
    f_spec_before = frozen_static["f_spec_before"]
    f_spec_live = frozen_static["f_spec_live"]
    assert isinstance(f_spec_before, torch.Tensor) and isinstance(f_spec_live, torch.Tensor)
    assert torch.equal(f_spec_before, f_spec_live)


def test_model_reconstruction_loss_reaches_gate_d_c_and_static_branch_but_not_frozen_medicalnet() -> None:
    model = PointGuidedMRIModel(
        PointGuidedConfig(num_semantic_classes=3, num_points=3, point_candidate_multiplier=3, offset_hidden_channels=12),
        trajectory_config=TrajectoryConfig(
            lambda_travel=0.01,
            lambda_overlap=0.01,
            lambda_step=0.01,
            k_max=1,
            selection_temperature=0.7,
            write_scale=0.2,
        ),
    ).train()
    assert model.trajectory is not None and model.decoder is not None
    result = model.forward_reconstruction(torch.randn(1, 3, 7, 7, 7), chunk_size=11)
    result.prediction.square().mean().backward()

    def has_gradient(parameters: object) -> bool:
        return any(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) and bool(parameter.grad.abs().sum() > 0.0)
            for parameter in parameters  # type: ignore[union-attr]
        )

    assert has_gradient(model.decoder.parameters())
    assert has_gradient(model.trajectory.update_net.parameters())
    assert has_gradient(model.trajectory.state_initializer.parameters())
    assert has_gradient(model.base_plane_projector.parameters())
    assert has_gradient(model.spectral_anchor_builder.band_projector.parameters())
    assert all(parameter.grad is None for parameter in model.semantic_prior.backbone.parameters())
