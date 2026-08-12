"""CPU invariants for the Phase-6 static SWT-Haar spectral anchor."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from smagm.features.point_guided.config import PointGuidedConfig
from smagm.features.point_guided.spectral_anchor import (
    SPECTRAL_ANCHOR_CHANNELS,
    SPECTRAL_ANCHOR_CHANNELS_PER_BAND,
    SpectralAnchor,
    StaticSpectralAnchor,
)
from smagm.features.point_guided.swt_haar import SWT_HAAR_BAND_NAMES, SwtHaarBands
from smagm.features.point_guided.triplane_projection import BaseTriPlanes


def _config(**overrides: object) -> PointGuidedConfig:
    return PointGuidedConfig(num_semantic_classes=3, **overrides)


def _base_planes(*, dtype: torch.dtype = torch.float32) -> BaseTriPlanes:
    torch.manual_seed(67)
    return BaseTriPlanes(
        xy=torch.randn(2, 64, 6, 7, dtype=dtype),
        xz=torch.randn(2, 64, 5, 7, dtype=dtype),
        yz=torch.randn(2, 64, 5, 6, dtype=dtype),
    )


def _assert_anchor_close(actual: SpectralAnchor, expected: SpectralAnchor) -> None:
    for name in ("xy", "xz", "yz"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0.0, atol=0.0)


def test_anchor_uses_one_shared_persistent_conv2d_for_all_21_band_plane_applications() -> None:
    builder = StaticSpectralAnchor(_config(), input_channels=64)
    assert isinstance(builder.band_projector, torch.nn.Conv2d)
    assert builder.band_projector.in_channels == 64
    assert builder.band_projector.out_channels == 8
    assert builder.band_projector.kernel_size == (1, 1)
    assert builder.band_projector.bias is not None  # PyTorch-default implementation detail.
    assert sum(isinstance(module, torch.nn.Conv2d) for module in builder.modules()) == 1
    assert sum(parameter.numel() for parameter in builder.parameters()) == 64 * 8 + 8

    calls = [0]

    def count_call(*_args: object) -> None:
        calls[0] += 1

    hook = builder.band_projector.register_forward_hook(count_call)
    try:
        anchor = builder(_base_planes())
    finally:
        hook.remove()

    assert calls == [21]
    assert anchor.xy.shape == (2, 56, 6, 7)
    assert anchor.xz.shape == (2, 56, 5, 7)
    assert anchor.yz.shape == (2, 56, 5, 6)


def test_anchor_concatenates_the_canonical_seven_band_blocks_in_exact_order() -> None:
    builder = StaticSpectralAnchor(_config(), input_channels=64).eval()
    with torch.no_grad():
        builder.band_projector.weight.fill_(1.0)
        assert builder.band_projector.bias is not None
        builder.band_projector.bias.zero_()

    base = _base_planes()
    class DistinguishableSwt(torch.nn.Module):
        def forward(self, plane: torch.Tensor) -> SwtHaarBands:
            return SwtHaarBands(
                **{
                    field: torch.full_like(plane, float(index + 1))
                    for index, field in enumerate(("ll2", "lh1", "hl1", "hh1", "lh2", "hl2", "hh2"))
                }
            )

    builder.swt = DistinguishableSwt()
    anchor = builder(base)
    for plane_name, base_plane in (("xy", base.xy), ("xz", base.xz), ("yz", base.yz)):
        actual_plane = getattr(anchor, plane_name)
        for index in range(7):
            expected_block = torch.full_like(base_plane[:, :8], 64.0 * (index + 1))
            start = index * SPECTRAL_ANCHOR_CHANNELS_PER_BAND
            torch.testing.assert_close(actual_plane[:, start : start + 8], expected_block)
    assert SWT_HAAR_BAND_NAMES == ("LL2", "LH1", "HL1", "HH1", "LH2", "HL2", "HH2")


def test_default_main_path_has_no_normalization_and_band_gn_is_the_only_optional_ablation() -> None:
    main = StaticSpectralAnchor(_config(), input_channels=64)
    ablation = StaticSpectralAnchor(_config(anchor_norm="band_gn"), input_channels=64)

    assert main.anchor_norm == "none"
    assert main.band_gn is None
    assert not any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in main.modules())
    assert ablation.band_gn is not None
    assert ablation.band_gn.num_groups == 7
    assert ablation.band_gn.num_channels == 56
    assert not ablation.band_gn.affine
    with pytest.raises(ValueError, match="anchor_norm"):
        _config(anchor_norm="layer_norm")
    with pytest.raises(ValueError, match="anchor_norm"):
        _config(anchor_norm=True)  # type: ignore[arg-type]


def test_anchor_rejects_wrong_base_channel_width_and_preserves_base_planes_without_mutation() -> None:
    builder = StaticSpectralAnchor(_config(), input_channels=64)
    base = _base_planes()
    before = tuple(plane.clone() for plane in (base.xy, base.xz, base.yz))
    builder(base)
    for actual, expected in zip((base.xy, base.xz, base.yz), before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    wrong = BaseTriPlanes(
        xy=torch.randn(1, 63, 4, 5),
        xz=torch.randn(1, 63, 3, 5),
        yz=torch.randn(1, 63, 3, 4),
    )
    with pytest.raises(ValueError, match="64 channels"):
        builder(wrong)


def test_spectral_anchor_typed_contract_fails_closed_for_malformed_plane() -> None:
    anchor = StaticSpectralAnchor(_config(), input_channels=64)(_base_planes())
    with pytest.raises(ValueError, match="56"):
        replace(anchor, xy=anchor.xy[:, :55])
    with pytest.raises(ValueError, match="retained DHW"):
        SpectralAnchor(xy=anchor.xy, xz=anchor.xz[..., :-1], yz=anchor.yz)


def test_anchor_preserves_float64_and_fixed_haar_filters_are_not_trainable_parameters() -> None:
    builder = StaticSpectralAnchor(_config(), input_channels=64).double()
    base = _base_planes(dtype=torch.float64)
    anchor = builder(base)

    for plane in (anchor.xy, anchor.xz, anchor.yz):
        assert plane.dtype == torch.float64
    assert set(name for name, _ in builder.named_parameters()) == {
        "band_projector.weight",
        "band_projector.bias",
    }
    assert set(name for name, _ in builder.swt.named_buffers()) == {
        "low_filter",
        "high_filter",
        "ll_filter",
        "lh_filter",
        "hl_filter",
        "hh_filter",
    }
    assert all(not buffer.requires_grad for buffer in builder.swt.buffers())
    assert SPECTRAL_ANCHOR_CHANNELS == 56
