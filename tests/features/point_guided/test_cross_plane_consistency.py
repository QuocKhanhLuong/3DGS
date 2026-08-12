"""CPU invariants for deterministic Phase-7 cross-plane consistency."""

from __future__ import annotations

import pytest
import torch

from smagm.features.point_guided.cross_plane_consistency import (
    CONSISTENCY_EPSILON,
    CONSISTENCY_DESCRIPTOR_CHANNELS,
    POINT_SPECTRAL_CHANNELS,
    CrossPlaneConsistency,
    consistency_descriptor,
)
from smagm.features.point_guided.spectral_anchor import (
    SPECTRAL_ANCHOR_CHANNELS,
    SPECTRAL_ANCHOR_CHANNELS_PER_BAND,
)
from smagm.features.point_guided.swt_haar import SWT_HAAR_BAND_NAMES


def _feature(
    *,
    ll2: float = 0.0,
    lh1: float = 0.0,
    hl1: float = 0.0,
    hh1: float = 0.0,
    lh2: float = 0.0,
    hl2: float = 0.0,
    hh2: float = 0.0,
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Create one [B=1, N=1, 56] raw feature with distinct band blocks."""

    values = (ll2, lh1, hl1, hh1, lh2, hl2, hh2)
    blocks = [
        torch.full((1, 1, SPECTRAL_ANCHOR_CHANNELS_PER_BAND), value, dtype=dtype)
        for value in values
    ]
    return torch.cat(blocks, dim=-1).requires_grad_(requires_grad)


def _replace_band(feature: torch.Tensor, name: str, value: torch.Tensor) -> torch.Tensor:
    index = SWT_HAAR_BAND_NAMES.index(name)
    start = index * SPECTRAL_ANCHOR_CHANNELS_PER_BAND
    result = feature.clone()
    result[..., start : start + SPECTRAL_ANCHOR_CHANNELS_PER_BAND] = value
    return result


def test_descriptor_exactly_preserves_ll2_and_computes_elementwise_scale_energies() -> None:
    feature = torch.arange(SPECTRAL_ANCHOR_CHANNELS, dtype=torch.float64).reshape(1, 1, -1)
    descriptor = consistency_descriptor(feature)
    blocks = feature.reshape(1, 1, len(SWT_HAAR_BAND_NAMES), SPECTRAL_ANCHOR_CHANNELS_PER_BAND)
    expected_e1 = torch.sqrt(blocks[..., 1, :].square() + blocks[..., 2, :].square() + blocks[..., 3, :].square() + CONSISTENCY_EPSILON)
    expected_e2 = torch.sqrt(blocks[..., 4, :].square() + blocks[..., 5, :].square() + blocks[..., 6, :].square() + CONSISTENCY_EPSILON)
    expected = torch.cat((blocks[..., 0, :], expected_e1, expected_e2), dim=-1)

    assert descriptor.shape == (1, 1, CONSISTENCY_DESCRIPTOR_CHANNELS)
    torch.testing.assert_close(descriptor, expected, rtol=0.0, atol=0.0)


def test_descriptor_is_orientation_insensitive_within_each_swt_scale_but_raw_bands_remain_distinct() -> None:
    feature = _feature(ll2=1.0, lh1=2.0, hl1=3.0, hh1=4.0, lh2=5.0, hl2=6.0, hh2=7.0)
    swapped = feature.clone()
    for left, right in (("LH1", "HH1"), ("LH2", "HL2")):
        left_index = SWT_HAAR_BAND_NAMES.index(left)
        right_index = SWT_HAAR_BAND_NAMES.index(right)
        width = SPECTRAL_ANCHOR_CHANNELS_PER_BAND
        left_slice = slice(left_index * width, (left_index + 1) * width)
        right_slice = slice(right_index * width, (right_index + 1) * width)
        swapped[..., left_slice], swapped[..., right_slice] = (
            feature[..., right_slice],
            feature[..., left_slice],
        )

    torch.testing.assert_close(consistency_descriptor(feature), consistency_descriptor(swapped), rtol=0.0, atol=0.0)
    assert not torch.equal(feature, swapped)
    assert not torch.equal(feature[..., 8:16], swapped[..., 8:16])


def test_zero_and_identical_features_have_finite_equal_reliability() -> None:
    module = CrossPlaneConsistency()
    zero = torch.zeros(2, 3, SPECTRAL_ANCHOR_CHANNELS)
    zero_result = module(zero, zero, zero)

    assert torch.isfinite(zero_result.q_xy).all()
    assert torch.isfinite(zero_result.reliability).all()
    torch.testing.assert_close(zero_result.pairwise_cosines, torch.ones_like(zero_result.pairwise_cosines))
    torch.testing.assert_close(
        zero_result.reliability,
        torch.full_like(zero_result.reliability, 1.0 / 3.0),
    )

    identical = _feature(ll2=2.0, lh1=3.0, hl1=-5.0, hh1=7.0, lh2=-11.0, hl2=13.0, hh2=17.0)
    identical_result = module(identical, identical, identical)
    torch.testing.assert_close(
        identical_result.reliability,
        torch.full_like(identical_result.reliability, 1.0 / 3.0),
    )
    torch.testing.assert_close(identical_result.reliability.sum(dim=-1), torch.ones(1, 1))


def test_two_agreeing_planes_outweigh_a_discordant_plane_without_hard_drop() -> None:
    module = CrossPlaneConsistency()
    agreeing = _feature(ll2=1.0)
    outlier = _feature(ll2=-1.0)
    result = module(agreeing, agreeing, outlier)

    assert result.alpha_xy.item() > result.alpha_yz.item()
    assert result.alpha_xz.item() > result.alpha_yz.item()
    assert result.alpha_yz.item() > 0.0
    assert torch.all(result.reliability >= 0.0)
    torch.testing.assert_close(result.reliability.sum(dim=-1), torch.ones(1, 1))


def test_weighted_concat_preserves_xy_xz_yz_block_layout_and_raw_band_order() -> None:
    module = CrossPlaneConsistency()
    f_xy = torch.arange(SPECTRAL_ANCHOR_CHANNELS, dtype=torch.float64).reshape(1, 1, -1)
    f_xz = (100.0 + torch.arange(SPECTRAL_ANCHOR_CHANNELS, dtype=torch.float64)).reshape(1, 1, -1)
    f_yz = (200.0 + torch.arange(SPECTRAL_ANCHOR_CHANNELS, dtype=torch.float64)).reshape(1, 1, -1)
    result = module(f_xy, f_xz, f_yz)

    assert result.spectral_evidence.shape == (1, 1, POINT_SPECTRAL_CHANNELS)
    width = SPECTRAL_ANCHOR_CHANNELS
    torch.testing.assert_close(result.f_spec[..., :width], result.alpha_xy.unsqueeze(-1) * f_xy)
    torch.testing.assert_close(result.f_spec[..., width : 2 * width], result.alpha_xz.unsqueeze(-1) * f_xz)
    torch.testing.assert_close(result.f_spec[..., 2 * width :], result.alpha_yz.unsqueeze(-1) * f_yz)
    torch.testing.assert_close(f_xy[..., 0:8], torch.arange(8, dtype=torch.float64).reshape(1, 1, -1))
    torch.testing.assert_close(f_xy[..., 8:16], torch.arange(8, 16, dtype=torch.float64).reshape(1, 1, -1))


def test_module_preserves_dtype_and_gradients_and_has_no_trainable_state() -> None:
    module = CrossPlaneConsistency()
    f_xy = _feature(ll2=2.0, lh1=3.0, hl1=5.0, hh1=7.0, dtype=torch.float64, requires_grad=True)
    f_xz = _feature(ll2=11.0, lh1=13.0, hl1=17.0, hh1=19.0, dtype=torch.float64, requires_grad=True)
    f_yz = _feature(ll2=23.0, lh1=29.0, hl1=31.0, hh1=37.0, dtype=torch.float64, requires_grad=True)
    result = module(f_xy, f_xz, f_yz)

    assert result.q_xy.dtype == torch.float64
    assert result.reliability.dtype == torch.float64
    assert result.f_spec.dtype == torch.float64
    assert sum(parameter.numel() for parameter in module.parameters()) == 0
    assert list(module.buffers()) == []

    (result.f_spec.square().sum() + result.reliability.square().sum()).backward()
    for feature in (f_xy, f_xz, f_yz):
        assert feature.grad is not None
        assert torch.isfinite(feature.grad).all()


def test_contract_rejects_malformed_or_incompatible_feature_triplets() -> None:
    module = CrossPlaneConsistency()
    feature = torch.zeros(1, 2, SPECTRAL_ANCHOR_CHANNELS)

    with pytest.raises(ValueError, match="rank-3"):
        module(feature[0], feature, feature)
    with pytest.raises(ValueError, match="56"):
        module(feature[..., :-1], feature[..., :-1], feature[..., :-1])
    with pytest.raises(TypeError, match="floating"):
        module(feature.long(), feature.long(), feature.long())
    with pytest.raises(ValueError, match="match f_xy shape"):
        module(feature, feature[:, :1], feature)
    with pytest.raises(ValueError, match="finite"):
        module(feature, feature, _replace_band(feature, "LL2", torch.full((1, 2, 8), float("nan"))))
