"""CPU invariants for the locked two-level stationary SWT-Haar transform."""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from smagm.features.point_guided.swt_haar import (
    SWT_HAAR_BAND_NAMES,
    SwtHaarBands,
    TwoLevelSwtHaar,
)


def _same_reflect_reference(
    plane: torch.Tensor,
    filter_2d: torch.Tensor,
    *,
    dilation: int,
) -> torch.Tensor:
    """Independent explicit reference for the locked same-grid padding rule."""

    before = dilation // 2
    after = dilation - before
    padded = F.pad(plane, (before, after, before, after), mode="reflect")
    channels = plane.shape[1]
    weight = filter_2d.to(dtype=plane.dtype, device=plane.device).expand(channels, -1, -1, -1)
    return F.conv2d(padded, weight, dilation=dilation, groups=channels)


def _energy(value: torch.Tensor) -> torch.Tensor:
    return value.square().mean().sqrt()


def test_filters_are_normalized_fixed_non_trainable_buffers() -> None:
    transform = TwoLevelSwtHaar()
    scale = 1.0 / math.sqrt(2.0)

    torch.testing.assert_close(transform.low_filter, torch.tensor((scale, scale)))
    torch.testing.assert_close(transform.high_filter, torch.tensor((scale, -scale)))
    torch.testing.assert_close(
        transform.ll_filter,
        torch.tensor(((scale * scale, scale * scale), (scale * scale, scale * scale))).view(1, 1, 2, 2),
    )
    torch.testing.assert_close(
        transform.lh_filter,
        torch.tensor(((scale * scale, -scale * scale), (scale * scale, -scale * scale))).view(1, 1, 2, 2),
    )
    torch.testing.assert_close(
        transform.hl_filter,
        torch.tensor(((scale * scale, scale * scale), (-scale * scale, -scale * scale))).view(1, 1, 2, 2),
    )
    torch.testing.assert_close(
        transform.hh_filter,
        torch.tensor(((scale * scale, -scale * scale), (-scale * scale, scale * scale))).view(1, 1, 2, 2),
    )

    assert not tuple(transform.parameters())
    assert set(transform.state_dict()) == {
        "low_filter",
        "high_filter",
        "ll_filter",
        "lh_filter",
        "hl_filter",
        "hh_filter",
    }
    for _, buffer in transform.named_buffers():
        assert not buffer.requires_grad
        assert bool(torch.isfinite(buffer).all())


@pytest.mark.parametrize("height,width", ((6, 7), (5, 7), (5, 6)))
def test_all_seven_bands_preserve_odd_even_non_square_plane_shapes(height: int, width: int) -> None:
    torch.manual_seed(height * 10 + width)
    plane = torch.randn(2, 64, height, width)

    bands = TwoLevelSwtHaar()(plane)

    assert isinstance(bands, SwtHaarBands)
    assert len(bands.as_tuple()) == 7
    for band in bands.as_tuple():
        assert band.shape == plane.shape
        assert band.dtype == plane.dtype
        assert band.device == plane.device


@pytest.mark.parametrize("shape", ((1, 3, 1, 6), (1, 3, 6, 1)))
def test_singleton_plane_axis_fails_closed_for_reflect_padding(shape: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError, match="reflect-padded SWT-Haar"):
        TwoLevelSwtHaar()(torch.randn(shape))


@pytest.mark.parametrize("invalid", (float("nan"), float("inf"), -float("inf")))
def test_non_finite_plane_fails_closed(invalid: float) -> None:
    plane = torch.ones(1, 1, 3, 4)
    plane[0, 0, 1, 2] = invalid

    with pytest.raises(ValueError, match="plane must be finite"):
        TwoLevelSwtHaar()(plane)


def test_constant_plane_has_negligible_all_high_pass_bands_under_reflect_padding() -> None:
    # Float64 makes the numerical tolerance deliberately stricter than the
    # float32 operation needed in normal frontend execution.
    plane = torch.full((1, 2, 7, 8), 3.25, dtype=torch.float64)
    bands = TwoLevelSwtHaar()(plane)

    for high_pass in (bands.lh1, bands.hl1, bands.hh1, bands.lh2, bands.hl2, bands.hh2):
        assert float(high_pass.abs().max()) < 1e-12


def test_ramps_prove_first_symbol_is_h_and_second_symbol_is_w() -> None:
    height, width = 9, 10
    w_ramp = torch.arange(width, dtype=torch.float64).view(1, 1, 1, width).expand(1, 1, height, width)
    h_ramp = torch.arange(height, dtype=torch.float64).view(1, 1, height, 1).expand(1, 1, height, width)
    transform = TwoLevelSwtHaar()

    along_w = transform(w_ramp)
    along_h = transform(h_ramp)

    # LH = low H / high W; HL = high H / low W.
    assert float(_energy(along_w.lh1)) > 0.1
    assert float(_energy(along_w.hl1)) < 1e-12
    assert float(_energy(along_h.hl1)) > 0.1
    assert float(_energy(along_h.lh1)) < 1e-12


def test_level_two_uses_dilation_two_not_a_second_level_one_copy() -> None:
    plane = torch.zeros(1, 1, 9, 11, dtype=torch.float64)
    plane[:, :, 4, 5] = 1.0
    transform = TwoLevelSwtHaar()

    bands = transform(plane)
    ll1 = _same_reflect_reference(plane, transform.ll_filter, dilation=1)
    expected_lh2 = _same_reflect_reference(ll1, transform.lh_filter, dilation=2)
    wrong_lh2 = _same_reflect_reference(ll1, transform.lh_filter, dilation=1)

    torch.testing.assert_close(bands.lh2, expected_lh2, rtol=0.0, atol=1e-12)
    assert not torch.allclose(bands.lh2, wrong_lh2, rtol=0.0, atol=1e-12)


def test_small_translation_preserves_same_grid_alignment_away_from_reflect_boundaries() -> None:
    torch.manual_seed(53)
    plane = torch.randn(1, 2, 15, 17, dtype=torch.float64)
    shifted_plane = plane.roll(shifts=(1, -1), dims=(-2, -1))
    transform = TwoLevelSwtHaar()

    reference = transform(plane)
    shifted = transform(shifted_plane)
    # The four-pixel margin excludes reflect-boundary influence from both SWT
    # levels.  In the remaining interior a one-pixel input translation produces
    # the same one-pixel coefficient-grid translation without resampling.
    for original, translated in zip(reference.as_tuple(), shifted.as_tuple(), strict=True):
        expected = original.roll(shifts=(1, -1), dims=(-2, -1))
        torch.testing.assert_close(translated[..., 4:-4, 4:-4], expected[..., 4:-4, 4:-4])


def test_public_result_contains_exactly_the_seven_stable_bands_without_ll1() -> None:
    expected_names = ("LL2", "LH1", "HL1", "HH1", "LH2", "HL2", "HH2")
    expected_fields = ("ll2", "lh1", "hl1", "hh1", "lh2", "hl2", "hh2")

    assert SWT_HAAR_BAND_NAMES == expected_names
    assert tuple(SwtHaarBands.__dataclass_fields__) == expected_fields

    bands = TwoLevelSwtHaar()(torch.randn(1, 1, 5, 6))
    assert not hasattr(bands, "ll1")
    assert len(bands.as_tuple()) == len(expected_names)


def test_float64_input_is_preserved_and_remains_differentiable() -> None:
    torch.manual_seed(61)
    plane = torch.randn(1, 2, 5, 7, dtype=torch.float64, requires_grad=True)

    transform = TwoLevelSwtHaar().to(device=plane.device, dtype=torch.float64)
    assert all(buffer.dtype == torch.float64 for buffer in transform.buffers())
    assert all(buffer.device == plane.device for buffer in transform.buffers())

    bands = transform(plane)
    assert all(band.dtype == torch.float64 for band in bands.as_tuple())
    loss = sum(band.square().mean() for band in bands.as_tuple())
    loss.backward()

    assert plane.grad is not None
    assert plane.grad.dtype == torch.float64
    assert bool(torch.isfinite(plane.grad).all())
