import torch
import pytest

from smagm.features.analytic import ANALYTIC_CHANNEL_NAMES, analytic_feature_bank


def test_constant_image_has_finite_zero_differential_channels() -> None:
    image = torch.ones((2, 1, 17, 19), dtype=torch.float64, requires_grad=True)
    output = analytic_feature_bank(image)
    assert output.tensor.shape == (2, len(ANALYTIC_CHANNEL_NAMES), 17, 19)
    assert torch.isfinite(output.tensor).all()
    assert torch.allclose(output.tensor[:, 0], torch.zeros_like(output.tensor[:, 0]))
    for channel in (1, 2, 4, 5, 6):
        assert torch.allclose(output.tensor[:, channel], torch.zeros_like(output.tensor[:, channel]), atol=1e-12)
    assert torch.all(output.tensor[:, 3] > 0)
    output.tensor.sum().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()


def test_horizontal_ramp_has_positive_interior_u_gradient() -> None:
    ramp = torch.arange(21, dtype=torch.float64).reshape(1, 1, 1, 21).expand(1, 1, 15, 21).clone()
    output = analytic_feature_bank(ramp)
    interior = output.tensor[0, 1, 2:-2, 2:-2]
    assert torch.all(interior > 0)
    assert torch.allclose(interior, interior.mean().expand_as(interior), atol=1e-10, rtol=1e-10)
    assert torch.allclose(output.tensor[0, 2, 2:-2, 2:-2], torch.zeros_like(interior), atol=1e-10)


def test_masked_pixels_are_zero_and_do_not_break_normalization() -> None:
    image = torch.randn((1, 1, 13, 13), dtype=torch.float32)
    mask = torch.ones_like(image, dtype=torch.bool)
    mask[:, :, :3, :4] = False
    output = analytic_feature_bank(image, mask)
    expanded_mask = mask.expand_as(output.tensor)
    assert torch.all(output.tensor[~expanded_mask] == 0)
    assert torch.isfinite(output.tensor).all()


@pytest.mark.parametrize("dtype,atol", [(torch.float32, 2e-6), (torch.float64, 1e-12)])
def test_spacing_scales_derivatives_and_physical_contrast_windows(dtype: torch.dtype, atol: float) -> None:
    """Physical spacings affect axes independently, without changing dtype."""
    v, u = torch.meshgrid(
        torch.arange(13, dtype=dtype), torch.arange(15, dtype=dtype), indexing="ij"
    )
    image = (3.0 * u + 5.0 * v + 0.1 * u * v).reshape(1, 1, 13, 15)
    unit = analytic_feature_bank(image, spacing_uv_mm=(1.0, 1.0))
    spaced = analytic_feature_bank(image, spacing_uv_mm=(2.0, 4.0))
    interior = (slice(None), slice(None), slice(2, -2), slice(2, -2))

    assert spaced.tensor.dtype is dtype
    assert torch.allclose(spaced.tensor[interior][:, 1], unit.tensor[interior][:, 1] / 2.0, atol=atol, rtol=atol)
    assert torch.allclose(spaced.tensor[interior][:, 2], unit.tensor[interior][:, 2] / 4.0, atol=atol, rtol=atol)
    assert torch.allclose(spaced.tensor[interior][:, 4], unit.tensor[interior][:, 4] / 16.0, atol=atol, rtol=atol)

    # r=2 mm becomes a one-pixel radius in each axis at 2x4-mm spacing, while
    # r=4 mm becomes radii (v,u)=(1,2).  These values are an independent mean.
    normalized = spaced.tensor[0, 0]
    v_index, u_index = 6, 7
    r2 = torch.stack(
        [normalized[max(0, min(12, v_index + dv)), max(0, min(14, u_index + du))]
         for dv in range(-1, 2) for du in range(-1, 2)]
    ).mean()
    r4 = torch.stack(
        [normalized[max(0, min(12, v_index + dv)), max(0, min(14, u_index + du))]
         for dv in range(-1, 2) for du in range(-2, 3)]
    ).mean()
    assert torch.allclose(spaced.tensor[0, 5, v_index, u_index], normalized[v_index, u_index] - r2, atol=atol, rtol=atol)
    assert torch.allclose(spaced.tensor[0, 6, v_index, u_index], normalized[v_index, u_index] - r4, atol=atol, rtol=atol)


def test_anisotropic_spacing_scales_nonzero_quadratic_laplacian() -> None:
    v, u = torch.meshgrid(torch.arange(13, dtype=torch.float64), torch.arange(15, dtype=torch.float64), indexing="ij")
    image = (3.0 * u.square() + 5.0 * v.square()).reshape(1, 1, 13, 15)
    spacing_uv_mm = (2.0, 4.0)
    output = analytic_feature_bank(image, spacing_uv_mm=spacing_uv_mm)

    mean = image.mean()
    normalized_scale = ((image - mean).square().mean()).sqrt()
    expected = (6.0 / spacing_uv_mm[0] ** 2 + 10.0 / spacing_uv_mm[1] ** 2) / normalized_scale
    interior = output.tensor[0, 4, 2:-2, 2:-2]
    assert expected > 0.0
    assert torch.allclose(interior, torch.full_like(interior, expected), atol=1e-12, rtol=1e-12)
