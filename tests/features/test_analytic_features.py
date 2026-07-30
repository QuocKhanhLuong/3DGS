import torch

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
