from __future__ import annotations

import pytest
import torch

from smagm.features.point_guided.losses import (
    CHARBONNIER_EPSILON,
    ReconstructionLossConfig,
    charbonnier_loss,
    gradient_agreement_loss,
    reconstruction_loss,
    structural_similarity_loss,
)


def test_charbonnier_matches_analytic_masked_reference_and_ignores_invalid_target() -> None:
    prediction = torch.arange(8, dtype=torch.float64).reshape(1, 1, 2, 2, 2).requires_grad_()
    target = torch.zeros_like(prediction)
    valid_mask = torch.zeros_like(prediction, dtype=torch.bool)
    valid_mask[..., 0, 0, 0] = True
    valid_mask[..., 0, 1, 1] = True
    valid_mask[..., 1, 0, 1] = True
    target[..., 0, 0, 0] = -1.0
    target[..., 0, 1, 1] = 2.0
    target[..., 1, 0, 1] = 7.0
    target[~valid_mask] = float("nan")

    loss = charbonnier_loss(prediction, target, valid_mask)
    expected = torch.sqrt(
        (prediction.detach()[valid_mask] - target[valid_mask]).square() + CHARBONNIER_EPSILON**2
    ).mean()

    torch.testing.assert_close(loss, expected)
    assert loss.dtype is torch.float64
    loss.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())
    torch.testing.assert_close(prediction.grad[~valid_mask], torch.zeros_like(prediction.grad[~valid_mask]))
    assert bool(prediction.grad[valid_mask].abs().sum() > 0.0)


def test_reconstruction_loss_composes_live_float64_components_and_backpropagates() -> None:
    values = torch.linspace(-1.0, 1.0, steps=64, dtype=torch.float64).reshape(1, 1, 4, 4, 4)
    prediction = values.detach().clone().requires_grad_()
    target = torch.zeros_like(prediction)
    config = ReconstructionLossConfig(lambda_ssim=0.2, lambda_grad=0.1, ssim_data_range=2.0)

    result = reconstruction_loss(prediction, target, config=config)

    assert set(result.components) == {"charbonnier", "ssim", "gradient"}
    assert result.valid_voxel_count == prediction.numel()
    assert all(value.dtype is torch.float64 for value in (*result.components.values(), result.total))
    torch.testing.assert_close(result.total, result.charbonnier + 0.2 * result.ssim + 0.1 * result.gradient)

    result.total.backward()
    assert prediction.grad is not None
    assert prediction.grad.dtype is torch.float64
    assert bool(torch.isfinite(prediction.grad).all())
    assert bool(prediction.grad.abs().sum() > 0.0)


def test_local_3d_ssim_matches_analytic_single_window_reference() -> None:
    prediction = torch.arange(27, dtype=torch.float64).reshape(1, 1, 3, 3, 3) / 13.0
    target = torch.flip(prediction, dims=(-3,)) + 0.25
    data_range = 2.0

    loss = structural_similarity_loss(prediction, target, data_range=data_range)

    mean_prediction = prediction.mean()
    mean_target = target.mean()
    variance_prediction = prediction.square().mean() - mean_prediction.square()
    variance_target = target.square().mean() - mean_target.square()
    covariance = (prediction * target).mean() - mean_prediction * mean_target
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    expected_ssim = ((2.0 * mean_prediction * mean_target + c1) * (2.0 * covariance + c2)) / (
        (mean_prediction.square() + mean_target.square() + c1) * (variance_prediction + variance_target + c2)
    )
    torch.testing.assert_close(loss, 1.0 - expected_ssim)


def test_local_3d_ssim_uses_only_wholly_valid_windows() -> None:
    torch.manual_seed(73)
    prediction = torch.randn(1, 1, 5, 5, 5, dtype=torch.float64)
    target = prediction.detach().clone() + 0.2
    valid_mask = torch.zeros_like(prediction, dtype=torch.bool)
    valid_mask[..., 1:4, 1:4, 1:4] = True
    target[~valid_mask] = float("nan")

    baseline = structural_similarity_loss(prediction, target, valid_mask, data_range=2.0)
    changed_outside_support = prediction.detach().clone()
    changed_outside_support[~valid_mask] += 10_000.0
    isolated = structural_similarity_loss(changed_outside_support, target, valid_mask, data_range=2.0)

    torch.testing.assert_close(isolated, baseline)
    no_window_mask = torch.zeros_like(valid_mask)
    no_window_mask[..., 2, 2, 2] = True
    with pytest.raises(ValueError, match="wholly valid"):
        structural_similarity_loss(prediction, target, no_window_mask, data_range=2.0)


@pytest.mark.parametrize("axis", (0, 1, 2), ids=("D", "H", "W"))
def test_gradient_agreement_uses_each_dhw_axis(axis: int) -> None:
    depth, height, width = 3, 4, 5
    d, h, w = torch.meshgrid(
        torch.arange(depth, dtype=torch.float64),
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    target = (d, h, w)[axis].unsqueeze(0).unsqueeze(0)
    prediction = torch.zeros_like(target, requires_grad=True)

    loss = gradient_agreement_loss(prediction, target)

    pair_counts = (
        (depth - 1) * height * width,
        depth * (height - 1) * width,
        depth * height * (width - 1),
    )
    expected = prediction.new_tensor(pair_counts[axis] / sum(pair_counts))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())
    assert bool(prediction.grad.abs().sum() > 0.0)


def test_e1_losses_fail_closed_for_invalid_supervision_or_missing_local_support() -> None:
    prediction = torch.zeros(1, 1, 3, 3, 3)
    target = torch.zeros_like(prediction)
    valid_mask = torch.ones_like(prediction, dtype=torch.bool)

    with pytest.raises(ValueError, match="dtype and device"):
        reconstruction_loss(prediction, target.double())
    with pytest.raises(ValueError, match="bool tensor"):
        reconstruction_loss(prediction, target, valid_mask.to(dtype=torch.uint8))
    with pytest.raises(ValueError, match="at least one output voxel"):
        reconstruction_loss(prediction, target, torch.zeros_like(valid_mask))
    with pytest.raises(ValueError, match="finite on valid_mask"):
        reconstruction_loss(prediction, torch.full_like(target, float("nan")), valid_mask)
    with pytest.raises(ValueError, match="positive and finite"):
        ReconstructionLossConfig(ssim_data_range=0.0)
    with pytest.raises(ValueError, match="at least 3"):
        structural_similarity_loss(prediction[..., :2, :, :], target[..., :2, :, :])

    isolated_mask = torch.zeros_like(valid_mask)
    isolated_mask[..., 1, 1, 1] = True
    with pytest.raises(ValueError, match="adjacent D/H/W"):
        gradient_agreement_loss(prediction, target, isolated_mask)
