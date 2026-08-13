from __future__ import annotations

import math

import pytest
import torch

from smagm.features.point_guided.point_guided_metrics import compute_point_guided_metrics


def test_masked_metrics_are_finite_and_ignore_invalid_voxels() -> None:
    prediction = torch.tensor([[[[1.0, 3.0], [100.0, 5.0]]]])
    target = torch.tensor([[[[1.0, 1.0], [200.0, 7.0]]]])
    valid_mask = torch.tensor([[[[True, True], [False, True]]]])
    metrics = compute_point_guided_metrics(prediction, target, valid_mask)
    assert metrics.mae == pytest.approx(4.0 / 3.0)
    assert metrics.mse == pytest.approx(8.0 / 3.0)
    assert metrics.nmse == pytest.approx((8.0 / 3.0) / 17.0)
    assert math.isfinite(metrics.psnr)
    assert 0.0 <= metrics.ssim <= 1.0


def test_exact_reconstruction_has_expected_limits() -> None:
    target = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2)
    metrics = compute_point_guided_metrics(target, target)
    assert metrics.mae == 0.0
    assert metrics.mse == 0.0
    assert metrics.nmse == 0.0
    assert math.isinf(metrics.psnr)
    assert metrics.ssim == pytest.approx(1.0)


def test_metric_validation_is_fail_closed() -> None:
    prediction = torch.zeros(2, 2)
    target = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="valid_mask"):
        compute_point_guided_metrics(prediction, target, torch.zeros(2, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="finite"):
        compute_point_guided_metrics(torch.tensor([[float("nan")]]), torch.zeros(1, 1))
