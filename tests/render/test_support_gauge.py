"""Gauge-fixing regressions: runtime conversion is invariant, render stays pure."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from smagm.contracts.coordinates import PhysicalPlane
from smagm.contracts.episode import prediction_digest_from_render_result
from smagm.gaussians import (
    AmplitudeGaugePolicy,
    GaussianBatch,
    RawGaussianParameters,
    fix_log_amplitude_gauge,
    gaussian_batch_from_raw,
)
from smagm.renderer import RenderConfig, RenderResult, render_plane


def _raw(dtype: torch.dtype, *, shift: float = 0.0, groups: torch.Tensor | None = None, requires_grad: bool = False) -> RawGaussianParameters:
    count = 4
    return RawGaussianParameters(
        centers_ras_mm=torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [0.0, 0.7, 0.0], [0.7, 0.7, 0.0]], dtype=dtype),
        covariance_factor=torch.eye(3, dtype=dtype).expand(count, -1, -1).clone(),
        raw_log_support_amplitude=torch.tensor([[-2.0], [-1.0], [0.0], [1.0]], dtype=dtype).add(shift).requires_grad_(requires_grad),
        appearance=torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=dtype, requires_grad=requires_grad),
        appearance_valid=torch.ones((count, 1), dtype=torch.bool),
        patient_state_index=groups,
    )


def _plane() -> PhysicalPlane:
    return PhysicalPlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.5, 0.5), 1.0, (3, 3), (0.0, 0.0, 1.0))


@pytest.mark.parametrize("dtype,atol", [(torch.float32, 2e-6), (torch.float64, 1e-12)])
@pytest.mark.parametrize("shift", [-17.5, 23.75])
def test_common_log_amplitude_shifts_have_identical_runtime_render_and_support(dtype, atol, shift) -> None:
    baseline = gaussian_batch_from_raw(_raw(dtype))
    shifted = gaussian_batch_from_raw(_raw(dtype, shift=shift))
    assert baseline.gauge_policy is AmplitudeGaugePolicy.MEAN_CENTERED_LOG_AMPLITUDE_PER_PATIENT_STATE
    assert torch.allclose(baseline.log_support_amplitude, shifted.log_support_amplitude, atol=atol, rtol=0)
    baseline_render = render_plane(baseline, _plane())
    shifted_render = render_plane(shifted, _plane())
    assert torch.allclose(baseline_render.intensity, shifted_render.intensity, atol=atol, rtol=0, equal_nan=True)
    assert torch.allclose(baseline_render.support_mass, shifted_render.support_mass, atol=atol, rtol=0)
    assert torch.equal(baseline_render.unsupported_mask, shifted_render.unsupported_mask)


def test_gauge_groups_are_centered_independently_and_invalid_indices_or_shapes_fail() -> None:
    amplitude = torch.tensor([[-2.0], [2.0], [11.0], [15.0]], dtype=torch.float64)
    groups = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    fixed = fix_log_amplitude_gauge(amplitude, groups)
    assert torch.allclose(fixed.values, torch.tensor([[-2.0], [2.0], [-2.0], [2.0]], dtype=torch.float64))
    assert torch.allclose(fixed.values[groups == 0].mean(), torch.zeros((), dtype=torch.float64))
    assert torch.allclose(fixed.values[groups == 1].mean(), torch.zeros((), dtype=torch.float64))
    singleton = fix_log_amplitude_gauge(amplitude, torch.tensor([0, 0, 1, 2], dtype=torch.int64))
    assert torch.equal(singleton.values, torch.tensor([[-2.0], [2.0], [0.0], [0.0]], dtype=torch.float64))
    with pytest.raises(ValueError):
        fix_log_amplitude_gauge(amplitude, torch.tensor([0, -1, 1, 1], dtype=torch.int64))
    with pytest.raises(ValueError):
        fix_log_amplitude_gauge(amplitude, torch.tensor([0, 0, 1, 1], dtype=torch.int32).reshape(4, 1))


def test_gauge_conversion_has_finite_float64_gradients() -> None:
    raw = _raw(torch.float64, requires_grad=True)
    batch = gaussian_batch_from_raw(raw)
    result = render_plane(batch, _plane())
    torch.nan_to_num(result.intensity, nan=0.0).sum().backward()
    assert raw.raw_log_support_amplitude.grad is not None
    assert raw.appearance.grad is not None
    assert torch.isfinite(raw.raw_log_support_amplitude.grad).all()
    assert torch.isfinite(raw.appearance.grad).all()


def test_every_raw_to_runtime_conversion_calls_gauge_once_and_render_calls_it_never(monkeypatch) -> None:
    import smagm.gaussians as gaussians

    calls = 0
    original = gaussians.fix_log_amplitude_gauge

    def instrumented(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(gaussians, "fix_log_amplitude_gauge", instrumented)
    raw = _raw(torch.float64)
    batch_a = gaussians.gaussian_batch_from_raw(raw)
    batch_b = GaussianBatch.from_raw(raw)
    assert calls == 2
    render_plane(batch_a, _plane())
    render_plane(batch_b, _plane())
    assert calls == 2


def test_legacy_direct_batch_remains_compatible_without_runtime_gauge() -> None:
    batch = GaussianBatch(
        centers_ras_mm=torch.zeros((2, 3), dtype=torch.float64),
        covariance_factor=torch.eye(3, dtype=torch.float64).expand(2, -1, -1).clone(),
        log_support_amplitude=torch.tensor([[2.0], [3.0]], dtype=torch.float64),
        appearance=torch.ones((2, 1), dtype=torch.float64),
        appearance_valid=torch.ones((2, 1), dtype=torch.bool),
    )
    assert batch.gauge_policy is AmplitudeGaugePolicy.LEGACY_RAW
    assert render_plane(batch, _plane()).intensity.shape == (3, 3)


def test_legacy_raw_shift_positive_control_changes_support_classification() -> None:
    def legacy(log_amplitude: float) -> GaussianBatch:
        return GaussianBatch(
            centers_ras_mm=torch.zeros((1, 3), dtype=torch.float64),
            covariance_factor=torch.eye(3, dtype=torch.float64).unsqueeze(0),
            log_support_amplitude=torch.tensor([[log_amplitude]], dtype=torch.float64),
            appearance=torch.ones((1, 1), dtype=torch.float64),
            appearance_valid=torch.ones((1, 1), dtype=torch.bool),
        )

    plane = PhysicalPlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0), 1.0, (1, 1), (0.0, 0.0, 1.0))
    config = RenderConfig(support_epsilon=1e-8)
    low = render_plane(legacy(-30.0), plane, config=config)
    high = render_plane(legacy(5.0), plane, config=config)
    assert low.unsupported_mask.item() is True
    assert high.unsupported_mask.item() is False


def test_hand_built_mean_centered_batch_cannot_forge_raw_conversion_provenance() -> None:
    converted = gaussian_batch_from_raw(_raw(torch.float64))
    with pytest.raises((PermissionError, TypeError, ValueError), match="provenance|factory|raw"):
        GaussianBatch(
            centers_ras_mm=converted.centers_ras_mm,
            covariance_factor=converted.covariance_factor,
            log_support_amplitude=converted.log_support_amplitude,
            appearance=converted.appearance,
            appearance_valid=converted.appearance_valid,
            gauge_policy=AmplitudeGaugePolicy.MEAN_CENTERED_LOG_AMPLITUDE_PER_PATIENT_STATE,
            gauge_config_hash=converted.gauge_config_hash,
        )


def test_digest_canonicalizes_quiet_nan_payloads_and_binds_all_output_fields() -> None:
    # Different IEEE quiet-NaN payloads must not make two identical unsupported
    # predictions hash differently.  Mask/support/PSF remain schema-bound.
    nan_a = torch.from_numpy(np.array([[0x7FF8000000000001]], dtype=np.uint64).view(np.float64))
    nan_b = torch.from_numpy(np.array([[0x7FF80000000000A5]], dtype=np.uint64).view(np.float64))
    common = dict(
        support_mass=torch.tensor([[0.0]], dtype=torch.float64),
        supported_psf_mass=torch.tensor([[0.0]], dtype=torch.float64),
        unsupported_mask=torch.tensor([[True]], dtype=torch.bool),
    )
    first = RenderResult(intensity=nan_a, **common)
    second = RenderResult(intensity=nan_b, **common)
    plane_hash = hashlib.sha256(b"plane").hexdigest()
    version = RenderConfig().renderer_version
    assert prediction_digest_from_render_result(first, plane_hash=plane_hash, renderer_version=version) == prediction_digest_from_render_result(second, plane_hash=plane_hash, renderer_version=version)
    changed_schema_value = RenderResult(
        intensity=nan_a,
        support_mass=torch.tensor([[1.0]], dtype=torch.float64),
        supported_psf_mass=common["supported_psf_mass"],
        unsupported_mask=common["unsupported_mask"],
    )
    assert prediction_digest_from_render_result(first, plane_hash=plane_hash, renderer_version=version) != prediction_digest_from_render_result(changed_schema_value, plane_hash=plane_hash, renderer_version=version)
