"""Compatibility import path for the T0 Gaussian tensor contract."""

from ..gaussians import (
    AmplitudeGaugePolicy,
    GaugeFixedLogAmplitude,
    GaussianBatch,
    RawGaussianParameters,
    fix_log_amplitude_gauge,
    gaussian_batch_from_raw,
)

__all__ = [
    "AmplitudeGaugePolicy",
    "GaugeFixedLogAmplitude",
    "GaussianBatch",
    "RawGaussianParameters",
    "fix_log_amplitude_gauge",
    "gaussian_batch_from_raw",
]
