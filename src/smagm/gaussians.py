"""Autograd-preserving tensor contract for the T0 Gaussian patient state."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
import hashlib
import json
import math

import torch


_GAUGE_FACTORY_TOKEN = object()


class AmplitudeGaugePolicy(str, Enum):
    """Named support-amplitude policy for Phase-1 runtime conversion."""

    LEGACY_RAW = "LEGACY_RAW"
    MEAN_CENTERED_LOG_AMPLITUDE_PER_PATIENT_STATE = "MEAN_CENTERED_LOG_AMPLITUDE_PER_PATIENT_STATE"


@dataclass(frozen=True)
class GaugeFixedLogAmplitude:
    """Differentiable gauge-fixed tensor and immutable provenance."""

    values: torch.Tensor
    policy: AmplitudeGaugePolicy
    config_hash: str


def _gauge_config_hash(policy: AmplitudeGaugePolicy) -> str:
    payload = json.dumps({"policy": policy.value, "version": 1}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fix_log_amplitude_gauge(
    raw_log_amplitude: torch.Tensor,
    patient_state_index: torch.Tensor | None = None,
    *,
    policy: AmplitudeGaugePolicy = AmplitudeGaugePolicy.MEAN_CENTERED_LOG_AMPLITUDE_PER_PATIENT_STATE,
) -> GaugeFixedLogAmplitude:
    """Pure, differentiable per-patient mean centering for every conversion.

    The caller owns *one* invocation per raw-to-runtime conversion.  This
    utility has no renderer state and retains gradients to raw amplitudes.
    """
    if not isinstance(raw_log_amplitude, torch.Tensor) or raw_log_amplitude.ndim != 2 or raw_log_amplitude.shape[1] != 1:
        raise ValueError("raw_log_amplitude must have shape [N, 1]")
    if raw_log_amplitude.dtype not in (torch.float32, torch.float64):
        raise TypeError("raw_log_amplitude must use float32 or float64")
    if raw_log_amplitude.shape[0] == 0 or not bool(torch.isfinite(raw_log_amplitude).all()):
        raise ValueError("raw_log_amplitude must be non-empty and finite")
    policy = AmplitudeGaugePolicy(policy)
    if policy is not AmplitudeGaugePolicy.MEAN_CENTERED_LOG_AMPLITUDE_PER_PATIENT_STATE:
        raise ValueError("new Phase-1 runtime conversion requires mean-centered log amplitude policy")
    if patient_state_index is None:
        group = torch.zeros(raw_log_amplitude.shape[0], dtype=torch.long, device=raw_log_amplitude.device)
    else:
        if not isinstance(patient_state_index, torch.Tensor) or patient_state_index.shape != (raw_log_amplitude.shape[0],):
            raise ValueError("patient_state_index must have shape [N]")
        if patient_state_index.device != raw_log_amplitude.device or patient_state_index.dtype not in (torch.int32, torch.int64):
            raise ValueError("patient_state_index must be integer and share raw_log_amplitude device")
        if bool((patient_state_index < 0).any()):
            raise ValueError("patient_state_index must be non-negative")
        group = patient_state_index.to(dtype=torch.long)
    # ``unique`` partitions only metadata; each mean remains a differentiable
    # tensor expression over the raw values of that patient state.
    centered = torch.empty_like(raw_log_amplitude)
    for state in torch.unique(group, sorted=True):
        selected = group == state
        centered[selected] = raw_log_amplitude[selected] - raw_log_amplitude[selected].mean(dim=0, keepdim=True)
    return GaugeFixedLogAmplitude(centered, policy, _gauge_config_hash(policy))


@dataclass(frozen=True)
class RawGaussianParameters:
    """Raw validated inputs whose conversion always fixes the amplitude gauge."""

    centers_ras_mm: torch.Tensor
    covariance_factor: torch.Tensor
    raw_log_support_amplitude: torch.Tensor
    appearance: torch.Tensor
    appearance_valid: torch.Tensor
    patient_state_index: torch.Tensor | None = None
    covariance_epsilon: float = 1e-8
    primitive_kind: tuple[str, ...] | None = None
    primitive_id: tuple[str, ...] | None = None


@dataclass(frozen=True)
class GaussianBatch:
    """General-SPD Gaussian tensors in canonical RAS millimetres.

    The record retains supplied tensors directly: no clone, detach, cast, or
    parameter registration occurs during validation, so upstream gradients are
    preserved.  ``covariance_factor`` is a lower-triangular factor ``L`` with
    positive diagonal and covariance ``L @ L.T + covariance_epsilon * I``.
    """

    centers_ras_mm: torch.Tensor
    covariance_factor: torch.Tensor
    log_support_amplitude: torch.Tensor
    appearance: torch.Tensor
    appearance_valid: torch.Tensor
    covariance_epsilon: float = 1e-8
    primitive_kind: tuple[str, ...] | None = None
    primitive_id: tuple[str, ...] | None = None
    gauge_policy: AmplitudeGaugePolicy = AmplitudeGaugePolicy.LEGACY_RAW
    gauge_config_hash: str | None = None
    _factory_token: InitVar[object | None] = None
    _factory_created: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        object.__setattr__(self, "_factory_created", _factory_token is _GAUGE_FACTORY_TOKEN)
        self.validate()

    def validate(self) -> None:
        """Revalidate tensors, including after optimizer or in-place updates."""

        fields = {
            "centers_ras_mm": self.centers_ras_mm,
            "covariance_factor": self.covariance_factor,
            "log_support_amplitude": self.log_support_amplitude,
            "appearance": self.appearance,
            "appearance_valid": self.appearance_valid,
        }
        if any(not isinstance(value, torch.Tensor) for value in fields.values()):
            raise TypeError("GaussianBatch fields must be torch.Tensor instances")
        if self.centers_ras_mm.ndim != 2 or self.centers_ras_mm.shape[1] != 3:
            raise ValueError("centers_ras_mm must have shape [N, 3]")
        count = self.centers_ras_mm.shape[0]
        if count == 0:
            raise ValueError("GaussianBatch must contain at least one Gaussian")
        if self.covariance_factor.shape != (count, 3, 3):
            raise ValueError("covariance_factor must have shape [N, 3, 3]")
        if self.log_support_amplitude.shape != (count, 1):
            raise ValueError("log_support_amplitude must have shape [N, 1]")
        if self.appearance.ndim != 2 or self.appearance.shape[0] != count or self.appearance.shape[1] == 0:
            raise ValueError("appearance must have shape [N, M] with M > 0")
        if self.appearance_valid.shape != self.appearance.shape or self.appearance_valid.dtype is not torch.bool:
            raise ValueError("appearance_valid must be bool with shape [N, M]")
        device = self.centers_ras_mm.device
        dtype = self.centers_ras_mm.dtype
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("the CPU reference requires float32 or float64 tensors")
        for name, value in fields.items():
            if value.device != device:
                raise ValueError(f"{name} must share centers_ras_mm device")
        for name in ("covariance_factor", "log_support_amplitude", "appearance"):
            if fields[name].dtype != dtype:
                raise ValueError(f"{name} must share centers_ras_mm dtype")
        for name, value in fields.items():
            if value.dtype is not torch.bool and not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
        finfo = torch.finfo(dtype)
        maximum_safe_coordinate = math.sqrt(finfo.max) / 4.0
        if not bool(torch.all(self.centers_ras_mm.abs() <= maximum_safe_coordinate)):
            raise ValueError("centers_ras_mm would overflow renderer distance arithmetic")
        maximum_safe_factor = math.sqrt(finfo.max / 6.0)
        if not bool(torch.all(self.covariance_factor.abs() <= maximum_safe_factor)):
            raise ValueError("covariance_factor would overflow covariance accumulation")
        maximum_safe_log = 0.5 * (math.log(finfo.max) - math.log(count)) - 2.0
        if not bool(torch.all(self.log_support_amplitude <= maximum_safe_log)):
            raise ValueError("log_support_amplitude would overflow renderer accumulation")
        maximum_safe_appearance = math.sqrt(finfo.max / count)
        if not bool(torch.all(self.appearance.abs() <= maximum_safe_appearance)):
            raise ValueError("appearance would overflow renderer accumulation")
        if not bool(torch.all(self.covariance_factor.triu(diagonal=1) == 0)):
            raise ValueError("covariance_factor must be lower triangular")
        if not bool(torch.all(torch.diagonal(self.covariance_factor, dim1=-2, dim2=-1) > 0)):
            raise ValueError("covariance_factor diagonal must be strictly positive")
        if (
            isinstance(self.covariance_epsilon, bool)
            or not isinstance(self.covariance_epsilon, (int, float))
            or not math.isfinite(float(self.covariance_epsilon))
            or self.covariance_epsilon < finfo.tiny
            or self.covariance_epsilon > finfo.max / 2.0
        ):
            raise ValueError("covariance_epsilon must be positive, finite, and dtype-representable")
        self._validate_ids("primitive_kind", self.primitive_kind, count)
        self._validate_ids("primitive_id", self.primitive_id, count)
        policy = AmplitudeGaugePolicy(self.gauge_policy)
        if policy is AmplitudeGaugePolicy.LEGACY_RAW:
            if self.gauge_config_hash is not None:
                raise ValueError("legacy GaussianBatch cannot claim Phase-1 gauge provenance")
        elif self.gauge_config_hash != _gauge_config_hash(policy) or not self._factory_created:
            raise ValueError("Phase-1 GaussianBatch requires matching gauge provenance")
        object.__setattr__(self, "gauge_policy", policy)
        if self.primitive_kind is not None:
            object.__setattr__(self, "primitive_kind", tuple(self.primitive_kind))
        if self.primitive_id is not None:
            object.__setattr__(self, "primitive_id", tuple(self.primitive_id))

    @staticmethod
    def _validate_ids(name: str, values: tuple[str, ...] | None, count: int) -> None:
        if values is not None and (len(values) != count or any(not isinstance(value, str) or not value for value in values)):
            raise ValueError(f"{name}, when supplied, must contain N non-empty strings")

    @property
    def count(self) -> int:
        return self.centers_ras_mm.shape[0]

    @property
    def appearance_channels(self) -> int:
        return self.appearance.shape[1]

    def covariance(self) -> torch.Tensor:
        identity = torch.eye(3, dtype=self.centers_ras_mm.dtype, device=self.centers_ras_mm.device)
        return self.covariance_factor @ self.covariance_factor.transpose(-1, -2) + self.covariance_epsilon * identity

    @classmethod
    def from_raw(cls, raw: RawGaussianParameters) -> "GaussianBatch":
        return gaussian_batch_from_raw(raw)


def gaussian_batch_from_raw(raw: RawGaussianParameters) -> GaussianBatch:
    """Canonical Phase-1 raw-parameter conversion with exactly one gauge fix."""
    if not isinstance(raw, RawGaussianParameters):
        raise TypeError("raw must be RawGaussianParameters")
    fixed = fix_log_amplitude_gauge(raw.raw_log_support_amplitude, raw.patient_state_index)
    return GaussianBatch(
        centers_ras_mm=raw.centers_ras_mm,
        covariance_factor=raw.covariance_factor,
        log_support_amplitude=fixed.values,
        appearance=raw.appearance,
        appearance_valid=raw.appearance_valid,
        covariance_epsilon=raw.covariance_epsilon,
        primitive_kind=raw.primitive_kind,
        primitive_id=raw.primitive_id,
        gauge_policy=fixed.policy,
        gauge_config_hash=fixed.config_hash,
        _factory_token=_GAUGE_FACTORY_TOKEN,
    )
