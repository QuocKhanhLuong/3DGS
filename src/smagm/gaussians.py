"""Autograd-preserving tensor contract for the T0 Gaussian patient state."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


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

    def __post_init__(self) -> None:
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
