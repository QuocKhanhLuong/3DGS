"""Safe feature-to-Gaussian bridge for the executable T1-A reference.

This module deliberately has no support-to-support communication, learned birth,
pruning, splitting, merging, or propagation.  E0/E1/E2 can therefore use the
same topology and bridge architecture while differing only in their evidence
encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from ..gaussians import GaussianBatch, RawGaussianParameters, gaussian_batch_from_raw
from .fixed_support import FixedSupportBatch


@dataclass(frozen=True)
class FixedGaussianHeadConfig:
    input_dim: int
    appearance_channels: int = 1
    hidden_dim: int = 32
    max_center_offset_mm: float | tuple[float, float, float] = 0.5
    min_scale_mm: float = 0.5
    max_scale_mm: float = 6.0
    max_off_diagonal_mm: float = 1.0
    use_reliability_amplitude: bool = True

    def __post_init__(self) -> None:
        for name in ("input_dim", "appearance_channels", "hidden_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("min_scale_mm",
            "max_scale_mm",
            "max_off_diagonal_mm",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        offsets = _center_offset_limits(self.max_center_offset_mm)
        if any(value < 0.0 for value in offsets) or self.min_scale_mm <= 0 or self.max_scale_mm <= self.min_scale_mm:
            raise ValueError("centre and scale bounds are invalid")
        if self.max_off_diagonal_mm < 0:
            raise ValueError("max_off_diagonal_mm must be non-negative")


def _center_offset_limits(value: float | tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(value, bool):
        raise ValueError("max_center_offset_mm must be finite")
    if isinstance(value, (int, float)):
        limits = (float(value),) * 3
    else:
        limits = tuple(float(component) for component in value)
    if len(limits) != 3 or any(not math.isfinite(component) for component in limits):
        raise ValueError("max_center_offset_mm must be a finite scalar or three finite local RAS-mm limits")
    return limits


@dataclass(frozen=True)
class RawFixedGaussianOutput:
    center_offset_raw: torch.Tensor  # [N, 3]
    covariance_raw: torch.Tensor  # [N, 6]: d0,l10,d1,l20,l21,d2
    log_amplitude_raw: torch.Tensor  # [N, 1]
    appearance_raw: torch.Tensor  # [N, M]

    def __post_init__(self) -> None:
        if self.center_offset_raw.ndim != 2 or self.center_offset_raw.shape[1] != 3:
            raise ValueError("center_offset_raw must have shape [N, 3]")
        count = self.center_offset_raw.shape[0]
        if self.covariance_raw.shape != (count, 6):
            raise ValueError("covariance_raw must have shape [N, 6]")
        if self.log_amplitude_raw.shape != (count, 1):
            raise ValueError("log_amplitude_raw must have shape [N, 1]")
        if self.appearance_raw.ndim != 2 or self.appearance_raw.shape[0] != count or self.appearance_raw.shape[1] <= 0:
            raise ValueError("appearance_raw must have shape [N, M]")
        device = self.center_offset_raw.device
        dtype = self.center_offset_raw.dtype
        for tensor in (self.covariance_raw, self.log_amplitude_raw, self.appearance_raw):
            if tensor.device != device or tensor.dtype != dtype:
                raise ValueError("raw Gaussian outputs must share device and dtype")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("raw Gaussian outputs must be finite")


class FixedGaussianHead(nn.Module):
    """Independent per-support MLP used identically across T1 variants."""

    def __init__(self, config: FixedGaussianHeadConfig) -> None:
        super().__init__()
        self.config = config
        output_dim = 3 + 6 + 1 + config.appearance_channels
        self.network = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, output_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        # Start near the middle of the declared scale interval.  The raw values
        # remain unconstrained; conversion applies the named bounded mapping.
        proportion = 0.5
        diagonal_bias = math.log(proportion / (1.0 - proportion))
        with torch.no_grad():
            covariance_start = 3
            for index in (0, 2, 5):
                final.bias[covariance_start + index] = diagonal_bias

    def forward(self, feature_vectors: torch.Tensor) -> RawFixedGaussianOutput:
        if not isinstance(feature_vectors, torch.Tensor) or feature_vectors.ndim != 2:
            raise ValueError("feature_vectors must have shape [N, C]")
        if feature_vectors.shape[1] != self.config.input_dim:
            raise ValueError("feature_vectors channel count disagrees with head config")
        if feature_vectors.dtype not in (torch.float32, torch.float64):
            raise TypeError("feature_vectors must use float32 or float64")
        raw = self.network(feature_vectors)
        cursor = 0
        center = raw[:, cursor : cursor + 3]
        cursor += 3
        covariance = raw[:, cursor : cursor + 6]
        cursor += 6
        amplitude = raw[:, cursor : cursor + 1]
        cursor += 1
        appearance = raw[:, cursor : cursor + self.config.appearance_channels]
        return RawFixedGaussianOutput(center, covariance, amplitude, appearance)


def _safe_covariance_factor(raw: torch.Tensor, config: FixedGaussianHeadConfig) -> torch.Tensor:
    count = raw.shape[0]
    factor = raw.new_zeros((count, 3, 3))
    diagonal_raw = raw[:, (0, 2, 5)]
    diagonal = config.min_scale_mm + (config.max_scale_mm - config.min_scale_mm) * torch.sigmoid(diagonal_raw)
    off_diagonal = config.max_off_diagonal_mm * torch.tanh(raw[:, (1, 3, 4)])
    factor[:, 0, 0] = diagonal[:, 0]
    factor[:, 1, 0] = off_diagonal[:, 0]
    factor[:, 1, 1] = diagonal[:, 1]
    factor[:, 2, 0] = off_diagonal[:, 1]
    factor[:, 2, 1] = off_diagonal[:, 2]
    factor[:, 2, 2] = diagonal[:, 2]
    return factor


def construct_fixed_gaussians(
    supports: FixedSupportBatch,
    raw_output: RawFixedGaussianOutput,
    *,
    config: FixedGaussianHeadConfig,
) -> GaussianBatch:
    """Convert one fixed support set into a gauge-safe runtime GaussianBatch."""

    if not isinstance(supports, FixedSupportBatch) or not isinstance(raw_output, RawFixedGaussianOutput):
        raise TypeError("supports and raw_output must use T1-A contract types")
    if supports.count != raw_output.center_offset_raw.shape[0]:
        raise ValueError("support count and raw output count disagree")
    if raw_output.appearance_raw.shape[1] != config.appearance_channels:
        raise ValueError("appearance channels disagree with head config")
    local_limits = raw_output.center_offset_raw.new_tensor(_center_offset_limits(config.max_center_offset_mm))
    local_offsets = local_limits * torch.tanh(raw_output.center_offset_raw)
    world_offsets = torch.einsum("ni,nij->nj", local_offsets, supports.support_basis_ras)
    centers = supports.centers_ras_mm + world_offsets
    factor = _safe_covariance_factor(raw_output.covariance_raw, config)
    log_amplitude = raw_output.log_amplitude_raw
    if config.use_reliability_amplitude:
        log_amplitude = log_amplitude + torch.log(supports.reliability.clamp_min(1e-4))
    appearance = torch.tanh(raw_output.appearance_raw)
    appearance_valid = torch.ones_like(appearance, dtype=torch.bool)
    primitive_ids = tuple(
        f"fixed:{plane_hash}:{observation_id}:{int(v)}:{int(u)}"
        for plane_hash, observation_id, (v, u) in zip(
            supports.source_plane_hashes, supports.observation_ids, supports.feature_indices_vu.tolist()
        )
    )
    patient_state_index = torch.zeros(supports.count, dtype=torch.long, device=centers.device)
    return gaussian_batch_from_raw(
        RawGaussianParameters(
            centers_ras_mm=centers,
            covariance_factor=factor,
            raw_log_support_amplitude=log_amplitude,
            appearance=appearance,
            appearance_valid=appearance_valid,
            patient_state_index=patient_state_index,
            primitive_kind=("fixed_support",) * supports.count,
            primitive_id=primitive_ids,
        )
    )
