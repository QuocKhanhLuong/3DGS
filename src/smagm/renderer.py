"""Pure-PyTorch physical-plane Gaussian MRI reference renderer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Literal, Sequence

import torch

from .contracts.coordinates import PhysicalPlane
from .gaussians import GaussianBatch


RENDERER_OUTPUT_SCHEMA_VERSION = "render-result-v1"


@dataclass(frozen=True)
class SlabProfile:
    """Normalized through-plane quadrature positions in units of thickness."""

    normalized_offsets: tuple[float, ...]
    weights: tuple[float, ...]
    kind: Literal["delta", "box", "discrete"] = "discrete"

    def __post_init__(self) -> None:
        if not self.normalized_offsets or len(self.normalized_offsets) != len(self.weights):
            raise ValueError("profile offsets and weights must be non-empty and same length")
        offsets = tuple(float(value) for value in self.normalized_offsets)
        weights = tuple(float(value) for value in self.weights)
        if not all(math.isfinite(value) for value in offsets + weights):
            raise ValueError("profile offsets and weights must be finite")
        if any(abs(offset) > 0.5 for offset in offsets):
            raise ValueError("normalized profile offsets must lie within [-0.5, 0.5]")
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ValueError("profile weights must be non-negative with positive mass")
        total = sum(weights)
        if not math.isfinite(total):
            raise ValueError("profile weight mass must be finite")
        object.__setattr__(self, "normalized_offsets", offsets)
        object.__setattr__(self, "weights", tuple(weight / total for weight in weights))

    @classmethod
    def delta(cls) -> "SlabProfile":
        return cls((0.0,), (1.0,), "delta")

    @classmethod
    def box(cls, samples: int) -> "SlabProfile":
        if not isinstance(samples, int) or samples <= 0:
            raise ValueError("box samples must be a positive integer")
        return cls(tuple((index + 0.5) / samples - 0.5 for index in range(samples)), (1.0,) * samples, "box")

    @classmethod
    def discrete(cls, normalized_offsets: Sequence[float], weights: Sequence[float]) -> "SlabProfile":
        return cls(tuple(normalized_offsets), tuple(weights), "discrete")


@dataclass(frozen=True)
class RenderConfig:
    """Controls for the through-plane profile-aware Gaussian reference renderer."""

    support_epsilon: float = 1e-8
    pixel_chunk_size: int | None = None
    gaussian_chunk_size: int | None = None
    profile: SlabProfile = SlabProfile.delta()
    minimum_supported_psf_mass: float = 0.999

    def __post_init__(self) -> None:
        if (
            isinstance(self.support_epsilon, bool)
            or not isinstance(self.support_epsilon, (int, float))
            or not math.isfinite(float(self.support_epsilon))
            or self.support_epsilon <= 0.0
        ):
            raise ValueError("support_epsilon must be positive and finite")
        for name, value in (("pixel_chunk_size", self.pixel_chunk_size), ("gaussian_chunk_size", self.gaussian_chunk_size)):
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be None or a positive integer")
        if not isinstance(self.profile, SlabProfile):
            raise TypeError("profile must be a SlabProfile")
        if (
            isinstance(self.minimum_supported_psf_mass, bool)
            or not isinstance(self.minimum_supported_psf_mass, (int, float))
            or not math.isfinite(float(self.minimum_supported_psf_mass))
            or not 0.0 < self.minimum_supported_psf_mass <= 1.0
        ):
            raise ValueError("minimum_supported_psf_mass must lie in (0, 1]")

    @property
    def renderer_version(self) -> str:
        """Controlled implementation/config identifier, never caller supplied.

        Chunk sizes are deliberately included: they are expected to be
        numerically equivalent but remain part of a reproducible render record.
        """
        payload = {
            "gaussian_chunk_size": self.gaussian_chunk_size,
            "implementation": "through-plane-profile-aware-gaussian-reference-renderer/v1",
            "minimum_supported_psf_mass": self.minimum_supported_psf_mass,
            "pixel_chunk_size": self.pixel_chunk_size,
            "profile": {
                "kind": self.profile.kind,
                "normalized_offsets": self.profile.normalized_offsets,
                "weights": self.profile.weights,
            },
            "support_epsilon": self.support_epsilon,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return f"through-plane-profile-aware-gaussian-reference-renderer/v1:{digest}"

    @property
    def renderer_output_schema_version(self) -> str:
        """Version of the detached `RenderResult` digest envelope."""
        return RENDERER_OUTPUT_SCHEMA_VERSION


@dataclass(frozen=True)
class RenderResult:
    """A selected-modality image, support mass, and explicit unsupported mask."""

    intensity: torch.Tensor  # [H, W]
    support_mass: torch.Tensor  # [H, W]
    supported_psf_mass: torch.Tensor  # [H, W], range [0, 1]
    unsupported_mask: torch.Tensor  # [H, W], bool


def _plane_points(plane: PhysicalPlane, gaussian: GaussianBatch) -> torch.Tensor:
    dtype, device = gaussian.centers_ras_mm.dtype, gaussian.centers_ras_mm.device
    height, width = plane.shape_hw
    v, u = torch.meshgrid(torch.arange(height, dtype=dtype, device=device), torch.arange(width, dtype=dtype, device=device), indexing="ij")
    origin = torch.tensor(plane.pixel_center_origin_ras_mm, dtype=dtype, device=device)
    axis_u = torch.tensor(plane.axis_u_ras, dtype=dtype, device=device)
    axis_v = torch.tensor(plane.axis_v_ras, dtype=dtype, device=device)
    return origin + (u * plane.spacing_uv_mm[0]).unsqueeze(-1) * axis_u + (v * plane.spacing_uv_mm[1]).unsqueeze(-1) * axis_v


def _gaussian_terms(points: torch.Tensor, gaussian: GaussianBatch, appearance_channel: int, gaussian_chunk_size: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return additive numerator and support without explicitly inverting Sigma."""
    numerator = points.new_zeros((points.shape[0],))
    support = points.new_zeros((points.shape[0],))
    step = gaussian_chunk_size or gaussian.count
    for start in range(0, gaussian.count, step):
        stop = min(start + step, gaussian.count)
        centers = gaussian.centers_ras_mm[start:stop]
        # L is the stored parameterization, but the contract covariance is
        # L L^T + epsilon I.  Cholesky retains a triangular solve in the hot
        # quadratic-form path while honoring the named numerical policy.
        factor = torch.linalg.cholesky(gaussian.covariance()[start:stop])
        difference = points[:, None, :] - centers[None, :, :]
        rhs = difference.permute(1, 2, 0)  # [G, 3, P]
        solved = torch.linalg.solve_triangular(factor, rhs, upper=False)
        exponent = -0.5 * solved.square().sum(dim=1).transpose(0, 1)
        density = torch.exp(gaussian.log_support_amplitude[start:stop, 0])[None, :] * torch.exp(exponent)
        valid = gaussian.appearance_valid[start:stop, appearance_channel].to(dtype=density.dtype)[None, :]
        weighted_density = density * valid
        support = support + weighted_density.sum(dim=1)
        numerator = numerator + (weighted_density * gaussian.appearance[start:stop, appearance_channel][None, :]).sum(dim=1)
    return numerator, support


def render_plane(gaussians: GaussianBatch, plane: PhysicalPlane, *, appearance_channel: int = 0, config: RenderConfig | None = None) -> RenderResult:
    """Render a normalized additive MRI image for one physical plane.

    A delta profile is the closed-form thin-plane evaluation. Other profiles
    evaluate the normalized latent field ``N(x) / S(x)`` at every quadrature
    depth and then apply normalized PSF weights to those intensities. Support
    mass is reported as the same weighted average of depth-wise support.
    ``supported_psf_mass`` reports the PSF weight with numerically supported
    samples; supported samples are renormalized and the configured coverage
    threshold determines the explicit unsupported mask. Chunk boundaries do not
    alter the operator and remain inside the differentiable tensor path;
    manifest/selection operations are outside it.  The function is deliberately
    side-effect free: it neither knows about episode ledgers nor issues
    prediction receipts.
    """
    if not isinstance(gaussians, GaussianBatch) or not isinstance(plane, PhysicalPlane):
        raise TypeError("gaussians and plane must use their T0 contract types")
    gaussians.validate()
    if not 0 <= appearance_channel < gaussians.appearance_channels:
        raise IndexError("appearance_channel is outside GaussianBatch appearance channels")
    config = config or RenderConfig()
    finfo = torch.finfo(gaussians.centers_ras_mm.dtype)
    if not finfo.tiny <= config.support_epsilon <= finfo.max:
        raise ValueError("support_epsilon must be representable in the renderer dtype")
    points = _plane_points(plane, gaussians).reshape(-1, 3)
    if not bool(torch.isfinite(points).all()):
        raise ValueError("plane coordinates are not finite in the renderer dtype")
    normal = torch.tensor(plane.signed_normal_ras, dtype=points.dtype, device=points.device)
    pixel_step = config.pixel_chunk_size or points.shape[0]
    intensity_chunks: list[torch.Tensor] = []
    support_chunks: list[torch.Tensor] = []
    supported_psf_chunks: list[torch.Tensor] = []
    for start in range(0, points.shape[0], pixel_step):
        stop = min(start + pixel_step, points.shape[0])
        chunk_points = points[start:stop]
        chunk_intensity = chunk_points.new_zeros((chunk_points.shape[0],))
        chunk_support = chunk_points.new_zeros((chunk_points.shape[0],))
        chunk_supported_psf = chunk_points.new_zeros((chunk_points.shape[0],))
        for offset, weight in zip(config.profile.normalized_offsets, config.profile.weights):
            shifted = chunk_points + (float(offset) * plane.thickness_mm) * normal
            local_numerator, local_support = _gaussian_terms(shifted, gaussians, appearance_channel, config.gaussian_chunk_size)
            local_unsupported = local_support <= config.support_epsilon
            local_intensity = local_numerator / local_support.clamp_min(config.support_epsilon)
            local_supported = ~local_unsupported
            chunk_intensity = chunk_intensity + float(weight) * torch.where(
                local_supported,
                local_intensity,
                torch.zeros_like(local_intensity),
            )
            chunk_support = chunk_support + float(weight) * local_support
            chunk_supported_psf = (
                chunk_supported_psf
                + float(weight) * local_supported.to(dtype=chunk_points.dtype)
            )
        chunk_intensity = chunk_intensity / chunk_supported_psf.clamp_min(
            torch.finfo(chunk_points.dtype).eps
        )
        intensity_chunks.append(chunk_intensity)
        support_chunks.append(chunk_support)
        supported_psf_chunks.append(chunk_supported_psf)
    intensity = torch.cat(intensity_chunks)
    support = torch.cat(support_chunks)
    supported_psf_mass = torch.cat(supported_psf_chunks)
    unsupported = supported_psf_mass < config.minimum_supported_psf_mass
    intensity = torch.where(unsupported, torch.full_like(intensity, float("nan")), intensity)
    height, width = plane.shape_hw
    return RenderResult(
        intensity.reshape(height, width),
        support.reshape(height, width),
        supported_psf_mass.reshape(height, width),
        unsupported.reshape(height, width),
    )
