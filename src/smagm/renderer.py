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
    tile_shape_hw: tuple[int, int] = (32, 32)

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
        if not isinstance(self.tile_shape_hw, Sequence) or isinstance(self.tile_shape_hw, (str, bytes)):
            raise ValueError("tile_shape_hw must contain two positive integers")
        tile_shape = tuple(self.tile_shape_hw)
        if (
            len(tile_shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in tile_shape)
        ):
            raise ValueError("tile_shape_hw must contain two positive integers")
        if not isinstance(self.profile, SlabProfile):
            raise TypeError("profile must be a SlabProfile")
        if (
            isinstance(self.minimum_supported_psf_mass, bool)
            or not isinstance(self.minimum_supported_psf_mass, (int, float))
            or not math.isfinite(float(self.minimum_supported_psf_mass))
            or not 0.0 < self.minimum_supported_psf_mass <= 1.0
        ):
            raise ValueError("minimum_supported_psf_mass must lie in (0, 1]")
        object.__setattr__(self, "tile_shape_hw", tile_shape)

    @property
    def renderer_version(self) -> str:
        """Controlled implementation/config identifier, never caller supplied.

        Chunk sizes are deliberately included: they are expected to be
        numerically equivalent but remain part of a reproducible render record.
        """
        payload = {
            "gaussian_chunk_size": self.gaussian_chunk_size,
            "implementation": "through-plane-profile-aware-gaussian-reference-renderer/v2",
            "minimum_supported_psf_mass": self.minimum_supported_psf_mass,
            "pixel_chunk_size": self.pixel_chunk_size,
            "profile": {
                "kind": self.profile.kind,
                "normalized_offsets": self.profile.normalized_offsets,
                "weights": self.profile.weights,
            },
            "support_epsilon": self.support_epsilon,
            "tile_shape_hw": self.tile_shape_hw,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return f"through-plane-profile-aware-gaussian-reference-renderer/v2:{digest}"

    @property
    def renderer_output_schema_version(self) -> str:
        """Version of the detached `RenderResult` digest envelope."""
        return RENDERER_OUTPUT_SCHEMA_VERSION


@dataclass(frozen=True)
class RenderResult:
    """A selected-modality image, support mass, and explicit unsupported mask.

    ``pixel_gaussian_candidate_pairs`` is a non-semantic telemetry diagnostic:
    the number of plane-pixel/Gaussian pairs retained after culling, before
    through-plane quadrature.  It is deliberately excluded from prediction
    digests and defaults to zero so manual construction of the historical four
    result tensors remains source compatible.
    """

    intensity: torch.Tensor  # [H, W]
    support_mass: torch.Tensor  # [H, W]
    supported_psf_mass: torch.Tensor  # [H, W], range [0, 1]
    unsupported_mask: torch.Tensor  # [H, W], bool
    pixel_gaussian_candidate_pairs: int = 0


@dataclass(frozen=True)
class _PreparedGaussianTerms:
    """Per-render Gaussian values and conservative plane bounds.

    Continuous tensors remain attached to autograd.  The bounds are detached
    because they control a discrete, conservative work-selection operation;
    selected terms still retain their exact differentiable evaluation path.
    """

    centers_ras_mm: torch.Tensor  # [G, 3]
    cholesky_factor: torch.Tensor  # [G, 3, 3]
    support_amplitude: torch.Tensor  # [G]
    appearance: torch.Tensor  # [G]
    appearance_valid: torch.Tensor  # [G], bool
    plane_u_bounds_mm: torch.Tensor  # [G, 2]
    plane_v_bounds_mm: torch.Tensor  # [G, 2]
    plane_normal_bounds_mm: torch.Tensor  # [G, 2]


def _plane_points(plane: PhysicalPlane, gaussian: GaussianBatch) -> torch.Tensor:
    dtype, device = gaussian.centers_ras_mm.dtype, gaussian.centers_ras_mm.device
    height, width = plane.shape_hw
    v, u = torch.meshgrid(torch.arange(height, dtype=dtype, device=device), torch.arange(width, dtype=dtype, device=device), indexing="ij")
    origin = torch.tensor(plane.pixel_center_origin_ras_mm, dtype=dtype, device=device)
    axis_u = torch.tensor(plane.axis_u_ras, dtype=dtype, device=device)
    axis_v = torch.tensor(plane.axis_v_ras, dtype=dtype, device=device)
    return origin + (u * plane.spacing_uv_mm[0]).unsqueeze(-1) * axis_u + (v * plane.spacing_uv_mm[1]).unsqueeze(-1) * axis_v


def _zero_density_log_floor(dtype: torch.dtype) -> float:
    """Return a strict log floor below which ``torch.exp`` is exactly zero."""
    if dtype is torch.float32:
        smallest_subnormal = math.ldexp(1.0, -149)
    elif dtype is torch.float64:
        smallest_subnormal = math.ldexp(1.0, -1074)
    else:  # GaussianBatch validation currently makes this unreachable.
        raise TypeError("renderer supports only float32 and float64 Gaussian tensors")
    # Keep a margin below the smallest subnormal so a conservative geometric
    # bound never drops a nonzero contribution because of a rounding boundary.
    return math.log(smallest_subnormal) - 2.0


def _prepare_gaussian_terms(gaussian: GaussianBatch, plane: PhysicalPlane, appearance_channel: int) -> _PreparedGaussianTerms:
    """Compute every per-Gaussian continuous value and culling bound once."""
    covariance = gaussian.covariance()
    cholesky_factor = torch.linalg.cholesky(covariance)
    support_amplitude = torch.exp(gaussian.log_support_amplitude[:, 0])
    appearance_valid = gaussian.appearance_valid[:, appearance_channel]

    dtype, device = gaussian.centers_ras_mm.dtype, gaussian.centers_ras_mm.device
    origin = torch.tensor(plane.pixel_center_origin_ras_mm, dtype=dtype, device=device)
    axis_u = torch.tensor(plane.axis_u_ras, dtype=dtype, device=device)
    axis_v = torch.tensor(plane.axis_v_ras, dtype=dtype, device=device)
    normal = torch.tensor(plane.signed_normal_ras, dtype=dtype, device=device)
    centre_offset = gaussian.centers_ras_mm - origin
    projected_u = centre_offset @ axis_u
    projected_v = centre_offset @ axis_v
    projected_normal = centre_offset @ normal

    # For q = (x-mu)^T Sigma^-1 (x-mu), q >= ||x-mu||^2 / trace(Sigma).
    # The resulting sphere therefore contains every point whose density can be
    # nonzero in the active floating-point dtype.  The small expansion makes
    # its plane-aligned bounds conservative after finite-precision arithmetic.
    max_quadratic = (2.0 * (gaussian.log_support_amplitude[:, 0].detach() - _zero_density_log_floor(dtype))).clamp_min(0.0)
    covariance_trace = torch.diagonal(covariance.detach(), dim1=-2, dim2=-1).sum(dim=-1)
    radius = torch.sqrt(covariance_trace * max_quadratic)
    radius = radius + 8.0 * torch.finfo(dtype).eps * (radius + 1.0)

    def _bounds(projected: torch.Tensor) -> torch.Tensor:
        return torch.stack((projected - radius, projected + radius), dim=1).detach()

    return _PreparedGaussianTerms(
        centers_ras_mm=gaussian.centers_ras_mm,
        cholesky_factor=cholesky_factor,
        support_amplitude=support_amplitude,
        appearance=gaussian.appearance[:, appearance_channel],
        appearance_valid=appearance_valid,
        plane_u_bounds_mm=_bounds(projected_u),
        plane_v_bounds_mm=_bounds(projected_v),
        plane_normal_bounds_mm=_bounds(projected_normal),
    )


def _gaussian_terms(
    points: torch.Tensor,
    prepared: _PreparedGaussianTerms,
    gaussian_indices: torch.Tensor,
    gaussian_chunk_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return additive numerator and support without explicitly inverting Sigma."""
    numerator = points.new_zeros((points.shape[0],))
    support = points.new_zeros((points.shape[0],))
    if gaussian_indices.numel() == 0:
        return numerator, support
    step = gaussian_chunk_size or gaussian_indices.numel()
    for start in range(0, gaussian_indices.numel(), step):
        indices = gaussian_indices[start:start + step]
        centers = prepared.centers_ras_mm.index_select(0, indices)
        factor = prepared.cholesky_factor.index_select(0, indices)
        difference = points[:, None, :] - centers[None, :, :]
        rhs = difference.permute(1, 2, 0)  # [G, 3, P]
        solved = torch.linalg.solve_triangular(factor, rhs, upper=False)
        exponent = -0.5 * solved.square().sum(dim=1).transpose(0, 1)
        density = prepared.support_amplitude.index_select(0, indices)[None, :] * torch.exp(exponent)
        valid = prepared.appearance_valid.index_select(0, indices).to(dtype=density.dtype)[None, :]
        weighted_density = density * valid
        support = support + weighted_density.sum(dim=1)
        numerator = numerator + (weighted_density * prepared.appearance.index_select(0, indices)[None, :]).sum(dim=1)
    return numerator, support


def _render_sampled_points(
    points: torch.Tensor,
    prepared: _PreparedGaussianTerms,
    gaussian_indices: torch.Tensor,
    *,
    normal: torch.Tensor,
    plane: PhysicalPlane,
    config: RenderConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render selected points with one already prepared Gaussian subset."""
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
            local_numerator, local_support = _gaussian_terms(shifted, prepared, gaussian_indices, config.gaussian_chunk_size)
            local_unsupported = local_support <= config.support_epsilon
            local_intensity = local_numerator / local_support.clamp_min(config.support_epsilon)
            local_supported = ~local_unsupported
            chunk_intensity = chunk_intensity + float(weight) * torch.where(
                local_supported,
                local_intensity,
                torch.zeros_like(local_intensity),
            )
            chunk_support = chunk_support + float(weight) * local_support
            chunk_supported_psf = chunk_supported_psf + float(weight) * local_supported.to(dtype=chunk_points.dtype)
        chunk_intensity = chunk_intensity / chunk_supported_psf.clamp_min(torch.finfo(chunk_points.dtype).eps)
        intensity_chunks.append(chunk_intensity)
        support_chunks.append(chunk_support)
        supported_psf_chunks.append(chunk_supported_psf)
    return torch.cat(intensity_chunks), torch.cat(support_chunks), torch.cat(supported_psf_chunks)


def _profile_normal_bounds(plane: PhysicalPlane, profile: SlabProfile) -> tuple[float, float]:
    offsets_mm = tuple(float(offset) * plane.thickness_mm for offset in profile.normalized_offsets)
    return min(offsets_mm), max(offsets_mm)


def _overlaps(bounds: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return (bounds[:, 1] >= lower) & (bounds[:, 0] <= upper)


def _coarse_plane_candidates(prepared: _PreparedGaussianTerms, plane: PhysicalPlane, profile: SlabProfile) -> torch.Tensor:
    """Cull Gaussians whose conservative support bounds miss the full plane."""
    height, width = plane.shape_hw
    normal_lower, normal_upper = _profile_normal_bounds(plane, profile)
    visible = prepared.appearance_valid.detach()
    visible = visible & _overlaps(prepared.plane_u_bounds_mm, 0.0, (width - 1) * plane.spacing_uv_mm[0])
    visible = visible & _overlaps(prepared.plane_v_bounds_mm, 0.0, (height - 1) * plane.spacing_uv_mm[1])
    visible = visible & _overlaps(prepared.plane_normal_bounds_mm, normal_lower, normal_upper)
    return torch.nonzero(visible, as_tuple=False).reshape(-1)


def _tile_candidates(
    prepared: _PreparedGaussianTerms,
    coarse_indices: torch.Tensor,
    plane: PhysicalPlane,
    *,
    v_start: int,
    v_stop: int,
    u_start: int,
    u_stop: int,
    normal_lower: float,
    normal_upper: float,
) -> torch.Tensor:
    """Cull the coarse set again against one conservative physical plane tile."""
    if coarse_indices.numel() == 0:
        return coarse_indices
    u_lower = u_start * plane.spacing_uv_mm[0]
    u_upper = (u_stop - 1) * plane.spacing_uv_mm[0]
    v_lower = v_start * plane.spacing_uv_mm[1]
    v_upper = (v_stop - 1) * plane.spacing_uv_mm[1]
    bounds_u = prepared.plane_u_bounds_mm.index_select(0, coarse_indices)
    bounds_v = prepared.plane_v_bounds_mm.index_select(0, coarse_indices)
    bounds_n = prepared.plane_normal_bounds_mm.index_select(0, coarse_indices)
    visible = _overlaps(bounds_u, u_lower, u_upper)
    visible = visible & _overlaps(bounds_v, v_lower, v_upper)
    visible = visible & _overlaps(bounds_n, normal_lower, normal_upper)
    return coarse_indices[visible]


def _validate_render_inputs(
    gaussians: GaussianBatch,
    plane: PhysicalPlane,
    appearance_channel: int,
    config: RenderConfig | None,
) -> RenderConfig:
    if not isinstance(gaussians, GaussianBatch) or not isinstance(plane, PhysicalPlane):
        raise TypeError("gaussians and plane must use their T0 contract types")
    gaussians.validate()
    if not 0 <= appearance_channel < gaussians.appearance_channels:
        raise IndexError("appearance_channel is outside GaussianBatch appearance channels")
    config = config or RenderConfig()
    if not isinstance(config, RenderConfig):
        raise TypeError("config must be a RenderConfig or None")
    finfo = torch.finfo(gaussians.centers_ras_mm.dtype)
    if not finfo.tiny <= config.support_epsilon <= finfo.max:
        raise ValueError("support_epsilon must be representable in the renderer dtype")
    return config


def _result_from_flat(
    intensity: torch.Tensor,
    support: torch.Tensor,
    supported_psf_mass: torch.Tensor,
    *,
    plane: PhysicalPlane,
    config: RenderConfig,
    pixel_gaussian_candidate_pairs: int,
) -> RenderResult:
    unsupported = supported_psf_mass < config.minimum_supported_psf_mass
    intensity = torch.where(unsupported, torch.full_like(intensity, float("nan")), intensity)
    height, width = plane.shape_hw
    return RenderResult(
        intensity.reshape(height, width),
        support.reshape(height, width),
        supported_psf_mass.reshape(height, width),
        unsupported.reshape(height, width),
        pixel_gaussian_candidate_pairs=pixel_gaussian_candidate_pairs,
    )


def _render_plane(
    gaussians: GaussianBatch,
    plane: PhysicalPlane,
    *,
    appearance_channel: int,
    config: RenderConfig | None,
    spatial_culling: bool,
) -> RenderResult:
    config = _validate_render_inputs(gaussians, plane, appearance_channel, config)
    points = _plane_points(plane, gaussians).reshape(-1, 3)
    if not bool(torch.isfinite(points).all()):
        raise ValueError("plane coordinates are not finite in the renderer dtype")
    normal = torch.tensor(plane.signed_normal_ras, dtype=points.dtype, device=points.device)
    prepared = _prepare_gaussian_terms(gaussians, plane, appearance_channel)

    if not spatial_culling:
        all_indices = torch.arange(gaussians.count, dtype=torch.long, device=points.device)
        intensity, support, supported_psf_mass = _render_sampled_points(
            points,
            prepared,
            all_indices,
            normal=normal,
            plane=plane,
            config=config,
        )
        return _result_from_flat(
            intensity,
            support,
            supported_psf_mass,
            plane=plane,
            config=config,
            pixel_gaussian_candidate_pairs=points.shape[0] * gaussians.count,
        )

    height, width = plane.shape_hw
    normal_lower, normal_upper = _profile_normal_bounds(plane, config.profile)
    coarse_indices = _coarse_plane_candidates(prepared, plane, config.profile)
    intensity = points.new_zeros((points.shape[0],))
    support = points.new_zeros((points.shape[0],))
    supported_psf_mass = points.new_zeros((points.shape[0],))
    candidate_pairs = 0
    tile_height, tile_width = config.tile_shape_hw
    for v_start in range(0, height, tile_height):
        v_stop = min(v_start + tile_height, height)
        for u_start in range(0, width, tile_width):
            u_stop = min(u_start + tile_width, width)
            rows = torch.arange(v_start, v_stop, dtype=torch.long, device=points.device)
            columns = torch.arange(u_start, u_stop, dtype=torch.long, device=points.device)
            pixel_indices = (rows[:, None] * width + columns[None, :]).reshape(-1)
            tile_indices = _tile_candidates(
                prepared,
                coarse_indices,
                plane,
                v_start=v_start,
                v_stop=v_stop,
                u_start=u_start,
                u_stop=u_stop,
                normal_lower=normal_lower,
                normal_upper=normal_upper,
            )
            if tile_indices.numel() == 0:
                continue
            tile_intensity, tile_support, tile_supported_psf_mass = _render_sampled_points(
                points.index_select(0, pixel_indices),
                prepared,
                tile_indices,
                normal=normal,
                plane=plane,
                config=config,
            )
            intensity = intensity.index_copy(0, pixel_indices, tile_intensity)
            support = support.index_copy(0, pixel_indices, tile_support)
            supported_psf_mass = supported_psf_mass.index_copy(0, pixel_indices, tile_supported_psf_mass)
            candidate_pairs += pixel_indices.numel() * tile_indices.numel()
    return _result_from_flat(
        intensity,
        support,
        supported_psf_mass,
        plane=plane,
        config=config,
        pixel_gaussian_candidate_pairs=candidate_pairs,
    )


def render_plane(gaussians: GaussianBatch, plane: PhysicalPlane, *, appearance_channel: int = 0, config: RenderConfig | None = None) -> RenderResult:
    """Render one physical MRI plane with conservative coarse and tile culling.

    A delta profile is the closed-form thin-plane evaluation. Other profiles
    evaluate the normalized latent field ``N(x) / S(x)`` at every quadrature
    depth and then apply normalized PSF weights to those intensities. Support
    mass is reported as the same weighted average of depth-wise support.
    ``supported_psf_mass`` reports the PSF weight with numerically supported
    samples; supported samples are renormalized and the configured coverage
    threshold determines the explicit unsupported mask.  Per-Gaussian values
    are prepared once; coarse-plane and tile bounds omit only densities that
    are already exactly zero through floating-point underflow.  Chunk boundaries
    remain inside the differentiable tensor path; manifest/selection operations
    are outside it.  The function is deliberately side-effect free: it neither
    knows about episode ledgers nor issues prediction receipts.
    """
    return _render_plane(
        gaussians,
        plane,
        appearance_channel=appearance_channel,
        config=config,
        spatial_culling=True,
    )


def render_plane_brute_force_reference(
    gaussians: GaussianBatch,
    plane: PhysicalPlane,
    *,
    appearance_channel: int = 0,
    config: RenderConfig | None = None,
) -> RenderResult:
    """Render the explicit unculled reference operator for equivalence tests.

    It shares the prepared covariance/appearance values with the optimized path
    but evaluates every batch Gaussian for every plane pixel.  This function is
    intentionally public so tests and diagnostics can distinguish numeric
    operator changes from spatial-culling changes.
    """
    return _render_plane(
        gaussians,
        plane,
        appearance_channel=appearance_channel,
        config=config,
        spatial_culling=False,
    )
