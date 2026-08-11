"""Deterministic, geometry-only initial point placement."""

from __future__ import annotations

import torch
from torch import nn

from .config import PointGuidedConfig
from .contracts import PointGuidedGeometryError, VolumeGeometry
from .sampling import voxel_dhw_to_ras_mm


def _normalise_brain_mask(
    brain_mask: torch.Tensor | None,
    *,
    batch_size: int,
    geometry: VolumeGeometry,
    device: torch.device,
) -> torch.Tensor | None:
    if brain_mask is None:
        return None
    if not isinstance(brain_mask, torch.Tensor):
        raise TypeError("brain_mask must be a torch.Tensor or None")
    if brain_mask.ndim == 5:
        if brain_mask.shape[1] != 1:
            raise ValueError("brain_mask rank-5 form must have shape [B, 1, D, H, W]")
        brain_mask = brain_mask[:, 0]
    if brain_mask.ndim != 4 or brain_mask.shape[0] != batch_size or tuple(brain_mask.shape[1:]) != tuple(geometry.shape_dhw):
        raise ValueError("brain_mask must have shape [B, D, H, W] or [B, 1, D, H, W]")
    if brain_mask.device != device:
        raise ValueError("brain_mask and requested output device must agree")
    if brain_mask.is_floating_point() and not bool(torch.isfinite(brain_mask).all()):
        raise ValueError("brain_mask must be finite")
    is_binary = (brain_mask == 0) | (brain_mask == 1)
    if not bool(is_binary.all()):
        raise ValueError("brain_mask must be binary (zero outside and one inside)")
    return brain_mask.to(dtype=torch.bool)


def _radical_inverse(indices: torch.Tensor, base: int, *, dtype: torch.dtype) -> torch.Tensor:
    """Evaluate one Halton radical-inverse coordinate without randomness."""

    result = torch.zeros_like(indices, dtype=dtype)
    quotient = indices.clone()
    factor = 1.0 / float(base)
    while bool((quotient > 0).any()):
        result = result + torch.remainder(quotient, base).to(dtype=dtype) * factor
        quotient = torch.div(quotient, base, rounding_mode="floor")
        factor /= float(base)
    return result


def _full_volume_halton_candidates(
    geometry: VolumeGeometry,
    candidate_count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Continuous non-Cartesian Halton candidates in ``[d, h, w]`` space."""

    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    # Start at one because the all-zero Halton point is a rigid corner.  The
    # prime bases make this a deterministic low-discrepancy 3-D sequence, not
    # a Cartesian grid of voxel centres.
    indices = torch.arange(1, candidate_count + 1, dtype=torch.long, device=device)
    unit_dhw = torch.stack(
        tuple(_radical_inverse(indices, base, dtype=dtype) for base in (2, 3, 5)),
        dim=-1,
    )
    upper = torch.as_tensor(
        tuple(length - 1 for length in geometry.shape_dhw),
        dtype=dtype,
        device=device,
    )
    return unit_dhw * upper


def _deterministic_mask_candidates(
    mask_dhw: torch.Tensor,
    candidate_count: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return deterministic continuous candidates associated with valid mask voxels.

    The binary mask has voxel resolution, so a sub-voxel candidate is legal
    when its nearest voxel centre is a valid mask element. Offsets stay
    strictly inside that voxel's half-cell and reflect inward at the outer
    volume box, giving the masked path the same non-Cartesian continuous
    point contract as the unmasked Halton path.
    """

    valid_count = int(mask_dhw.sum().item())
    if valid_count <= candidate_count:
        kept_indices = torch.nonzero(mask_dhw.reshape(-1), as_tuple=False).flatten()
    else:
        # A bounded reservoir keyed solely by linear voxel index avoids
        # materialising every valid coordinate for large masks.  The LCG is
        # deterministic, has no learned/data-value input, and FPS below turns
        # this candidate set into a quasi-uniform physical point set.
        flat = mask_dhw.reshape(-1)
        kept_indices = torch.empty(0, dtype=torch.long, device=flat.device)
        kept_scores = torch.empty(0, dtype=torch.long, device=flat.device)
        chunk_size = 262_144
        modulus = 2_147_483_647
        for start in range(0, flat.numel(), chunk_size):
            stop = min(start + chunk_size, flat.numel())
            local = torch.nonzero(flat[start:stop], as_tuple=False).flatten()
            if local.numel() == 0:
                continue
            indices = local + start
            scores = torch.remainder(indices * 1_103_515_245 + 12_345, modulus)
            combined_indices = torch.cat((kept_indices, indices))
            combined_scores = torch.cat((kept_scores, scores))
            if combined_indices.numel() > candidate_count:
                order = torch.argsort(combined_scores, stable=True)[:candidate_count]
                kept_indices = combined_indices[order]
                kept_scores = combined_scores[order]
            else:
                kept_indices = combined_indices
                kept_scores = combined_scores

    depth, height, width = mask_dhw.shape
    d = torch.div(kept_indices, height * width, rounding_mode="floor")
    remainder = torch.remainder(kept_indices, height * width)
    h = torch.div(remainder, width, rounding_mode="floor")
    w = torch.remainder(remainder, width)
    voxel_centres = torch.stack((d, h, w), dim=-1).to(dtype=dtype)
    # Quasi-random offsets keyed by the immutable valid-voxel index prevent
    # the masked path from collapsing into a Cartesian lattice.  At a volume
    # boundary, reflect a would-be outward offset inward rather than clipping
    # it to zero, so even a corner-only mask remains genuinely sub-voxel.
    raw_offsets = torch.stack(
        tuple(_radical_inverse(kept_indices + 1, base, dtype=dtype) - 0.5 for base in (2, 3, 5)),
        dim=-1,
    )
    upper = torch.as_tensor((depth - 1, height - 1, width - 1), dtype=dtype, device=mask_dhw.device)
    magnitude = raw_offsets.abs().clamp_min(0.01) * 0.98
    can_step_negative = voxel_centres > 0.0
    can_step_positive = voxel_centres < upper
    prefer_positive = raw_offsets >= 0.0
    direction = torch.where(
        prefer_positive & can_step_positive,
        torch.ones_like(raw_offsets),
        torch.where(
            (~prefer_positive) & can_step_negative,
            -torch.ones_like(raw_offsets),
            torch.where(
                can_step_positive,
                torch.ones_like(raw_offsets),
                torch.where(can_step_negative, -torch.ones_like(raw_offsets), torch.zeros_like(raw_offsets)),
            ),
        ),
    )
    return voxel_centres + direction * magnitude


def _farthest_quasi_uniform_points(candidate_ras_mm: torch.Tensor, num_points: int) -> torch.Tensor:
    if candidate_ras_mm.ndim != 2 or candidate_ras_mm.shape[1] != 3:
        raise ValueError("candidate_ras_mm must have shape [M, 3]")
    if candidate_ras_mm.shape[0] < num_points:
        raise PointGuidedGeometryError("fewer valid point candidates than requested num_points")

    # A centre-nearest seed followed by farthest-point traversal is deterministic
    # and distributes selected points in physical (not voxel-index) space.
    centre = candidate_ras_mm.mean(dim=0, keepdim=True)
    seed = torch.argmin(((candidate_ras_mm - centre) ** 2).sum(dim=-1))
    chosen = torch.empty(num_points, dtype=torch.long, device=candidate_ras_mm.device)
    nearest_sq = torch.full(
        (candidate_ras_mm.shape[0],),
        float("inf"),
        dtype=candidate_ras_mm.dtype,
        device=candidate_ras_mm.device,
    )
    current = seed
    for index in range(num_points):
        chosen[index] = current
        delta = candidate_ras_mm - candidate_ras_mm[current]
        nearest_sq = torch.minimum(nearest_sq, (delta * delta).sum(dim=-1))
        nearest_sq[current] = -1.0
        if index + 1 < num_points:
            current = torch.argmax(nearest_sq)
    return candidate_ras_mm[chosen]


def initialize_quasi_uniform_points(
    geometry: VolumeGeometry,
    batch_size: int,
    num_points: int,
    *,
    brain_mask: torch.Tensor | None = None,
    candidate_multiplier: int = 4,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return exactly ``num_points`` deterministic RAS-mm physical points.

    Selection depends only on the immutable geometry and optional binary brain
    mask.  It deliberately has no image-volume argument, which makes image
    value dependence impossible.  A clear error is raised when a mask cannot
    provide the requested number of valid centres.
    """

    if batch_size <= 0 or num_points <= 0 or candidate_multiplier <= 0:
        raise ValueError("batch_size, num_points, and candidate_multiplier must be positive")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be torch.float32 or torch.float64")
    resolved_device = torch.device(device) if device is not None else (
        brain_mask.device if isinstance(brain_mask, torch.Tensor) else torch.device("cpu")
    )
    mask = _normalise_brain_mask(brain_mask, batch_size=batch_size, geometry=geometry, device=resolved_device)
    candidate_count = num_points * candidate_multiplier
    selected_by_batch: list[torch.Tensor] = []

    for batch_index in range(batch_size):
        if mask is None:
            candidates_dhw = _full_volume_halton_candidates(
                geometry,
                candidate_count,
                device=resolved_device,
                dtype=dtype,
            )
        else:
            valid_count = int(mask[batch_index].sum().item())
            if valid_count < num_points:
                raise PointGuidedGeometryError(
                    f"brain_mask batch item {batch_index} has {valid_count} valid voxel centres, fewer than requested {num_points}"
                )
            candidates_dhw = _deterministic_mask_candidates(
                mask[batch_index],
                min(candidate_count, valid_count),
                dtype=dtype,
            )

        candidates_ras_mm = voxel_dhw_to_ras_mm(candidates_dhw.to(dtype=dtype), geometry)
        selected_by_batch.append(_farthest_quasi_uniform_points(candidates_ras_mm, num_points))

    return torch.stack(selected_by_batch, dim=0)


class DeterministicPointInitializer(nn.Module):
    """Configuration-bound geometry/mask-only point initializer."""

    def __init__(self, config: PointGuidedConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        geometry: VolumeGeometry,
        batch_size: int,
        brain_mask: torch.Tensor | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return initialize_quasi_uniform_points(
            geometry,
            batch_size,
            self.config.num_points,
            brain_mask=brain_mask,
            candidate_multiplier=self.config.point_candidate_multiplier,
            device=device,
            dtype=dtype,
        )


__all__ = ["DeterministicPointInitializer", "initialize_quasi_uniform_points"]
