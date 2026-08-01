"""Physical kernel interpolation floor built only from legal context pixels."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Sequence

import torch

from ..gaussians import GaussianBatch, RawGaussianParameters, gaussian_batch_from_raw

if TYPE_CHECKING:
    from ..training.episode import ContextEvidence


@dataclass(frozen=True)
class SparseInterpolationConfig:
    """Deterministic context-pixel kernel sizes in physical millimetres."""

    stride_vu: tuple[int, int] = (1, 1)
    tangent_scale_fraction: float = 0.75
    normal_scale_fraction: float = 0.5
    maximum_points: int | None = None

    def __post_init__(self) -> None:
        if len(self.stride_vu) != 2 or any(not isinstance(v, int) or v <= 0 for v in self.stride_vu):
            raise ValueError("stride_vu must contain two positive integers")
        if any(not math.isfinite(v) or v <= 0 for v in (self.tangent_scale_fraction, self.normal_scale_fraction)):
            raise ValueError("interpolation scale fractions must be finite and positive")
        if self.maximum_points is not None and (not isinstance(self.maximum_points, int) or self.maximum_points <= 0):
            raise ValueError("maximum_points must be None or a positive integer")


def _ras_factor(
    basis_rows: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    local_covariance = torch.diag_embed(scales.square())
    covariance = basis_rows.transpose(-1, -2) @ local_covariance @ basis_rows
    epsilon = torch.finfo(covariance.dtype).eps * 16
    identity = torch.eye(3, dtype=covariance.dtype, device=covariance.device)
    return torch.linalg.cholesky(covariance + epsilon * identity)


def construct_sparse_interpolation_gaussians(
    context_evidence: Sequence["ContextEvidence"],
    *,
    modality_ids: Sequence[str],
    config: SparseInterpolationConfig | None = None,
) -> GaussianBatch:
    """Represent context-only physical kernel interpolation as fixed Gaussians.

    This is a receipt-compatible R0 floor, not a learned Gaussian model: it
    consumes normalized context pixels directly and owns no trainable module.
    """

    items = tuple(context_evidence)
    modalities = tuple(modality_ids)
    config = config or SparseInterpolationConfig()
    if not items or not modalities or len(set(modalities)) != len(modalities):
        raise ValueError("interpolation requires context evidence and unique modality IDs")
    if any(item.modality_id not in modalities for item in items):
        raise ValueError("all context modalities must be declared")

    centers: list[torch.Tensor] = []
    factors: list[torch.Tensor] = []
    appearances: list[torch.Tensor] = []
    validities: list[torch.Tensor] = []
    primitive_ids: list[str] = []
    remaining = config.maximum_points
    for item in items:
        image = item.normalized_image[0, 0]
        valid = item.valid_mask[0, 0]
        height, width = image.shape
        indices = [
            (v, u)
            for v in range(0, height, config.stride_vu[0])
            for u in range(0, width, config.stride_vu[1])
            if bool(valid[v, u].detach().cpu())
        ]
        if remaining is not None:
            indices = indices[:remaining]
            remaining -= len(indices)
        if not indices:
            if remaining == 0:
                break
            continue
        index = torch.tensor(indices, dtype=torch.int64, device=image.device)
        dtype = image.dtype
        origin = torch.tensor(item.plane.pixel_center_origin_ras_mm, dtype=dtype, device=image.device)
        axis_u = torch.tensor(item.plane.axis_u_ras, dtype=dtype, device=image.device)
        axis_v = torch.tensor(item.plane.axis_v_ras, dtype=dtype, device=image.device)
        normal = torch.tensor(item.plane.signed_normal_ras, dtype=dtype, device=image.device)
        point = (
            origin
            + index[:, 1:2] * float(item.plane.spacing_uv_mm[0]) * axis_u
            + index[:, 0:1] * float(item.plane.spacing_uv_mm[1]) * axis_v
        )
        basis = torch.stack((axis_u, axis_v, normal)).expand(len(indices), -1, -1)
        scale = image.new_tensor((
            float(item.plane.spacing_uv_mm[0]) * config.tangent_scale_fraction,
            float(item.plane.spacing_uv_mm[1]) * config.tangent_scale_fraction,
            float(item.plane.thickness_mm) * config.normal_scale_fraction,
        )).expand(len(indices), -1)
        appearance = image.new_zeros((len(indices), len(modalities)))
        appearance[:, modalities.index(item.modality_id)] = image[index[:, 0], index[:, 1]]
        appearance_valid = torch.zeros_like(appearance, dtype=torch.bool)
        appearance_valid[:, modalities.index(item.modality_id)] = True
        centers.append(point)
        factors.append(_ras_factor(basis, scale))
        appearances.append(appearance)
        validities.append(appearance_valid)
        primitive_ids.extend(f"interpolation:{item.observation_id}:{v}:{u}" for v, u in indices)
        if remaining == 0:
            break
    if not centers:
        raise ValueError("interpolation has no legal valid context pixels")
    all_centers = torch.cat(centers)
    all_appearance = torch.cat(appearances)
    return gaussian_batch_from_raw(RawGaussianParameters(
        centers_ras_mm=all_centers,
        covariance_factor=torch.cat(factors),
        raw_log_support_amplitude=all_centers.new_zeros((all_centers.shape[0], 1)),
        appearance=all_appearance,
        appearance_valid=torch.cat(validities),
        patient_state_index=torch.zeros(all_centers.shape[0], dtype=torch.int64, device=all_centers.device),
        covariance_epsilon=float(torch.finfo(all_centers.dtype).tiny),
        primitive_kind=("interpolation",) * all_centers.shape[0],
        primitive_id=tuple(primitive_ids),
    ))


__all__ = ["SparseInterpolationConfig", "construct_sparse_interpolation_gaussians"]
