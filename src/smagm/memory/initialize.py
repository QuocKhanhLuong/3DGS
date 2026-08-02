"""Initialize structural and volumetric seed Gaussian banks from anchors."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import torch

from ..anchors import AnchorBatch
from ..gaussians import (
    GaussianBatch,
    RawGaussianParameters,
    gaussian_batch_from_raw,
    restore_gauge_fixed_gaussian_batch,
)
from .appearance import validate_appearance_slots
from .contracts import GaussianMemory, GaussianMemoryBank, PrimitiveKind, gaussian_memory_hash
from .observability import initial_observability


@dataclass(frozen=True)
class SeedMemoryConfig:
    structural_tangent_fraction: float = 0.7
    structural_normal_fraction: float = 0.15
    volumetric_scale_fraction: float = 0.9
    initial_uncertainty: float = 1.0
    field_center_offset_fraction: float = 0.05

    def __post_init__(self) -> None:
        if any(value <= 0 for value in self.__dict__.values()):
            raise ValueError("seed-memory scales and uncertainty must be positive")

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _covariance_factor(anchors: AnchorBatch, local_scales: torch.Tensor) -> torch.Tensor:
    covariance = anchors.frame_axes_ras @ torch.diag_embed(local_scales.square()) @ anchors.frame_axes_ras.transpose(-1, -2)
    epsilon = torch.finfo(covariance.dtype).eps * 16
    identity = torch.eye(3, dtype=covariance.dtype, device=covariance.device)
    return torch.linalg.cholesky(covariance + epsilon * identity)


def _bank(
    anchors: AnchorBatch, kind: PrimitiveKind, scales: torch.Tensor, *, uncertainty: float,
    field_values: torch.Tensor | None,
) -> GaussianMemoryBank:
    count = anchors.count
    appearance, valid = validate_appearance_slots(anchors.appearance, anchors.appearance_valid)
    ids = tuple(f"{kind.value.lower()}:{anchor_id}" for anchor_id in anchors.anchor_ids)
    centers = anchors.centers_ras_mm
    if field_values is not None:
        if field_values.shape != (count, 1) or not bool(torch.isfinite(field_values).all()):
            raise ValueError("field_values must be finite with shape [N,1]")
        normal = anchors.frame_axes_ras[:, :, 2]
        multiplier = 1.0 if kind is PrimitiveKind.STRUCTURAL else 0.25
        centers = centers + multiplier * torch.tanh(field_values) * scales[:, 2:3] * normal
    raw = RawGaussianParameters(
        centers_ras_mm=centers,
        covariance_factor=_covariance_factor(anchors, scales),
        raw_log_support_amplitude=torch.zeros((count, 1), dtype=appearance.dtype, device=appearance.device),
        appearance=appearance, appearance_valid=valid,
        patient_state_index=torch.zeros(count, dtype=torch.int64, device=appearance.device),
        primitive_kind=(kind.value,) * count, primitive_id=ids,
    )
    gaussians = gaussian_batch_from_raw(raw)
    provenance = tuple(hashlib.sha256(f"{kind.value}:{anchor_id}:{anchor_hash}".encode()).hexdigest() for anchor_id, anchor_hash in zip(anchors.anchor_ids, anchors.geometry.provenance_hashes))
    return GaussianMemoryBank(
        kind=kind, gaussians=gaussians, anchor_ids=anchors.anchor_ids,
        parent_primitive_ids=(None,) * count, provenance_hashes=provenance,
        observability=initial_observability(anchors.observability, initial_uncertainty=uncertainty),
    )


def _apply_field_to_volumetric_centers(
    anchors: AnchorBatch,
    gaussians: GaussianBatch,
    field_values: torch.Tensor,
    *,
    volumetric_scales: torch.Tensor,
    field_center_offset_fraction: float,
) -> GaussianBatch:
    """Apply the context-derived field scaffold without replacing head outputs.

    The head remains the owner of volumetric covariance, amplitude, and
    appearance.  The shared field contributes only a bounded local-normal
    placement offset, matching the anchor-only volumetric seed path.  Keeping
    this as a differentiable tensor expression is important for sparse target
    planes: the thin structural bank can be numerically unsupported between
    context planes even though the thicker volumetric bank still renders.
    """

    if field_values.shape != (anchors.count, 1) or not bool(torch.isfinite(field_values).all()):
        raise ValueError("field_values must be finite with shape [N,1]")
    if gaussians.count != anchors.count:
        raise ValueError("field-conditioned volumetric Gaussian count must match anchors")
    scaled_field = field_values * field_center_offset_fraction
    normal = anchors.frame_axes_ras[:, :, 2]
    center_offset = 0.25 * torch.tanh(scaled_field) * volumetric_scales[:, 2:3] * normal
    return restore_gauge_fixed_gaussian_batch(
        centers_ras_mm=gaussians.centers_ras_mm + center_offset,
        covariance_factor=gaussians.covariance_factor,
        log_support_amplitude=gaussians.log_support_amplitude,
        appearance=gaussians.appearance,
        appearance_valid=gaussians.appearance_valid,
        covariance_epsilon=gaussians.covariance_epsilon,
        primitive_kind=gaussians.primitive_kind,
        primitive_id=gaussians.primitive_id,
        gauge_policy=gaussians.gauge_policy,
        gauge_config_hash=gaussians.gauge_config_hash or "",
    )


def initialize_seed_memory(
    anchors: AnchorBatch, *, config: SeedMemoryConfig | None = None,
    field_values: torch.Tensor | None = None,
    volumetric_gaussians: GaussianBatch | None = None,
) -> GaussianMemory:
    """Create the immutable dual-bank seed memory.

    The structural bank is created from the anchor frame and shared
    StructuralField.  A head-produced volumetric bank keeps the head's
    appearance/covariance outputs and receives only the same bounded
    context-derived local-normal field offset as the anchor-only fallback.
    This keeps R4 connected to the field without collapsing the dual banks or
    letting the field absorb the Gaussian head's responsibility.
    """
    config = config or SeedMemoryConfig()
    scales = anchors.support_scales_mm
    structural_scales = torch.stack((
        scales[:, 0] * config.structural_tangent_fraction,
        scales[:, 1] * config.structural_tangent_fraction,
        scales[:, 2] * config.structural_normal_fraction,
    ), dim=1)
    volumetric_scales = scales * config.volumetric_scale_fraction
    scaled_field = None if field_values is None else field_values * config.field_center_offset_fraction
    structural = _bank(anchors, PrimitiveKind.STRUCTURAL, structural_scales, uncertainty=config.initial_uncertainty, field_values=scaled_field)
    if volumetric_gaussians is None:
        volumetric = _bank(anchors, PrimitiveKind.VOLUMETRIC, volumetric_scales, uncertainty=config.initial_uncertainty, field_values=scaled_field)
    else:
        if volumetric_gaussians.count != anchors.count:
            raise ValueError("head-produced volumetric Gaussian count must match anchor count")
        if volumetric_gaussians.appearance_channels != anchors.appearance.shape[1]:
            raise ValueError("head-produced volumetric appearance channels must match anchor modalities")
        if volumetric_gaussians.primitive_kind != (PrimitiveKind.VOLUMETRIC.value,) * anchors.count:
            raise ValueError("head-produced volumetric Gaussians must be typed as VOLUMETRIC")
        if field_values is not None:
            volumetric_gaussians = _apply_field_to_volumetric_centers(
                anchors,
                volumetric_gaussians,
                field_values,
                volumetric_scales=volumetric_scales,
                field_center_offset_fraction=config.field_center_offset_fraction,
            )
        provenance = tuple(
            hashlib.sha256(f"{PrimitiveKind.VOLUMETRIC.value}:{anchor_id}:{anchor_hash}".encode()).hexdigest()
            for anchor_id, anchor_hash in zip(anchors.anchor_ids, anchors.geometry.provenance_hashes)
        )
        volumetric = GaussianMemoryBank(
            kind=PrimitiveKind.VOLUMETRIC,
            gaussians=volumetric_gaussians,
            anchor_ids=anchors.anchor_ids,
            parent_primitive_ids=(None,) * anchors.count,
            provenance_hashes=provenance,
            observability=initial_observability(anchors.observability, initial_uncertainty=config.initial_uncertainty),
        )
    digest = gaussian_memory_hash(structural, volumetric, anchors.modality_ids)
    return GaussianMemory(structural, volumetric, anchors.modality_ids, digest)
