"""Deterministic anchor frame refinement from supported field gradients."""

from __future__ import annotations

import torch

from .contracts import AnchorGeometryBatch


def refine_frames_from_gradients(
    geometry: AnchorGeometryBatch, gradients_ras: torch.Tensor, supported: torch.Tensor,
    *, minimum_gradient_norm: float = 1e-6,
) -> AnchorGeometryBatch:
    if gradients_ras.shape != geometry.centers_ras_mm.shape or supported.shape != (gradients_ras.shape[0],):
        raise ValueError("gradients and support must match anchor centres")
    if supported.dtype is not torch.bool or minimum_gradient_norm <= 0:
        raise ValueError("supported must be bool and minimum norm positive")
    frames = geometry.frame_axes_ras.clone()
    validity = geometry.frame_validity.clone()
    for index in range(gradients_ras.shape[0]):
        norm = torch.linalg.vector_norm(gradients_ras[index])
        if not bool(supported[index]) or float(norm.detach()) < minimum_gradient_norm:
            continue
        normal = gradients_ras[index] / norm
        references = torch.eye(3, dtype=normal.dtype, device=normal.device)
        reference = references[torch.argmin(normal.abs())]
        tangent_u = torch.linalg.cross(reference, normal)
        tangent_u = tangent_u / torch.linalg.vector_norm(tangent_u)
        tangent_v = torch.linalg.cross(normal, tangent_u)
        frames[index] = torch.stack((tangent_u, tangent_v, normal), dim=1)
        validity[index] = True
    return AnchorGeometryBatch(
        anchor_ids=geometry.anchor_ids, centers_ras_mm=geometry.centers_ras_mm,
        frame_axes_ras=frames, frame_validity=validity, support_scales_mm=geometry.support_scales_mm,
        geometry_confidence=geometry.geometry_confidence, disagreement=geometry.disagreement,
        contributing_observation_ids=geometry.contributing_observation_ids,
        contributing_plane_hashes=geometry.contributing_plane_hashes,
        provenance_hashes=geometry.provenance_hashes,
    )
