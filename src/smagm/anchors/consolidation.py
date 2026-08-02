"""Deterministic physical suppression and bounded cross-plane consolidation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import torch

from .contracts import AnchorGeometryBatch, LiftedCandidateBatch


@dataclass(frozen=True)
class ConsolidationConfig:
    nms_radius_mm: float = 1.0
    merge_radius_mm: float = 1.5
    maximum_component_diameter_mm: float = 3.0
    support_scale_mm: float = 3.0

    def __post_init__(self) -> None:
        values = tuple(float(v) for v in self.__dict__.values())
        if any(not math.isfinite(v) or v <= 0 for v in values) or self.maximum_component_diameter_mm < self.merge_radius_mm:
            raise ValueError("consolidation distances must be positive finite millimetres")


def physical_nms(candidates: LiftedCandidateBatch, *, radius_mm: float) -> LiftedCandidateBatch:
    if not math.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError("radius_mm must be positive and finite")
    order = sorted(range(len(candidates.candidate_ids)), key=lambda i: (-float(candidates.score[i, 0].detach()), candidates.candidate_ids[i]))
    kept: list[int] = []
    for index in order:
        point = candidates.centers_ras_mm[index]
        if all(float(torch.linalg.vector_norm(point - candidates.centers_ras_mm[other]).detach()) > radius_mm for other in kept):
            kept.append(index)
    return candidates.select(torch.tensor(kept, dtype=torch.int64, device=candidates.centers_ras_mm.device))


def _provenance_hash(ids: tuple[str, ...], observations: tuple[str, ...], planes: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps({"candidates": ids, "observations": observations, "planes": planes}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def consolidate_candidates(
    candidates: LiftedCandidateBatch, *, config: ConsolidationConfig | None = None,
) -> AnchorGeometryBatch:
    """Greedy bounded-diameter components with deterministic canonical order."""

    config = config or ConsolidationConfig()
    order = sorted(range(len(candidates.candidate_ids)), key=lambda i: candidates.candidate_ids[i])
    clusters: list[list[int]] = []
    for index in order:
        point = candidates.centers_ras_mm[index]
        assigned = False
        for cluster in clusters:
            distances = torch.linalg.vector_norm(candidates.centers_ras_mm[cluster] - point, dim=1)
            if float(distances.min().detach()) <= config.merge_radius_mm and float(distances.max().detach()) <= config.maximum_component_diameter_mm:
                cluster.append(index)
                assigned = True
                break
        if not assigned:
            clusters.append([index])
    centers, frames, confidences, disagreements = [], [], [], []
    observation_groups, plane_groups, anchor_ids, provenance = [], [], [], []
    for cluster in clusters:
        scores = candidates.score[cluster, 0].clamp_min(torch.finfo(candidates.score.dtype).eps)
        weights = scores / scores.sum()
        points = candidates.centers_ras_mm[cluster]
        center = (weights[:, None] * points).sum(dim=0)
        representative = min(cluster, key=lambda i: candidates.candidate_ids[i])
        ids = tuple(sorted(candidates.candidate_ids[i] for i in cluster))
        observations = tuple(sorted(set(candidates.observation_ids[i] for i in cluster)))
        planes = tuple(sorted(set(candidates.plane_hashes[i] for i in cluster)))
        digest = _provenance_hash(ids, observations, planes)
        centers.append(center)
        frames.append(candidates.plane_axes_ras[representative])
        confidences.append(scores.mean().clamp(0, 1).reshape(1))
        disagreements.append(torch.linalg.vector_norm(points - center, dim=1).mean().reshape(1))
        observation_groups.append(observations); plane_groups.append(planes)
        anchor_ids.append("anchor-" + digest[:16]); provenance.append(digest)
    count = len(clusters)
    reference = candidates.centers_ras_mm
    return AnchorGeometryBatch(
        anchor_ids=tuple(anchor_ids), centers_ras_mm=torch.stack(centers), frame_axes_ras=torch.stack(frames),
        # Every anchor owns a complete local (t1, t2, n) frame.  The source
        # plane basis is the deterministic fallback until a supported
        # structural gradient refines the normal; it is not a global-z frame.
        frame_validity=torch.tensor([[True, True, True]] * count, dtype=torch.bool, device=reference.device),
        support_scales_mm=torch.full((count, 3), config.support_scale_mm, dtype=reference.dtype, device=reference.device),
        geometry_confidence=torch.stack(confidences), disagreement=torch.stack(disagreements),
        contributing_observation_ids=tuple(observation_groups), contributing_plane_hashes=tuple(plane_groups),
        provenance_hashes=tuple(provenance),
    )
