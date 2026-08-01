"""Physical lifting and context-only anchor bootstrap orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import torch

from .aggregation import AggregationConfig, CachedPlaneEvidence, aggregate_anchor_evidence
from .candidates import CandidateSelectionConfig, select_structural_candidates
from .consolidation import ConsolidationConfig, consolidate_candidates, physical_nms
from .contracts import AnchorBatch, LiftedCandidateBatch, StructuralCandidateBatch


@dataclass(frozen=True)
class AnchorBootstrapConfig:
    candidate: CandidateSelectionConfig = CandidateSelectionConfig()
    consolidation: ConsolidationConfig = ConsolidationConfig()
    aggregation: AggregationConfig = AggregationConfig()

    @property
    def config_hash(self) -> str:
        payload = {"candidate": self.candidate.__dict__, "consolidation": self.consolidation.__dict__, "aggregation": self.aggregation.__dict__}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def lift_candidates(candidates: StructuralCandidateBatch) -> LiftedCandidateBatch:
    indices = candidates.feature_indices_vu
    centers = candidates.transform.ras_mm_from_feature_vu(indices[:, 0], indices[:, 1])
    plane = candidates.transform.input_plane
    assert plane is not None
    axis_u = torch.as_tensor(plane.axis_u_ras, dtype=centers.dtype, device=centers.device)
    axis_v = torch.as_tensor(plane.axis_v_ras, dtype=centers.dtype, device=centers.device)
    # NIfTI tensor order is [v, u] while the source slice normal is retained
    # independently for provenance.  Anchor frames are a geometric basis and
    # must be right-handed; the cross-product normal is equivalent for the
    # covariance/support contract and avoids rejecting valid L/P/S affines.
    normal = torch.linalg.cross(axis_u, axis_v)
    normal = normal / torch.linalg.vector_norm(normal)
    axis = torch.stack((axis_u, axis_v, normal), dim=1)
    axes = axis.unsqueeze(0).expand(centers.shape[0], -1, -1).clone()
    plane_hash = candidates.transform.source_plane_hash
    return LiftedCandidateBatch(
        candidate_ids=candidates.candidate_ids, centers_ras_mm=centers, score=candidates.score,
        modality_ids=(candidates.modality_id,) * centers.shape[0],
        observation_ids=(candidates.observation_id,) * centers.shape[0],
        plane_hashes=(plane_hash,) * centers.shape[0], plane_axes_ras=axes,
    )


def bootstrap_anchors(
    evidence: tuple[CachedPlaneEvidence, ...], *, patient_id: str,
    modality_ids: tuple[str, ...], config: AnchorBootstrapConfig | None = None,
) -> AnchorBatch:
    """Build anchors only from explicitly context-only cached evidence."""

    if not evidence or not patient_id:
        raise ValueError("bootstrap requires patient identity and cached context evidence")
    if any(not item.context_only for item in evidence):
        raise PermissionError("anchor bootstrap rejects non-context or target-derived evidence")
    config = config or AnchorBootstrapConfig()
    lifted = []
    for item in evidence:
        raw = lift_candidates(select_structural_candidates(item.features, config=config.candidate))
        registration_tag = hashlib.sha256(item.registration_id.encode("utf-8")).hexdigest()[:16]
        lifted.append(LiftedCandidateBatch(
            candidate_ids=tuple(f"{candidate_id}:registration:{registration_tag}" for candidate_id in raw.candidate_ids),
            centers_ras_mm=raw.centers_ras_mm,
            score=raw.score,
            modality_ids=raw.modality_ids,
            observation_ids=raw.observation_ids,
            plane_hashes=raw.plane_hashes,
            plane_axes_ras=raw.plane_axes_ras,
        ))
    combined = LiftedCandidateBatch(
        candidate_ids=tuple(v for item in lifted for v in item.candidate_ids),
        centers_ras_mm=torch.cat([item.centers_ras_mm for item in lifted]),
        score=torch.cat([item.score for item in lifted]),
        modality_ids=tuple(v for item in lifted for v in item.modality_ids),
        observation_ids=tuple(v for item in lifted for v in item.observation_ids),
        plane_hashes=tuple(v for item in lifted for v in item.plane_hashes),
        plane_axes_ras=torch.cat([item.plane_axes_ras for item in lifted]),
    )
    suppressed = physical_nms(combined, radius_mm=config.consolidation.nms_radius_mm)
    geometry = consolidate_candidates(suppressed, config=config.consolidation)
    return aggregate_anchor_evidence(
        geometry, evidence, patient_id=patient_id, modality_ids=modality_ids, config=config.aggregation
    )
