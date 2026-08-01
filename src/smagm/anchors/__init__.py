"""Physical support-anchor construction from legal cached evidence."""

from .aggregation import AggregationConfig, CachedPlaneEvidence, EmptyAnchorEvidenceError, aggregate_anchor_evidence
from .bootstrap import AnchorBootstrapConfig, bootstrap_anchors, lift_candidates
from .candidates import CandidateSelectionConfig, select_structural_candidates
from .consolidation import ConsolidationConfig, consolidate_candidates, physical_nms
from .contracts import AnchorBatch, AnchorGeometryBatch, LiftedCandidateBatch, StructuralCandidateBatch
from .frames import refine_frames_from_gradients
from .index import query_anchor_neighbors

__all__ = [
    "AggregationConfig", "AnchorBatch", "AnchorBootstrapConfig", "AnchorGeometryBatch",
    "CachedPlaneEvidence", "CandidateSelectionConfig", "ConsolidationConfig",
    "EmptyAnchorEvidenceError", "LiftedCandidateBatch", "StructuralCandidateBatch",
    "aggregate_anchor_evidence", "bootstrap_anchors", "consolidate_candidates",
    "lift_candidates", "physical_nms", "query_anchor_neighbors",
    "refine_frames_from_gradients", "select_structural_candidates",
]
