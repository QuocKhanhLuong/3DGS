"""Isolated evaluation over serialized immutable predictions."""

from .audit import AuditTarget, FreezeRecord, SerializedPredictions, evaluate_audit_targets, open_serialized_audit_targets, open_serialized_predictions
from .budget import BudgetCurve, summarize_budget_curve
from .medical_fidelity import ROIFidelity, evaluate_roi_fidelity
from .metrics import ReconstructionMetrics, compute_reconstruction_metrics
from .statistics import PairedSummary, paired_patient_summary
from .uncertainty import UncertaintyAssociation, evaluate_uncertainty_association

__all__ = [
    "AuditTarget", "BudgetCurve", "FreezeRecord", "PairedSummary", "ROIFidelity",
    "ReconstructionMetrics", "SerializedPredictions", "UncertaintyAssociation",
    "compute_reconstruction_metrics", "evaluate_audit_targets", "evaluate_roi_fidelity",
    "evaluate_uncertainty_association", "open_serialized_audit_targets", "open_serialized_predictions",
    "paired_patient_summary", "summarize_budget_curve",
]
