"""Immutable geometry and observation-legality contracts."""

from .coordinates import PhysicalPlane, SourceAffineTransform, SourceConvention, TargetGrid
from .observation import (
    AccessLevel,
    AvailabilityObservationMeta,
    ObservationLedger,
    ObservationMeta,
    SparseAvailabilityManifest,
    SparseManifest,
    validate_patient_split_manifests,
)
from .episode import (
    AcquisitionCapability,
    AcquisitionCostEntry,
    AcquisitionCostSchedule,
    DeploymentAcquisitionLedger,
    EpisodeAssignment,
    EpisodeController,
    EpisodeLedger,
    FrozenPatientState,
    PredictionReceiptCapability,
    PredictionReceiptRecord,
    PredictionRegistrar,
    TargetCommitCapability,
    prediction_digest_from_render_result,
)

__all__ = [
    "AccessLevel",
    "AcquisitionCapability",
    "AcquisitionCostEntry",
    "AcquisitionCostSchedule",
    "AvailabilityObservationMeta",
    "DeploymentAcquisitionLedger",
    "EpisodeAssignment",
    "EpisodeController",
    "EpisodeLedger",
    "FrozenPatientState",
    "ObservationLedger",
    "ObservationMeta",
    "PhysicalPlane",
    "PredictionReceiptCapability",
    "PredictionReceiptRecord",
    "PredictionRegistrar",
    "SourceAffineTransform",
    "SourceConvention",
    "SparseManifest",
    "SparseAvailabilityManifest",
    "TargetGrid",
    "TargetCommitCapability",
    "prediction_digest_from_render_result",
    "validate_patient_split_manifests",
]
