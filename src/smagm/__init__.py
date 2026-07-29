"""T0 legal physical forward operator for sparse MRI reconstruction research."""

from .contracts.coordinates import (
    PhysicalPlane,
    SourceAffineTransform,
    SourceConvention,
    TargetGrid,
)
from .contracts.observation import (
    AccessLevel,
    AvailabilityObservationMeta,
    ObservationLedger,
    ObservationMeta,
    SparseAvailabilityManifest,
    SparseManifest,
    validate_patient_split_manifests,
)
from .contracts.episode import (
    AcquisitionCostEntry,
    AcquisitionCostSchedule,
    DeploymentAcquisitionLedger,
    EpisodeAssignment,
    EpisodeController,
    EpisodeLedger,
    FrozenPatientState,
    PredictionReceiptCapability,
    PredictionRegistrar,
    TargetCommitCapability,
)
from .gaussians import (
    AmplitudeGaugePolicy,
    GaugeFixedLogAmplitude,
    GaussianBatch,
    RawGaussianParameters,
    fix_log_amplitude_gauge,
    gaussian_batch_from_raw,
)
from .renderer import RenderConfig, RenderResult, SlabProfile, render_plane

__all__ = [
    "AccessLevel",
    "AcquisitionCostEntry",
    "AcquisitionCostSchedule",
    "AmplitudeGaugePolicy",
    "AvailabilityObservationMeta",
    "DeploymentAcquisitionLedger",
    "EpisodeAssignment",
    "EpisodeController",
    "EpisodeLedger",
    "FrozenPatientState",
    "GaugeFixedLogAmplitude",
    "GaussianBatch",
    "ObservationLedger",
    "ObservationMeta",
    "PhysicalPlane",
    "PredictionReceiptCapability",
    "PredictionRegistrar",
    "RawGaussianParameters",
    "RenderConfig",
    "RenderResult",
    "SlabProfile",
    "SourceAffineTransform",
    "SourceConvention",
    "SparseManifest",
    "SparseAvailabilityManifest",
    "TargetGrid",
    "TargetCommitCapability",
    "fix_log_amplitude_gauge",
    "gaussian_batch_from_raw",
    "validate_patient_split_manifests",
    "render_plane",
]
