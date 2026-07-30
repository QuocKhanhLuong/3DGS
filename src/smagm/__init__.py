"""Executable sparse-MRI Gaussian reconstruction research scaffold."""

from .baselines.fixed_gaussian import (
    FixedGaussianHead,
    FixedGaussianHeadConfig,
    RawFixedGaussianOutput,
    construct_fixed_gaussians,
)
from .baselines.fixed_support import FixedSupportBatch, FixedSupportConfig, sample_fixed_supports
from .contracts.coordinates import (
    PhysicalPlane,
    SourceAffineTransform,
    SourceConvention,
    TargetGrid,
)
from .contracts.observation import (
    AccessLevel,
    ALLOWED_COHORT_SPLITS,
    TRAINING_LEDGER_SPLITS,
    AvailabilityObservationMeta,
    PatientSplitRegistry,
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
from .features.analytic import ANALYTIC_CHANNEL_NAMES, AnalyticFeatureOutput, analytic_feature_bank
from .features.contracts import EncoderFeatureMaps, FeatureGridToPlaneTransform
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
    "ALLOWED_COHORT_SPLITS",
    "ANALYTIC_CHANNEL_NAMES",
    "AcquisitionCostEntry",
    "AcquisitionCostSchedule",
    "AmplitudeGaugePolicy",
    "AnalyticFeatureOutput",
    "AvailabilityObservationMeta",
    "DeploymentAcquisitionLedger",
    "EncoderFeatureMaps",
    "EpisodeAssignment",
    "EpisodeController",
    "EpisodeLedger",
    "FeatureGridToPlaneTransform",
    "FixedGaussianHead",
    "FixedGaussianHeadConfig",
    "FixedSupportBatch",
    "FixedSupportConfig",
    "FrozenPatientState",
    "GaugeFixedLogAmplitude",
    "GaussianBatch",
    "ObservationLedger",
    "ObservationMeta",
    "PhysicalPlane",
    "PatientSplitRegistry",
    "PredictionReceiptCapability",
    "PredictionRegistrar",
    "RawFixedGaussianOutput",
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
    "TRAINING_LEDGER_SPLITS",
    "analytic_feature_bank",
    "construct_fixed_gaussians",
    "fix_log_amplitude_gauge",
    "gaussian_batch_from_raw",
    "render_plane",
    "sample_fixed_supports",
    "validate_patient_split_manifests",
]
