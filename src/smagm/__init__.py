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
from .data.io import DecodedObservation, DecoderConfig, decode_observation
from .data.normalization import NormalizationConfig, PreprocessingRecord
from .gaussians import (
    AmplitudeGaugePolicy,
    GaugeFixedLogAmplitude,
    GaussianBatch,
    RawGaussianParameters,
    fix_log_amplitude_gauge,
    gaussian_batch_from_raw,
)
from .renderer import RenderConfig, RenderResult, SlabProfile, render_plane
from .losses.reconstruction import ReconstructionLossConfig, ReconstructionLossResult, reconstruction_loss
from .training.episode import LegalEpisodeConfig, LegalEpisodeStep, build_legal_episode_step
from .training.trainer import T1CTrainer, TrainerConfig, TrainingStepOutput, TrainStepReport

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
    "DecodedObservation",
    "DecoderConfig",
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
    "LegalEpisodeConfig",
    "LegalEpisodeStep",
    "NormalizationConfig",
    "PhysicalPlane",
    "PatientSplitRegistry",
    "PredictionReceiptCapability",
    "PredictionRegistrar",
    "PreprocessingRecord",
    "RawFixedGaussianOutput",
    "RawGaussianParameters",
    "RenderConfig",
    "RenderResult",
    "ReconstructionLossConfig",
    "ReconstructionLossResult",
    "SlabProfile",
    "SourceAffineTransform",
    "SourceConvention",
    "SparseManifest",
    "SparseAvailabilityManifest",
    "TargetGrid",
    "TargetCommitCapability",
    "T1CTrainer",
    "TrainerConfig",
    "TrainingStepOutput",
    "TrainStepReport",
    "TRAINING_LEDGER_SPLITS",
    "analytic_feature_bank",
    "construct_fixed_gaussians",
    "build_legal_episode_step",
    "decode_observation",
    "fix_log_amplitude_gauge",
    "gaussian_batch_from_raw",
    "render_plane",
    "reconstruction_loss",
    "sample_fixed_supports",
    "validate_patient_split_manifests",
]
