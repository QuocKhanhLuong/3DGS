"""Executable sparse-MRI Gaussian reconstruction research scaffold."""

from ._torch_compat import ensure_eager_optimizer_compatibility

ensure_eager_optimizer_compatibility()

from .baselines.fixed_gaussian import (
    FixedGaussianHead,
    FixedGaussianHeadConfig,
    RawFixedGaussianOutput,
    construct_fixed_gaussians,
)
from .baselines.fixed_support import FixedSupportBatch, FixedSupportConfig, sample_fixed_supports
from .baselines import FreeGaussianState, RepresentationPlan, RepresentationVariant, SparseInterpolationConfig, resolve_representation_plan
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
from .anchors import AnchorBatch, AnchorBootstrapConfig, CachedPlaneEvidence, bootstrap_anchors
from .fields import GlobalStructuralField, GlobalStructuralFieldConfig, SharedStructuralField, StructuralFieldConfig, query_structural_field
from .memory import GaussianMemory, PropagationConfig, initialize_seed_memory, propagate_memory
from .state import PatientState, build_initial_patient_state, load_patient_state, save_patient_state
from .contracts.outputs import PlaneReconstruction, ReconstructionPackage, VolumeReconstruction
from .losses.reconstruction import ReconstructionLossConfig, ReconstructionLossResult, reconstruction_loss
from .training.episode import LegalEpisodeConfig, LegalEpisodeStep, build_legal_episode_step
from .training.representations import RepresentationEpisodeResult, build_representation_episode_step
from .training.trainer import T1CTrainer, TrainerConfig, TrainingStepOutput, TrainStepReport

__all__ = [
    "AccessLevel",
    "ALLOWED_COHORT_SPLITS",
    "ANALYTIC_CHANNEL_NAMES",
    "AcquisitionCostEntry",
    "AcquisitionCostSchedule",
    "AnchorBatch",
    "AnchorBootstrapConfig",
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
    "FreeGaussianState",
    "GaugeFixedLogAmplitude",
    "GaussianBatch",
    "GaussianMemory",
    "GlobalStructuralField",
    "GlobalStructuralFieldConfig",
    "ObservationLedger",
    "ObservationMeta",
    "LegalEpisodeConfig",
    "LegalEpisodeStep",
    "NormalizationConfig",
    "PhysicalPlane",
    "PatientState",
    "PlaneReconstruction",
    "PatientSplitRegistry",
    "PredictionReceiptCapability",
    "PredictionRegistrar",
    "PreprocessingRecord",
    "RawFixedGaussianOutput",
    "RawGaussianParameters",
    "RenderConfig",
    "RenderResult",
    "ReconstructionPackage",
    "RepresentationEpisodeResult",
    "RepresentationPlan",
    "RepresentationVariant",
    "ReconstructionLossConfig",
    "ReconstructionLossResult",
    "SlabProfile",
    "SourceAffineTransform",
    "SourceConvention",
    "SparseManifest",
    "SparseAvailabilityManifest",
    "SparseInterpolationConfig",
    "TargetGrid",
    "TargetCommitCapability",
    "T1CTrainer",
    "TrainerConfig",
    "TrainingStepOutput",
    "TrainStepReport",
    "TRAINING_LEDGER_SPLITS",
    "analytic_feature_bank",
    "bootstrap_anchors",
    "build_initial_patient_state",
    "CachedPlaneEvidence",
    "construct_fixed_gaussians",
    "build_legal_episode_step",
    "build_representation_episode_step",
    "decode_observation",
    "fix_log_amplitude_gauge",
    "gaussian_batch_from_raw",
    "initialize_seed_memory",
    "load_patient_state",
    "propagate_memory",
    "PropagationConfig",
    "query_structural_field",
    "render_plane",
    "reconstruction_loss",
    "resolve_representation_plan",
    "sample_fixed_supports",
    "save_patient_state",
    "SharedStructuralField",
    "StructuralFieldConfig",
    "VolumeReconstruction",
    "validate_patient_split_manifests",
]
