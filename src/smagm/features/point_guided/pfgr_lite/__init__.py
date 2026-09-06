"""Additive PFGR-Lite frontend package.

Configuration, contracts, geometry, and provenance are safe to import in a
target-free process.  The composition/model symbol is resolved lazily so a
configuration import cannot pull in Gate-E teacher/objective or CLI modules.
"""

from .config import (
    EffectTeacherConfig,
    PFGRLiteConfig,
    PFGRPolicyConfig,
    StaticSynthesisConfig,
    ValueModelConfig,
)
from .provenance import (
    CalibrationIdentity,
    ProducerCompatibility,
    SourceProvenance,
    ValueFitIdentity,
    batchnorm_state_digest,
    canonical_digest,
    canonical_json,
    module_parameter_digest,
    module_state_digest,
    tensor_digest,
)
from .static_geometry import (
    FeatureLattice,
    MultiScaleFeatureGeometry,
    derive_feature_lattices,
    derive_multiscale_feature_geometries,
    derive_multiscale_feature_geometry,
    derive_static_feature_geometry,
    resample_plane_between_lattices,
    sample_source_to_lattice,
)
from .static_synthesis import (
    StaticSynthesisHead,
)
from .types import (
    ActionProposal,
    ActionProposalBatch,
    CompletedBehaviorTrace,
    Decision,
    DescriptorBundle,
    GainCalibration,
    GainLabel,
    InferenceBundle,
    ObservationContext,
    OperationCounters,
    PFGRRouteResult,
    PFGRState,
    ProducerDependencies,
    ResumeState,
    SparseFootprint,
    StageState,
    ValueBankManifest,
    V_DESCRIPTOR_DIMS,
    build_descriptor_bundle,
    clone_dynamic_planes,
    dynamic_planes_digest,
)


def __getattr__(name: str):
    if name == "PFGRLiteModel":
        from .model import PFGRLiteModel

        return PFGRLiteModel
    raise AttributeError(name)


__all__ = [
    "ActionProposal",
    "ActionProposalBatch",
    "CalibrationIdentity",
    "CompletedBehaviorTrace",
    "Decision",
    "DescriptorBundle",
    "EffectTeacherConfig",
    "FeatureLattice",
    "GainCalibration",
    "GainLabel",
    "InferenceBundle",
    "MultiScaleFeatureGeometry",
    "ObservationContext",
    "OperationCounters",
    "PFGRLiteConfig",
    "PFGRLiteModel",
    "PFGRPolicyConfig",
    "PFGRRouteResult",
    "PFGRState",
    "ProducerCompatibility",
    "ProducerDependencies",
    "ResumeState",
    "SourceProvenance",
    "SparseFootprint",
    "StageState",
    "StaticSynthesisConfig",
    "StaticSynthesisHead",
    "ValueBankManifest",
    "ValueFitIdentity",
    "ValueModelConfig",
    "V_DESCRIPTOR_DIMS",
    "batchnorm_state_digest",
    "build_descriptor_bundle",
    "canonical_digest",
    "canonical_json",
    "clone_dynamic_planes",
    "derive_feature_lattices",
    "derive_multiscale_feature_geometries",
    "derive_multiscale_feature_geometry",
    "derive_static_feature_geometry",
    "dynamic_planes_digest",
    "module_parameter_digest",
    "module_state_digest",
    "resample_plane_between_lattices",
    "sample_source_to_lattice",
    "tensor_digest",
]
