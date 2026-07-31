"""Small teacher-free feature objectives used by T1-B."""

from .compose import ObjectiveResult, compose_objective
from .reconstruction import (
    EmptyReconstructionMaskError,
    ReconstructionLossConfig,
    ReconstructionLossResult,
    reconstruction_loss,
)

from .structural import (
    EmptyComparisonError,
    StructuralLossResult,
    appearance_sensitivity_loss,
    cross_modality_structural_consistency_loss,
    reliability_regularization_loss,
    structural_consistency_loss,
    structural_consistency_result,
    structural_variance_floor_loss,
)

__all__ = [
    "EmptyComparisonError",
    "EmptyReconstructionMaskError",
    "ObjectiveResult",
    "ReconstructionLossConfig",
    "ReconstructionLossResult",
    "StructuralLossResult",
    "appearance_sensitivity_loss",
    "cross_modality_structural_consistency_loss",
    "compose_objective",
    "reliability_regularization_loss",
    "reconstruction_loss",
    "structural_consistency_loss",
    "structural_consistency_result",
    "structural_variance_floor_loss",
]
