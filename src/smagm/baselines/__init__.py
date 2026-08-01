"""Deterministic T1-A support and Gaussian bridge baselines."""

from .fixed_gaussian import (
    FixedGaussianHead,
    FixedGaussianHeadConfig,
    RawFixedGaussianOutput,
    construct_fixed_gaussians,
)
from .fixed_support import FixedSupportBatch, FixedSupportConfig, sample_fixed_supports
from .free_gaussian import FreeGaussianState
from .interpolation import SparseInterpolationConfig, construct_sparse_interpolation_gaussians
from .representations import RepresentationPlan, RepresentationVariant, resolve_representation_plan

__all__ = [
    "FixedGaussianHead",
    "FixedGaussianHeadConfig",
    "FixedSupportBatch",
    "FixedSupportConfig",
    "FreeGaussianState",
    "RawFixedGaussianOutput",
    "RepresentationPlan",
    "RepresentationVariant",
    "SparseInterpolationConfig",
    "construct_fixed_gaussians",
    "construct_sparse_interpolation_gaussians",
    "sample_fixed_supports",
    "resolve_representation_plan",
]
