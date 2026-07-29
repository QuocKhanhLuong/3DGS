"""Immutable geometry and observation-legality contracts."""

from .coordinates import PhysicalPlane, SourceAffineTransform, SourceConvention, TargetGrid
from .observation import (
    AccessLevel,
    ObservationLedger,
    ObservationMeta,
    SparseManifest,
    validate_patient_split_manifests,
)

__all__ = [
    "AccessLevel",
    "ObservationLedger",
    "ObservationMeta",
    "PhysicalPlane",
    "SourceAffineTransform",
    "SourceConvention",
    "SparseManifest",
    "TargetGrid",
    "validate_patient_split_manifests",
]
