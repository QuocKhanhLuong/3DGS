"""Compatibility import path for immutable sparse manifests."""

from ..contracts.observation import (
    AccessLevel,
    AvailabilityObservationMeta,
    ObservationMeta,
    SparseAvailabilityManifest,
    SparseManifest,
    validate_patient_split_manifests,
)

__all__ = [
    "AccessLevel",
    "AvailabilityObservationMeta",
    "ObservationMeta",
    "SparseManifest",
    "SparseAvailabilityManifest",
    "validate_patient_split_manifests",
]
