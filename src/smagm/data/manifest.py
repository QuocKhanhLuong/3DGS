"""Compatibility import path for immutable sparse manifests."""

from ..contracts.observation import (
    AccessLevel,
    ObservationMeta,
    SparseManifest,
    validate_patient_split_manifests,
)

__all__ = [
    "AccessLevel",
    "ObservationMeta",
    "SparseManifest",
    "validate_patient_split_manifests",
]
