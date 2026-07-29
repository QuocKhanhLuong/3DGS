"""Compatibility import path for immutable sparse manifests."""

from ..contracts.observation import (
    AccessLevel,
    ALLOWED_COHORT_SPLITS,
    TRAINING_LEDGER_SPLITS,
    AvailabilityObservationMeta,
    PatientSplitRegistry,
    ObservationMeta,
    SparseAvailabilityManifest,
    SparseManifest,
    validate_patient_split_manifests,
)

__all__ = [
    "AccessLevel",
    "ALLOWED_COHORT_SPLITS",
    "AvailabilityObservationMeta",
    "PatientSplitRegistry",
    "ObservationMeta",
    "SparseManifest",
    "SparseAvailabilityManifest",
    "TRAINING_LEDGER_SPLITS",
    "validate_patient_split_manifests",
]
