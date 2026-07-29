"""T0 legal physical forward operator for sparse MRI reconstruction research."""

from .contracts.coordinates import (
    PhysicalPlane,
    SourceAffineTransform,
    SourceConvention,
    TargetGrid,
)
from .contracts.observation import (
    AccessLevel,
    ObservationLedger,
    ObservationMeta,
    SparseManifest,
    validate_patient_split_manifests,
)
from .gaussians import GaussianBatch
from .renderer import RenderConfig, RenderResult, SlabProfile, render_plane

__all__ = [
    "AccessLevel",
    "GaussianBatch",
    "ObservationLedger",
    "ObservationMeta",
    "PhysicalPlane",
    "RenderConfig",
    "RenderResult",
    "SlabProfile",
    "SourceAffineTransform",
    "SourceConvention",
    "SparseManifest",
    "TargetGrid",
    "validate_patient_split_manifests",
    "render_plane",
]
