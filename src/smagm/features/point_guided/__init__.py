"""Modular point-guided MRI frontend with deliberate future-only interfaces."""

from .config import PointGuidedConfig
from .contracts import (
    EmptySparseSupportError,
    FrontendOutput,
    PointField,
    PointGuidedGeometryError,
    SparsePoU,
    VolumeGeometry,
)
from .model import PointGuidedMRIModel

__all__ = [
    "EmptySparseSupportError",
    "FrontendOutput",
    "PointField",
    "PointGuidedConfig",
    "PointGuidedGeometryError",
    "PointGuidedMRIModel",
    "SparsePoU",
    "VolumeGeometry",
]
