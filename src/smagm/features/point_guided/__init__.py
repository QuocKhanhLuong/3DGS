"""Modular point-guided MRI frontend with bounded lazy public exports.

Importing contracts/configuration (or the additive PFGR-Lite package) must not
eagerly import the legacy model, Gate-E teacher/objective, data, or CLI
modules.  ``PointGuidedMRIModel`` retains its historical public name through
the module-level lazy attribute below.
"""

from .config import PointGuidedConfig
from .contracts import (
    EmptySparseSupportError,
    FrontendOutput,
    PointField,
    PointGuidedGeometryError,
    SparsePoU,
    VolumeGeometry,
)


def __getattr__(name: str):
    if name == "PointGuidedMRIModel":
        from .model import PointGuidedMRIModel

        return PointGuidedMRIModel
    raise AttributeError(name)

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
