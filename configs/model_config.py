"""Public configuration entry point for the point-guided MRI frontend.

The implementation lives with the model so tests and runtime use exactly one
typed contract. This file is intentionally a re-export, not a second source of
hyperparameters.
"""

from smagm.features.point_guided.config import PointGuidedConfig

__all__ = ["PointGuidedConfig"]
