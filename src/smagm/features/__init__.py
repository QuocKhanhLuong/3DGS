"""Teacher-free feature contracts and analytic evidence for T1-A."""

from .analytic import ANALYTIC_CHANNEL_NAMES, AnalyticFeatureOutput, analytic_feature_bank
from .contracts import EncoderFeatureMaps, FeatureGridToPlaneTransform

__all__ = [
    "ANALYTIC_CHANNEL_NAMES",
    "AnalyticFeatureOutput",
    "EncoderFeatureMaps",
    "FeatureGridToPlaneTransform",
    "analytic_feature_bank",
]
