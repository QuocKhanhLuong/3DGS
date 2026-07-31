"""Teacher-free feature contracts and analytic evidence for T1-A."""

from .analytic import ANALYTIC_CHANNEL_NAMES, AnalyticFeatureOutput, analytic_feature_bank
from .conditioning import IntensityPerturbation, apply_intensity_perturbation
from .contracts import EncoderFeatureMaps, FeatureGridToPlaneTransform
from .encoder import EncoderConfig, EncoderParameterReport, EvidenceEncoder

__all__ = [
    "ANALYTIC_CHANNEL_NAMES",
    "AnalyticFeatureOutput",
    "EncoderFeatureMaps",
    "FeatureGridToPlaneTransform",
    "analytic_feature_bank",
    "EncoderConfig",
    "EncoderParameterReport",
    "EvidenceEncoder",
    "IntensityPerturbation",
    "apply_intensity_perturbation",
]
