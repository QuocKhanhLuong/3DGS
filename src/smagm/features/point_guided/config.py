"""Explicit, locked configuration for the point-guided MRI frontend."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal, Sequence

from .contracts import NUM_COARSE_SEMANTIC_CLASSES


def _positive_floats(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(value <= 0.0 or not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain positive finite values")
    return result


@dataclass(frozen=True)
class PointGuidedConfig:
    """Configuration for the locked frontend, not future trajectory modules.

    ``directional_offsets_mm`` and all affinity choices are deliberately
    explicit because changing them changes the research contract.
    """

    num_semantic_classes: int
    input_modalities: tuple[str, str, str] = ("T1", "T2", "FLAIR")
    num_points: int = 2048
    alternative_num_points: int = 3072
    support_radius_mm: float = 4.0
    max_displacement_mm: float = 2.0
    directional_offsets_mm: tuple[float, ...] = (1.0, 2.0, 3.0)
    coarse_backbone: str = "MedicalNet_ResNet10"
    freeze_coarse_backbone: bool = True
    detach_backbone_features: bool = True
    spectral_tap: Literal["conv1_pre_maxpool", "layer1"] = "conv1_pre_maxpool"
    projection_mode: Literal[
        "mean",
        "max",
        "pointwise_weighted",
        "axis_local_weighted",
    ] = "axis_local_weighted"
    anchor_norm: Literal["none", "band_gn"] = "none"
    medicalnet_checkpoint_path: str | Path | None = None
    medicalnet_checkpoint_sha256: str | None = None
    require_pretrained_backbone: bool = False
    offset_hidden_channels: int = 128
    point_candidate_multiplier: int = 4
    max_local_voxels_per_point: int = 4096
    semantic_distance: str = "L1"
    spatial_kernel: str = "quadratic_compact"
    affinity_composition: str = "multiplication"
    pou_normalization: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.num_semantic_classes, int)
            or isinstance(self.num_semantic_classes, bool)
            or self.num_semantic_classes != NUM_COARSE_SEMANTIC_CLASSES
        ):
            raise ValueError("num_semantic_classes is locked to exactly 3 production coarse semantic classes")
        if tuple(self.input_modalities) != ("T1", "T2", "FLAIR"):
            raise ValueError("input_modalities is locked to ('T1', 'T2', 'FLAIR')")
        if not isinstance(self.freeze_coarse_backbone, bool):
            raise ValueError("freeze_coarse_backbone must be a bool")
        if not isinstance(self.detach_backbone_features, bool):
            raise ValueError("detach_backbone_features must be a bool")
        if self.spectral_tap not in ("conv1_pre_maxpool", "layer1"):
            raise ValueError("spectral_tap must be 'conv1_pre_maxpool' or 'layer1'")
        if self.projection_mode not in (
            "mean",
            "max",
            "pointwise_weighted",
            "axis_local_weighted",
        ):
            raise ValueError(
                "projection_mode must be 'mean', 'max', 'pointwise_weighted', or "
                "'axis_local_weighted'"
            )
        if not isinstance(self.anchor_norm, str) or self.anchor_norm not in ("none", "band_gn"):
            raise ValueError("anchor_norm must be 'none' or 'band_gn'")
        for name in (
            "num_points",
            "alternative_num_points",
            "offset_hidden_channels",
            "point_candidate_multiplier",
            "max_local_voxels_per_point",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        try:
            support_radius_mm = float(self.support_radius_mm)
            max_displacement_mm = float(self.max_displacement_mm)
        except (TypeError, ValueError) as error:
            raise ValueError("support_radius_mm and max_displacement_mm must be numeric") from error
        if (
            support_radius_mm <= 0.0
            or max_displacement_mm <= 0.0
            or not math.isfinite(support_radius_mm)
            or not math.isfinite(max_displacement_mm)
        ):
            raise ValueError("support_radius_mm and max_displacement_mm must be positive and finite")
        # The support is a research constant, not a tunable kernel width.  A
        # caller may choose a smaller safety cap for diagnostic experiments,
        # but can never relax the 2-mm physical bound.
        if not math.isclose(support_radius_mm, 4.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("support_radius_mm is locked to exactly 4.0 mm")
        if max_displacement_mm > 2.0:
            raise ValueError("max_displacement_mm must not exceed the locked 2.0-mm bound")
        object.__setattr__(self, "support_radius_mm", support_radius_mm)
        object.__setattr__(self, "max_displacement_mm", max_displacement_mm)
        offsets = _positive_floats(self.directional_offsets_mm, "directional_offsets_mm")
        if offsets != (1.0, 2.0, 3.0):
            raise ValueError("directional_offsets_mm is locked to (1.0, 2.0, 3.0)")
        object.__setattr__(self, "directional_offsets_mm", offsets)
        if self.coarse_backbone != "MedicalNet_ResNet10":
            raise ValueError("coarse_backbone is locked to MedicalNet_ResNet10")
        if self.semantic_distance != "L1":
            raise ValueError("semantic_distance is locked to L1")
        if self.spatial_kernel != "quadratic_compact":
            raise ValueError("spatial_kernel is locked to quadratic_compact")
        if self.affinity_composition != "multiplication" or not self.pou_normalization:
            raise ValueError("the locked PoU uses multiplicative normalized affinities")
        if self.medicalnet_checkpoint_sha256 is not None:
            digest = self.medicalnet_checkpoint_sha256.lower()
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("medicalnet_checkpoint_sha256 must be a SHA-256 hex digest")
            object.__setattr__(self, "medicalnet_checkpoint_sha256", digest)
        if self.require_pretrained_backbone:
            if self.medicalnet_checkpoint_path is None:
                raise ValueError("require_pretrained_backbone requires medicalnet_checkpoint_path")
            if self.medicalnet_checkpoint_sha256 is None:
                raise ValueError(
                    "require_pretrained_backbone requires a SHA-256 digest from an approved official manifest"
                )
