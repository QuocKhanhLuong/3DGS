"""Typed T1-C optimization stages; stage names are policy, not architecture."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class TrainingStage(str, Enum):
    STRUCTURAL_WARMUP = "structural_warmup"
    JOINT_SPARSE_RECONSTRUCTION = "joint_sparse_reconstruction"
    RECONSTRUCTION_REFINEMENT = "reconstruction_refinement"


@dataclass(frozen=True)
class StageConfig:
    stage: TrainingStage = TrainingStage.JOINT_SPARSE_RECONSTRUCTION
    reconstruction_weight: float = 1.0
    structural_weight: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", TrainingStage(self.stage))
        if any(not math.isfinite(value) or value < 0.0 for value in (self.reconstruction_weight, self.structural_weight)):
            raise ValueError("stage weights must be finite and non-negative")
        if self.reconstruction_weight + self.structural_weight <= 0.0:
            raise ValueError("at least one stage objective must be enabled")
