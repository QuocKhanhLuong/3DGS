"""Resolved T1-C optimization stages, independent of model APIs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class TrainingStage(str, Enum):
    STRUCTURAL_WARMUP = "structural_warmup"
    JOINT_RECONSTRUCTION = "joint_reconstruction"
    RECONSTRUCTION_DOMINANT = "reconstruction_dominant"


@dataclass(frozen=True)
class StageConfig:
    """Weights and legal target access for one bounded optimization stage."""

    stage: TrainingStage = TrainingStage.JOINT_RECONSTRUCTION
    reconstruction_weight: float = 1.0
    structural_weight: float = 0.0
    auxiliary_only: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", TrainingStage(self.stage))
        if any(not math.isfinite(value) or value < 0.0 for value in (self.reconstruction_weight, self.structural_weight)):
            raise ValueError("stage weights must be finite and non-negative")
        if self.reconstruction_weight + self.structural_weight <= 0.0:
            raise ValueError("at least one stage objective must be enabled")
        if self.auxiliary_only and self.reconstruction_weight != 0.0:
            raise ValueError("an auxiliary-only stage must set reconstruction_weight to zero")
        if self.stage is TrainingStage.STRUCTURAL_WARMUP and self.auxiliary_only is False and self.reconstruction_weight == 0.0:
            # A caller can run a diagnostic warm-up with target reconstruction,
            # but it must say so explicitly rather than silently exposing one.
            raise ValueError("a zero-reconstruction structural warm-up must be marked auxiliary_only")

    def to_dict(self) -> dict[str, object]:
        return {
            "auxiliary_only": self.auxiliary_only,
            "reconstruction_weight": self.reconstruction_weight,
            "stage": self.stage.value,
            "structural_weight": self.structural_weight,
        }


@dataclass(frozen=True)
class TrainingSchedule:
    """A deterministic step-index policy for warm-up, joint, and refinement."""

    structural_warmup_steps: int = 0
    joint_reconstruction_steps: int = 1
    structural_warmup: StageConfig = StageConfig(
        stage=TrainingStage.STRUCTURAL_WARMUP,
        reconstruction_weight=0.0,
        structural_weight=1.0,
        auxiliary_only=True,
    )
    joint_reconstruction: StageConfig = StageConfig(
        stage=TrainingStage.JOINT_RECONSTRUCTION,
        reconstruction_weight=1.0,
        structural_weight=1.0,
    )
    reconstruction_dominant: StageConfig = StageConfig(
        stage=TrainingStage.RECONSTRUCTION_DOMINANT,
        reconstruction_weight=1.0,
        structural_weight=0.1,
    )

    def __post_init__(self) -> None:
        for name in ("structural_warmup_steps", "joint_reconstruction_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        expected = (
            ("structural_warmup", TrainingStage.STRUCTURAL_WARMUP),
            ("joint_reconstruction", TrainingStage.JOINT_RECONSTRUCTION),
            ("reconstruction_dominant", TrainingStage.RECONSTRUCTION_DOMINANT),
        )
        for name, stage in expected:
            config = getattr(self, name)
            if not isinstance(config, StageConfig) or config.stage is not stage:
                raise ValueError(f"{name} must describe the {stage.value} stage")

    def stage_for_step(self, completed_steps: int) -> StageConfig:
        if not isinstance(completed_steps, int) or completed_steps < 0:
            raise ValueError("completed_steps must be a non-negative integer")
        if completed_steps < self.structural_warmup_steps:
            return self.structural_warmup
        if completed_steps < self.structural_warmup_steps + self.joint_reconstruction_steps:
            return self.joint_reconstruction
        return self.reconstruction_dominant

    def to_dict(self) -> dict[str, object]:
        return {
            "joint_reconstruction": self.joint_reconstruction.to_dict(),
            "joint_reconstruction_steps": self.joint_reconstruction_steps,
            "reconstruction_dominant": self.reconstruction_dominant.to_dict(),
            "structural_warmup": self.structural_warmup.to_dict(),
            "structural_warmup_steps": self.structural_warmup_steps,
        }
