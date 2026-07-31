"""Legal fixed-topology T1-C training contracts."""

from .episode import LegalEpisodeConfig, LegalEpisodeStep, build_legal_episode_step
from .metrics import gradient_norm, parameter_count
from .objective import T1CObjectiveConfig, resolve_objective
from .provenance import RunProvenance, canonical_hash, capture_run_provenance, module_state_hash
from .sampling import MatchedVariantSchedule, build_matched_variant_schedule
from .schedule import StageConfig, TrainingStage
from .trainer import T1CTrainer, TrainerConfig, TrainingStepOutput, TrainStepReport

__all__ = [
    "LegalEpisodeConfig",
    "LegalEpisodeStep",
    "MatchedVariantSchedule",
    "RunProvenance",
    "StageConfig",
    "T1CTrainer",
    "T1CObjectiveConfig",
    "TrainerConfig",
    "TrainingStage",
    "TrainingStepOutput",
    "TrainStepReport",
    "build_legal_episode_step",
    "build_matched_variant_schedule",
    "canonical_hash",
    "capture_run_provenance",
    "gradient_norm",
    "module_state_hash",
    "parameter_count",
    "resolve_objective",
]
