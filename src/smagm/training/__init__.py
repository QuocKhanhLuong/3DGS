"""Legal fixed-topology T1-C training contracts."""

from .episode import (
    ContextEvidence,
    ContextOnlyEpisodeStep,
    LegalEpisodeConfig,
    LegalEpisodeStep,
    build_context_only_episode_step,
    build_legal_episode_step,
)
from .metrics import gradient_norm, parameter_count
from .objective import T1CObjectiveConfig, T1CObjectiveResult, compose_t1c_objective, resolve_objective
from .provenance import RunProvenance, canonical_hash, capture_run_provenance, module_state_hash
from .sampling import MatchedExperimentIdentity, MatchedVariantSchedule, build_matched_variant_schedule
from .schedule import StageConfig, TrainingSchedule, TrainingStage
from .trainer import T1CTrainer, TrainerConfig, TrainingStepOutput, TrainStepReport

__all__ = [
    "ContextEvidence",
    "ContextOnlyEpisodeStep",
    "LegalEpisodeConfig",
    "LegalEpisodeStep",
    "MatchedExperimentIdentity",
    "MatchedVariantSchedule",
    "RunProvenance",
    "StageConfig",
    "TrainingSchedule",
    "T1CTrainer",
    "T1CObjectiveConfig",
    "T1CObjectiveResult",
    "TrainerConfig",
    "TrainingStage",
    "TrainingStepOutput",
    "TrainStepReport",
    "build_legal_episode_step",
    "build_context_only_episode_step",
    "build_matched_variant_schedule",
    "canonical_hash",
    "capture_run_provenance",
    "gradient_norm",
    "module_state_hash",
    "parameter_count",
    "compose_t1c_objective",
    "resolve_objective",
]
