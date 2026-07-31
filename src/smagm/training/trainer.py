"""Optimizer ownership for legal T1-C episode steps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os
import tempfile

import torch

from ..baselines.fixed_gaussian import FixedGaussianHead
from ..contracts.episode import EpisodeAssignment, EpisodeLedger
from ..features.encoder import EvidenceEncoder
from .episode import LegalEpisodeConfig, LegalEpisodeStep, build_legal_episode_step
from .metrics import gradient_norm


@dataclass(frozen=True)
class TrainerConfig:
    gradient_clip_norm: float | None = 10.0
    accumulation_steps: int = 1
    precision: str = "float32"
    validation_interval: int = 1
    checkpoint_interval: int = 1
    early_stopping_patience: int | None = None

    def __post_init__(self) -> None:
        if self.gradient_clip_norm is not None and (
            not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0
        ):
            raise ValueError("gradient_clip_norm must be None or positive and finite")
        if not isinstance(self.accumulation_steps, int) or self.accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be a positive integer")
        if self.precision not in ("float32", "float64"):
            raise ValueError("precision must be float32 or float64")
        for name in ("validation_interval", "checkpoint_interval"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.early_stopping_patience is not None and (
            not isinstance(self.early_stopping_patience, int) or self.early_stopping_patience <= 0
        ):
            raise ValueError("early_stopping_patience must be None or a positive integer")


@dataclass(frozen=True)
class TrainStepReport:
    variant: str
    assignment_hash: str
    state_version: str
    receipt_record_hash: str
    audit_hash: str
    target_id: str
    legal_target_pixel_count: int
    supported_fraction: float
    loss: float
    encoder_gradient_norm: float
    head_gradient_norm: float
    cache_bytes: int
    support_count: int
    optimizer_updated: bool
    step_index: int


@dataclass(frozen=True)
class TrainingStepOutput:
    step: LegalEpisodeStep
    report: TrainStepReport


class T1CTrainer:
    """One legal optimizer-step controller; it never bypasses the episode path."""

    def __init__(
        self,
        *,
        encoder: EvidenceEncoder,
        gaussian_head: FixedGaussianHead,
        optimizer: torch.optim.Optimizer,
        episode_config: LegalEpisodeConfig | None = None,
        trainer_config: TrainerConfig | None = None,
    ) -> None:
        if not isinstance(encoder, EvidenceEncoder) or not isinstance(gaussian_head, FixedGaussianHead):
            raise TypeError("trainer requires the T1-B encoder and fixed Gaussian head")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch optimizer")
        owned = {id(parameter) for parameter in list(encoder.parameters()) + list(gaussian_head.parameters())}
        optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        if owned != optimized:
            raise ValueError("optimizer parameters must exactly match encoder and Gaussian head")
        self.encoder = encoder
        self.gaussian_head = gaussian_head
        self.optimizer = optimizer
        self.episode_config = episode_config or LegalEpisodeConfig()
        self.trainer_config = trainer_config or TrainerConfig()
        expected_dtype = torch.float32 if self.trainer_config.precision == "float32" else torch.float64
        if any(parameter.dtype != expected_dtype for parameter in list(encoder.parameters()) + list(gaussian_head.parameters())):
            raise ValueError("trainer precision must match encoder and Gaussian-head parameter dtype")
        self._accumulation_cursor = 0
        self._step_index = 0

    def train_step(
        self,
        *,
        ledger: EpisodeLedger,
        assignment: EpisodeAssignment,
        target_id: str,
    ) -> TrainingStepOutput:
        if self._accumulation_cursor == 0:
            self.optimizer.zero_grad(set_to_none=True)
        step = build_legal_episode_step(
            ledger=ledger,
            assignment=assignment,
            target_id=target_id,
            encoder=self.encoder,
            gaussian_head=self.gaussian_head,
            config=self.episode_config,
        )
        if not bool(torch.isfinite(step.loss.total)):
            raise FloatingPointError("non-finite legal episode loss")
        (step.loss.total / self.trainer_config.accumulation_steps).backward()
        encoder_norm = gradient_norm(self.encoder.parameters())
        head_norm = gradient_norm(self.gaussian_head.parameters())
        if head_norm <= 0.0:
            raise FloatingPointError("Gaussian head received no finite training gradient")
        if self.encoder.config.variant in ("e1", "e2") and encoder_norm <= 0.0:
            raise FloatingPointError("learned encoder received no finite training gradient")
        parameters = list(self.encoder.parameters()) + list(self.gaussian_head.parameters())
        self._accumulation_cursor += 1
        optimizer_updated = self._accumulation_cursor == self.trainer_config.accumulation_steps
        if optimizer_updated:
            if self.trainer_config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters, self.trainer_config.gradient_clip_norm, error_if_nonfinite=True)
            self.optimizer.step()
            self._accumulation_cursor = 0
        self._step_index += 1
        report = TrainStepReport(
            variant=self.encoder.config.variant,
            assignment_hash=assignment.assignment_hash,
            state_version=step.state_version,
            receipt_record_hash=step.receipt_record_hash,
            audit_hash=step.audit_hash,
            target_id=target_id,
            legal_target_pixel_count=step.loss.legal_pixel_count,
            supported_fraction=step.loss.supported_fraction,
            loss=float(step.loss.total.detach().cpu()),
            encoder_gradient_norm=encoder_norm,
            head_gradient_norm=head_norm,
            cache_bytes=step.cache_bytes,
            support_count=step.support_count,
            optimizer_updated=optimizer_updated,
            step_index=self._step_index,
        )
        return TrainingStepOutput(step, report)

    def checkpoint_state(self) -> dict[str, object]:
        """Return a CPU-serializable checkpoint bound to the exact contracts."""

        return {
            "schema": "smagm-t1c-checkpoint-v1",
            "variant": self.encoder.config.variant,
            "encoder_config_hash": self.encoder.config.config_hash,
            "episode_config_hash": self.episode_config.config_hash,
            "trainer_config": self.trainer_config.__dict__,
            "encoder": self.encoder.state_dict(),
            "gaussian_head": self.gaussian_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_index": self._step_index,
            "accumulation_cursor": self._accumulation_cursor,
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        """Atomically serialize a checkpoint; artifact provenance is external."""

        destination = Path(path)
        if not destination.parent.is_dir():
            raise FileNotFoundError("checkpoint parent directory does not exist")
        handle, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        os.close(handle)
        try:
            torch.save(self.checkpoint_state(), temporary)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return destination

    def load_checkpoint(self, path: str | Path) -> None:
        """Load only the typed T1-C state schema using safe tensor loading."""

        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("schema") != "smagm-t1c-checkpoint-v1":
            raise ValueError("checkpoint schema is not a T1-C checkpoint")
        if payload.get("variant") != self.encoder.config.variant:
            raise ValueError("checkpoint encoder variant mismatch")
        if payload.get("encoder_config_hash") != self.encoder.config.config_hash:
            raise ValueError("checkpoint encoder configuration mismatch")
        if payload.get("episode_config_hash") != self.episode_config.config_hash:
            raise ValueError("checkpoint episode configuration mismatch")
        if payload.get("trainer_config") != self.trainer_config.__dict__:
            raise ValueError("checkpoint trainer configuration mismatch")
        self.encoder.load_state_dict(payload["encoder"])
        self.gaussian_head.load_state_dict(payload["gaussian_head"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self._step_index = int(payload["step_index"])
        self._accumulation_cursor = int(payload["accumulation_cursor"])
