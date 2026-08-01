"""Optimizer ownership and checkpoint safety for legal episodic training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from types import MappingProxyType
from typing import Mapping

import torch

from ..baselines.fixed_gaussian import FixedGaussianHead
from ..contracts.episode import EpisodeAssignment, EpisodeLedger
from ..features.conditioning import IntensityPerturbation, apply_intensity_perturbation
from ..features.encoder import EvidenceEncoder
from ..losses.structural import (
    EmptyComparisonError,
    appearance_sensitivity_loss,
    reliability_regularization_loss,
    structural_consistency_loss,
    structural_variance_floor_loss,
)
from .episode import (
    ContextEvidence,
    ContextOnlyEpisodeStep,
    LegalEpisodeConfig,
    LegalEpisodeStep,
    build_context_only_episode_step,
    build_legal_episode_step,
)
from .metrics import gradient_norm
from .objective import T1CObjectiveConfig, T1CObjectiveResult, compose_t1c_objective
from .schedule import TrainingSchedule


def _hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrainerConfig:
    gradient_clip_norm: float | None = 10.0
    accumulation_steps: int = 1
    precision: str = "float32"
    checkpoint_interval: int = 1
    schedule: TrainingSchedule = TrainingSchedule()
    objective: T1CObjectiveConfig = T1CObjectiveConfig()

    def __post_init__(self) -> None:
        if self.gradient_clip_norm is not None and (
            not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0
        ):
            raise ValueError("gradient_clip_norm must be None or positive and finite")
        if not isinstance(self.accumulation_steps, int) or self.accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be a positive integer")
        if self.precision not in ("float32", "float64"):
            raise ValueError("precision must be float32 or float64")
        for name in ("checkpoint_interval",):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.schedule, TrainingSchedule) or not isinstance(self.objective, T1CObjectiveConfig):
            raise TypeError("schedule and objective must use typed T1-C contracts")

    def to_dict(self) -> dict[str, object]:
        return {
            "accumulation_steps": self.accumulation_steps,
            "checkpoint_interval": self.checkpoint_interval,
            "gradient_clip_norm": self.gradient_clip_norm,
            "objective": self.objective.to_dict(),
            "precision": self.precision,
            "schedule": self.schedule.to_dict(),
        }


@dataclass(frozen=True)
class TrainStepReport:
    variant: str
    stage: str
    assignment_hash: str
    state_version: str | None
    receipt_record_hash: str | None
    audit_hash: str | None
    target_id: str | None
    reconstruction_intensity_loss: float | None
    reconstruction_gradient_loss: float | None
    reconstruction_frequency_loss: float | None
    legal_target_pixel_count: int
    supported_fraction: float
    unsupported_fraction: float
    structural_components: Mapping[str, float]
    inactive_components: Mapping[str, str]
    loss: float
    encoder_gradient_norm: float
    head_gradient_norm: float
    cache_bytes: int
    encoder_runtime_seconds: float
    encoder_forward_passes: int
    support_count: int
    support_topology_hash: str
    optimizer_updated: bool
    step_index: int
    optimizer_step_index: int

    def __post_init__(self) -> None:
        if (
            not self.stage
            or self.legal_target_pixel_count < 0
            or self.cache_bytes <= 0
            or self.support_count <= 0
            or not math.isfinite(self.encoder_runtime_seconds)
            or self.encoder_runtime_seconds < 0.0
            or self.encoder_forward_passes <= 0
        ):
            raise ValueError("training report contains invalid stage or episode diagnostics")
        if not 0.0 <= self.supported_fraction <= 1.0 or not 0.0 <= self.unsupported_fraction <= 1.0:
            raise ValueError("support fractions must lie in [0, 1]")
        object.__setattr__(self, "structural_components", MappingProxyType(dict(self.structural_components)))
        object.__setattr__(self, "inactive_components", MappingProxyType(dict(self.inactive_components)))


@dataclass(frozen=True)
class TrainingStepOutput:
    step: LegalEpisodeStep | ContextOnlyEpisodeStep
    objective: T1CObjectiveResult
    report: TrainStepReport


class T1CTrainer:
    """One legal optimizer controller; target access never bypasses episode.py."""

    def __init__(
        self,
        *,
        encoder: EvidenceEncoder,
        gaussian_head: FixedGaussianHead,
        optimizer: torch.optim.Optimizer,
        episode_config: LegalEpisodeConfig | None = None,
        trainer_config: TrainerConfig | None = None,
        matched_experiment_identity: str = "",
        resolved_config_hash: str = "",
        sampler_state_hash: str = "",
        manifest_hash: str = "",
        split_registry_hash: str = "",
        scheduled_assignment_hashes: tuple[str, ...] = (),
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
        for name, digest in (
            ("matched_experiment_identity", matched_experiment_identity),
            ("resolved_config_hash", resolved_config_hash),
            ("sampler_state_hash", sampler_state_hash),
            ("manifest_hash", manifest_hash),
            ("split_registry_hash", split_registry_hash),
        ):
            if digest and (len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)):
                raise ValueError(f"{name} must be empty or a SHA-256 digest")
        self.matched_experiment_identity = matched_experiment_identity
        self.resolved_config_hash = resolved_config_hash
        self.sampler_state_hash = sampler_state_hash
        if len(set(scheduled_assignment_hashes)) != len(scheduled_assignment_hashes):
            raise ValueError("scheduled_assignment_hashes must be unique")
        for assignment_hash in scheduled_assignment_hashes:
            if len(assignment_hash) != 64 or any(char not in "0123456789abcdef" for char in assignment_hash):
                raise ValueError("scheduled_assignment_hashes must contain SHA-256 digests")
        self.manifest_hash = manifest_hash
        self.split_registry_hash = split_registry_hash
        self.scheduled_assignment_hashes = tuple(scheduled_assignment_hashes)
        self._accumulation_cursor = 0
        self._step_index = 0
        self._optimizer_step_index = 0
        self._last_assignment_hash = ""
        self._last_manifest_hash = ""
        self._last_split_registry_hash = ""
        self._schedule_cursor = 0

    def _structural_components(
        self, evidence: tuple[ContextEvidence, ...]
    ) -> tuple[dict[str, torch.Tensor], dict[str, str], float, int]:
        values: dict[str, list[torch.Tensor]] = {}
        inactive: dict[str, str] = {}
        declared = self.trainer_config.objective.weights
        runtime_seconds = 0.0
        forward_passes = 0
        for item in evidence:
            perturbed_image = apply_intensity_perturbation(item.normalized_image, IntensityPerturbation(scale=1.05, bias=0.01))
            started = time.perf_counter()
            perturbed = self.encoder(perturbed_image, item.plane, item.modality_id, item.valid_mask)
            runtime_seconds += time.perf_counter() - started
            forward_passes += 1
            terms = {
                "structural_consistency": lambda: structural_consistency_loss(
                    item.features.structural, perturbed.structural, item.features.valid_feature_mask
                ),
                "appearance_sensitivity": lambda: appearance_sensitivity_loss(
                    item.features.appearance, perturbed.appearance, item.features.valid_feature_mask
                ),
                "reliability_regularization": lambda: reliability_regularization_loss(
                    item.features.reliability, item.features.valid_feature_mask
                ),
                "variance_floor": lambda: structural_variance_floor_loss(
                    item.features.structural, item.features.valid_feature_mask
                ),
            }
            for name in declared:
                if name not in terms:
                    inactive[name] = "UNSUPPORTED_DECLARED_STRUCTURAL_COMPONENT"
                    continue
                try:
                    values.setdefault(name, []).append(terms[name]())
                except EmptyComparisonError as exc:
                    inactive[name] = f"EMPTY_LEGAL_COMPARISON: {exc}"
        active = {name: torch.stack(terms).mean() for name, terms in values.items() if terms}
        for name in declared:
            if name not in active and name not in inactive:
                inactive[name] = "NO_LEGAL_CONTEXT_EVIDENCE"
        return active, inactive, runtime_seconds, forward_passes

    @staticmethod
    def _ledger_hashes(ledger: EpisodeLedger) -> tuple[str, str]:
        manifest_hash = ledger.manifest_hash
        registry = getattr(ledger, "_split_registry", None)
        split_registry_hash = getattr(registry, "registry_hash", "")
        if not split_registry_hash:
            raise ValueError("EpisodeLedger must expose a split-registry hash for T1-C training")
        return manifest_hash, split_registry_hash

    def _validate_ledger_binding(self, ledger: EpisodeLedger, assignment: EpisodeAssignment) -> None:
        manifest_hash, split_registry_hash = self._ledger_hashes(ledger)
        if self.manifest_hash and manifest_hash != self.manifest_hash:
            raise ValueError("episode ledger manifest does not match the trainer run binding")
        if self.split_registry_hash and split_registry_hash != self.split_registry_hash:
            raise ValueError("episode ledger split registry does not match the trainer run binding")
        if self.scheduled_assignment_hashes and assignment.assignment_hash not in self.scheduled_assignment_hashes:
            raise ValueError("episode assignment is outside the immutable T1-C schedule")
        if self.scheduled_assignment_hashes and (
            self._schedule_cursor >= len(self.scheduled_assignment_hashes)
            or assignment.assignment_hash != self.scheduled_assignment_hashes[self._schedule_cursor]
        ):
            raise ValueError("episode assignment does not match the immutable T1-C schedule cursor")
        if self._last_manifest_hash and manifest_hash != self._last_manifest_hash:
            raise ValueError("episode ledger manifest does not match the checkpoint-resume binding")
        if self._last_split_registry_hash and split_registry_hash != self._last_split_registry_hash:
            raise ValueError("episode ledger split registry does not match the checkpoint-resume binding")

    def _record_ledger_bindings(self, ledger: EpisodeLedger, assignment: EpisodeAssignment) -> None:
        self._last_assignment_hash = assignment.assignment_hash
        self._last_manifest_hash, self._last_split_registry_hash = self._ledger_hashes(ledger)

    @property
    def schedule_cursor(self) -> int:
        """Completed assignment count in the immutable episode schedule."""

        return self._schedule_cursor

    def train_step(
        self,
        *,
        ledger: EpisodeLedger,
        assignment: EpisodeAssignment,
        target_id: str | None = None,
    ) -> TrainingStepOutput:
        self._validate_ledger_binding(ledger, assignment)
        if self._accumulation_cursor == 0:
            self.optimizer.zero_grad(set_to_none=True)
        stage = self.trainer_config.schedule.stage_for_step(self._step_index)
        if stage.auxiliary_only:
            if target_id is not None and target_id not in assignment.target_ids:
                raise PermissionError("the optional warm-up target_id must be a declared target")
            step: LegalEpisodeStep | ContextOnlyEpisodeStep = build_context_only_episode_step(
                ledger=ledger,
                assignment=assignment,
                encoder=self.encoder,
                gaussian_head=self.gaussian_head,
                config=self.episode_config,
            )
            reconstruction = None
            state_version = receipt_record_hash = audit_hash = target_value = None
            legal_count = 0
            supported_fraction = 0.0
            reconstruction_components: Mapping[str, torch.Tensor] = {}
        else:
            if target_id is None:
                raise ValueError("a reconstruction-enabled stage requires exactly one target_id")
            step = build_legal_episode_step(
                ledger=ledger,
                assignment=assignment,
                target_id=target_id,
                encoder=self.encoder,
                gaussian_head=self.gaussian_head,
                config=self.episode_config,
            )
            reconstruction = step.loss
            state_version, receipt_record_hash, audit_hash, target_value = step.state_version, step.receipt_record_hash, step.audit_hash, step.target_id
            legal_count = step.loss.legal_pixel_count
            supported_fraction = step.loss.supported_fraction
            reconstruction_components = step.loss.components
        active, inactive, structural_runtime, structural_passes = self._structural_components(step.context_evidence)
        objective = compose_t1c_objective(
            stage=stage,
            reconstruction=reconstruction,
            structural_components=active,
            inactive_components=inactive,
            config=self.trainer_config.objective,
        )
        if not bool(torch.isfinite(objective.total)):
            raise FloatingPointError("non-finite legal episode objective")
        (objective.total / self.trainer_config.accumulation_steps).backward()
        encoder_norm = gradient_norm(self.encoder.parameters())
        head_norm = gradient_norm(self.gaussian_head.parameters())
        if not stage.auxiliary_only and head_norm <= 0.0:
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
            self._optimizer_step_index += 1
        self._step_index += 1
        self._record_ledger_bindings(ledger, assignment)
        if self.scheduled_assignment_hashes:
            self._schedule_cursor += 1
        structural_report = {name: float(value.detach().cpu()) for name, value in active.items()}
        report = TrainStepReport(
            variant=self.encoder.config.variant,
            stage=stage.stage.value,
            assignment_hash=assignment.assignment_hash,
            state_version=state_version,
            receipt_record_hash=receipt_record_hash,
            audit_hash=audit_hash,
            target_id=target_value,
            reconstruction_intensity_loss=(float(reconstruction_components["intensity"].detach().cpu()) if "intensity" in reconstruction_components else None),
            reconstruction_gradient_loss=(float(reconstruction_components["gradient"].detach().cpu()) if "gradient" in reconstruction_components else None),
            reconstruction_frequency_loss=(float(reconstruction_components["frequency"].detach().cpu()) if "frequency" in reconstruction_components else None),
            legal_target_pixel_count=legal_count,
            supported_fraction=supported_fraction,
            unsupported_fraction=1.0 - supported_fraction,
            structural_components=structural_report,
            inactive_components={
                name: component.reason for name, component in objective.inactive_components.items()
            },
            loss=float(objective.total.detach().cpu()),
            encoder_gradient_norm=encoder_norm,
            head_gradient_norm=head_norm,
            cache_bytes=step.cache_bytes,
            encoder_runtime_seconds=step.encoder_runtime_seconds + structural_runtime,
            encoder_forward_passes=len(step.context_evidence) + structural_passes,
            support_count=step.support_count,
            support_topology_hash=step.support_topology_hash,
            optimizer_updated=optimizer_updated,
            step_index=self._step_index,
            optimizer_step_index=self._optimizer_step_index,
        )
        return TrainingStepOutput(step, objective, report)

    def checkpoint_state(self) -> dict[str, object]:
        """Serialize only optimizer-boundary state; accumulated gradients are not serializable here."""

        if self._accumulation_cursor != 0:
            raise RuntimeError("cannot checkpoint inside an incomplete gradient-accumulation window")
        required_bindings = {
            "matched_experiment_identity": self.matched_experiment_identity,
            "resolved_config_hash": self.resolved_config_hash,
            "sampler_state_hash": self.sampler_state_hash,
            "manifest_hash": self.manifest_hash,
            "split_registry_hash": self.split_registry_hash,
            "last_assignment_hash": self._last_assignment_hash,
            "last_manifest_hash": self._last_manifest_hash,
            "last_split_registry_hash": self._last_split_registry_hash,
            "schedule_cursor": self._schedule_cursor,
            "step_index": self._step_index,
            "optimizer_step_index": self._optimizer_step_index,
            "accumulation_steps": self.trainer_config.accumulation_steps,
        }
        if not self.scheduled_assignment_hashes or any(not value for value in required_bindings.values()):
            raise RuntimeError("checkpoint requires complete immutable run, manifest, split, and schedule bindings")
        if self._last_assignment_hash not in self.scheduled_assignment_hashes:
            raise RuntimeError("checkpoint assignment is absent from the immutable T1-C schedule")
        if (
            self._schedule_cursor <= 0
            or self._schedule_cursor > len(self.scheduled_assignment_hashes)
            or self.scheduled_assignment_hashes[self._schedule_cursor - 1] != self._last_assignment_hash
        ):
            raise RuntimeError("checkpoint schedule cursor does not match the last completed assignment")
        if self._last_manifest_hash != self.manifest_hash or self._last_split_registry_hash != self.split_registry_hash:
            raise RuntimeError("checkpoint ledger bindings differ from the immutable trainer run binding")
        checkpoint_binding = {
            **required_bindings,
            "scheduled_assignment_hashes": self.scheduled_assignment_hashes,
        }
        return {
            "schema": "smagm-episodic-trainer-checkpoint-v3",
            "variant": self.encoder.config.variant,
            "encoder_config_hash": self.encoder.config.config_hash,
            "gaussian_head_config_hash": _hash(self.gaussian_head.config.__dict__),
            "episode_config_hash": self.episode_config.config_hash,
            "trainer_config_hash": _hash(self.trainer_config.to_dict()),
            "matched_experiment_identity": self.matched_experiment_identity,
            "resolved_config_hash": self.resolved_config_hash,
            "sampler_state_hash": self.sampler_state_hash,
            "manifest_hash": self.manifest_hash,
            "split_registry_hash": self.split_registry_hash,
            "scheduled_assignment_hashes": self.scheduled_assignment_hashes,
            "last_assignment_hash": self._last_assignment_hash,
            "last_manifest_hash": self._last_manifest_hash,
            "last_split_registry_hash": self._last_split_registry_hash,
            "schedule_cursor": self._schedule_cursor,
            "step_index": self._step_index,
            "optimizer_step_index": self._optimizer_step_index,
            "accumulation_steps": self.trainer_config.accumulation_steps,
            "encoder": self.encoder.state_dict(),
            "gaussian_head": self.gaussian_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_index": self._step_index,
            "optimizer_step_index": self._optimizer_step_index,
            "accumulation_cursor": self._accumulation_cursor,
            "torch_rng_state": torch.get_rng_state(),
            "checkpoint_binding_hash": _hash(checkpoint_binding),
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        destination = Path(path)
        if not destination.parent.is_dir():
            raise FileNotFoundError("checkpoint parent directory does not exist")
        state = self.checkpoint_state()
        handle, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        os.close(handle)
        try:
            torch.save(state, temporary)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return destination

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore exact boundary state only after all immutable bindings match."""

        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        expected = {
            "schema": "smagm-episodic-trainer-checkpoint-v3",
            "variant": self.encoder.config.variant,
            "encoder_config_hash": self.encoder.config.config_hash,
            "gaussian_head_config_hash": _hash(self.gaussian_head.config.__dict__),
            "episode_config_hash": self.episode_config.config_hash,
            "trainer_config_hash": _hash(self.trainer_config.to_dict()),
            "matched_experiment_identity": self.matched_experiment_identity,
            "resolved_config_hash": self.resolved_config_hash,
            "sampler_state_hash": self.sampler_state_hash,
            "manifest_hash": self.manifest_hash,
            "split_registry_hash": self.split_registry_hash,
            "scheduled_assignment_hashes": self.scheduled_assignment_hashes,
        }
        if not isinstance(payload, dict) or any(payload.get(name) != value for name, value in expected.items()):
            raise ValueError("checkpoint schema or immutable T1-C binding mismatch")
        resume_binding = {
            "matched_experiment_identity": self.matched_experiment_identity,
            "resolved_config_hash": self.resolved_config_hash,
            "sampler_state_hash": self.sampler_state_hash,
            "manifest_hash": self.manifest_hash,
            "split_registry_hash": self.split_registry_hash,
            "last_assignment_hash": payload.get("last_assignment_hash", ""),
            "last_manifest_hash": payload.get("last_manifest_hash", ""),
            "last_split_registry_hash": payload.get("last_split_registry_hash", ""),
            "schedule_cursor": payload.get("schedule_cursor", -1),
            "step_index": payload.get("step_index", -1),
            "optimizer_step_index": payload.get("optimizer_step_index", -1),
            "accumulation_steps": self.trainer_config.accumulation_steps,
            "scheduled_assignment_hashes": self.scheduled_assignment_hashes,
        }
        if any(
            not isinstance(value, str) or len(value) != 64
            for name, value in resume_binding.items()
            if name not in {
                "scheduled_assignment_hashes",
                "schedule_cursor",
                "step_index",
                "optimizer_step_index",
                "accumulation_steps",
            }
        ):
            raise ValueError("checkpoint contains incomplete resume bindings")
        for name in ("schedule_cursor", "step_index", "optimizer_step_index", "accumulation_steps"):
            if not isinstance(resume_binding[name], int):
                raise ValueError("checkpoint contains non-integer optimizer or schedule state")
        if (
            not 0 <= resume_binding["schedule_cursor"] <= len(self.scheduled_assignment_hashes)
            or resume_binding["accumulation_steps"] != self.trainer_config.accumulation_steps
            or resume_binding["step_index"] != resume_binding["schedule_cursor"]
            or resume_binding["step_index"] % resume_binding["accumulation_steps"] != 0
            or resume_binding["optimizer_step_index"]
            != resume_binding["step_index"] // resume_binding["accumulation_steps"]
        ):
            raise ValueError("checkpoint optimizer, step, and schedule cursor state are inconsistent")
        if (
            payload.get("checkpoint_binding_hash") != _hash(resume_binding)
            or payload["last_assignment_hash"] not in self.scheduled_assignment_hashes
            or payload["last_manifest_hash"] != self.manifest_hash
            or payload["last_split_registry_hash"] != self.split_registry_hash
            or resume_binding["schedule_cursor"] <= 0
            or self.scheduled_assignment_hashes[resume_binding["schedule_cursor"] - 1]
            != payload["last_assignment_hash"]
        ):
            raise ValueError("checkpoint resume bindings are corrupt or do not match this run")
        if int(payload.get("accumulation_cursor", -1)) != 0:
            raise ValueError("checkpoint must be saved at an optimizer-step boundary")
        self.encoder.load_state_dict(payload["encoder"])
        self.gaussian_head.load_state_dict(payload["gaussian_head"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self._step_index = int(payload["step_index"])
        self._optimizer_step_index = int(payload["optimizer_step_index"])
        self._accumulation_cursor = 0
        self._last_assignment_hash = str(payload.get("last_assignment_hash", ""))
        self._last_manifest_hash = str(payload.get("last_manifest_hash", ""))
        self._last_split_registry_hash = str(payload.get("last_split_registry_hash", ""))
        self._schedule_cursor = int(payload["schedule_cursor"])
        torch.set_rng_state(payload["torch_rng_state"])
