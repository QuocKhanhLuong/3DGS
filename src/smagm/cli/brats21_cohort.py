"""Process-owned cohort training state for the BraTS21 product controller.

This module deliberately owns learned/global state only.  It never receives a
patient ledger, context payload, target payload, segmentation, anchor bank, or
patient-specific Gaussian state.  Those values stay inside one legal episode
transaction and are discarded by the caller after the optimizer boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
import tempfile
from typing import Any, TypeVar

import torch
from torch import nn

from ..experiments.complexity import parameter_counts, profile_supported_operator_flops


T = TypeVar("T")


def _frozen_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _module_hash(modules: Mapping[str, nn.Module]) -> str:
    digest = hashlib.sha256()
    for module_name, module in modules.items():
        digest.update(module_name.encode("utf-8"))
        for name, value in module.state_dict().items():
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class GlobalCheckpointManager:
    """Atomic target-free checkpoint owner for one cohort process.

    It owns only process-level learned state snapshots.  Patient ledgers,
    target bytes, anchors, Gaussian banks, and prediction receipts are never
    accepted by this API, so they cannot accidentally enter a global resume
    artifact.
    """

    path: Path
    save_count: int = 0
    restore_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("global checkpoint path must be a pathlib.Path")

    @staticmethod
    def _file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def save(
        self,
        cohort: "BraTS21CohortModel",
        *,
        model_binding_hash: str,
        cohort_hash: str,
        split_hash: str,
    ) -> str:
        """Atomically save the one target-free global cohort snapshot."""

        payload = cohort.snapshot(
            model_binding_hash=model_binding_hash,
            cohort_hash=cohort_hash,
            split_hash=split_hash,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=self.path.parent,
            suffix=self.path.suffix + ".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            torch.save(payload, temporary)
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
        self.save_count += 1
        return self._file_hash(self.path)

    def restore(
        self,
        cohort: "BraTS21CohortModel",
        *,
        model_binding_hash: str,
        cohort_hash: str,
        split_hash: str,
    ) -> int:
        """Restore a matching snapshot and return its global-step cursor."""

        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError("global cohort checkpoint must be a mapping")
        cohort.restore(
            payload,
            model_binding_hash=model_binding_hash,
            cohort_hash=cohort_hash,
            split_hash=split_hash,
        )
        self.restore_count += 1
        return cohort.global_step


@dataclass
class BraTS21CohortModel:
    """The sole process-level owner of product learned state.

    ``evidence_projector`` is optional only while retaining the historical
    prefix ablation.  Production configuration supplies it and includes it in
    the optimizer/checkpoint/module accounting.
    """

    encoder: nn.Module
    gaussian_head: nn.Module
    structural_field: nn.Module
    optimizer: Any
    evidence_projector: nn.Module | None = None
    scheduler: Any | None = None
    amp_scaler: Any | None = None
    wandb_logger: Any | None = None
    checkpoint_manager: GlobalCheckpointManager | None = None
    global_step: int = 0
    profiler_invocations: int = 0
    wandb_initializations: int = 0
    encoder_flop_telemetry: dict[str, object] | None = None

    def __post_init__(self) -> None:
        for name, module in self.modules().items():
            if not isinstance(module, nn.Module):
                raise TypeError(f"{name} must be a torch.nn.Module")
        if isinstance(self.global_step, bool) or not isinstance(self.global_step, int) or self.global_step < 0:
            raise ValueError("global_step must be a non-negative integer")
        if isinstance(self.profiler_invocations, bool) or not isinstance(self.profiler_invocations, int) or self.profiler_invocations < 0:
            raise ValueError("profiler_invocations must be a non-negative integer")
        if isinstance(self.wandb_initializations, bool) or not isinstance(self.wandb_initializations, int) or self.wandb_initializations < 0:
            raise ValueError("wandb_initializations must be a non-negative integer")
        if not callable(getattr(self.optimizer, "zero_grad", None)) or not callable(getattr(self.optimizer, "step", None)):
            raise TypeError("cohort optimizer must provide zero_grad and step")
        if self.checkpoint_manager is not None and not isinstance(self.checkpoint_manager, GlobalCheckpointManager):
            raise TypeError("checkpoint_manager must be a GlobalCheckpointManager or None")
        if self.encoder_flop_telemetry is not None and not isinstance(self.encoder_flop_telemetry, dict):
            raise TypeError("encoder_flop_telemetry must be a mapping or None")

    def modules(self) -> dict[str, nn.Module]:
        modules: dict[str, nn.Module] = {
            "encoder": self.encoder,
            "gaussian_head": self.gaussian_head,
            "structural_field": self.structural_field,
        }
        if self.evidence_projector is not None:
            modules["evidence_projector"] = self.evidence_projector
        return modules

    @property
    def module_counts(self) -> dict[str, object]:
        return parameter_counts(self.modules())

    @property
    def state_hash(self) -> str:
        return _module_hash(self.modules())

    def start_wandb(self) -> None:
        if self.wandb_logger is not None and self.wandb_initializations == 0:
            self.wandb_logger.start()
            self.wandb_initializations = 1

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def optimizer_step(self) -> int:
        """Commit one successful shared-model update and advance global step."""

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1
        return self.global_step

    def run_with_optional_profiler(
        self,
        operation: Callable[[], T],
        *,
        enabled: bool,
        scope: str,
    ) -> tuple[T, dict[str, object]]:
        """Profile at most one explicitly diagnostic operation per process."""

        invoke_profiler = bool(enabled and self.profiler_invocations == 0)
        result, telemetry = profile_supported_operator_flops(operation, enabled=invoke_profiler, scope=scope)
        if invoke_profiler:
            self.profiler_invocations += 1
        return result, telemetry

    def snapshot(
        self,
        *,
        model_binding_hash: str,
        cohort_hash: str,
        split_hash: str,
    ) -> dict[str, object]:
        """Return a target-free, patient-state-free resumable global snapshot."""

        if not all(isinstance(value, str) and value for value in (model_binding_hash, cohort_hash, split_hash)):
            raise ValueError("snapshot bindings must be non-empty strings")
        return {
            "schema": "smagm-brats21-global-training-checkpoint-v2",
            "model_binding_hash": model_binding_hash,
            "cohort_hash": cohort_hash,
            "cohort_split_hash": split_hash,
            "global_step": self.global_step,
            "profiler_invocations": self.profiler_invocations,
            "modules": {name: _frozen_state(module) for name, module in self.modules().items()},
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
            "amp_scaler": None if self.amp_scaler is None else self.amp_scaler.state_dict(),
            "target_payload_not_in_checkpoint": True,
            "patient_state_not_in_checkpoint": True,
        }

    def restore(
        self,
        payload: Mapping[str, object],
        *,
        model_binding_hash: str,
        cohort_hash: str,
        split_hash: str,
    ) -> None:
        """Restore only a matching target-free cohort snapshot."""

        if payload.get("schema") != "smagm-brats21-global-training-checkpoint-v2":
            raise ValueError("global cohort checkpoint schema is invalid")
        bindings = {
            "model_binding_hash": model_binding_hash,
            "cohort_hash": cohort_hash,
            "cohort_split_hash": split_hash,
        }
        if any(payload.get(name) != value for name, value in bindings.items()):
            raise ValueError("global cohort checkpoint does not match the resolved model/cohort/split")
        if payload.get("target_payload_not_in_checkpoint") is not True or payload.get("patient_state_not_in_checkpoint") is not True:
            raise ValueError("global cohort checkpoint must explicitly exclude target and patient state")
        modules = payload.get("modules")
        if not isinstance(modules, Mapping) or set(modules) != set(self.modules()):
            raise ValueError("global cohort checkpoint module inventory is invalid")
        try:
            for name, module in self.modules().items():
                state = modules[name]
                if not isinstance(state, Mapping):
                    raise TypeError(name)
                module.load_state_dict(state)
            optimizer_state = payload.get("optimizer")
            if not isinstance(optimizer_state, Mapping):
                raise TypeError("optimizer")
            self.optimizer.load_state_dict(optimizer_state)
            if self.scheduler is not None:
                scheduler_state = payload.get("scheduler")
                if not isinstance(scheduler_state, Mapping):
                    raise TypeError("scheduler")
                self.scheduler.load_state_dict(scheduler_state)
            if self.amp_scaler is not None:
                scaler_state = payload.get("amp_scaler")
                if not isinstance(scaler_state, Mapping):
                    raise TypeError("amp_scaler")
                self.amp_scaler.load_state_dict(scaler_state)
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            raise ValueError("global cohort checkpoint state is invalid") from error
        global_step = payload.get("global_step")
        profiler_invocations = payload.get("profiler_invocations", 0)
        if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
            raise ValueError("global cohort checkpoint global_step is invalid")
        if isinstance(profiler_invocations, bool) or not isinstance(profiler_invocations, int) or profiler_invocations < 0:
            raise ValueError("global cohort checkpoint profiler cursor is invalid")
        self.global_step = global_step
        self.profiler_invocations = profiler_invocations

    def validation(self, operation: Callable[[], T]) -> T:
        """Run a no-grad validation transaction without mutating global weights."""

        before = self.state_hash
        with torch.no_grad():
            result = operation()
        if self.state_hash != before:
            raise RuntimeError("validation mutated global learned state")
        return result

    def save_global_checkpoint(
        self,
        *,
        model_binding_hash: str,
        cohort_hash: str,
        split_hash: str,
    ) -> str:
        """Delegate the live target-free snapshot to the owned manager."""

        if self.checkpoint_manager is None:
            raise RuntimeError("cohort model has no global checkpoint manager")
        return self.checkpoint_manager.save(
            self,
            model_binding_hash=model_binding_hash,
            cohort_hash=cohort_hash,
            split_hash=split_hash,
        )

    def restore_global_checkpoint(
        self,
        *,
        model_binding_hash: str,
        cohort_hash: str,
        split_hash: str,
    ) -> int:
        """Restore the owned global checkpoint exactly once per process."""

        if self.checkpoint_manager is None:
            raise RuntimeError("cohort model has no global checkpoint manager")
        return self.checkpoint_manager.restore(
            self,
            model_binding_hash=model_binding_hash,
            cohort_hash=cohort_hash,
            split_hash=split_hash,
        )


__all__ = ["BraTS21CohortModel", "GlobalCheckpointManager"]
