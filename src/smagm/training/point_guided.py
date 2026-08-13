"""Server-ready training orchestration for the locked point-guided model.

This module is intentionally separate from the legacy sparse-plane trainers.
The model is called target-free first; T1ce is introduced only when the
already-built Gate-E context is scored, and segmentation is introduced after
that for the training-only semantic auxiliary term.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import csv
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Sampler

from ..features.point_guided.baseline_checkpoint import (
    load_training_resume_checkpoint,
    save_clean_inference_checkpoint,
    save_training_resume_checkpoint,
)
from ..features.point_guided.baseline_inference import baseline_checkpoint_metadata
from ..features.point_guided.baseline_inference import GateGInferenceConfig
from ..features.point_guided.baseline_metrics import (
    ReconstructionMetrics,
    SemanticDiceMetrics,
    compute_reconstruction_metrics,
    semantic_dice,
)
from ..features.point_guided.baseline_training import (
    BaselineTrainingConfig,
    build_baseline_optimizer,
)
from ..features.point_guided.config import PointGuidedConfig
from ..features.point_guided.model import PointGuidedMRIModel
from ..features.point_guided.medicalnet_resnet10 import sha256_file
from ..features.point_guided.semantic_supervision import (
    build_coarse_semantic_target,
    compute_semantic_grounding_loss,
)
from ..features.point_guided.training_objective import SupervisionConfig
from ..features.point_guided.trajectory_cost import TrajectoryConfig
from ..data.brats21_point_guided import (
    BraTS21PointGuidedDataset,
    PointGuidedBatch,
    collate_point_guided_samples,
    deterministic_subject_split,
    discover_point_guided_subjects,
)


_STOP_REASONS = ("k_max", "nonpositive_utility", "candidates_exhausted")
_STAT_KEYS = (
    "total_loss",
    "reconstruction_loss",
    "semantic_loss",
    "reward_loss",
    "local_loss",
    "monotonic_loss",
    "update_regularization",
    "MAE",
    "PSNR",
    "SSIM",
    "k_used",
    "path_length_mm",
    "predicted_reward",
    "utility",
    "update_magnitude",
    "dice_normal",
    "dice_edema",
    "dice_core",
    "examples",
)


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation indices without DistributedSampler padding.

    ``DistributedSampler`` pads validation/test indices so every rank has the
    same number of batches.  That is useful for training, but it duplicates
    validation accounting when the cohort size is not divisible by world size.
    This sampler gives each rank a disjoint strided subset instead.
    """

    def __init__(self, dataset: Sequence[Any], *, rank: int, world_size: int) -> None:
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise ValueError("rank/world_size must describe a valid distributed rank")
        self.dataset_size = len(dataset)
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, self.dataset_size, self.world_size))

    def __len__(self) -> int:
        if self.rank >= self.dataset_size:
            return 0
        return (self.dataset_size - 1 - self.rank) // self.world_size + 1


@dataclass(frozen=True)
class PointGuidedTrainingSettings:
    """Runtime settings; architecture and Gate-E constants stay elsewhere."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation: int = 1
    gradient_clip: float | None = 1.0
    decoder_chunk_size: int = 8192
    lambda_semantic: float = 0.2
    semantic_class_weights: tuple[float, float, float] | None = None
    num_workers: int = 0
    seed: int = 20260813
    amp: bool = True
    amp_dtype: str = "fp16"
    early_stopping_patience: int = 10
    log_interval: int = 1
    prediction_interval: int = 1
    normalization_space: str = "masked_robust_01_[0,1]"

    def __post_init__(self) -> None:
        for name in ("learning_rate",):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        weight_decay = float(self.weight_decay)
        if not math.isfinite(weight_decay) or weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        object.__setattr__(self, "weight_decay", weight_decay)
        for name in (
            "epochs",
            "batch_size",
            "gradient_accumulation",
            "decoder_chunk_size",
            "num_workers",
            "early_stopping_patience",
            "log_interval",
            "prediction_interval",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < (0 if name == "num_workers" else 1):
                raise ValueError(f"{name} must be a valid positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        semantic = float(self.lambda_semantic)
        if not math.isfinite(semantic) or semantic < 0.0:
            raise ValueError("lambda_semantic must be finite and non-negative")
        object.__setattr__(self, "lambda_semantic", semantic)
        if self.semantic_class_weights is not None:
            weights = tuple(float(value) for value in self.semantic_class_weights)
            if len(weights) != 3 or any(not math.isfinite(value) or value < 0.0 for value in weights) or not any(value > 0.0 for value in weights):
                raise ValueError("semantic_class_weights must contain three finite non-negative values with one positive value")
            object.__setattr__(self, "semantic_class_weights", weights)  # type: ignore[assignment]
        if self.gradient_clip is not None:
            clip = float(self.gradient_clip)
            if not math.isfinite(clip) or clip <= 0.0:
                raise ValueError("gradient_clip must be positive and finite when supplied")
            object.__setattr__(self, "gradient_clip", clip)
        if self.amp_dtype not in ("fp16", "bf16"):
            raise ValueError("amp_dtype must be fp16 or bf16")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git_head() -> str | None:
    try:
        return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _set_seed(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value % (2**32 - 1))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def initialize_distributed(device_name: str) -> DistributedContext:
    """Detect torchrun and initialize one process per GPU without DataParallel."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    requested = str(device_name)
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if world_size > 1:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device(requested)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        if not torch.distributed.is_available():
            raise RuntimeError("torch.distributed is unavailable in this PyTorch build")
        backend = "nccl" if device.type == "cuda" else "gloo"
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size, device=device)


def destroy_distributed(context: DistributedContext) -> None:
    if context.is_distributed and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def _broadcast_run_dir(context: DistributedContext, run_dir: Path) -> Path:
    if not context.is_distributed:
        return run_dir
    values: list[str | None] = [str(run_dir) if context.is_main else None]
    torch.distributed.broadcast_object_list(values, src=0)
    if not values[0]:
        raise RuntimeError("distributed run directory broadcast returned no path")
    return Path(values[0])


class _TrainingContextModule(nn.Module):
    """Make the custom target-free context call visible to DDP's reducer."""

    def __init__(self, model: PointGuidedMRIModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        observations: Tensor,
        brain_mask: Tensor,
        spacing_mm: tuple[float, float, float],
        voxel_to_ras_mm: tuple[tuple[float, ...], ...],
        chunk_size: int,
    ) -> tuple[object, Tensor]:
        context = self.model.forward_training_context(
            observations,
            brain_mask=brain_mask,
            spacing_mm=spacing_mm,
            voxel_to_ras_mm=voxel_to_ras_mm,
            chunk_size=chunk_size,
        )
        # Return a tensor in addition to the typed context so DDP can discover
        # the autograd graph even though the context is a frozen dataclass.
        return context, context.reconstruction.prediction


def _unwrap_model(module: nn.Module) -> PointGuidedMRIModel:
    if isinstance(module, DistributedDataParallel):
        module = module.module
    if isinstance(module, _TrainingContextModule):
        module = module.model
    if not isinstance(module, PointGuidedMRIModel):
        raise TypeError("expected the current PointGuidedMRIModel")
    return module


def _autocast_context(context: DistributedContext, settings: PointGuidedTrainingSettings):
    enabled = bool(settings.amp and context.device.type == "cuda")
    dtype = torch.float16 if settings.amp_dtype == "fp16" else torch.bfloat16
    return torch.autocast(device_type=context.device.type, dtype=dtype, enabled=enabled)


def _scaler(context: DistributedContext, settings: PointGuidedTrainingSettings) -> torch.cuda.amp.GradScaler | None:
    enabled = bool(settings.amp and settings.amp_dtype == "fp16" and context.device.type == "cuda")
    if not enabled:
        return None
    return torch.cuda.amp.GradScaler(enabled=True)


def _cuda_memory(context: DistributedContext) -> dict[str, int]:
    if context.device.type != "cuda":
        return {"allocated": 0, "reserved": 0, "max_allocated": 0}
    return {
        "allocated": int(torch.cuda.memory_allocated(context.device)),
        "reserved": int(torch.cuda.memory_reserved(context.device)),
        "max_allocated": int(torch.cuda.max_memory_allocated(context.device)),
    }


def _field(batch: object, name: str) -> Any:
    if isinstance(batch, Mapping):
        return batch[name]
    return getattr(batch, name)


def _prepare_batch(
    batch: PointGuidedBatch,
    context: DistributedContext,
) -> tuple[Tensor, Tensor, tuple[float, float, float], tuple[tuple[float, ...], ...]]:
    observations = _field(batch, "observations").to(context.device, non_blocking=True)
    brain_mask = _field(batch, "brain_mask").to(context.device, non_blocking=True)
    # The collator rejects mixed geometry, so the first item is a valid
    # representative for the whole homogeneous batch.  Do not flatten a
    # [B,3]/[B,4,4] tensor into an invalid 3B/4B geometry tuple.
    spacing_tensor = _field(batch, "spacing_xyz_mm")
    affine_tensor = _field(batch, "voxel_to_ras_mm")
    spacing = tuple(float(value) for value in spacing_tensor[0])
    affine = tuple(tuple(float(value) for value in row) for row in affine_tensor[0])
    return observations, brain_mask, spacing, affine


def _route_stats(context: object, points: Tensor) -> dict[str, Any]:
    route = getattr(context, "trajectory")
    batch = points.shape[0]
    if not route.steps:
        zero = torch.zeros(batch, dtype=points.dtype, device=points.device)
        return {
            "k_used": 0.0,
            "path_length_mm": 0.0,
            "predicted_reward": 0.0,
            "utility": 0.0,
            "update_magnitude": 0.0,
            "stop_reasons": Counter(route.stop_reasons),
        }
    indices = route.selected_indices
    active = indices >= 0
    k_used = active.sum(dim=1).to(dtype=points.dtype)
    selected_points = []
    for row in range(batch):
        valid = indices[row][indices[row] >= 0]
        selected_points.append(points[row, valid] if valid.numel() else points.new_empty((0, 3)))
    path = torch.stack(
        tuple(
            item[1:].sub(item[:-1]).norm(dim=-1).sum() if item.shape[0] > 1 else points.new_zeros(())
            for item in selected_points
        )
    )
    rewards = torch.stack(tuple(step.selected_reward for step in route.steps), dim=1)
    utilities = torch.stack(tuple(step.selected_utility for step in route.steps), dim=1)
    updates = torch.stack(tuple(step.selected_update_norm for step in route.steps), dim=1)
    count = active.sum(dim=1).clamp_min(1).to(dtype=points.dtype)
    return {
        "k_used": float(k_used.mean().detach().cpu()),
        "path_length_mm": float(path.mean().detach().cpu()),
        "predicted_reward": float((rewards * active.to(rewards.dtype)).sum(dim=1).div(count).mean().detach().cpu()),
        "utility": float((utilities * active.to(utilities.dtype)).sum(dim=1).div(count).mean().detach().cpu()),
        "update_magnitude": float((updates * active.to(updates.dtype)).sum(dim=1).div(count).mean().detach().cpu()),
        "stop_reasons": Counter(route.stop_reasons),
    }


def _record_stats(
    accumulator: dict[str, float],
    *,
    objective: object,
    total_loss: Tensor,
    semantic_loss: Tensor,
    reconstruction_metrics: ReconstructionMetrics,
    semantic_metrics: SemanticDiceMetrics | None,
    route_stats: Mapping[str, Any],
    batch_size: int,
) -> None:
    count = float(batch_size)
    values = {
        "total_loss": float(total_loss.detach().cpu()),
        "reconstruction_loss": float(objective.reconstruction.total.detach().cpu()),
        "semantic_loss": float(semantic_loss.detach().cpu()),
        "reward_loss": float(objective.reward.detach().cpu()),
        "local_loss": float(objective.local.detach().cpu()),
        "monotonic_loss": float(objective.monotonic.detach().cpu()),
        "update_regularization": float(objective.delta.detach().cpu()),
        "MAE": reconstruction_metrics.mae,
        "PSNR": reconstruction_metrics.psnr,
        "SSIM": reconstruction_metrics.ssim,
        "k_used": float(route_stats["k_used"]),
        "path_length_mm": float(route_stats["path_length_mm"]),
        "predicted_reward": float(route_stats["predicted_reward"]),
        "utility": float(route_stats["utility"]),
        "update_magnitude": float(route_stats["update_magnitude"]),
    }
    for name, value in values.items():
        accumulator[name] = accumulator.get(name, 0.0) + value * count
    accumulator["examples"] = accumulator.get("examples", 0.0) + count
    if semantic_metrics is not None:
        for name, value in (
            ("dice_normal", semantic_metrics.dice_normal),
            ("dice_edema", semantic_metrics.dice_edema),
            ("dice_core", semantic_metrics.dice_core),
        ):
            accumulator[name] = accumulator.get(name, 0.0) + value * count
    reasons = accumulator.setdefault("stop_reasons", Counter())
    reasons.update(route_stats["stop_reasons"])


def _reduce_stats(accumulator: dict[str, Any], context: DistributedContext) -> dict[str, Any]:
    numeric_keys = _STAT_KEYS
    if context.is_distributed:
        values = torch.tensor(
            [float(accumulator.get(key, 0.0)) for key in numeric_keys],
            dtype=torch.float64,
            device=context.device,
        )
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        for key, value in zip(numeric_keys, values.tolist()):
            accumulator[key] = float(value)
        stop_values = torch.tensor(
            [float(accumulator.get("stop_reasons", Counter()).get(reason, 0)) for reason in _STOP_REASONS],
            dtype=torch.float64,
            device=context.device,
        )
        torch.distributed.all_reduce(stop_values, op=torch.distributed.ReduceOp.SUM)
        accumulator["stop_reasons"] = Counter({reason: int(value) for reason, value in zip(_STOP_REASONS, stop_values.tolist())})
    count = max(float(accumulator.get("examples", 0.0)), 1.0)
    result = {
        key: float(accumulator.get(key, 0.0)) / count
        for key in numeric_keys
        if key != "examples"
    }
    result["examples"] = count
    result["stop_reasons"] = dict(accumulator.get("stop_reasons", Counter()))
    return result


def _empty_stats() -> dict[str, Any]:
    return {"stop_reasons": Counter()}


class PointGuidedTrainer:
    """One explicit train/validation loop over the new full-volume adapter."""

    def __init__(
        self,
        model: PointGuidedMRIModel,
        optimizer: torch.optim.Optimizer,
        settings: PointGuidedTrainingSettings,
        supervision: SupervisionConfig,
        context: DistributedContext,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.settings = settings
        self.supervision = supervision
        self.context = context
        self.scaler = _scaler(context, settings)
        self.global_step = 0
        self.train_context_module: nn.Module | None = None
        self.eval_context_module: nn.Module | None = None

    def _forward_objective(self, batch: PointGuidedBatch, *, training: bool) -> tuple[object, Tensor, Tensor, ReconstructionMetrics, SemanticDiceMetrics | None, dict[str, Any]]:
        observations, brain_mask, spacing, affine = _prepare_batch(batch, self.context)
        # This call is intentionally target-free.  The target is the next
        # local variable passed only to compute_training_objective below.
        context = self._context_module(training)(
            observations,
            brain_mask,
            spacing,
            affine,
            self.settings.decoder_chunk_size,
        )[0]
        # T1ce is fetched and introduced only after target-free context
        # construction; it is never part of the wrapped model call.
        raw_target = _field(batch, "target_t1ce")
        if raw_target is None:
            raise ValueError("point-guided training requires a T1ce target after target-free context construction")
        target = raw_target.to(self.context.device, non_blocking=True)
        raw_segmentation = _field(batch, "segmentation")
        segmentation = None if raw_segmentation is None else raw_segmentation.to(self.context.device, non_blocking=True)
        objective = self.model.compute_training_objective(
            context,
            target,
            config=self.supervision,
            valid_mask=brain_mask,
            generator=torch.Generator(device=observations.device).manual_seed(self.settings.seed + self.global_step),
        )
        semantic_loss = target.new_zeros(())
        semantic_metrics = None
        if segmentation is not None:
            # Segmentation enters only after the target-after-inference
            # objective has been constructed and never enters routing.
            semantic_target = build_coarse_semantic_target(
                segmentation,
                brain_mask,
                ignore_index=255,
            )
            semantic_result = compute_semantic_grounding_loss(
                context.frontend.s_coarse.clamp_min(torch.finfo(context.frontend.s_coarse.dtype).tiny).log(),
                semantic_target,
                ignore_index=255,
                class_weights=self.settings.semantic_class_weights,
            )
            semantic_loss = semantic_result.loss
            semantic_metrics = semantic_dice(context.frontend.s_coarse.detach(), semantic_target.detach(), ignore_index=255)
        total = objective.total + self.settings.lambda_semantic * semantic_loss
        if self.context.is_distributed:
            # Every DDP parameter must participate in the reducer graph even
            # when a bounded route has no selected step on one batch.  This is
            # an exact zero-valued keepalive and changes no objective value or
            # gradient for parameters already connected to the loss.
            total = total + sum(
                (parameter.sum() * 0.0)
                for parameter in self.model.parameters()
                if parameter.requires_grad
            )
        metrics = compute_reconstruction_metrics(
            context.reconstruction.prediction.detach(),
            target.detach(),
            brain_mask,
            data_range=self.supervision.ssim_data_range,
            intensity_space=self.settings.normalization_space,
        )
        route_stats = _route_stats(context, context.frontend.refined_points_ras_mm)
        return objective, total, semantic_loss, metrics, semantic_metrics, route_stats

    def _context_module(self, training: bool) -> nn.Module:
        module = self.train_context_module if training else self.eval_context_module
        if module is None:
            raise RuntimeError(
                "PointGuidedTrainer requires separate training and evaluation context modules"
            )
        return module

    def run_epoch(self, loader: DataLoader[PointGuidedBatch], *, training: bool) -> dict[str, Any]:
        self.model.train(training)
        self._context_module(training).train(training)
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        accumulator = _empty_stats()
        batch_count = 0
        for batch_index, batch in enumerate(loader):
            batch_count += 1
            try:
                inference_context = torch.no_grad() if not training else torch.enable_grad()
                with inference_context, _autocast_context(self.context, self.settings):
                    objective, total_loss, semantic_loss, reconstruction_metrics, semantic_metrics, route_stats = self._forward_objective(
                        batch,
                        training=training,
                    )
                    loss = total_loss
                    if training:
                        loss = loss / float(self.settings.gradient_accumulation)
                if training:
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    should_step = (batch_index + 1) % self.settings.gradient_accumulation == 0
                    if should_step:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        if self.settings.gradient_clip is not None:
                            torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], self.settings.gradient_clip)
                        if self.scaler is not None:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.global_step += 1
                _record_stats(
                    accumulator,
                    objective=objective,
                    total_loss=total_loss,
                    semantic_loss=semantic_loss,
                    reconstruction_metrics=reconstruction_metrics,
                    semantic_metrics=semantic_metrics,
                    route_stats=route_stats,
                    batch_size=int(_field(batch, "observations").shape[0]),
                )
            except RuntimeError as error:
                if "out of memory" in str(error).lower() and self.context.device.type == "cuda":
                    raise RuntimeError(
                        "CUDA OOM in point-guided training. Reduce decoder_chunk_size, "
                        "counterfactual_candidates, K_max, gradient accumulation, or batch size; "
                        "scientific settings were not changed automatically. DDP does not pool GPU memory."
                    ) from error
                raise
        if training and batch_count % self.settings.gradient_accumulation != 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            if self.settings.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], self.settings.gradient_clip)
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
        return _reduce_stats(accumulator, self.context)

    def bind_context_modules(self, train_module: nn.Module, eval_module: nn.Module | None = None) -> None:
        """Bind separate training and validation context modules.

        Training may use DDP, while validation must use the raw local module
        because ``DistributedEvalSampler`` intentionally produces uneven,
        non-padded shards. The one-module form remains available through
        :meth:`bind_context_module` for CPU synthetic callers.
        """

        if not isinstance(train_module, nn.Module):
            raise TypeError("train_module must be a torch.nn.Module")
        if eval_module is not None and not isinstance(eval_module, nn.Module):
            raise TypeError("eval_module must be a torch.nn.Module when supplied")
        self.train_context_module = train_module
        self.eval_context_module = train_module if eval_module is None else eval_module

    def bind_context_module(self, module: nn.Module) -> None:
        """Compatibility alias for one-module CPU/synthetic callers."""

        self.bind_context_modules(module, module)

    def context_module_for(self, *, training: bool) -> nn.Module:
        """Return the context module selected for an epoch."""

        return self._context_module(training)


def _settings_from_mapping(value: Mapping[str, Any]) -> PointGuidedTrainingSettings:
    allowed = {field for field in PointGuidedTrainingSettings.__dataclass_fields__}
    values = {key: item for key, item in value.items() if key in allowed}
    if isinstance(values.get("semantic_class_weights"), list):
        values["semantic_class_weights"] = tuple(values["semantic_class_weights"])
    return PointGuidedTrainingSettings(**values)


def normalization_policy_from_config(value: object | None) -> str:
    """Return the explicit full-volume normalization policy name."""

    if value is None:
        return "masked_zscore"
    if isinstance(value, Mapping):
        policy = value.get("normalization_policy", "masked_zscore")
    else:
        policy = getattr(value, "normalization_policy", "masked_zscore")
    policy = str(policy)
    if policy not in ("masked_zscore", "masked_robust_01"):
        raise ValueError(
            "normalization_policy must be 'masked_zscore' or 'masked_robust_01'"
        )
    return policy


def normalization_space_from_config(value: object | None) -> str:
    """Return the declared metric/intensity-space label for run metadata."""

    policy = normalization_policy_from_config(value)
    if policy == "masked_robust_01":
        return "masked_robust_01_[0,1]"
    return "masked_zscore_explicit_metric_range"


def validate_metric_data_range(
    normalization_config: object | None,
    supervision: SupervisionConfig,
) -> None:
    """Reject silent PSNR/SSIM range assumptions for masked z-score data."""

    policy = normalization_policy_from_config(normalization_config)
    explicit = None
    if isinstance(normalization_config, Mapping):
        explicit = normalization_config.get("metric_data_range")
    elif normalization_config is not None:
        explicit = getattr(normalization_config, "metric_data_range", None)
    if policy == "masked_zscore":
        if explicit is None:
            raise ValueError(
                "masked_zscore requires data.normalization.metric_data_range "
                "for PSNR/SSIM; it cannot silently use 1.0"
            )
        explicit_value = float(explicit)
        if not math.isfinite(explicit_value) or explicit_value <= 0.0:
            raise ValueError("metric_data_range must be positive and finite")
        if not math.isclose(explicit_value, supervision.ssim_data_range, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(
                "data.normalization.metric_data_range must equal "
                "supervision.ssim_data_range"
            )
    elif explicit is not None:
        explicit_value = float(explicit)
        if not math.isfinite(explicit_value) or explicit_value <= 0.0:
            raise ValueError("metric_data_range must be positive and finite")
        if not math.isclose(explicit_value, supervision.ssim_data_range, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(
                "data.normalization.metric_data_range must equal "
                "supervision.ssim_data_range"
            )


def build_model_from_config(raw: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> tuple[PointGuidedMRIModel, SupervisionConfig, PointGuidedTrainingSettings]:
    overrides = dict(overrides or {})
    model_values = dict(raw.get("model", {}))
    trajectory_values = dict(raw.get("trajectory", {}))
    supervision_values = dict(raw.get("supervision", {}))
    training_values = dict(raw.get("training", {}))
    for key in ("medicalnet_checkpoint_path", "medicalnet_checkpoint_sha256", "require_pretrained_backbone"):
        if key in overrides and overrides[key] is not None:
            model_values[key] = overrides[key]
    for key in ("k_max", "lambda_travel", "lambda_overlap", "lambda_step"):
        if key in overrides and overrides[key] is not None:
            trajectory_values[key] = overrides[key]
    for key in ("counterfactual_candidates",):
        if key in overrides and overrides[key] is not None:
            supervision_values[key] = overrides[key]
    for key in (
        "decoder_chunk_size",
        "lambda_semantic",
        "amp",
        "amp_dtype",
        "batch_size",
        "gradient_accumulation",
        "gradient_clip",
        "learning_rate",
        "weight_decay",
        "semantic_class_weights",
    ):
        if key in overrides and overrides[key] is not None:
            training_values[key] = overrides[key]
    config = PointGuidedConfig(**model_values)
    trajectory = TrajectoryConfig(**trajectory_values)
    model = PointGuidedMRIModel(config, trajectory_config=trajectory)
    return model, SupervisionConfig(**supervision_values), _settings_from_mapping(training_values)


def _make_loader(
    dataset: BraTS21PointGuidedDataset,
    *,
    context: DistributedContext,
    settings: PointGuidedTrainingSettings,
    training: bool,
) -> tuple[DataLoader[PointGuidedBatch], Sampler[Any] | None]:
    sampler: Sampler[Any] | None
    if context.is_distributed and training:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=settings.seed,
        )
        shuffle = False
    elif context.is_distributed:
        sampler = DistributedEvalSampler(dataset, rank=context.rank, world_size=context.world_size)
        shuffle = False
    else:
        sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(settings.seed)) if training else None
        shuffle = sampler is None and training
    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=settings.num_workers,
        pin_memory=context.device.type == "cuda",
        collate_fn=collate_point_guided_samples,
        persistent_workers=settings.num_workers > 0,
    )
    return loader, sampler


def _environment(context: DistributedContext, model: PointGuidedMRIModel, settings: PointGuidedTrainingSettings) -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(context.device),
        "rank": context.rank,
        "world_size": context.world_size,
        "git_head": _git_head(),
        "amp": settings.amp,
        "amp_dtype": settings.amp_dtype,
        "pretrained_backbone_verified": bool(model.semantic_prior.pretrained_loaded),
        "medicalnet_checkpoint_loaded": bool(model.semantic_prior.checkpoint_loaded),
        "normalization_space": settings.normalization_space,
    }


def _resolve_split(
    subject_ids: Sequence[str],
    *,
    seed: int,
    split_fractions: Sequence[float] = (0.8, 0.1, 0.1),
    split_file: str | Path | None,
    max_train_subjects: int | None,
    max_val_subjects: int | None,
    max_test_subjects: int | None,
) -> tuple[dict[str, Any], str]:
    """Use an existing split exactly, otherwise create the declared split."""

    if split_file is not None:
        path = Path(split_file)
        if not path.is_file():
            raise FileNotFoundError(f"split file does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("split file must contain a JSON object")
        groups = {
            "train": tuple(payload.get("train_subject_ids", payload.get("train", ()))),
            "val": tuple(payload.get("val_subject_ids", payload.get("val", payload.get("validation", ())))),
            "test": tuple(payload.get("test_subject_ids", payload.get("test", ()))),
        }
        all_ids = tuple(sorted(str(item) for item in subject_ids))
        selected = tuple(item for group in groups.values() for item in group)
        if any(item not in all_ids for item in selected) or len(selected) != len(set(selected)):
            raise ValueError("split file contains unknown or overlapping subject IDs")
        excluded = tuple(payload.get("excluded_subject_ids", ()))
        if any(item not in all_ids for item in excluded) or len(excluded) != len(set(excluded)):
            raise ValueError("split file contains unknown or overlapping excluded subject IDs")
        if set(selected).intersection(excluded) or set(selected).union(excluded) != set(all_ids):
            raise ValueError("split file must partition every discovered subject exactly once")
        split_hash = payload.get("split_hash")
        if not isinstance(split_hash, str) or len(split_hash) != 64:
            raise ValueError("split file must contain a 64-character split_hash")
        normalized = dict(payload)
        normalized.update(
            {
                "train": groups["train"],
                "val": groups["val"],
                "test": groups["test"],
                "excluded_subject_ids": excluded,
                "split_hash": split_hash,
            }
        )
        return normalized, split_hash
    split_object = deterministic_subject_split(
        subject_ids,
        seed=seed,
        split_fractions=split_fractions,
        max_train_subjects=max_train_subjects,
        max_val_subjects=max_val_subjects,
        max_test_subjects=max_test_subjects,
    )
    payload = split_object.to_dict()
    payload.update({
        "train": tuple(split_object.train_subject_ids),
        "val": tuple(split_object.val_subject_ids),
        "test": tuple(split_object.test_subject_ids),
    })
    return payload, split_object.split_hash


def _write_epoch_logs(run_dir: Path, record: Mapping[str, Any], *, header_written: bool) -> None:
    with (run_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    path = run_dir / "metrics.csv"
    fields = tuple(record.keys())
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not header_written:
            writer.writeheader()
        writer.writerow({key: record[key] for key in fields})


def _save_overfit_predictions(
    *,
    model: PointGuidedMRIModel,
    dataset: BraTS21PointGuidedDataset,
    run_dir: Path,
    epoch: int,
    settings: PointGuidedTrainingSettings,
) -> tuple[str, ...]:
    """Save target-free debug predictions for the declared overfit cohort."""

    trajectory = model.trajectory
    if trajectory is None:
        raise RuntimeError("overfit prediction snapshots require the explicit trajectory")
    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise RuntimeError("overfit prediction snapshots require a model with parameters") from error
    inference_config = GateGInferenceConfig(
        lambda_travel=trajectory.config.lambda_travel,
        lambda_overlap=trajectory.config.lambda_overlap,
        lambda_step=trajectory.config.lambda_step,
        k_max=trajectory.config.k_max,
        decoder_chunk_size=settings.decoder_chunk_size,
    )
    directory = run_dir / "predictions" / f"epoch-{epoch:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    subject_ids: list[str] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        with torch.no_grad():
            result = model.forward_baseline_inference(
                sample.observations.unsqueeze(0).to(device, non_blocking=True),
                brain_mask=sample.brain_mask.unsqueeze(0).to(device, non_blocking=True),
                spacing_mm=sample.spacing_xyz_mm,
                voxel_to_ras_mm=sample.voxel_to_ras_mm,
                inference_config=inference_config,
            )
        torch.save(
            {
                "epoch": epoch,
                "subject_id": sample.subject_id,
                "prediction": result.prediction[0, 0].detach().cpu(),
                "normalization_space": settings.normalization_space,
                "target_free": True,
            },
            directory / f"{sample.subject_id}_t1ce_pred.pt",
        )
        subject_ids.append(sample.subject_id)
    return tuple(subject_ids)


def run_training(
    *,
    raw_config: Mapping[str, Any],
    data_root: str | Path,
    output_root: str | Path,
    run_name: str | None = None,
    split_file: str | Path | None = None,
    resume: str | Path | None = None,
    overfit: bool = False,
    max_train_subjects: int | None = None,
    max_val_subjects: int | None = None,
    max_test_subjects: int | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    overrides = dict(overrides or {})
    device_name = str(overrides.pop("device", raw_config.get("training", {}).get("device", "cpu")))
    context = initialize_distributed(device_name)
    try:
        _set_seed(int(raw_config.get("training", {}).get("seed", 20260813)), context.rank)
        model, supervision, settings = build_model_from_config(raw_config, overrides)
        data_config = raw_config.get("data", {})
        normalization_config = data_config.get("normalization")
        validate_metric_data_range(normalization_config, supervision)
        settings = replace(
            settings,
            normalization_space=normalization_space_from_config(normalization_config),
        )
        model.to(context.device)
        subjects = discover_point_guided_subjects(data_root)
        split, split_hash = _resolve_split(
            tuple(subject.subject_id for subject in subjects),
            seed=int(raw_config.get("data", {}).get("split_seed", 20260813)),
            split_fractions=tuple(raw_config.get("data", {}).get("split_fractions", (0.8, 0.1, 0.1))),
            split_file=split_file,
            max_train_subjects=max_train_subjects,
            max_val_subjects=max_val_subjects,
            max_test_subjects=max_test_subjects,
        )
        root = Path(output_root)
        if resume is not None:
            run_dir = Path(resume).resolve().parent.parent
        else:
            name = run_name or datetime.now(timezone.utc).strftime("point-guided-%Y%m%dT%H%M%SZ")
            run_dir = root / name
        run_dir = _broadcast_run_dir(context, run_dir)
        if context.is_main:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            resolved_config = json.loads(_canonical_json(raw_config))
            resolved_config.setdefault("training", {})["device"] = device_name
            for key in ("medicalnet_checkpoint_path", "medicalnet_checkpoint_sha256", "require_pretrained_backbone"):
                if key in overrides:
                    resolved_config.setdefault("model", {})[key] = overrides[key]
            for key in ("k_max", "lambda_travel", "lambda_overlap", "lambda_step"):
                if key in overrides:
                    resolved_config.setdefault("trajectory", {})[key] = overrides[key]
            for key in ("counterfactual_candidates",):
                if key in overrides:
                    resolved_config.setdefault("supervision", {})[key] = overrides[key]
            for key in (
                "decoder_chunk_size",
                "lambda_semantic",
                "amp",
                "amp_dtype",
                "batch_size",
                "gradient_accumulation",
                "gradient_clip",
                "learning_rate",
                "weight_decay",
                "semantic_class_weights",
            ):
                if key in overrides:
                    resolved_config.setdefault("training", {})[key] = overrides[key]
            _atomic_json(
                run_dir / "config.json",
                {
                    "resolved": resolved_config,
                    "training_settings": asdict(settings),
                    "supervision": asdict(supervision),
                    "overfit": overfit,
                },
            )
            _atomic_json(run_dir / "split.json", split)
            _atomic_json(run_dir / "environment.json", _environment(context, model, settings))
        if context.is_distributed:
            torch.distributed.barrier()

        baseline_config = BaselineTrainingConfig(
            learning_rate=settings.learning_rate,
            weight_decay=settings.weight_decay,
            batch_size=settings.batch_size,
            epochs=settings.epochs,
            max_steps=max(1, settings.epochs * max(1, len(split["train"]))),
            decoder_chunk_size=settings.decoder_chunk_size,
            device=str(context.device),
            seed=settings.seed,
            gradient_clip_norm=settings.gradient_clip,
            checkpoint_directory=str(run_dir / "checkpoints"),
        )
        optimizer, ownership = build_baseline_optimizer(model, baseline_config)
        raw_context_module = _TrainingContextModule(model)
        train_context_module: nn.Module = raw_context_module
        if context.is_distributed:
            train_context_module = DistributedDataParallel(
                raw_context_module,
                device_ids=[context.local_rank] if context.device.type == "cuda" else None,
                find_unused_parameters=False,
            )
        trainer = PointGuidedTrainer(model, optimizer, settings, supervision, context)
        trainer.bind_context_modules(train_context_module, raw_context_module)

        start_epoch = 1
        best_metric = math.inf
        if resume is not None:
            state = load_training_resume_checkpoint(
                resume,
                model=model,
                optimizer=optimizer,
                scaler=trainer.scaler,
                expected_split_hash=split_hash,
            )
            start_epoch = int(state["epoch"]) + 1
            trainer.global_step = int(state["global_step"])
            best_metric = float(state["best_validation_reconstruction_loss"])
            if context.is_distributed:
                torch.distributed.barrier()

        train_ids = tuple(split["train"])
        val_ids = train_ids if overfit else tuple(split["val"])
        require_segmentation = bool(data_config.get("require_segmentation", True))
        train_dataset = BraTS21PointGuidedDataset(
            data_root,
            subject_ids=train_ids,
            normalization_config=normalization_config,
            require_segmentation=require_segmentation,
        )
        val_dataset = BraTS21PointGuidedDataset(
            data_root,
            subject_ids=val_ids,
            normalization_config=normalization_config,
            require_segmentation=require_segmentation,
        )
        train_loader, train_sampler = _make_loader(train_dataset, context=context, settings=settings, training=True)
        val_loader, val_sampler = _make_loader(val_dataset, context=context, settings=settings, training=False)
        header_written = (run_dir / "metrics.csv").is_file()
        patience_count = 0
        first_train_reconstruction_loss: float | None = None
        overfit_prediction_epochs: list[int] = []
        summary: dict[str, Any] = {}
        wandb_run = None
        wandb_config = raw_config.get("_wandb", {})
        if context.is_main and isinstance(wandb_config, Mapping) and bool(wandb_config.get("enabled", False)):
            try:
                import wandb  # type: ignore[import-not-found]
            except ImportError as error:
                raise RuntimeError("--wandb was requested but the optional wandb dependency is not installed") from error
            wandb_run = wandb.init(
                project=str(wandb_config.get("project", "point-guided-brats21")),
                name=wandb_config.get("run_name"),
                config={"training": asdict(settings), "supervision": asdict(supervision), "git_head": _git_head()},
            )
        for epoch in range(start_epoch, settings.epochs + 1):
            if context.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(context.device)
            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)
            if isinstance(val_sampler, DistributedSampler):
                val_sampler.set_epoch(epoch)
            started = time.perf_counter()
            train_stats = trainer.run_epoch(train_loader, training=True)
            val_stats = trainer.run_epoch(val_loader, training=False)
            epoch_seconds = time.perf_counter() - started
            row: dict[str, Any] = {
                "epoch": epoch,
                "global_step": trainer.global_step,
                "train/total_loss": train_stats.get("total_loss", math.nan),
                "train/reconstruction_loss": train_stats.get("reconstruction_loss", math.nan),
                "train/semantic_loss": train_stats.get("semantic_loss", math.nan),
                "train/reward_loss": train_stats.get("reward_loss", math.nan),
                "train/local_loss": train_stats.get("local_loss", math.nan),
                "train/monotonic_loss": train_stats.get("monotonic_loss", math.nan),
                "train/update_regularization": train_stats.get("update_regularization", math.nan),
                "val/total_loss": val_stats.get("total_loss", math.nan),
                "val/reconstruction_loss": val_stats.get("reconstruction_loss", math.nan),
                "val/MAE": val_stats.get("MAE", math.nan),
                "val/PSNR": val_stats.get("PSNR", math.nan),
                "val/SSIM": val_stats.get("SSIM", math.nan),
                "val/dice_normal": val_stats.get("dice_normal", math.nan),
                "val/dice_edema": val_stats.get("dice_edema", math.nan),
                "val/dice_core": val_stats.get("dice_core", math.nan),
                "trajectory/mean_K_used": val_stats.get("k_used", math.nan),
                "trajectory/mean_path_length_mm": val_stats.get("path_length_mm", math.nan),
                "trajectory/mean_predicted_reward": val_stats.get("predicted_reward", math.nan),
                "trajectory/mean_utility": val_stats.get("utility", math.nan),
                "trajectory/mean_update_magnitude": val_stats.get("update_magnitude", math.nan),
                "trajectory/stop_reason_histogram": json.dumps(val_stats.get("stop_reasons", {}), sort_keys=True),
                "system/epoch_seconds": epoch_seconds,
                "system/gpu_allocated": _cuda_memory(context)["allocated"],
                "system/gpu_reserved": _cuda_memory(context)["reserved"],
                "system/gpu_peak_allocated": _cuda_memory(context)["max_allocated"],
                "validation_is_held_out": not overfit,
            }
            if first_train_reconstruction_loss is None:
                first_train_reconstruction_loss = float(train_stats.get("reconstruction_loss", math.nan))
            stop_training = False
            if context.is_main:
                if overfit and epoch % settings.prediction_interval == 0:
                    _save_overfit_predictions(
                        model=model,
                        dataset=train_dataset,
                        run_dir=run_dir,
                        epoch=epoch,
                        settings=settings,
                    )
                    overfit_prediction_epochs.append(epoch)
                _write_epoch_logs(run_dir, row, header_written=header_written)
                header_written = True
                if wandb_run is not None:
                    wandb_run.log({key: value for key, value in row.items() if isinstance(value, (int, float, bool))}, step=trainer.global_step)
                metric = float(val_stats.get("reconstruction_loss", math.inf))
                if metric < best_metric:
                    best_metric = metric
                    patience_count = 0
                    save_clean_inference_checkpoint(run_dir / "checkpoints" / "best_model.pt", model)
                else:
                    patience_count += 1
                save_training_resume_checkpoint(
                    run_dir / "checkpoints" / "last_train.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=trainer.scaler,
                    epoch=epoch,
                    global_step=trainer.global_step,
                    best_validation_reconstruction_loss=best_metric,
                    training_config={"settings": asdict(settings), "supervision": asdict(supervision)},
                    split_hash=split_hash,
                    metadata={"git_head": _git_head(), "overfit": overfit, "validation_is_held_out": not overfit},
                )
                summary = {
                    "status": "completed_software_run",
                    "overfit": overfit,
                    "validation_is_held_out": not overfit,
                    "best_validation_reconstruction_loss": best_metric,
                    "last_epoch": epoch,
                    "global_step": trainer.global_step,
                    "pretrained_backbone_verified": bool(model.semantic_prior.pretrained_loaded),
                    "ownership": [asdict(item) for item in ownership],
                    "last_metrics": row,
                    "overfit_prediction_epochs": tuple(overfit_prediction_epochs),
                    "overfit_train_reconstruction_loss_decreased": (
                        overfit
                        and first_train_reconstruction_loss is not None
                        and float(train_stats.get("reconstruction_loss", math.inf)) < first_train_reconstruction_loss
                    ),
                    "split_hash": split_hash,
                    "git_head": _git_head(),
                }
                _atomic_json(run_dir / "summary.json", summary)
                stop_training = patience_count >= settings.early_stopping_patience
            if context.is_distributed:
                stop_flag = torch.tensor(
                    [1 if stop_training else 0],
                    dtype=torch.int64,
                    device=context.device,
                )
                torch.distributed.broadcast(stop_flag, src=0)
                torch.distributed.barrier()
                stop_training = bool(int(stop_flag.item()))
            if stop_training:
                break
        if wandb_run is not None:
            wandb_run.finish()
        if context.is_main:
            return summary
        return None
    finally:
        destroy_distributed(context)


def preflight(
    *,
    data_root: str | Path,
    checkpoint: str | Path | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Read-only server preflight; never downloads data or weights."""

    subjects = discover_point_guided_subjects(data_root)
    result: dict[str, Any] = {
        "dataset_root": str(Path(data_root).resolve()),
        "subject_count": len(subjects),
        "subject_ids_sample": [subject.subject_id for subject in subjects[:5]],
        "git_head": _git_head(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "medicalnet_checkpoint": None if checkpoint is None else str(Path(checkpoint).resolve()),
    }
    if checkpoint is not None and not Path(checkpoint).is_file():
        raise FileNotFoundError(f"MedicalNet checkpoint does not exist: {checkpoint}")
    if expected_sha256 is not None:
        actual_sha256 = sha256_file(checkpoint) if checkpoint is not None else None
        expected = str(expected_sha256).lower()
        if actual_sha256 != expected:
            raise ValueError(
                "MedicalNet checkpoint SHA-256 mismatch in preflight: "
                f"expected={expected}, actual={actual_sha256}"
            )
        result["medicalnet_sha256"] = actual_sha256
        result["medicalnet_integrity_verified"] = True
    else:
        result["medicalnet_sha256"] = None
        result["medicalnet_integrity_verified"] = False
    return result


__all__ = [
    "DistributedContext",
    "DistributedEvalSampler",
    "PointGuidedTrainer",
    "PointGuidedTrainingSettings",
    "build_model_from_config",
    "destroy_distributed",
    "initialize_distributed",
    "normalization_policy_from_config",
    "normalization_space_from_config",
    "preflight",
    "run_training",
    "validate_metric_data_range",
]
