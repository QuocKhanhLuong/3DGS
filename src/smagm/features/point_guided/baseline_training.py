"""Gate-F F1/F2 baseline ownership, policy, and synthetic smoke helpers.

This module is deliberately small: it classifies the already-implemented
model, constructs one conventional engineering-baseline optimizer, and runs a
single target-after-inference smoke step.  It contains no real-data adapter,
held-out evaluation, scheduler, checkpoint policy, or Gate-G inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor

from .availability import ExactNoRevisitPolicy
from .model import PointGuidedMRIModel
from .training_objective import SupervisionConfig, TrainingObjectiveResult


def _positive_finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


@dataclass(frozen=True)
class BaselineTrainingConfig:
    """Explicit engineering-only settings for the first Gate-F baseline.

    Adam is selected because it is the repository's existing simple optimizer
    convention.  These values are reproducible starting values, not locked
    scientific constants or validation results.
    """

    optimizer_name: Literal["adam"] = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    batch_size: int = 1
    epochs: int = 1
    max_steps: int = 1
    decoder_chunk_size: int = 31
    device: str = "cpu"
    seed: int = 20260812
    gradient_clip_norm: float | None = None
    logging_interval: int = 1
    checkpoint_directory: str = "artifacts/point_guided_baseline"

    def __post_init__(self) -> None:
        if self.optimizer_name != "adam":
            raise ValueError("Gate-F baseline optimizer_name is the explicit engineering choice 'adam'")
        object.__setattr__(self, "learning_rate", _positive_finite("learning_rate", self.learning_rate))
        weight_decay = float(self.weight_decay)
        if not math.isfinite(weight_decay) or weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        object.__setattr__(self, "weight_decay", weight_decay)
        if len(self.betas) != 2 or any(not math.isfinite(float(value)) or not 0.0 <= float(value) < 1.0 for value in self.betas):
            raise ValueError("betas must contain two finite values in [0,1)")
        object.__setattr__(self, "betas", (float(self.betas[0]), float(self.betas[1])))
        for name in ("batch_size", "epochs", "max_steps", "decoder_chunk_size", "logging_interval"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string")
        if self.gradient_clip_norm is not None:
            object.__setattr__(self, "gradient_clip_norm", _positive_finite("gradient_clip_norm", self.gradient_clip_norm))
        if not isinstance(self.checkpoint_directory, str) or not self.checkpoint_directory:
            raise ValueError("checkpoint_directory must be a non-empty string")


@dataclass(frozen=True)
class ParameterOwnership:
    """One exact Gate-F model-parameter ownership row."""

    module: str
    parameter_count: int
    requires_grad: bool
    optimizer_member: bool
    authority: str


@dataclass(frozen=True)
class BaselineSmokeResult:
    """Compact F2 evidence from one synthetic forward/backward/update step."""

    prediction_shape: tuple[int, int, int, int, int]
    total_loss: float
    max_displacement_mm: float
    gradient_modules: tuple[str, ...]
    changed_modules: tuple[str, ...]
    selected_indices: Tensor
    ownership: tuple[ParameterOwnership, ...]


# Backward-compatible Gate-F spelling; Gate-G imports the neutral helper.
BaselineNoRevisitPolicy = ExactNoRevisitPolicy


def _components(model: PointGuidedMRIModel) -> tuple[tuple[str, torch.nn.Module, str], ...]:
    if model.trajectory is None or model.decoder is None:
        raise ValueError("Gate-F baseline requires PointGuidedMRIModel with an explicit TrajectoryConfig")
    return (
        (
            "semantic_head",
            model.semantic_prior.semantic_head,
            "Gate-F MAIN: no semantic-head checkpoint or explicit lock is present",
        ),
        (
            "point_refiner.offset_predictor",
            model.point_refiner.offset_predictor,
            "human Gate-F resolution: trainable; 1,419 parameters; bounds unchanged",
        ),
        ("base_plane_projector", model.base_plane_projector, "Gate-F MAIN axis-conditioned B projector"),
        (
            "spectral_anchor_builder.band_projector",
            model.spectral_anchor_builder.band_projector,
            "Gate-F MAIN shared spectral 1x1 projector; SWT filters remain fixed",
        ),
        ("trajectory.state_initializer", model.trajectory.state_initializer, "Gate-F MAIN Z0 initializer"),
        ("trajectory.reward_net", model.trajectory.reward_net, "Gate-F MAIN RewardNet"),
        ("trajectory.update_net", model.trajectory.update_net, "Gate-F MAIN UpdateNet"),
        ("decoder", model.decoder, "Gate-F MAIN implicit decoder"),
    )


def _parameter_ids(optimizer: torch.optim.Optimizer | None) -> set[int]:
    if optimizer is None:
        return set()
    result: list[int] = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    if len(result) != len(set(result)):
        raise RuntimeError("baseline optimizer contains a duplicate parameter")
    return set(result)


def resolve_parameter_ownership(
    model: PointGuidedMRIModel,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[ParameterOwnership, ...]:
    """Classify every model parameter against the explicit Gate-F decision."""

    if not isinstance(model, PointGuidedMRIModel):
        raise TypeError("model must be a PointGuidedMRIModel")
    optimizer_ids = _parameter_ids(optimizer)
    component_ids: set[int] = set()
    records: list[ParameterOwnership] = []
    for name, module, authority in _components(model):
        parameters = tuple(module.parameters())
        if not parameters:
            raise RuntimeError(f"Gate-F component {name} unexpectedly has no parameters")
        requires = {parameter.requires_grad for parameter in parameters}
        if len(requires) != 1:
            raise RuntimeError(f"Gate-F component {name} mixes frozen and trainable parameters")
        trainable = requires.pop()
        if not trainable:
            raise RuntimeError(f"Gate-F component {name} is unexpectedly frozen")
        parameter_ids = {id(parameter) for parameter in parameters}
        if component_ids.intersection(parameter_ids):
            raise RuntimeError(f"Gate-F parameter ownership overlaps at {name}")
        component_ids.update(parameter_ids)
        members = parameter_ids.intersection(optimizer_ids)
        if optimizer is not None and members != parameter_ids:
            raise RuntimeError(f"Gate-F optimizer membership must include every parameter of {name} exactly once")
        records.append(
            ParameterOwnership(
                module=name,
                parameter_count=sum(parameter.numel() for parameter in parameters),
                requires_grad=trainable,
                optimizer_member=bool(optimizer is not None),
                authority=authority,
            )
        )

    backbone_parameters = tuple(model.semantic_prior.backbone.parameters())
    if any(parameter.requires_grad for parameter in backbone_parameters):
        raise RuntimeError("MedicalNet backbone must remain frozen in the Gate-F MAIN baseline")
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    if optimizer_ids.intersection(backbone_ids):
        raise RuntimeError("frozen MedicalNet backbone must not appear in the baseline optimizer")
    records.append(
        ParameterOwnership(
            module="semantic_prior.backbone",
            parameter_count=sum(parameter.numel() for parameter in backbone_parameters),
            requires_grad=False,
            optimizer_member=False,
            authority="Gate-F MAIN frozen MedicalNet backbone",
        )
    )

    all_parameters = {id(parameter) for parameter in model.parameters()}
    if all_parameters != component_ids.union(backbone_ids):
        unknown = len(all_parameters.difference(component_ids.union(backbone_ids)))
        raise RuntimeError(f"GATE-F TRAINABLE-SET AMBIGUITY: {unknown} model parameter(s) are unclassified")
    if optimizer is not None and optimizer_ids != component_ids:
        unknown = len(optimizer_ids.difference(component_ids))
        raise RuntimeError(f"baseline optimizer contains {unknown} unexpected parameter(s)")
    if any(parameter.requires_grad for parameter in model.spectral_anchor_builder.swt.parameters()):
        raise RuntimeError("SWT-Haar filters must remain fixed buffers, not trainable parameters")
    return tuple(records)


def build_baseline_optimizer(
    model: PointGuidedMRIModel,
    config: BaselineTrainingConfig,
) -> tuple[torch.optim.Optimizer, tuple[ParameterOwnership, ...]]:
    """Build the exact one-set Gate-F baseline Adam optimizer."""

    if not isinstance(config, BaselineTrainingConfig):
        raise TypeError("config must be a BaselineTrainingConfig")
    ownership = resolve_parameter_ownership(model)
    parameters = [parameter for _, module, _ in _components(model) for parameter in module.parameters()]
    optimizer = torch.optim.Adam(
        parameters,
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    return optimizer, resolve_parameter_ownership(model, optimizer)


def _snapshots(model: PointGuidedMRIModel) -> dict[str, tuple[Tensor, ...]]:
    return {
        name: tuple(parameter.detach().clone() for parameter in module.parameters())
        for name, module, _ in _components(model)
    }


def _module_changed(before: tuple[Tensor, ...], module: torch.nn.Module) -> bool:
    return any(not torch.equal(previous, current.detach()) for previous, current in zip(before, module.parameters()))


def _finite_nonzero_gradient(module: torch.nn.Module) -> bool:
    gradients = tuple(parameter.grad for parameter in module.parameters())
    # A module is connected when at least one of its parameters receives a
    # finite nonzero gradient.  Individual bias gradients can legitimately be
    # exactly zero (for example, a symmetric softmax collapse), while the
    # trainable module still participates through its weights.
    return bool(gradients) and any(
        gradient is not None and bool(torch.isfinite(gradient).all()) and bool(torch.count_nonzero(gradient))
        for gradient in gradients
    )


def _assert_no_exact_revisit(selected_indices: Tensor) -> None:
    if not isinstance(selected_indices, Tensor) or selected_indices.ndim != 2 or selected_indices.dtype != torch.long:
        raise ValueError("selected_indices must be a [B,K] long tensor")
    for row in selected_indices:
        active = row[row >= 0]
        if torch.unique(active).numel() != active.numel():
            raise RuntimeError("Gate-F MAIN trajectory revisited an exact selected candidate")


def run_synthetic_smoke(
    model: PointGuidedMRIModel,
    observations: Tensor,
    target_t1ce: Tensor,
    *,
    training_config: BaselineTrainingConfig,
    supervision_config: SupervisionConfig,
) -> BaselineSmokeResult:
    """Run the F2 forward, Gate-E objective, backward, and one Adam step."""

    if not isinstance(observations, Tensor) or observations.ndim != 5 or observations.shape[1] != 3:
        raise ValueError("observations must have shape [B,3,D,H,W]")
    if not isinstance(target_t1ce, Tensor) or target_t1ce.shape != (observations.shape[0], 1, *observations.shape[-3:]):
        raise ValueError("target_t1ce must have shape [B,1,D,H,W] aligned with observations")
    if observations.dtype != target_t1ce.dtype or observations.device != target_t1ce.device:
        raise ValueError("observations and target_t1ce must share dtype and device")
    if not bool(torch.isfinite(observations).all()) or not bool(torch.isfinite(target_t1ce).all()):
        raise ValueError("synthetic smoke tensors must be finite")
    torch.manual_seed(training_config.seed)
    model.train()
    optimizer, ownership = build_baseline_optimizer(model, training_config)
    before = _snapshots(model)
    backbone_before = tuple(parameter.detach().clone() for parameter in model.semantic_prior.backbone.parameters())
    swt_before = tuple(buffer.detach().clone() for _, buffer in model.spectral_anchor_builder.swt.named_buffers())
    optimizer.zero_grad(set_to_none=True)
    context = model.forward_training_context(
        observations,
        chunk_size=training_config.decoder_chunk_size,
        availability_policy=BaselineNoRevisitPolicy(),
    )
    objective: TrainingObjectiveResult = model.compute_training_objective(
        context,
        target_t1ce,
        config=supervision_config,
        generator=torch.Generator(device=observations.device).manual_seed(training_config.seed),
    )
    if not bool(torch.isfinite(objective.total)):
        raise RuntimeError("Gate-F smoke objective is non-finite")
    objective.total.backward()
    gradient_modules = tuple(name for name, module, _ in _components(model) if _finite_nonzero_gradient(module))
    required = tuple(name for name, _, _ in _components(model))
    if gradient_modules != required:
        missing = sorted(set(required).difference(gradient_modules))
        raise RuntimeError(f"Gate-F smoke missing finite nonzero gradients for: {missing}")
    if any(parameter.grad is not None for parameter in model.semantic_prior.backbone.parameters()):
        raise RuntimeError("frozen MedicalNet backbone received a gradient in Gate-F smoke")
    if training_config.gradient_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], training_config.gradient_clip_norm)
    optimizer.step()
    changed_modules = tuple(
        name for name, module, _ in _components(model) if _module_changed(before[name], module)
    )
    if changed_modules != required:
        missing = sorted(set(required).difference(changed_modules))
        raise RuntimeError(f"Gate-F smoke did not update: {missing}")
    if any(not torch.equal(before_value, parameter.detach()) for before_value, parameter in zip(backbone_before, model.semantic_prior.backbone.parameters())):
        raise RuntimeError("frozen MedicalNet backbone changed after the Gate-F optimizer step")
    if any(not torch.equal(before_value, buffer.detach()) for before_value, (_, buffer) in zip(swt_before, model.spectral_anchor_builder.swt.named_buffers())):
        raise RuntimeError("fixed SWT-Haar filter buffer changed after the Gate-F optimizer step")

    with torch.no_grad():
        post_context = model.forward_training_context(
            observations,
            chunk_size=training_config.decoder_chunk_size,
            availability_policy=BaselineNoRevisitPolicy(),
        )
    _assert_no_exact_revisit(post_context.trajectory.selected_indices)
    max_displacement = float(torch.linalg.vector_norm(post_context.frontend.displacement_ras_mm, dim=-1).max())
    if max_displacement > 2.0 + 1e-6:
        raise RuntimeError("Gate-F optimizer step violated the locked 2-mm displacement bound")
    if post_context.reconstruction.prediction.shape != target_t1ce.shape:
        raise RuntimeError("Gate-F prediction shape must exactly match [B,1,D,H,W] target shape")
    return BaselineSmokeResult(
        prediction_shape=tuple(int(value) for value in post_context.reconstruction.prediction.shape),
        total_loss=float(objective.total.detach()),
        max_displacement_mm=max_displacement,
        gradient_modules=gradient_modules,
        changed_modules=changed_modules,
        selected_indices=post_context.trajectory.selected_indices.detach().clone(),
        ownership=ownership,
    )


__all__ = [
    "BaselineNoRevisitPolicy",
    "BaselineSmokeResult",
    "BaselineTrainingConfig",
    "ParameterOwnership",
    "build_baseline_optimizer",
    "resolve_parameter_ownership",
    "run_synthetic_smoke",
]
