"""Gate-G deterministic, target-free baseline inference policy.

This module owns only the operational G1--G4 overlay: hard greedy selection,
exact no revisit, per-subject stopping, compact diagnostics, and one final
Gate-D decode.  It reuses the trained frontend modules and introduces no
parameters, targets, losses, optimizers, or held-out evaluation machinery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .availability import ExactNoRevisitPolicy
from .contracts import POINT_SPECTRAL_EVIDENCE_CHANNELS, VolumeGeometry
from .decoder import ImplicitTriPlaneDecoder
from .reward import GateBDescriptorContext
from .spectral_query import FeatureGridGeometry
from .trajectory import AdaptiveRewardCostTrajectory, TrajectoryResult
from .trajectory_cost import TrajectoryConfig
from .triplane_projection import BaseTriPlanes


_CHECKPOINT_SCHEMA = "point-guided-gate-f-baseline-v1"


def _positive_finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


@dataclass(frozen=True)
class GateGInferenceConfig:
    """Initial operational Gate-G values from ``PLAN_GATE_F_G.md``.

    These are explicit PLAN initial values, not validation-selected settings.
    ``decoder_chunk_size`` is an engineering memory bound only.  Selection is
    always hard greedy, so no temperature field belongs to this config.
    """

    lambda_travel: float = 0.05
    lambda_overlap: float = 0.20
    lambda_step: float = 0.05
    k_max: int = 64
    decoder_chunk_size: int = 65_536

    def __post_init__(self) -> None:
        for name in ("lambda_travel", "lambda_overlap", "lambda_step"):
            object.__setattr__(self, name, _positive_finite(name, getattr(self, name)))
        for name in ("k_max", "decoder_chunk_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def route_config(self, *, write_scale: float) -> TrajectoryConfig:
        """Adapt G's public hard-policy config to the shared trajectory type."""

        return TrajectoryConfig(
            lambda_travel=self.lambda_travel,
            lambda_overlap=self.lambda_overlap,
            lambda_step=self.lambda_step,
            k_max=self.k_max,
            # The common solver validates this field even though inference
            # passes ``training=False`` and never consumes a soft temperature.
            selection_temperature=1.0,
            write_scale=write_scale,
        )


@dataclass(frozen=True)
class BaselineInferenceResult:
    """Target-free Gate-G result with selected-step-only diagnostics."""

    prediction: Tensor  # [B,1,D,H,W]
    selected_indices: Tensor  # [B,K], -1 for inactive compact diagnostics
    k_used: Tensor  # [B] long
    stop_reasons: tuple[str, ...]
    path_length_mm: Tensor  # [B]
    reward_mean: Tensor  # [B], selected-step aggregate (zero if no selection)
    reward_max: Tensor  # [B], selected-step aggregate (zero if no selection)
    utility_mean: Tensor  # [B], selected-step aggregate (zero if no selection)
    utility_max: Tensor  # [B], selected-step aggregate (zero if no selection)
    update_magnitude_mean: Tensor  # [B], selected-step aggregate
    update_magnitude_max: Tensor  # [B], selected-step aggregate
    candidate_evaluations: Tensor  # [B] long, actual dense RewardNet score count
    eligible_candidate_evaluations: Tensor  # [B] long, pre-mask eligible count
    semantic_probabilities: Tensor | None = None  # [B,3,D,H,W], target-free diagnostic output

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, Tensor) or self.prediction.ndim != 5 or self.prediction.shape[1] != 1:
            raise ValueError("prediction must be a [B,1,D,H,W] tensor")
        batch = self.prediction.shape[0]
        if batch <= 0 or not self.prediction.is_floating_point() or not bool(torch.isfinite(self.prediction).all()):
            raise ValueError("prediction must be finite with positive batch size")
        if (
            not isinstance(self.selected_indices, Tensor)
            or self.selected_indices.ndim != 2
            or self.selected_indices.shape[0] != batch
            or self.selected_indices.dtype != torch.long
            or self.selected_indices.device != self.prediction.device
        ):
            raise ValueError("selected_indices must be a device-matched [B,K] long tensor")
        if len(self.stop_reasons) != batch or any(not isinstance(reason, str) for reason in self.stop_reasons):
            raise ValueError("stop_reasons must contain one string per subject")
        for name in ("k_used", "candidate_evaluations", "eligible_candidate_evaluations"):
            value = getattr(self, name)
            if (
                not isinstance(value, Tensor)
                or value.shape != (batch,)
                or value.dtype != torch.long
                or value.device != self.prediction.device
                or bool((value < 0).any())
            ):
                raise ValueError(f"{name} must be a nonnegative device-matched [B] long tensor")
        if bool((self.eligible_candidate_evaluations > self.candidate_evaluations).any()):
            raise ValueError("eligible candidate evaluations cannot exceed actual dense RewardNet scores")
        if self.semantic_probabilities is not None:
            if (
                not isinstance(self.semantic_probabilities, Tensor)
                or self.semantic_probabilities.shape != (batch, 3, *self.prediction.shape[-3:])
                or self.semantic_probabilities.device != self.prediction.device
                or self.semantic_probabilities.dtype != self.prediction.dtype
            ):
                raise ValueError("semantic_probabilities must be a device-matched [B,3,D,H,W] tensor")
        if not torch.equal(self.k_used, (self.selected_indices >= 0).sum(dim=1, dtype=torch.long)):
            raise ValueError("k_used must equal the count of nonnegative selected indices")
        for name in (
            "path_length_mm",
            "reward_mean",
            "reward_max",
            "utility_mean",
            "utility_max",
            "update_magnitude_mean",
            "update_magnitude_max",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, Tensor)
                or value.shape != (batch,)
                or value.dtype != self.prediction.dtype
                or value.device != self.prediction.device
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"{name} must be a finite device-matched [B] floating tensor")


def _selected_statistics(result: TrajectoryResult, name: str) -> tuple[Tensor, Tensor]:
    """Aggregate only actual selected steps, never inactive zero sentinels."""

    batch = result.final_state.xy.shape[0]
    dtype = result.final_state.xy.dtype
    device = result.final_state.xy.device
    indices = result.selected_indices
    if not result.steps:
        zeros = torch.zeros(batch, dtype=dtype, device=device)
        return zeros, zeros
    values = torch.stack(tuple(getattr(step, name) for step in result.steps), dim=1)
    active = indices >= 0
    counts = active.sum(dim=1)
    totals = (values * active.to(dtype=dtype)).sum(dim=1)
    mean = torch.where(counts > 0, totals / counts.clamp_min(1).to(dtype=dtype), torch.zeros_like(totals))
    maximum = values.masked_fill(~active, -torch.inf).max(dim=1).values
    maximum = torch.where(counts > 0, maximum, torch.zeros_like(maximum))
    return mean, maximum


def _path_length_mm(refined_points_ras_mm: Tensor, selected_indices: Tensor) -> Tensor:
    batch = refined_points_ras_mm.shape[0]
    result = torch.zeros(batch, dtype=refined_points_ras_mm.dtype, device=refined_points_ras_mm.device)
    for row in range(batch):
        indices = selected_indices[row]
        indices = indices[indices >= 0]
        if indices.numel() > 1:
            selected = refined_points_ras_mm[row, indices]
            result[row] = torch.linalg.vector_norm(selected[1:] - selected[:-1], dim=-1).sum()
    return result


def run_baseline_inference(
    trajectory: AdaptiveRewardCostTrajectory,
    decoder: ImplicitTriPlaneDecoder,
    base_planes: BaseTriPlanes,
    refined_points_ras_mm: Tensor,
    point_semantic: Tensor,
    f_spec: Tensor,
    reliability: Tensor,
    gate_b_descriptors: GateBDescriptorContext,
    feature_geometry: FeatureGridGeometry,
    geometry: VolumeGeometry,
    *,
    config: GateGInferenceConfig,
    semantic_probabilities: Tensor | None = None,
) -> BaselineInferenceResult:
    """Execute G1--G4 once from fixed target-free frontend evidence.

    Callers must place the model in ``eval`` and enter ``torch.no_grad``;
    enforcing both here prevents accidental Gate-E/training semantics from
    entering the public Gate-G path.
    """

    if not isinstance(trajectory, AdaptiveRewardCostTrajectory) or not isinstance(decoder, ImplicitTriPlaneDecoder):
        raise TypeError("Gate-G requires the completed shared trajectory and implicit decoder")
    if not isinstance(config, GateGInferenceConfig):
        raise TypeError("config must be a GateGInferenceConfig")
    if trajectory.training or decoder.training:
        raise RuntimeError("Gate-G baseline inference requires eval-mode trajectory and decoder")
    if torch.is_grad_enabled():
        raise RuntimeError("Gate-G baseline inference requires torch.no_grad()")
    if not isinstance(f_spec, Tensor) or f_spec.shape[-1] != POINT_SPECTRAL_EVIDENCE_CHANNELS:
        raise ValueError("f_spec must retain the exact 168-channel Gate-B evidence")

    route = trajectory._run(
        base_planes,
        refined_points_ras_mm,
        point_semantic,
        f_spec,
        reliability,
        gate_b_descriptors,
        feature_geometry,
        geometry,
        trace_states=None,
        availability_policy=ExactNoRevisitPolicy(),
        route_config=config.route_config(write_scale=trajectory.config.write_scale),
    )
    # This is intentionally the sole dense Gate-D invocation, after Z_K.
    prediction = decoder(
        route.final_state,
        feature_geometry,
        geometry,
        chunk_size=config.decoder_chunk_size,
    )
    reward_mean, reward_max = _selected_statistics(route, "selected_reward")
    utility_mean, utility_max = _selected_statistics(route, "selected_utility")
    update_mean, update_max = _selected_statistics(route, "selected_update_norm")
    return BaselineInferenceResult(
        prediction=prediction,
        selected_indices=route.selected_indices,
        k_used=route.route_lengths,
        stop_reasons=route.stop_reasons,
        path_length_mm=_path_length_mm(refined_points_ras_mm, route.selected_indices),
        reward_mean=reward_mean,
        reward_max=reward_max,
        utility_mean=utility_mean,
        utility_max=utility_max,
        update_magnitude_mean=update_mean,
        update_magnitude_max=update_max,
        candidate_evaluations=route.candidate_evaluations,
        eligible_candidate_evaluations=route.eligible_candidate_evaluations,
        semantic_probabilities=semantic_probabilities,
    )


def _canonical_config(value: object) -> Mapping[str, Any]:
    if not hasattr(value, "__dataclass_fields__"):
        raise TypeError("checkpoint metadata requires dataclass-backed model and trajectory configs")
    # JSON round-trip gives a stable built-in-only structure and handles a
    # possible Path checkpoint field without silently omitting it.
    return json.loads(json.dumps(asdict(value), sort_keys=True, default=str))


def baseline_checkpoint_metadata(model: nn.Module) -> Mapping[str, Any]:
    """Return the strict architecture metadata required by a future F4 checkpoint."""

    trajectory = getattr(model, "trajectory", None)
    model_config = getattr(model, "config", None)
    if not isinstance(trajectory, AdaptiveRewardCostTrajectory) or model_config is None:
        raise TypeError("checkpoint metadata requires a PointGuidedMRIModel with an explicit trajectory")
    return {
        "schema": _CHECKPOINT_SCHEMA,
        "model_config": _canonical_config(model_config),
        "trajectory_config": _canonical_config(trajectory.config),
        "decoder_architecture": "96->64->32->1",
        "gate_e_architecture": "target-after-inference objective",
    }


def load_validated_baseline_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> None:
    """Strictly load a later F4 checkpoint after exact metadata validation.

    It accepts no partial state dict and does not create, label, or imply a
    trained checkpoint.  It exists so later operational inference can reject
    architecture/configuration mismatches deterministically.
    """

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"baseline checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or set(payload) != {"metadata", "state_dict"}:
        raise ValueError("baseline checkpoint must contain exactly metadata and state_dict")
    if payload["metadata"] != baseline_checkpoint_metadata(model):
        raise ValueError("baseline checkpoint metadata does not match the current model architecture/configuration")
    state_dict = payload["state_dict"]
    if not isinstance(state_dict, Mapping) or not all(isinstance(name, str) and isinstance(value, Tensor) for name, value in state_dict.items()):
        raise ValueError("baseline checkpoint state_dict must map parameter names to tensors")
    model.load_state_dict(state_dict, strict=True)


__all__ = [
    "BaselineInferenceResult",
    "GateGInferenceConfig",
    "baseline_checkpoint_metadata",
    "load_validated_baseline_checkpoint",
    "run_baseline_inference",
]
