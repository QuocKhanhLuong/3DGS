"""Gate-C C3 typed reward-cost configuration and explicit routing costs."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


LOCKED_SUPPORT_RADIUS_MM = 4.0


def _positive_finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


@dataclass(frozen=True)
class TrajectoryConfig:
    """Explicit Gate-C runtime choices; no Gate-F/G tuned defaults are implied.

    ``bounded_travel_cost`` and ``separate_halt_from_utility`` are opt-in so
    older Gate-F/G configurations keep their historical behavior.  In the
    corrected point-guided training configuration, physical travel is mapped
    to ``[0, 1)`` before weighting and ``lambda_step`` is retained as the
    serialized minimum raw RewardNet gain required to continue rather than a
    constant subtracted from every candidate ranking score.
    """

    lambda_travel: float
    lambda_overlap: float
    lambda_step: float
    k_max: int
    selection_temperature: float
    write_scale: float
    support_radius_mm: float = LOCKED_SUPPORT_RADIUS_MM
    bounded_travel_cost: bool = False
    separate_halt_from_utility: bool = False
    training_exploration_steps: int = 0

    def __post_init__(self) -> None:
        for name in ("lambda_travel", "lambda_overlap", "lambda_step", "selection_temperature", "write_scale"):
            object.__setattr__(self, name, _positive_finite(name, getattr(self, name)))
        if not isinstance(self.k_max, int) or isinstance(self.k_max, bool) or self.k_max <= 0:
            raise ValueError("k_max must be a positive integer")
        radius = _positive_finite("support_radius_mm", self.support_radius_mm)
        if radius != LOCKED_SUPPORT_RADIUS_MM:
            raise ValueError("support_radius_mm must be exactly 4.0 mm in Gate-C MAIN")
        object.__setattr__(self, "support_radius_mm", radius)
        for name in ("bounded_travel_cost", "separate_halt_from_utility"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        if (
            not isinstance(self.training_exploration_steps, int)
            or isinstance(self.training_exploration_steps, bool)
            or self.training_exploration_steps < 0
            or self.training_exploration_steps > self.k_max
        ):
            raise ValueError("training_exploration_steps must be an integer in [0, k_max]")

    @property
    def halt_reward_threshold(self) -> float:
        """Minimum raw expected gain used when halt and ranking are separated."""

        return self.lambda_step


def _points(name: str, value: Tensor, *, rank: int) -> None:
    if not isinstance(value, Tensor) or value.ndim != rank or value.shape[-1] != 3 or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor ending in XYZ dimension 3")
    if value.shape[0] <= 0 or (rank == 3 and value.shape[1] < 0) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must have positive batch and finite values")


def travel_cost(
    candidate_points_ras_mm: Tensor,
    previous_indices: Tensor,
    *,
    support_radius_mm: float = LOCKED_SUPPORT_RADIUS_MM,
    bounded: bool = False,
) -> Tensor:
    """Return route locality cost; first step is always zero.

    The legacy form is physical distance divided by the locked 4-mm support
    radius.  The corrected form additionally maps that non-negative ratio
    ``d`` to ``d / (1 + d)``, preserving ordering while preventing an
    unbounded millimetre-scale term from overwhelming RewardNet's ``[0, 1]``
    score range.
    """

    _points("candidate_points_ras_mm", candidate_points_ras_mm, rank=3)
    if not isinstance(previous_indices, Tensor) or previous_indices.shape != (candidate_points_ras_mm.shape[0],):
        raise ValueError("previous_indices must have shape [B]")
    if not isinstance(bounded, bool):
        raise ValueError("bounded must be a bool")
    radius = _positive_finite("support_radius_mm", support_radius_mm)
    if radius != LOCKED_SUPPORT_RADIUS_MM:
        raise ValueError("support_radius_mm must be exactly 4.0 mm")
    batch, points, _ = candidate_points_ras_mm.shape
    if previous_indices.dtype != torch.long or previous_indices.device != candidate_points_ras_mm.device:
        raise ValueError("previous_indices must be a device-matched torch.long tensor")
    if bool((previous_indices >= points).any()) or bool((previous_indices < -1).any()):
        raise ValueError("previous_indices must contain -1 or valid candidate indices")
    safe_indices = previous_indices.clamp_min(0)
    previous = candidate_points_ras_mm[torch.arange(batch, device=candidate_points_ras_mm.device), safe_indices]
    distances = torch.linalg.vector_norm(candidate_points_ras_mm - previous.unsqueeze(1), dim=-1) / radius
    if bounded:
        distances = distances / (1.0 + distances)
    return torch.where(previous_indices.unsqueeze(1) < 0, torch.zeros_like(distances), distances)


def overlap_cost(candidate_points_ras_mm: Tensor, visited_points_ras_mm: Tensor, *, support_radius_mm: float = LOCKED_SUPPORT_RADIUS_MM) -> Tensor:
    """Maximum compact support overlap with any earlier selected physical point."""

    _points("candidate_points_ras_mm", candidate_points_ras_mm, rank=3)
    _points("visited_points_ras_mm", visited_points_ras_mm, rank=3)
    if visited_points_ras_mm.shape[0] != candidate_points_ras_mm.shape[0] or visited_points_ras_mm.dtype != candidate_points_ras_mm.dtype or visited_points_ras_mm.device != candidate_points_ras_mm.device:
        raise ValueError("visited points must share candidate batch, dtype, and device")
    radius = _positive_finite("support_radius_mm", support_radius_mm)
    if radius != LOCKED_SUPPORT_RADIUS_MM:
        raise ValueError("support_radius_mm must be exactly 4.0 mm")
    result = torch.zeros(candidate_points_ras_mm.shape[:2], dtype=candidate_points_ras_mm.dtype, device=candidate_points_ras_mm.device)
    for visited in visited_points_ras_mm.unbind(dim=1):
        distance = torch.linalg.vector_norm(candidate_points_ras_mm - visited.unsqueeze(1), dim=-1)
        result = torch.maximum(result, torch.square(torch.clamp(1.0 - distance / (2.0 * radius), min=0.0)))
    return result


def route_utility(reward: Tensor, travel: Tensor, overlap: Tensor, config: TrajectoryConfig) -> Tensor:
    """Return the candidate ranking score used by the route solver.

    Legacy configurations retain ``R - lambda_travel*T - lambda_overlap*O -
    lambda_step``.  Corrected configurations remove the candidate-independent
    step term from ranking; stopping is then decided separately from raw
    expected gain.
    """

    if not isinstance(config, TrajectoryConfig):
        raise TypeError("config must be a TrajectoryConfig")
    for name, value in (("reward", reward), ("travel", travel), ("overlap", overlap)):
        if not isinstance(value, Tensor) or value.ndim != 2 or not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be a finite floating [B,N] tensor")
        if value.shape != reward.shape or value.dtype != reward.dtype or value.device != reward.device:
            raise ValueError(f"{name} must match reward shape, dtype, and device")
    if bool((travel < 0.0).any()) or bool((overlap < 0.0).any()):
        raise ValueError("travel and overlap costs must be nonnegative")
    utility = reward - config.lambda_travel * travel - config.lambda_overlap * overlap
    if not config.separate_halt_from_utility:
        utility = utility - config.lambda_step
    return utility


__all__ = [
    "LOCKED_SUPPORT_RADIUS_MM",
    "TrajectoryConfig",
    "overlap_cost",
    "route_utility",
    "travel_cost",
]
