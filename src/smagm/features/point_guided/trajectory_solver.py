"""Gate-C C4 adaptive hard/straight-through next-point selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SelectionResult:
    """One batched route decision; inactive rows use index ``-1`` and zero weights."""

    indices: Tensor  # [B], long; -1 denotes economic stop
    hard_weights: Tensor  # [B,N]
    weights: Tensor  # [B,N], hard forward / soft backward in training
    active: Tensor  # [B], bool
    max_utility: Tensor  # [B]

    def __post_init__(self) -> None:
        if not isinstance(self.indices, Tensor) or self.indices.ndim != 1 or self.indices.dtype != torch.long:
            raise ValueError("indices must be a rank-1 torch.long tensor")
        if not isinstance(self.active, Tensor) or self.active.shape != self.indices.shape or self.active.dtype != torch.bool:
            raise ValueError("active must be a bool tensor aligned with indices")
        if not isinstance(self.max_utility, Tensor) or self.max_utility.shape != self.indices.shape:
            raise ValueError("max_utility must align with indices")
        for name, value in (("hard_weights", self.hard_weights), ("weights", self.weights)):
            if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[0] != self.indices.shape[0]:
                raise ValueError(f"{name} must have shape [B,N]")
        if self.hard_weights.shape != self.weights.shape or self.weights.dtype != self.max_utility.dtype:
            raise ValueError("selection weights must share shape and utility dtype")


class AdaptiveRouteSolver(nn.Module):
    """Parameter-free receding-horizon utility solver, never a global route planner."""

    def forward(
        self,
        utility: Tensor,
        running: Tensor,
        *,
        training: bool,
        temperature: float,
    ) -> SelectionResult:
        if not isinstance(utility, Tensor) or utility.ndim != 2 or not utility.is_floating_point() or not bool(torch.isfinite(utility).all()):
            raise ValueError("utility must be a finite floating tensor [B,N]")
        if not isinstance(running, Tensor) or running.shape != utility.shape[:1] or running.dtype != torch.bool or running.device != utility.device:
            raise ValueError("running must be a device-matched bool tensor [B]")
        if not isinstance(temperature, float) or not torch.isfinite(torch.tensor(temperature)) or temperature <= 0.0:
            raise ValueError("temperature must be positive and finite")
        # Gate C deliberately permits revisiting a point.  Explicit travel,
        # overlap, and step costs discourage redundant choices economically;
        # a structural no-revisit mask belongs to the later Gate-G policy.
        max_utility, raw_indices = utility.max(dim=-1)
        active = running & (max_utility > 0.0)
        indices = torch.where(active, raw_indices, torch.full_like(raw_indices, -1))
        hard = torch.zeros_like(utility)
        if bool(active.any()):
            hard.scatter_(1, raw_indices.unsqueeze(1), active.to(dtype=utility.dtype).unsqueeze(1))
        if training:
            # A stopped row remains all-zero; every candidate stays eligible
            # for running rows, including previously selected candidates.
            soft = torch.softmax(utility / temperature, dim=-1)
            soft = torch.where(active.unsqueeze(1), soft, torch.zeros_like(soft))
            weights = hard + soft - soft.detach()
        else:
            weights = hard
        return SelectionResult(indices=indices, hard_weights=hard, weights=weights, active=active, max_utility=max_utility)


__all__ = ["AdaptiveRouteSolver", "SelectionResult"]
