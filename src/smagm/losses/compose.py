"""Typed composition of reconstruction and optional structural objectives."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import torch

from .reconstruction import ReconstructionLossResult


@dataclass(frozen=True)
class ObjectiveResult:
    total: torch.Tensor
    components: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        if self.total.ndim != 0 or not bool(torch.isfinite(self.total)):
            raise ValueError("objective total must be one finite scalar")
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))


def compose_objective(
    reconstruction: ReconstructionLossResult,
    *,
    structural_components: Mapping[str, torch.Tensor] | None = None,
    structural_weights: Mapping[str, float] | None = None,
) -> ObjectiveResult:
    """Compose already-computed components without owning data or masks."""

    if reconstruction.status != "OK":
        raise ValueError("a skipped reconstruction result cannot enter optimization")
    supplied = dict(structural_components or {})
    weights = dict(structural_weights or {})
    if set(supplied) != set(weights):
        raise ValueError("each structural component requires exactly one explicit weight")
    components = {f"reconstruction/{key}": value for key, value in reconstruction.components.items()}
    total = reconstruction.total
    for name, value in supplied.items():
        weight = weights[name]
        if not isinstance(value, torch.Tensor) or value.ndim != 0 or not bool(torch.isfinite(value)):
            raise ValueError(f"structural component {name} must be a finite scalar")
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"structural weight {name} must be finite and non-negative")
        components[f"structural/{name}"] = value
        total = total + weight * value
    return ObjectiveResult(total, components)
