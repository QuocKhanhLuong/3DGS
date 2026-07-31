"""Training-side composition of legal reconstruction and structural terms."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import torch

from ..losses.compose import ObjectiveResult, compose_objective
from ..losses.reconstruction import ReconstructionLossResult
from .schedule import StageConfig


@dataclass(frozen=True)
class InactiveObjectiveComponent:
    """An explicit reason why an otherwise declared component did not run."""

    name: str
    reason: str

    def __post_init__(self) -> None:
        if not self.name or not self.reason:
            raise ValueError("inactive objective components require a name and reason")


@dataclass(frozen=True)
class T1CObjectiveConfig:
    """Explicit structural weights; no target term is implied by this config."""

    structural_weights: tuple[tuple[str, float], ...] = (
        ("structural_consistency", 1.0),
        ("appearance_sensitivity", 0.1),
        ("reliability_regularization", 0.1),
        ("variance_floor", 0.1),
    )

    def __post_init__(self) -> None:
        names = [name for name, _ in self.structural_weights]
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("structural objective names must be unique and non-empty")
        if any(not math.isfinite(weight) or weight < 0.0 for _, weight in self.structural_weights):
            raise ValueError("structural weights must be finite and non-negative")

    @property
    def weights(self) -> dict[str, float]:
        return dict(self.structural_weights)

    def to_dict(self) -> dict[str, object]:
        return {"structural_weights": list(self.structural_weights)}


@dataclass(frozen=True)
class T1CObjectiveResult:
    """A finite total with active terms and typed inactive-term diagnostics."""

    total: torch.Tensor
    components: Mapping[str, torch.Tensor]
    inactive_components: Mapping[str, InactiveObjectiveComponent]

    def __post_init__(self) -> None:
        if not isinstance(self.total, torch.Tensor) or self.total.ndim != 0 or not bool(torch.isfinite(self.total)):
            raise ValueError("T1-C objective total must be one finite scalar")
        for name, value in self.components.items():
            if not name or not isinstance(value, torch.Tensor) or value.ndim != 0 or not bool(torch.isfinite(value)):
                raise ValueError("active objective components must be named finite scalars")
        if any(not name or not isinstance(component, InactiveObjectiveComponent) or component.name != name for name, component in self.inactive_components.items()):
            raise ValueError("inactive objective components must use explicit typed reasons")
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))
        object.__setattr__(self, "inactive_components", MappingProxyType(dict(self.inactive_components)))


def resolve_objective(
    reconstruction: ReconstructionLossResult,
    *,
    structural_components: Mapping[str, torch.Tensor] | None = None,
    config: T1CObjectiveConfig | None = None,
) -> ObjectiveResult:
    """Compatibility helper for a joint reconstruction objective."""

    config = config or T1CObjectiveConfig()
    return compose_objective(
        reconstruction,
        structural_components=structural_components,
        structural_weights=config.weights,
    )


def compose_t1c_objective(
    *,
    stage: StageConfig,
    reconstruction: ReconstructionLossResult | None,
    structural_components: Mapping[str, torch.Tensor],
    inactive_components: Mapping[str, str] | None = None,
    config: T1CObjectiveConfig | None = None,
) -> T1CObjectiveResult:
    """Compose declared components without opening target data or mutating state."""

    if not isinstance(stage, StageConfig):
        raise TypeError("stage must be a StageConfig")
    config = config or T1CObjectiveConfig()
    supplied = dict(structural_components)
    inactive = dict(inactive_components or {})
    unknown = set(supplied) - set(config.weights)
    if unknown:
        raise ValueError(f"structural components lack declared weights: {sorted(unknown)}")
    if stage.reconstruction_weight > 0.0:
        if reconstruction is None:
            raise ValueError("a reconstruction-enabled stage requires a reconstruction result")
        if reconstruction.status != "OK":
            raise ValueError("a skipped reconstruction result cannot enter optimization")
        total = reconstruction.total * stage.reconstruction_weight
        components = {f"reconstruction/{name}": value for name, value in reconstruction.components.items()}
    else:
        if reconstruction is not None:
            raise ValueError("an auxiliary-only stage must not receive target reconstruction")
        total: torch.Tensor | None = None
        components: dict[str, torch.Tensor] = {}
    for name, value in supplied.items():
        if not isinstance(value, torch.Tensor) or value.ndim != 0 or not bool(torch.isfinite(value)):
            raise ValueError(f"structural component {name} must be a finite scalar")
        weight = config.weights[name] * stage.structural_weight
        components[f"structural/{name}"] = value
        contribution = value * weight
        total = contribution if total is None else total + contribution
    if total is None:
        raise ValueError("objective has no active legal components")
    typed_inactive = {name: InactiveObjectiveComponent(name, reason) for name, reason in inactive.items()}
    return T1CObjectiveResult(total, components, typed_inactive)
