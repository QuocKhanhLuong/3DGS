"""Training-side resolution of already legal objective components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from ..losses.compose import ObjectiveResult, compose_objective
from ..losses.reconstruction import ReconstructionLossResult


@dataclass(frozen=True)
class T1CObjectiveConfig:
    structural_weights: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        names = [name for name, _ in self.structural_weights]
        if len(set(names)) != len(names):
            raise ValueError("structural objective names must be unique")

    @property
    def weights(self) -> dict[str, float]:
        return dict(self.structural_weights)


def resolve_objective(
    reconstruction: ReconstructionLossResult,
    *,
    structural_components: Mapping[str, torch.Tensor] | None = None,
    config: T1CObjectiveConfig | None = None,
) -> ObjectiveResult:
    config = config or T1CObjectiveConfig()
    return compose_objective(
        reconstruction,
        structural_components=structural_components,
        structural_weights=config.weights,
    )
