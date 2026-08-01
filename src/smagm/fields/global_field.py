"""Matched low-capacity global-coordinate structural-field baseline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import torch
from torch import nn


@dataclass(frozen=True)
class GlobalStructuralFieldConfig:
    evidence_dim: int
    coordinate_scale_mm: float
    hidden_width: int = 32
    hidden_layers: int = 2
    activation: str = "silu"

    def __post_init__(self) -> None:
        if self.evidence_dim <= 0 or self.coordinate_scale_mm <= 0:
            raise ValueError("global field needs positive evidence dimension and coordinate scale")
        if not 16 <= self.hidden_width <= 64 or not 2 <= self.hidden_layers <= 4:
            raise ValueError("global field must match the locked low-capacity envelope")
        if self.activation not in ("silu", "softplus"):
            raise ValueError("activation must be silu or softplus")

    @property
    def config_hash(self) -> str:
        canonical = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class GlobalStructuralField(nn.Module):
    """Map canonical RAS-mm coordinates and pooled context evidence to a scalar."""

    def __init__(self, config: GlobalStructuralFieldConfig) -> None:
        super().__init__()
        self.config = config
        activation: type[nn.Module] = nn.SiLU if config.activation == "silu" else nn.Softplus
        layers: list[nn.Module] = [nn.Linear(3 + config.evidence_dim, config.hidden_width), activation()]
        for _ in range(config.hidden_layers - 1):
            layers.extend((nn.Linear(config.hidden_width, config.hidden_width), activation()))
        layers.append(nn.Linear(config.hidden_width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, points_ras_mm: torch.Tensor, pooled_evidence: torch.Tensor) -> torch.Tensor:
        if points_ras_mm.ndim != 2 or points_ras_mm.shape[1] != 3:
            raise ValueError("points_ras_mm must have shape [Q,3]")
        if pooled_evidence.ndim == 1:
            pooled_evidence = pooled_evidence.expand(points_ras_mm.shape[0], -1)
        if pooled_evidence.shape != (points_ras_mm.shape[0], self.config.evidence_dim):
            raise ValueError("pooled_evidence must have shape [C] or [Q,C]")
        if not bool(torch.isfinite(points_ras_mm).all() and torch.isfinite(pooled_evidence).all()):
            raise ValueError("global field inputs must be finite")
        normalized_coordinates = points_ras_mm / float(self.config.coordinate_scale_mm)
        output = self.network(torch.cat((normalized_coordinates, pooled_evidence), dim=-1))
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("global field produced non-finite values")
        return output


__all__ = ["GlobalStructuralField", "GlobalStructuralFieldConfig"]
