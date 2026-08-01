"""One low-capacity StructuralField shared across anchors and patients."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import torch
from torch import nn


@dataclass(frozen=True)
class StructuralFieldConfig:
    evidence_dim: int
    hidden_width: int = 32
    hidden_layers: int = 2
    activation: str = "silu"

    def __post_init__(self) -> None:
        if self.evidence_dim <= 0 or not 16 <= self.hidden_width <= 64 or not 2 <= self.hidden_layers <= 4:
            raise ValueError("StructuralField must stay within the locked low-capacity envelope")
        if self.activation not in ("silu", "softplus"):
            raise ValueError("activation must be silu or softplus")

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class SharedStructuralField(nn.Module):
    """Map normalized local coordinate plus compact evidence to one scalar."""

    def __init__(self, config: StructuralFieldConfig) -> None:
        super().__init__()
        self.config = config
        activation: type[nn.Module] = nn.SiLU if config.activation == "silu" else nn.Softplus
        layers: list[nn.Module] = [nn.Linear(3 + config.evidence_dim, config.hidden_width), activation()]
        for _ in range(config.hidden_layers - 1):
            layers.extend((nn.Linear(config.hidden_width, config.hidden_width), activation()))
        layers.append(nn.Linear(config.hidden_width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, local_coordinates: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        if local_coordinates.shape[:-1] != evidence.shape[:-1] or local_coordinates.shape[-1] != 3 or evidence.shape[-1] != self.config.evidence_dim:
            raise ValueError("local coordinates and evidence have incompatible field shapes")
        values = self.network(torch.cat((local_coordinates, evidence), dim=-1))
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError("StructuralField produced non-finite values")
        return values
