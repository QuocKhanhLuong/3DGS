"""Type-only boundaries for research decisions deliberately left unresolved."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

from torch import Tensor, nn

from .contracts import PointField


class SpectralAnchorBase(nn.Module, ABC):
    """TODO: choose global spectral representation and modality fusion."""

    @abstractmethod
    def forward(self, x: Tensor) -> Any:
        raise NotImplementedError


class InitialDynamicTriPlaneBase(nn.Module, ABC):
    """TODO: choose initial tri-plane encoder, channels, resolution, and projection."""

    @abstractmethod
    def forward(self, x: Tensor) -> Any:
        raise NotImplementedError


class TrajectoryHistory(Protocol):
    """TODO: trajectory-history representation is intentionally undecided."""


class TrajectorySelectorBase(nn.Module, ABC):
    """TODO: specify selector scoring, cardinality, revisit, and stop policy."""

    @abstractmethod
    def forward(
        self,
        dynamic_triplane: Any,
        spectral_anchor: Any,
        point_field: PointField,
        history: TrajectoryHistory,
    ) -> Any:
        raise NotImplementedError


class LocalTrajectoryUpdaterBase(nn.Module, ABC):
    """TODO: specify local fusion, correction, scatter, and conflict handling."""

    @abstractmethod
    def forward(
        self,
        local_dynamic_triplane: Any,
        local_spectral_anchor: Any,
        point_info: PointField,
        history: TrajectoryHistory,
    ) -> Any:
        raise NotImplementedError


class StoppingPolicyBase(Protocol):
    """TODO: choose fixed, learned, convergence, or no-useful-point stopping."""


class FinalTriPlaneDecoderBase(nn.Module, ABC):
    """TODO: specify tri-plane querying, fusion, activation, and T1ce decoder."""

    @abstractmethod
    def forward(self, dynamic_triplane: Any) -> Tensor:
        raise NotImplementedError


class ReconstructionLossConfig(Protocol):
    """TODO: reconstruction losses are intentionally not selected yet."""


# Concise aliases do not add behaviour; they preserve the research terminology.
SpectralAnchor = SpectralAnchorBase
InitialDynamicTriPlane = InitialDynamicTriPlaneBase
TrajectorySelector = TrajectorySelectorBase
LocalTrajectoryUpdater = LocalTrajectoryUpdaterBase
StoppingPolicy = StoppingPolicyBase
FinalTriPlaneDecoder = FinalTriPlaneDecoderBase
ReconstructionLossConfigBase = ReconstructionLossConfig


__all__ = [
    "FinalTriPlaneDecoder",
    "FinalTriPlaneDecoderBase",
    "InitialDynamicTriPlane",
    "InitialDynamicTriPlaneBase",
    "LocalTrajectoryUpdater",
    "LocalTrajectoryUpdaterBase",
    "ReconstructionLossConfig",
    "ReconstructionLossConfigBase",
    "SpectralAnchor",
    "SpectralAnchorBase",
    "StoppingPolicy",
    "StoppingPolicyBase",
    "TrajectoryHistory",
    "TrajectorySelector",
    "TrajectorySelectorBase",
]
