"""PFGR-Lite static B0/B1/B2/B-light synthesis heads.

The heads consume the three feature maps already emitted by the one
MedicalNet traversal.  B/A from the legacy frontend remains an explicit
feature-only branch; these modules produce only the 32-channel dynamic Z0
planes consumed by the bounded PFGR updater/decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..medicalnet_resnet10 import MedicalNetFeatures
from ..state_init import DYNAMIC_STATE_CHANNELS, DynamicStateInitializer, DynamicTriPlanes
from ..triplane_projection import BaseTriPlanes
from .config import StaticSynthesisConfig
from .static_geometry import (
    FeatureLattice,
    MultiScaleFeatureGeometry,
    resample_plane_between_lattices,
    sample_source_to_lattice,
    source_plane_means,
)


BASE_CHANNELS = 64
SOURCE_CHANNELS = 3
STATE_CHANNELS = DYNAMIC_STATE_CHANNELS


def _check_feature(name: str, value: Tensor, channels: int) -> None:
    if not isinstance(value, Tensor) or value.ndim != 5 or value.shape[1] != channels or not value.is_floating_point():
        raise ValueError(f"{name} must have shape [B,{channels},D,H,W]")
    if value.shape[0] <= 0 or any(size <= 0 for size in value.shape[-3:]) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite with positive spatial dimensions")


def _check_source(source: Tensor) -> None:
    if not isinstance(source, Tensor) or source.ndim != 5 or source.shape[1] != SOURCE_CHANNELS or not source.is_floating_point():
        raise ValueError("source must have shape [B,3,D,H,W]")
    if source.shape[0] <= 0 or not bool(torch.isfinite(source).all()):
        raise ValueError("source must be finite with positive batch size")


class _Residual2d(nn.Module):
    """Conv3x3-SiLU-Conv3x3 residual block with zero final residual."""

    def __init__(self, channels: int = BASE_CHANNELS) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.activation = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        # Zero-initialising the residual branch makes all variants begin at
        # their deterministic projection/B baseline while retaining gradients.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, value: Tensor) -> Tensor:
        return value + self.conv2(self.activation(self.conv1(value)))


class _AxisConditionedCollapse(nn.Module):
    """Normalized per-axis weighted collapse of one 3-D feature volume."""

    def __init__(self, channels: int = BASE_CHANNELS) -> None:
        super().__init__()
        self.xy_logits = nn.Conv3d(channels, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=True)
        self.xz_logits = nn.Conv3d(channels, 1, kernel_size=(1, 3, 1), padding=(0, 1, 0), bias=True)
        self.yz_logits = nn.Conv3d(channels, 1, kernel_size=(1, 1, 3), padding=(0, 0, 1), bias=True)
        for scorer in (self.xy_logits, self.xz_logits, self.yz_logits):
            nn.init.zeros_(scorer.weight)
            nn.init.zeros_(scorer.bias)

    def forward(self, value: Tensor) -> BaseTriPlanes:
        xy_weight = torch.softmax(self.xy_logits(value), dim=2)
        xz_weight = torch.softmax(self.xz_logits(value), dim=3)
        yz_weight = torch.softmax(self.yz_logits(value), dim=4)
        return BaseTriPlanes(
            xy=(value * xy_weight).sum(dim=2),
            xz=(value * xz_weight).sum(dim=3),
            yz=(value * yz_weight).sum(dim=4),
        )


@dataclass(frozen=True)
class _ScalePlanes:
    xy: Tensor
    xz: Tensor
    yz: Tensor


class StaticSynthesisHead(nn.Module):
    """Produce graph-preserving 32-channel Z0 planes for one PFGR variant."""

    _SCALE_CHANNELS = {"shallow": 64, "layer1": 64, "deep": 512}

    def __init__(self, config: StaticSynthesisConfig | None = None) -> None:
        super().__init__()
        self.config = StaticSynthesisConfig() if config is None else config
        if not isinstance(self.config, StaticSynthesisConfig):
            raise TypeError("config must be StaticSynthesisConfig")
        variant = self.config.variant
        self.variant = variant
        # B1 and B2 deliberately instantiate identical source-slot modules.
        # B1 feeds explicit zero source slots; B2 feeds ordered observations.
        if variant in ("b1_multiscale_v1", "b2_ordered_multiscale_v1"):
            self.scale_projectors = nn.ModuleDict(
                {
                    "shallow": nn.Conv3d(67, 64, kernel_size=1, bias=True),
                    "layer1": nn.Conv3d(67, 64, kernel_size=1, bias=True),
                    "deep": nn.Conv3d(515, 64, kernel_size=1, bias=True),
                }
            )
            self.scale_collapses = nn.ModuleDict({name: _AxisConditionedCollapse() for name in ("shallow", "layer1", "deep")})
            self.scale_residuals = nn.ModuleDict(
                {name: nn.ModuleList([_Residual2d() for _ in range(2)]) for name in ("shallow", "layer1", "deep")}
            )
        else:
            self.scale_projectors = nn.ModuleDict()
            self.scale_collapses = nn.ModuleDict()
            self.scale_residuals = nn.ModuleDict()

        if variant == "b_light_ordered_v1":
            self.light_projection = nn.Conv2d(67, 64, kernel_size=1, bias=True)
            self.light_residuals = nn.ModuleList([_Residual2d() for _ in range(1)])
        else:
            self.light_projection = None
            self.light_residuals = nn.ModuleList()

        # The existing Gate-C state initializer is a shared 64->32 map applied
        # identically to all three planes.  Reusing its class keeps B0's
        # module semantics and gives every PFGR variant one explicit state
        # initializer owner.
        self.state_initializer = DynamicStateInitializer(input_channels=BASE_CHANNELS, state_channels=STATE_CHANNELS)

    @property
    def final_projection(self) -> nn.Conv2d:
        """Compatibility view of the shared 64->32 state map."""

        return self.state_initializer.shared_projection

    @property
    def source_slots_active(self) -> bool:
        return self.variant in ("b2_ordered_multiscale_v1", "b_light_ordered_v1")

    @property
    def source_slot_channels(self) -> tuple[int, int, int]:
        """Reserved ordered source slots for shallow/Layer1/deep branches."""

        return (3, 3, 3)

    @property
    def inactive_source_slots(self) -> tuple[str, ...]:
        if self.variant == "b2_ordered_multiscale_v1":
            return ()
        if self.variant == "b_light_ordered_v1":
            return ("layer1", "deep")
        return ("shallow", "layer1", "deep")

    @property
    def effective_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def total_parameter_count(self) -> int:
        return self.effective_parameter_count

    def _validate_inputs(
        self,
        features: MedicalNetFeatures,
        source: Tensor,
        base_planes: BaseTriPlanes,
        geometries: MultiScaleFeatureGeometry,
        selected_lattice: FeatureLattice,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(features, MedicalNetFeatures):
            raise TypeError("features must be MedicalNetFeatures from the shared traversal")
        _check_feature("features.shallow", features.shallow, 64)
        _check_feature("features.layer1", features.layer1, 64)
        _check_feature("features.deep", features.deep, 512)
        _check_source(source)
        if not isinstance(base_planes, BaseTriPlanes):
            raise TypeError("base_planes must be BaseTriPlanes")
        if not isinstance(geometries, MultiScaleFeatureGeometry):
            raise TypeError("geometries must be MultiScaleFeatureGeometry")
        if not isinstance(selected_lattice, FeatureLattice):
            raise TypeError("selected_lattice must be FeatureLattice")
        if source.shape[0] != features.shallow.shape[0] or source.shape[0] != base_planes.xy.shape[0]:
            raise ValueError("source, features, and base planes must share batch size")
        return features.shallow, features.layer1, features.deep

    @staticmethod
    def _resample_base(base_planes: BaseTriPlanes, selected: FeatureLattice, target: FeatureLattice) -> BaseTriPlanes:
        return BaseTriPlanes(
            xy=resample_plane_between_lattices(base_planes.xy, selected, target, plane_name="xy"),
            xz=resample_plane_between_lattices(base_planes.xz, selected, target, plane_name="xz"),
            yz=resample_plane_between_lattices(base_planes.yz, selected, target, plane_name="yz"),
        )

    @staticmethod
    def _apply_residuals(planes: BaseTriPlanes, residuals: nn.ModuleList) -> BaseTriPlanes:
        values = []
        for name in ("xy", "xz", "yz"):
            value = getattr(planes, name)
            for block in residuals:
                value = block(value)
            values.append(value)
        return BaseTriPlanes(xy=values[0], xz=values[1], yz=values[2])

    @staticmethod
    def _fuse_scale(
        planes: BaseTriPlanes,
        source_lattice: FeatureLattice,
        target_lattice: FeatureLattice,
    ) -> BaseTriPlanes:
        if source_lattice.feature_shape_dhw == target_lattice.feature_shape_dhw and source_lattice.feature_geometry == target_lattice.feature_geometry:
            return planes
        return BaseTriPlanes(
            xy=resample_plane_between_lattices(planes.xy, source_lattice, target_lattice, plane_name="xy"),
            xz=resample_plane_between_lattices(planes.xz, source_lattice, target_lattice, plane_name="xz"),
            yz=resample_plane_between_lattices(planes.yz, source_lattice, target_lattice, plane_name="yz"),
        )

    def _multiscale(self, features: MedicalNetFeatures, source: Tensor, base: BaseTriPlanes, geometries: MultiScaleFeatureGeometry) -> BaseTriPlanes:
        feature_map = {"shallow": features.shallow, "layer1": features.layer1, "deep": features.deep}
        branch_outputs: dict[str, BaseTriPlanes] = {}
        target = geometries.shallow
        for name in ("shallow", "layer1", "deep"):
            value = feature_map[name]
            aligned_source = sample_source_to_lattice(source, geometries[name])
            if self.variant == "b1_multiscale_v1":
                # Keep the three source slots physically present and explicitly
                # zero in B1 so its parameter count matches B2 exactly.
                aligned_source = torch.zeros_like(aligned_source)
            input_value = torch.cat((value, aligned_source), dim=1)
            projected = self.scale_projectors[name](input_value)
            planes = self.scale_collapses[name](projected)
            if name == "shallow":
                # The existing B plane participates in the shallow residual
                # path, preserving an S0 gradient to its axis scorers.
                planes = BaseTriPlanes(
                    xy=planes.xy + base.xy,
                    xz=planes.xz + base.xz,
                    yz=planes.yz + base.yz,
                )
            planes = self._apply_residuals(planes, self.scale_residuals[name])
            branch_outputs[name] = self._fuse_scale(planes, geometries[name], target)
        # Coarse-to-fine fusion retains true physical centre resampling.  The
        # B plane was added before the shallow residual path above.
        return BaseTriPlanes(
            xy=branch_outputs["shallow"].xy + branch_outputs["layer1"].xy + branch_outputs["deep"].xy,
            xz=branch_outputs["shallow"].xz + branch_outputs["layer1"].xz + branch_outputs["deep"].xz,
            yz=branch_outputs["shallow"].yz + branch_outputs["layer1"].yz + branch_outputs["deep"].yz,
        )

    def _light(self, source: Tensor, base: BaseTriPlanes, geometries: MultiScaleFeatureGeometry) -> BaseTriPlanes:
        if self.light_projection is None:
            raise RuntimeError("B-light projection is not initialized")
        aligned = sample_source_to_lattice(source, geometries.shallow)
        source_xy, source_xz, source_yz = source_plane_means(aligned)
        outputs: list[Tensor] = []
        for base_plane, source_plane in zip((base.xy, base.xz, base.yz), (source_xy, source_xz, source_yz)):
            value = self.light_projection(torch.cat((base_plane, source_plane), dim=1))
            for block in self.light_residuals:
                value = block(value)
            outputs.append(value)
        return BaseTriPlanes(xy=outputs[0], xz=outputs[1], yz=outputs[2])

    def forward(
        self,
        features: MedicalNetFeatures,
        source: Tensor,
        base_planes: BaseTriPlanes,
        geometries: MultiScaleFeatureGeometry,
        *,
        selected_lattice: FeatureLattice | None = None,
    ) -> DynamicTriPlanes:
        """Return graph-preserving Z0 on the shallow feature lattice."""

        selected = geometries.shallow if selected_lattice is None else selected_lattice
        self._validate_inputs(features, source, base_planes, geometries, selected)
        base_shallow = self._resample_base(base_planes, selected, geometries.shallow)
        if self.variant == "b0_legacy_v1":
            fused = base_shallow
        elif self.variant in ("b1_multiscale_v1", "b2_ordered_multiscale_v1"):
            fused = self._multiscale(features, source, base_shallow, geometries)
        elif self.variant == "b_light_ordered_v1":
            fused = self._light(source, base_shallow, geometries)
        else:  # defensive guard against externally mutated frozen config
            raise ValueError(f"unsupported static synthesis variant: {self.variant!r}")
        return DynamicTriPlanes(
            xy=self.state_initializer.shared_projection(fused.xy),
            xz=self.state_initializer.shared_projection(fused.xz),
            yz=self.state_initializer.shared_projection(fused.yz),
        )

    # Explicit name used in context construction and handoff docs.
    initial_planes = forward


PFGRStaticHead = StaticSynthesisHead
B0LegacyStaticHead = StaticSynthesisHead
B1MultiscaleStaticHead = StaticSynthesisHead
B2OrderedMultiscaleStaticHead = StaticSynthesisHead
BLiteOrderedStaticHead = StaticSynthesisHead


__all__ = [
    "B0LegacyStaticHead",
    "B1MultiscaleStaticHead",
    "B2OrderedMultiscaleStaticHead",
    "BLiteOrderedStaticHead",
    "PFGRStaticHead",
    "STATE_CHANNELS",
    "StaticSynthesisHead",
]
