"""Compact teacher-free T1-B evidence encoders.

The three variants deliberately share one output contract and one geometry
path.  They produce ``[B, C, Hf, Wf]`` maps on a half-pixel grid and never
communicate between support points or batch items.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Literal, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..contracts.coordinates import PhysicalPlane
from .analytic import analytic_feature_bank
from .contracts import EncoderFeatureMaps, FeatureGridToPlaneTransform


EncoderVariant = Literal["e0", "e1", "e2"]


@dataclass(frozen=True)
class EncoderConfig:
    """Auditable T1-B encoder configuration."""

    variant: EncoderVariant = "e2"
    output_stride: int = 1
    structural_channels: int = 16
    appearance_channels: int = 8
    reliability_channels: int = 1
    hidden_channels: int = 24
    analytic_eps: float = 1e-6
    local_radii_mm: tuple[int, int] = (2, 4)

    def __post_init__(self) -> None:
        if self.variant not in ("e0", "e1", "e2"):
            raise ValueError("variant must be one of e0, e1, or e2")
        if self.output_stride not in (1, 2, 4):
            raise ValueError("output_stride must be one of 1, 2, or 4")
        for name in ("structural_channels", "appearance_channels", "hidden_channels"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.reliability_channels != 1:
            raise ValueError("reliability_channels is fixed at exactly one")
        if isinstance(self.analytic_eps, bool) or not isinstance(self.analytic_eps, (int, float)):
            raise ValueError("analytic_eps must be positive and finite")
        if not math.isfinite(float(self.analytic_eps)) or self.analytic_eps <= 0.0:
            raise ValueError("analytic_eps must be positive and finite")
        if tuple(self.local_radii_mm) != (2, 4):
            raise ValueError("the reference analytic contract requires local_radii_mm=(2, 4)")
        object.__setattr__(self, "local_radii_mm", (2, 4))

    def to_dict(self) -> dict[str, object]:
        return {
            "analytic_eps": float(self.analytic_eps),
            "appearance_channels": self.appearance_channels,
            "hidden_channels": self.hidden_channels,
            "local_radii_mm": self.local_radii_mm,
            "output_stride": self.output_stride,
            "reliability_channels": self.reliability_channels,
            "structural_channels": self.structural_channels,
            "variant": self.variant,
        }

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EncoderParameterReport:
    """Parameter and fixed-adapter accounting for an encoder instance."""

    variant: str
    parameter_count: int
    trainable_parameter_count: int
    adapter_operation_count: int
    analytic_channel_count: int
    output_stride: int


def _validate_input(
    image: torch.Tensor,
    planes: PhysicalPlane | Sequence[PhysicalPlane],
    modality_ids: str | Sequence[str],
    valid_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, tuple[PhysicalPlane, ...], tuple[str, ...], torch.Tensor]:
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[1] != 1:
        raise ValueError("image must have shape [B, 1, H, W]")
    if image.dtype not in (torch.float32, torch.float64):
        raise TypeError("image must use float32 or float64")
    if not bool(torch.isfinite(image).all()):
        raise ValueError("image must be finite")
    batch, _, height, width = image.shape
    if isinstance(planes, PhysicalPlane):
        if batch != 1:
            raise ValueError("a single PhysicalPlane may only bind an image batch of one")
        plane_items = (planes,)
    else:
        plane_items = tuple(planes)
        if len(plane_items) != batch:
            raise ValueError("planes must contain exactly one PhysicalPlane per batch item")
    for plane in plane_items:
        if not isinstance(plane, PhysicalPlane) or tuple(plane.shape_hw) != (height, width):
            raise ValueError("every PhysicalPlane must match the common image shape")
    if isinstance(modality_ids, str):
        if batch != 1:
            raise ValueError("a single modality ID may only bind an image batch of one")
        ids = (modality_ids,)
    else:
        ids = tuple(modality_ids)
        if len(ids) != batch:
            raise ValueError("modality_ids must contain exactly one ID per batch item")
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("modality_ids must contain non-empty strings")
    if valid_mask is None:
        mask = torch.ones_like(image, dtype=torch.bool)
    else:
        if valid_mask.shape != image.shape or valid_mask.dtype is not torch.bool or valid_mask.device != image.device:
            raise ValueError("valid_mask must be bool with the same shape and device as image")
        if not bool(valid_mask.flatten(1).any(dim=1).all()):
            raise ValueError("every image requires at least one valid input pixel")
        mask = valid_mask
    return image, plane_items, ids, mask


def _pad_for_stride(value: torch.Tensor, stride: int, *, fill: float = 0.0) -> torch.Tensor:
    if stride == 1:
        return value
    height, width = value.shape[-2:]
    pad_v = (-height) % stride
    pad_u = (-width) % stride
    return F.pad(value, (0, pad_u, 0, pad_v), value=fill)


def _downsample_valid_mask(mask: torch.Tensor, stride: int) -> torch.Tensor:
    """Downsample validity by exact all-pixels pooling after right/bottom padding."""
    padded = _pad_for_stride(mask.to(dtype=torch.float32), stride)
    if stride == 1:
        pooled = padded
    else:
        pooled = F.avg_pool2d(padded, kernel_size=stride, stride=stride)
    return (pooled == 1.0).to(dtype=torch.bool)


def _downsample_features(value: torch.Tensor, stride: int) -> torch.Tensor:
    padded = _pad_for_stride(value, stride)
    if stride == 1:
        return padded
    return F.avg_pool2d(padded, kernel_size=stride, stride=stride)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(1, channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = F.silu(self.norm1(self.conv1(value)))
        value = self.norm2(self.conv2(value))
        return F.silu(value + residual)


class _MicroCNN(nn.Module):
    """Small per-pixel CNN with explicit powers-of-two downsampling only."""

    def __init__(self, input_channels: int, config: EncoderConfig) -> None:
        super().__init__()
        hidden = config.hidden_channels
        self.output_stride = config.output_stride
        self.stem = nn.Conv2d(input_channels, hidden, kernel_size=3, padding=1)
        self.stem_norm = nn.GroupNorm(1, hidden)
        self.blocks = nn.ModuleList([_ResidualBlock(hidden), _ResidualBlock(hidden)])
        self.structural_head = nn.Conv2d(hidden, config.structural_channels, kernel_size=1)
        self.appearance_head = nn.Conv2d(hidden, config.appearance_channels, kernel_size=1)
        self.reliability_head = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = F.silu(self.stem_norm(self.stem(value)))
        for block in self.blocks:
            value = block(value)
        if self.output_stride in (2, 4):
            value = _pad_for_stride(value, 2)
            value = F.avg_pool2d(value, kernel_size=2, stride=2)
        if self.output_stride == 4:
            value = _pad_for_stride(value, 2)
            value = F.avg_pool2d(value, kernel_size=2, stride=2)
        return self.structural_head(value), self.appearance_head(value), self.reliability_head(value)


class EvidenceEncoder(nn.Module):
    """Teacher-free E0/E1/E2 encoder with a common ``EncoderFeatureMaps`` output."""

    _E0_STRUCTURAL = (0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1)
    _E0_APPEARANCE = (0, 5, 6, 0, 1, 2, 3, 4)

    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or EncoderConfig()
        input_channels = 1 if self.config.variant == "e1" else 7
        self.micro_cnn = None if self.config.variant == "e0" else _MicroCNN(input_channels, self.config)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @property
    def parameter_report(self) -> EncoderParameterReport:
        adapter_ops = 0 if self.config.variant != "e0" else len(self._E0_STRUCTURAL) + len(self._E0_APPEARANCE) + 1
        return EncoderParameterReport(
            variant=self.config.variant,
            parameter_count=self.parameter_count,
            trainable_parameter_count=self.trainable_parameter_count,
            adapter_operation_count=adapter_ops,
            analytic_channel_count=8,
            output_stride=self.config.output_stride,
        )

    def state_hash(self) -> str:
        """Hash learned state for cache binding; E0 has a stable empty state."""
        digest = hashlib.sha256()
        for name, value in self.state_dict().items():
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _e0(self, analytic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        structural = analytic[:, self._E0_STRUCTURAL]
        appearance = analytic[:, self._E0_APPEARANCE]
        reliability = analytic[:, 7:8].clamp(0.0, 1.0)
        return structural, appearance, reliability

    def forward(
        self,
        image: torch.Tensor,
        planes: PhysicalPlane | Sequence[PhysicalPlane],
        modality_ids: str | Sequence[str],
        valid_mask: torch.Tensor | None = None,
    ) -> EncoderFeatureMaps:
        image, plane_items, ids, input_mask = _validate_input(image, planes, modality_ids, valid_mask)
        spacing = tuple(plane.spacing_uv_mm for plane in plane_items)
        analytic = analytic_feature_bank(
            image,
            valid_mask=input_mask,
            local_radii=self.config.local_radii_mm,
            spacing_uv_mm=spacing,
            eps=self.config.analytic_eps,
        )
        if self.config.variant == "e0":
            structural, appearance, reliability = self._e0(analytic.tensor)
            source_mask = analytic.valid_mask
            structural = _downsample_features(structural, self.config.output_stride)
            appearance = _downsample_features(appearance, self.config.output_stride)
            reliability = _downsample_features(reliability, self.config.output_stride)
        else:
            if self.config.variant == "e1":
                cnn_input = torch.where(input_mask, image, torch.zeros_like(image))
                source_mask = input_mask
            else:
                cnn_input = analytic.tensor[:, :7]
                source_mask = analytic.valid_mask
            assert self.micro_cnn is not None
            structural, appearance, reliability_logits = self.micro_cnn(cnn_input)
            reliability = torch.sigmoid(reliability_logits)
        feature_mask = _downsample_valid_mask(source_mask, self.config.output_stride)
        structural = torch.where(feature_mask, structural, torch.zeros_like(structural))
        appearance = torch.where(feature_mask, appearance, torch.zeros_like(appearance))
        reliability = torch.where(feature_mask, reliability.clamp(0.0, 1.0), torch.zeros_like(reliability))
        expected_shape = tuple((length + self.config.output_stride - 1) // self.config.output_stride for length in image.shape[-2:])
        if tuple(structural.shape[-2:]) != expected_shape or tuple(feature_mask.shape[-2:]) != expected_shape:
            raise RuntimeError("encoder output shape does not match the locked half-pixel feature grid")
        transforms = tuple(
            FeatureGridToPlaneTransform(
                input_shape_hw=tuple(image.shape[-2:]),
                feature_shape_hw=expected_shape,
                stride_vu=(self.config.output_stride, self.config.output_stride),
                input_plane=plane,
            )
            for plane in plane_items
        )
        return EncoderFeatureMaps(
            structural=structural,
            appearance=appearance,
            reliability=reliability,
            grid_to_planes=transforms,
            modality_ids=ids,
            valid_feature_mask=feature_mask,
        )

    def encode(
        self,
        image: torch.Tensor,
        planes: PhysicalPlane | Sequence[PhysicalPlane],
        modality_ids: str | Sequence[str],
        valid_mask: torch.Tensor | None = None,
    ) -> EncoderFeatureMaps:
        """Named alias for callers that prefer an explicit encoder operation."""
        return self.forward(image, planes, modality_ids, valid_mask)

