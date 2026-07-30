"""Differentiable analytic evidence channels for the teacher-free T1 baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


ANALYTIC_CHANNEL_NAMES = (
    "normalized_intensity",
    "gradient_u",
    "gradient_v",
    "gradient_magnitude",
    "laplacian",
    "local_contrast_r2",
    "local_contrast_r4",
    "valid_mask",
)


@dataclass(frozen=True)
class AnalyticFeatureOutput:
    """Analytic tensor and explicit channel metadata."""

    tensor: torch.Tensor  # [B, 8, H, W]
    valid_mask: torch.Tensor  # [B, 1, H, W], bool
    channel_names: tuple[str, ...] = ANALYTIC_CHANNEL_NAMES

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, torch.Tensor) or self.tensor.ndim != 4:
            raise ValueError("tensor must have shape [B, C, H, W]")
        if self.tensor.shape[1] != len(self.channel_names):
            raise ValueError("tensor channels and channel_names disagree")
        if self.valid_mask.shape != (self.tensor.shape[0], 1, *self.tensor.shape[-2:]):
            raise ValueError("valid_mask must have shape [B, 1, H, W]")
        if self.valid_mask.dtype is not torch.bool:
            raise TypeError("valid_mask must be bool")
        if not bool(torch.isfinite(self.tensor).all()):
            raise ValueError("analytic tensor must be finite")


def _validate_image(image: torch.Tensor, valid_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[1] != 1:
        raise ValueError("image must have shape [B, 1, H, W]")
    if image.dtype not in (torch.float32, torch.float64):
        raise TypeError("image must use float32 or float64")
    if image.shape[0] <= 0 or image.shape[-2] <= 0 or image.shape[-1] <= 0:
        raise ValueError("image dimensions must be positive")
    if not bool(torch.isfinite(image).all()):
        raise ValueError("image must be finite")
    if valid_mask is None:
        mask = torch.ones_like(image, dtype=torch.bool)
    else:
        if not isinstance(valid_mask, torch.Tensor) or valid_mask.shape != image.shape:
            raise ValueError("valid_mask must share image shape")
        if valid_mask.dtype is not torch.bool or valid_mask.device != image.device:
            raise ValueError("valid_mask must be bool and share image device")
        if not bool(valid_mask.flatten(1).any(dim=1).all()):
            raise ValueError("every image requires at least one valid pixel")
        mask = valid_mask
    return image, mask


def _masked_standardize(image: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    weight = mask.to(dtype=image.dtype)
    count = weight.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    mean = (image * weight).sum(dim=(-2, -1), keepdim=True) / count
    variance = ((image - mean).square() * weight).sum(dim=(-2, -1), keepdim=True) / count
    scale = variance.clamp_min(eps * eps).sqrt()
    normalized = (image - mean) / scale
    return torch.where(mask, normalized, torch.zeros_like(normalized))


def _replicate_conv2d(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    kernel = kernel.to(dtype=image.dtype, device=image.device).reshape(1, 1, *kernel.shape)
    pad_h = kernel.shape[-2] // 2
    pad_w = kernel.shape[-1] // 2
    padded = F.pad(image, (pad_w, pad_w, pad_h, pad_h), mode="replicate")
    return F.conv2d(padded, kernel)


def _local_contrast(image: torch.Tensor, radius: int) -> torch.Tensor:
    kernel_size = 2 * radius + 1
    padded = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    local_mean = F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)
    return image - local_mean


def analytic_feature_bank(
    image: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    local_radii: Sequence[int] = (2, 4),
    eps: float = 1e-6,
) -> AnalyticFeatureOutput:
    """Build the fixed T1-A structural evidence bank.

    The operation is differentiable with respect to ``image``.  Replicate
    padding makes constant images yield zero differential channels at the
    boundary and keeps the pixel-centre grid unchanged.
    """

    image, mask = _validate_image(image, valid_mask)
    radii = tuple(local_radii)
    if radii != (2, 4):
        raise ValueError("the reference channel contract currently requires local_radii=(2, 4)")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or eps <= 0:
        raise ValueError("eps must be positive")
    normalized = _masked_standardize(image, mask, float(eps))
    gradient_u = _replicate_conv2d(
        normalized,
        torch.tensor(((0.0, 0.0, 0.0), (-0.5, 0.0, 0.5), (0.0, 0.0, 0.0))),
    )
    gradient_v = _replicate_conv2d(
        normalized,
        torch.tensor(((0.0, -0.5, 0.0), (0.0, 0.0, 0.0), (0.0, 0.5, 0.0))),
    )
    gradient_magnitude = torch.sqrt(gradient_u.square() + gradient_v.square() + float(eps) ** 2)
    laplacian = _replicate_conv2d(
        normalized,
        torch.tensor(((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0))),
    )
    contrast_r2 = _local_contrast(normalized, 2)
    contrast_r4 = _local_contrast(normalized, 4)
    weight = mask.to(dtype=image.dtype)
    channels = (
        normalized,
        gradient_u,
        gradient_v,
        gradient_magnitude,
        laplacian,
        contrast_r2,
        contrast_r4,
        weight,
    )
    tensor = torch.cat(tuple(torch.where(mask, value, torch.zeros_like(value)) for value in channels), dim=1)
    return AnalyticFeatureOutput(tensor=tensor, valid_mask=mask)
