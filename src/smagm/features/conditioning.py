"""Declared intensity perturbations for teacher-free structural comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class IntensityPerturbation:
    """Pointwise affine perturbation that preserves the pixel geometry."""

    scale: float = 1.0
    bias: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.scale)) or not math.isfinite(float(self.bias)):
            raise ValueError("intensity perturbation parameters must be finite")
        if self.scale <= 0.0:
            raise ValueError("intensity perturbation scale must be positive")


def apply_intensity_perturbation(image: torch.Tensor, perturbation: IntensityPerturbation) -> torch.Tensor:
    """Apply a differentiable geometry-preserving affine intensity change."""
    if not isinstance(image, torch.Tensor) or image.dtype not in (torch.float32, torch.float64):
        raise TypeError("image must be a float32 or float64 tensor")
    if not isinstance(perturbation, IntensityPerturbation):
        raise TypeError("perturbation must be an IntensityPerturbation")
    result = image * image.new_tensor(perturbation.scale) + image.new_tensor(perturbation.bias)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("intensity perturbation produced non-finite values")
    return result

