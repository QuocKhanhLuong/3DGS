"""Minimal teacher-free structural objectives with explicit topology masks."""

from __future__ import annotations

from dataclasses import dataclass

import torch


class EmptyComparisonError(ValueError):
    """Raised when a requested masked comparison has no legal locations."""


@dataclass(frozen=True)
class StructuralLossResult:
    """Scalar loss plus the number of legally compared feature locations."""

    loss: torch.Tensor
    compared_count: int
    component: str

    def __post_init__(self) -> None:
        if self.loss.ndim != 0 or not bool(torch.isfinite(self.loss)):
            raise ValueError("loss result must contain one finite scalar")
        if self.compared_count <= 0:
            raise ValueError("a loss result must contain at least one comparison")


def _expanded_mask(mask: torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(reference, torch.Tensor) or reference.ndim != 4:
        raise ValueError(f"{name} reference must have shape [B, C, H, W]")
    if not isinstance(mask, torch.Tensor) or mask.dtype is not torch.bool:
        raise TypeError(f"{name} must be a bool tensor")
    if mask.shape not in ((reference.shape[0], 1, *reference.shape[-2:]), reference.shape):
        raise ValueError(f"{name} must have shape [B, 1, H, W] or match the feature tensor")
    if mask.device != reference.device:
        raise ValueError(f"{name} must share the feature device")
    return mask.expand_as(reference)


def _pair(reference: torch.Tensor, other: torch.Tensor, mask: torch.Tensor, component: str) -> StructuralLossResult:
    if reference.shape != other.shape or reference.dtype != other.dtype or reference.device != other.device:
        raise ValueError("paired feature tensors must share shape, dtype, and device")
    if not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(other).all()):
        raise ValueError("paired feature tensors must be finite")
    legal = _expanded_mask(mask, reference, "valid_mask")
    count = int(legal.sum().detach().cpu())
    if count == 0:
        raise EmptyComparisonError(f"{component} has no legal masked comparisons")
    value = (reference - other).abs()[legal].mean()
    return StructuralLossResult(value, count, component)


def structural_consistency_loss(
    z_str: torch.Tensor,
    z_str_perturbed: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Compare structural maps under a declared geometry-preserving perturbation."""
    return _pair(z_str, z_str_perturbed, valid_mask, "structural_consistency").loss


def structural_consistency_result(
    z_str: torch.Tensor,
    z_str_perturbed: torch.Tensor,
    valid_mask: torch.Tensor,
) -> StructuralLossResult:
    """Return structural consistency with an explicit comparison diagnostic."""
    return _pair(z_str, z_str_perturbed, valid_mask, "structural_consistency")


def appearance_sensitivity_loss(
    z_app: torch.Tensor,
    z_app_perturbed: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    minimum_change: float = 0.01,
) -> torch.Tensor:
    """Penalize appearance maps that fail to respond to an intensity change.

    This is a margin objective, not an invariance claim: it does not require
    ``Z_app`` to match across intensity perturbations.
    """
    if minimum_change < 0.0 or not torch.isfinite(torch.as_tensor(minimum_change)):
        raise ValueError("minimum_change must be finite and non-negative")
    if z_app.shape != z_app_perturbed.shape:
        raise ValueError("appearance feature tensors must share shape")
    if not bool(torch.isfinite(z_app).all()) or not bool(torch.isfinite(z_app_perturbed).all()):
        raise ValueError("appearance feature tensors must be finite")
    legal = _expanded_mask(valid_mask, z_app, "valid_mask")
    count = int(legal.sum().detach().cpu())
    if count == 0:
        raise EmptyComparisonError("appearance_sensitivity has no legal masked comparisons")
    delta = (z_app - z_app_perturbed).abs()
    return (torch.as_tensor(float(minimum_change), dtype=z_app.dtype, device=z_app.device) - delta[legal]).clamp_min(0.0).mean()


def reliability_regularization_loss(
    reliability: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    target_mean: float = 0.5,
) -> torch.Tensor:
    """Keep legal reliability away from the trivial all-zero solution."""
    if reliability.ndim != 4 or reliability.shape[1] != 1:
        raise ValueError("reliability must have shape [B, 1, H, W]")
    if not 0.0 < target_mean < 1.0:
        raise ValueError("target_mean must lie strictly between zero and one")
    if not bool(torch.isfinite(reliability).all()) or bool((reliability < 0.0).any()) or bool((reliability > 1.0).any()):
        raise ValueError("reliability must be finite and bounded in [0, 1]")
    legal = _expanded_mask(valid_mask, reliability, "valid_mask")
    count = int(legal.sum().detach().cpu())
    if count == 0:
        raise EmptyComparisonError("reliability_regularization has no legal masked comparisons")
    target = reliability.new_tensor(float(target_mean))
    return (reliability[legal] - target).square().mean()


def structural_variance_floor_loss(
    z_str: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    variance_floor: float = 1e-3,
) -> torch.Tensor:
    """Return a positive diagnostic loss for collapsed structural channels."""
    if variance_floor <= 0.0 or not bool(torch.isfinite(torch.as_tensor(variance_floor))):
        raise ValueError("variance_floor must be positive and finite")
    legal = _expanded_mask(valid_mask, z_str, "valid_mask")
    count = int(legal.sum().detach().cpu())
    if count == 0:
        raise EmptyComparisonError("structural_variance_floor has no legal masked comparisons")
    values = z_str.permute(1, 0, 2, 3).masked_select(legal.permute(1, 0, 2, 3)).reshape(z_str.shape[1], -1)
    if values.shape[1] < 2:
        raise EmptyComparisonError("structural_variance_floor requires at least two legal values per channel")
    std = values.std(dim=1, unbiased=False)
    floor = z_str.new_tensor(float(variance_floor))
    return (floor - std).clamp_min(0.0).mean()


def cross_modality_structural_consistency_loss(
    z_str_a: torch.Tensor,
    z_str_b: torch.Tensor,
    valid_overlap_mask: torch.Tensor,
    *,
    registered: bool,
    registration_confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compare only registered physical correspondences; reject unregistered pairs."""
    if not registered:
        raise ValueError("cross-modality structural consistency requires registered observations")
    if registration_confidence is None:
        return structural_consistency_loss(z_str_a, z_str_b, valid_overlap_mask)
    if registration_confidence.shape != valid_overlap_mask.shape or registration_confidence.dtype not in (torch.float32, torch.float64):
        raise ValueError("registration_confidence must match the explicit valid-overlap mask")
    if not bool(torch.isfinite(registration_confidence).all()) or bool((registration_confidence < 0.0).any()) or bool((registration_confidence > 1.0).any()):
        raise ValueError("registration_confidence must be finite and bounded in [0, 1]")
    confidence_mask = valid_overlap_mask & (registration_confidence > 0.0)
    return _pair(z_str_a, z_str_b, confidence_mask, "cross_modality_structural_consistency").loss
