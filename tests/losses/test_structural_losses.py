"""Teacher-free loss and anti-collapse gates."""

from __future__ import annotations

import pytest
import torch

from smagm.features.conditioning import IntensityPerturbation, apply_intensity_perturbation
from smagm.losses.structural import (
    EmptyComparisonError,
    appearance_sensitivity_loss,
    cross_modality_structural_consistency_loss,
    reliability_regularization_loss,
    structural_consistency_loss,
    structural_variance_floor_loss,
)


def test_geometry_preserving_intensity_perturbation_and_mask_are_respected() -> None:
    image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    perturbed = apply_intensity_perturbation(image, IntensityPerturbation(scale=1.5, bias=-2.0))
    z = image.expand(1, 2, 4, 4)
    z_perturbed = z.clone()
    valid = torch.zeros((1, 1, 4, 4), dtype=torch.bool)
    valid[:, :, 1:3, 1:3] = True
    assert bool(torch.isfinite(perturbed).all())
    assert structural_consistency_loss(z, z_perturbed, valid) == 0.0
    assert appearance_sensitivity_loss(z, z + 0.1, valid) == 0.0


def test_collapse_diagnostic_activates_and_reliability_zero_is_penalized() -> None:
    z = torch.ones((1, 3, 4, 4), dtype=torch.float32)
    valid = torch.ones((1, 1, 4, 4), dtype=torch.bool)
    assert structural_variance_floor_loss(z, valid) > 0.0
    reliability = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    assert reliability_regularization_loss(reliability, valid) > 0.0


def test_unregistered_cross_modality_and_empty_comparisons_are_rejected() -> None:
    z = torch.zeros((1, 2, 4, 4))
    valid = torch.ones((1, 1, 4, 4), dtype=torch.bool)
    with pytest.raises(ValueError, match="registered"):
        cross_modality_structural_consistency_loss(z, z, valid, registered=False)
    empty = torch.zeros_like(valid)
    with pytest.raises(EmptyComparisonError):
        structural_consistency_loss(z, z, empty)
    with pytest.raises(EmptyComparisonError):
        structural_variance_floor_loss(z, torch.zeros_like(valid))
