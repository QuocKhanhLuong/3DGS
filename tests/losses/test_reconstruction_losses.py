from __future__ import annotations

import pytest
import torch

from smagm.losses.compose import compose_objective
from smagm.losses.reconstruction import (
    EmptyReconstructionMaskError,
    ReconstructionLossConfig,
    reconstruction_loss,
)
from smagm.renderer import RenderResult


def _prediction(*, requires_grad: bool = True) -> RenderResult:
    intensity = torch.tensor(
        [[1.0, 2.0, float("nan")], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]],
        requires_grad=requires_grad,
    )
    unsupported = torch.zeros((3, 3), dtype=torch.bool)
    unsupported[0, 2] = True
    return RenderResult(
        intensity=intensity,
        support_mass=torch.ones((3, 3)),
        supported_psf_mass=(~unsupported).to(dtype=torch.float32),
        unsupported_mask=unsupported,
    )


def test_loss_uses_exact_target_valid_and_renderer_supported_intersection() -> None:
    prediction = _prediction()
    target = torch.zeros((3, 3))
    target_valid = torch.ones((3, 3), dtype=torch.bool)
    target_valid[2, 2] = False
    result = reconstruction_loss(
        prediction,
        target,
        target_valid,
        config=ReconstructionLossConfig(intensity="mse", gradient_weight=0.2, frequency_weight=0.1),
    )
    assert result.status == "OK"
    assert result.legal_pixel_count == 7
    assert result.target_valid_pixel_count == 8
    assert result.supported_fraction == pytest.approx(7 / 8)
    assert set(result.components) == {"intensity", "gradient", "frequency"}
    result.total.backward()
    assert prediction.intensity.grad is not None
    assert torch.isfinite(prediction.intensity.grad).all()


def test_empty_legal_mask_is_typed_and_never_silently_optimized() -> None:
    prediction = _prediction()
    mask = torch.zeros((3, 3), dtype=torch.bool)
    with pytest.raises(EmptyReconstructionMaskError):
        reconstruction_loss(prediction, torch.zeros((3, 3)), mask)
    skipped = reconstruction_loss(
        prediction,
        torch.zeros((3, 3)),
        mask,
        config=ReconstructionLossConfig(empty_mask_policy="skip"),
    )
    assert skipped.status == "SKIPPED_EMPTY_LEGAL_MASK"
    assert skipped.legal_pixel_count == 0
    with pytest.raises(ValueError, match="skipped"):
        compose_objective(skipped)


def test_objective_composition_requires_explicit_one_to_one_structural_weights() -> None:
    result = reconstruction_loss(_prediction(), torch.zeros((3, 3)), torch.ones((3, 3), dtype=torch.bool))
    structural = torch.tensor(0.25, requires_grad=True)
    objective = compose_objective(
        result,
        structural_components={"variance": structural},
        structural_weights={"variance": 0.5},
    )
    assert "structural/variance" in objective.components
    with pytest.raises(ValueError, match="exactly one"):
        compose_objective(result, structural_components={"variance": structural})
