"""Focused tests for training-only coarse semantic supervision."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from smagm.features.point_guided.semantic_supervision import (
    SemanticGroundingLossResult,
    build_coarse_semantic_target,
    compute_semantic_grounding_loss,
)


def test_build_coarse_semantic_target_maps_brats_labels_and_ignores_outside_brain() -> None:
    segmentation = torch.tensor(
        [[[[0, 2, 1], [4, 0, 2]]]],
        dtype=torch.long,
    )
    brain_mask = torch.tensor(
        [[[[True, True, True], [True, False, False]]]],
        dtype=torch.bool,
    )

    target = build_coarse_semantic_target(segmentation, brain_mask)

    expected = torch.tensor([[[[0, 1, 2], [2, -100, -100]]]], dtype=torch.long)
    assert target.shape == (1, 1, 2, 3)
    assert torch.equal(target, expected)


def test_build_coarse_semantic_target_accepts_singleton_channel_inputs() -> None:
    segmentation = torch.tensor([[[[0, 2], [1, 4]]]], dtype=torch.long).unsqueeze(1)
    brain_mask = torch.ones_like(segmentation, dtype=torch.bool)

    target = build_coarse_semantic_target(segmentation, brain_mask)

    assert target.shape == (1, 1, 2, 2)
    assert target.tolist() == [[[[0, 1], [2, 2]]]]


def test_semantic_grounding_loss_matches_weighted_cross_entropy_and_exposes_three_dice_values() -> None:
    target = torch.tensor([[[[0, 1, 2, -100]]]], dtype=torch.long)
    logits = torch.tensor(
        [
            [
                [[[4.0, 0.0, 0.0, 9.0]]],
                [[[0.0, 5.0, 0.0, -9.0]]],
                [[[0.0, 0.0, 6.0, -8.0]]],
            ]
        ]
    )
    weights = torch.tensor([1.0, 2.0, 3.0])

    result = compute_semantic_grounding_loss(
        logits,
        target,
        class_weights=weights,
    )

    expected_loss = F.cross_entropy(logits, target, weight=weights, ignore_index=-100)
    assert isinstance(result, SemanticGroundingLossResult)
    torch.testing.assert_close(result.loss, expected_loss)
    torch.testing.assert_close(result.total, result.loss)
    torch.testing.assert_close(result.dice, torch.ones(3))
    torch.testing.assert_close(result.dice_per_class, torch.ones(3))
    torch.testing.assert_close(result.mean_dice, torch.ones(()))
    assert result.valid_voxel_count == 3
    assert set(result.metrics) == {"dice_class_0", "dice_class_1", "dice_class_2", "mean_dice"}


def test_semantic_grounding_loss_is_differentiable_but_dice_is_a_metric() -> None:
    torch.manual_seed(23)
    logits = torch.randn(1, 3, 2, 2, 2, requires_grad=True)
    target = torch.tensor(
        [[[[0, 1], [2, -100]], [[1, 2], [0, 1]]]],
        dtype=torch.long,
    )

    result = compute_semantic_grounding_loss(logits, target)
    result.loss.backward()

    assert logits.grad is not None
    assert bool(torch.isfinite(result.loss))
    assert bool(torch.isfinite(logits.grad).all())
    assert not result.dice.requires_grad


def test_all_ignored_target_has_zero_loss_and_finite_zero_dice() -> None:
    logits = torch.zeros(1, 3, 2, 2, 2, requires_grad=True)
    target = torch.full((1, 2, 2, 2), -100, dtype=torch.long)

    result = compute_semantic_grounding_loss(logits, target)

    assert result.valid_voxel_count == 0
    torch.testing.assert_close(result.loss, logits.sum() * 0.0)
    torch.testing.assert_close(result.dice, torch.zeros(3))
    torch.testing.assert_close(result.mean_dice, torch.zeros(()))
    result.loss.backward()
    assert logits.grad is not None
    assert bool(torch.equal(logits.grad, torch.zeros_like(logits)))


@pytest.mark.parametrize(
    ("segmentation", "brain_mask", "match"),
    (
        (
            torch.tensor([[[[3]]]], dtype=torch.long),
            torch.ones(1, 1, 1, 1, dtype=torch.bool),
            "drawn from",
        ),
        (
            torch.tensor([[[[float("nan")]]]]),
            torch.ones(1, 1, 1, 1, dtype=torch.bool),
            "finite",
        ),
        (
            torch.tensor([[[[0]]]], dtype=torch.long),
            torch.ones(1, 1, 1, 1),
            "boolean",
        ),
        (
            torch.tensor([[[[0]]]], dtype=torch.long),
            torch.ones(1, 1, 1, 2, dtype=torch.bool),
            "matching",
        ),
    ),
)
def test_build_coarse_semantic_target_validates_labels_mask_and_shapes(
    segmentation: torch.Tensor,
    brain_mask: torch.Tensor,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        build_coarse_semantic_target(segmentation, brain_mask)


@pytest.mark.parametrize(
    "ignore_index",
    (True, 0, 1, 2, 2**63, -2**63 - 1),
)
def test_semantic_supervision_rejects_invalid_ignore_index(ignore_index: object) -> None:
    segmentation = torch.zeros(1, 1, 1, 1, dtype=torch.long)
    mask = torch.ones_like(segmentation, dtype=torch.bool)

    with pytest.raises(ValueError, match="ignore_index"):
        build_coarse_semantic_target(segmentation, mask, ignore_index=ignore_index)  # type: ignore[arg-type]


def test_semantic_grounding_loss_rejects_invalid_logits_target_and_class_weights() -> None:
    logits = torch.zeros(1, 3, 1, 1, 1)
    target = torch.zeros(1, 1, 1, 1, dtype=torch.long)

    with pytest.raises(ValueError, match=r"\[B,3,D,H,W\]"):
        compute_semantic_grounding_loss(torch.zeros(1, 2, 1, 1, 1), target)
    with pytest.raises(ValueError, match="finite"):
        compute_semantic_grounding_loss(torch.full_like(logits, float("nan")), target)
    with pytest.raises(ValueError, match="0, 1, 2, or ignore_index"):
        compute_semantic_grounding_loss(logits, torch.full_like(target, 3))
    with pytest.raises(ValueError, match=r"shape \[3\]"):
        compute_semantic_grounding_loss(logits, target, class_weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="non-negative"):
        compute_semantic_grounding_loss(logits, target, class_weights=[1.0, -1.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        compute_semantic_grounding_loss(logits, target, class_weights=[1.0, float("nan"), 1.0])


def test_semantic_supervision_does_not_enter_model_inference_or_gate_e_modules() -> None:
    package_root = Path(__file__).resolve().parents[3] / "src" / "smagm" / "features" / "point_guided"
    model_source = (package_root / "model.py").read_text(encoding="utf-8")
    objective_source = (package_root / "training_objective.py").read_text(encoding="utf-8")
    assert "semantic_supervision" not in model_source
    assert "semantic_supervision" not in objective_source
