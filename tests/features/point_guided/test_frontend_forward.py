"""End-to-end CPU smoke tests for the locked frontend boundary."""

from __future__ import annotations

import pytest
import torch

from smagm.features.point_guided import PointGuidedConfig, PointGuidedMRIModel
from smagm.features.point_guided.sampling import ras_mm_in_bounds


def _model() -> PointGuidedMRIModel:
    return PointGuidedMRIModel(
        PointGuidedConfig(
            num_semantic_classes=3,
            num_points=4,
            point_candidate_multiplier=3,
        )
    )


def test_frontend_returns_only_the_locked_point_field_and_sparse_pou() -> None:
    model = _model().eval()
    x = torch.randn(1, 3, 9, 9, 9)
    brain_mask = torch.ones(1, 1, 9, 9, 9, dtype=torch.bool)

    with torch.no_grad():
        output = model.forward_frontend(x, brain_mask, spacing_mm=(1.0, 1.5, 2.0))

    assert output.S_coarse.shape == (1, 3, 9, 9, 9)
    assert output.initial_points.shape == output.refined_points.shape == (1, 4, 3)
    assert output.displacement.shape == (1, 4, 3)
    assert output.point_semantic.shape == (1, 4, 3)
    assert torch.allclose(output.S_coarse.sum(dim=1), torch.ones_like(output.S_coarse[:, 0]), atol=1e-5)
    assert torch.linalg.vector_norm(output.displacement, dim=-1).amax() <= 2.0 + 1e-6
    assert bool(ras_mm_in_bounds(output.refined_points, output.geometry).all())
    assert output.sparse_pou.normalized_weight.ndim == 1
    assert output.sparse_pou.normalized_weight.numel() > 0


def test_downstream_loss_reaches_the_offset_predictor_through_refined_points() -> None:
    model = _model().train()
    x = torch.randn(1, 3, 9, 9, 9)
    output = model.forward_frontend(x, spacing_mm=(1.0, 1.0, 1.0))
    loss = (
        output.refined_points.square().mean()
        + output.point_semantic.square().mean()
        + output.sparse_pou.raw_affinity.mean()
    )
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.point_refiner.offset_predictor.parameters()
        if parameter.requires_grad
    ]
    assert gradients and all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients)


def test_full_forward_refuses_to_synthesize_an_unresolved_t1ce_volume() -> None:
    with pytest.raises(NotImplementedError, match="Full T1ce synthesis is unresolved"):
        _model()(torch.randn(1, 3, 9, 9, 9))
