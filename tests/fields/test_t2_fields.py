from __future__ import annotations

import hashlib

import torch

from smagm.anchors import AnchorBatch, AnchorGeometryBatch
from smagm.anchors.contracts import anchor_evidence_hash
from smagm.fields import SharedStructuralField, StructuralFieldConfig, blend_local_fields, query_structural_field


def _anchors(order: torch.Tensor | None = None) -> AnchorBatch:
    centers = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
    frames = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)
    evidence = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    appearance = torch.ones(2, 1, dtype=torch.float64)
    valid = torch.ones(2, 1, dtype=torch.bool)
    observability = torch.ones(2, 3, dtype=torch.float64)
    ids = ("a", "b")
    hashes = tuple(hashlib.sha256(value.encode()).hexdigest() for value in ids)
    if order is not None:
        chosen = tuple(int(i) for i in order.tolist())
        centers, frames, evidence, appearance, valid, observability = (v[order] for v in (centers, frames, evidence, appearance, valid, observability))
        ids = tuple(ids[i] for i in chosen); hashes = tuple(hashes[i] for i in chosen)
    geometry = AnchorGeometryBatch(
        ids, centers, frames, torch.ones(2, 3, dtype=torch.bool), torch.full((2, 3), 3.0, dtype=torch.float64),
        torch.ones(2, 1, dtype=torch.float64), torch.zeros(2, 1, dtype=torch.float64),
        (("o",), ("o",)), (("p",), ("p",)), hashes,
    )
    digest = anchor_evidence_hash(patient_id="p", geometry=geometry, evidence=evidence, appearance=appearance, appearance_valid=valid, observability=observability)
    return AnchorBatch("p", geometry, evidence, appearance, valid, observability, ("t1",), digest)


def test_shared_field_is_permutation_stable_supported_and_differentiable() -> None:
    torch.manual_seed(3)
    field = SharedStructuralField(StructuralFieldConfig(evidence_dim=2, hidden_width=16)).double()
    points = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    first = query_structural_field(field, _anchors(), points)
    second = query_structural_field(field, _anchors(torch.tensor([1, 0])), points)
    assert first.supported.item()
    assert torch.allclose(first.value, second.value, atol=1e-12)
    first.value.sum().backward()
    assert points.grad is not None and torch.isfinite(points.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in field.parameters())


def test_shared_field_reports_unsupported_without_confident_fill() -> None:
    field = SharedStructuralField(StructuralFieldConfig(evidence_dim=2, hidden_width=16)).double()
    output = query_structural_field(field, _anchors(), torch.tensor([[100.0, 0.0, 0.0]], dtype=torch.float64))
    assert not output.supported.item()
    assert torch.isnan(output.value).all()


def test_zero_overlap_disagreement_has_finite_backward() -> None:
    local_values = torch.ones(1, 1, 1, requires_grad=True)
    local_coordinates = torch.zeros(1, 1, 3)
    output = blend_local_fields(local_values, local_coordinates, torch.ones(1, 1, dtype=torch.bool))
    assert output.disagreement.item() == 0.0
    (output.value.sum() + output.disagreement.sum()).backward()
    assert local_values.grad is not None and torch.isfinite(local_values.grad).all()
