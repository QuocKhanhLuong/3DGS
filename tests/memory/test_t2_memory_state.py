from __future__ import annotations

import hashlib

import pytest
import torch

from smagm.anchors import AnchorBatch, AnchorGeometryBatch
from smagm.anchors.contracts import anchor_evidence_hash
from smagm.memory import GaussianMemory, GaussianMemoryBank, PrimitiveKind, PrimitiveObservability, initialize_seed_memory
from smagm.state import build_initial_patient_state


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _anchors() -> AnchorBatch:
    geometry = AnchorGeometryBatch(
        ("a",), torch.zeros(1, 3), torch.eye(3).unsqueeze(0), torch.tensor([[True, True, False]]),
        torch.tensor([[2.0, 3.0, 4.0]]), torch.ones(1, 1), torch.zeros(1, 1),
        (("obs",),), ((_digest("plane"),),), (_digest("anchor"),),
    )
    evidence = torch.ones(1, 4)
    appearance = torch.tensor([[2.0, 0.0]])
    valid = torch.tensor([[True, False]])
    observability = torch.tensor([[1.0, 0.8, 0.1]])
    digest = anchor_evidence_hash(patient_id="p", geometry=geometry, evidence=evidence, appearance=appearance, appearance_valid=valid, observability=observability)
    return AnchorBatch("p", geometry, evidence, appearance, valid, observability, ("t1", "t2"), digest)


def test_seed_memory_has_distinct_banks_spd_covariance_and_missing_modality_mask() -> None:
    memory = initialize_seed_memory(_anchors())
    assert memory.structural.kind is PrimitiveKind.STRUCTURAL
    assert memory.volumetric.kind is PrimitiveKind.VOLUMETRIC
    assert memory.primitive_count == 2
    assert not memory.structural.gaussians.appearance_valid[0, 1]
    assert torch.linalg.eigvalsh(memory.structural.gaussians.covariance()).min() > 0
    assert torch.linalg.eigvalsh(memory.volumetric.gaussians.covariance()).min() > 0
    structural_diagonal = torch.diagonal(memory.structural.gaussians.covariance()[0])
    assert structural_diagonal[2] < structural_diagonal[0]


def test_initial_patient_state_binds_context_anchor_field_and_memory_without_target() -> None:
    state = build_initial_patient_state(
        patient_id="p", manifest_hash=_digest("manifest"), config_hash=_digest("config"),
        context_observation_ids=("obs",), cache_key_hashes=(_digest("cache"),), anchors=_anchors(),
        field_config_hash=_digest("field-config"), field_model_hash=_digest("field-model"),
    )
    assert state.update_round == 0
    assert state.parent_state_version is None
    assert len(state.state_version) == 64
    assert "target" not in state.__dict__


def test_anchor_and_memory_hashes_bind_geometry_and_observability() -> None:
    anchors = _anchors()
    with pytest.raises(ValueError, match="evidence_hash"):
        AnchorBatch(
            anchors.patient_id,
            AnchorGeometryBatch(
                anchors.geometry.anchor_ids,
                anchors.geometry.centers_ras_mm + torch.tensor([[1.0, 0.0, 0.0]]),
                anchors.geometry.frame_axes_ras,
                anchors.geometry.frame_validity,
                anchors.geometry.support_scales_mm,
                anchors.geometry.geometry_confidence,
                anchors.geometry.disagreement,
                anchors.geometry.contributing_observation_ids,
                anchors.geometry.contributing_plane_hashes,
                anchors.geometry.provenance_hashes,
            ),
            anchors.evidence, anchors.appearance, anchors.appearance_valid,
            anchors.observability, anchors.modality_ids, anchors.evidence_hash,
        )
    memory = initialize_seed_memory(anchors)
    changed_observability = PrimitiveObservability(
        evidence_count=memory.structural.observability.evidence_count + 1.0,
        coverage=memory.structural.observability.coverage,
        disagreement=memory.structural.observability.disagreement,
        uncertainty=memory.structural.observability.uncertainty,
        propagation_depth=memory.structural.observability.propagation_depth,
        update_round=memory.structural.observability.update_round,
    )
    changed_structural = GaussianMemoryBank(
        memory.structural.kind, memory.structural.gaussians,
        memory.structural.anchor_ids, memory.structural.parent_primitive_ids,
        memory.structural.provenance_hashes, changed_observability,
    )
    with pytest.raises(ValueError, match="memory_hash"):
        GaussianMemory(
            changed_structural, memory.volumetric, memory.modality_ids, memory.memory_hash,
        )
