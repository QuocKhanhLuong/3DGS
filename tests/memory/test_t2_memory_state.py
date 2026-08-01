from __future__ import annotations

import hashlib

import torch

from smagm.anchors import AnchorBatch, AnchorGeometryBatch
from smagm.anchors.contracts import anchor_evidence_hash
from smagm.memory import PrimitiveKind, initialize_seed_memory
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
