from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from smagm.anchors import AnchorBatch, AnchorGeometryBatch
from smagm.anchors.contracts import anchor_evidence_hash
from smagm.contracts.coordinates import PhysicalPlane
from smagm.gaussians import restore_gauge_fixed_gaussian_batch
from smagm.memory import (
    GaussianMemory,
    GaussianMemoryBank,
    PropagationConfig,
    gaussian_memory_hash,
    initialize_seed_memory,
    propagate_memory,
)
from smagm.renderer import render_plane
from smagm.state import apply_memory_update, build_initial_patient_state, load_patient_state, save_patient_state


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _anchors(*, requires_grad: bool = False) -> AnchorBatch:
    centers = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=requires_grad)
    geometry = AnchorGeometryBatch(
        ("a",), centers, torch.eye(3).unsqueeze(0), torch.tensor([[True, True, True]]),
        torch.tensor([[4.0, 4.0, 4.0]]), torch.ones(1, 1), torch.zeros(1, 1),
        (("obs",),), ((_digest("plane"),),), (_digest("anchor"),),
    )
    evidence = torch.ones(1, 4)
    appearance = torch.tensor([[2.0]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    observability = torch.tensor([[1.0, 0.8, 0.1]])
    digest = anchor_evidence_hash(patient_id="p", geometry=geometry, evidence=evidence, appearance=appearance, appearance_valid=valid, observability=observability)
    return AnchorBatch("p", geometry, evidence, appearance, valid, observability, ("t1",), digest)


def _two_anchors() -> AnchorBatch:
    centers = torch.tensor([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    geometry = AnchorGeometryBatch(
        ("a", "b"), centers, torch.eye(3).unsqueeze(0).repeat(2, 1, 1),
        torch.ones(2, 3, dtype=torch.bool), torch.full((2, 3), 4.0),
        torch.ones(2, 1), torch.zeros(2, 1),
        (("obs-a",), ("obs-b",)),
        ((_digest("plane-a"),), (_digest("plane-b"),)),
        (_digest("anchor-a"), _digest("anchor-b")),
    )
    evidence = torch.ones(2, 4)
    appearance = torch.tensor([[1.0], [2.0]])
    valid = torch.ones(2, 1, dtype=torch.bool)
    observability = torch.tensor([[1.0, 0.8, 0.1], [1.0, 0.9, 0.2]])
    digest = anchor_evidence_hash(
        patient_id="p", geometry=geometry, evidence=evidence, appearance=appearance,
        appearance_valid=valid, observability=observability,
    )
    return AnchorBatch("p", geometry, evidence, appearance, valid, observability, ("t1",), digest)


def _multi_modality_anchor(*, second_value: float) -> AnchorBatch:
    base = _anchors()
    appearance = torch.tensor([[0.2, second_value]])
    valid = torch.ones_like(appearance, dtype=torch.bool)
    digest = anchor_evidence_hash(
        patient_id=base.patient_id, geometry=base.geometry, evidence=base.evidence,
        appearance=appearance, appearance_valid=valid, observability=base.observability,
    )
    return AnchorBatch(
        base.patient_id, base.geometry, base.evidence, appearance, valid,
        base.observability, ("t1", "t2"), digest,
    )


def _bank_with_log_amplitudes(bank: GaussianMemoryBank, values: torch.Tensor) -> GaussianMemoryBank:
    gaussian = bank.gaussians
    restored = restore_gauge_fixed_gaussian_batch(
        centers_ras_mm=gaussian.centers_ras_mm,
        covariance_factor=gaussian.covariance_factor,
        log_support_amplitude=values,
        appearance=gaussian.appearance,
        appearance_valid=gaussian.appearance_valid,
        covariance_epsilon=gaussian.covariance_epsilon,
        primitive_kind=gaussian.primitive_kind,
        primitive_id=gaussian.primitive_id,
        gauge_policy=gaussian.gauge_policy,
        gauge_config_hash=gaussian.gauge_config_hash or "",
    )
    return GaussianMemoryBank(
        bank.kind, restored, bank.anchor_ids, bank.parent_primitive_ids,
        bank.provenance_hashes, bank.observability,
    )


def _state():
    return build_initial_patient_state(
        patient_id="p", manifest_hash=_digest("manifest"), config_hash=_digest("config"),
        context_observation_ids=("obs",), cache_key_hashes=(_digest("cache"),), anchors=_anchors(),
        field_config_hash=_digest("field-config"), field_model_hash=_digest("field-model"),
    )


def test_p0_returns_exact_seed_state_and_no_transaction() -> None:
    anchors = _anchors(); memory = initialize_seed_memory(anchors)
    result, transactions = propagate_memory(
        memory, anchors, config=PropagationConfig(variant="p0"),
        bounds_min_ras_mm=torch.tensor([-5.0, -5.0, -5.0]), bounds_max_ras_mm=torch.tensor([5.0, 5.0, 5.0]),
    )
    assert result is memory
    assert transactions == ()


def test_p1_is_deterministic_bounded_and_uncertainty_monotone() -> None:
    anchors = _anchors(); memory = initialize_seed_memory(anchors)
    config = PropagationConfig(rounds=2, step_mm=1.0, maximum_structural_primitives=8, maximum_volumetric_primitives=8)
    kwargs = dict(memory=memory, anchors=anchors, config=config, bounds_min_ras_mm=torch.tensor([-5.0] * 3), bounds_max_ras_mm=torch.tensor([5.0] * 3))
    first, transactions = propagate_memory(**kwargs)
    second, second_transactions = propagate_memory(**kwargs)
    assert first.memory_hash == second.memory_hash
    assert transactions == second_transactions
    assert first.primitive_count <= 16
    assert torch.all(first.structural.observability.uncertainty[first.structural.gaussians.count // 2 :] >= memory.structural.observability.uncertainty.min())
    assert first.structural.observability.propagation_depth.max() >= 1
    assert all(len(item.transaction_hash) == 64 for item in transactions)
    assert torch.all(first.structural.gaussians.centers_ras_mm.abs() <= 5)


def test_p1_preserves_existing_gauge_fixed_amplitudes_exactly() -> None:
    anchors = _two_anchors()
    seed = initialize_seed_memory(anchors)
    values = torch.tensor([[-0.75], [0.75]])
    structural = _bank_with_log_amplitudes(seed.structural, values)
    volumetric = _bank_with_log_amplitudes(seed.volumetric, values)
    memory = GaussianMemory(
        structural,
        volumetric,
        seed.modality_ids,
        gaussian_memory_hash(structural, volumetric, seed.modality_ids),
    )
    propagated, _ = propagate_memory(
        memory,
        anchors,
        config=PropagationConfig(rounds=1, maximum_structural_primitives=8, maximum_volumetric_primitives=8),
        bounds_min_ras_mm=torch.tensor([-5.0] * 3),
        bounds_max_ras_mm=torch.tensor([5.0] * 3),
    )
    assert propagated.structural.gaussians.count > structural.gaussians.count
    assert torch.equal(
        propagated.structural.gaussians.log_support_amplitude[: structural.gaussians.count],
        values,
    )
    assert torch.equal(
        propagated.volumetric.gaussians.log_support_amplitude[: volumetric.gaussians.count],
        values,
    )
    assert torch.equal(
        propagated.structural.gaussians.log_support_amplitude[structural.gaussians.count :],
        torch.zeros_like(propagated.structural.gaussians.log_support_amplitude[structural.gaussians.count :]),
    )


def test_propagated_state_is_immutable_versioned_and_safe_round_trip(tmp_path) -> None:
    state = _state()
    propagated, _ = propagate_memory(
        state.memory, state.anchors, config=PropagationConfig(rounds=1),
        bounds_min_ras_mm=torch.tensor([-5.0] * 3), bounds_max_ras_mm=torch.tensor([5.0] * 3),
    )
    updated = apply_memory_update(state, propagated)
    assert updated.parent_state_version == state.state_version
    assert updated.state_version != state.state_version
    path = save_patient_state(updated, tmp_path / "state.pt")
    restored = load_patient_state(path)
    assert restored.state_version == updated.state_version
    assert restored.memory.memory_hash == updated.memory.memory_hash
    assert torch.equal(restored.memory.structural.gaussians.centers_ras_mm, updated.memory.structural.gaussians.centers_ras_mm)


def test_state_restore_rejects_tampered_amplitude_gauge(tmp_path) -> None:
    path = save_patient_state(_state(), tmp_path / "state.pt")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["memory"]["structural"]["gaussian"]["log_support_amplitude"] += 1.0
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    with pytest.raises(ValueError, match="mean-centered gauge"):
        load_patient_state(tampered)


def test_p0_p1_tranche_has_no_adaptive_topology_or_anchor_move_api() -> None:
    repository = Path(__file__).resolve().parents[2]
    assert not (repository / "src" / "smagm" / "memory" / "topology.py").exists()
    assert not (repository / "src" / "smagm" / "anchors" / "adaptation.py").exists()


def test_propagated_bank_renders_and_backpropagates_to_patient_geometry() -> None:
    anchors = _anchors(requires_grad=True)
    memory = initialize_seed_memory(anchors)
    propagated, _ = propagate_memory(
        memory, anchors, config=PropagationConfig(rounds=1),
        bounds_min_ras_mm=torch.tensor([-5.0] * 3), bounds_max_ras_mm=torch.tensor([5.0] * 3),
    )
    plane = PhysicalPlane(
        pixel_center_origin_ras_mm=(-2.0, -2.0, 0.0), axis_u_ras=(1.0, 0.0, 0.0), axis_v_ras=(0.0, 1.0, 0.0),
        spacing_uv_mm=(1.0, 1.0), thickness_mm=1.0, shape_hw=(5, 5), signed_normal_ras=(0.0, 0.0, 1.0),
    )
    result = render_plane(propagated.volumetric.gaussians, plane)
    result.intensity[~result.unsupported_mask].sum().backward()
    assert anchors.centers_ras_mm.grad is not None
    assert torch.isfinite(anchors.centers_ras_mm.grad).all()


def test_p1_enforces_per_anchor_children_across_rounds_and_patient_budget() -> None:
    anchors = _anchors(); memory = initialize_seed_memory(anchors)
    propagated, transactions = propagate_memory(
        memory,
        anchors,
        config=PropagationConfig(
            rounds=3,
            children_per_parent_per_round=2,
            maximum_children_per_anchor=1,
            maximum_patient_primitives=3,
            maximum_structural_primitives=8,
            maximum_volumetric_primitives=8,
        ),
        bounds_min_ras_mm=torch.tensor([-5.0] * 3),
        bounds_max_ras_mm=torch.tensor([5.0] * 3),
    )
    assert propagated.primitive_count <= 3
    assert sum(parent is not None for parent in propagated.structural.parent_primitive_ids) <= 1
    assert sum(parent is not None for parent in propagated.volumetric.parent_primitive_ids) <= 1
    assert any(item.rejected_duplicate_or_budget > 0 for item in transactions)


def test_p1_rejects_unsupported_and_uncertain_frontier_without_children() -> None:
    anchors = _anchors(); memory = initialize_seed_memory(anchors)
    unsupported, unsupported_transactions = propagate_memory(
        memory,
        anchors,
        config=PropagationConfig(rounds=1),
        bounds_min_ras_mm=torch.tensor([-5.0] * 3),
        bounds_max_ras_mm=torch.tensor([5.0] * 3),
        supported_anchor_mask=torch.zeros(1, dtype=torch.bool),
    )
    assert unsupported.memory_hash == memory.memory_hash
    assert unsupported_transactions[0].rejected_no_meaningful_gain > 0
    uncertain, uncertain_transactions = propagate_memory(
        memory,
        anchors,
        config=PropagationConfig(rounds=1, maximum_uncertainty=0.05),
        bounds_min_ras_mm=torch.tensor([-5.0] * 3),
        bounds_max_ras_mm=torch.tensor([5.0] * 3),
    )
    assert uncertain.memory_hash == memory.memory_hash
    assert uncertain_transactions[0].rejected_uncertainty > 0


def test_p1_uses_oriented_source_volume_containment_for_oblique_affines() -> None:
    cosine = 2.0 ** -0.5
    centers = torch.tensor([[0.0, 2.0 ** 0.5, 1.0]])
    frame = torch.tensor([[[cosine, -cosine, 0.0], [cosine, cosine, 0.0], [0.0, 0.0, 1.0]]])
    geometry = AnchorGeometryBatch(
        ("oblique-anchor",), centers, frame, torch.ones(1, 3, dtype=torch.bool),
        torch.full((1, 3), 4.0), torch.ones(1, 1), torch.zeros(1, 1),
        (("observation",),), ((_digest("oblique-plane"),),), (_digest("oblique-anchor"),),
    )
    evidence = torch.ones(1, 4)
    appearance = torch.ones(1, 1)
    appearance_valid = torch.ones(1, 1, dtype=torch.bool)
    observability = torch.tensor([[1.0, 0.8, 0.1]])
    digest = anchor_evidence_hash(
        patient_id="p", geometry=geometry, evidence=evidence, appearance=appearance,
        appearance_valid=appearance_valid, observability=observability,
    )
    anchors = AnchorBatch("p", geometry, evidence, appearance, appearance_valid, observability, ("t1",), digest)
    memory = initialize_seed_memory(anchors)
    affine = torch.tensor([
        [cosine, -cosine, 0.0, 0.0],
        [cosine, cosine, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    propagated, transactions = propagate_memory(
        memory,
        anchors,
        config=PropagationConfig(rounds=1, step_mm=1.5),
        bounds_min_ras_mm=torch.tensor([-2.0 ** 0.5, 0.0, 0.0]),
        bounds_max_ras_mm=torch.tensor([2.0 ** 0.5, 2.0 * 2.0 ** 0.5, 2.0]),
        source_affine_ras_from_index=affine,
        source_shape_xyz=(3, 3, 3),
    )
    inverse = torch.linalg.inv(affine)
    centers_h = torch.cat((propagated.structural.gaussians.centers_ras_mm, torch.ones(propagated.structural.gaussians.count, 1)), dim=1)
    indices = (centers_h @ inverse.T)[:, :3]
    assert bool((indices >= -1e-5).all())
    assert bool((indices <= 2.0 + 1e-5).all())
    assert propagated.primitive_count == memory.primitive_count
    assert transactions[0].rejected_out_of_bounds > 0


def test_cross_modality_gate_uses_appearance_agreement_not_slot_presence() -> None:
    agreeing = _multi_modality_anchor(second_value=0.2)
    agreeing_memory = initialize_seed_memory(agreeing)
    propagated, transactions = propagate_memory(
        agreeing_memory,
        agreeing,
        config=PropagationConfig(rounds=1, minimum_cross_modality_agreement=0.9),
        bounds_min_ras_mm=torch.tensor([-5.0] * 3), bounds_max_ras_mm=torch.tensor([5.0] * 3),
    )
    assert propagated.primitive_count > agreeing_memory.primitive_count
    assert transactions[0].rejected_no_meaningful_gain == 0

    disagreeing = _multi_modality_anchor(second_value=0.9)
    disagreeing_memory = initialize_seed_memory(disagreeing)
    rejected, rejected_transactions = propagate_memory(
        disagreeing_memory,
        disagreeing,
        config=PropagationConfig(rounds=1, minimum_cross_modality_agreement=0.9),
        bounds_min_ras_mm=torch.tensor([-5.0] * 3), bounds_max_ras_mm=torch.tensor([5.0] * 3),
    )
    assert rejected.memory_hash == disagreeing_memory.memory_hash
    assert rejected_transactions[0].rejected_no_meaningful_gain > 0
