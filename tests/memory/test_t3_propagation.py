from __future__ import annotations

import hashlib
import inspect
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
    validate_seed_and_reserve_budgets,
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


def _rotated_anchors(anchors: AnchorBatch, rotation_ras: torch.Tensor) -> AnchorBatch:
    """Apply one proper RAS rotation to all physical anchor-frame quantities."""

    geometry = anchors.geometry
    rotated_geometry = AnchorGeometryBatch(
        geometry.anchor_ids,
        anchors.centers_ras_mm @ rotation_ras.T,
        rotation_ras.unsqueeze(0) @ anchors.frame_axes_ras,
        geometry.frame_validity,
        geometry.support_scales_mm,
        geometry.geometry_confidence,
        geometry.disagreement,
        geometry.contributing_observation_ids,
        geometry.contributing_plane_hashes,
        geometry.provenance_hashes,
    )
    digest = anchor_evidence_hash(
        patient_id=anchors.patient_id,
        geometry=rotated_geometry,
        evidence=anchors.evidence,
        appearance=anchors.appearance,
        appearance_valid=anchors.appearance_valid,
        observability=anchors.observability,
    )
    return AnchorBatch(
        anchors.patient_id,
        rotated_geometry,
        anchors.evidence,
        anchors.appearance,
        anchors.appearance_valid,
        anchors.observability,
        anchors.modality_ids,
        digest,
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
    assert any(item.rejected_budget > 0 for item in transactions)
    assert all(not hasattr(item, "rejected_duplicate_or_budget") for item in transactions)


def test_p1_reports_exhaustive_separate_proposal_rejection_counters() -> None:
    anchors = _anchors()
    memory = initialize_seed_memory(anchors)
    propagated, transactions = propagate_memory(
        memory,
        anchors,
        config=PropagationConfig(
            rounds=1,
            children_per_parent_per_round=3,
            maximum_children_per_anchor=8,
            maximum_structural_primitives=8,
            maximum_volumetric_primitives=8,
        ),
        bounds_min_ras_mm=torch.tensor([-5.0] * 3),
        bounds_max_ras_mm=torch.tensor([5.0] * 3),
    )
    transaction = transactions[0]
    assert propagated.primitive_count == memory.primitive_count + transaction.accepted_count
    assert transaction.proposal_count == 6
    assert transaction.accepted_count == len(transaction.accepted_primitive_ids)
    assert transaction.rejected_duplicate == 1
    assert transaction.rejected_budget == 0
    assert transaction.proposal_count == transaction.accepted_count + sum((
        transaction.rejected_out_of_bounds,
        transaction.rejected_unsupported,
        transaction.rejected_duplicate,
        transaction.rejected_budget,
        transaction.rejected_uncertainty,
        transaction.rejected_invalid,
        transaction.rejected_no_meaningful_gain,
    ))
    assert not hasattr(transaction, "rejected_duplicate_or_budget")


def test_p1_preflights_seed_capacities_without_changing_p0() -> None:
    anchors = _anchors()
    memory = initialize_seed_memory(anchors)
    at_capacity, transactions = propagate_memory(
        memory,
        anchors,
        config=PropagationConfig(
            rounds=1,
            maximum_structural_primitives=1,
            maximum_volumetric_primitives=1,
            maximum_patient_primitives=2,
        ),
        bounds_min_ras_mm=torch.tensor([-5.0] * 3),
        bounds_max_ras_mm=torch.tensor([5.0] * 3),
    )
    transaction = transactions[0]
    assert at_capacity.memory_hash == memory.memory_hash
    assert transaction.proposal_count == 2
    assert transaction.accepted_count == 0
    assert transaction.rejected_budget == 2

    two_anchor_memory = initialize_seed_memory(_two_anchors())
    with pytest.raises(ValueError, match="structural seed primitive count"):
        propagate_memory(
            two_anchor_memory,
            _two_anchors(),
            config=PropagationConfig(maximum_structural_primitives=1, maximum_volumetric_primitives=2),
            bounds_min_ras_mm=torch.tensor([-5.0] * 3),
            bounds_max_ras_mm=torch.tensor([5.0] * 3),
        )
    p0_result, p0_transactions = propagate_memory(
        two_anchor_memory,
        _two_anchors(),
        config=PropagationConfig(variant="p0", maximum_structural_primitives=1, maximum_volumetric_primitives=1),
        bounds_min_ras_mm=torch.tensor([-5.0] * 3),
        bounds_max_ras_mm=torch.tensor([5.0] * 3),
    )
    assert p0_result is two_anchor_memory
    assert p0_transactions == ()


def test_p1_uses_declared_structural_tangent_or_no_propagation_policy() -> None:
    anchors = _anchors()
    memory = initialize_seed_memory(anchors)
    tangent_config = PropagationConfig(
        rounds=1,
        children_per_parent_per_round=4,
        structural_propagation_policy="tangent_only",
        maximum_structural_primitives=8,
        maximum_volumetric_primitives=8,
    )
    tangent, _ = propagate_memory(
        memory,
        anchors,
        config=tangent_config,
        bounds_min_ras_mm=torch.tensor([-5.0] * 3),
        bounds_max_ras_mm=torch.tensor([5.0] * 3),
    )
    structural_children = tangent.structural.gaussians.centers_ras_mm[memory.structural.gaussians.count :]
    volumetric_children = tangent.volumetric.gaussians.centers_ras_mm[memory.volumetric.gaussians.count :]
    structural_local = structural_children - memory.structural.gaussians.centers_ras_mm[0]
    volumetric_local = volumetric_children - memory.volumetric.gaussians.centers_ras_mm[0]
    assert structural_children.shape[0] == 4
    assert torch.allclose(structural_local[:, 2], torch.zeros(4))
    assert torch.allclose(structural_local[:, :2].abs().sum(dim=1), torch.ones(4))
    assert volumetric_children.shape[0] == 2
    assert torch.allclose(volumetric_local[:, :2], torch.zeros(2, 2))
    assert torch.allclose(volumetric_local[:, 2].abs(), torch.ones(2))

    no_structural, transactions = propagate_memory(
        memory,
        anchors,
        config=PropagationConfig(
            rounds=1,
            children_per_parent_per_round=2,
            structural_propagation_policy="none",
            maximum_structural_primitives=8,
            maximum_volumetric_primitives=8,
        ),
        bounds_min_ras_mm=torch.tensor([-5.0] * 3),
        bounds_max_ras_mm=torch.tensor([5.0] * 3),
    )
    assert no_structural.structural.gaussians.count == memory.structural.gaussians.count
    assert no_structural.volumetric.gaussians.count == memory.volumetric.gaussians.count + 2
    assert transactions[0].proposal_count == 2


def test_p1_is_equivariant_under_proper_ras_rotation() -> None:
    anchors = _anchors()
    rotation = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    rotated_anchors = _rotated_anchors(anchors, rotation)
    config = PropagationConfig(
        rounds=1,
        children_per_parent_per_round=2,
        maximum_structural_primitives=8,
        maximum_volumetric_primitives=8,
    )
    kwargs = dict(
        config=config,
        bounds_min_ras_mm=torch.tensor([-5.0] * 3),
        bounds_max_ras_mm=torch.tensor([5.0] * 3),
    )
    propagated, transactions = propagate_memory(initialize_seed_memory(anchors), anchors, **kwargs)
    rotated, rotated_transactions = propagate_memory(
        initialize_seed_memory(rotated_anchors),
        rotated_anchors,
        **kwargs,
    )
    assert torch.allclose(
        rotated.structural.gaussians.centers_ras_mm,
        propagated.structural.gaussians.centers_ras_mm @ rotation.T,
        atol=1e-6,
    )
    assert torch.allclose(
        rotated.volumetric.gaussians.centers_ras_mm,
        propagated.volumetric.gaussians.centers_ras_mm @ rotation.T,
        atol=1e-6,
    )
    assert transactions[0].proposal_count == rotated_transactions[0].proposal_count
    assert transactions[0].accepted_count == rotated_transactions[0].accepted_count


def test_p1_api_excludes_target_and_segmentation_inputs() -> None:
    parameters = inspect.signature(propagate_memory).parameters.values()
    forbidden = ("target", "audit", "segment", "label", "image", "payload")
    assert not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    assert not any(any(token in parameter.name.lower() for token in forbidden) for parameter in parameters)


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


def test_explicit_seed_and_reserve_budget_leaves_declared_p1_capacity() -> None:
    anchors = _anchors()
    memory = initialize_seed_memory(anchors)
    config = PropagationConfig(
        variant="p1",
        rounds=1,
        maximum_total_anchors=1,
        structural_seed_budget=1,
        volumetric_seed_budget=1,
        propagation_reserved_budget=2,
        maximum_structural_primitives=2,
        maximum_volumetric_primitives=2,
        maximum_patient_primitives=4,
    )
    validate_seed_and_reserve_budgets(memory, anchors, config=config)
    assert memory.primitive_count + config.propagation_reserved_budget == config.maximum_patient_primitives

    with pytest.raises(ValueError, match="propagation_reserved_budget"):
        PropagationConfig(variant="p1", rounds=1, propagation_reserved_budget=0)
    with pytest.raises(ValueError, match="seed primitive count plus propagation_reserved_budget"):
        validate_seed_and_reserve_budgets(
            memory,
            anchors,
            config=PropagationConfig(
                variant="p1",
                rounds=1,
                structural_seed_budget=1,
                volumetric_seed_budget=1,
                propagation_reserved_budget=3,
                maximum_structural_primitives=2,
                maximum_volumetric_primitives=2,
                maximum_patient_primitives=4,
            ),
        )
