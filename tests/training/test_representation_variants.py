from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch

from smagm.anchors import AnchorBootstrapConfig, CandidateSelectionConfig
from smagm.baselines import (
    RepresentationVariant,
    SparseInterpolationConfig,
    resolve_representation_plan,
)
from smagm.baselines.fixed_gaussian import FixedGaussianHead, FixedGaussianHeadConfig
from smagm.baselines.fixed_support import FixedSupportConfig
from smagm.contracts.coordinates import PhysicalPlane
from smagm.contracts.episode import EpisodeAssignment, EpisodeLedger
from smagm.contracts.observation import AvailabilityObservationMeta, PatientSplitRegistry, SparseAvailabilityManifest
from smagm.features.encoder import EncoderConfig, EvidenceEncoder
from smagm.fields import GlobalStructuralField, GlobalStructuralFieldConfig, SharedStructuralField, StructuralFieldConfig
from smagm.losses.reconstruction import ReconstructionLossConfig
from smagm.memory import PropagationConfig
from smagm.renderer import RenderConfig
from smagm.training import (
    AnchorEvidenceProjector,
    AnchorEvidenceProjectorConfig,
    LegalEpisodeConfig,
    build_representation_episode_step,
)


def _payload(phase: float, shape: tuple[int, int]) -> bytes:
    v, u = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    array = np.asarray(np.sin(u / 3.0 + phase) + np.cos(v / 4.0), dtype=np.float32)
    buffer = BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _plane(observation_id: str, z: float, shape: tuple[int, int]) -> PhysicalPlane:
    return PhysicalPlane(
        (0.0, 0.0, z),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0),
        1.0,
        shape,
        (0.0, 0.0, 1.0),
        observation_id=observation_id,
    )


def _episode(tmp_path, suffix: str) -> tuple[EpisodeLedger, EpisodeAssignment]:
    shape = (9, 9)
    payloads = {"context": _payload(0.0, shape), "target": _payload(0.2, shape)}
    entries = tuple(
        AvailabilityObservationMeta(
            observation_id=name,
            patient_id="patient",
            split="train",
            relative_path=f"{suffix}-{name}.npy",
            modality_id="T2",
            plane=_plane(name, float(index), shape),
            is_synthetic=True,
        )
        for index, name in enumerate(payloads)
    )
    for name, payload in payloads.items():
        (tmp_path / f"{suffix}-{name}.npy").write_bytes(payload)
    manifest = SparseAvailabilityManifest(
        entries,
        manifest_id=f"variants-{suffix}",
        integrity_digests={name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
    )
    assignment = EpisodeAssignment.create(
        manifest,
        episode_id=f"episode-{suffix}",
        patient_id="patient",
        context_ids=("context",),
        target_ids=("target",),
    )
    ledger = EpisodeLedger(
        manifest,
        assignment,
        tmp_path,
        split_registry=PatientSplitRegistry.create((manifest,)),
    )
    return ledger, assignment


def _multi_target_episode(tmp_path) -> tuple[EpisodeLedger, EpisodeAssignment]:
    shape = (9, 9)
    payloads = {
        "context": _payload(0.0, shape),
        "target-a": _payload(0.2, shape),
        "target-b": _payload(0.4, shape),
    }
    entries = tuple(
        AvailabilityObservationMeta(
            observation_id=name,
            patient_id="patient",
            split="train",
            relative_path=f"multi-{name}.npy",
            modality_id="T2",
            plane=_plane(name, float(index), shape),
            is_synthetic=True,
        )
        for index, name in enumerate(payloads)
    )
    for name, payload in payloads.items():
        (tmp_path / f"multi-{name}.npy").write_bytes(payload)
    manifest = SparseAvailabilityManifest(
        entries,
        manifest_id="variants-multi-target",
        integrity_digests={name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
    )
    assignment = EpisodeAssignment.create(
        manifest,
        episode_id="episode-multi-target",
        patient_id="patient",
        context_ids=("context",),
        target_ids=("target-a", "target-b"),
    )
    return (
        EpisodeLedger(
            manifest,
            assignment,
            tmp_path,
            split_registry=PatientSplitRegistry.create((manifest,)),
        ),
        assignment,
    )


def _multimodal_episode(tmp_path) -> tuple[EpisodeLedger, EpisodeAssignment]:
    shape = (9, 9)
    specification = (
        ("context-t1", "T1", 0.0),
        ("context-t2", "T2", 0.3),
        ("target-t2", "T2", 0.5),
    )
    payloads = {name: _payload(phase, shape) for name, _, phase in specification}
    entries = tuple(
        AvailabilityObservationMeta(
            observation_id=name,
            patient_id="patient",
            split="train",
            relative_path=f"mapped-{name}.npy",
            modality_id=modality,
            plane=_plane(name, float(index), shape),
            is_synthetic=True,
        )
        for index, (name, modality, _) in enumerate(specification)
    )
    for name, payload in payloads.items():
        (tmp_path / f"mapped-{name}.npy").write_bytes(payload)
    manifest = SparseAvailabilityManifest(
        entries,
        manifest_id="variants-explicit-mapping",
        integrity_digests={name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
    )
    assignment = EpisodeAssignment.create(
        manifest,
        episode_id="episode-explicit-mapping",
        patient_id="patient",
        context_ids=("context-t1", "context-t2"),
        target_ids=("target-t2",),
    )
    return (
        EpisodeLedger(
            manifest,
            assignment,
            tmp_path,
            split_registry=PatientSplitRegistry.create((manifest,)),
        ),
        assignment,
    )


def _config() -> LegalEpisodeConfig:
    return LegalEpisodeConfig(
        supports=FixedSupportConfig(step_vu=(4, 4)),
        renderer=RenderConfig(support_epsilon=1e-10),
        reconstruction_loss=ReconstructionLossConfig(intensity="mse"),
        modality_to_appearance_channel={"T2": 0},
    )


def _encoder_head(*, appearance_channels: int = 1) -> tuple[EvidenceEncoder, FixedGaussianHead]:
    return (
        EvidenceEncoder(EncoderConfig(variant="e2")),
        FixedGaussianHead(FixedGaussianHeadConfig(input_dim=25, appearance_channels=appearance_channels)),
    )


def _anchor_projector(head: FixedGaussianHead) -> AnchorEvidenceProjector:
    return AnchorEvidenceProjector(
        AnchorEvidenceProjectorConfig(evidence_dim=52, head_input_dim=head.config.input_dim)
    )


def test_representation_switches_have_exact_module_inventories_and_reject_t4_or_mismatched_p1() -> None:
    expected_removed = {
        "r0": ("encoder", "physical_anchors", "shared_local_field", "global_coordinate_field"),
        "r1": ("physical_anchors", "shared_local_field", "global_coordinate_field"),
        "r2": ("encoder", "physical_anchors", "shared_local_field", "global_coordinate_field"),
        "r3": ("shared_local_field", "global_coordinate_field"),
        "r4": ("global_coordinate_field",),
        "r5": ("shared_local_field",),
    }
    for variant, removed in expected_removed.items():
        plan = resolve_representation_plan(variant)
        assert all(module not in plan.active_modules for module in removed)
        assert plan.plan_hash == resolve_representation_plan(variant).plan_hash
    full = resolve_representation_plan("r4", propagation_variant="p1")
    assert full.active_modules[-1] == "bounded_fixed_propagation"
    with pytest.raises(ValueError, match="P1"):
        resolve_representation_plan("r3", propagation_variant="p1")
    with pytest.raises(ValueError, match="P0 and bounded fixed P1"):
        resolve_representation_plan("r4", propagation_variant="p4")


@pytest.mark.parametrize("variant", ("r0", "r2"))
def test_interpolation_and_free_gaussian_baselines_use_no_encoder_or_anchor_field(tmp_path, variant: str) -> None:
    ledger, assignment = _episode(tmp_path, variant)
    result = build_representation_episode_step(
        ledger=ledger,
        assignment=assignment,
        target_id="target",
        representation_variant=variant,
        config=_config(),
        interpolation_config=SparseInterpolationConfig(stride_vu=(3, 3)),
    )
    assert result.patient_state is None
    assert [event.event for event in ledger.event_records][-2:] == ["REGISTER_PREDICTION", "REVEAL_TARGET"]
    assert result.loss.status == "OK"
    if variant == "r2":
        assert result.free_gaussian_state is not None
        result.loss.total.backward()
        gradients = [parameter.grad for parameter in result.free_gaussian_state.parameters()]
        assert all(value is not None and bool(torch.isfinite(value).all()) for value in gradients)
        assert any(float(value.abs().sum()) > 0 for value in gradients if value is not None)
    else:
        assert result.free_gaussian_state is None


@pytest.mark.parametrize("variant", ("r3", "r4", "r5"))
def test_anchor_representation_switches_build_only_the_selected_field(tmp_path, variant: str) -> None:
    torch.manual_seed(9)
    ledger, assignment = _episode(tmp_path, variant)
    encoder, head = _encoder_head()
    projector = _anchor_projector(head)
    local = SharedStructuralField(StructuralFieldConfig(evidence_dim=52, hidden_width=16)) if variant == "r4" else None
    global_field = GlobalStructuralField(
        GlobalStructuralFieldConfig(evidence_dim=52, coordinate_scale_mm=16.0, hidden_width=16)
    ) if variant == "r5" else None
    result = build_representation_episode_step(
        ledger=ledger,
        assignment=assignment,
        target_id="target",
        representation_variant=variant,
        config=_config(),
        encoder=encoder,
        gaussian_head=head,
        anchor_evidence_projector=projector,
        local_field=local,
        global_field=global_field,
        bootstrap_config=AnchorBootstrapConfig(candidate=CandidateSelectionConfig(maximum_candidates=4)),
        propagation_config=PropagationConfig(variant="p0"),
    )
    assert result.patient_state is not None
    assert result.patient_state.memory.primitive_count > 0
    assert result.propagation_transactions == ()
    assert result.representation_plan.variant is RepresentationVariant(variant.replace("r3", "direct_anchor_gaussian").replace("r4", "anchor_field").replace("r5", "global_field"))
    result.loss.total.backward()
    selected_field = local if local is not None else global_field
    if selected_field is not None:
        gradients = [parameter.grad for parameter in selected_field.parameters()]
        assert all(value is not None and bool(torch.isfinite(value).all()) for value in gradients)
        assert any(float(value.abs().sum()) > 0 for value in gradients if value is not None)
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
        for parameter in projector.parameters()
    )


def test_anchor_representation_dispatcher_preserves_p1_transactions(tmp_path) -> None:
    torch.manual_seed(23)
    ledger, assignment = _episode(tmp_path, "r4-p1")
    encoder, head = _encoder_head()
    projector = _anchor_projector(head)
    field = SharedStructuralField(StructuralFieldConfig(evidence_dim=52, hidden_width=16))
    result = build_representation_episode_step(
        ledger=ledger,
        assignment=assignment,
        target_id="target",
        representation_variant="r4",
        propagation_variant="p1",
        config=_config(),
        encoder=encoder,
        gaussian_head=head,
        anchor_evidence_projector=projector,
        local_field=field,
        bootstrap_config=AnchorBootstrapConfig(candidate=CandidateSelectionConfig(maximum_candidates=4)),
        propagation_config=PropagationConfig(variant="p1", rounds=1),
    )
    assert len(result.propagation_transactions) == 1
    transaction = result.propagation_transactions[0]
    assert transaction.round_index == 1
    assert transaction.child_memory_hash == result.patient_state.memory.memory_hash


def test_anchor_representation_dispatcher_forwards_phase_timing_to_static_r4(tmp_path) -> None:
    """Keep the product R4 diagnostic keyword on the dispatcher contract."""

    torch.manual_seed(31)
    ledger, assignment = _episode(tmp_path, "r4-phase-timing")
    encoder, head = _encoder_head()
    projector = _anchor_projector(head)
    field = SharedStructuralField(StructuralFieldConfig(evidence_dim=52, hidden_width=16))
    result = build_representation_episode_step(
        ledger=ledger,
        assignment=assignment,
        target_id="target",
        representation_variant="r4",
        propagation_variant="p0",
        config=_config(),
        encoder=encoder,
        gaussian_head=head,
        anchor_evidence_projector=projector,
        local_field=field,
        bootstrap_config=AnchorBootstrapConfig(candidate=CandidateSelectionConfig(maximum_candidates=4)),
        propagation_config=PropagationConfig(variant="p0"),
        collect_phase_timing=True,
    )
    assert result.phase_timing_ms is not None
    expected = {
        "encoder_wall_time_ms",
        "anchor_build_wall_time_ms",
        "field_query_wall_time_ms",
        "propagation_wall_time_ms",
        "renderer_wall_time_ms",
        "loss_wall_time_ms",
    }
    assert expected <= result.phase_timing_ms.keys()
    assert all(
        isinstance(result.phase_timing_ms[name], float) and result.phase_timing_ms[name] >= 0.0
        for name in expected
    )


def test_static_memory_channel_order_follows_explicit_mapping_not_modality_sort(tmp_path) -> None:
    torch.manual_seed(14)
    ledger, assignment = _multimodal_episode(tmp_path)
    encoder, head = _encoder_head(appearance_channels=2)
    projector = _anchor_projector(head)
    config = LegalEpisodeConfig(
        supports=FixedSupportConfig(step_vu=(4, 4)),
        renderer=RenderConfig(support_epsilon=1e-10),
        reconstruction_loss=ReconstructionLossConfig(intensity="mse"),
        modality_to_appearance_channel={"T2": 0, "T1": 1},
    )
    result = build_representation_episode_step(
        ledger=ledger,
        assignment=assignment,
        target_id="target-t2",
        representation_variant="r3",
        config=config,
        encoder=encoder,
        gaussian_head=head,
        anchor_evidence_projector=projector,
        bootstrap_config=AnchorBootstrapConfig(candidate=CandidateSelectionConfig(maximum_candidates=4)),
        propagation_config=PropagationConfig(variant="p0"),
    )
    assert result.patient_state is not None
    assert result.patient_state.memory.modality_ids == ("T2", "T1")
    assert result.loss.status == "OK"


def test_anchor_projector_is_default_and_prefix_is_explicit_ablation(tmp_path) -> None:
    ledger, assignment = _episode(tmp_path, "missing-projector")
    encoder, head = _encoder_head()
    with pytest.raises(ValueError, match="requires an AnchorEvidenceProjector"):
        build_representation_episode_step(
            ledger=ledger,
            assignment=assignment,
            target_id="target",
            representation_variant="r3",
            config=_config(),
            encoder=encoder,
            gaussian_head=head,
            propagation_config=PropagationConfig(variant="p0"),
        )
    assert ledger.event_records == ()

    legacy_ledger, legacy_assignment = _episode(tmp_path, "prefix-ablation")
    legacy_encoder, legacy_head = _encoder_head()
    legacy = build_representation_episode_step(
        ledger=legacy_ledger,
        assignment=legacy_assignment,
        target_id="target",
        representation_variant="r3",
        config=_config(),
        encoder=legacy_encoder,
        gaussian_head=legacy_head,
        propagation_config=PropagationConfig(variant="p0"),
        gaussian_head_input_adapter="anchor_evidence_prefix",
    )
    assert legacy.loss.status == "OK"
    assert legacy.patient_state is not None


def test_fixed_support_gaussian_switch_uses_the_maintained_t1c_path(tmp_path) -> None:
    torch.manual_seed(12)
    ledger, assignment = _episode(tmp_path, "r1")
    encoder, head = _encoder_head()
    result = build_representation_episode_step(
        ledger=ledger,
        assignment=assignment,
        target_id="target",
        representation_variant="r1",
        config=_config(),
        encoder=encoder,
        gaussian_head=head,
    )
    assert result.patient_state is None and result.free_gaussian_state is None
    assert result.loss.status == "OK"
    result.loss.total.backward()
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in head.parameters())


def test_dispatcher_rejects_unused_modules_before_opening_context(tmp_path) -> None:
    ledger, assignment = _episode(tmp_path, "reject-unused")
    encoder, head = _encoder_head()
    with pytest.raises(ValueError, match="reject encoder"):
        build_representation_episode_step(
            ledger=ledger,
            assignment=assignment,
            target_id="target",
            representation_variant="r0",
            encoder=encoder,
            gaussian_head=head,
        )
    assert ledger.event_records == ()


def test_dispatcher_rejects_multi_target_assignment_before_opening_context(tmp_path) -> None:
    ledger, assignment = _multi_target_episode(tmp_path)
    with pytest.raises(ValueError, match="exactly one target"):
        build_representation_episode_step(
            ledger=ledger,
            assignment=assignment,
            target_id="target-a",
            representation_variant="r0",
            config=_config(),
        )
    assert ledger.event_records == ()


def test_static_pipeline_introduces_no_t4_routing_package() -> None:
    repository = Path(__file__).resolve().parents[2]
    assert not (repository / "src" / "smagm" / "routing").exists()
