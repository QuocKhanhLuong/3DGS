from __future__ import annotations

import hashlib
from io import BytesIO

import numpy as np
import torch

from smagm.baselines.fixed_gaussian import FixedGaussianHead, FixedGaussianHeadConfig
from smagm.baselines.fixed_support import FixedSupportConfig
from smagm.contracts.coordinates import PhysicalPlane
from smagm.contracts.episode import EpisodeAssignment, EpisodeLedger
from smagm.contracts.observation import AvailabilityObservationMeta, PatientSplitRegistry, SparseAvailabilityManifest
from smagm.features.encoder import EncoderConfig, EvidenceEncoder
from smagm.fields import SharedStructuralField, StructuralFieldConfig
from smagm.losses.reconstruction import ReconstructionLossConfig
from smagm.memory import PropagationConfig
from smagm.renderer import RenderConfig
from smagm.training import LegalEpisodeConfig, build_static_episode_step


def _payload(phase: float, shape: tuple[int, int]) -> bytes:
    v, u = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    array = np.asarray(np.sin(u / 3.0 + phase) + np.cos(v / 4.0), dtype=np.float32)
    buffer = BytesIO(); np.save(buffer, array, allow_pickle=False); return buffer.getvalue()


def _plane(observation_id: str, z: float, shape: tuple[int, int]) -> PhysicalPlane:
    return PhysicalPlane((0.0, 0.0, z), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0), 1.0, shape, (0.0, 0.0, 1.0), observation_id=observation_id)


def test_static_anchor_field_propagation_obeys_receipt_barrier_and_backpropagates(tmp_path) -> None:
    shape = (17, 15); payloads = {"context": _payload(0.0, shape), "target": _payload(0.3, shape)}
    entries = tuple(AvailabilityObservationMeta(
        observation_id=name, patient_id="patient", split="train", relative_path=f"{name}.npy",
        modality_id="T2", plane=_plane(name, float(index), shape), is_synthetic=True,
    ) for index, name in enumerate(payloads))
    for name, payload in payloads.items(): (tmp_path / f"{name}.npy").write_bytes(payload)
    manifest = SparseAvailabilityManifest(entries, manifest_id="static", integrity_digests={name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()})
    assignment = EpisodeAssignment.create(manifest, episode_id="episode", patient_id="patient", context_ids=("context",), target_ids=("target",))
    ledger = EpisodeLedger(manifest, assignment, tmp_path, split_registry=PatientSplitRegistry.create((manifest,)))
    torch.manual_seed(4); encoder = EvidenceEncoder(EncoderConfig(variant="e2"))
    head = FixedGaussianHead(FixedGaussianHeadConfig(input_dim=25, appearance_channels=1))
    field = SharedStructuralField(StructuralFieldConfig(evidence_dim=52, hidden_width=16))
    config = LegalEpisodeConfig(
        supports=FixedSupportConfig(step_vu=(4, 4), border_vu=(1, 1)), renderer=RenderConfig(support_epsilon=1e-10),
        reconstruction_loss=ReconstructionLossConfig(intensity="mse"), modality_to_appearance_channel={"T2": 0},
    )
    result = build_static_episode_step(
        ledger=ledger, assignment=assignment, target_id="target", encoder=encoder, gaussian_head=head,
        config=config, patient_id="patient", manifest_hash=manifest.manifest_hash,
        patient_config_hash=hashlib.sha256(b"config").hexdigest(), field_model=field,
        field_config_hash=field.config.config_hash, propagation_config=PropagationConfig(variant="p1", rounds=1),
    )
    event_names = [event.event for event in ledger.event_records]
    assert event_names.index("REGISTER_PREDICTION") < event_names.index("REVEAL_TARGET")
    assert result.patient_state.context_observation_ids == ("context",)
    assert "target" not in result.patient_state.__dict__
    result.loss.total.backward()
    encoder_grads = {name: None if parameter.grad is None else float(parameter.grad.abs().sum()) for name, parameter in encoder.named_parameters()}
    head_grads = {name: None if parameter.grad is None else float(parameter.grad.abs().sum()) for name, parameter in head.named_parameters()}
    field_grads = {name: None if parameter.grad is None else float(parameter.grad.abs().sum()) for name, parameter in field.named_parameters()}
    assert any(value is not None and value > 0 for value in encoder_grads.values()), encoder_grads
    assert any(value is not None and value > 0 for value in head_grads.values()), head_grads
    assert all(value is not None and value > 0 for value in field_grads.values()), field_grads


def test_r4_far_target_keeps_field_on_the_live_volumetric_render_path(tmp_path) -> None:
    """A sparse interior target must not make the thin structural bank vanish from autograd."""

    shape = (17, 15)
    payloads = {"context": _payload(0.0, shape), "target": _payload(0.3, shape)}
    entries = tuple(
        AvailabilityObservationMeta(
            observation_id=name,
            patient_id="patient",
            split="train",
            relative_path=f"{name}.npy",
            modality_id="T2",
            plane=_plane(name, z, shape),
            is_synthetic=True,
        )
        for name, z in (("context", 0.0), ("target", 12.0))
    )
    for name, payload in payloads.items():
        (tmp_path / f"{name}.npy").write_bytes(payload)
    manifest = SparseAvailabilityManifest(
        entries,
        manifest_id="static-far-target",
        integrity_digests={name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
    )
    assignment = EpisodeAssignment.create(
        manifest,
        episode_id="far-target-episode",
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
    torch.manual_seed(19)
    encoder = EvidenceEncoder(EncoderConfig(variant="e2"))
    head = FixedGaussianHead(FixedGaussianHeadConfig(input_dim=25, appearance_channels=1))
    field = SharedStructuralField(StructuralFieldConfig(evidence_dim=52, hidden_width=16))
    config = LegalEpisodeConfig(
        supports=FixedSupportConfig(step_vu=(4, 4), border_vu=(1, 1)),
        renderer=RenderConfig(support_epsilon=1e-10),
        reconstruction_loss=ReconstructionLossConfig(intensity="mse"),
        modality_to_appearance_channel={"T2": 0},
    )
    result = build_static_episode_step(
        ledger=ledger,
        assignment=assignment,
        target_id="target",
        encoder=encoder,
        gaussian_head=head,
        config=config,
        patient_id="patient",
        manifest_hash=manifest.manifest_hash,
        patient_config_hash=hashlib.sha256(b"far-target-config").hexdigest(),
        field_model=field,
        field_config_hash=field.config.config_hash,
        propagation_config=PropagationConfig(variant="p0"),
    )
    result.loss.total.backward()
    field_gradient = torch.sqrt(
        sum((parameter.grad.square().sum() for parameter in field.parameters() if parameter.grad is not None), torch.tensor(0.0))
    )
    assert result.loss.legal_pixel_count > 0
    assert result.patient_state.memory.structural.gaussians.centers_ras_mm.grad_fn is not None
    assert result.patient_state.memory.volumetric.gaussians.centers_ras_mm.grad_fn is not None
    assert field_gradient.isfinite() and field_gradient > 0


def test_deferred_target_perturbation_cannot_change_context_state_before_reveal(tmp_path) -> None:
    """Target bytes are deferred; all state construction must remain identical."""

    shape = (17, 15)
    context_payload = _payload(0.0, shape)
    target_payloads = (_payload(0.3, shape), _payload(1.7, shape))
    entries = tuple(
        AvailabilityObservationMeta(
            observation_id=name,
            patient_id="patient",
            split="train",
            relative_path=f"{name}.npy",
            modality_id="T2",
            plane=_plane(name, float(index), shape),
            is_synthetic=True,
        )
        for index, name in enumerate(("context", "target"))
    )
    (tmp_path / "context.npy").write_bytes(context_payload)
    manifest = SparseAvailabilityManifest(
        entries,
        manifest_id="deferred-target-invariance",
        # The deferred target is bound by a reference digest rather than by
        # materialized target bytes, matching the product source-reference path.
        integrity_digests={
            "context": hashlib.sha256(context_payload).hexdigest(),
            "target": hashlib.sha256(b"deferred-target-reference").hexdigest(),
        },
    )
    assignment = EpisodeAssignment.create(
        manifest,
        episode_id="deferred-target-invariance-episode",
        patient_id="patient",
        context_ids=("context",),
        target_ids=("target",),
    )
    registry = PatientSplitRegistry.create((manifest,))
    config = LegalEpisodeConfig(
        supports=FixedSupportConfig(step_vu=(4, 4), border_vu=(1, 1)),
        renderer=RenderConfig(support_epsilon=1e-10),
        reconstruction_loss=ReconstructionLossConfig(intensity="mse"),
        modality_to_appearance_channel={"T2": 0},
    )

    def run_one(target_payload: bytes):
        ledger = EpisodeLedger(
            manifest,
            assignment,
            tmp_path,
            split_registry=registry,
            deferred_target_readers={"target": lambda: target_payload},
        )
        torch.manual_seed(41)
        encoder = EvidenceEncoder(EncoderConfig(variant="e2"))
        head = FixedGaussianHead(FixedGaussianHeadConfig(input_dim=25, appearance_channels=1))
        field = SharedStructuralField(StructuralFieldConfig(evidence_dim=52, hidden_width=16))
        result = build_static_episode_step(
            ledger=ledger,
            assignment=assignment,
            target_id="target",
            encoder=encoder,
            gaussian_head=head,
            config=config,
            patient_id="patient",
            manifest_hash=manifest.manifest_hash,
            patient_config_hash=hashlib.sha256(b"config").hexdigest(),
            field_model=field,
            field_config_hash=field.config.config_hash,
            propagation_config=PropagationConfig(variant="p1", rounds=1),
        )
        return ledger, result

    first_ledger, first = run_one(target_payloads[0])
    second_ledger, second = run_one(target_payloads[1])

    assert first.context_step.preprocessing.record_hash == second.context_step.preprocessing.record_hash
    assert first.context_step.feature_cache_key_hashes == second.context_step.feature_cache_key_hashes
    assert first.patient_state.anchors.evidence_hash == second.patient_state.anchors.evidence_hash
    assert first.patient_state.memory.memory_hash == second.patient_state.memory.memory_hash
    assert first.patient_state.state_version == second.patient_state.state_version
    assert torch.equal(first.prediction.intensity, second.prediction.intensity)
    assert not torch.equal(first.target, second.target)
    first_events = [event.event for event in first_ledger.event_records]
    second_events = [event.event for event in second_ledger.event_records]
    assert first_events.index("REGISTER_PREDICTION") < first_events.index("REVEAL_TARGET")
    assert second_events.index("REGISTER_PREDICTION") < second_events.index("REVEAL_TARGET")
