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
    field_grads = {name: None if parameter.grad is None else float(parameter.grad.abs().sum()) for name, parameter in field.named_parameters()}
    assert any(value is not None and value > 0 for value in encoder_grads.values()), encoder_grads
    assert all(value is not None and value > 0 for value in field_grads.values()), field_grads
