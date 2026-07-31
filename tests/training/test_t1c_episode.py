from __future__ import annotations

import hashlib
from io import BytesIO
import json

import numpy as np
import pytest
import torch

from smagm.baselines.fixed_gaussian import FixedGaussianHead, FixedGaussianHeadConfig
from smagm.baselines.fixed_support import FixedSupportConfig
from smagm.contracts.coordinates import PhysicalPlane
from smagm.contracts.episode import EpisodeAssignment, EpisodeLedger
from smagm.contracts.observation import AvailabilityObservationMeta, PatientSplitRegistry, SparseAvailabilityManifest
from smagm.features.encoder import EncoderConfig, EvidenceEncoder
from smagm.data.episodes import EpisodeSamplingConfig
from smagm.losses.reconstruction import ReconstructionLossConfig
from smagm.renderer import RenderConfig
from smagm.training.episode import LegalEpisodeConfig
from smagm.training.provenance import module_state_hash
from smagm.training.sampling import build_matched_variant_schedule
from smagm.training.trainer import T1CTrainer, TrainerConfig


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


def _episode(tmp_path, *, episode_id: str = "episode"):
    shape = (17, 15)
    payloads = {"context": _payload(0.0, shape), "target": _payload(0.3, shape)}
    entries = tuple(
        AvailabilityObservationMeta(
            observation_id=identifier,
            patient_id="patient-a",
            split="train",
            relative_path=f"{identifier}.npy",
            modality_id="T2",
            plane=_plane(identifier, float(index), shape),
            is_synthetic=True,
        )
        for index, identifier in enumerate(payloads)
    )
    for identifier, payload in payloads.items():
        (tmp_path / f"{identifier}.npy").write_bytes(payload)
    manifest = SparseAvailabilityManifest(
        entries,
        manifest_id="m",
        integrity_digests={identifier: hashlib.sha256(payload).hexdigest() for identifier, payload in payloads.items()},
    )
    assignment = EpisodeAssignment.create(
        manifest,
        episode_id=episode_id,
        patient_id="patient-a",
        context_ids=("context",),
        target_ids=("target",),
    )
    registry = PatientSplitRegistry.create((manifest,))
    return EpisodeLedger(manifest, assignment, tmp_path, split_registry=registry), assignment


def _trainer(variant: str, seed: int = 7, trainer_config: TrainerConfig | None = None) -> T1CTrainer:
    torch.manual_seed(seed)
    encoder = EvidenceEncoder(EncoderConfig(variant=variant))
    torch.manual_seed(seed + 10_000)
    head = FixedGaussianHead(
        FixedGaussianHeadConfig(
            input_dim=25,
            min_scale_mm=1.5,
            max_scale_mm=5.0,
            max_center_offset_mm=0.2,
        )
    )
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-3)
    return T1CTrainer(
        encoder=encoder,
        gaussian_head=head,
        optimizer=optimizer,
        episode_config=LegalEpisodeConfig(
            supports=FixedSupportConfig(step_vu=(4, 4), border_vu=(1, 1)),
            renderer=RenderConfig(support_epsilon=1e-10),
            reconstruction_loss=ReconstructionLossConfig(intensity="mse"),
        ),
        trainer_config=trainer_config,
    )


@pytest.mark.parametrize("variant", ["e0", "e1", "e2"])
def test_legal_step_orders_context_prediction_reveal_and_preserves_gradients(tmp_path, variant: str) -> None:
    ledger, assignment = _episode(tmp_path, episode_id=f"episode-{variant}")
    output = _trainer(variant).train_step(ledger=ledger, assignment=assignment, target_id="target")
    assert [event.event for event in ledger.event_records] == [
        "OPEN_CONTEXT",
        "COMMIT_TARGET",
        "REGISTER_PREDICTION",
        "REVEAL_TARGET",
    ]
    assert [row.role for row in ledger.audit_records] == ["CONTEXT", "TARGET"]
    assert output.step.context_ids == ("context",)
    assert len(output.step.feature_cache_key_hashes) == 1
    assert output.report.target_id == "target"
    assert output.report.legal_target_pixel_count > 0
    assert output.report.head_gradient_norm > 0.0
    if variant == "e0":
        assert output.report.encoder_gradient_norm == 0.0
    else:
        assert output.report.encoder_gradient_norm > 0.0


def test_variants_share_assignment_support_topology_and_independent_state(tmp_path) -> None:
    reports = []
    state_hashes = []
    for index, variant in enumerate(("e0", "e1", "e2")):
        path = tmp_path / variant
        path.mkdir()
        ledger, assignment = _episode(path, episode_id="matched-episode")
        trainer = _trainer(variant, seed=10 + index)
        reports.append(trainer.train_step(ledger=ledger, assignment=assignment, target_id="target").report)
        state_hashes.append(module_state_hash(trainer.encoder, trainer.gaussian_head))
    assert len({report.assignment_hash for report in reports}) == 1
    assert len({report.support_count for report in reports}) == 1
    assert len(set(state_hashes)) == 3


def test_common_gaussian_head_initialization_is_isolated_from_variant_rng_use() -> None:
    trainers = [_trainer(variant, seed=31) for variant in ("e0", "e1", "e2")]
    head_hashes = [module_state_hash(trainer.gaussian_head) for trainer in trainers]
    assert len(set(head_hashes)) == 1


def test_trainer_rejects_optimizer_that_does_not_exactly_own_encoder_and_head() -> None:
    encoder = EvidenceEncoder(EncoderConfig(variant="e1"))
    head = FixedGaussianHead(FixedGaussianHeadConfig(input_dim=25))
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="exactly match"):
        T1CTrainer(encoder=encoder, gaussian_head=head, optimizer=optimizer)


def test_matched_schedule_returns_the_exact_same_assignments_for_all_variants(tmp_path) -> None:
    ledger, assignment = _episode(tmp_path)
    matched = build_matched_variant_schedule(
        ledger._manifest,  # exact sealed manifest; no payload access occurs
        patient_id="patient-a",
        config=EpisodeSamplingConfig(context_count=1, target_count=1, seed=3),
    )
    hashes = {
        variant: tuple(item.assignment_hash for item in matched.for_variant(variant).assignments)
        for variant in ("e0", "e1", "e2")
    }
    assert len(set(hashes.values())) == 1
    assert ledger.event_records == ()
    assert assignment.patient_id == "patient-a"


def test_checkpoint_round_trip_binds_variant_configs_optimizer_and_step(tmp_path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    ledger, assignment = _episode(data_root)
    trainer = _trainer("e1", seed=21)
    output = trainer.train_step(ledger=ledger, assignment=assignment, target_id="target")
    checkpoint = trainer.save_checkpoint(tmp_path / "checkpoint.pt")
    expected = module_state_hash(trainer.encoder, trainer.gaussian_head)
    restored = _trainer("e1", seed=99)
    restored.load_checkpoint(checkpoint)
    assert module_state_hash(restored.encoder, restored.gaussian_head) == expected
    assert restored.checkpoint_state()["step_index"] == output.report.step_index


def test_gradient_accumulation_updates_only_at_declared_boundary(tmp_path) -> None:
    trainer = _trainer("e1", trainer_config=TrainerConfig(accumulation_steps=2))
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_ledger, first_assignment = _episode(first_root, episode_id="first")
    second_ledger, second_assignment = _episode(second_root, episode_id="second")
    first = trainer.train_step(ledger=first_ledger, assignment=first_assignment, target_id="target")
    second = trainer.train_step(ledger=second_ledger, assignment=second_assignment, target_id="target")
    assert first.report.optimizer_updated is False
    assert second.report.optimizer_updated is True


def test_target_is_never_encoded_or_cached_before_or_after_reveal(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger, assignment = _episode(tmp_path)
    trainer = _trainer("e2")
    calls: list[torch.Tensor] = []
    original = trainer.encoder.forward

    def spy(image, planes, modality_ids, valid_mask=None):
        calls.append(image.detach().clone())
        return original(image, planes, modality_ids, valid_mask)

    monkeypatch.setattr(trainer.encoder, "forward", spy)
    output = trainer.train_step(ledger=ledger, assignment=assignment, target_id="target")
    assert len(calls) == len(assignment.context_ids) == 1
    assert len(output.step.feature_cache_key_hashes) == len(assignment.context_ids)
    assert output.step.target_id not in output.step.context_ids


def test_t1c_does_not_create_blocked_t2_or_later_packages() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[2] / "src" / "smagm"
    blocked = ("anchors", "fields", "memory", "state", "routing", "reconstruction", "evaluation")
    assert [name for name in blocked if (root / name).exists()] == []


def test_matched_experiment_config_locks_common_downstream_opportunity() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    config = json.loads((root / "configs" / "experiments" / "t1c_synthetic.json").read_text(encoding="utf-8"))
    assert config["variants"] == ["e0", "e1", "e2"]
    fairness = config["fairness"]
    assert fairness["common_feature_channels"] == [16, 8, 1]
    assert fairness["common_gaussian_head_hidden_dim"] == 32
    assert fairness["hardware_class"] == "cpu"
    assert "never audit" in fairness["checkpoint_selection_rule"]
