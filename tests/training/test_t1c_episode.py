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
from smagm.data.episodes import EpisodeSamplingConfig, EpisodeSamplingError, EpisodeSamplingFailureReason
from smagm.losses.reconstruction import ReconstructionLossConfig
from smagm.renderer import RenderConfig
from smagm.training.episode import LegalEpisodeConfig
from smagm.training.objective import T1CObjectiveConfig
from smagm.training.provenance import module_state_hash
from smagm.training.sampling import build_matched_variant_schedule
from smagm.training.schedule import StageConfig, TrainingSchedule, TrainingStage
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


def _episode(
    tmp_path,
    *,
    episode_id: str = "episode",
    context_modality: str = "T2",
    target_modality: str = "T2",
    payload_phase: float = 0.0,
    manifest_id: str = "m",
):
    shape = (17, 15)
    payloads = {"context": _payload(payload_phase, shape), "target": _payload(payload_phase + 0.3, shape)}
    entries = tuple(
        AvailabilityObservationMeta(
            observation_id=identifier,
            patient_id="patient-a",
            split="train",
            relative_path=f"{identifier}.npy",
            modality_id=context_modality if identifier == "context" else target_modality,
            plane=_plane(identifier, float(index), shape),
            is_synthetic=True,
        )
        for index, identifier in enumerate(payloads)
    )
    for identifier, payload in payloads.items():
        (tmp_path / f"{identifier}.npy").write_bytes(payload)
    manifest = SparseAvailabilityManifest(
        entries,
        manifest_id=manifest_id,
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


def _trainer(
    variant: str,
    seed: int = 7,
    trainer_config: TrainerConfig | None = None,
    *,
    manifest_hash: str = "",
    split_registry_hash: str = "",
    scheduled_assignment_hashes: tuple[str, ...] = (),
) -> T1CTrainer:
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
            modality_to_appearance_channel={"T2": 0},
        ),
        trainer_config=trainer_config,
        matched_experiment_identity="a" * 64 if manifest_hash else "",
        resolved_config_hash="b" * 64 if manifest_hash else "",
        sampler_state_hash="c" * 64 if manifest_hash else "",
        manifest_hash=manifest_hash,
        split_registry_hash=split_registry_hash,
        scheduled_assignment_hashes=scheduled_assignment_hashes,
    )


def _checkpoint_trainer(
    variant: str,
    ledger: EpisodeLedger,
    assignments: tuple[EpisodeAssignment, ...],
    *,
    seed: int = 7,
    trainer_config: TrainerConfig | None = None,
) -> T1CTrainer:
    return _trainer(
        variant,
        seed=seed,
        trainer_config=trainer_config,
        manifest_hash=ledger.manifest_hash,
        split_registry_hash=ledger._split_registry.registry_hash,
        scheduled_assignment_hashes=tuple(item.assignment_hash for item in assignments),
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
    assert len({report.support_topology_hash for report in reports}) == 1
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
    trainer = _checkpoint_trainer("e1", ledger, (assignment,), seed=21)
    output = trainer.train_step(ledger=ledger, assignment=assignment, target_id="target")
    checkpoint = trainer.save_checkpoint(tmp_path / "checkpoint.pt")
    expected = module_state_hash(trainer.encoder, trainer.gaussian_head)
    restored = _checkpoint_trainer("e1", ledger, (assignment,), seed=99)
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


def test_checkpoint_rejects_incomplete_accumulation_window(tmp_path) -> None:
    trainer = _trainer("e1", trainer_config=TrainerConfig(accumulation_steps=2))
    ledger, assignment = _episode(tmp_path)
    trainer.train_step(ledger=ledger, assignment=assignment, target_id="target")
    with pytest.raises(RuntimeError, match="incomplete gradient-accumulation"):
        trainer.save_checkpoint(tmp_path / "mid-window.pt")


def test_schedule_runs_context_only_warmup_then_joint_and_reconstruction_dominant(tmp_path) -> None:
    schedule = TrainingSchedule(structural_warmup_steps=1, joint_reconstruction_steps=1)
    trainer = _trainer("e1", trainer_config=TrainerConfig(schedule=schedule))
    warmup_root = tmp_path / "warmup"
    joint_root = tmp_path / "joint"
    dominant_root = tmp_path / "dominant"
    for root in (warmup_root, joint_root, dominant_root):
        root.mkdir()
    warmup_ledger, warmup_assignment = _episode(warmup_root, episode_id="warmup")
    warmup = trainer.train_step(ledger=warmup_ledger, assignment=warmup_assignment)
    assert warmup.report.stage == "structural_warmup"
    assert warmup.report.legal_target_pixel_count == 0
    assert warmup.report.reconstruction_intensity_loss is None
    assert [event.event for event in warmup_ledger.event_records] == ["OPEN_CONTEXT"]
    joint_ledger, joint_assignment = _episode(joint_root, episode_id="joint")
    joint = trainer.train_step(ledger=joint_ledger, assignment=joint_assignment, target_id="target")
    assert joint.report.stage == "joint_reconstruction"
    assert joint.report.reconstruction_intensity_loss is not None
    dominant_ledger, dominant_assignment = _episode(dominant_root, episode_id="dominant")
    dominant = trainer.train_step(ledger=dominant_ledger, assignment=dominant_assignment, target_id="target")
    assert dominant.report.stage == "reconstruction_dominant"
    assert dominant.report.reconstruction_intensity_loss is not None


def test_unavailable_structural_component_is_reported_not_silently_zeroed(tmp_path) -> None:
    config = TrainerConfig(objective=T1CObjectiveConfig(structural_weights=(("not_implemented", 1.0),)))
    ledger, assignment = _episode(tmp_path)
    result = _trainer("e1", trainer_config=config).train_step(ledger=ledger, assignment=assignment, target_id="target")
    assert result.report.inactive_components["not_implemented"] == "UNSUPPORTED_DECLARED_STRUCTURAL_COMPONENT"


def test_modality_to_appearance_channel_is_explicit_bounded_and_hashable() -> None:
    first = LegalEpisodeConfig(modality_to_appearance_channel={"T2": 0, "FLAIR": 1})
    second = LegalEpisodeConfig(modality_to_appearance_channel={"FLAIR": 1, "T2": 0})
    assert first.appearance_channel_for("T2", available_channels=2) == 0
    assert first.appearance_channel_for("FLAIR", available_channels=2) == 1
    assert first.modality_mapping_hash == second.modality_mapping_hash
    with pytest.raises(ValueError, match="no appearance channel"):
        first.appearance_channel_for("DWI", available_channels=2)
    with pytest.raises(ValueError, match="outside"):
        first.appearance_channel_for("FLAIR", available_channels=1)


def test_legal_episode_rejects_assignment_without_target_modality_context(tmp_path) -> None:
    ledger, assignment = _episode(tmp_path, context_modality="T2", target_modality="FLAIR")
    trainer = _trainer("e1")
    with pytest.raises(EpisodeSamplingError) as exc_info:
        trainer.train_step(ledger=ledger, assignment=assignment, target_id="target")
    assert exc_info.value.reason is EpisodeSamplingFailureReason.MISSING_CONTEXT_MODALITY
    assert ledger.event_records == ()


def test_checkpoint_resume_matches_uninterrupted_execution_at_optimizer_boundary(tmp_path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    resumed_first_root = tmp_path / "resumed-first"
    resumed_second_root = tmp_path / "resumed-second"
    for root in (first_root, second_root, resumed_first_root, resumed_second_root):
        root.mkdir()
    first_ledger, first_assignment = _episode(first_root, episode_id="first")
    second_ledger, second_assignment = _episode(second_root, episode_id="second")
    uninterrupted = _checkpoint_trainer("e1", first_ledger, (first_assignment, second_assignment), seed=71)
    uninterrupted.train_step(ledger=first_ledger, assignment=first_assignment, target_id="target")
    uninterrupted.train_step(ledger=second_ledger, assignment=second_assignment, target_id="target")
    expected = module_state_hash(uninterrupted.encoder, uninterrupted.gaussian_head)
    resumed_first_ledger, resumed_first_assignment = _episode(resumed_first_root, episode_id="first")
    resumed_second_ledger, resumed_second_assignment = _episode(resumed_second_root, episode_id="second")
    resumable = _checkpoint_trainer(
        "e1", resumed_first_ledger, (resumed_first_assignment, resumed_second_assignment), seed=71
    )
    resumable.train_step(ledger=resumed_first_ledger, assignment=resumed_first_assignment, target_id="target")
    checkpoint = resumable.save_checkpoint(tmp_path / "boundary.pt")
    restored = _checkpoint_trainer(
        "e1", resumed_first_ledger, (resumed_first_assignment, resumed_second_assignment), seed=999
    )
    restored.load_checkpoint(checkpoint)
    restored.train_step(ledger=resumed_second_ledger, assignment=resumed_second_assignment, target_id="target")
    assert module_state_hash(restored.encoder, restored.gaussian_head) == expected


def test_checkpoint_rejects_a_different_head_contract_with_compatible_shapes(tmp_path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    ledger, assignment = _episode(data_root)
    trainer = _checkpoint_trainer("e1", ledger, (assignment,), seed=73)
    trainer.train_step(ledger=ledger, assignment=assignment, target_id="target")
    checkpoint = trainer.save_checkpoint(tmp_path / "head-contract.pt")
    changed_encoder = EvidenceEncoder(EncoderConfig(variant="e1"))
    changed_head = FixedGaussianHead(
        FixedGaussianHeadConfig(input_dim=25, min_scale_mm=1.6, max_scale_mm=5.0, max_center_offset_mm=0.2)
    )
    changed_optimizer = torch.optim.Adam(list(changed_encoder.parameters()) + list(changed_head.parameters()), lr=1e-3)
    incompatible = T1CTrainer(
        encoder=changed_encoder,
        gaussian_head=changed_head,
        optimizer=changed_optimizer,
        episode_config=LegalEpisodeConfig(
            supports=FixedSupportConfig(step_vu=(4, 4), border_vu=(1, 1)),
            renderer=RenderConfig(support_epsilon=1e-10),
            reconstruction_loss=ReconstructionLossConfig(intensity="mse"),
            modality_to_appearance_channel={"T2": 0},
        ),
        matched_experiment_identity="a" * 64,
        resolved_config_hash="b" * 64,
        sampler_state_hash="c" * 64,
        manifest_hash=ledger.manifest_hash,
        split_registry_hash=ledger._split_registry.registry_hash,
        scheduled_assignment_hashes=(assignment.assignment_hash,),
    )
    with pytest.raises(ValueError, match="immutable T1-C binding"):
        incompatible.load_checkpoint(checkpoint)


def test_checkpoint_rejects_missing_or_tampered_run_bindings(tmp_path) -> None:
    ledger, assignment = _episode(tmp_path)
    unbound = _trainer("e1")
    unbound.train_step(ledger=ledger, assignment=assignment, target_id="target")
    with pytest.raises(RuntimeError, match="complete immutable run"):
        unbound.save_checkpoint(tmp_path / "unbound.pt")
    bound_root = tmp_path / "bound"
    bound_root.mkdir()
    bound_ledger, bound_assignment = _episode(bound_root)
    trainer = _checkpoint_trainer("e1", bound_ledger, (bound_assignment,), seed=91)
    trainer.train_step(ledger=bound_ledger, assignment=bound_assignment, target_id="target")
    checkpoint = trainer.save_checkpoint(tmp_path / "bound.pt")
    tampered = torch.load(checkpoint, map_location="cpu", weights_only=True)
    tampered["last_manifest_hash"] = "f" * 64
    tampered_path = tmp_path / "tampered.pt"
    torch.save(tampered, tampered_path)
    restored = _checkpoint_trainer("e1", bound_ledger, (bound_assignment,), seed=92)
    with pytest.raises(ValueError, match="resume bindings"):
        restored.load_checkpoint(tampered_path)


def test_resumed_trainer_rejects_a_ledger_outside_manifest_and_schedule(tmp_path) -> None:
    first_root = tmp_path / "first"
    other_root = tmp_path / "other"
    first_root.mkdir()
    other_root.mkdir()
    ledger, assignment = _episode(first_root, episode_id="first")
    trainer = _checkpoint_trainer("e1", ledger, (assignment,), seed=93)
    trainer.train_step(ledger=ledger, assignment=assignment, target_id="target")
    checkpoint = trainer.save_checkpoint(tmp_path / "resume.pt")
    other_ledger, other_assignment = _episode(
        other_root, episode_id="other", payload_phase=1.0, manifest_id="different-manifest"
    )
    restored = _checkpoint_trainer("e1", ledger, (assignment,), seed=94)
    restored.load_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="manifest"):
        restored.train_step(ledger=other_ledger, assignment=other_assignment, target_id="target")
    assert other_ledger.event_records == ()


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
    # One declared intensity-perturbation view is legal structural supervision;
    # both encoder calls still originate from the sole context observation.
    assert len(calls) == 2 * len(assignment.context_ids) == 2
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
