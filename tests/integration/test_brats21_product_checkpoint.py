from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from smagm.baselines.fixed_gaussian import FixedGaussianHead, FixedGaussianHeadConfig
from smagm.cli.brats21_product import (
    _global_checkpoint_source_patient,
    _load_product_config,
    _promote_global_model_checkpoint,
    _reconcile_global_promotion_journal,
    _product_metric_rows,
    _quarantine_partial_file,
    _runtime_config,
    _source_hashes_for_patient,
    _training_summary_complete,
    _write_product_metric_reports,
)
from smagm.cli.brats21_smoke import (
    _global_model_binding_hash,
    _hash_untracked_provenance,
    _load_global_model_checkpoint,
    _load_r4_progress_checkpoint,
    _make_optimizer,
    _optimizer_learning_rate,
    _source_grid_from_geometry,
    _source_volume_bounds_from_geometry,
    _save_r4_progress_checkpoint,
)
from smagm.contracts.coordinates import PhysicalPlane
from smagm.contracts.episode import EpisodeAssignment
from smagm.contracts.observation import AvailabilityObservationMeta, SparseAvailabilityManifest
from smagm.data.brats21_prepare import PreparedBraTS21
from smagm.features.encoder import EncoderConfig, EvidenceEncoder
from smagm.fields import SharedStructuralField, StructuralFieldConfig
from smagm.memory import PropagationConfig


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_git_provenance_skips_generated_and_binary_payloads(tmp_path: Path) -> None:
    source = tmp_path / "new_module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    generated = tmp_path / "experiments" / "prepared"
    generated.mkdir(parents=True)
    payload = generated / "context.npy"
    payload.write_bytes(b"not-read-as-provenance-content")

    digest, skipped = _hash_untracked_provenance(tmp_path, ["new_module.py", "experiments/prepared/context.npy"])

    assert len(digest) == 64
    assert skipped == ("experiments/prepared/context.npy",)


def _plane(name: str, depth: float) -> PhysicalPlane:
    return PhysicalPlane(
        (0.0, 0.0, depth),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0),
        1.0,
        (4, 4),
        (0.0, 0.0, 1.0),
        observation_id=name,
    )


def _bundle() -> PreparedBraTS21:
    entries = (
        AvailabilityObservationMeta("context", "patient", "train", "context.npy", "flair", _plane("context", 0.0), True),
        AvailabilityObservationMeta("target", "patient", "train", "target.npy", "flair", _plane("target", 1.0), True),
    )
    manifest = SparseAvailabilityManifest(
        entries,
        manifest_id="product-checkpoint-test",
        integrity_digests={entry.observation_id: _digest(entry.observation_id) for entry in entries},
    )
    assignment = EpisodeAssignment.create(
        manifest,
        episode_id="product-checkpoint-episode",
        patient_id="patient",
        context_ids=("context",),
        target_ids=("target",),
    )
    return PreparedBraTS21(
        root=".",
        manifest=manifest,
        assignment=assignment,
        manifest_json={},
        evaluator_json={"target_plane": _plane("target", 1.0).to_canonical_dict()},
    )


def _modules() -> tuple[EvidenceEncoder, FixedGaussianHead, SharedStructuralField]:
    return (
        EvidenceEncoder(EncoderConfig(variant="e2")),
        FixedGaussianHead(FixedGaussianHeadConfig(input_dim=25, appearance_channels=1, hidden_dim=16)),
        SharedStructuralField(StructuralFieldConfig(evidence_dim=52, hidden_width=16, hidden_layers=2)),
    )


def test_product_optimizer_updates_encoder_head_and_field() -> None:
    encoder, head, field = _modules()
    optimizer, _ = _make_optimizer(encoder, head, field, 1e-3)
    expected = {id(parameter) for module in (encoder, head, field) for parameter in module.parameters()}
    if hasattr(optimizer, "param_groups"):
        actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    else:
        actual = {id(parameter) for parameter in optimizer.parameters}
    assert actual == expected


def test_product_optimizer_learning_rate_is_explicit_for_native_backend() -> None:
    encoder, head, field = _modules()
    optimizer, _ = _make_optimizer(encoder, head, field, 1e-3)
    assert _optimizer_learning_rate(optimizer) == pytest.approx(1e-3)


def test_r4_progress_checkpoint_round_trip_excludes_target_payload(tmp_path) -> None:
    bundle = _bundle()
    encoder, head, field = _modules()
    optimizer, _ = _make_optimizer(encoder, head, field, 1e-3)
    config_hash = _digest("config")
    split_hash = _digest("split")
    propagation = PropagationConfig(variant="p0", rounds=0)
    for parameter in tuple(encoder.parameters()) + tuple(head.parameters()) + tuple(field.parameters()):
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    path = tmp_path / "progress_checkpoint.pt"
    _save_r4_progress_checkpoint(
        path=path,
        bundle=bundle,
        config_hash=config_hash,
        split_hash=split_hash,
        propagation=propagation,
        steps=3,
        completed_steps=1,
        reports=[{"step": 1}],
        encoder=encoder,
        gaussian_head=head,
        field=field,
        optimizer=optimizer,
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["target_payload_not_in_checkpoint"] is True
    assert "target" not in payload

    restored_encoder, restored_head, restored_field = _modules()
    restored_optimizer, _ = _make_optimizer(restored_encoder, restored_head, restored_field, 1e-3)
    completed, reports = _load_r4_progress_checkpoint(
        path=path,
        bundle=bundle,
        config_hash=config_hash,
        split_hash=split_hash,
        propagation=propagation,
        steps=3,
        encoder=restored_encoder,
        gaussian_head=restored_head,
        field=restored_field,
        optimizer=restored_optimizer,
    )
    assert completed == 1
    assert reports == [{"step": 1}]
    assert all(torch.equal(value, restored_encoder.state_dict()[name]) for name, value in encoder.state_dict().items())
    assert all(torch.equal(value, restored_head.state_dict()[name]) for name, value in head.state_dict().items())
    assert all(torch.equal(value, restored_field.state_dict()[name]) for name, value in field.state_dict().items())
    if hasattr(optimizer, "state"):
        assert len(optimizer.state) == len(restored_optimizer.state) > 0
    else:
        assert optimizer.step_index == restored_optimizer.step_index == 1


def test_global_model_promotion_round_trip_excludes_patient_and_target_payloads(tmp_path) -> None:
    encoder, head, field = _modules()
    optimizer, _ = _make_optimizer(encoder, head, field, 1e-3)
    for parameter in tuple(encoder.parameters()) + tuple(head.parameters()) + tuple(field.parameters()):
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    split_hash = _digest("split")
    config = {"training": {"steps": 3}, "episode_split": "train", "split_hash": split_hash}
    validation_config = {"training": {"steps": 1}, "episode_split": "validation", "split_hash": _digest("other-split")}
    assert _global_model_binding_hash(config) == _global_model_binding_hash(validation_config)
    checkpoint = tmp_path / "patient_checkpoint.pt"
    torch.save(
        {
            "schema": "smagm-brats21-r4-checkpoint-v1",
            "config_hash": _digest("patient-config"),
            "split_hash": split_hash,
            "target_payload_not_in_checkpoint": True,
            "global_model_eligible": True,
            "training_updates_applied": True,
            "model_binding_hash": _global_model_binding_hash(config),
            "encoder": {name: value.detach().cpu() for name, value in encoder.state_dict().items()},
            "gaussian_head": {name: value.detach().cpu() for name, value in head.state_dict().items()},
            "field": {name: value.detach().cpu() for name, value in field.state_dict().items()},
            "optimizer": optimizer.state_dict(),
        },
        checkpoint,
    )
    training_summary = tmp_path / "summary.json"
    training_summary.write_text(
        json.dumps({
            "schema": "smagm-brats21-real-smoke-v1",
            "split_hash": split_hash,
            "manifest_hash": _digest("manifest"),
            "assignment_hash": _digest("assignment"),
            "training_updates_applied": True,
            "e2_r4_p1": {"checkpoint": str(checkpoint), "training_updates_applied": True},
        }),
        encoding="utf-8",
    )
    global_checkpoint = tmp_path / "global_model_checkpoint.pt"
    promoted = _promote_global_model_checkpoint(
        training_summary,
        global_checkpoint,
        cohort_hash=_digest("cohort"),
        split_hash=split_hash,
        source_patient_pseudonym="patient-safe",
        global_update_index=1,
    )
    assert promoted["global_update_index"] == 1
    payload = torch.load(global_checkpoint, map_location="cpu", weights_only=True)
    assert payload["schema"] == "smagm-brats21-global-training-checkpoint-v1"
    assert payload["cohort_split_hash"] == split_hash
    assert payload["target_payload_not_in_checkpoint"] is True
    assert payload["patient_state_not_in_checkpoint"] is True
    assert "target" not in payload and "patient_state" not in payload
    assert _global_checkpoint_source_patient(global_checkpoint) == "patient-safe"

    restored_encoder, restored_head, restored_field = _modules()
    restored_optimizer, _ = _make_optimizer(restored_encoder, restored_head, restored_field, 1e-3)
    loaded = _load_global_model_checkpoint(
        global_checkpoint,
        config=config,
        split_hash=split_hash,
        encoder=restored_encoder,
        gaussian_head=restored_head,
        field=restored_field,
        optimizer=restored_optimizer,
    )
    assert loaded["global_update_index"] == 1
    assert all(torch.equal(value, restored_encoder.state_dict()[name]) for name, value in encoder.state_dict().items())
    assert all(torch.equal(value, restored_head.state_dict()[name]) for name, value in head.state_dict().items())
    assert all(torch.equal(value, restored_field.state_dict()[name]) for name, value in field.state_dict().items())


def test_global_binding_ignores_nested_episode_step_budget() -> None:
    cohort_split_hash = _digest("cohort-split")
    train_config = {
        "training": {
            "cohort_split_hash": cohort_split_hash,
            "split_hash": cohort_split_hash,
            "training": {"steps": 100},
        },
        "episode_split": "train",
        "split_hash": cohort_split_hash,
    }
    validation_config = {
        "training": {
            "cohort_split_hash": cohort_split_hash,
            "split_hash": cohort_split_hash,
            "training": {"steps": 1},
        },
        "episode_split": "validation",
        "split_hash": cohort_split_hash,
    }
    assert _global_model_binding_hash(train_config) == _global_model_binding_hash(validation_config)


def test_global_promotion_journal_reconciles_only_exact_successor(tmp_path, monkeypatch) -> None:
    cohort_hash = _digest("cohort")
    split_hash = _digest("split")
    config_hash = _digest("config")
    manifest_hash = _digest("manifest")
    assignment_hash = _digest("assignment")
    model_binding_hash = _digest("model-binding")
    summary_path = tmp_path / "patient-summary.json"
    summary_path.write_text(
        json.dumps({"manifest_hash": manifest_hash, "assignment_hash": assignment_hash}),
        encoding="utf-8",
    )
    global_path = tmp_path / "global.pt"
    torch.save(
        {
            "schema": "smagm-brats21-global-training-checkpoint-v1",
            "cohort_hash": cohort_hash,
            "split_hash": split_hash,
            "cohort_split_hash": split_hash,
            "global_update_index": 1,
            "source_patient_pseudonym": "patient-safe",
            "source_manifest_hash": manifest_hash,
            "source_assignment_hash": assignment_hash,
            "model_binding_hash": model_binding_hash,
            "target_payload_not_in_checkpoint": True,
            "patient_state_not_in_checkpoint": True,
        },
        global_path,
    )
    monkeypatch.setattr("smagm.cli.brats21_product._training_summary_complete", lambda *args, **kwargs: True)
    result = {"patient_pseudonym": "patient-safe", "summary": str(summary_path)}
    state = {
        "global_update_count": 0,
        "global_checkpoint_sha256": None,
        "completed": {},
        "pending_promotion": {
            "global_update_index": 1,
            "previous_global_update_index": 0,
            "previous_global_checkpoint_sha256": None,
            "cohort_hash": cohort_hash,
            "split_hash": split_hash,
            "config_hash": config_hash,
            "completion_key": "train:patient-safe",
            "training_summary": str(summary_path),
            "result": result,
            "patient_pseudonym": "patient-safe",
            "source_patient_pseudonym": "patient-safe",
            "propagation_variant": "p1",
            "source_manifest_hash": manifest_hash,
            "source_assignment_hash": assignment_hash,
            "model_binding_hash": model_binding_hash,
        },
    }
    update_index, checkpoint_hash = _reconcile_global_promotion_journal(
        state, global_path, cohort_hash=cohort_hash, split_hash=split_hash, config_hash=config_hash,
    )
    assert update_index == 1
    assert checkpoint_hash == hashlib.sha256(global_path.read_bytes()).hexdigest()
    assert "pending_promotion" not in state
    assert state["completed"]["train:patient-safe"] == result

    lone_state = {"global_update_count": 0, "global_checkpoint_sha256": None, "completed": {}}
    with pytest.raises(ValueError, match="exact next journaled"):
        _reconcile_global_promotion_journal(
            lone_state, global_path, cohort_hash=cohort_hash, split_hash=split_hash, config_hash=config_hash,
        )


def test_full_source_grid_uses_affine_and_declared_tensor_axis_order() -> None:
    grid = _source_grid_from_geometry(
        {
            "shape_xyz": [6, 8, 5],
            "affine": [
                [2.0, 0.0, 0.0, 10.0],
                [0.0, 3.0, 0.0, -4.0],
                [0.0, 0.0, 4.0, 2.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        inplane_stride_vu=(2, 4),
        preprocessing_hash="a" * 64,
        modality_id="flair",
    )
    assert grid.shape_dhw == (5, 3, 2)
    assert grid.index_to_ras_mm[0][0] == 0.0
    assert grid.index_to_ras_mm[1][0] == 12.0
    assert grid.index_to_ras_mm[0][1] == 4.0
    assert grid.index_to_ras_mm[1][1] == 0.0
    assert grid.world_from_dhw(3, 2, 1) == (18.0, 8.0, 14.0)


def test_propagation_bounds_cover_physical_source_volume_from_affine() -> None:
    lower, upper = _source_volume_bounds_from_geometry(
        {
            "shape_xyz": [3, 4, 5],
            "affine": [
                [0.0, -2.0, 0.0, 10.0],
                [1.0, 0.0, 0.0, -4.0],
                [0.0, 0.0, 1.5, 2.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    )
    assert tuple(lower.tolist()) == pytest.approx((4.0, -4.0, 2.0))
    assert tuple(upper.tolist()) == pytest.approx((10.0, -2.0, 8.0))


def test_incomplete_patient_summary_is_not_accepted_as_success(tmp_path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({
            "schema": "smagm-brats21-real-smoke-v1",
            "scientific_pass_recorded": False,
            "target_reveal_barrier_verified": True,
            "e2_r4_p1": {"checkpoint": str(tmp_path / "missing.pt")},
        }),
        encoding="utf-8",
    )
    assert not _training_summary_complete(summary)


def test_patient_completion_requires_the_requested_propagation_variant(tmp_path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({
            "schema": "smagm-brats21-real-smoke-v1",
            "scientific_pass_recorded": False,
            "target_reveal_barrier_verified": True,
            "e2_r4_p0": {},
        }),
        encoding="utf-8",
    )
    assert not _training_summary_complete(summary, expected_propagation_variant="p1")


def test_invalid_training_marker_quarantine_preserves_resumable_directory(tmp_path) -> None:
    training = tmp_path / "training"
    training.mkdir()
    marker = training / "summary.json"
    marker.write_text("invalid", encoding="utf-8")
    progress = training / "e2_r4_p1" / "progress_checkpoint.pt"
    progress.parent.mkdir()
    progress.write_bytes(b"progress")
    quarantined = _quarantine_partial_file(marker)
    assert not marker.exists()
    assert quarantined.read_text(encoding="utf-8") == "invalid"
    assert progress.is_file()


def test_full_product_runtime_config_declares_dataset_evaluation_and_output_contract() -> None:
    config, _ = _load_product_config(Path("configs/experiments/brats21_product_full.json"))
    runtime = _runtime_config(
        config,
        propagation_variant="p1",
        steps=100,
        wandb_mode="online",
        split_name="train",
    )
    assert runtime["dataset_root"] == "data/preprocessed/BraTS21"
    assert runtime["evaluation"]["metric_config"]["data_range"] == 1.0
    assert runtime["output_paths"]["global_checkpoint"] == "global_model_checkpoint.pt"
    assert runtime["output_paths"]["refuse_overwrite_success"] is True
    assert runtime["output_paths"]["patient_metrics"] == "patient_metrics.csv"
    assert runtime["output_paths"]["aggregate_metrics"] == "aggregate_metrics.json"
    assert runtime["experiment_name"] == "brats21-structure-constrained-full-e2-r4-p1"
    assert config["product"]["disk_policy"] == "advisory"
    assert config["product"]["validation"]["cadence"] == "final"
    assert config["product"]["validation"]["selection_policy"] == "fixed_geometry_only_no_checkpoint_selection"
    assert config["product"]["max_global_steps"] == 100
    propagation = config["training"]["propagation"]
    assert propagation["propagation_reserved_budget"] > 0
    assert config["training"]["training"]["gaussian_head_input_adapter"] == "anchor_evidence_projector"


def test_full_product_is_the_only_launch_mode() -> None:
    config, _ = _load_product_config(Path("configs/experiments/brats21_product_full.json"))
    assert config["product"]["stage"] == "full"
    assert "readiness_prerequisites" not in config["product"]


def test_product_metric_reports_are_patient_macro_and_support_explicit(tmp_path: Path) -> None:
    training = tmp_path / "patients" / "patient-safe" / "training"
    training.mkdir(parents=True)
    summary = {
        "schema": "smagm-brats21-real-smoke-v1",
        "patient_pseudonymous_id": "patient-safe",
        "split": "train",
        "r0": {"loss": 0.4, "runtime_seconds": 2.0},
        "e2_r4_p1": {
            "final_evaluation_loss": 0.2,
            "runtime_seconds": 3.0,
            "anchor_count": 4,
            "structural_gaussian_count": 4,
            "volumetric_gaussian_count": 4,
            "propagation_child_count": 2,
        },
        "evaluations": {
            "r0": {"metrics": [{"evaluable_voxels": 10, "supported_voxels": 8, "unsupported_voxels": 2, "mae": 0.4}]},
            "e2_r4_p1": {"metrics": [{
                "evaluable_voxels": 10, "supported_voxels": 10, "unsupported_voxels": 0,
                "mae": 0.2, "complete_mae": 0.2, "complete_metric_status": "COMPUTED_ALL_DECLARED_TARGET_VOXELS",
                "metric_scope": "support_conditioned", "data_range": 1.0, "roi_contrast_error": 0.03,
            }]},
        },
        "resource_metrics": {"cache_size_bytes": 12, "checkpoint_size_bytes": 34},
    }
    summary_path = training / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    state = {"completed": {"train:patient-safe": {"summary": str(summary_path)}}}
    result = _write_product_metric_reports(
        state=state,
        output_dir=tmp_path,
        output_paths={"patient_metrics": "patient_metrics.csv", "aggregate_metrics": "aggregate_metrics.json"},
        propagation_variant="p1",
    )
    assert result["row_count"] == 2
    aggregate = json.loads((tmp_path / "aggregate_metrics.json").read_text(encoding="utf-8"))
    assert aggregate["variants"]["e2_r4_p1"]["pooled_support"]["supported_fraction"] == pytest.approx(1.0)
    assert aggregate["variants"]["r0"]["pooled_support"]["supported_fraction"] == pytest.approx(0.8)
    csv_text = (tmp_path / "patient_metrics.csv").read_text(encoding="utf-8")
    assert "patient-safe" in csv_text
    assert "complete_mae" in csv_text and "complete_metric_status" in csv_text
    assert aggregate["variants"]["e2_r4_p1"]["metrics"]["complete_mae"]["mean"] == pytest.approx(0.2)
    interval = aggregate["variants"]["e2_r4_p1"]["metrics"]["complete_mae"]["bootstrap_confidence_interval"]
    assert interval["method"] == "deterministic_patient_bootstrap_percentile"
    assert interval["low"] == pytest.approx(0.2)
    assert interval["high"] == pytest.approx(0.2)
    assert aggregate["aggregation"]["missing_policy"] == "omit_non_finite_with_count"


def test_product_source_hashes_require_full_inventory_binding(tmp_path) -> None:
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps({
            "schema": "smagm-brats21-dataset-inventory-v2",
            "cohort_hash": "c" * 64,
            "validation_scope": {"all_patients": "headers-only validation and source hashes"},
            "valid_patients": [],
        }),
        encoding="utf-8",
    )
    config = {"data": {"inventory_report": str(inventory), "cohort_hash": "c" * 64}}
    with pytest.raises(ValueError, match="full finite-value validation"):
        _source_hashes_for_patient(config, "BraTS2021_00000")
