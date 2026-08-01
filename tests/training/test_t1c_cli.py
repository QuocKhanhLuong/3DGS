from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from smagm.cli.train import load_resolved_config, run_synthetic_training
from smagm.training.sampling import MatchedExperimentIdentity


ROOT = Path(__file__).resolve().parents[2]


def test_config_file_drives_synthetic_cli_and_writes_resolved_artifacts(tmp_path) -> None:
    config_path = ROOT / "configs" / "experiments" / "t1c_synthetic.json"
    output = tmp_path / "run"
    resolved, resolved_hash = load_resolved_config(
        config_path,
        variant="e1",
        seed=43,
        steps=2,
        output_dir=output,
    )
    report = run_synthetic_training(
        config=resolved,
        resolved_config_hash=resolved_hash,
        output_dir=output,
        repository_root=ROOT,
        allow_dirty=True,
    )
    persisted = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
    assert persisted["selected_variant"] == "e1"
    assert persisted["seed"] == 43
    assert persisted["training"]["steps"] == 2
    assert report["variant"] == "e1"
    assert report["checkpoint_selection"] == {
        "assignment_hash": report["assignment_hash"],
        "optimizer_step_index": 2,
        "schedule_cursor": 2,
        "training_step_budget": 2,
        "rule": "last_eligible_optimizer_step_never_audit",
    }
    assert report["encoder_forward_passes"] == 2
    expected_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    assert report["reproducible"] is (not expected_dirty)
    assert {path.name for path in output.iterdir()} == {
        "artifact_manifest.json",
        "checkpoint.pt",
        "episode_ledger.json",
        "metrics.jsonl",
        "provenance.json",
        "resolved_config.json",
        "summary.json",
    }
    metrics = (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(metrics) == 2
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["dirty"] is expected_dirty
    required = {
        "commit",
        "dirty",
        "config_hash",
        "manifest_hash",
        "split_registry_hash",
        "assignment_schedule_hash",
        "modality_mapping_hash",
        "preprocessing_policy_hash",
        "preprocessing_record_hash",
        "opened_file_ledger_hash",
        "dependency_manifest_hash",
        "artifact_manifest_hash",
        "encoder_variant",
        "encoder_config_hash",
        "encoder_state_hash",
        "gaussian_head_initialization_hash",
        "renderer_config_hash",
        "amplitude_gauge_hash",
        "frozen_patient_state_schema_version",
        "seed",
        "environment",
        "device",
        "parameter_count",
        "run_started_at",
        "run_ended_at",
        "artifact_hashes",
    }
    assert required <= set(provenance)
    assert {name for name, _ in provenance["artifact_hashes"]} == {
        "checkpoint.pt",
        "artifact_manifest.json",
        "episode_ledger.json",
        "metrics.jsonl",
        "resolved_config.json",
    }
    artifact_manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert artifact_manifest["schema"] == "smagm-artifact-manifest-v1"
    assert set(artifact_manifest["artifacts"]) == {
        "checkpoint.pt",
        "episode_ledger.json",
        "metrics.jsonl",
        "resolved_config.json",
    }
    assert report == json.loads((output / "summary.json").read_text(encoding="utf-8"))


def test_matched_protocol_hash_excludes_only_variant_and_output_location(tmp_path) -> None:
    config_path = ROOT / "configs" / "experiments" / "t1c_synthetic.json"
    resolved = [
        load_resolved_config(config_path, variant=variant, steps=2, output_dir=tmp_path / variant)[0]
        for variant in ("e0", "e1", "e2")
    ]
    assert len({item["matched_protocol_hash"] for item in resolved}) == 1
    assert len({item["selected_variant"] for item in resolved}) == 3


def test_ephemeral_quality_run_still_binds_nonpersistent_artifact_digests() -> None:
    config_path = ROOT / "configs" / "experiments" / "t1c_synthetic.json"
    resolved, resolved_hash = load_resolved_config(config_path, variant="e0")
    report = run_synthetic_training(
        config=resolved,
        resolved_config_hash=resolved_hash,
        repository_root=ROOT,
        allow_dirty=True,
    )
    assert set(report["artifact_digests"]) == {
        "ephemeral/artifact_manifest.json",
        "ephemeral/checkpoint.pt",
        "ephemeral/episode_ledger.json",
        "ephemeral/metrics.jsonl",
        "ephemeral/resolved_config.json",
    }


def test_matched_experiment_identity_freezes_nested_conditions() -> None:
    identity = MatchedExperimentIdentity.from_resolved_conditions(
        manifest_hash="a" * 64,
        split_registry_hash="b" * 64,
        assignment_schedule_hash="c" * 64,
        modality_mapping_hash="d" * 64,
        shared_conditions={"nested": {"channels": [16, 8, 1]}},
    )
    with pytest.raises(TypeError):
        identity.shared_conditions["nested"]["channels"] += (2,)  # type: ignore[index]


def test_matched_variant_runs_execute_the_same_downstream_protocol(tmp_path) -> None:
    config_path = ROOT / "configs" / "experiments" / "t1c_synthetic.json"
    reports = []
    for variant in ("e0", "e1", "e2"):
        output = tmp_path / variant
        resolved, resolved_hash = load_resolved_config(
            config_path, variant=variant, steps=2, output_dir=output
        )
        reports.append(
            run_synthetic_training(
                config=resolved,
                resolved_config_hash=resolved_hash,
                output_dir=output,
                repository_root=ROOT,
                allow_dirty=True,
            )
        )
    for field in (
        "assignment_schedule_hash",
        "head_initialization_hash",
        "manifest_hash",
        "matched_experiment_identity",
        "matched_protocol_hash",
        "split_registry_hash",
        "support_count",
        "support_topology_hash",
    ):
        assert len({report[field] for report in reports}) == 1
    assert len({json.dumps(report["checkpoint_selection"], sort_keys=True) for report in reports}) == 1
    assert reports[0]["checkpoint_selection"] == {
        "assignment_hash": reports[0]["assignment_hash"],
        "optimizer_step_index": 2,
        "schedule_cursor": 2,
        "training_step_budget": 2,
        "rule": "last_eligible_optimizer_step_never_audit",
    }
