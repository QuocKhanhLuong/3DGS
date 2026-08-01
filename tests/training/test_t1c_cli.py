from __future__ import annotations

import json
from pathlib import Path

from smagm.cli.train import load_resolved_config, run_synthetic_training


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
    assert report["reproducible"] is False  # test runs from a development-dirty checkout
    assert {path.name for path in output.iterdir()} == {
        "checkpoint.pt",
        "metrics.jsonl",
        "provenance.json",
        "resolved_config.json",
        "summary.json",
    }
    metrics = (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(metrics) == 2
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    required = {
        "commit",
        "dirty",
        "config_hash",
        "manifest_hash",
        "split_registry_hash",
        "assignment_schedule_hash",
        "modality_mapping_hash",
        "preprocessing_policy_hash",
        "encoder_variant",
        "encoder_config_hash",
        "encoder_state_hash",
        "gaussian_head_initialization_hash",
        "renderer_config_hash",
        "amplitude_gauge_hash",
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
        "metrics.jsonl",
        "resolved_config.json",
    }


def test_matched_protocol_hash_excludes_only_variant_and_output_location(tmp_path) -> None:
    config_path = ROOT / "configs" / "experiments" / "t1c_synthetic.json"
    resolved = [
        load_resolved_config(config_path, variant=variant, steps=2, output_dir=tmp_path / variant)[0]
        for variant in ("e0", "e1", "e2")
    ]
    assert len({item["matched_protocol_hash"] for item in resolved}) == 1
    assert len({item["selected_variant"] for item in resolved}) == 3
