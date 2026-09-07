from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from smagm.cli.pfgr_lite import CLIError, COMMANDS, _config_for_command, _parser, _resolve_cached_value_config, _validate_resolved_pfgr_config, main
from smagm.features.point_guided.pfgr_lite.config import PFGRLiteConfig


def test_cli_registers_frozen_command_family_and_strict_unknown_flags() -> None:
    parser = _parser()
    for command in COMMANDS:
        with pytest.raises(SystemExit) as outcome:
            parser.parse_args([command, "--help"])
        assert outcome.value.code == 0
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "--scenario", "adaptive", "--budget", "4", "--unknown", "1"])


def test_synthetic_dry_manifest_is_honest_and_non_mutating(tmp_path: Path) -> None:
    output = tmp_path / "runs"
    assert main(
        [
            "preflight",
            "--synthetic",
            "--config",
            "configs/pfgr_lite/synthetic.json",
            "--output-root",
            str(output),
            "--run-name",
            "preflight",
            "--dry-manifest",
        ]
    ) == 0
    run = output / "preflight"
    manifest = json.loads((run / "dry_manifest.json").read_text(encoding="utf-8"))
    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "DRY_MANIFEST"
    assert manifest["scientific_status"] == "NOT_EVALUATED"
    assert receipt["capability"] == "engineering_only"
    assert not (run / "weights.pt").exists()
    assert main(
        [
            "preflight",
            "--synthetic",
            "--config",
            "configs/pfgr_lite/synthetic.json",
            "--output-root",
            str(output),
            "--run-name",
            "preflight",
            "--dry-manifest",
        ]
    ) == 1


def test_static_base_flag_is_bound_into_strict_execution_config() -> None:
    parser = _parser()
    args = parser.parse_args(["static-train", "--config", "configs/pfgr_lite/main.json", "--base", "b1"])
    config, details = _config_for_command(args, stage="S0")
    assert config.static.variant == "b1_multiscale_v1"
    assert details["execution"]["pfgr_config"]["static"]["variant"] == "b1_multiscale_v1"


def test_cached_value_config_binds_only_unresolved_normalization() -> None:
    requested = PFGRLiteConfig(engineering_only=True)
    cached = replace(requested, observation_normalization="measured-recipe-hash")
    bundle = SimpleNamespace(config={"pfgr_config": cached.as_dict()})
    bound = _resolve_cached_value_config(requested, bundle)
    assert bound.observation_normalization == "measured-recipe-hash"

    explicit = replace(requested, observation_normalization="caller-supplied-policy")
    with pytest.raises(CLIError, match="normalization identity"):
        _resolve_cached_value_config(explicit, bundle)


def test_factory_normalization_resolution_rejects_explicit_wrong_policy() -> None:
    requested = PFGRLiteConfig(engineering_only=True)
    resolved = replace(requested, observation_normalization="measured-recipe-hash")
    _validate_resolved_pfgr_config(requested, resolved)
    with pytest.raises(CLIError, match="unresolved policy label"):
        _validate_resolved_pfgr_config(
            replace(requested, observation_normalization="caller-supplied-policy"),
            resolved,
        )


def test_real_dry_manifest_reports_missing_inputs_without_claiming_success(tmp_path: Path) -> None:
    output = tmp_path / "runs"
    assert main(
        [
            "evaluate",
            "--config",
            "configs/pfgr_lite/main.json",
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--scenario",
            "adaptive",
            "--budget",
            "4",
            "--output-root",
            str(output),
            "--run-name",
            "missing",
            "--dry-manifest",
        ]
    ) == 0
    payload = json.loads((output / "missing" / "dry_manifest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKED"
    assert payload["scientific_status"] == "NOT_EVALUATED"
    assert payload["missing_inputs"]


def test_every_subcommand_help_is_available_in_a_fresh_parser() -> None:
    parser = _parser()
    # argparse's help path exits cleanly; this loop also guards accidental
    # imports of teacher/oracle dependencies during top-level parser creation.
    for command in COMMANDS:
        with pytest.raises(SystemExit) as outcome:
            parser.parse_args([command, "--help"])
        assert outcome.value.code == 0


def test_existing_run_directory_is_never_mutated_on_reservation_failure(tmp_path: Path) -> None:
    output = tmp_path / "runs"
    existing = output / "owned-by-teammate"
    existing.mkdir(parents=True)
    sentinel = existing / "sentinel.txt"
    sentinel.write_bytes(b"leave this byte-for-byte unchanged\n")
    assert main([
        "preflight",
        "--synthetic",
        "--config",
        "configs/pfgr_lite/synthetic.json",
        "--output-root",
        str(output),
        "--run-name",
        existing.name,
    ]) == 1
    assert sentinel.read_bytes() == b"leave this byte-for-byte unchanged\n"
    assert not (existing / "receipt.json").exists()


def test_wandb_receipt_logs_finite_metrics_and_counts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Summary(dict):
        pass

    class FakeRun:
        id = "offline-test-id"
        url = None

        def __init__(self) -> None:
            self.logged: list[dict[str, object]] = []
            self.summary = Summary()

        def log(self, payload: dict[str, object]) -> None:
            self.logged.append(dict(payload))

        def finish(self) -> None:
            return None

    run = FakeRun()
    fake = SimpleNamespace(init=lambda **_: run)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    output = tmp_path / "runs"
    assert main(
        [
            "preflight",
            "--synthetic",
            "--config",
            "configs/pfgr_lite/synthetic.json",
            "--output-root",
            str(output),
            "--run-name",
            "wandb",
            "--wandb",
        ]
    ) == 0
    assert run.logged
    assert "counts.target_reads" in run.logged[0]
    assert "metrics" not in run.logged[0]  # flattened numeric fields only
    assert run.summary
