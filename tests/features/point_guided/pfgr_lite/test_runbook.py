from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from smagm.cli.pfgr_lite import _parser, main


def _runbook_block(needle: str) -> str:
    text = Path("RUNBOOK_PFGR_LITE.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```bash\s*\n(.*?)```", text, re.DOTALL)
    return next(block for block in blocks if needle in block)


def _synthetic_bank(tmp_path: Path, run_prefix: str) -> tuple[Path, Path, Path]:
    """Create one actual bounded producer/bank fixture for runbook helper tests."""

    output = tmp_path / "runs"
    assert main(
        [
            "smoke",
            "--synthetic",
            "--config",
            "configs/pfgr_lite/synthetic.json",
            "--output-root",
            str(output),
            "--run-name",
            f"{run_prefix}-producer",
            "--device",
            "cpu",
            "--no-amp",
            "--max-subjects",
            "1",
            "--max-steps",
            "1",
        ]
    ) == 0
    checkpoint = output / f"{run_prefix}-producer" / "inference.pt"
    assert checkpoint.is_file()
    assert main(
        [
            "bank-build",
            "--synthetic",
            "--config",
            "configs/pfgr_lite/synthetic.json",
            "--output-root",
            str(output),
            "--run-name",
            f"{run_prefix}-bank",
            "--device",
            "cpu",
            "--no-amp",
            "--checkpoint",
            str(checkpoint),
            "--max-subjects",
            "1",
            "--max-states",
            "1",
            "--candidate-count",
            "2",
            "--query-count",
            "8",
            "--teacher-mode",
            "iid_fixed_q",
        ]
    ) == 0
    bank = output / f"{run_prefix}-bank" / "s2" / "bank" / "index.json"
    assert bank.is_file()
    return output, checkpoint, bank


def test_runbook_check_validates_r0_r10_commands_and_configs(tmp_path: Path) -> None:
    output = tmp_path / "runs"
    assert main(
        [
            "runbook-check",
            "--runbook",
            "RUNBOOK_PFGR_LITE.md",
            "--config-dir",
            "configs/pfgr_lite",
            "--output-root",
            str(output),
            "--run-name",
            "runbook",
        ]
    ) == 0
    receipt = json.loads((output / "runbook" / "receipt.json").read_text(encoding="utf-8"))
    report = receipt["metrics"]
    assert receipt["status"] == "SOFTWARE_PASS"
    assert report["missing_sections"] == []
    assert report["missing_commands"] == []
    assert report["shell_errors"] == []
    assert report["config_errors"] == []


def test_runbook_check_fails_closed_for_missing_section_and_command(tmp_path: Path) -> None:
    runbook = tmp_path / "bad.md"
    runbook.write_text("# R0\n```bash\necho ok\n```\n", encoding="utf-8")
    output = tmp_path / "runs"
    assert main(
        [
            "runbook-check",
            "--runbook",
            str(runbook),
            "--config-dir",
            "configs/pfgr_lite",
            "--output-root",
            str(output),
            "--run-name",
            "bad",
        ]
    ) == 1
    receipt = json.loads((output / "bad" / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "SOFTWARE_FAIL"
    assert receipt["metrics"]["missing_sections"]
    assert receipt["metrics"]["missing_commands"]


def test_runbook_has_bounded_synthetic_execution_not_only_shell_validation(tmp_path: Path) -> None:
    """The documented smoke command executes the real bounded stage service."""

    output = tmp_path / "runs"
    assert main(
        [
            "smoke",
            "--synthetic",
            "--config",
            "configs/pfgr_lite/synthetic.json",
            "--output-root",
            str(output),
            "--run-name",
            "r1-execution",
            "--device",
            "cpu",
            "--no-amp",
            "--max-subjects",
            "2",
            "--max-steps",
            "1",
        ]
    ) == 0
    run_dir = output / "r1-execution"
    receipt = json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "SOFTWARE_PASS"
    assert receipt["scientific_status"] == "NOT_EVALUATED"
    assert (run_dir / "inference.pt").is_file()
    assert (run_dir / "resume.pt").is_file()


def test_runbook_bank_index_matches_lowercase_cli_artifact_layout(tmp_path: Path) -> None:
    """The documented R5 variable must point at the path the stage service emits."""

    runbook = Path("RUNBOOK_PFGR_LITE.md").read_text(encoding="utf-8")
    assert 'PFGR_BANK_INDEX="$PFGR_R5_DIR/s2/bank/index.json"' in runbook
    output = tmp_path / "runs"
    assert main(
        [
            "smoke",
            "--synthetic",
            "--config",
            "configs/pfgr_lite/synthetic.json",
            "--output-root",
            str(output),
            "--run-name",
            "r5-producer",
            "--device",
            "cpu",
            "--no-amp",
            "--max-subjects",
            "1",
            "--max-steps",
            "1",
        ]
    ) == 0
    checkpoint = output / "r5-producer" / "inference.pt"
    assert checkpoint.is_file()
    assert main(
        [
            "bank-build",
            "--synthetic",
            "--config",
            "configs/pfgr_lite/synthetic.json",
            "--output-root",
            str(output),
            "--run-name",
            "r5-bank",
            "--device",
            "cpu",
            "--no-amp",
            "--checkpoint",
            str(checkpoint),
            "--max-subjects",
            "1",
            "--max-states",
            "1",
            "--candidate-count",
            "2",
            "--query-count",
            "8",
            "--teacher-mode",
            "iid_fixed_q",
        ]
    ) == 0
    emitted = output / "r5-bank" / "s2" / "bank" / "index.json"
    assert emitted.is_file()


def test_runbook_receipt_writer_uses_actual_dry_context_and_refuses_overwrite(tmp_path: Path) -> None:
    output, checkpoint, bank = _synthetic_bank(tmp_path, "review")
    value_checkpoint = output / "review-value" / "value.pt"
    assert main(
        [
            "value-fit",
            "--synthetic",
            "--config",
            "configs/pfgr_lite/synthetic.json",
            "--output-root",
            str(output),
            "--run-name",
            "review-value",
            "--device",
            "cpu",
            "--no-amp",
            "--checkpoint",
            str(checkpoint),
            "--bank-index",
            str(bank),
            "--value-input",
            "126",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--max-steps",
            "1",
        ]
    ) == 0
    assert value_checkpoint.is_file()
    assert main(
        [
            "calibrate",
            "--synthetic",
            "--config",
            "configs/pfgr_lite/synthetic.json",
            "--output-root",
            str(output),
            "--run-name",
            "review-request",
            "--device",
            "cpu",
            "--no-amp",
            "--checkpoint",
            str(checkpoint),
            "--value-checkpoint",
            str(value_checkpoint),
            "--teacher-mode",
            "exact_footprint",
            "--max-subjects",
            "1",
            "--dry-manifest",
        ]
    ) == 0
    context = output / "review-request" / "review_context.json"
    receipt = output / "review-request" / "engineering-review.json"
    assert context.is_file()
    writer = _runbook_block("PFGR_REVIEW_CONTEXT")
    environment = os.environ.copy()
    environment.update(
        {
            "POINT_GUIDED_PYTHON": sys.executable,
            "PFGR_REVIEW_CONTEXT": str(context),
            "PFGR_REVIEW_RECEIPT": str(receipt),
            "PFGR_REVIEWER": "Synthetic Reviewer",
            "PFGR_REVIEW_DECISION": "ENGINEERING_DIAGNOSTIC",
        }
    )
    first = subprocess.run(["bash"], input=writer, text=True, capture_output=True, env=environment, check=False)
    assert first.returncode == 0, first.stderr
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "pfgr-lite-review-receipt-v1"
    assert payload["decision"] == "ENGINEERING_DIAGNOSTIC"
    context_payload = json.loads(context.read_text(encoding="utf-8"))
    assert payload["config_hash"] == context_payload["config_hash"]
    assert payload["cohort_hash"] == context_payload["cohort_hash"]
    assert payload["artifacts"] == context_payload["expected_artifacts"]
    second = subprocess.run(["bash"], input=writer, text=True, capture_output=True, env=environment, check=False)
    assert second.returncode != 0
    assert "refuse overwrite reviewer receipt" in second.stderr


def test_runbook_same_bank_v_join_executes_against_three_actual_variants(tmp_path: Path) -> None:
    output, checkpoint, bank = _synthetic_bank(tmp_path, "vjoin")
    for variant in (126, 270, 366):
        fit_name = f"R6-v{variant}-TEST"
        assert main(
            [
                "value-fit",
                "--synthetic",
                "--config",
                "configs/pfgr_lite/synthetic.json",
                "--output-root",
                str(output),
                "--run-name",
                fit_name,
                "--device",
                "cpu",
                "--no-amp",
                "--checkpoint",
                str(checkpoint),
                "--bank-index",
                str(bank),
                "--value-input",
                str(variant),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--max-steps",
                "1",
            ]
        ) == 0
        assert main(
            [
                "value-evaluate",
                "--synthetic",
                "--config",
                "configs/pfgr_lite/synthetic.json",
                "--output-root",
                str(output),
                "--run-name",
                f"R6-eval-v{variant}-TEST",
                "--device",
                "cpu",
                "--no-amp",
                "--checkpoint",
                str(checkpoint),
                "--bank-index",
                str(bank),
                "--value-checkpoint",
                str(output / fit_name / "value.pt"),
                "--batch-size",
                "2",
            ]
        ) == 0
    full_block = _runbook_block("PFGR_V126_PAIRS")
    join_block = full_block[full_block.index("PFGR_V126_PAIRS=") :]
    script = 'set -euo pipefail\nrequire_file(){ test -f "$1" || exit 2; }\n' + join_block
    environment = os.environ.copy()
    environment.update(
        {
            "POINT_GUIDED_PYTHON": sys.executable,
            "PYTHONPATH": str(Path.cwd() / "src") + os.pathsep + environment.get("PYTHONPATH", ""),
            "OUTPUT_ROOT": str(output),
            "PFGR_RUN_ID": "TEST",
        }
    )
    completed = subprocess.run(["bash"], input=script, text=True, capture_output=True, env=environment, check=False)
    assert completed.returncode == 0, completed.stderr
    joined = output / "R6-v-paired-TEST" / "value_evaluate_pairs.json"
    payload = json.loads(joined.read_text(encoding="utf-8"))
    assert payload["join_status"] == "PASS"
    assert payload["input_variants"] == [126, 270, 366]
    assert payload["row_count"] > 0


def test_runbook_r2_synthetic_fallback_uses_synthetic_config_and_artifact(tmp_path: Path) -> None:
    output = tmp_path / "runs"
    assert main(
        [
            "smoke",
            "--synthetic",
            "--config",
            "configs/pfgr_lite/synthetic.json",
            "--output-root",
            str(output),
            "--run-name",
            "R1-synthetic-TEST",
            "--device",
            "cpu",
            "--no-amp",
            "--max-subjects",
            "1",
            "--max-steps",
            "1",
        ]
    ) == 0
    full_block = _runbook_block("PFGR_R2_CHECKPOINT")
    environment = os.environ.copy()
    environment.update(
        {
            "POINT_GUIDED_PYTHON": sys.executable,
            "PYTHONPATH": str(Path.cwd() / "src") + os.pathsep + environment.get("PYTHONPATH", ""),
            "OUTPUT_ROOT": str(output),
            "PFGR_RUN_ID": "TEST",
            "REPO_ROOT": str(Path.cwd()),
            "PFGR_R1_REAL_DIR": str(output / "R1-real-TEST"),
            "PFGR_R1_SYNTH_DIR": str(output / "R1-synthetic-TEST"),
            "PFGR_ROLES": str(output / "roles.json"),
        }
    )
    script = 'set -euo pipefail\nrequire_artifact(){ test -e "$1" || exit 2; }\n' + full_block
    completed = subprocess.run(["bash"], input=script, text=True, capture_output=True, env=environment, check=False)
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads((output / "R2-TEST" / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["capability"] == "engineering_only"
    assert receipt["scientific_status"] == "NOT_EVALUATED"
    assert (output / "R2-TEST" / "benchmark.json").is_file()


def test_runbook_literal_flags_exist_on_the_matching_cli_parser() -> None:
    """Resolve shell continuations and catch stale flags without running data work."""

    text = Path("RUNBOOK_PFGR_LITE.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```(?:bash|sh|shell)\s*\n(.*?)```", text, re.IGNORECASE | re.DOTALL)
    parser = _parser()
    subparsers = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    seen_commands: set[str] = set()
    for block in blocks:
        logical = block.replace("\\\n", " ")
        for line in logical.splitlines():
            match = re.search(r"smagm\.cli\.pfgr_lite\s+([a-z][a-z-]*)\b(.*)", line)
            if match is None:
                continue
            command = match.group(1)
            command_parser = subparsers.choices[command]
            known_flags = {
                flag
                for action in command_parser._actions
                for flag in action.option_strings
            }
            flags = set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", match.group(2)))
            missing = sorted(flags - known_flags)
            assert not missing, f"runbook {command} has stale CLI flags: {missing}"
            seen_commands.add(command)
    assert seen_commands >= set(subparsers.choices)
