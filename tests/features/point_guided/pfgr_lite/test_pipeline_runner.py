"""Runner software tests: fake compute, real files/argv/packaging, no training."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[4] / "scripts/run_pfgr_r4_evidence.py"
spec = importlib.util.spec_from_file_location("pfgr_pipeline_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


@pytest.fixture
def args(tmp_path):
    data = tmp_path / "BraTS space;$(not-a-shell)"
    data.mkdir()
    checkpoint = tmp_path / "MedicalNet ' $(not-a-command).pth"
    checkpoint.write_bytes(b"synthetic test bytes only")
    split = tmp_path / "split.json"
    split.write_text(
        '{"train": ["train-01"], "val": ["val-00", "val-01", "val-02", "val-03"], "test": []}'
    )
    return runner.parser().parse_args(
        [
            "--data-root",
            str(data),
            "--medicalnet-checkpoint",
            str(checkpoint),
            "--medicalnet-sha256",
            runner.sha256(checkpoint),
            "--split-file",
            str(split),
            "--run-root",
            str(tmp_path / "runs space"),
            "--run-name",
            "fresh",
            "--device",
            "cpu",
            "--profile",
            "provisional",
            "--static-epochs",
            "7",
            "--updater-epochs",
            "9",
            "--seed",
            "73",
            "--benchmark",
        ]
    )


def paired(subject="val-00"):
    return {
        "schema_version": "pfgr-lite-metrics-v1",
        "subject_id": subject,
        "before": {"mae": 0.4, "psnr": 15.0, "ssim": None, "masked_charbonnier": 0.401},
        "after": {"mae": 0.3, "psnr": 16.0, "ssim": None, "masked_charbonnier": 0.301},
        "improvement": {
            "mae": 0.1,
            "psnr": 1.0,
            "ssim": None,
            "masked_charbonnier": 0.1,
        },
    }


class FakeCompute:
    def __init__(self, *, fail=None, package_real=False, mismatch=False, missing=None):
        self.called = []
        self.fail = fail
        self.package_real = package_real
        self.mismatch = mismatch
        self.missing = missing

    def __call__(self, stage, logs, emit, timeout):
        self.called.append(stage)
        (logs / "stdout.txt").write_text("fake-compute fixture\n")
        (logs / "stderr.txt").write_text(
            "fixture failure\n" if stage.name == self.fail else ""
        )
        emit(
            "output",
            stage=stage.name,
            stream="stdout",
            path=str(logs / "stdout.txt"),
            offset_bytes=0,
            count_bytes=21,
        )
        root = Path(stage.run_dir)
        root.mkdir(parents=True)
        if stage.name == self.fail:
            return 7
        if stage.command == "package":
            if self.package_real:
                from smagm.features.point_guided.pfgr_lite.artifacts import (
                    package_evidence,
                )

                inputs = [
                    Path(stage.argv[i + 1])
                    for i, v in enumerate(stage.argv)
                    if v == "--run-dir"
                ]
                manifest = package_evidence(inputs, root / "evidence")
            else:
                (root / "evidence").mkdir()
                archive = root / "evidence/evidence.zip"
                with zipfile.ZipFile(archive, "w") as handle:
                    handle.writestr("fixture.txt", "software test")
                manifest = {
                    "archive": {
                        "path": "evidence.zip",
                        "sha256": runner.sha256(archive),
                    },
                    "included": [
                        {"source_path": name}
                        for name in (
                            "pipeline_manifest.json",
                            "pipeline_summary.json",
                            "pipeline_events.jsonl",
                            "pipeline_metrics.csv",
                        )
                    ],
                }
            runner.json_write(root / "manifest.json", manifest)
            runner.json_write(root / "receipt.json", {"archive": manifest["archive"]})
            return 0
        for item in stage.produces:
            path = Path(item)
            if path.name == self.missing:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}" if path.suffix == ".json" else "fixture checkpoint")
        if self.missing == "inference.pt":
            return 0
        runner.json_write(
            root / "receipt.json",
            {
                "counts": {"subjects": 4},
                "runtime_model": {"parameter_count": 100, "flops": None},
                "environment": {"cuda_available": False},
            },
        )
        if stage.command in {"smoke", "static-train", "updater-train"}:
            config_path = Path(stage.argv[stage.argv.index("--config") + 1])
            config = runner.json_read(config_path)
            config["stage_options"]["seed"] = int(
                stage.argv[stage.argv.index("--seed") + 1]
            )
            runner.json_write(root / "resolved_config.json", config)
            runner.json_write(
                root / "stage_runtime.json",
                {
                    "schema_version": "pfgr-lite-stage-runtime-v1",
                    "execution_config": config,
                    "cursor": {
                        "epoch": 1,
                        "update": 2,
                        "sample_order": ["train-01"],
                        "route_rng_state": "present",
                    },
                    "rng_streams": ["numpy", "python", "torch_cpu"],
                    "optimizer_state_present": True,
                },
            )
            runner.json_write(
                root / "metrics.json",
                {"loss": 12.5, "paired_dense_metrics": [paired("train-01")]},
            )
            phase = root / ("s1" if stage.command == "updater-train" else "s0")
            phase.mkdir()
            (phase / "stage_history.jsonl").write_text(
                json.dumps(
                    {
                        "schema_version": "pfgr-lite-stage-history-v1",
                        "stage": phase.name.upper(),
                        "scope": "segment",
                        "record": {
                            "epoch": 0,
                            "update": 1,
                            "subject_ids": ["train-01"],
                            "objective": 12.5,
                        },
                    }
                )
                + "\n"
            )
        if stage.command == "evaluate":
            (root / "paired_subjects.jsonl").write_text(
                "".join(json.dumps(paired(f"val-{i:02d}")) + "\n" for i in range(4))
            )
        if stage.command == "headroom-evaluate":
            subjects = []
            for i in range(4):
                subject = f"val-{i:02d}"
                if self.mismatch and stage.arm == "u_plus_spectral" and i == 3:
                    subject = "different-val"
                subjects.append(
                    {
                        "subject_id": subject,
                        "no_op": {"metric": paired(subject)},
                        "oracle": {"metric": paired(subject)},
                        "random_controls": [{"seed": 17, "metric": paired(subject)}],
                        "exact_pool_audit": {
                            "top1_regret": None,
                            "observed_best_gain": 0.01,
                            "pool_coverage": {"finite_exact_candidate_count": 31},
                            "ranking": {"pairwise_order_agreement": 0.75},
                            "sign_agreement": {"fraction": 0.8},
                        },
                        "diagnostic_timing": {
                            "phases": {"exact_pool_audit": {"elapsed_seconds": 0.2}}
                        },
                    }
                )
            runner.json_write(
                root / "next1/headroom_metrics.json",
                {
                    "schema_version": "pfgr-lite-headroom-result-v2",
                    "subjects": subjects,
                    "scientific_status": "INCONCLUSIVE",
                },
            )
            runner.json_write(
                root / "next1/next1_evidence.json",
                {
                    "schema_version": "pfgr-lite-headroom-evidence-v1",
                    "subjects": subjects,
                    "scientific_status": "INCONCLUSIVE",
                },
            )
            runner.json_write(
                root / "next1/headroom_decision.json",
                {
                    "schema_version": "pfgr-lite-headroom-decision-v1",
                    "decision": "INCONCLUSIVE",
                },
            )
        return 0


def test_plan_only_dependency_wiring_and_live_parser(args, capsys):
    args.plan_only = True
    calls = FakeCompute()
    code, destination = runner.run(args, calls)
    assert code == 0 and calls.called == []
    manifest = runner.json_read(destination / "metadata/pipeline_manifest.json")
    stages = manifest["stages"]
    assert manifest["r5_status"] == "CLOSED"
    assert not {"bank-build", "value-fit", "calibrate", "resume"} & {
        s["command"] for s in stages
    }
    with pytest.raises(FileExistsError):
        runner.run(args, calls)
    for stage in stages:
        argv = stage["argv"]
        if stage["command"] == "updater-train":
            assert argv[argv.index("--checkpoint") + 1].endswith("R3-b2/inference.pt")
            assert argv[argv.index("--epochs") + 1] == "9"
            assert "--max-steps" not in argv
        if stage["command"] == "static-train":
            assert argv[argv.index("--epochs") + 1] == "7"
        if stage["command"] == "evaluate":
            assert argv[argv.index("--config") + 1].endswith(
                f"R3-{stage['arm']}/resolved_config.json"
            )
            assert argv[argv.index("--checkpoint") + 1].endswith(
                f"R3-{stage['arm']}/inference.pt"
            )
        if stage["command"] == "headroom-evaluate":
            assert "--exact-pool-audit" in argv
            assert argv[argv.index("--base-checkpoint") + 1].endswith(
                "R3-b2/inference.pt"
            )
    assert str(args.data_root) in stages[0]["argv"]
    assert "$(not-a-shell)" in capsys.readouterr().out


def test_plan_can_describe_missing_server_inputs_but_execution_rejects(args):
    args.plan_only = True
    args.data_root = args.run_root / "remote-data"
    args.medicalnet_checkpoint = args.run_root / "remote-checkpoint"
    args.split_file = args.run_root / "remote-split"
    args.python = args.run_root / "remote-venv/bin/python"
    code, destination = runner.run(args, FakeCompute())
    manifest = runner.json_read(destination / "metadata/pipeline_manifest.json")
    assert code == 0
    assert manifest["input_sha256"][str(args.medicalnet_checkpoint)] is None
    assert manifest["data_root_status"] == "unavailable_plan_only"
    args.plan_only = False
    with pytest.raises(ValueError, match="missing BraTS"):
        runner.run(args, FakeCompute())


@pytest.mark.parametrize("arm", ["b0", "b1", "b2", "b_light"])
def test_final_static_evaluation_accepts_published_execution_envelope(
    args, tmp_path, arm
):
    from smagm.cli.pfgr_lite import _config_for_command, _parser

    cli = _parser()
    training = cli.parse_args(
        [
            "static-train",
            "--config",
            str(args.config),
            "--base",
            arm,
            "--device",
            "cpu",
            "--seed",
            "73",
        ]
    )
    expected, details = _config_for_command(training, stage="S0")
    published = tmp_path / f"{arm}-resolved_config.json"
    runner.json_write(published, details["execution"])
    evaluation = cli.parse_args(
        [
            "evaluate",
            "--config",
            str(published),
            "--checkpoint",
            str(tmp_path / "not-loaded-by-config-check.pt"),
            "--scenario",
            "noop",
            "--budget",
            "0",
            "--device",
            "cpu",
            "--seed",
            "73",
        ]
    )
    observed, resolved = _config_for_command(evaluation)
    assert observed.static.variant == expected.static.variant
    assert resolved["execution"]["stage_options"]["seed"] == 73


@pytest.mark.parametrize("failure", ["R0-preflight", "R3-b1", "R4-u_only", "package"])
def test_fail_fast_preserves_exit_and_attempts_package(args, failure):
    compute = FakeCompute(fail=failure)
    code, destination = runner.run(args, compute)
    assert code == 7
    assert compute.called[-1].command == "package"
    failed_index = next(
        i for i, stage in enumerate(compute.called) if stage.name == failure
    )
    assert all(
        stage.command == "package" for stage in compute.called[failed_index + 1 :]
    )
    summary = runner.json_read(destination / "metadata/pipeline_summary.json")
    assert summary["status"] == "failed"
    assert summary["failure"]["stage"] == failure
    assert summary["r5_status"] == "CLOSED"
    assert summary["package"]["status"] == (
        "failed" if failure == "package" else "succeeded"
    )


def test_success_summary_metrics_no_imputation_or_average(args):
    code, destination = runner.run(args, FakeCompute())
    assert code == 0
    summary = runner.json_read(destination / "metadata/pipeline_summary.json")
    assert summary["scientific_status"] == "INCONCLUSIVE"
    rows = summary["rows"]
    assert any(row["metric"] == "before.mae" and row["value"] == 0.4 for row in rows)
    assert all(
        row["value"] is None
        for row in rows
        if row["metric"]
        in {"before.ssim", "after.ssim", "improvement.ssim", "top1_regret", "flops"}
    )
    assert any(row["metric_scope"] == "training_forward_pre_update" for row in rows)
    assert any(row["metric_scope"] == "heldout_final_checkpoint" for row in rows)
    random = next(row for row in rows if row["scope"] == "random")
    assert random["random_seed"] == 17 and random["training_seed"] == 73
    assert random["seed_kind"] == "random_control"
    assert any(
        row["metric"] == "parameter_count" and row["value"] == 100 for row in rows
    )
    assert not any("average" in row["metric"] for row in rows)
    assert ",null," in (destination / "metadata/pipeline_metrics.csv").read_text()
    assert Path(summary["package"]["archive"]["path"]).is_file()
    assert (destination / "metadata/stage_logs/R4-u_only/command.json").is_file()


def test_hash_mismatch_does_not_reserve_or_execute(args):
    args.medicalnet_sha256 = "0" * 64
    compute = FakeCompute()
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        runner.run(args, compute)
    assert compute.called == []
    assert not args.run_root.exists()


def test_changed_predecessor_and_missing_outputs_fail_closed(args):
    compute = FakeCompute(missing="inference.pt")
    code, destination = runner.run(args, compute)
    assert code == 1
    assert [s.name for s in compute.called][-2:] == ["R1-synthetic", "package"]
    assert (
        runner.json_read(destination / "metadata/pipeline_summary.json")["status"]
        == "failed"
    )


def test_final_validation_subject_mismatch_is_failure(args):
    code, destination = runner.run(args, FakeCompute(mismatch=True))
    assert code == 1
    summary = runner.json_read(destination / "metadata/pipeline_summary.json")
    assert "subject IDs" in summary["failure"]["error"]
    assert summary["failure"]["stage"] == "R4B-u_plus_spectral"


def test_predecessor_bytes_cannot_change_between_updater_arms(args):
    compute = FakeCompute()

    def tamper(stage, *rest):
        result = compute(stage, *rest)
        if stage.name == "R4-u_only":
            base = Path(stage.argv[stage.argv.index("--checkpoint") + 1])
            base.write_bytes(b"changed checkpoint")
        return result

    code, destination = runner.run(args, tamper)
    summary = runner.json_read(destination / "metadata/pipeline_summary.json")
    assert code == 1
    assert summary["failure"]["stage"] == "R4-u_plus_spectral"
    assert "dependency" in summary["failure"]["error"]
    assert "R4-u_plus_spectral" not in [stage.name for stage in compute.called]
    assert compute.called[-1].command == "package"


def test_effective_training_seed_must_match_requested(args):
    compute = FakeCompute()

    def bad_seed(stage, *rest):
        result = compute(stage, *rest)
        if stage.name == "R1-synthetic":
            path = Path(stage.run_dir) / "resolved_config.json"
            config = runner.json_read(path)
            config["stage_options"]["seed"] = 74
            runner.json_write(path, config)
        return result

    code, destination = runner.run(args, bad_seed)
    assert code == 1
    summary = runner.json_read(destination / "metadata/pipeline_summary.json")
    assert "training seed" in summary["failure"]["error"]
    assert compute.called[-1].command == "package"


def test_explicit_roles_are_hash_bound_and_never_regenerated(args):
    args.roles_file = args.split_file.parent / "reviewed roles.json"
    args.roles_file.write_text('{"schema_version": "fixture_only"}')
    args.plan_only = True
    _, destination = runner.run(args, FakeCompute())
    manifest = runner.json_read(destination / "metadata/pipeline_manifest.json")
    preflight = manifest["stages"][0]
    assert "--write-roles" not in preflight["argv"]
    assert preflight["argv"][preflight["argv"].index("--roles-file") + 1] == str(
        args.roles_file
    )
    assert manifest["input_sha256"][str(args.roles_file)] == runner.sha256(
        args.roles_file
    )


def test_unknown_metrics_keep_nulls_and_diagnostic_reasons(tmp_path):
    root = tmp_path / "stage"
    root.mkdir()
    row = paired()
    row["before"]["psnr"] = float("inf")
    row["before"]["ssim_reason"] = "no_valid_window"
    row["before"]["ssim_reduction"] = "center_observed_valid_window_v1"
    (root / "metrics.json").write_text(json.dumps({"paired_dense_metrics": [row]}))
    stage = runner.Stage("test", "updater-train", [], str(root), [], [], "u_only", 9)
    rows = runner.collect_metrics(stage)
    assert next(r for r in rows if r["metric"] == "before.psnr")["value"] is None
    assert (
        next(r for r in rows if r["metric"] == "before.ssim_reason")["value"]
        == "no_valid_window"
    )
    assert (
        next(r for r in rows if r["metric"] == "before.ssim_reduction")["value"]
        == "center_observed_valid_window_v1"
    )


def test_history_tail_retains_actual_epoch_update_and_line_indices(tmp_path):
    root = tmp_path / "stage"
    (root / "s1").mkdir(parents=True)
    path = root / "s1/stage_history.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "record": {
                        "epoch": i // 10,
                        "update": i,
                        "subject_ids": ["train-01"],
                        "objective": i / 100,
                    }
                }
            )
            + "\n"
            for i in range(105)
        )
    )
    stage = runner.Stage("history", "updater-train", [], str(root), [], [], "u_only", 9)
    rows = runner.collect_metrics(stage)
    updates = [r for r in rows if r["metric"] == "update"]
    assert len(updates) == 100
    assert updates[0]["value"] == 5 and updates[-1]["value"] == 104
    assert updates[0]["json_path"] == "line[6].record.update"
    assert all(
        r["subject_id"] == "train-01"
        and r["scope"] == "training_history_last_100_records"
        for r in rows
    )
    assert len(path.read_text().splitlines()) == 105


def test_real_executor_preserves_argv_output_and_nonzero(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    events = []
    text = "literal; $(touch forbidden) ' quote"
    stage = runner.Stage(
        "executor",
        "smoke",
        [
            sys.executable,
            "-c",
            "import sys; print(sys.argv[1]); print('error', file=sys.stderr); sys.exit(6)",
            text,
        ],
        str(tmp_path / "unused"),
        [],
        [],
    )
    assert (
        runner.execute(
            stage, logs, lambda event, **fields: events.append((event, fields)), 10
        )
        == 6
    )
    assert (logs / "stdout.txt").read_text().strip() == text
    assert (logs / "stderr.txt").read_text().strip() == "error"
    assert events and all("text" not in fields for _, fields in events)
    assert sum(fields["count_bytes"] for _, fields in events) == sum(
        (logs / f).stat().st_size for f in ("stdout.txt", "stderr.txt")
    )


def test_timeout_attempts_packaging(args):
    compute = FakeCompute()

    def timeout(stage, *rest):
        if stage.name == "R0-preflight":
            raise subprocess.TimeoutExpired(stage.argv, 1)
        return compute(stage, *rest)

    code, destination = runner.run(args, timeout)
    assert code == 124
    assert compute.called[-1].command == "package"
    assert (
        "TimeoutExpired"
        in runner.json_read(destination / "metadata/pipeline_summary.json")["failure"][
            "error"
        ]
    )


def test_real_packager_includes_runner_metadata_and_logs(args):
    code, destination = runner.run(args, FakeCompute(package_real=True))
    summary = runner.json_read(destination / "metadata/pipeline_summary.json")
    assert code == 0, summary["package"].get("error")
    with zipfile.ZipFile(summary["package"]["archive"]["path"]) as archive:
        names = archive.namelist()
        for expected in (
            "pipeline_manifest.json",
            "pipeline_summary.json",
            "pipeline_events.jsonl",
            "pipeline_metrics.csv",
            "stdout.txt",
            "stderr.txt",
            "command.json",
            "exit.json",
            "headroom_metrics.json",
        ):
            assert any(Path(name).name == expected for name in names), expected
        assert not any(name.endswith(".pt") for name in names)
