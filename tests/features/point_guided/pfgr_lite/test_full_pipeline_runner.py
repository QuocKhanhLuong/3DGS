"""Full-runner DAG tests use mock compute and actual immutable files/packaging."""

from __future__ import annotations

import importlib.util
import json
import shlex
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load("pfgr_full_runner_test", REPO / "scripts/run_pfgr_full_pipeline.py")
legacy = load(
    "pfgr_legacy_runner_test_fixtures",
    Path(__file__).with_name("test_pipeline_runner.py"),
)


@pytest.fixture
def args(tmp_path):
    data = tmp_path / "BraTS space;$(never-shell)"
    data.mkdir()
    checkpoint = tmp_path / "MedicalNet ' $(never-shell).pth"
    checkpoint.write_bytes(b"synthetic test bytes only")
    split = tmp_path / "split.json"
    runner.core.json_write(
        split,
        {
            "train": ["train-01"],
            "val": [f"val-{i:02d}" for i in range(4)],
            "test": [f"test-{i:02d}" for i in range(4)],
        },
    )
    return runner.parser().parse_args(
        [
            "--data-root",
            str(data),
            "--medicalnet-checkpoint",
            str(checkpoint),
            "--medicalnet-sha256",
            runner.core.sha256(checkpoint),
            "--split-file",
            str(split),
            "--run-root",
            str(tmp_path / "runs with spaces"),
            "--run-name",
            "fresh",
            "--device",
            "cpu",
            "--static-epochs",
            "7",
            "--updater-epochs",
            "9",
            "--value-epochs",
            "11",
            "--seed",
            "73",
        ]
    )


class FullFake(legacy.FakeCompute):
    def __init__(
        self,
        *,
        adaptive=False,
        pair_mismatch=False,
        cohort_mismatch=False,
        mutate_after=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.adaptive = adaptive
        self.pair_mismatch = pair_mismatch
        self.cohort_mismatch = cohort_mismatch
        self.mutate_after = mutate_after

    def __call__(self, stage, logs, emit, timeout):
        root = Path(stage.run_dir)
        if stage.command == "runner-helper":
            self.called.append(stage)
            if stage.name == self.fail:
                root.mkdir(parents=True)
                return 7
            specification = json.loads(runner.flag(stage, "--spec-json"))
            kind = runner.flag(stage, "--helper")
            if kind == "resume-compare":
                root.mkdir(parents=True)
                runner.core.json_write(
                    root / "resume_comparison.json",
                    {
                        "schema_version": "pfgr-lite-full-resume-comparison-v1",
                        "status": "SOFTWARE_PASS",
                        "committed_updates": [1, 2, 2],
                        "cursor_fields_compared": [["update"], ["update"], ["update"]],
                        "difference_paths": [],
                        "kind": specification["kind"],
                        "tensor_comparisons": 1,
                        "changed_tensors_after_resume": 1,
                        "scope": "mock-compute DAG fixture only",
                    },
                )
                return 0
            if kind == "adaptive-check" and self.adaptive:
                root.mkdir(parents=True)
                path = Path(specification["adaptive_path"])
                runner.core.json_write(
                    root / "adaptive_availability.json",
                    {
                        "schema_version": "pfgr-lite-full-adaptive-availability-v1",
                        "status": "AVAILABLE",
                        "adaptive_path": str(path),
                        "adaptive_sha256": runner.core.sha256(path),
                        "reason": None,
                        "scope": "mock-compute DAG fixture only",
                    },
                )
                return 0
            return runner.helper_main(stage.argv[3:])
        rc = super().__call__(stage, logs, emit, timeout)
        if rc or stage.command == "package":
            return rc
        if stage.name == "R0-preflight":
            runner.core.json_write(
                root / "roles.json",
                {
                    "producer_fit_subject_ids": ["train-01"],
                    "calibration_fit_subject_ids": ["fit-01"],
                    "calibration_allowance_subject_ids": ["allow-01"],
                    "baseline_validation_subject_ids": [
                        f"val-{i:02d}" for i in range(4)
                    ],
                    "baseline_test_subject_ids": [f"test-{i:02d}" for i in range(4)],
                },
            )
        if stage.command == "resume" and stage.name != "R10-value-resumed":
            parent = Path(runner.flag(stage, "--resume-checkpoint")).parent
            runtime = runner.core.json_read(parent / "stage_runtime.json")
            runtime["cursor"]["update"] = 2
            runner.core.json_write(root / "stage_runtime.json", runtime)
            runner.core.json_write(
                root / "resolved_config.json",
                runner.core.json_read(parent / "resolved_config.json"),
            )
        if stage.command == "bank-build":
            runner.core.json_write(
                root / "s2/bank/index.json",
                {
                    "schema_version": "pfgr-lite-value-bank-v1",
                    "gain_scale": {"digest": "gain-scale"},
                    "rows": [
                        {
                            "row_id": 0,
                            "row_hash": "immutable-row",
                            "shard": "000.json",
                            "offset": 0,
                        }
                    ],
                },
            )
        if stage.command == "value-evaluate":
            bank = Path(runner.flag(stage, "--bank-index"))
            variant = int(stage.arm[1:])
            digest = runner.core.sha256(bank)
            row = {
                "row_key": "shared-key",
                "row_hash": "immutable-row",
                "row_id": 0,
                "shard": "000.json",
                "offset": 0,
                "bank_manifest_hash": digest,
                "input_variant": variant,
                "subject_id": "train-01",
                "context_id": "ctx",
                "state_version": 0,
                "point_id": 0,
                "action_id": "action",
                "proposal_hash": "proposal",
                "state_digest": "state",
                "measured_raw_gain": 0.01,
                "group_key": "group",
                "measured_rank": 1,
                "predicted_raw": 0.02,
                "predicted_scaled": 0.2,
                "predicted_rank": 1,
            }
            if self.pair_mismatch and variant == 222:
                row["context_id"] = "wrong-context"
            runner.core.json_write(
                root / "value_evaluate_pairs.json",
                {
                    "schema_version": "pfgr-lite-value-evaluation-pairs-v1",
                    "bank_manifest_hash": digest,
                    "gain_scale_hash": "gain-scale",
                    "input_variant": variant,
                    "row_count": 1,
                    "group_count": 1,
                    "rows": [row],
                },
            )
            runner.core.json_write(
                root / "value_evaluate.json",
                {
                    "mse_raw": 0.01,
                    "sign_accuracy": 0.5,
                    "top1_regret": 0.02,
                    "constant_training_mean_mse_raw": 0.03,
                    "rank_correlation": None,
                },
            )
        if stage.command == "calibrate":
            runner.core.json_write(
                root / "calibration.json",
                {
                    "status": "SOFTWARE_PASS",
                    "calibration": {
                        "capability": "adaptive" if self.adaptive else "diagnostic",
                        "a": 0.5,
                        "b": -0.1,
                        "allowance": 0.01,
                    },
                    "insufficient_data": False,
                },
            )
            if self.adaptive:
                (root / "adaptive.pt").write_bytes(
                    b"mock adaptive checkpoint; strict behavior tested separately"
                )
        if "execution_authorization.json" in {Path(p).name for p in stage.produces}:
            runner.core.json_write(
                root / "execution_authorization.json",
                {
                    "schema_version": "pfgr-lite-execution-authorization-v1",
                    "human_reviewed": False,
                    "authorizes_main": False,
                    "decision": "EXPLORATORY_EXECUTION",
                },
            )
        if stage.name.startswith(("R8-", "R9-")):
            role = runner.flag(stage, "--split-role")
            prefix = "val" if role == "validation" else "test"
            subjects = []
            for i in range(4):
                subject = f"{prefix}-{i:02d}"
                if (
                    self.cohort_mismatch
                    and stage.name == "R9-random-k1-seed17"
                    and i == 3
                ):
                    subject = "undeclared-test"
                row = legacy.paired(subject)
                row.update(
                    context_id=subject + "-context",
                    z0_digest="z0",
                    z0_state_digest="state",
                    route={
                        "k": int(runner.flag(stage, "--budget")),
                        "stop_reason": "budget",
                    },
                    pipeline_elapsed_seconds=0.01,
                )
                if runner.flag(stage, "--scenario") in {"noop", "static"}:
                    row["after"] = dict(row["before"])
                subjects.append(row)
            (root / "paired_subjects.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in subjects)
            )
            (root / "action_metrics.jsonl").write_text("")
        if self.mutate_after == stage.name:
            Path(runner.flag(stage, "--bank-index")).write_text('{"mutated": true}')
        return rc


def summary(destination):
    return runner.core.json_read(destination / "metadata/pipeline_summary.json")


def test_plan_only_complete_dag_budget_and_argv(args, capsys):
    args.plan_only = True
    fake = FullFake()
    code, destination = runner.run(args, fake)
    assert code == 0 and not fake.called
    manifest = runner.core.json_read(destination / "metadata/pipeline_manifest.json")
    assert manifest["last_stage"] == "R10" and manifest["profile"] == "PROVISIONAL"
    assert manifest["human_reviewed"] is False and manifest["authorizes_main"] is False
    plan = runner.build_plan(args, destination)
    runner.core.validate_live_flags(
        [s for s in plan if s.command != "runner-helper"],
        allowed=runner.ALLOWED,
        static_only=False,
    )
    assert plan[-1].name == "R10-value-comparison"
    assert any(s.name == "R2-benchmark" for s in plan)
    for stage in plan:
        assert shlex.split(shlex.join(stage.argv)) == stage.argv
        assert (
            "--headroom-decision" not in stage.argv
            and "--review-receipt" not in stage.argv
        )
        if stage.name.startswith("R6-fit-"):
            assert runner.flag(stage, "--epochs") == "11"
        if (
            stage.name.startswith(("R5", "R6", "R7", "R8", "R9"))
            and stage.command != "runner-helper"
        ):
            assert "--engineering-only" in stage.argv
    by_name = {s.name: s for s in plan}
    updater = str(destination / "stages/R4-u_plus_spectral/inference.pt")
    bank = str(destination / "stages/R5-bank/s2/bank/index.json")
    value = str(destination / "stages/R6-fit-v366/value.pt")
    for variant in runner.VALUE_VARIANTS:
        s = by_name[f"R6-fit-v{variant}"]
        assert (
            runner.flag(s, "--checkpoint") == updater
            and runner.flag(s, "--bank-index") == bank
        )
        assert (
            str(destination / "stages/R4-u_plus_spectral/resolved_config.json")
            in s.requires
        )
    for phase, role in (("R8", "validation"), ("R9", "test")):
        controls = [s for s in plan if s.name.startswith(phase + "-")]
        assert len(controls) == 42
        assert {
            (
                runner.flag(s, "--scenario"),
                int(runner.flag(s, "--budget")),
                int(runner.flag(s, "--seed")),
            )
            for s in controls
        } == {
            (p, k, seed)
            for seed in runner.RANDOM_SEEDS
            for p, ks in [("noop", (0,)), ("static", (0,))]
            + [(p, runner.BUDGETS) for p in runner.POLICIES]
            for k in ks
        }
        for s in controls:
            assert runner.flag(s, "--split-role") == role
            if runner.flag(s, "--scenario") in {
                "adaptive",
                "fixed_learned",
                "parallel_topk",
            }:
                assert runner.flag(s, "--value-checkpoint") == value
            if role == "test":
                assert "--exploratory-run" in s.argv
    i, r, u = [
        by_name["R10-value-" + n] for n in ("interrupted", "resumed", "uninterrupted")
    ]
    for flag in (
        "--bank-index",
        "--epochs",
        "--batch-size",
        "--learning-rate",
        "--value-input",
        "--seed",
        "--config",
    ):
        assert len({runner.flag(s, flag) for s in (i, r, u)}) == 1
    assert [runner.flag(s, "--max-steps") for s in (i, r, u)] == ["1", "2", "2"]
    with pytest.raises(FileExistsError):
        runner.run(args, fake)


def test_diagnostic_calibration_continues_r9_r10_and_real_package(args):
    fake = FullFake(package_real=True)
    code, destination = runner.run(args, fake)
    result = summary(destination)
    assert code == 0, result["errors"] or result["package"].get("error")
    assert (
        result["status"] == "completed_with_unavailable"
        and result["adaptive"]["status"] == "NOT_AVAILABLE"
    )
    assert result["unavailable_count"] == 18
    assert not any(runner.flag(s, "--scenario") == "adaptive" for s in fake.called)
    assert "R9-parallel_topk-k4-seed41" in {s.name for s in fake.called}
    assert {"R10-synthetic-comparison", "R10-value-comparison"} <= {
        s.name for s in fake.called
    }
    assert any(r["metric"] == "mse_raw" and r["value"] == 0.01 for r in result["rows"])
    assert any(
        r["metric"] == "calibration.a" and r["value"] == 0.5 for r in result["rows"]
    )
    assert any(
        r["seed_kind"] == "policy_randomization" and r["training_seed"] == 73
        for r in result["rows"]
    )
    assert result["cohorts"]["test:4"]["subject_ids"] == [
        f"test-{i:02d}" for i in range(4)
    ]
    exit_payload = runner.core.json_read(
        destination / "metadata/stage_logs/R6-fit-v366/command.json"
    )
    bank = str(destination / "stages/R5-bank/s2/bank/index.json")
    assert exit_payload["input_sha256"][bank] == runner.core.sha256(Path(bank))
    archive = Path(result["package"]["archive"]["path"])
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert any(n.endswith("resume_comparison.json") for n in names)
        assert any(n.endswith("control_join.json") for n in names)
        assert any(n.endswith("value_join.json") for n in names)
        assert not any(n.endswith(".pt") for n in names)


def test_available_adaptive_branch_uses_validated_optional_hash(args):
    fake = FullFake(adaptive=True, package_real=True)
    code, destination = runner.run(args, fake)
    result = summary(destination)
    assert code == 0, result["errors"] or result["package"].get("error")
    assert (
        result["status"] == "completed" and result["adaptive"]["status"] == "AVAILABLE"
    )
    adaptive = str(destination / "stages/R7-calibration/adaptive.pt")
    command = runner.core.json_read(
        destination / "metadata/stage_logs/R9-adaptive-k4-seed41/command.json"
    )
    assert command["input_sha256"][adaptive] == runner.core.sha256(Path(adaptive))


@pytest.mark.parametrize(
    "failure",
    ["R4B-u_plus_spectral", "R7-calibration", "R6-fit-v126", "R10-synthetic-resumed"],
)
def test_failure_is_error_and_packages_independent_later_work(args, failure):
    fake = FullFake(fail=failure, package_real=True)
    code, destination = runner.run(args, fake)
    result = summary(destination)
    assert code == 1 and result["status"] == "error"
    assert result["package"]["status"] == "succeeded", result["package"]
    assert "R10-value-comparison" in {s.name for s in fake.called}
    assert "R9-random-k4-seed41" in {s.name for s in fake.called}
    if failure == "R7-calibration":
        assert result["adaptive"]["status"] == "ERROR"
        assert not any(r["status"] == "not_available" for r in result["stages"])


def test_cross_variant_mismatch_records_error_without_blocking_predeclared_v366(args):
    fake = FullFake(pair_mismatch=True, package_real=True)
    code, destination = runner.run(args, fake)
    assert code == 1
    result = summary(destination)
    assert "R9-fixed_learned-k1-seed17" in {s.name for s in fake.called}
    assert "R9-random-k1-seed17" in {s.name for s in fake.called}
    assert result["package"]["status"] == "succeeded"


def test_declared_test_cohort_mismatch_fails(args):
    fake = FullFake(cohort_mismatch=True, package_real=True)
    code, destination = runner.run(args, fake)
    assert code == 1
    assert any(
        "predeclared ordered cohort" in (e["error"] or "")
        for e in summary(destination)["errors"]
    )


def test_changed_bank_is_not_trusted_by_later_fit(args):
    fake = FullFake(mutate_after="R6-evaluate-v126", package_real=True)
    code, destination = runner.run(args, fake)
    assert code == 1
    assert "R6-fit-v270" not in {s.name for s in fake.called}
    assert any(
        "dependency" in (e["error"] or "") for e in summary(destination)["errors"]
    )


def test_adaptive_absence_and_invalid_claim_are_distinct(tmp_path):
    payload = tmp_path / "calibration.json"
    specification = {
        "inputs": [str(payload)],
        "adaptive_path": str(tmp_path / "adaptive.pt"),
    }
    runner.core.json_write(
        payload,
        {"status": "INCONCLUSIVE", "calibration": None, "insufficient_data": True},
    )
    assert (
        runner.adaptive_availability(specification)["reason"]
        == "insufficient_calibration_data"
    )
    runner.core.json_write(
        payload, {"status": "SOFTWARE_PASS", "calibration": {"capability": "adaptive"}}
    )
    with pytest.raises(ValueError, match="without a published"):
        runner.adaptive_availability(specification)
    runner.core.json_write(payload, {"status": "ERROR", "calibration": None})
    with pytest.raises(ValueError, match="recognized completed"):
        runner.adaptive_availability(specification)


def test_value_join_rejects_wrong_bank_bytes(tmp_path):
    index = tmp_path / "index.json"
    runner.core.json_write(index, {"rows": [], "gain_scale": {"digest": "scale"}})
    pairs = []
    for v in runner.VALUE_VARIANTS:
        path = tmp_path / f"v{v}.json"
        runner.core.json_write(
            path,
            {
                "input_variant": v,
                "bank_manifest_hash": "wrong",
                "gain_scale_hash": "scale",
                "row_count": 1,
                "group_count": 1,
            },
        )
        pairs.append(path)
    with pytest.raises(ValueError, match="actual bank index bytes"):
        runner.value_join([index] + pairs)


def test_actual_cached_fit_cursor_weights_optimizer_rng_comparison(
    tmp_path, monkeypatch
):
    """Actual cached optimizer steps, injected container seam; no MRI/MedicalNet work."""
    from types import SimpleNamespace

    from smagm.features.point_guided.pfgr_lite import checkpoint
    from smagm.features.point_guided.pfgr_lite.value_net import fit_value

    fixtures = load(
        "pfgr_cached_value_fixture", Path(__file__).with_name("test_value_fit.py")
    )
    bank = fixtures._bank(tmp_path)
    options = {
        "input_variant": 366,
        "epochs": 3,
        "batch_size": 2,
        "seed": 71,
        "learning_rate": 1e-3,
        "device": "cpu",
    }
    interrupted = fit_value(bank, max_updates=1, **options)
    resumed = fit_value(bank, resume=interrupted.resume_state, max_updates=2, **options)
    reference = fit_value(bank, max_updates=2, **options)
    paths = [
        tmp_path / (name + ".pt") for name in ("interrupted", "resumed", "reference")
    ]
    values = {}
    for path, fit in zip(paths, (interrupted, resumed, reference)):
        path.write_bytes(b"container seam fixture; actual fit payload injected")
        values[path] = SimpleNamespace(
            stage_state=fit.stage_state,
            bank_state={"cached_value_fit": fit.resume_state},
        )
    monkeypatch.setattr(checkpoint, "load_resume", values.__getitem__)
    result = runner.resume_comparison(paths, "value")
    assert result["status"] == "SOFTWARE_PASS" and result["committed_updates"] == [
        1,
        2,
        2,
    ]
    assert (
        result["tensor_comparisons"] > 10 and result["changed_tensors_after_resume"] > 0
    )
    resumed.resume_state["model_state_dict"][
        next(iter(resumed.resume_state["model_state_dict"]))
    ].add_(1.0)
    result = runner.resume_comparison(paths, "value")
    assert result["status"] == "MISMATCH" and any(
        p.startswith("model_state_dict.") for p in result["difference_paths"]
    )
