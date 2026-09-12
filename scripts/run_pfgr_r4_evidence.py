#!/usr/bin/env python3
"""Run the explicit PFGR R0--R4B CLI DAG; stdlib only, never open R5."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
PREFIX = "pfgr-lite-pipeline-"
ALLOWED = {
    "preflight",
    "runbook-check",
    "smoke",
    "benchmark",
    "static-train",
    "updater-train",
    "evaluate",
    "headroom-evaluate",
    "package",
}
METRIC_FIELDS = ("mae", "psnr", "ssim", "masked_charbonnier")
CSV_FIELDS = (
    "stage",
    "arm",
    "subject_id",
    "training_seed",
    "random_seed",
    "seed_kind",
    "metric_scope",
    "scope",
    "metric",
    "value",
    "artifact",
    "json_path",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_read(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise TypeError(f"JSON object required: {path}")
    return result


def json_write(path: Path, value: Any) -> None:
    # Only the runner's reserved metadata files are replaced, never CLI runs.
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--medicalnet-checkpoint", type=Path, required=True)
    p.add_argument("--medicalnet-sha256", required=True)
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--roles-file", type=Path)
    p.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="existing parent or new parent; each run reserves an exclusive child",
    )
    p.add_argument(
        "--run-name", help="new simple child name; default UTC plus random suffix"
    )
    p.add_argument("--config", type=Path, default=REPO / "configs/pfgr_lite/main.json")
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--device", required=True)
    p.add_argument("--profile", choices=("bootstrap", "provisional"), required=True)
    p.add_argument("--static-epochs", type=positive, required=True)
    p.add_argument("--updater-epochs", type=positive, required=True)
    p.add_argument("--static-max-steps", type=positive)
    p.add_argument("--updater-max-steps", type=positive)
    p.add_argument("--train-max-subjects", type=positive)
    p.add_argument(
        "--seed",
        type=int,
        default=20260907,
        help="training/initialization seed; R4B random seeds remain 17,29,41",
    )
    p.add_argument(
        "--base",
        choices=("b2",),
        default="b2",
        help="locked predeclared shared R4 predecessor",
    )
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="bounded R2 parity pilot using real R1 checkpoint",
    )
    p.add_argument("--benchmark-repeats", type=positive, default=3)
    p.add_argument("--candidate-chunk-size", type=positive, default=1)
    p.add_argument("--decode-chunk-size", type=positive, default=1024)
    p.add_argument(
        "--stage-timeout-seconds",
        type=positive,
        help="optional process-group timeout, including packaging",
    )
    p.add_argument("--package-max-file-mib", type=positive, default=64)
    p.add_argument("--package-max-archive-mib", type=positive, default=512)
    p.add_argument(
        "--plan-only",
        action="store_true",
        help="validate local config and live parser flags, record missing server inputs, save argv plan; execute no CLI or training",
    )
    return p


@dataclass(frozen=True)
class Stage:
    name: str
    command: str
    argv: list[str]
    run_dir: str
    requires: list[str]
    produces: list[str]
    arm: str | None = None
    training_seed: int | None = None


def build_plan(args: argparse.Namespace, destination: Path) -> list[Stage]:
    cli = [str(args.python), "-u", "-m", "smagm.cli.pfgr_lite"]
    stage_root = destination / "stages"
    roles = args.roles_file or stage_root / "R0-preflight" / "roles.json"
    common = [
        "--config",
        str(args.config),
        "--data-root",
        str(args.data_root),
        "--split-file",
        str(args.split_file),
        "--device",
        args.device,
        "--no-amp",
        "--seed",
        str(args.seed),
        "--candidate-chunk-size",
        str(args.candidate_chunk_size),
        "--decode-chunk-size",
        str(args.decode_chunk_size),
    ]
    fresh = [
        "--medicalnet-checkpoint",
        str(args.medicalnet_checkpoint),
        "--medicalnet-sha256",
        args.medicalnet_sha256,
    ]
    with_roles = ["--roles-file", str(roles)]
    plan: list[Stage] = []

    def add(
        name: str,
        command: str,
        flags: list[str],
        requires: list[Path] | None = None,
        products: tuple[str, ...] = ("receipt.json",),
        arm: str | None = None,
        training: bool = False,
    ) -> None:
        if command not in ALLOWED:
            raise ValueError(f"forbidden pipeline command: {command}")
        out = stage_root / name
        plan.append(
            Stage(
                name,
                command,
                cli
                + [command]
                + flags
                + ["--output-root", str(stage_root), "--run-name", name],
                str(out),
                [str(p) for p in requires or []],
                [str(out / p) for p in products],
                arm,
                args.seed if training else None,
            )
        )

    add(
        "R0-preflight",
        "preflight",
        common + fresh + (with_roles if args.roles_file else ["--write-roles"]),
        products=("receipt.json",)
        if args.roles_file
        else ("receipt.json", "roles.json"),
    )
    add(
        "R0-runbook",
        "runbook-check",
        [
            "--runbook",
            str(REPO / "RUNBOOK_PFGR_LITE.md"),
            "--config-dir",
            str(REPO / "configs/pfgr_lite"),
        ],
    )
    smoke_products = (
        "receipt.json",
        "inference.pt",
        "resume.pt",
        "stage_runtime.json",
        "resolved_config.json",
    )
    add(
        "R1-synthetic",
        "smoke",
        [
            "--synthetic",
            "--config",
            str(REPO / "configs/pfgr_lite/synthetic.json"),
            "--device",
            "cpu",
            "--no-amp",
            "--seed",
            str(args.seed),
            "--max-subjects",
            "2",
            "--max-steps",
            "2",
            "--epochs",
            "1",
        ],
        products=smoke_products,
        training=True,
    )
    add(
        "R1-real",
        "smoke",
        common
        + fresh
        + with_roles
        + ["--max-subjects", "2", "--max-steps", "2", "--epochs", "1"],
        [roles],
        smoke_products,
        training=True,
    )
    if args.benchmark:
        pilot = stage_root / "R1-real/inference.pt"
        add(
            "R2-benchmark",
            "benchmark",
            common
            + with_roles
            + [
                "--checkpoint",
                str(pilot),
                "--split-role",
                "producer_fit",
                "--max-subjects",
                "2",
                "--max-states",
                "2",
                "--candidate-count",
                "4",
                "--teacher-mode",
                "iid_fixed_q",
                "--query-count",
                "64",
                "--repeats",
                str(args.benchmark_repeats),
            ],
            [roles, pilot],
        )
    budget = (
        ["--max-subjects", str(args.train_max_subjects)]
        if args.train_max_subjects
        else []
    )
    for arm in ("b0", "b1", "b2", "b_light"):
        flags = ["--base", arm, "--epochs", str(args.static_epochs)] + budget
        if args.static_max_steps is not None:
            flags += ["--max-steps", str(args.static_max_steps)]
        add(
            "R3-" + arm,
            "static-train",
            common + fresh + with_roles + flags,
            [roles],
            smoke_products,
            arm,
            True,
        )
    base = stage_root / f"R3-{args.base}/inference.pt"
    for arm in ("b0", "b1", "b2", "b_light"):
        static = stage_root / f"R3-{arm}/inference.pt"
        resolved = stage_root / f"R3-{arm}/resolved_config.json"
        # The published execution envelope contains the actual static variant;
        # evaluation has no --base flag and MAIN must never hydrate by guess.
        eval_common = list(common)
        eval_common[eval_common.index("--config") + 1] = str(resolved)
        add(
            f"R3-validation-{arm}",
            "evaluate",
            eval_common
            + with_roles
            + [
                "--checkpoint",
                str(static),
                "--scenario",
                "noop",
                "--budget",
                "0",
                "--split-role",
                "validation",
                "--max-subjects",
                "4",
            ],
            [roles, static, resolved],
            ("receipt.json", "paired_subjects.jsonl", "metrics.json"),
            arm,
            True,
        )
    for arm in ("u_only", "u_plus_spectral"):
        flags = [
            "--checkpoint",
            str(base),
            "--spectral-arm",
            arm,
            "--epochs",
            str(args.updater_epochs),
        ] + budget
        if args.updater_max_steps is not None:
            flags += ["--max-steps", str(args.updater_max_steps)]
        add(
            "R4-" + arm,
            "updater-train",
            common + with_roles + flags,
            [roles, base],
            smoke_products,
            arm,
            True,
        )
    for arm in ("u_only", "u_plus_spectral"):
        updater = stage_root / f"R4-{arm}/inference.pt"
        add(
            "R4B-" + arm,
            "headroom-evaluate",
            common
            + with_roles
            + [
                "--checkpoint",
                str(updater),
                "--base-checkpoint",
                str(base),
                "--split-role",
                "validation",
                "--max-subjects",
                "4",
                "--candidate-count",
                "32",
                "--query-count",
                "1024",
                "--teacher-mode",
                "iid_fixed_q",
                "--practical-margin",
                "0.0",
                "--exact-pool-audit",
            ],
            [roles, base, updater],
            (
                "receipt.json",
                "next1/headroom_metrics.json",
                "next1/next1_evidence.json",
                "next1/headroom_decision.json",
            ),
            arm,
            True,
        )
    return plan


def validate_live_flags(plan: list[Stage]) -> None:
    """Inspect literal current argparse contracts without importing torch or running CLI."""
    tree = ast.parse((REPO / "src/smagm/cli/pfgr_lite.py").read_text())
    common: set[str] = set()
    commands: dict[str, set[str]] = {}
    variables: dict[str, str] = {}
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or function.name not in {
            "_add_common",
            "_parser",
        }:
            continue
        for node in function.body:
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "add_parser"
            ):
                command = ast.literal_eval(node.value.args[0])
                variables[node.targets[0].id] = command
                commands[command] = set()
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "add_argument"
            ):
                flags = {
                    arg.value
                    for arg in call.args
                    if isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and arg.value.startswith("--")
                }
                if function.name == "_add_common":
                    common.update(flags)
                elif (
                    isinstance(call.func.value, ast.Name)
                    and call.func.value.id in variables
                ):
                    commands[variables[call.func.value.id]].update(flags)
            elif isinstance(call.func, ast.Name) and call.func.id == "_add_common":
                commands[variables[call.args[0].id]].update(common)
    for stage in plan:
        if stage.command not in ALLOWED or stage.command not in commands:
            raise ValueError(f"unsupported current CLI command: {stage.command}")
        unknown = {arg for arg in stage.argv[5:] if arg.startswith("--")} - commands[
            stage.command
        ]
        if unknown:
            raise ValueError(
                f"current CLI contract mismatch for {stage.name}: {sorted(unknown)}"
            )
        if stage.command == "evaluate" and (
            stage.argv[stage.argv.index("--scenario") + 1] != "noop"
            or stage.argv[stage.argv.index("--budget") + 1] != "0"
        ):
            raise ValueError(
                "runner evaluation is restricted to final static noop budget 0"
            )


def validate_inputs(args: argparse.Namespace) -> dict[str, str | None]:
    for name in (
        "data_root",
        "medicalnet_checkpoint",
        "split_file",
        "config",
        "python",
        "run_root",
        "roles_file",
    ):
        value = getattr(args, name)
        if value is not None:
            # Keep a venv interpreter symlink: resolving it selects system Python.
            setattr(
                args,
                name,
                value.expanduser().absolute()
                if name == "python"
                else value.expanduser().resolve(),
            )
    if not args.data_root.is_dir() and not args.plan_only:
        raise ValueError(f"missing BraTS directory: {args.data_root}")
    if not args.plan_only and (
        not args.python.is_file() or not os.access(args.python, os.X_OK)
    ):
        raise ValueError(f"Python interpreter is not executable: {args.python}")
    if not args.device.strip():
        raise ValueError("device must be explicit")
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    paths = [args.config, args.split_file, args.medicalnet_checkpoint]
    if args.roles_file:
        paths.append(args.roles_file)
        if args.roles_file.is_file():
            json_read(args.roles_file)
    for path in paths:
        if not path.is_file() and not args.plan_only:
            raise ValueError(f"missing input file: {path}")
    if args.split_file.is_file():
        json_read(args.split_file)
    config = json_read(args.config)
    pfgr = config.get("pfgr_config", {})
    if (
        config.get("schema_version") != "pfgr-lite-execution-config-v1"
        or pfgr.get("engineering_only") is not False
        or pfgr.get("num_points") != 2048
    ):
        raise ValueError(
            "real pipeline requires MAIN execution config with engineering_only=false and N=2048"
        )
    if config.get("stage_options", {}).get("max_updates") is not None:
        raise ValueError(
            "config max_updates must be null; use explicit runner --static-max-steps/--updater-max-steps"
        )
    variants = {
        "b0": "b0_legacy_v1",
        "b1": "b1_multiscale_v1",
        "b2": "b2_ordered_multiscale_v1",
        "b_light": "b_light_ordered_v1",
    }
    if pfgr.get("static", {}).get("variant") != variants[args.base]:
        raise ValueError(
            "--base must match config pfgr_config.static.variant for strict checkpoint hydration"
        )
    expected = args.medicalnet_sha256.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("MedicalNet expected SHA256 must contain 64 hex digits")
    hashes = {str(path): sha256(path) if path.is_file() else None for path in paths}
    if (
        hashes[str(args.medicalnet_checkpoint)] is not None
        and hashes[str(args.medicalnet_checkpoint)] != expected
    ):
        raise ValueError("MedicalNet checkpoint SHA256 mismatch")
    args.medicalnet_sha256 = expected
    return hashes


class Events:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def __call__(self, event: str, **fields: Any) -> None:
        payload = {
            "schema_version": PREFIX + "event-v1",
            "time": utc_now(),
            "event": event,
            **fields,
        }
        with self.lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()


def execute(
    stage: Stage, logs: Path, emit: Callable[..., None], timeout: int | None
) -> int:
    """Stream raw output to separate files and timestamped bounded JSONL chunks."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    env["PYTHONUNBUFFERED"] = "1"
    # This entrypoint intentionally has no W&B/network publishing switch.
    env["WANDB_MODE"] = "disabled"
    errors: list[BaseException] = []
    with (
        (logs / "stdout.txt").open("xb") as out,
        (logs / "stderr.txt").open("xb") as err,
    ):
        process = subprocess.Popen(
            stage.argv,
            cwd=REPO,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
        )

        def pump(pipe: Any, target: Any, stream: str) -> None:
            try:
                while chunk := pipe.read1(8192):
                    offset = target.tell()
                    target.write(chunk)
                    target.flush()
                    emit(
                        "output",
                        stage=stage.name,
                        stream=stream,
                        path=target.name,
                        offset_bytes=offset,
                        count_bytes=len(chunk),
                    )
            except BaseException as error:  # noqa: BLE001 - persist capture failure and stop child
                errors.append(error)
                process.terminate()
            finally:
                pipe.close()

        threads = [
            threading.Thread(target=pump, args=(process.stdout, out, "stdout")),
            threading.Thread(target=pump, args=(process.stderr, err, "stderr")),
        ]
        for thread in threads:
            thread.start()
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
            raise
        finally:
            for thread in threads:
                thread.join()
        if errors:
            raise RuntimeError(f"output capture failed: {errors[0]}")
    return code


def _scalars(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _scalars(item, f"{prefix}.{key}" if prefix else key)
    elif value is None or isinstance(value, (bool, int, float)):
        yield (
            prefix,
            None if isinstance(value, float) and not math.isfinite(value) else value,
        )
    elif isinstance(value, str) and any(
        word in prefix.rsplit(".", 1)[-1]
        for word in (
            "status",
            "reason",
            "version",
            "policy",
            "scope",
            "definition",
            "reduction",
            "device",
        )
    ):
        yield prefix, value


def collect_metrics(stage: Stage) -> list[dict[str, Any]]:
    """Copy measured scalars; no inferred counters, loss averages or new metrics."""
    root = Path(stage.run_dir)
    rows: list[dict[str, Any]] = []

    def add(
        payload: dict[str, Any],
        artifact: Path,
        scope: str,
        *,
        subject: str | None = None,
        random_seed: int | None = None,
        prefix: str = "",
    ) -> None:
        metric_scope = (
            "heldout_final_checkpoint"
            if stage.command in {"headroom-evaluate", "evaluate"}
            else "training_forward_pre_update"
            if stage.command in {"smoke", "static-train", "updater-train"}
            and scope == "training_pair"
            else "runtime_or_training_diagnostic"
        )
        for key, value in _scalars(payload):
            rows.append(
                dict(
                    zip(
                        CSV_FIELDS,
                        (
                            stage.name,
                            stage.arm,
                            subject,
                            stage.training_seed,
                            random_seed,
                            "random_control"
                            if random_seed is not None
                            else "training_initialization"
                            if stage.training_seed is not None
                            else None,
                            metric_scope,
                            scope,
                            key,
                            value,
                            str(artifact),
                            f"{prefix}.{key}".lstrip("."),
                        ),
                    )
                )
            )

    def paired(
        payload: Any,
        path: Path,
        scope: str,
        subject: str | None,
        seed: int | None,
        prefix: str,
    ) -> None:
        payload = payload if isinstance(payload, dict) else {}
        measured = {
            section: {
                **{key: None for key in METRIC_FIELDS},
                **payload.get(section, {}),
            }
            for section in ("before", "after", "improvement")
        }
        add(measured, path, scope, subject=subject, random_seed=seed, prefix=prefix)
        # Formula/mask/range/denominator fields remain at the exact source path.
        add(
            {k: v for k, v in payload.items() if k not in measured},
            path,
            scope + ":contract",
            subject=subject,
            random_seed=seed,
            prefix=prefix,
        )

    for relative in (
        "metrics.json",
        "benchmark.json",
        "parity.json",
        "stage_runtime.json",
        "environment.json",
        "weights.json",
    ):
        path = root / relative
        if path.is_file():
            payload = json_read(path)
            add(
                {
                    k: v
                    for k, v in payload.items()
                    if k
                    not in {
                        "history",
                        "paired_dense_metrics",
                        "paired_dense_metrics_aggregate",
                    }
                },
                path,
                relative,
            )
            for i, row in enumerate(payload.get("paired_dense_metrics", [])):
                paired(
                    row,
                    path,
                    "training_pair",
                    row.get("subject_id"),
                    None,
                    f"paired_dense_metrics[{i}]",
                )
    receipt = root / "receipt.json"
    if receipt.is_file():
        payload = json_read(receipt)
        for section in ("counts", "runtime_model", "environment", "seed_resolution"):
            add(payload.get(section, {}), receipt, "cli_" + section, prefix=section)
    for phase in ("s0", "s1"):
        history = root / phase / "stage_history.jsonl"
        if history.is_file():
            # The full immutable history stays in the evidence ZIP. Keep an
            # exact tail in the compact summary, never average update losses.
            with history.open(encoding="utf-8") as handle:
                tail = deque(enumerate(handle, start=1), maxlen=100)
            for line_number, line in tail:
                record = json.loads(line).get("record", {})
                subjects = record.get("subject_ids", [])
                subject = record.get("subject_id") or (
                    subjects[0] if len(subjects) == 1 else None
                )
                add(
                    record,
                    history,
                    "training_history_last_100_records",
                    subject=subject,
                    prefix=f"line[{line_number}].record",
                )
    subject_rows = root / "paired_subjects.jsonl"
    if subject_rows.is_file():
        with subject_rows.open(encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                row = json.loads(line)
                paired(
                    row,
                    subject_rows,
                    "static_final",
                    row.get("subject_id"),
                    None,
                    f"line[{i + 1}]",
                )
    headroom = root / "next1/headroom_metrics.json"
    if headroom.is_file():
        payload = json_read(headroom)
        add(
            {
                k: v
                for k, v in payload.items()
                if k not in {"subjects", "dense_metrics", "headroom_decision"}
            },
            headroom,
            "headroom_summary",
        )
        for i, record in enumerate(payload.get("subjects", [])):
            subject = record.get("subject_id")
            start = f"subjects[{i}]"
            for scope in ("no_op", "oracle"):
                paired(
                    record.get(scope, {}).get("metric"),
                    headroom,
                    scope,
                    subject,
                    None,
                    f"{start}.{scope}.metric",
                )
            for j, control in enumerate(record.get("random_controls", [])):
                paired(
                    control.get("metric"),
                    headroom,
                    "random",
                    subject,
                    control.get("seed"),
                    f"{start}.random_controls[{j}].metric",
                )
            for scope in (
                "exact_pool_audit",
                "diagnostic_timing",
                "timing_seconds",
                "write_norms",
            ):
                add(
                    record.get(scope, {}),
                    headroom,
                    scope,
                    subject=subject,
                    prefix=f"{start}.{scope}",
                )
            add(
                {
                    "oracle_minus_random": record.get("oracle_minus_random"),
                    "mask_denominator": record.get("mask_denominator"),
                },
                headroom,
                "headroom_subject",
                subject=subject,
                prefix=start,
            )
    return rows


def validation_subjects(stage: Stage) -> list[str] | None:
    root = Path(stage.run_dir)
    if stage.command == "evaluate":
        with (root / "paired_subjects.jsonl").open(encoding="utf-8") as handle:
            subjects = [
                json.loads(line).get("subject_id") for line in handle if line.strip()
            ]
    elif stage.command == "headroom-evaluate":
        subjects = [
            row.get("subject_id")
            for row in json_read(root / "next1/headroom_metrics.json").get(
                "subjects", []
            )
        ]
    else:
        return None
    if (
        len(subjects) != 4
        or len(set(subjects)) != 4
        or any(not isinstance(s, str) or not s for s in subjects)
    ):
        raise ValueError(
            "final checkpoint diagnostics require four distinct validation subject IDs"
        )
    return subjects


def package_stage(
    args: argparse.Namespace, destination: Path, run_dirs: list[str]
) -> Stage:
    argv = [str(args.python), "-u", "-m", "smagm.cli.pfgr_lite", "package"]
    for path in run_dirs + [str(destination / "metadata")]:
        argv += ["--run-dir", path]
    argv += ["--output-root", str(destination), "--run-name", "package"]
    argv += [
        "--max-file-size-mib",
        str(args.package_max_file_mib),
        "--max-archive-size-mib",
        str(args.package_max_archive_mib),
    ]
    out = destination / "package"
    return Stage(
        "package",
        "package",
        argv,
        str(out),
        [],
        [str(out / "receipt.json"), str(out / "manifest.json")],
    )


def run(
    args: argparse.Namespace, executor: Callable[..., int] = execute
) -> tuple[int, Path]:
    input_hashes = validate_inputs(args)
    name = (
        args.run_name
        or f"pfgr-r4-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:10]}"
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("run-name must be a simple alphanumeric directory name")
    destination = args.run_root / name
    plan = build_plan(args, destination)
    validate_live_flags(plan + [package_stage(args, destination, [])])
    args.run_root.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)  # Reservation rejects existing dirs and symlinks.
    metadata = destination / "metadata"
    logs = metadata / "stage_logs"
    metadata.mkdir()
    logs.mkdir()
    emit = Events(metadata / "pipeline_events.jsonl")
    manifest = {
        "schema_version": PREFIX + "manifest-v1",
        "created_at": utc_now(),
        "run_dir": str(destination),
        "mode": "plan_only" if args.plan_only else "execute",
        "profile": args.profile,
        "scientific_status": "NOT_EVALUATED",
        "r5_status": "CLOSED",
        "predeclared_base": args.base,
        "training_seed": args.seed,
        "random_control_seeds": [17, 29, 41],
        "input_sha256": input_hashes,
        "input_status": {
            path: "hashed" if digest else "unavailable_plan_only"
            for path, digest in input_hashes.items()
        },
        "data_root_status": "directory_present"
        if args.data_root.is_dir()
        else "unavailable_plan_only",
        "runner_sha256": sha256(Path(__file__)),
        "cli_sha256": sha256(REPO / "src/smagm/cli/pfgr_lite.py"),
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "stages": [asdict(stage) for stage in plan],
        "package_template": asdict(
            package_stage(args, destination, [stage.run_dir for stage in plan])
        ),
        "resume_policy": "none; every invocation requires a fresh run directory",
        "validation_scope": "stdlib input/hash/config/argv validation; runtime tensor/checkpoint/role/device validation belongs to CLI preflight and stage factories",
    }
    json_write(metadata / "pipeline_manifest.json", manifest)
    emit("plan", stage=None, mode=manifest["mode"], stages=len(plan))
    if args.plan_only:
        for stage in plan + [
            package_stage(args, destination, [s.run_dir for s in plan])
        ]:
            print(shlex.join(stage.argv), flush=True)
        return 0, destination

    records: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    dependencies = dict(input_hashes)
    failure: dict[str, Any] | None = None
    exit_code = 0
    cohort: list[str] | None = None

    def launch(stage: Stage) -> dict[str, Any]:
        started = time.perf_counter()
        # Packaging logs are outside its source directory, so source metadata
        # remains frozen while the CLI hashes and archives it.
        stage_logs = (
            destination / "package_logs"
            if stage.command == "package"
            else logs / stage.name
        )
        stage_logs.mkdir()
        stage_emit = (
            Events(stage_logs / "pipeline_events.jsonl")
            if stage.command == "package"
            else emit
        )
        record: dict[str, Any] = {
            "stage": stage.name,
            "command": stage.command,
            "arm": stage.arm,
            "argv": stage.argv,
            "status": "running",
            "started_at": utc_now(),
            "run_dir": stage.run_dir,
            "exit_code": None,
            "duration_seconds": None,
            "stdout": str(stage_logs / "stdout.txt"),
            "stderr": str(stage_logs / "stderr.txt"),
            "artifacts": [],
        }
        json_write(
            stage_logs / "command.json",
            {
                "schema_version": PREFIX + "command-v1",
                "argv": stage.argv,
                "cwd": str(REPO),
                "runner_sha256": manifest["runner_sha256"],
                "cli_sha256": manifest["cli_sha256"],
                "input_sha256": {p: dependencies.get(p) for p in stage.requires},
            },
        )
        stage_emit("stage_start", stage=stage.name, argv=stage.argv)
        print(f"[{utc_now()}] {stage.name} started", flush=True)
        try:
            if Path(stage.run_dir).exists() or Path(stage.run_dir).is_symlink():
                raise FileExistsError(f"refusing to overwrite stage: {stage.run_dir}")
            for path in stage.requires:
                actual = sha256(Path(path))
                if path not in dependencies or actual != dependencies[path]:
                    raise ValueError(
                        f"dependency missing from verified predecessor or changed: {path}"
                    )
            for path, digest in input_hashes.items():
                if sha256(Path(path)) != digest:
                    raise ValueError(f"input bytes changed since planning: {path}")
            if sha256(REPO / "src/smagm/cli/pfgr_lite.py") != manifest["cli_sha256"]:
                raise ValueError("CLI source changed during pipeline")
            record["exit_code"] = executor(
                stage, stage_logs, stage_emit, args.stage_timeout_seconds
            )
            record["process_exit_code"] = record["exit_code"]
            if record["exit_code"] != 0:
                raise RuntimeError(
                    f"{stage.name} exited with code {record['exit_code']}"
                )
            if stage.command in {"smoke", "static-train", "updater-train"}:
                effective = json_read(Path(stage.run_dir) / "resolved_config.json")
                if effective.get("stage_options", {}).get("seed") != args.seed:
                    raise ValueError(
                        "published StageOptions seed differs from requested training seed"
                    )
            for item in stage.produces:
                digest = sha256(Path(item))
                dependencies[item] = digest
                record["artifacts"].append(
                    {
                        "path": item,
                        "sha256": digest,
                        "size_bytes": Path(item).stat().st_size,
                    }
                )
            required = set(stage.produces)
            for path in sorted(Path(stage.run_dir).rglob("*")):
                if (
                    path.is_file()
                    and not path.is_symlink()
                    and str(path) not in required
                ):
                    record["artifacts"].append(
                        {
                            "path": str(path),
                            "sha256": sha256(path),
                            "size_bytes": path.stat().st_size,
                        }
                    )
            record["status"] = "succeeded"
        except BaseException as error:  # noqa: BLE001 - boundary records failures/interrupts before packaging
            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"
            record["traceback"] = traceback.format_exc()
            (stage_logs / "traceback.txt").write_text(
                record["traceback"], encoding="utf-8"
            )
            if record["exit_code"] in (None, 0):
                record["exit_code"] = (
                    130
                    if isinstance(error, KeyboardInterrupt)
                    else 124
                    if isinstance(error, subprocess.TimeoutExpired)
                    else 1
                )
        finally:
            record["duration_seconds"] = time.perf_counter() - started
            record["finished_at"] = utc_now()
            record["runner_sha256_at_exit"] = sha256(Path(__file__))
            record["cli_sha256_at_exit"] = sha256(REPO / "src/smagm/cli/pfgr_lite.py")
            json_write(
                stage_logs / "exit.json",
                {"schema_version": PREFIX + "exit-v1", **record},
            )
            stage_emit("stage_exit", **record)
            print(
                f"[{utc_now()}] {stage.name} {record['status']} ({record['duration_seconds']:.2f}s)",
                flush=True,
            )
        return record

    def summarize(package: dict[str, Any] | None = None) -> None:
        summary = {
            "schema_version": PREFIX + "summary-v1",
            "run_dir": str(destination),
            "status": "failed"
            if failure
            else "pipeline_stages_completed"
            if len(records) == len(plan)
            else "running",
            "scientific_status": "INCONCLUSIVE",
            "r5_status": "CLOSED",
            "profile": args.profile,
            "training_seed": args.seed,
            "predeclared_base": args.base,
            "failure": failure,
            "stages": records,
            "rows": metrics,
            "missing_values": "null means absent/nonfinite/unmeasured; no imputation or new aggregation",
            "package": package,
            "artifacts": {
                "manifest": str(metadata / "pipeline_manifest.json"),
                "events": str(metadata / "pipeline_events.jsonl"),
                "csv": str(metadata / "pipeline_metrics.csv"),
                "logs": str(logs),
            },
            "package_snapshot": "summary/events in ZIP are frozen immediately before packaging; final local summary adds package outcome",
        }
        json_write(metadata / "pipeline_summary.json", summary)
        with (metadata / "pipeline_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in metrics:
                writer.writerow({k: "null" if v is None else v for k, v in row.items()})

    for stage in plan:
        record = launch(stage)
        records.append(record)
        try:
            metrics.extend(collect_metrics(stage))
            if record["status"] == "succeeded":
                observed = validation_subjects(stage)
                if observed is not None:
                    if cohort is None:
                        cohort = observed
                    elif observed != cohort:
                        raise ValueError(
                            "final static/U audits disagree on ordered validation subject IDs"
                        )
                    record["validation_subject_ids"] = observed
        except Exception as error:  # noqa: BLE001 - malformed evidence must produce a failed summary
            record["status"] = "failed"
            record["error"] = (
                f"metric collection failed: {type(error).__name__}: {error}"
            )
            record["exit_code"] = 1
            emit("metric_collection_failed", stage=stage.name, error=record["error"])
            json_write(
                logs / stage.name / "exit.json",
                {"schema_version": PREFIX + "exit-v1", **record},
            )
        if record["status"] == "failed":
            failure = {"stage": stage.name, "error": record["error"]}
            exit_code = record["exit_code"]
        summarize()
        if failure:
            break
    run_dirs = [s.run_dir for s in plan if Path(s.run_dir).is_dir()]
    packaging = package_stage(args, destination, run_dirs)
    package_record = launch(packaging)
    if package_record["status"] == "succeeded":
        try:
            package_manifest = json_read(Path(packaging.run_dir) / "manifest.json")
            archive = package_manifest.get("archive", {})
            archive_path = Path(packaging.run_dir) / "evidence" / "evidence.zip"
            if archive.get("path") != "evidence.zip" or sha256(
                archive_path
            ) != archive.get("sha256"):
                raise ValueError(
                    "packaged archive is absent or fails its manifest hash"
                )
            included_names = {
                Path(row["source_path"]).name
                for row in package_manifest.get("included", [])
            }
            required_names = {
                "pipeline_manifest.json",
                "pipeline_summary.json",
                "pipeline_events.jsonl",
                "pipeline_metrics.csv",
            }
            if not required_names <= included_names:
                raise ValueError(
                    f"package omitted required runner metadata: {sorted(required_names - included_names)}"
                )
            package_record["archive"] = {**archive, "path": str(archive_path)}
            package_record["manifest"] = str(Path(packaging.run_dir) / "manifest.json")
            package_record["evidence_status"] = package_manifest.get("evidence_status")
        except (ValueError, OSError, KeyError, TypeError) as error:
            package_record.update(
                status="failed",
                error=f"archive verification failed: {error}",
                exit_code=1,
            )
            json_write(
                destination / "package_logs/exit.json",
                {"schema_version": PREFIX + "exit-v1", **package_record},
            )
    if package_record["status"] != "succeeded" and failure is None:
        failure = {"stage": "package", "error": package_record["error"]}
        exit_code = package_record["exit_code"]
    summarize(package_record)
    emit("pipeline_exit", stage=None, exit_code=exit_code, r5_status="CLOSED")
    return max(
        0, min(255, exit_code if exit_code >= 0 else 128 - exit_code)
    ), destination


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        code, destination = run(args)
    except (ValueError, TypeError, OSError) as error:
        print(f"PFGR pipeline input/reservation failure: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "exit_code": code,
                "run_dir": str(destination),
                "plan_only": args.plan_only,
                "r5_status": "CLOSED",
            }
        ),
        flush=True,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
