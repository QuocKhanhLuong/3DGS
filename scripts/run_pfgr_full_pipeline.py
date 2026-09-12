#!/usr/bin/env python3
"""Execute the predeclared exploratory PFGR R0--R10 DAG on explicit inputs.

Orchestration is stdlib-only. Tensor comparisons run in separate helper processes;
all model/data/role logic belongs to the existing strict PFGR CLI.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import shlex
import sys
import uuid
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
# Load the sibling by path: this works both as a script and under test importlib.
_spec = importlib.util.spec_from_file_location(
    "_pfgr_r4_primitives", Path(__file__).with_name("run_pfgr_r4_evidence.py")
)
assert _spec is not None and _spec.loader is not None
core = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = core
_spec.loader.exec_module(core)
Stage = core.Stage
ALLOWED = core.ALLOWED | {
    "bank-build",
    "bank-verify",
    "value-fit",
    "value-evaluate",
    "calibrate",
    "resume",
}
VALUE_VARIANTS = (126, 270, 366, 222)
POLICIES = ("random", "fixed_learned", "parallel_topk", "adaptive")
BUDGETS = (1, 2, 4)
RANDOM_SEEDS = (17, 29, 41)
PREFIX = core.PREFIX


def parser() -> argparse.ArgumentParser:
    p = core.parser()
    p.description = __doc__
    for action in p._actions:
        if action.dest == "profile":
            action.required = False
            action.choices = ("provisional",)
            action.default = "provisional"
    p.set_defaults(benchmark=True)
    p.add_argument(
        "--value-epochs",
        type=core.positive,
        required=True,
        help="explicit PROVISIONAL fit budget, separate from R10 mechanics",
    )
    p.add_argument("--value-batch-size", type=core.positive, default=32)
    p.add_argument("--value-learning-rate", type=float, default=1e-3)
    p.add_argument("--bank-max-subjects", type=core.positive, default=16)
    p.add_argument("--bank-max-states", type=core.positive, default=3)
    p.add_argument("--bank-candidate-count", type=core.positive, default=32)
    p.add_argument("--bank-replay-count", type=core.positive, default=2)
    p.add_argument("--teacher-query-count", type=core.positive, default=1024)
    p.add_argument(
        "--calibration-max-subjects",
        type=core.positive,
        default=128,
        help="cap on disjoint train-role collection; actual role minima still apply",
    )
    p.add_argument("--evaluation-max-subjects", type=core.positive, default=4)
    p.add_argument(
        "--real-resume-check",
        action="store_true",
        help="also run bounded real S0 interruption/resume/reference; separate from training budget",
    )
    return p


def flag(stage: Stage, name: str, default: Any = None) -> Any:
    return stage.argv[stage.argv.index(name) + 1] if name in stage.argv else default


def build_plan(args: argparse.Namespace, destination: Path) -> list[Stage]:
    plan = core.build_plan(args, destination)
    root = destination / "stages"
    roles = args.roles_file or root / "R0-preflight/roles.json"
    updater = root / "R4-u_plus_spectral/inference.pt"
    resolved = root / "R4-u_plus_spectral/resolved_config.json"
    bank = root / "R5-bank/s2/bank/index.json"
    value = root / "R6-fit-v366/value.pt"
    adaptive = root / "R7-calibration/adaptive.pt"
    cli = [str(args.python), "-u", "-m", "smagm.cli.pfgr_lite"]
    common = [
        "--config",
        str(resolved),
        "--data-root",
        str(args.data_root),
        "--split-file",
        str(args.split_file),
        "--roles-file",
        str(roles),
        "--device",
        args.device,
        "--no-amp",
        "--seed",
        str(args.seed),
        "--candidate-chunk-size",
        str(args.candidate_chunk_size),
        "--decode-chunk-size",
        str(args.decode_chunk_size),
        "--engineering-only",
    ]
    common_deps = [roles, updater, resolved]

    def add(name, command, flags, requires=(), products=("receipt.json",), arm=None):
        out = root / name
        stage = Stage(
            name,
            command,
            cli + [command] + flags + ["--output-root", str(root), "--run-name", name],
            str(out),
            [str(p) for p in requires],
            [str(out / p) for p in products],
            arm,
            args.seed,
        )
        plan.append(stage)
        return stage

    def helper(name, kind, inputs, payload=None):
        out = root / name
        product = {
            "value-join": "value_join.json",
            "resume-compare": "resume_comparison.json",
            "adaptive-check": "adaptive_availability.json",
        }[kind]
        specification = {"inputs": [str(p) for p in inputs], **(payload or {})}
        plan.append(
            Stage(
                name,
                "runner-helper",
                [
                    str(args.python),
                    "-u",
                    str(Path(__file__)),
                    "--helper",
                    kind,
                    "--spec-json",
                    json.dumps(specification, sort_keys=True),
                    "--output-dir",
                    str(out),
                ],
                str(out),
                [str(p) for p in inputs],
                [str(out / product)],
            )
        )

    add(
        "R5-bank",
        "bank-build",
        common
        + [
            "--checkpoint",
            str(updater),
            "--max-subjects",
            str(args.bank_max_subjects),
            "--max-states",
            str(args.bank_max_states),
            "--candidate-count",
            str(args.bank_candidate_count),
            "--query-count",
            str(args.teacher_query_count),
            "--teacher-mode",
            "iid_fixed_q",
        ],
        common_deps,
        ("receipt.json", "s2/bank/index.json", "resolved_config.json", "metrics.json"),
        "u_plus_spectral",
    )
    add(
        "R5-verify",
        "bank-verify",
        common
        + [
            "--checkpoint",
            str(updater),
            "--bank-index",
            str(bank),
            "--replay-count",
            str(args.bank_replay_count),
        ],
        common_deps + [bank],
    )
    fit_flags = [
        "--checkpoint",
        str(updater),
        "--bank-index",
        str(bank),
        "--epochs",
        str(args.value_epochs),
        "--batch-size",
        str(args.value_batch_size),
        "--learning-rate",
        str(args.value_learning_rate),
    ]
    pair_files = []
    for variant in VALUE_VARIANTS:
        fit = add(
            f"R6-fit-v{variant}",
            "value-fit",
            common + fit_flags + ["--value-input", str(variant)],
            common_deps + [bank, root / "R5-verify/receipt.json"],
            ("receipt.json", "value.pt", "value_fit.json"),
            f"v{variant}",
        )
        evaluation = add(
            f"R6-evaluate-v{variant}",
            "value-evaluate",
            common
            + [
                "--checkpoint",
                str(updater),
                "--bank-index",
                str(bank),
                "--value-checkpoint",
                str(Path(fit.run_dir) / "value.pt"),
                "--batch-size",
                str(args.value_batch_size),
            ],
            common_deps + [bank, Path(fit.run_dir) / "value.pt"],
            ("receipt.json", "value_evaluate.json", "value_evaluate_pairs.json"),
            f"v{variant}",
        )
        pair_files.append(Path(evaluation.run_dir) / "value_evaluate_pairs.json")
    helper("R6-same-bank-join", "value-join", [bank] + pair_files)
    add(
        "R7-calibration",
        "calibrate",
        common
        + [
            "--checkpoint",
            str(updater),
            "--value-checkpoint",
            str(value),
            "--exploratory-run",
            "--max-subjects",
            str(args.calibration_max_subjects),
            "--teacher-mode",
            "iid_fixed_q",
            "--query-count",
            str(args.teacher_query_count),
        ],
        common_deps + [value, root / "R6-evaluate-v366/receipt.json"],
        ("receipt.json", "calibration.json", "execution_authorization.json"),
        "v366",
    )
    helper(
        "R7-adaptive-availability",
        "adaptive-check",
        [root / "R7-calibration/calibration.json", updater, value],
        {
            "adaptive_path": str(adaptive),
            "producer_path": str(updater),
            "value_path": str(value),
        },
    )
    for phase, role in (("R8", "validation"), ("R9", "test")):
        for policy, budget, seed in [
            (p, 0, s) for s in RANDOM_SEEDS for p in ("noop", "static")
        ] + [(p, k, s) for s in RANDOM_SEEDS for p in POLICIES for k in BUDGETS]:
            checkpoint = adaptive if policy == "adaptive" else updater
            deps = common_deps + [checkpoint]
            flags = common + [
                "--checkpoint",
                str(checkpoint),
                "--scenario",
                policy,
                "--budget",
                str(budget),
                "--split-role",
                role,
                "--max-subjects",
                str(args.evaluation_max_subjects),
                "--teacher-mode",
                "iid_fixed_q",
                "--query-count",
                str(args.teacher_query_count),
            ]
            # One explicit seed token; validation and test share the predeclared matrix.
            flags[flags.index("--seed") + 1] = str(seed)
            if policy in {"fixed_learned", "parallel_topk", "adaptive"}:
                flags += ["--value-checkpoint", str(value)]
                deps += [value, root / "R6-evaluate-v366/receipt.json"]
            if policy == "adaptive":
                deps.append(
                    root / "R7-adaptive-availability/adaptive_availability.json"
                )
            if role == "test":
                flags += ["--exploratory-run"]
            add(
                f"{phase}-{policy}-k{budget}-seed{seed}",
                "evaluate",
                flags,
                deps,
                (
                    "receipt.json",
                    "metrics.json",
                    "paired_subjects.jsonl",
                    "action_metrics.jsonl",
                )
                + (("execution_authorization.json",) if role == "test" else ()),
                policy,
            )
    # Mechanics budgets are fixed and intentionally separate from R3/R4/R6 fit budgets.
    smoke_products = (
        "receipt.json",
        "inference.pt",
        "resume.pt",
        "stage_runtime.json",
        "resolved_config.json",
    )
    for kind in ("synthetic", "real") if args.real_resume_check else ("synthetic",):
        if kind == "synthetic":
            flags = [
                "--synthetic",
                "--engineering-only",
                "--config",
                str(REPO / "configs/pfgr_lite/synthetic.json"),
                "--device",
                "cpu",
                "--no-amp",
                "--seed",
                str(args.seed),
            ]
            dependencies = []
            fresh = []
        else:
            flags = list(common)
            flags[flags.index("--config") + 1] = str(args.config)
            dependencies = [roles]
            fresh = [
                "--medicalnet-checkpoint",
                str(args.medicalnet_checkpoint),
                "--medicalnet-sha256",
                args.medicalnet_sha256,
            ]
        mechanics = flags + [
            "--epochs",
            "2",
            "--max-subjects",
            "2",
            "--batch-size",
            "1",
        ]
        interrupted = add(
            f"R10-{kind}-interrupted",
            "smoke",
            mechanics + fresh + ["--max-steps", "1"],
            dependencies,
            smoke_products,
        )
        resume_source = Path(interrupted.run_dir) / "resume.pt"
        resumed = add(
            f"R10-{kind}-resumed",
            "resume",
            mechanics + ["--resume-checkpoint", str(resume_source), "--max-steps", "2"],
            dependencies + [resume_source],
            smoke_products,
        )
        reference = add(
            f"R10-{kind}-uninterrupted",
            "smoke",
            mechanics + fresh + ["--max-steps", "2"],
            dependencies,
            smoke_products,
        )
        helper(
            f"R10-{kind}-comparison",
            "resume-compare",
            [
                resume_source,
                Path(resumed.run_dir) / "resume.pt",
                Path(reference.run_dir) / "resume.pt",
            ],
            {"kind": kind},
        )
    cached = common + [
        "--bank-index",
        str(bank),
        "--value-input",
        "366",
        "--epochs",
        "3",
        "--batch-size",
        str(args.value_batch_size),
        "--learning-rate",
        str(args.value_learning_rate),
    ]
    cached_products = ("receipt.json", "value-resume.pt", "value_fit_incomplete.json")
    interrupted = add(
        "R10-value-interrupted",
        "value-fit",
        cached + ["--checkpoint", str(updater), "--max-steps", "1"],
        common_deps + [bank],
        cached_products,
    )
    resume_source = Path(interrupted.run_dir) / "value-resume.pt"
    resumed = add(
        "R10-value-resumed",
        "resume",
        cached + ["--resume-checkpoint", str(resume_source), "--max-steps", "2"],
        common_deps + [bank, resume_source],
        cached_products,
    )
    reference = add(
        "R10-value-uninterrupted",
        "value-fit",
        cached + ["--checkpoint", str(updater), "--max-steps", "2"],
        common_deps + [bank],
        cached_products,
    )
    helper(
        "R10-value-comparison",
        "resume-compare",
        [
            resume_source,
            Path(resumed.run_dir) / "value-resume.pt",
            Path(reference.run_dir) / "value-resume.pt",
        ],
        {"kind": "value"},
    )
    # Config files (including upstream resolved envelopes) are byte dependencies too.
    return [
        replace(
            s,
            requires=list(dict.fromkeys(s.requires + [str(Path(flag(s, "--config")))])),
        )
        if flag(s, "--config")
        else s
        for s in plan
    ]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL object rows required: {path}")
    return rows


def finite(value: Any) -> float | None:
    return (
        float(value)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        else None
    )


def value_join(paths: list[Path]) -> dict[str, Any]:
    index, *pair_paths = paths
    bank = core.json_read(index)
    pairs = [core.json_read(path) for path in pair_paths]
    if len(pairs) != len(VALUE_VARIANTS) or {
        p.get("input_variant") for p in pairs
    } != set(VALUE_VARIANTS):
        raise ValueError("same-bank join requires V126/V270/V366/V222 exactly once")
    for key in ("bank_manifest_hash", "gain_scale_hash", "row_count", "group_count"):
        if pairs[0].get(key) is None or any(p.get(key) != pairs[0][key] for p in pairs):
            raise ValueError(f"same-bank pair metadata mismatch: {key}")
    if pairs[0]["bank_manifest_hash"] != core.sha256(index):
        raise ValueError("V pairs do not bind actual bank index bytes")
    if pairs[0]["gain_scale_hash"] != bank.get("gain_scale", {}).get("digest"):
        raise ValueError("V pairs do not bind bank gain scale")
    indexed = {r["row_id"]: r for r in bank["rows"]}
    if len(indexed) != len(bank["rows"]):
        raise ValueError("bank row IDs are duplicated")
    rows_by_variant = {}
    for pair in pairs:
        rows = pair.get("rows", [])
        mapped = {r["row_key"]: r for r in rows}
        if not mapped or len(mapped) != len(rows) or len(rows) != pair["row_count"]:
            raise ValueError("empty, duplicate, or incomplete V paired rows")
        for row in rows:
            entry = indexed.get(row["row_id"], {})
            if any(row.get(k) != entry.get(k) for k in ("row_hash", "shard", "offset")):
                raise ValueError("paired row does not join immutable bank index")
            if row.get("bank_manifest_hash") != pair["bank_manifest_hash"]:
                raise ValueError("paired row bank differs from envelope")
        rows_by_variant[pair["input_variant"]] = mapped
    reference = rows_by_variant[366]
    identity_keys = (
        "row_hash",
        "row_id",
        "shard",
        "offset",
        "bank_manifest_hash",
        "subject_id",
        "context_id",
        "state_version",
        "point_id",
        "action_id",
        "proposal_hash",
        "state_digest",
        "measured_raw_gain",
        "group_key",
        "measured_rank",
    )
    for mapped in rows_by_variant.values():
        if set(mapped) != set(reference):
            raise ValueError("V variants evaluated different immutable row sets")
        if any(
            any(mapped[key].get(field) != row.get(field) for field in identity_keys)
            for key, row in reference.items()
        ):
            raise ValueError("V paired row action/context/target identity mismatch")
    return {
        "schema_version": "pfgr-lite-full-value-join-v1",
        "status": "SOFTWARE_PASS",
        "scientific_status": "NOT_EVALUATED",
        "scope": "same-bank cached fit/evaluation; no held-out V generalization claim",
        "bank_index_sha256": core.sha256(index),
        "bank_manifest_hash": pairs[0]["bank_manifest_hash"],
        "gain_scale_hash": pairs[0]["gain_scale_hash"],
        "row_count": len(reference),
        "predeclared_downstream_variant": 366,
        "source_sha256": {str(p): core.sha256(p) for p in paths},
        "rows": [
            {
                "row_key": key,
                **{f: row.get(f) for f in identity_keys},
                "variants": {
                    str(v): {
                        k: rows_by_variant[v][key].get(k)
                        for k in ("predicted_scaled", "predicted_raw", "predicted_rank")
                    }
                    for v in VALUE_VARIANTS
                },
            }
            for key, row in reference.items()
        ],
    }


def resume_comparison(paths: list[Path], kind: str) -> dict[str, Any]:
    """Compare actual strict snapshots, never just their names or printed cursors."""
    import numpy as np
    import torch

    from smagm.features.point_guided.pfgr_lite.checkpoint import load_resume

    interrupted, resumed, reference = [load_resume(p) for p in paths]
    if [b.stage_state.update for b in (interrupted, resumed, reference)] != [1, 2, 2]:
        raise ValueError(
            "resume mechanics require actual committed cursors 1 -> 2 versus uninterrupted 2"
        )
    if interrupted.stage_state.completion != "pending":
        raise ValueError("the interruption source was already complete")
    differences = []
    tensor_count = 0

    def compare(left, right, name):
        nonlocal tensor_count
        if isinstance(left, torch.Tensor):
            tensor_count += 1
            equal = (
                isinstance(right, torch.Tensor)
                and left.dtype == right.dtype
                and left.shape == right.shape
                and torch.equal(left.cpu(), right.cpu())
            )
        elif isinstance(left, np.ndarray):
            equal = (
                isinstance(right, np.ndarray)
                and left.dtype == right.dtype
                and np.array_equal(left, right)
            )
        elif isinstance(left, dict):
            equal = isinstance(right, dict) and set(left) == set(right)
            if equal:
                for key in left:
                    compare(left[key], right[key], f"{name}.{key}")
                return
        elif isinstance(left, (list, tuple)):
            equal = isinstance(right, type(left)) and len(left) == len(right)
            if equal:
                for i, (a, b) in enumerate(zip(left, right)):
                    compare(a, b, f"{name}[{i}]")
                return
        else:
            equal = left == right
        if not equal:
            differences.append(name)

    compare(asdict(resumed.stage_state), asdict(reference.stage_state), "stage_state")
    if kind == "value":
        payloads = [
            b.bank_state["cached_value_fit"] for b in (interrupted, resumed, reference)
        ]
        left, right = payloads[1:]
        immutable = (
            "bank_manifest_hash",
            "gain_scale_hash",
            "producer_compatibility_hash",
            "input_variant",
            "fit_config_hash",
            "initial_weights_hash",
            "optimizer_provenance",
        )
        for key in immutable:
            if key not in payloads[0]:
                raise ValueError(f"cached fit missing immutable identity {key}")
            for payload in payloads[1:]:
                compare(payloads[0][key], payload[key], f"identity.{key}")
        for key in (
            "model_state_dict",
            "optimizer_state",
            "stage_payload",
            "rng_state",
            "bank_state",
        ):
            if key not in left or key not in right:
                raise ValueError(f"cached fit missing comparison state {key}")
            compare(left[key], right[key], key)
        weights = left["model_state_dict"]
        old_weights = payloads[0]["model_state_dict"]
        cursors = [p["stage_payload"] for p in payloads]
    else:
        payloads = [
            b.bank_state["stage_runtime"] for b in (interrupted, resumed, reference)
        ]
        left, right = payloads[1:]
        for key in ("training_config_hash", "input_manifest_hash", "split_role_hash"):
            for payload in payloads[1:]:
                compare(payloads[0][key], payload[key], f"identity.{key}")
        for key in ("optimizer_state", "rng_state", "cursor", "parameter_names"):
            compare(left[key], right[key], key)
        compare(
            dict(resumed.inference.state_dict),
            dict(reference.inference.state_dict),
            "weights",
        )
        weights = dict(resumed.inference.state_dict)
        old_weights = dict(interrupted.inference.state_dict)
        cursors = [p["cursor"] for p in payloads]
    changed = sum(not torch.equal(weights[k], old_weights[k]) for k in weights)
    if tensor_count == 0 or not weights or changed == 0:
        raise ValueError("resume did not demonstrate actual tensor state advancement")
    result = {
        "schema_version": "pfgr-lite-full-resume-comparison-v1",
        "kind": kind,
        "status": "SOFTWARE_PASS" if not differences else "MISMATCH",
        "scientific_status": "NOT_EVALUATED",
        "equality": "exact CPU tensor/value equality; no tolerance",
        "source_sha256": {str(p): core.sha256(p) for p in paths},
        "committed_updates": [
            b.stage_state.update for b in (interrupted, resumed, reference)
        ],
        "cursor_fields_compared": [sorted(c) for c in cursors],
        "tensor_comparisons": tensor_count,
        "changed_tensors_after_resume": changed,
        "difference_paths": differences,
    }
    return result


def adaptive_availability(specification: dict[str, Any]) -> dict[str, Any]:
    """Valid diagnostic absence is successful evidence; malformed/failed output is not."""
    path = Path(specification["inputs"][0])
    payload = core.json_read(path)
    calibration = payload.get("calibration")
    adaptive = Path(specification["adaptive_path"])
    result = {
        "schema_version": "pfgr-lite-full-adaptive-availability-v1",
        "scientific_status": "NOT_EVALUATED",
        "calibration_sha256": core.sha256(path),
        "adaptive_path": str(adaptive),
        "adaptive_sha256": None,
    }
    if "calibration" not in payload or payload.get("status") not in {
        "INCONCLUSIVE",
        "SOFTWARE_PASS",
    }:
        raise ValueError("calibration output is not a recognized completed diagnostic")
    capability = (
        calibration.get("capability") if isinstance(calibration, dict) else None
    )
    if calibration is None or capability == "diagnostic":
        if adaptive.exists():
            raise ValueError(
                "diagnostic calibration unexpectedly published adaptive.pt"
            )
        result.update(
            status="NOT_AVAILABLE",
            reason="insufficient_calibration_data"
            if calibration is None
            else "diagnostic_calibration_capability",
        )
        return result
    if capability != "adaptive" or not adaptive.is_file():
        raise ValueError(
            "adaptive calibration claims capability without a published artifact"
        )
    from smagm.features.point_guided.pfgr_lite.checkpoint import (
        load_inference_bundle,
        load_value_artifact,
    )

    producer = load_inference_bundle(Path(specification["producer_path"]))
    value = load_value_artifact(
        Path(specification["value_path"]),
        expected_producer=producer.producer,
        expected_role_manifest=producer.role_manifest,
    )
    bundle = load_inference_bundle(adaptive)
    if (
        bundle.capability != "adaptive"
        or bundle.producer.compatibility_hash != producer.producer.compatibility_hash
    ):
        raise ValueError("adaptive artifact capability/producer mismatch")
    if (
        bundle.value_fit_identity is None
        or bundle.value_fit_identity.digest != value.value_fit_identity.digest
    ):
        raise ValueError("adaptive artifact uses a different V identity")
    result.update(
        status="AVAILABLE", reason=None, adaptive_sha256=core.sha256(adaptive)
    )
    return result


def helper_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Internal immutable evidence helper; no data loading or training"
    )
    p.add_argument(
        "--helper",
        choices=("value-join", "resume-compare", "adaptive-check"),
        required=True,
    )
    p.add_argument("--spec-json", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    specification = json.loads(args.spec_json)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    paths = [Path(p) for p in specification["inputs"]]
    if args.helper == "value-join":
        result, filename = value_join(paths), "value_join.json"
    elif args.helper == "resume-compare":
        result, filename = (
            resume_comparison(paths, specification["kind"]),
            "resume_comparison.json",
        )
    else:
        result, filename = (
            adaptive_availability(specification),
            "adaptive_availability.json",
        )
    core.json_write(args.output_dir / filename, result)
    return 1 if result["status"] == "MISMATCH" else 0


def collect_metrics(stage: Stage) -> list[dict[str, Any]]:
    rows = core.collect_metrics(stage)
    if stage.name.startswith(("R8-", "R9-")):
        for row in rows:
            row.update(
                random_seed=int(flag(stage, "--seed")),
                seed_kind="policy_randomization",
                metric_scope="target_after_policy_evaluation",
            )
            if row["scope"] == "static_final":
                row["scope"] = "policy_final"
    root = Path(stage.run_dir)
    # Exact raw measurements including V constants/rank/regret/sign and calibrated a/b.
    for filename in (
        "value_fit.json",
        "value_evaluate.json",
        "value_fit_incomplete.json",
        "calibration.json",
        "calibration/metrics.json",
        "calibration/calibration_metrics.json",
        "resume_comparison.json",
        "adaptive_availability.json",
    ):
        path = root / filename
        if not path.is_file():
            continue
        payload = core.json_read(path)
        for key, value in core._scalars(payload):
            rows.append(
                dict(
                    zip(
                        core.CSV_FIELDS,
                        (
                            stage.name,
                            stage.arm,
                            None,
                            stage.training_seed,
                            None,
                            "training_initialization",
                            "cached_value_or_calibration_or_resume_diagnostic",
                            filename,
                            key,
                            value,
                            str(path),
                            key,
                        ),
                    )
                )
            )
    # CLI counts/metrics retain collected calibration work, including null counters.
    receipt = root / "receipt.json"
    if receipt.is_file() and stage.command in {
        "calibrate",
        "bank-build",
        "bank-verify",
        "resume",
    }:
        for key, value in core._scalars(core.json_read(receipt).get("metrics", {})):
            rows.append(
                dict(
                    zip(
                        core.CSV_FIELDS,
                        (
                            stage.name,
                            stage.arm,
                            None,
                            stage.training_seed,
                            None,
                            None,
                            "runtime_or_calibration_diagnostic",
                            "receipt_metrics",
                            key,
                            value,
                            str(receipt),
                            "metrics." + key,
                        ),
                    )
                )
            )
    return rows


def control_join(plan: list[Stage], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Exact per-subject pairing against the same-seed/role noop from the same U."""
    successful = {r["stage"] for r in records if r["status"] == "succeeded"}
    observed = {}
    unavailable = []
    for stage in plan:
        if not stage.name.startswith(("R8-", "R9-")):
            continue
        if stage.name not in successful:
            unavailable.append(stage.name)
            continue
        rows = read_jsonl(Path(stage.run_dir) / "paired_subjects.jsonl")
        observed[stage.name] = (stage, rows)
    joined = []
    for name, (stage, rows) in observed.items():
        seed = int(flag(stage, "--seed"))
        reference_name = f"{name[:2]}-noop-k0-seed{seed}"
        reference = observed.get(reference_name)
        if reference is None:
            unavailable.append(name + ": missing same-seed noop")
            continue
        references = {r["subject_id"]: r for r in reference[1]}
        if [r["subject_id"] for r in rows] != [r["subject_id"] for r in reference[1]]:
            raise ValueError(f"control cohort mismatch: {name}")
        for row in rows:
            baseline = references[row["subject_id"]]
            for key in ("context_id", "z0_digest", "z0_state_digest", "before"):
                if row.get(key) is None or row.get(key) != baseline.get(key):
                    raise ValueError(
                        f"control initial-state/context mismatch: {name}:{row['subject_id']}:{key}"
                    )
            measured = {}
            for metric in core.METRIC_FIELDS:
                a, b = (
                    finite(row.get("after", {}).get(metric)),
                    finite(baseline.get("after", {}).get(metric)),
                )
                measured[metric] = (
                    None
                    if a is None or b is None
                    else a - b
                    if metric in {"psnr", "ssim"}
                    else b - a
                )
            joined.append(
                {
                    "stage": name,
                    "split_role": flag(stage, "--split-role"),
                    "seed": seed,
                    "subject_id": row["subject_id"],
                    "context_id": row["context_id"],
                    "requested_budget": int(flag(stage, "--budget")),
                    "policy": flag(stage, "--scenario"),
                    "measured_k": row.get("route", {}).get("k"),
                    "stop_reason": row.get("route", {}).get("stop_reason"),
                    "pipeline_elapsed_seconds": finite(
                        row.get("pipeline_elapsed_seconds")
                    ),
                    "after": row.get("after"),
                    "improvement_over_noop": measured,
                    "source": str(Path(stage.run_dir) / "paired_subjects.jsonl"),
                    "source_sha256": core.sha256(
                        Path(stage.run_dir) / "paired_subjects.jsonl"
                    ),
                }
            )
    return {
        "schema_version": "pfgr-lite-full-control-join-v1",
        "status": "SOFTWARE_PASS" if joined else "NOT_AVAILABLE",
        "scientific_status": "NOT_EVALUATED",
        "downstream_variant": 366,
        "rows": joined,
        "unavailable": unavailable,
        "frontier": None,
        "frontier_reason": "no aggregate winner/frontier selected; exact per-subject measured quality and elapsed time remain available",
        "metric_definition": "same role/seed/subject/context/Z0: noop.after minus policy.after for MAE/Charbonnier; reverse for PSNR/SSIM; null if unmeasured",
        "latency_definition": "source pipeline_elapsed_seconds includes route, decode, and post-prediction metric/teacher work; not deployment-only latency",
    }


def declared_cohort(stage: Stage, roles: dict[str, Any]) -> list[str] | None:
    if stage.command not in {"evaluate", "headroom-evaluate"}:
        return None
    role = flag(stage, "--split-role")
    key = {
        "validation": "baseline_validation_subject_ids",
        "test": "baseline_test_subject_ids",
    }.get(role)
    if key is None:
        return None
    subjects = roles.get(key)
    if (
        not isinstance(subjects, list)
        or not subjects
        or any(not isinstance(s, str) or not s for s in subjects)
    ):
        raise ValueError(f"role manifest lacks declared {role} cohort")
    return subjects[: int(flag(stage, "--max-subjects"))]


def package_stage(args: argparse.Namespace, destination: Path) -> Stage:
    """Archive the stage tree as one source, keeping argv bounded for 122 stages."""
    return core.package_stage(args, destination, [str(destination / "stages")])


def verify_package(record: dict[str, Any], stage: Stage) -> None:
    if record["status"] != "succeeded":
        return
    root = Path(stage.run_dir)
    manifest = core.json_read(root / "manifest.json")
    archive = root / "evidence/evidence.zip"
    digest = manifest.get("archive", {}).get("sha256")
    if (
        manifest.get("archive", {}).get("path") != "evidence.zip"
        or core.sha256(archive) != digest
    ):
        raise ValueError("packaged archive is absent or fails its manifest hash")
    included = {Path(row["source_path"]).name for row in manifest.get("included", [])}
    required = {
        "pipeline_manifest.json",
        "pipeline_summary.json",
        "pipeline_events.jsonl",
        "pipeline_metrics.csv",
        "control_join.json",
    }
    if not required <= included:
        raise ValueError(
            f"package omitted runner evidence: {sorted(required - included)}"
        )
    record.update(
        archive={**manifest["archive"], "path": str(archive)},
        manifest=str(root / "manifest.json"),
        evidence_status=manifest.get("evidence_status"),
    )


def run(args: argparse.Namespace, executor=core.execute) -> tuple[int, Path]:
    input_hashes = core.validate_inputs(args)
    if args.profile != "provisional":
        raise ValueError(
            "full exploratory execution requires explicit PROVISIONAL budget scope"
        )
    if not math.isfinite(args.value_learning_rate) or args.value_learning_rate <= 0:
        raise ValueError("value-learning-rate must be finite and positive")
    if args.teacher_query_count < 2:
        raise ValueError("IID teacher-query-count must be at least two")
    name = (
        args.run_name
        or f"pfgr-full-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:10]}"
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("run-name must be a simple alphanumeric directory name")
    destination = args.run_root / name
    plan = build_plan(args, destination)
    core.validate_live_flags(
        [s for s in plan if s.command != "runner-helper"]
        + [package_stage(args, destination)],
        allowed=ALLOWED,
        static_only=False,
    )
    # Pin all external file prerequisites, including synthetic mechanics config.
    products = {p for stage in plan for p in stage.produces}
    for stage in plan:
        for path in stage.requires:
            if path not in products and not Path(path).is_relative_to(destination):
                input_hashes[path] = (
                    core.sha256(Path(path)) if Path(path).is_file() else None
                )
    input_hashes[str(Path(core.__file__))] = core.sha256(Path(core.__file__))
    args.run_root.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    (destination / "stages").mkdir()
    metadata = destination / "metadata"
    logs = metadata / "stage_logs"
    logs.mkdir(parents=True)
    emit = core.Events(metadata / "pipeline_events.jsonl")
    manifest = {
        "schema_version": PREFIX + "manifest-v1",
        "created_at": core.utc_now(),
        "run_dir": str(destination),
        "full_pipeline": True,
        "last_stage": "R10",
        "mode": "plan_only" if args.plan_only else "execute",
        "profile": "PROVISIONAL",
        "scientific_status": "NOT_EVALUATED",
        "execution_scope": "EXPLORATORY",
        "human_reviewed": False,
        "authorizes_main": False,
        "predeclared_base": "b2",
        "predeclared_updater": "u_plus_spectral",
        "predeclared_downstream_value": 366,
        "random_control_seeds": list(RANDOM_SEEDS),
        "budgets": [0, *BUDGETS],
        "training_seed": args.seed,
        "input_sha256": input_hashes,
        "runner_sha256": core.sha256(Path(__file__)),
        "cli_sha256": core.sha256(REPO / "src/smagm/cli/pfgr_lite.py"),
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "stages": [asdict(s) for s in plan],
        "package_template": asdict(package_stage(args, destination)),
        "cohort_rule": "ordered role-manifest IDs truncated by the predeclared cap; role/data CLI validation remains authoritative",
        "test_selection": "predeclared controls and seeds; no winner chosen from test",
        "r10_budget": "smoke epochs2 batch1 subjects2; cached V epochs3 same bank/seed/batch/lr; interrupted max1 -> resumed max2 versus uninterrupted max2",
        "resume_policy": "fresh top-level directory; only explicit within-run R10 continuation",
        "input_status": {
            p: "hashed" if h else "unavailable_plan_only"
            for p, h in input_hashes.items()
        },
    }
    core.json_write(metadata / "pipeline_manifest.json", manifest)
    emit("plan", stage=None, mode=manifest["mode"], stages=len(plan), last_stage="R10")
    if args.plan_only:
        for stage in plan + [package_stage(args, destination)]:
            print(shlex.join(stage.argv), flush=True)
        return 0, destination

    dependencies = dict(input_hashes)
    owners = {p: s.name for s in plan for p in s.produces}
    records: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    availability = {"status": "PENDING", "reason": "calibration has not completed"}
    cohorts: dict[str, list[str]] = {}
    join = {
        "schema_version": "pfgr-lite-full-control-join-v1",
        "status": "NOT_AVAILABLE",
        "rows": [],
        "reason": "evaluation has not completed",
    }
    core.json_write(metadata / "control_join.json", join)
    cancelled = False

    def summarize(package=None):
        errors = [r for r in records if r["status"] in {"failed", "blocked"}]
        missing = [r for r in records if r["status"] == "not_available"]
        status = (
            "error"
            if errors or package and package["status"] != "succeeded"
            else ("completed_with_unavailable" if missing else "completed")
            if len(records) == len(plan)
            else "running"
        )
        summary = {
            "schema_version": PREFIX + "summary-v1",
            "full_pipeline": True,
            "last_stage": "R10",
            "run_dir": str(destination),
            "status": status,
            "scientific_status": "NOT_EVALUATED",
            "profile": "PROVISIONAL",
            "execution_scope": "EXPLORATORY",
            "human_reviewed": False,
            "authorizes_main": False,
            "adaptive": availability,
            "stages": records,
            "rows": metrics,
            "cohorts": {k: {"subject_ids": v} for k, v in cohorts.items()},
            "control_join": str(metadata / "control_join.json"),
            "errors": [
                {k: r.get(k) for k in ("stage", "status", "error")} for r in errors
            ],
            "unavailable_count": len(missing),
            "completed_stage_count": sum(r["status"] == "succeeded" for r in records),
            "missing_values": "null means absent/nonfinite/unmeasured; no imputation",
            "package": package,
            "package_snapshot": "ZIP metadata frozen before packaging; local summary adds archive result",
        }
        core.json_write(metadata / "pipeline_summary.json", summary)
        with (metadata / "pipeline_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=core.CSV_FIELDS)
            writer.writeheader()
            for row in metrics:
                writer.writerow({k: "null" if v is None else v for k, v in row.items()})

    for stage in plan:
        completed = {r["stage"] for r in records if r["status"] == "succeeded"}
        reason = None
        status = "blocked"
        if (
            flag(stage, "--scenario") == "adaptive"
            and availability["status"] == "NOT_AVAILABLE"
        ):
            status, reason = "not_available", availability["reason"]
        else:
            bad = [
                p
                for p in stage.requires
                if p not in dependencies or (p in owners and owners[p] not in completed)
            ]
            if bad:
                reason = "unavailable or failed verified predecessor: " + ", ".join(bad)
        if cancelled:
            reason = "run cancelled by interrupt"
        if reason:
            record = {
                "stage": stage.name,
                "command": stage.command,
                "status": status,
                "reason": reason,
                "error": reason if status == "blocked" else None,
                "argv": stage.argv,
                "run_dir": stage.run_dir,
                "exit_code": None,
                "artifacts": [],
            }
            emit("stage_skipped", **record)
        else:
            record = core.launch_stage(
                stage,
                args=args,
                destination=destination,
                logs=logs,
                emit=emit,
                manifest=manifest,
                dependencies=dependencies,
                input_hashes=input_hashes,
                executor=executor,
                runner_path=Path(__file__),
            )
            cancelled = record.get("exit_code") == 130
            try:
                metrics.extend(collect_metrics(stage))
                if record["status"] == "succeeded":
                    if stage.name == "R7-adaptive-availability":
                        availability = core.json_read(
                            Path(stage.run_dir) / "adaptive_availability.json"
                        )
                        if availability["status"] == "AVAILABLE":
                            artifact = availability["adaptive_path"]
                            if (
                                core.sha256(Path(artifact))
                                != availability["adaptive_sha256"]
                            ):
                                raise ValueError(
                                    "adaptive artifact changed after strict validation"
                                )
                            dependencies[artifact] = availability["adaptive_sha256"]
                        elif availability["status"] != "NOT_AVAILABLE":
                            raise ValueError("unknown adaptive availability status")
                    if stage.command in {"evaluate", "headroom-evaluate"}:
                        roles_path = (
                            args.roles_file
                            or destination / "stages/R0-preflight/roles.json"
                        )
                        expected = declared_cohort(stage, core.json_read(roles_path))
                        actual = (
                            [
                                r["subject_id"]
                                for r in read_jsonl(
                                    Path(stage.run_dir) / "paired_subjects.jsonl"
                                )
                            ]
                            if stage.command == "evaluate"
                            else core.validation_subjects(stage)
                        )
                        if expected != actual or len(set(actual)) != len(actual):
                            raise ValueError(
                                f"observed subjects differ from predeclared ordered cohort: {stage.name}"
                            )
                        key = (
                            flag(stage, "--split-role")
                            + ":"
                            + str(flag(stage, "--max-subjects"))
                        )
                        if key in cohorts and cohorts[key] != actual:
                            raise ValueError(
                                "same role/cap differs across evaluation controls"
                            )
                        cohorts[key] = actual
                        record["subject_ids"] = actual
            except Exception as error:  # noqa: BLE001 - persist malformed evidence and continue independent stages
                record.update(
                    status="failed",
                    exit_code=1,
                    error=f"evidence validation failed: {type(error).__name__}: {error}",
                )
                core.json_write(
                    logs / stage.name / "exit.json",
                    {"schema_version": PREFIX + "exit-v1", **record},
                )
                emit(
                    "evidence_validation_failed",
                    stage=stage.name,
                    error=record["error"],
                )
        if stage.name in {"R7-calibration", "R7-adaptive-availability"} and record[
            "status"
        ] in {"failed", "blocked"}:
            availability = {
                "status": "ERROR",
                "reason": record.get("error", "calibration execution failed"),
            }
        records.append(record)
        summarize()
    try:
        join = control_join(plan, records)
    except Exception as error:  # noqa: BLE001 - persist malformed evidence and continue independent stages
        join = {
            "schema_version": "pfgr-lite-full-control-join-v1",
            "status": "ERROR",
            "rows": [],
            "error": str(error),
        }
        records.append(
            {"stage": "control-join", "status": "failed", "error": str(error)}
        )
    core.json_write(metadata / "control_join.json", join)
    summarize()
    packaging = package_stage(args, destination)
    package = core.launch_stage(
        packaging,
        args=args,
        destination=destination,
        logs=logs,
        emit=emit,
        manifest=manifest,
        dependencies=dependencies,
        input_hashes=input_hashes,
        executor=executor,
        runner_path=Path(__file__),
    )
    try:
        verify_package(package, packaging)
    except Exception as error:  # noqa: BLE001 - persist malformed evidence and continue independent stages
        package.update(
            status="failed", exit_code=1, error=f"archive verification failed: {error}"
        )
        core.json_write(
            destination / "package_logs/exit.json",
            {"schema_version": PREFIX + "exit-v1", **package},
        )
    summarize(package)
    failed = (
        any(r["status"] in {"failed", "blocked"} for r in records)
        or package["status"] != "succeeded"
    )
    exit_code = 130 if cancelled else 1 if failed else 0
    emit("pipeline_exit", stage=None, exit_code=exit_code, last_stage="R10")
    return exit_code, destination


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--helper" in argv:
        return helper_main(argv)
    args = parser().parse_args(argv)
    try:
        code, destination = run(args)
    except (ValueError, TypeError, OSError) as error:
        print(f"PFGR full pipeline input/reservation failure: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "exit_code": code,
                "run_dir": str(destination),
                "plan_only": args.plan_only,
                "last_stage": "R10",
            }
        ),
        flush=True,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
