"""Executable PFGR-Lite command family.

The CLI is deliberately a thin boundary around the typed PFGR services.  It
owns argument/config resolution, run receipts, dry manifests and safe output
directories; model, data, teacher, value and policy mechanics remain in their
respective service modules.  Real-data commands therefore fail closed when a
reviewed dependency is absent instead of silently writing a parser-only
success.  ``--synthetic`` is an explicit engineering capability and can
exercise the bounded CPU path without claiming a trained or clinical result.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time
import traceback
from typing import Any


CLI_SCHEMA = "pfgr-lite-cli-v1"
RECEIPT_SCHEMA = "pfgr-lite-receipt-v1"
ENVIRONMENT_SCHEMA = "pfgr-lite-environment-v1"
COMMANDS = (
    "preflight",
    "smoke",
    "benchmark",
    "static-train",
    "updater-train",
    "bank-build",
    "bank-verify",
    "value-fit",
    "value-evaluate",
    "calibrate",
    "evaluate",
    "oracle-evaluate",
    "resume",
    "package",
    "runbook-check",
)
SCENARIOS = ("static", "noop", "random", "fixed_learned", "adaptive", "parallel_topk")
BUDGETS = (0, 1, 2, 4)
WandB_ENTITY = "khanhlq-work-hanoi-university-of-science-and-technology"
WandB_PROJECT = "smagm-point-guided"
_LAST_RESERVED_RUN_DIR: Path | None = None


class CLIError(ValueError):
    """Actionable command/config boundary error."""


def _jsonable(value: Any) -> Any:
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _jsonable(value.as_dict())
    if hasattr(value, "to_metadata") and callable(value.to_metadata):
        return _jsonable(value.to_metadata())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not (value == value and abs(value) != float("inf")):
            raise CLIError("nonfinite value cannot be written to a receipt")
        return value
    # Tensors are never allowed in CLI receipts.  Services must expose hashes
    # or scalar reductions instead of leaking patient data/predictions.
    if getattr(value, "__class__", None).__name__ == "Tensor":
        raise CLIError("tensor payload is not allowed in CLI receipt")
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSON file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CLIError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise CLIError(f"JSON root must be an object: {path}")
    return payload


REVIEW_RECEIPT_SCHEMA = "pfgr-lite-review-receipt-v1"


def _review_context(
    *,
    scope: str,
    config_hash: str,
    inputs: Any,
    bundle: Any,
    args: argparse.Namespace,
    policy: str,
    budget: int,
    value_identity_hash: str | None = None,
    split_role: str | None = None,
) -> dict[str, Any]:
    """Build the exact human-review payload before costly/target work."""

    role_manifest = getattr(inputs, "role_manifest", None) or getattr(bundle, "role_manifest", None)
    if role_manifest is None:
        raise CLIError("review context requires the complete TrainingRoleManifest identity")
    subjects = tuple(str(sample.subject_id) for sample in getattr(inputs, "samples", ()))
    if not subjects:
        raise CLIError("review context requires a nonempty selected subject cohort")
    split_hash = str(getattr(role_manifest, "baseline_split_hash", ""))
    producer = getattr(bundle, "producer", None)
    producer_hash = getattr(producer, "compatibility_hash", None)
    if not isinstance(producer_hash, str) or not producer_hash:
        raise CLIError("review context requires a complete producer compatibility identity")
    payload = {
        "schema_version": "pfgr-lite-review-context-v1",
        "scope": scope,
        "selected_subject_ids": subjects,
        "split_role": str(split_role or getattr(args, "split_role", "validation")),
        "baseline_split_hash": split_hash,
        "role_manifest_digest": role_manifest.digest,
        "producer_compatibility_hash": producer_hash,
        "value_fit_identity_hash": value_identity_hash,
        "config_hash": config_hash,
        "policy": policy,
        "budget": int(budget),
        "seed": int(args.seed),
        "teacher_mode": str(getattr(args, "teacher_mode", "iid_fixed_q")),
        "query_count": int(getattr(args, "query_count", 1024)),
        "candidate_count": int(getattr(args, "candidate_count", 32)),
        "candidate_chunk_size": int(getattr(args, "candidate_chunk_size", 1)),
        "decode_chunk_size": int(getattr(args, "decode_chunk_size", 1024)),
    }
    return payload


def _review_context_hash(context: Mapping[str, Any]) -> str:
    from smagm.features.point_guided.pfgr_lite.provenance import canonical_digest

    return canonical_digest(dict(context), prefix="pfgr-lite-review-cohort-v1|")


def _dry_review_context(
    args: argparse.Namespace,
    command: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Resolve a review payload without observations, targets, or rollouts.

    Reviewers need the exact cohort/config/artifact identities before issuing
    an approval receipt.  This helper intentionally loads only checkpoint and
    value metadata plus the reviewed split/role JSON; production MRI samples
    are not opened and no model route/teacher is executed.  A complete return
    value is a *review request*, never an approval or scientific result.
    """

    review_required = command == "calibrate" or (
        command == "evaluate" and str(getattr(args, "split_role", "validation")) == "test"
    )
    if not review_required:
        return None, []
    missing: list[str] = []
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint is None or not checkpoint.is_file():
        missing.append(f"checkpoint predecessor does not exist: {checkpoint}")
    value_checkpoint = getattr(args, "value_checkpoint", None)
    needs_value = command == "calibrate" or str(getattr(args, "scenario", "")) in {"adaptive", "fixed_learned", "parallel_topk"}
    if needs_value and (value_checkpoint is None or not value_checkpoint.is_file()):
        missing.append(f"value checkpoint predecessor does not exist: {value_checkpoint}")
    if command == "calibrate" and getattr(args, "evidence", None) is not None and not bool(getattr(args, "synthetic", False)):
        missing.append("production calibration evidence must be collected by S5; --evidence is engineering-only")
    if missing:
        return None, missing

    from types import SimpleNamespace

    from smagm.features.point_guided.pfgr_lite.checkpoint import load_inference_bundle, load_value_artifact
    from smagm.features.point_guided.pfgr_lite.config import PFGRLiteConfig
    from smagm.features.point_guided.pfgr_lite.types import TrainingRoleManifest

    try:
        bundle = load_inference_bundle(checkpoint)
        artifact = None
        if needs_value:
            artifact = load_value_artifact(
                value_checkpoint,
                expected_producer=bundle.producer,
                expected_role_manifest=bundle.role_manifest,
            )
        if args.synthetic:
            stage = "S5" if command == "calibrate" else None
            config, _ = _config_for_command(args, stage=stage)
            # This engineering fixture is target-free and bounded; it creates
            # only its deterministic observation records, never a target join
            # or policy rollout.  Use the exact role/sample selection that the
            # corresponding normal command will pass to the service.
            inputs = _synthetic_inputs(args, config, stage=stage)
            resolved_config = config
        else:
            split_file = getattr(args, "split_file", None)
            roles_file = getattr(args, "roles_file", None)
            if split_file is None or not split_file.is_file():
                missing.append(f"--split-file predecessor does not exist: {split_file}")
            if roles_file is None or not roles_file.is_file():
                missing.append(f"--roles-file predecessor does not exist: {roles_file}")
            if missing:
                return None, missing
            from smagm.data.brats21_point_guided import load_point_guided_split

            split = load_point_guided_split(split_file)
            role_manifest = TrainingRoleManifest.from_dict(_load_json(roles_file))
            if role_manifest.baseline_split_hash != split.split_hash:
                raise CLIError("review roles baseline split does not match the reviewed split")
            if bundle.role_manifest is not None and role_manifest.digest != bundle.role_manifest.digest:
                raise CLIError("review roles do not match the checkpoint role manifest")
            role = "calibration" if command == "calibrate" else str(getattr(args, "split_role", "validation"))
            role_ids = {
                "producer_fit": role_manifest.producer_fit_subject_ids,
                "calibration": tuple(sorted(set(role_manifest.calibration_fit_subject_ids) | set(role_manifest.calibration_allowance_subject_ids))),
                "calibration_fit": role_manifest.calibration_fit_subject_ids,
                "calibration_allowance": role_manifest.calibration_allowance_subject_ids,
                "validation": role_manifest.baseline_validation_subject_ids,
                "test": role_manifest.baseline_test_subject_ids,
            }
            if role not in role_ids:
                raise CLIError(f"unknown review split role {role!r}")
            selected = tuple(role_ids[role])[: getattr(args, "max_subjects", None) or len(role_ids[role])]
            if not selected:
                raise CLIError(f"review split role {role!r} has no selected subjects")
            inputs = SimpleNamespace(
                role_manifest=role_manifest,
                samples=tuple(SimpleNamespace(subject_id=subject_id) for subject_id in selected),
            )
            bundle_pfgr = bundle.config.get("pfgr_config") if isinstance(bundle.config, Mapping) else None
            if not isinstance(bundle_pfgr, Mapping):
                raise CLIError("checkpoint predecessor is missing its strict PFGR config envelope")
            resolved_config = PFGRLiteConfig.from_dict(bundle_pfgr)
            # A dry review request must fail on requested protocol drift before
            # a reviewer spends effort on an apparently valid cohort.  The
            # checkpoint's PFGR payload is the resolved producer identity; the
            # only tolerated request difference is W3's explicit unresolved
            # normalization policy label.  No MRI/model/teacher work occurs.
            requested_config, _ = _config_for_command(args, stage="S5" if command == "calibrate" else None)
            _validate_resolved_pfgr_config(requested_config, resolved_config)
        config_hash = hashlib.sha256(json.dumps(_jsonable(resolved_config.as_dict()), sort_keys=True).encode()).hexdigest()
        scope = "R7-calibration-cohort" if command == "calibrate" else "R9-final-evaluation"
        policy = "adaptive-calibration" if command == "calibrate" else str(args.scenario)
        budget = 4 if command == "calibrate" else int(args.budget)
        split_role = "calibration" if command == "calibrate" else str(args.split_role)
        context = _review_context(
            scope=scope,
            config_hash=config_hash,
            inputs=inputs,
            bundle=bundle,
            args=args,
            policy=policy,
            budget=budget,
            value_identity_hash=None if artifact is None else artifact.value_fit_identity.digest,
            split_role=split_role,
        )
        expected_artifacts: dict[str, str] = {"checkpoint_sha256": _sha256(checkpoint)}
        if value_checkpoint is not None:
            expected_artifacts["value_checkpoint_sha256"] = _sha256(value_checkpoint)
        role_manifest = getattr(inputs, "role_manifest", None) or bundle.role_manifest
        expected_artifacts.update(
            {
                "role_manifest_digest": role_manifest.digest,
                "split_hash": role_manifest.baseline_split_hash,
            }
        )
        return {
            "schema_version": "pfgr-lite-review-context-v1",
            "status": "REVIEW_REQUIRED",
            "scope": scope,
            "context": context,
            "cohort_hash": _review_context_hash(context),
            "config_hash": config_hash,
            "expected_artifacts": expected_artifacts,
            "decision_required": True,
            "scientific_status": "NOT_EVALUATED",
        }, []
    except (OSError, TypeError, ValueError, KeyError, CLIError) as error:
        missing.append(f"review context unresolved: {type(error).__name__}: {error}")
        return None, missing


def _validate_review_receipt(
    path: Path,
    *,
    synthetic: bool,
    config_hash: str | None = None,
    expected_context: Mapping[str, Any] | None = None,
    expected_scope: str | None = None,
    expected_artifacts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate a human-authored cost/held-out review boundary.

    A path-exists check is not an approval.  The receipt binds the requested
    scope and exact resolved config/cohort hashes, while ``ENGINEERING_DIAGNOSTIC``
    is the only accepted decision for synthetic fixtures.
    """

    payload = _load_json(path)
    required = {"schema_version", "scope", "decision", "reviewer", "created_at", "config_hash", "cohort_hash", "artifacts"}
    unknown = set(payload) - required
    missing = required - set(payload)
    if missing or unknown:
        raise CLIError(f"review receipt keys invalid; missing={sorted(missing)}, unknown={sorted(unknown)}")
    if payload["schema_version"] != REVIEW_RECEIPT_SCHEMA:
        raise CLIError("unknown review receipt schema")
    if not isinstance(payload["scope"], str) or not payload["scope"].strip():
        raise CLIError("review receipt scope must be nonempty")
    decision = payload["decision"]
    allowed = {"APPROVED", "ENGINEERING_DIAGNOSTIC"}
    if decision not in allowed:
        raise CLIError("review receipt decision must be APPROVED or ENGINEERING_DIAGNOSTIC")
    if synthetic and decision != "ENGINEERING_DIAGNOSTIC":
        raise CLIError("synthetic commands require ENGINEERING_DIAGNOSTIC review receipts")
    if not synthetic and decision != "APPROVED":
        raise CLIError("production commands require an explicit APPROVED review receipt")
    for name in ("reviewer", "created_at"):
        if not isinstance(payload[name], str) or not payload[name].strip():
            raise CLIError(f"review receipt {name} must be nonempty")
    for name in ("config_hash", "cohort_hash"):
        value = payload[name]
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
            raise CLIError(f"review receipt {name} must be a complete SHA256 digest")
    if config_hash is not None and payload["config_hash"] != config_hash:
        raise CLIError("review receipt config_hash does not match resolved PFGR config")
    if not isinstance(payload["artifacts"], Mapping):
        raise CLIError("review receipt artifacts must be a mapping of predecessor paths/hashes")
    if expected_scope is not None and payload["scope"] != expected_scope:
        raise CLIError(f"review receipt scope must be exactly {expected_scope!r}")
    if expected_context is not None:
        expected_hash = _review_context_hash(expected_context)
        if payload["cohort_hash"] != expected_hash:
            raise CLIError("review receipt cohort_hash does not match the actual selected cohort/work payload")
    if expected_artifacts is not None:
        for key, expected_value in expected_artifacts.items():
            if payload["artifacts"].get(key) != expected_value:
                raise CLIError(f"review receipt artifact {key!r} does not match the actual predecessor")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_SOURCE_SCOPE_ROOTS = (
    "src/smagm/cli/pfgr_lite.py",
    "src/smagm/features/point_guided/pfgr_lite/",
    "tests/features/point_guided/pfgr_lite/",
    "configs/pfgr_lite/",
    "RUNBOOK_PFGR_LITE.md",
    "README.md",
    "docs/README.md",
)
_SOURCE_SCOPE_SUFFIXES = {".py", ".json", ".md"}


def _scoped_source_paths(repo: Path) -> list[str]:
    """Resolve the reproducibility scope without including run artifacts.

    ``git ls-files --cached --others`` intentionally includes relevant
    untracked implementation/config/documentation files, while the explicit
    roots and suffix allow-list exclude user data, checkpoints, run outputs,
    ``.DS_Store`` and arbitrary untracked files elsewhere in the worktree.
    """

    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", *_SOURCE_SCOPE_ROOTS],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )
        candidates = result.stdout.splitlines() if result.returncode == 0 else []
    except OSError:
        candidates = []
    resolved: list[str] = []
    for item in candidates:
        path = item.strip().replace("\\", "/")
        if not path or Path(path).name.startswith("."):
            continue
        if Path(path).suffix.lower() not in _SOURCE_SCOPE_SUFFIXES:
            continue
        if not any(path == root.rstrip("/") or path.startswith(root) for root in _SOURCE_SCOPE_ROOTS):
            continue
        if (repo / path).is_file():
            resolved.append(path)
    return sorted(set(resolved))


def _scoped_source_digest(repo: Path, paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for relative in paths:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with (repo / relative).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _scoped_dirty_diff(repo: Path) -> tuple[str, bool]:
    """Hash tracked diff bytes and report dirty state for the same scope."""

    try:
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--", *_SOURCE_SCOPE_ROOTS],
            cwd=repo,
            check=False,
            capture_output=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", *_SOURCE_SCOPE_ROOTS],
            cwd=repo,
            check=False,
            capture_output=True,
        )
        payload = diff.stdout if diff.returncode == 0 else b""
        dirty = bool(status.stdout.strip()) or bool(payload)
    except OSError:
        payload = b""
        dirty = True
    return hashlib.sha256(payload).hexdigest(), dirty


def _source_receipt() -> dict[str, Any]:
    """Return source identity without dumping arbitrary environment values."""

    repo = Path(__file__).resolve().parents[3]

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )
        source_sha = result.stdout.strip() if result.returncode == 0 else "unknown"
    except OSError:
        source_sha = "unknown"
    try:
        diff_hash, dirty = _scoped_dirty_diff(repo)
    except OSError:
        diff_hash, dirty = hashlib.sha256(b"").hexdigest(), True
    paths = _scoped_source_paths(repo)
    try:
        source_scope_hash = _scoped_source_digest(repo, paths)
    except OSError:
        source_scope_hash = "unknown"
    return {
        "source_sha": source_sha,
        "working_tree_dirty": bool(dirty),
        "dirty_diff_sha256": diff_hash,
        "source_scope_sha256": source_scope_hash,
        "source_scope_file_count": len(paths),
        "source_scope_roots": list(_SOURCE_SCOPE_ROOTS),
    }


def _environment_receipt(device: str | None) -> dict[str, Any]:
    values: dict[str, Any] = {
        "schema_version": ENVIRONMENT_SCHEMA,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device_requested": device or "cpu",
        "amp": False,
    }
    try:
        import torch

        values.update(
            {
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
            }
        )
    except Exception as error:  # pragma: no cover - import diagnostics only
        values["torch_error"] = f"{type(error).__name__}: {error}"
    return values


def _add_common(parser: argparse.ArgumentParser, *, config_required: bool = False) -> None:
    parser.add_argument("--config", type=Path, required=config_required)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--roles-file", type=Path)
    parser.add_argument("--medicalnet-checkpoint", type=Path)
    parser.add_argument("--medicalnet-sha256")
    parser.add_argument("--output-root", type=Path, default=Path("runs/pfgr-lite"))
    parser.add_argument("--run-name")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("--synthetic", action="store_true", default=False)
    parser.add_argument("--dry-manifest", action="store_true", default=False)
    parser.add_argument("--review-receipt", type=Path)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb-entity", default=WandB_ENTITY)
    parser.add_argument("--wandb-project", default=WandB_PROJECT)
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--query-count", type=int, default=1024)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--teacher-mode", choices=("exact_footprint", "iid_fixed_q"), default="iid_fixed_q")
    parser.add_argument("--split-role", default="validation")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--candidate-chunk-size", type=int, default=1)
    parser.add_argument("--decode-chunk-size", type=int, default=1024)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m smagm.cli.pfgr_lite",
        description="PFGR-Lite target-free frontend, bounded diagnostics, and evidence runbook CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="validate environment, checkpoint and reviewed split")
    _add_common(preflight)
    preflight.add_argument("--write-roles", action="store_true")

    smoke = subparsers.add_parser("smoke", help="run bounded static/updater engineering smoke")
    _add_common(smoke)

    benchmark = subparsers.add_parser("benchmark", help="benchmark same-work full/sparse teacher parity")
    _add_common(benchmark, config_required=False)
    benchmark.add_argument("--checkpoint", type=Path)
    benchmark.add_argument("--max-states", type=int, default=2)
    benchmark.add_argument("--repeats", type=int, default=3)

    static = subparsers.add_parser("static-train", help="run explicit S0 static-base stage")
    _add_common(static)
    static.add_argument("--base", choices=("b0", "b1", "b2", "b_light"), default="b2")

    updater = subparsers.add_parser("updater-train", help="run explicit S1 updater stage")
    _add_common(updater)
    updater.add_argument("--checkpoint", type=Path)
    updater.add_argument("--spectral-arm", choices=("u_only", "u_plus_spectral"), default="u_plus_spectral")

    bank = subparsers.add_parser("bank-build", help="build a versioned measured value bank")
    _add_common(bank)
    bank.add_argument("--checkpoint", type=Path)
    bank.add_argument("--max-states", type=int, default=3)

    verify = subparsers.add_parser("bank-verify", help="replay and verify a value bank")
    _add_common(verify)
    verify.add_argument("--bank-index", type=Path, required=True)
    verify.add_argument("--checkpoint", type=Path)
    verify.add_argument("--replay-count", type=int, default=2)

    value_fit = subparsers.add_parser("value-fit", help="fit one signed ValueNet on an immutable bank")
    _add_common(value_fit, config_required=True)
    value_fit.add_argument("--bank-index", type=Path, required=True)
    # A V artifact must be joined to the exact producer envelope.  Engineering
    # banks carry that envelope in ``index.json``; production runs should pass
    # the preceding strict inference bundle so the join is independently
    # rechecked rather than inferred from a filename.
    value_fit.add_argument("--checkpoint", type=Path)
    value_fit.add_argument("--value-input", type=int, choices=(126, 222, 270, 366), default=366)
    value_fit.add_argument("--learning-rate", type=float, default=1e-3)

    value_eval = subparsers.add_parser("value-evaluate", help="evaluate same-bank ValueNet controls")
    _add_common(value_eval)
    value_eval.add_argument("--bank-index", type=Path, required=True)
    value_eval.add_argument("--checkpoint", type=Path)
    value_eval.add_argument("--value-checkpoint", type=Path, required=True)

    calibrate = subparsers.add_parser("calibrate", help="fit train-only policy calibration")
    _add_common(calibrate)
    calibrate.add_argument("--checkpoint", type=Path, required=True)
    calibrate.add_argument("--value-checkpoint", type=Path, required=True)
    calibrate.add_argument("--evidence", type=Path, help="diagnostic-only sealed evidence import for engineering fixtures; production S5 collects traces directly")

    evaluate = subparsers.add_parser("evaluate", help="evaluate one explicit target-free policy scenario")
    _add_common(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--value-checkpoint", type=Path)
    evaluate.add_argument("--scenario", choices=SCENARIOS, required=True)
    evaluate.add_argument("--budget", type=int, choices=BUDGETS, required=True)
    evaluate.add_argument("--local-footprint-audit", action="store_true")

    oracle = subparsers.add_parser("oracle-evaluate", help="run separate target-aware diagnostic oracle")
    _add_common(oracle)
    oracle.add_argument("--checkpoint", type=Path, required=True)
    oracle.add_argument("--oracle-mode", choices=("sampled_one", "greedy", "all_exact_one"), required=True)
    oracle.add_argument("--budget", type=int, choices=BUDGETS, required=True)
    oracle.add_argument("--confirmation-mode", choices=("exact_footprint", "iid_fixed_q"), default="exact_footprint")
    oracle.add_argument("--confirmation-query-count", type=int, default=1024)

    resume = subparsers.add_parser("resume", help="resume one explicit stage/checkpoint")
    _add_common(resume, config_required=True)
    resume.add_argument("--resume-checkpoint", type=Path, required=True)
    resume.add_argument("--bank-index", type=Path, help="immutable value bank for a value-fit resume")
    resume.add_argument("--value-input", type=int, choices=(126, 222, 270, 366), help="ValueNet descriptor width for a value-fit resume")
    resume.add_argument("--learning-rate", type=float, default=1e-3)

    package = subparsers.add_parser("package", help="package allow-listed run evidence")
    package.add_argument("--run-dir", type=Path, action="append", required=True)
    package.add_argument("--output-root", type=Path, default=Path("runs/pfgr-lite"))
    package.add_argument("--run-name")
    package.add_argument("--dry-manifest", action="store_true", default=False)

    check = subparsers.add_parser("runbook-check", help="validate executable Vietnamese runbook blocks")
    check.add_argument("--runbook", type=Path, default=Path("RUNBOOK_PFGR_LITE.md"))
    check.add_argument("--config-dir", type=Path, default=Path("configs/pfgr_lite"))
    check.add_argument("--output-root", type=Path, default=Path("runs/pfgr-lite"))
    check.add_argument("--run-name")
    check.add_argument("--dry-manifest", action="store_true", default=False)
    return parser


def _default_run_name(command: str) -> str:
    return f"{command}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}"


def _reserve_run(args: argparse.Namespace, command: str) -> Path:
    global _LAST_RESERVED_RUN_DIR
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    name = args.run_name or _default_run_name(command)
    if not name or name in {".", ".."} or Path(name).name != name:
        raise CLIError("run-name must be a simple nonempty directory name")
    run_dir = output_root / name
    try:
        run_dir.mkdir(mode=0o755, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite existing PFGR run: {run_dir}") from error
    _LAST_RESERVED_RUN_DIR = run_dir
    return run_dir


def _config_document(path: Path | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load strict execution config and return PFGR/frontend/normalization/options maps."""

    if path is None:
        path = Path("configs/pfgr_lite/main.json")
    document = _load_json(path)
    allowed_wrapped = {
        "schema_version",
        "pfgr_config",
        "frontend_sidecar",
        "normalization",
        "stage_options",
        "normalization_hash",
    }
    if "pfgr_config" in document:
        unknown = set(document) - allowed_wrapped
        if unknown:
            raise CLIError(f"unknown StageExecutionConfig keys: {sorted(unknown)}")
        pfgr = document.get("pfgr_config")
        frontend = document.get("frontend_sidecar", {})
        normalization = document.get("normalization", {})
        options = document.get("stage_options", {})
    else:
        # Plain PFGR config is accepted only when it contains no execution
        # fields; this keeps PFGRLiteConfig strict and avoids ignored options.
        pfgr = document
        frontend = {}
        normalization = {}
        options = {}
    from smagm.features.point_guided.pfgr_lite.config import PFGRLiteConfig, frontend_config_from_dict
    from smagm.features.point_guided.pfgr_lite.stages import StageExecutionConfig, StageOptions

    resolved = PFGRLiteConfig.from_dict(pfgr)
    if frontend:
        # StageExecutionConfig's sidecar is metadata; if a serialized frontend
        # config is supplied, validate its version and fields here.
        if isinstance(frontend, Mapping) and "schema_version" in frontend and "config" in frontend:
            frontend_config_from_dict(frontend)
    stage_options = StageOptions.from_dict(options) if options else StageOptions(engineering_only=resolved.engineering_only)
    execution = StageExecutionConfig(
        config=resolved,
        frontend_sidecar=frontend,
        normalization=normalization,
        stage_options=stage_options,
    )
    return resolved.as_dict(), dict(frontend), dict(normalization), execution.as_dict()


def _config_for_command(args: argparse.Namespace, *, stage: str | None = None) -> tuple[Any, dict[str, Any]]:
    from dataclasses import replace as dataclass_replace

    from smagm.features.point_guided.pfgr_lite.config import (
        PFGRLiteConfig,
        frontend_config_from_dict,
    )

    path = args.config if getattr(args, "config", None) is not None else None
    raw, frontend_sidecar, normalization, execution = _config_document(path)
    config = PFGRLiteConfig.from_dict(raw)
    if stage == "S0" and getattr(args, "base", None) is not None:
        variant_by_flag = {
            "b0": "b0_legacy_v1",
            "b1": "b1_multiscale_v1",
            "b2": "b2_ordered_multiscale_v1",
            "b_light": "b_light_ordered_v1",
        }
        try:
            variant = variant_by_flag[str(args.base)]
        except KeyError as error:  # argparse choices should make this unreachable
            raise CLIError(f"unknown static base variant: {args.base!r}") from error
        config = dataclass_replace(config, static=dataclass_replace(config.static, variant=variant))
    if getattr(args, "synthetic", False):
        # Small N is a capability marker, never a production default.
        config = PFGRLiteConfig(
            static=config.static,
            policy=config.policy,
            value=config.value,
            teacher=config.teacher,
            numeric_mode="fp32",
            candidate_count=config.candidate_count,
            state_channels=config.state_channels,
            correction_channels=config.correction_channels,
            write_scale=config.write_scale,
            support_radius_mm=config.support_radius_mm,
            max_displacement_mm=config.max_displacement_mm,
            build_chunk_size=min(config.build_chunk_size, 128),
            decode_chunk_size=min(config.decode_chunk_size, 128),
            device=getattr(args, "device", None),
            num_points=min(config.num_points, 4),
            engineering_only=True,
            observation_normalization=config.observation_normalization,
        )
    if stage is not None:
        opts_raw = dict(execution.get("stage_options", {}))
        opts_raw.update({"stage": stage, "device": getattr(args, "device", "cpu"), "engineering_only": config.engineering_only})
        if getattr(args, "epochs", None) is not None:
            opts_raw["epochs"] = args.epochs
        if getattr(args, "max_steps", None) is not None:
            opts_raw["max_updates"] = args.max_steps
        if getattr(args, "batch_size", None) is not None:
            opts_raw["batch_size"] = args.batch_size
        if stage == "S1":
            opts_raw["arm"] = getattr(args, "spectral_arm", opts_raw.get("arm", "u_plus_spectral"))
        if stage == "S2":
            requested_teacher_mode = getattr(args, "teacher_mode", "iid_fixed_q")
            if requested_teacher_mode not in {"exact_footprint", "iid_fixed_q"}:
                raise CLIError(f"unknown teacher mode: {requested_teacher_mode!r}")
            opts_raw["query_mode"] = "exact_dense" if requested_teacher_mode == "exact_footprint" else "iid_fixed_q"
            requested_q = int(getattr(args, "query_count", config.teacher.q_draws))
            if requested_teacher_mode == "exact_footprint":
                requested_q = 0
            elif requested_q < 2:
                raise CLIError("iid_fixed_q requires --query-count >= 2")
            # Q/mode are explicit S2 execution sidecars.  Keep the frozen
            # PFGR producer config unchanged so the same U/D/Z0 checkpoint
            # can be measured under exact and fixed-Q teacher controls.
            opts_raw["teacher_q_draws"] = requested_q
            requested_candidates = int(getattr(args, "candidate_count", 32))
            if requested_candidates > 32:
                raise CLIError("S2 candidate-count above the bounded 32-candidate stage ceiling is unsupported; pass an explicit <=32 value")
            opts_raw["candidate_count"] = requested_candidates
            opts_raw["candidates_per_state"] = requested_candidates
            if getattr(args, "max_states", None) is not None:
                requested_states = int(args.max_states)
                if requested_states < 1 or requested_states > 3:
                    raise CLIError("--max-states must be an explicit integer in [1, 3]; no silent stage clamp is applied")
                opts_raw["max_states_per_subject"] = requested_states
        else:
            # StageOptions' teacher query mode is an S2 measurement control;
            # all static/updater/stage-service paths use the exact dense
            # objective and must not inherit a calibration Q mode from a
            # shared execution JSON.
            opts_raw["query_mode"] = "exact_dense"
        from smagm.features.point_guided.pfgr_lite.stages import StageOptions

        options = StageOptions.from_dict(opts_raw)
        # Do not mutate the source config: execution sidecar is serialized in
        # the run receipt and strict PFGRLiteConfig remains authoritative.
        execution["stage_options"] = options.as_dict()
    # Command-level controls (notably ``--base`` and the explicit synthetic
    # capability) are part of the strict execution envelope; never leave the
    # stale PFGR document nested inside StageExecutionConfig.
    execution["pfgr_config"] = config.as_dict()
    frontend = None
    if frontend_sidecar and "schema_version" in frontend_sidecar:
        frontend = frontend_config_from_dict(frontend_sidecar)
    return config, {"frontend": frontend, "normalization": normalization, "execution": execution}


def _input_missing(args: argparse.Namespace, *, require_real: bool = True) -> list[str]:
    missing: list[str] = []
    if require_real and not getattr(args, "synthetic", False):
        # A hydrated PFGR inference checkpoint carries the frozen MedicalNet
        # source/provenance; a fresh constructor needs the external verified
        # checkpoint.  Never demand or silently mix both paths.
        required_names = ["data_root", "split_file"]
        if getattr(args, "checkpoint", None) is None:
            required_names.extend(("medicalnet_checkpoint", "medicalnet_sha256"))
        for name in required_names:
            value = getattr(args, name, None)
            if value is None:
                missing.append(f"--{name.replace('_', '-')}")
        for name, predicate in (
            ("data_root", lambda value: value.is_dir()),
            ("split_file", lambda value: value.is_file()),
            ("medicalnet_checkpoint", lambda value: value.is_file()),
        ):
            value = getattr(args, name, None)
            if value is not None and not predicate(value):
                missing.append(f"{name} path does not exist: {value}")
        checkpoint_bundle = getattr(args, "checkpoint", None)
        if checkpoint_bundle is not None and not checkpoint_bundle.is_file():
            missing.append(f"checkpoint path does not exist: {checkpoint_bundle}")
        checkpoint = getattr(args, "medicalnet_checkpoint", None)
        expected = getattr(args, "medicalnet_sha256", None)
        if checkpoint is not None and checkpoint.is_file() and expected:
            actual = _sha256(checkpoint)
            if actual.lower() != expected.lower():
                missing.append(f"checkpoint SHA256 mismatch: expected {expected}, actual {actual}")
    return missing


def _receipt_base(args: argparse.Namespace, command: str, *, status: str, run_dir: Path, scientific_status: str = "NOT_EVALUATED") -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA,
        "cli_schema": CLI_SCHEMA,
        "command": command,
        # Preserve the explicit argv supplied by embedders/tests; falling
        # back to sys.argv is only for the module entry point.
        "argv": list(getattr(args, "_argv", sys.argv[1:])),
        "status": status,
        "exit_code": 0,
        "scientific_status": scientific_status,
        "scientific_claim": "NOT_EVALUATED",
        "run_dir": str(run_dir.resolve()),
        "source": _source_receipt(),
        "environment": _environment_receipt(getattr(args, "device", None)),
        "config_hash": None,
        "effective_policy_hash": None,
        "stage": None,
        "role": getattr(args, "split_role", None),
        "capability": "engineering_only" if getattr(args, "synthetic", False) else "production_pending",
        "counts": {},
        "metrics": {},
    }


def _wandb_numeric_metrics(value: Any, *, prefix: str = "") -> dict[str, float | int]:
    """Flatten finite receipt numerics for W&B without coercing unknowns."""

    result: dict[str, float | int] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_prefix = f"{prefix}.{key_text}" if prefix else key_text
            result.update(_wandb_numeric_metrics(child, prefix=child_prefix))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            result.update(_wandb_numeric_metrics(child, prefix=f"{prefix}.{index}" if prefix else str(index)))
    elif isinstance(value, bool):
        return result
    elif isinstance(value, int):
        result[prefix] = int(value)
    elif isinstance(value, float) and math.isfinite(value):
        result[prefix] = float(value)
    return {key: value for key, value in result.items() if key}


def _wandb_receipt(args: argparse.Namespace, *, command: str, run_dir: Path, metrics: Mapping[str, Any] | None = None, counts: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return optional real W&B identity without fabricating a URL.

    W&B is deliberately opt-in and defaults to offline mode so a missing
    package or network never turns a bounded software command into a fake
    experiment.  Only IDs/URLs returned by an actual ``wandb.Run`` are
    persisted; skipped/unavailable states carry an explicit reason.
    """

    if not bool(getattr(args, "wandb", False)):
        return {"enabled": False, "status": "DISABLED"}
    try:
        import wandb  # type: ignore
    except Exception as error:  # pragma: no cover - optional dependency
        return {"enabled": True, "status": "SKIPPED", "reason": f"optional_dependency_unavailable:{type(error).__name__}"}
    mode = os.environ.get("WANDB_MODE", "offline")
    try:
        run = wandb.init(
            project=str(getattr(args, "wandb_project", WandB_PROJECT)),
            entity=str(getattr(args, "wandb_entity", WandB_ENTITY)),
            dir=str(run_dir),
            mode=mode,
            name=run_dir.name,
            reinit=True,
        )
        payload: dict[str, Any] = {"enabled": True, "status": "STARTED", "mode": mode}
        run_id = getattr(run, "id", None)
        run_url = getattr(run, "url", None)
        if isinstance(run_id, str) and run_id:
            payload["run_id"] = run_id
        if isinstance(run_url, str) and run_url:
            payload["url"] = run_url
        numeric = {}
        numeric.update(_wandb_numeric_metrics(metrics or {}, prefix="metrics"))
        numeric.update(_wandb_numeric_metrics(counts or {}, prefix="counts"))
        # A service may expose histogram bins/stop counts as nested numeric
        # mappings; flattening keeps every bin/count visible without inventing
        # zeroes for unavailable counters.
        try:
            if numeric and callable(getattr(run, "log", None)):
                run.log(numeric)
                payload["logged_metric_count"] = len(numeric)
            summary = getattr(run, "summary", None)
            if numeric and summary is not None and hasattr(summary, "update"):
                summary.update(numeric)
                payload["summary_metric_count"] = len(numeric)
            payload["metric_scope"] = "receipt_metrics_and_counts"
        except Exception as error:  # pragma: no cover - optional dependency
            payload["status"] = "SKIPPED"
            payload["reason"] = f"wandb_logging_failed:{type(error).__name__}"
        try:
            run.finish()
        except Exception:
            pass
        return payload
    except Exception as error:  # pragma: no cover - optional dependency
        return {"enabled": True, "status": "SKIPPED", "reason": f"wandb_init_failed:{type(error).__name__}"}


def _publish_receipt(args: argparse.Namespace, command: str, run_dir: Path, *, status: str = "SOFTWARE_PASS", scientific_status: str = "NOT_EVALUATED", **updates: Any) -> dict[str, Any]:
    receipt = _receipt_base(args, command, status=status, run_dir=run_dir, scientific_status=scientific_status)
    receipt.update(updates)
    receipt["wandb"] = _wandb_receipt(args, command=command, run_dir=run_dir, metrics=receipt.get("metrics"), counts=receipt.get("counts"))
    _write_json(run_dir / "receipt.json", receipt)
    return receipt


def _dry_manifest(args: argparse.Namespace, command: str, run_dir: Path, *, planned: Mapping[str, Any] | None = None) -> dict[str, Any]:
    missing = _input_missing(args, require_real=not getattr(args, "synthetic", False))
    review_context, review_missing = _dry_review_context(args, command)
    missing.extend(review_missing)
    planned_payload = dict(planned or {})
    if review_context is not None:
        review_path = run_dir / "review_context.json"
        _write_json(review_path, review_context)
        planned_payload.update(
            {
                "review_context": str(review_path),
                "review_scope": review_context["scope"],
                "decision_required": True,
            }
        )
    try:
        config, details = _config_for_command(args)
        config_payload: Any = config.as_dict()
        config_hash = hashlib.sha256(json.dumps(_jsonable(config_payload), sort_keys=True).encode()).hexdigest()
    except Exception as error:
        config_payload = None
        config_hash = None
        missing.append(f"config: {type(error).__name__}: {error}")
        details = {}
    payload = {
        "schema_version": CLI_SCHEMA,
        "status": "BLOCKED" if missing else "DRY_MANIFEST",
        "scientific_status": "NOT_EVALUATED",
        "command": command,
        "synthetic": bool(getattr(args, "synthetic", False)),
        "planned": _jsonable(planned_payload),
        "missing_inputs": missing,
        "config": config_payload,
        "config_hash": config_hash,
        "execution": _jsonable(details.get("execution")),
        "environment": _environment_receipt(getattr(args, "device", None)),
        "source": _source_receipt(),
    }
    _write_json(run_dir / "dry_manifest.json", payload)
    _publish_receipt(args, command, run_dir, status="BLOCKED" if missing else "SOFTWARE_PASS", manifest_only=True, missing_inputs=missing, config_hash=config_hash, planned=_jsonable(planned_payload))
    return payload


def _synthetic_inputs(args: argparse.Namespace, config: Any, *, stage: str | None = None) -> Any:
    """Build a tiny target-free fixture through the actual PFGR model seam."""

    import torch

    from smagm.features.point_guided import PointGuidedConfig
    from smagm.features.point_guided.contracts import VolumeGeometry
    from smagm.features.point_guided.pfgr_lite.data import DataAccessCounters, TargetFreeSample
    from smagm.features.point_guided.pfgr_lite.footprint import PFGRQueryLattice
    from smagm.features.point_guided.pfgr_lite.model import PFGRLiteModel
    from smagm.features.point_guided.pfgr_lite.sparse_write import make_action_writer, make_point_query, make_support_legal_mask
    from smagm.features.point_guided.pfgr_lite.stages import StageInputs, StageOptions
    from smagm.features.point_guided.pfgr_lite.types import OperationCounters

    seed = int(getattr(args, "seed", 20260907))
    torch.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except ImportError:  # pragma: no cover - NumPy is a PFGR runtime dependency
        pass
    n = min(int(config.num_points), 4)
    frontend = PointGuidedConfig(num_semantic_classes=3, num_points=n, point_candidate_multiplier=2, offset_hidden_channels=12, detach_backbone_features=False)
    model = PFGRLiteModel(config, frontend_config=frontend, query_lattice_factory=PFGRQueryLattice).to(torch.device(getattr(args, "device", "cpu"))).train()
    checkpoint_path = getattr(args, "checkpoint", None)
    if checkpoint_path is not None:
        from smagm.features.point_guided.pfgr_lite.checkpoint import hydrate_inference_model, load_inference_bundle

        bundle = load_inference_bundle(checkpoint_path)
        if bundle.config.get("pfgr_config") != config.as_dict():
            raise CLIError("synthetic checkpoint PFGR config does not match the supplied config")
        model = hydrate_inference_model(bundle, query_lattice_factory=PFGRQueryLattice).to(torch.device(getattr(args, "device", "cpu"))).train()
    geometry = VolumeGeometry.from_spacing((9, 9, 9), (1.0, 1.0, 1.0))
    counters = DataAccessCounters()
    operation_counters = OperationCounters()
    samples = []
    # One deterministic four-subject engineering cohort is shared by every
    # saved stage.  Producer-fit and calibration-fit/allowance IDs are
    # disjoint; a bounded command selects only the requested stage role, so a
    # small synthetic run cannot relabel the same subject as independent fit
    # and calibration evidence.
    producer_subject_ids = ("synthetic-00", "synthetic-01")
    calibration_subject_ids = ("synthetic-02", "synthetic-03")
    selected_subject_ids = calibration_subject_ids if stage == "S5" else producer_subject_ids
    observation_generator = torch.Generator(device="cpu")
    observation_generator.manual_seed(seed + 7919)
    selected_count = max(1, min(int(getattr(args, "max_subjects", None) or 1), len(selected_subject_ids)))
    for index, subject_id in enumerate(selected_subject_ids[:selected_count]):
        observations = torch.randn((3, 9, 9, 9), dtype=torch.float32, generator=observation_generator)
        mask = torch.ones((1, 9, 9, 9), dtype=torch.bool)
        samples.append(TargetFreeSample(subject_id, observations, mask, geometry, {"policy": config.observation_normalization}, "", ""))

    def target_provider(subject_id: str, **_: Any) -> torch.Tensor:
        for sample in samples:
            if sample.subject_id == subject_id:
                return torch.zeros((1, *sample.shape_dhw), dtype=torch.float32)
        raise ValueError(f"unknown synthetic subject {subject_id}")

    point_query = make_point_query()

    def writer(state: Any, context: Any, action: Any) -> Any:
        lattice = PFGRQueryLattice.build(context.geometry, context.feature_geometry, query_dtype=state.planes.xy.dtype, build_chunk_size=config.build_chunk_size)
        return make_action_writer(lattice)(state, context, action)

    def legal_mask(lattice: Any) -> Any:
        return make_support_legal_mask(lattice)

    role_manifest = None
    if stage in {"S0", "S1", "S2", "S4", "S5"}:
        # Engineering routes still carry a complete role identity so strict
        # updater provenance/resume envelopes never fall back to ``None`` or
        # sentinel strings.  These synthetic assignments are explicitly
        # engineering-only and can never mint production adaptive evidence.
        from smagm.features.point_guided.pfgr_lite.types import TrainingRoleManifest

        subject_ids = producer_subject_ids + calibration_subject_ids
        fit_ids = ("synthetic-02",)
        allowance_ids = ("synthetic-03",)
        producer_ids = producer_subject_ids
        role_manifest = TrainingRoleManifest(
            baseline_split_hash="synthetic-split-v1",
            baseline_train_subject_ids=subject_ids,
            baseline_validation_subject_ids=(),
            baseline_test_subject_ids=(),
            producer_fit_subject_ids=producer_ids,
            calibration_fit_subject_ids=fit_ids,
            calibration_allowance_subject_ids=allowance_ids,
            subject_group_ids=tuple((subject, f"group-{index:03d}") for index, subject in enumerate(subject_ids)),
            engineering_only=True,
        )
    metadata = {
        "data_counters": counters,
        # StageInputs uses ``counters`` as the canonical receipt key.  Keep
        # the same object under both names so deferred target joins and
        # service receipts observe the exact callback activity.
        "counters": counters,
        "operation_counters": operation_counters,
        "lattice_factory": PFGRQueryLattice,
        "legal_mask_builder": legal_mask,
        "synthetic": True,
        "initialization_id": "synthetic-cli-v1",
        # Stage provenance requires complete source/checkpoint identities even
        # for the explicit engineering fixture.  These names describe the
        # actual synthetic initialization; they are not claims about an
        # external or historical MedicalNet checkpoint.
        "source_id": str(Path(checkpoint_path).resolve()) if checkpoint_path is not None else "synthetic-untrained-initialization-v1",
        "checkpoint_id": str(Path(checkpoint_path).resolve()) if checkpoint_path is not None else "synthetic-untrained-initialization-v1",
    }
    options = StageOptions(stage=stage or "S0", device=getattr(args, "device", "cpu"), epochs=max(1, int(getattr(args, "epochs", None) or 1)), max_updates=getattr(args, "max_steps", None), engineering_only=True, query_chunk_size=min(config.decode_chunk_size, 128), candidate_chunk_size=max(1, int(getattr(args, "candidate_chunk_size", 1))))
    return StageInputs(
        samples=tuple(samples),
        model=model,
        config=config,
        stage_options=options,
        target_provider=target_provider,
        query=point_query,
        writer=writer,
        role_manifest=role_manifest,
        metadata=metadata,
    )


def _production_inputs(args: argparse.Namespace, config: Any, *, stage: str | None = None) -> Any:
    missing = _input_missing(args)
    if missing:
        raise CLIError("real-data preflight is incomplete: " + "; ".join(missing))
    if getattr(args, "roles_file", None) is None or not args.roles_file.is_file():
        raise CLIError("production StageInputs require the reviewed --roles-file produced by preflight; role assignment is never regenerated implicitly")
    from smagm.features.point_guided.pfgr_lite.stages import StageOptions, build_stage_inputs

    # Resolve the serialized frontend/normalization sidecar through the same
    # strict config path used for receipts.  Passing ``None`` here would make
    # build_stage_inputs silently construct an unproven three-channel prior.
    _, resolved_details = _config_for_command(args, stage=stage)
    frontend = resolved_details.get("frontend")
    normalization = resolved_details.get("normalization") or None
    options_payload = resolved_details.get("execution", {}).get("stage_options", {})
    options = StageOptions.from_dict(options_payload) if options_payload else StageOptions(stage=stage or "S0", device=args.device, epochs=max(1, int(args.epochs or 1)), max_updates=args.max_steps, engineering_only=False, query_chunk_size=args.decode_chunk_size, candidate_chunk_size=args.candidate_chunk_size)
    checkpoint_path = getattr(args, "checkpoint", None)
    # The W3 factory treats an inference checkpoint as the complete frozen
    # source.  Passing an external MedicalNet path alongside it is ambiguous
    # and correctly rejected; use the external path only for a fresh S0
    # constructor.
    medicalnet_path = None if checkpoint_path is not None else args.medicalnet_checkpoint
    medicalnet_sha = None if checkpoint_path is not None else args.medicalnet_sha256
    if checkpoint_path is not None:
        # ``hydrate_inference_model`` below validates the serialized frontend
        # sidecar from the bundle; an unrelated config sidecar with null
        # MedicalNet fields must not override that source of truth.
        frontend = None
    # S5 collection must see both disjoint calibration roles before any target
    # join; W3b's factory exposes this as the explicit combined role.
    subject_role = "calibration" if stage == "S5" else (args.split_role if stage not in {"S0", "S1", "S2"} else "producer_fit")
    return build_stage_inputs(
        config,
        data_root=args.data_root,
        split_file=args.split_file,
        roles_file=args.roles_file,
        frontend_config=frontend,
        checkpoint_path=checkpoint_path,
        medicalnet_checkpoint_path=medicalnet_path,
        medicalnet_checkpoint_sha256=medicalnet_sha,
        normalization_config=normalization,
        stage_options=options,
        max_subjects=args.max_subjects,
        subject_role=subject_role,
    )


def _inputs(args: argparse.Namespace, config: Any, *, stage: str | None = None) -> Any:
    if getattr(args, "synthetic", False):
        return _synthetic_inputs(args, config, stage=stage)
    return _production_inputs(args, config, stage=stage)


def _nested_module(model: Any, path: str) -> Any | None:
    value = model
    for part in path.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    return value


def _stage_optimizer(model: Any, stage: str, options: Any) -> Any | None:
    """Construct the same bounded optimizer ownership used by W3 stages."""

    import torch

    if model is None:
        return None
    names = {
        "S0": ("static_head", "frontend.base_plane_projector", "base_plane_projector", "decoder", "implicit_decoder"),
        "S1": ("updater", "update_net", "frontend.spectral_anchor_builder", "spectral_projector", "spectral_anchor_builder"),
    }.get(stage, ())
    modules: list[Any] = []
    seen_modules: set[int] = set()
    for name in names:
        module = _nested_module(model, name)
        if module is not None and hasattr(module, "parameters") and id(module) not in seen_modules:
            modules.append(module)
            seen_modules.add(id(module))
    params: list[Any] = []
    seen_params: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen_params:
                params.append(parameter)
                seen_params.add(id(parameter))
    if not params:
        return None
    return torch.optim.Adam(params, lr=float(options.learning_rate), weight_decay=float(options.weight_decay))


def _capture_rng_state() -> dict[str, Any]:
    import random

    state: dict[str, Any] = {"python": random.getstate(), "torch_cpu": __import__("torch").get_rng_state()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    torch = __import__("torch")
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _optimizer_parameter_names(model: Any, optimizer: Any) -> tuple[str, ...]:
    if optimizer is None or model is None:
        return ()
    by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    names: list[str] = []
    for group in optimizer.param_groups:
        for parameter in group.get("params", ()):
            name = by_id.get(id(parameter))
            if name is None:
                raise CLIError("optimizer contains a parameter not owned by the staged model")
            names.append(name)
    return tuple(names)


def _runtime_hashes(execution: Mapping[str, Any]) -> tuple[str, str]:
    # Match W3b's canonical identity envelopes exactly; a plain SHA-256 of
    # JSON would look equivalent but fail strict stage resume validation.
    from smagm.features.point_guided.pfgr_lite.provenance import canonical_digest

    execution_hash = canonical_digest(_jsonable(execution), prefix="pfgr-lite-execution-config-v1|")
    training = json.loads(json.dumps(_jsonable(execution)))
    stage_options = training.get("stage_options") if isinstance(training, Mapping) else None
    if isinstance(stage_options, Mapping):
        # W3b's immutable training identity retains the field with a null
        # value; deleting it would produce a different digest and make a
        # valid max_updates continuation look stale.
        stage_options["max_updates"] = None
    training_hash = canonical_digest(training, prefix="pfgr-lite-training-config-v1|")
    return execution_hash, training_hash


def _validate_resolved_pfgr_config(requested: Any, resolved: Any) -> None:
    """Validate the factory's concrete PFGR config without hiding recipe binding.

    The CLI document may carry the historical normalization policy label
    ``pfgr-observation-normalization-v1``.  W3's real-data factory resolves
    that label to the measured recipe identity and returns it through
    ``StageExecutionConfig``.  This is the sole permitted PFGR difference;
    every protocol dimension and stage option remains an exact match.
    """

    requested_payload = requested.as_dict() if hasattr(requested, "as_dict") else dict(requested)
    resolved_payload = resolved.as_dict() if hasattr(resolved, "as_dict") else dict(resolved)
    differences = {
        key
        for key in set(requested_payload) | set(resolved_payload)
        if requested_payload.get(key) != resolved_payload.get(key)
    }
    if differences - {"observation_normalization"}:
        raise CLIError(
            "factory-resolved PFGR config disagrees with supplied strict config: "
            f"{sorted(differences - {'observation_normalization'})}"
        )
    if "observation_normalization" in differences:
        requested_value = requested_payload.get("observation_normalization")
        if requested_value != "pfgr-observation-normalization-v1":
            raise CLIError(
                "requested observation_normalization must remain the explicit "
                "unresolved policy label until the factory resolves its measured recipe"
            )
        value = resolved_payload.get("observation_normalization")
        if not isinstance(value, str) or not value.strip() or value.lower() in {"unknown", "unset", "none", "null", "pfgr-observation-normalization-v1"}:
            raise CLIError("factory-resolved observation normalization identity is incomplete")


def _stage_runtime(
    *,
    result: Any,
    execution: Mapping[str, Any],
    optimizer: Any,
    model: Any,
    rng_state: Mapping[str, Any],
    producer_hash: str,
    split_role_hash: str | None,
    input_manifest_hash: str,
    sample_order: Sequence[str] = (),
) -> Mapping[str, Any]:
    """Validate and preserve W3b's stage-runtime envelope.

    W3b owns optimizer/RNG/cursor snapshots.  The CLI may serialize those
    snapshots, but it must not manufacture missing fields or silently replace
    an identity with a value derived from the current process.  Only the
    optional continuation annotation is accepted in addition to the strict
    runtime schema.
    """

    execution_hash, training_hash = _runtime_hashes(execution)
    expected = {
        "schema_version",
        "stage_state",
        "optimizer_state",
        "rng_state",
        "cursor",
        "parameter_names",
        "execution_config",
        "execution_config_hash",
        "training_config_hash",
        "producer_compatibility_hash",
        "split_role_hash",
        "input_manifest_hash",
    }
    optional = {"continuation"}

    supplied = getattr(result, "runtime_state", None)
    if supplied:
        if not isinstance(supplied, Mapping):
            raise CLIError("StageResult.runtime_state must be a mapping")
        raw = dict(supplied)
        unknown = set(raw) - expected - optional
        missing = expected - set(raw)
        if missing or unknown:
            raise CLIError(f"StageResult.runtime_state keys invalid; missing={sorted(missing)}, unknown={sorted(unknown)}")
        runtime = raw
    else:
        # This fallback is retained only for a direct engineering fixture
        # whose StageResult did not expose W3b runtime state.  S2 callers do
        # not invoke this helper because S2 is explicitly non-resumable.
        stage_state = asdict(result.stage_state)
        runtime = {
            "schema_version": "pfgr-lite-stage-runtime-v1",
            "stage_state": stage_state,
            "optimizer_state": {} if optimizer is None else optimizer.state_dict(),
            "rng_state": dict(rng_state),
            "cursor": {
                "epoch": int(result.stage_state.epoch),
                "batch_index": int(result.stage_state.update),
                "update": int(result.stage_state.update),
                "microstep": int(result.stage_state.microstep),
                "sample_order": list(sample_order),
                "route_rng_state": None,
            },
            "parameter_names": list(_optimizer_parameter_names(model, optimizer)),
            "execution_config": _jsonable(execution),
            "execution_config_hash": execution_hash,
            "training_config_hash": training_hash,
            "producer_compatibility_hash": producer_hash,
            "split_role_hash": split_role_hash,
            "input_manifest_hash": input_manifest_hash,
        }
    if set(runtime) - expected - optional or expected - set(runtime):
        raise CLIError("StageResult.runtime_state is incomplete or uses an unknown schema")
    if runtime.get("schema_version") != "pfgr-lite-stage-runtime-v1":
        raise CLIError("StageResult.runtime_state is incomplete or uses an unknown schema")
    if runtime.get("stage_state") != asdict(result.stage_state):
        raise CLIError("StageResult.runtime_state stage_state disagrees with StageResult")
    if runtime.get("execution_config_hash") != execution_hash:
        raise CLIError("StageResult.runtime_state execution_config_hash disagrees with resolved execution")
    if runtime.get("training_config_hash") != training_hash:
        raise CLIError("StageResult.runtime_state training_config_hash disagrees with resolved training config")
    if runtime.get("execution_config") != _jsonable(execution):
        raise CLIError("StageResult.runtime_state execution_config disagrees with resolved execution")
    if runtime.get("producer_compatibility_hash") != producer_hash:
        raise CLIError("StageResult.runtime_state producer compatibility identity disagrees with staged producer")
    if runtime.get("split_role_hash") != split_role_hash:
        raise CLIError("StageResult.runtime_state split-role identity disagrees with staged roles")
    if runtime.get("input_manifest_hash") != input_manifest_hash:
        raise CLIError("StageResult.runtime_state input manifest identity disagrees with staged inputs")
    return runtime


def _encode_artifact_context(model: Any, sample: Any, device: str) -> Any:
    """Encode one artifact provenance context without mutating train/eval state."""

    import torch

    if not hasattr(model, "encode_observations"):
        raise CLIError("model does not expose the PFGR observation encoder")
    modes = {module: bool(module.training) for module in model.modules()} if hasattr(model, "modules") else {}
    try:
        if hasattr(model, "eval"):
            model.eval()
        with torch.no_grad():
            return model.encode_observations(
                sample.observations.unsqueeze(0).to(device),
                sample.brain_mask.to(device),
                sample.geometry,
            )
    finally:
        # Restore every nested module's prior mode; ``model.train(flag)``
        # alone would erase a deliberately mixed legacy mode configuration.
        for module, training in modes.items():
            module.training = training


def _frontend_artifact_for_model(model: Any, fallback: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return only the canonical serialized frontend configuration.

    ``StageExecutionConfig.frontend_sidecar`` intentionally also carries
    factory provenance (checkpoint paths, loader identities, and hashes).  It
    is an execution receipt, not the strict W1 frontend artifact consumed by
    checkpoint hydration.  Publishing that whole sidecar makes a valid
    production checkpoint fail ``frontend_config_from_dict`` on resume.  The
    model's typed ``PointGuidedConfig`` is the source of truth; a fallback is
    accepted only when it is already the versioned ``{schema_version, config}``
    artifact.
    """

    from smagm.features.point_guided.pfgr_lite.config import frontend_config_from_dict, frontend_config_to_dict

    candidate = getattr(model, "frontend_config", None)
    if candidate is None:
        frontend = getattr(model, "frontend", None)
        candidate = getattr(frontend, "config", None)
    if candidate is not None:
        return frontend_config_to_dict(candidate)
    if isinstance(fallback, Mapping) and set(fallback) <= {"schema_version", "config"} and "schema_version" in fallback:
        # Parse and reserialize so aliases/order/path values are normalized by
        # the same strict W1 boundary used by checkpoint load.
        return frontend_config_to_dict(frontend_config_from_dict(fallback))
    raise CLIError("stage model did not expose a typed frontend_config for checkpoint publication")


def _stage_command(args: argparse.Namespace, command: str, *, stage: str) -> dict[str, Any]:
    run_dir = _reserve_run(args, command)
    if args.dry_manifest:
        return _dry_manifest(args, command, run_dir, planned={"stage": stage, "max_subjects": args.max_subjects, "max_steps": args.max_steps, "epochs": args.epochs or 1})
    from dataclasses import replace as dataclass_replace

    config, details = _config_for_command(args, stage=stage)
    inputs = _inputs(args, config, stage=stage)
    from smagm.features.point_guided.pfgr_lite.stages import StageExecutionConfig, _input_manifest_hash, run_stage

    # W3's factory historically emits ``checkpoint_id='none'`` for a fresh
    # run constructed directly from the verified MedicalNet file.  The strict
    # producer-stage envelope cannot carry that sentinel: bind the actual
    # initialization source path before the stage service snapshots its
    # provenance.  This does not invent a checkpoint or alter model weights.
    metadata = dict(getattr(inputs, "metadata", {}) or {})
    source_path = getattr(args, "checkpoint", None) or getattr(args, "medicalnet_checkpoint", None)
    if source_path is not None:
        source_text = str(Path(source_path).resolve())
        if str(metadata.get("checkpoint_id", "")).lower() in {"", "none", "unknown", "unset", "null"}:
            metadata["checkpoint_id"] = source_text
        if str(metadata.get("source_id", "")).lower() in {"", "none", "unknown", "unset", "null", "engineering-source"}:
            metadata["source_id"] = source_text
    if metadata != dict(getattr(inputs, "metadata", {}) or {}):
        inputs = dataclass_replace(inputs, metadata=metadata)

    # Synthetic fixtures are resolved once from the strict CLI envelope.  A
    # production StageInputs factory, however, has already hydrated the exact
    # frontend/normalization/checkpoint provenance; replacing its execution
    # object here would erase that evidence and can make a valid checkpoint
    # appear compatible with a different sidecar.
    resolved_execution = getattr(inputs, "execution", None)
    if not args.synthetic and resolved_execution is None:
        raise CLIError("production StageInputs must expose its factory-resolved StageExecutionConfig")
    if resolved_execution is None:
        resolved_execution = StageExecutionConfig.from_dict(details["execution"])
    options = resolved_execution.stage_options
    if args.synthetic:
        inputs = dataclass_replace(inputs, execution=resolved_execution, stage_options=options)
    else:
        # Preserve W3's resolved execution and only require that the strict
        # PFGR payload agrees with the command config at the seam.
        _validate_resolved_pfgr_config(config, resolved_execution.config)
        # The factory's measured normalization identity is the concrete
        # producer config used by run_stage and checkpoint publication.
        config = resolved_execution.config
        details = {**details, "execution": resolved_execution.as_dict()}
    rng_before = _capture_rng_state()
    result = run_stage(stage, config, inputs, run_dir / stage.lower())
    rng_after = _capture_rng_state()
    _write_json(run_dir / "resolved_config.json", details["execution"])
    published_metrics = dict(result.metrics)
    _write_json(run_dir / "stage_state.json", asdict(result.stage_state))
    artifact_paths: dict[str, str] = {}
    # Publish a strict target-free inference bundle from the actual model
    # state/context produced by the stage.  No target, teacher or prediction
    # tensor is placed in the bundle; missing producer context is a hard
    # failure rather than a synthetic metadata shortcut.
    model = getattr(inputs, "model", None)
    samples = tuple(getattr(inputs, "samples", ()))
    if model is not None and samples and hasattr(model, "encode_observations"):
        from smagm.features.point_guided.pfgr_lite.checkpoint import CHECKPOINT_CONFIG_SCHEMA, save_inference_bundle, save_resume
        from smagm.features.point_guided.pfgr_lite.types import InferenceBundle

        sample = samples[0]
        context = _encode_artifact_context(model, sample, args.device)
        source_provenance = getattr(context.producer, "source_provenance", None)
        traversal_count = getattr(source_provenance, "traversal_count", None)
        if isinstance(traversal_count, int) and not isinstance(traversal_count, bool):
            published_metrics["artifact_context_traversal_count"] = traversal_count
        # Persist the typed W1 frontend artifact, not the factory execution
        # sidecar (which may additionally contain checkpoint/load paths).
        # The latter remains available in resolved_config.json and the stage
        # receipt for provenance, but must never be fed to strict hydration.
        frontend_sidecar = _frontend_artifact_for_model(
            model,
            details["execution"].get("frontend_sidecar", {}),
        )
        role_manifest = getattr(inputs, "role_manifest", None)
        split_hash = getattr(role_manifest, "baseline_split_hash", "synthetic-split-v1" if args.synthetic else None)
        split_role_hash = getattr(role_manifest, "digest", None if args.synthetic else "")
        bundle_config = {
            "schema_version": CHECKPOINT_CONFIG_SCHEMA,
            "pfgr_config": config.as_dict(),
            "frontend_config": frontend_sidecar,
            "stage": "inference",
            "split_roles": {
                "producer_fit": "producer_fit",
                "calibration_fit": "calibration_fit",
                "calibration_allowance": "calibration_allowance",
            },
            "value_fit_identity_hash": None,
            "gain_scale_hash": None,
            "effective_policy_hash": None,
        }
        capability = "static" if stage == "S0" else "forced_diagnostic"
        bundle = InferenceBundle(
            state_dict=model.state_dict(),
            producer=context.producer,
            config=bundle_config,
            capability=capability,
            split_hash=split_hash,
            frontend_config=frontend_sidecar,
            role_manifest=role_manifest,
            stage_provenance=result.receipt.stage_provenance,
        )
        checkpoint_path = run_dir / "inference.pt"
        save_inference_bundle(checkpoint_path, bundle)
        artifact_paths["inference_checkpoint"] = str(checkpoint_path)
        # S2 builds an immutable bank and has no optimizer/RNG continuation
        # contract.  Do not synthesize a resume snapshot from an empty runtime
        # mapping; publish an explicit non-resumable receipt instead.
        if stage in {"S0", "S1"}:
            resume_path = run_dir / "resume.pt"
            runtime_state = _stage_runtime(
                result=result,
                execution=details["execution"],
                optimizer=None,
                model=model,
                rng_state=rng_after,
                producer_hash=context.producer.compatibility_hash,
                split_role_hash=split_role_hash,
                input_manifest_hash=_input_manifest_hash(inputs),
                sample_order=tuple(item.subject_id for item in samples),
            )
            save_resume(
                resume_path,
                bundle,
                result.stage_state,
                runtime_state["optimizer_state"],
                runtime_state["rng_state"],
                {"stage": stage, "parent_inference": "inference.pt", "producer_compatibility_hash": context.producer.compatibility_hash, "rng_before_keys": sorted(rng_before), "stage_runtime": runtime_state},
            )
            runtime_receipt = {key: value for key, value in runtime_state.items() if key not in {"optimizer_state", "rng_state", "cursor"}}
            cursor_receipt = dict(runtime_state.get("cursor", {})) if isinstance(runtime_state.get("cursor"), Mapping) else {}
            cursor_receipt["route_rng_state"] = sorted(cursor_receipt.get("route_rng_state", {})) if isinstance(cursor_receipt.get("route_rng_state"), Mapping) else "present"
            runtime_receipt["cursor"] = cursor_receipt
            runtime_receipt.update({"optimizer_state_present": bool(runtime_state.get("optimizer_state")), "rng_streams": sorted(runtime_state.get("rng_state", {}))})
            _write_json(run_dir / "stage_runtime.json", runtime_receipt)
            artifact_paths["stage_runtime"] = str(run_dir / "stage_runtime.json")
            artifact_paths["resume_checkpoint"] = str(resume_path)
        else:
            _write_json(run_dir / "resume.json", {"schema_version": "pfgr-lite-resume-v1", "resumable": False, "reason": "bank-build S2 has no optimizer/RNG continuation contract"})
            artifact_paths["resume_receipt"] = str(run_dir / "resume.json")
    _write_json(run_dir / "metrics.json", published_metrics)
    receipt = _publish_receipt(args, command, run_dir, stage=stage, counts={"subjects": result.receipt.subjects, "updates": result.receipt.route_updates, "gradient_steps": result.receipt.gradient_steps}, metrics=published_metrics, config_hash=hashlib.sha256(json.dumps(config.as_dict(), sort_keys=True).encode()).hexdigest(), artifacts=artifact_paths)
    return receipt


def _producer_from_bank(reader: Any) -> Any:
    """Reconstruct the producer envelope persisted by ``ValueBankWriter``.

    The reader intentionally exposes validation and rows, not a second public
    producer accessor.  This narrow CLI adapter consumes its canonical index
    envelope so V fitting can still be run on an engineering bank without
    inventing a compatibility hash.  Production callers should additionally
    pass ``--checkpoint``; that path joins the bank to the exact hydrated
    inference bundle below.
    """

    from dataclasses import fields

    from smagm.features.point_guided.pfgr_lite.provenance import ProducerCompatibility, SourceProvenance
    from smagm.features.point_guided.pfgr_lite.types import ProducerDependencies

    index = getattr(reader, "index", None)
    envelope = index.get("producer") if isinstance(index, Mapping) else None
    if not isinstance(envelope, Mapping):
        raise CLIError("value-bank index has no canonical producer envelope")
    compatibility_payload = envelope.get("compatibility")
    source_payload = envelope.get("source_provenance")
    if not isinstance(compatibility_payload, Mapping) or not isinstance(source_payload, Mapping):
        raise CLIError("value-bank producer envelope is incomplete (compatibility/source provenance required)")
    compatibility_values = dict(compatibility_payload)
    compatibility_values["component_versions"] = tuple(tuple(row) for row in compatibility_values.get("component_versions", ()))
    compatibility = ProducerCompatibility(**compatibility_values)
    source_values = dict(source_payload)
    # SourceProvenance.as_dict carries two compatibility aliases; require them
    # to agree before removing them for the dataclass constructor.
    if "sha256" in source_values and source_values["sha256"] != source_values.get("checkpoint_sha256"):
        raise CLIError("value-bank source provenance SHA-256 alias mismatch")
    if "integrity_verified" in source_values and source_values["integrity_verified"] != source_values.get("checkpoint_integrity_verified"):
        raise CLIError("value-bank source provenance integrity alias mismatch")
    source_values.pop("sha256", None)
    source_values.pop("integrity_verified", None)
    source_values["details"] = tuple(tuple(row) for row in source_values.get("details", ()))
    allowed_source = {field.name for field in fields(SourceProvenance)}
    if set(source_values) != allowed_source:
        raise CLIError("value-bank source provenance keys are incomplete or unknown")
    source = SourceProvenance(**source_values)
    defaults = {
        "observation_normalization": "pfgr-observation-normalization-v1",
        "geometry_query_version": "pfgr-lite-static-geometry-v1",
        "static_architecture": "b2_ordered_multiscale_v1",
        "semantic_architecture": "medicalnet-resnet10-semantic-1x1-v1",
        "point_architecture": "deterministic-points-refiner-v1",
        "updater_architecture": "update-net-270-128-96-v1",
        "decoder_architecture": "implicit-decoder-96-64-32-1-v1",
        "writer_architecture": "compact-writeback-4mm-v1",
        "candidate_geometry": "point-candidate-geometry-v1",
        "label_definition": "signed-conditional-mean-masked-global-charbonnier-v1",
        "config_version": "pfgr-lite-config-v1",
    }
    return ProducerDependencies(compatibility=compatibility, source_provenance=source, **defaults)


def _producer_and_bundle(args: argparse.Namespace, reader: Any | None = None) -> tuple[Any, Any | None]:
    """Resolve a strict producer from a checkpoint or a bank's envelope."""

    bundle = None
    if getattr(args, "checkpoint", None) is not None:
        from smagm.features.point_guided.pfgr_lite.checkpoint import load_inference_bundle

        bundle = load_inference_bundle(args.checkpoint)
        producer = bundle.producer
        if reader is not None:
            _validate_supplied_bank_joins(args, reader, bundle)
            # ValueBankReader performs the complete row/provenance validation;
            # this join prevents a V fit from silently using another U/D bank.
            reader.validate_producer(producer)
            if bundle.role_manifest is not None:
                reader.validate_split_role(bundle.role_manifest.digest)
    elif reader is not None:
        if not bool(getattr(args, "synthetic", False)):
            raise CLIError("production value-bank joins require the external frozen inference --checkpoint; bank metadata is not an authority")
        _validate_supplied_bank_joins(args, reader, None)
        producer = _producer_from_bank(reader)
    else:
        raise CLIError("a strict producer checkpoint or bank index is required")
    return producer, bundle


def _validate_supplied_bank_joins(args: argparse.Namespace, reader: Any, bundle: Any | None) -> None:
    """Join a bank to the caller's reviewed split/role artifacts.

    Production value commands must not silently trust a bank's embedded role
    envelope when the runbook supplies a different split file.  Synthetic
    engineering fixtures intentionally use the bank authority and therefore
    skip this external-data requirement.
    """

    if bool(getattr(args, "synthetic", False)):
        return
    split_file = getattr(args, "split_file", None)
    roles_file = getattr(args, "roles_file", None)
    if split_file is None or not split_file.is_file():
        raise CLIError("production value-bank commands require the reviewed --split-file artifact")
    if roles_file is None or not roles_file.is_file():
        raise CLIError("production value-bank commands require the reviewed --roles-file artifact")
    from smagm.data.brats21_point_guided import load_point_guided_split
    from smagm.features.point_guided.pfgr_lite.types import TrainingRoleManifest

    split = load_point_guided_split(split_file)
    roles = TrainingRoleManifest.from_dict(_load_json(roles_file))
    if roles.baseline_split_hash != split.split_hash:
        raise CLIError("supplied role manifest baseline split does not match --split-file")
    reader_roles = getattr(reader, "role_manifest", None)
    if reader_roles is None:
        raise CLIError("production value-bank is missing its complete TrainingRoleManifest")
    if roles.digest != reader_roles.digest:
        raise CLIError("supplied role manifest does not match the immutable value-bank role identity")
    bundle_roles = getattr(bundle, "role_manifest", None)
    if bundle_roles is None or bundle_roles.digest != roles.digest:
        raise CLIError("supplied role manifest does not match the frozen inference checkpoint role identity")
    if reader.manifest().split_role_hash != roles.digest:
        raise CLIError("value-bank split/role hash does not match supplied reviewed roles")


def _bank_verify_command(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = _reserve_run(args, "bank-verify")
    if args.dry_manifest:
        return _dry_manifest(args, "bank-verify", run_dir, planned={"bank_index": str(args.bank_index), "replay_count": args.replay_count})
    from smagm.features.point_guided.pfgr_lite.value_bank import ValueBankReader
    from smagm.features.point_guided.pfgr_lite.bank_audit import audit_bank_replay

    reader = ValueBankReader(args.bank_index)
    producer, bundle = _producer_and_bundle(args, reader)
    rows = reader.rows(include_diagnostic=True)
    replay_count = int(args.replay_count)
    if replay_count < 0:
        raise CLIError("--replay-count must be nonnegative")
    if replay_count and not rows:
        raise CLIError("bank replay requested but the immutable bank contains no rows")
    role_manifest = bundle.role_manifest if bundle is not None else reader.role_manifest
    if replay_count and role_manifest is None:
        raise CLIError("bank replay audit requires the complete TrainingRoleManifest identity")
    replay_audit = audit_bank_replay(reader, replay_count, producer=producer, role_manifest=role_manifest) if replay_count else {
        "schema_version": "pfgr-lite-bank-audit-v1",
        "audit_kind": "state_snapshot_and_row_identity",
        "requested_replay_count": 0,
        "rows_checked": 0,
        "snapshots_checked": 0,
        "bytes_checked": 0,
        "decoder_calls": 0,
        "teacher_calls": 0,
        "reconstruction_replay": False,
        "status": "PASS",
    }
    role_counts: dict[str, int] = {}
    for row in rows:
        role = str(getattr(row, "split_role", ""))
        role_counts[role] = role_counts.get(role, 0) + 1
    payload = {
        "schema_version": "pfgr-lite-bank-verify-v1",
        "status": "SOFTWARE_PASS",
        "bank_index": str(Path(args.bank_index).resolve()),
        "manifest_hash": reader.manifest_hash,
        "producer_compatibility_hash": producer.compatibility_hash,
        "split_role_hash": reader.manifest().split_role_hash,
        "row_count": len(rows),
        "role_counts": dict(sorted(role_counts.items())),
        "replay_count": replay_count,
        "replay": dict(replay_audit) | {"target_reads": 0, "same_bank": True, "note": "state snapshot/row identity audit only; decoder route replay is not claimed"},
        "checkpoint_joined": bundle is not None,
        "scientific_status": "NOT_EVALUATED",
    }
    _write_json(run_dir / "bank_verify.json", payload)
    return _publish_receipt(args, "bank-verify", run_dir, counts={"rows": len(rows), "replayed": replay_count, "target_reads": 0}, metrics=payload, scientific_status="NOT_EVALUATED", bank_manifest_hash=reader.manifest_hash)


def _resolve_cached_value_config(config: Any, bundle: Any | None) -> Any:
    """Bind V commands to the producer's resolved normalization identity.

    Production S0/S1 factories replace the historical PFGR normalization
    policy label with the measured recipe identity before writing a bundle.
    V commands do not reload observations, so they must compare their strict
    execution config to that cached producer envelope rather than silently
    fitting/evaluating a value model under the unresolved label.  The only
    tolerated difference is the known unresolved default; all other config
    drift (including an explicitly different normalization identity) fails
    closed.
    """

    if bundle is None:
        return config
    from dataclasses import replace
    from smagm.features.point_guided.pfgr_lite.config import PFGRLiteConfig

    envelope = getattr(bundle, "config", None)
    expected_payload = envelope.get("pfgr_config") if isinstance(envelope, Mapping) else None
    if not isinstance(expected_payload, Mapping):
        raise CLIError("producer checkpoint lacks the cached PFGR config required for value commands")
    expected = PFGRLiteConfig.from_dict(expected_payload)
    requested = config.as_dict() if hasattr(config, "as_dict") else dict(config)
    expected_dict = expected.as_dict()
    differences = {key for key in set(requested) | set(expected_dict) if requested.get(key) != expected_dict.get(key)}
    if differences - {"observation_normalization"}:
        raise CLIError(
            "value command PFGR config differs from its producer checkpoint: "
            f"{sorted(differences - {'observation_normalization'})}"
        )
    if "observation_normalization" in differences:
        unresolved = "pfgr-observation-normalization-v1"
        if requested.get("observation_normalization") != unresolved:
            raise CLIError("value command normalization identity does not match the producer checkpoint")
        config = replace(config, observation_normalization=expected.observation_normalization)
    return config


def _value_fit_command(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = _reserve_run(args, "value-fit")
    if args.dry_manifest:
        return _dry_manifest(args, "value-fit", run_dir, planned={"bank_index": str(args.bank_index), "input_variant": args.value_input, "requires_checkpoint_for_production": True})
    from smagm.features.point_guided.pfgr_lite.checkpoint import save_resume, save_value_artifact
    from smagm.features.point_guided.pfgr_lite.value_bank import ValueBankReader
    from smagm.features.point_guided.pfgr_lite.value_net import fit_value

    reader = ValueBankReader(args.bank_index)
    producer, bundle = _producer_and_bundle(args, reader)
    config, details = _config_for_command(args, stage="S3")
    config = _resolve_cached_value_config(config, bundle)
    max_updates = args.max_steps
    fit = fit_value(
        reader,
        config=config.value,
        input_variant=args.value_input,
        epochs=max(1, int(args.epochs or details["execution"].get("stage_options", {}).get("epochs", 1))),
        batch_size=max(1, int(args.batch_size or details["execution"].get("stage_options", {}).get("batch_size", 32))),
        seed=args.seed,
        device=args.device,
        learning_rate=args.learning_rate,
        max_updates=max_updates,
    )
    if not bool(fit.complete):
        # W4 rejects incomplete ValueArtifacts.  Persist W3a's complete
        # tensor/optimizer/cursor/RNG fit envelope through the existing strict
        # resume container when the producer bundle is available; never write
        # a hash-only placeholder or claim a completed fit.
        incomplete = {"metrics": fit.metrics, "resume_keys": sorted(fit.resume_state), "fit_complete": False}
        if bundle is None:
            incomplete["resume_state_artifact"] = "not-published: producer checkpoint is required"
            _write_json(run_dir / "value_fit_incomplete.json", incomplete)
            raise CLIError("value fit stopped before completion and no producer checkpoint was supplied for strict resume")
        from smagm.features.point_guided.pfgr_lite.types import StageState

        stage_state = fit.stage_state or StageState(
            stage="value_fit",
            substage="value_fit",
            epoch=int(fit.resume_state.get("stage_payload", {}).get("epoch", 0)),
            update=int(fit.resume_state.get("stage_payload", {}).get("update_count", 0)),
            completion="pending",
        )
        resume_path = run_dir / "value-resume.pt"
        save_resume(
            resume_path,
            bundle,
            stage_state,
            {},
            {},
            {
                "stage": "value_fit",
                "parent_inference": "inference.pt",
                "producer_compatibility_hash": producer.compatibility_hash,
                "bank_state": {
                    "manifest_hash": reader.manifest_hash,
                    "split_role_hash": reader.manifest().split_role_hash,
                    "gain_scale_hash": reader.gain_scale.digest,
                },
                "cached_value_fit": fit.resume_state,
            },
        )
        incomplete.update({"resume_state_artifact": str(resume_path), "resumable": True})
        _write_json(run_dir / "value_fit_incomplete.json", incomplete)
        return _publish_receipt(
            args,
            "value-fit",
            run_dir,
            status="INCONCLUSIVE",
            scientific_status="NOT_EVALUATED",
            counts={"rows": int(fit.metrics.get("row_count", 0)), "target_reads": 0, "teacher_calls": 0},
            metrics=incomplete,
            artifacts={"resume_checkpoint": str(resume_path), "value_fit_incomplete": str(run_dir / "value_fit_incomplete.json")},
        )
    role_manifest = bundle.role_manifest if bundle is not None else reader.role_manifest
    artifact_path = run_dir / "value.pt"
    save_value_artifact(
        artifact_path,
        fit,
        producer=producer,
        config=config.value,
        role_manifest=role_manifest,
        stage_provenance=reader.stage_provenance,
    )
    metrics = dict(fit.metrics)
    metrics.update({"artifact": str(artifact_path), "input_variant": args.value_input, "bank_manifest_hash": reader.manifest_hash, "producer_compatibility_hash": producer.compatibility_hash})
    _write_json(run_dir / "value_fit.json", metrics)
    return _publish_receipt(args, "value-fit", run_dir, status="SOFTWARE_PASS", counts={"rows": int(metrics.get("row_count", 0)), "target_reads": 0, "teacher_calls": 0}, metrics=metrics, artifacts={"value_artifact": str(artifact_path)}, config_hash=hashlib.sha256(json.dumps(config.as_dict(), sort_keys=True).encode()).hexdigest())


def _value_evaluation_pairs(reader: Any, model: Any, *, input_variant: int, device: str, batch_size: int) -> dict[str, Any]:
    """Emit row-keyed, target-free V ranking controls for one bank/artifact.

    Each V variant is evaluated in a separate CLI invocation, but every row
    carries the immutable bank row hash/action/context identity.  Consumers
    can therefore join V126/V270/V366 results exactly without rerunning U,
    the writer, or the target teacher.
    """

    import torch

    from smagm.features.point_guided.pfgr_lite.provenance import canonical_digest
    from smagm.features.point_guided.pfgr_lite.value_bank import row_digest

    # ``ValueBankReader`` materializes rows in the exact immutable index order.
    # Keep that positional relationship (row_id/shard/offset) instead of
    # indexing by action label: labels are an identity component, not a safe
    # dictionary key for a cross-variant join.  The reader already validates
    # every index checksum, but re-check the canonical row digest here so this
    # artifact cannot silently fall back to a partial/derived identity.
    all_rows = tuple(reader.rows(include_diagnostic=True))
    index_rows = reader.index.get("rows", ()) if isinstance(getattr(reader, "index", None), Mapping) else ()
    if len(all_rows) != len(index_rows):
        raise ValueError("value-bank row/index cardinality mismatch")
    indexed_rows: list[tuple[Any, Mapping[str, Any]]] = []
    for row, entry in zip(all_rows, index_rows):
        if not isinstance(entry, Mapping):
            raise ValueError("value-bank row index entry is malformed")
        actual_hash = row_digest(row)
        indexed_hash = entry.get("row_hash")
        if not isinstance(indexed_hash, str) or indexed_hash != actual_hash:
            raise ValueError("value-bank row digest/index mismatch")
        indexed_rows.append((row, entry))
    rows = tuple((row, entry) for row, entry in indexed_rows if not bool(getattr(row, "diagnostic", False)))
    predictions: list[float] = []
    model.to(device)
    model.eval()
    model_dtype = next(model.parameters()).dtype
    with torch.no_grad():
        for start in range(0, len(rows), max(1, int(batch_size))):
            chunk = rows[start : start + max(1, int(batch_size))]
            descriptors = torch.stack([getattr(row, f"v{input_variant}") for row, _ in chunk]).to(device=device, dtype=model_dtype)
            predicted = model(descriptors).detach().to(device="cpu", dtype=torch.float64).reshape(-1)
            predictions.extend(float(value) for value in predicted.tolist())
    scale = float(reader.gain_scale.scale)
    records: list[dict[str, Any]] = []
    for (row, entry), predicted_scaled in zip(rows, predictions):
        row_hash = str(entry["row_hash"])
        row_key = canonical_digest(
            {"bank_manifest_hash": reader.manifest_hash, "row_id": int(entry["row_id"]), "row_hash": row_hash},
            prefix="pfgr-lite-value-eval-row-v1|",
        )
        records.append(
            {
                "row_key": row_key,
                "row_hash": row_hash,
                "row_id": entry.get("row_id"),
                "shard": entry.get("shard"),
                "offset": entry.get("offset"),
                "bank_manifest_hash": reader.manifest_hash,
                "input_variant": int(input_variant),
                "subject_id": row.subject_key,
                "context_id": row.context_id,
                "state_version": int(row.state_version),
                "point_id": int(row.point_id),
                "action_id": row.action_id,
                "proposal_hash": row.proposal_hash,
                "state_digest": row.state_digest,
                "predicted_scaled": float(predicted_scaled),
                "predicted_raw": float(predicted_scaled * scale),
                "measured_raw_gain": float(row.raw_gain),
            }
        )
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault((str(record["subject_id"]), str(record["context_id"]), int(record["state_version"])), []).append(record)
    for group in groups.values():
        predicted_order = sorted(group, key=lambda item: (-float(item["predicted_raw"]), int(item["point_id"]), str(item["action_id"])))
        measured_order = sorted(group, key=lambda item: (-float(item["measured_raw_gain"]), int(item["point_id"]), str(item["action_id"])))
        group_key = canonical_digest(
            {"subject_id": group[0]["subject_id"], "context_id": group[0]["context_id"], "state_version": group[0]["state_version"], "bank_manifest_hash": reader.manifest_hash},
            prefix="pfgr-lite-value-eval-group-v1|",
        )
        for rank, item in enumerate(predicted_order, start=1):
            item["group_key"] = group_key
            item["predicted_rank"] = rank
        for rank, item in enumerate(measured_order, start=1):
            item["group_key"] = group_key
            item["measured_rank"] = rank
    return {
        "schema_version": "pfgr-lite-value-evaluation-pairs-v1",
        "bank_manifest_hash": reader.manifest_hash,
        "gain_scale_hash": reader.gain_scale.digest,
        "input_variant": int(input_variant),
        "row_count": len(records),
        "group_count": len(groups),
        "same_bank": True,
        "teacher_calls": 0,
        "target_volume_reads": 0,
        "rows": records,
    }


def _value_evaluate_command(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = _reserve_run(args, "value-evaluate")
    if args.dry_manifest:
        return _dry_manifest(args, "value-evaluate", run_dir, planned={"bank_index": str(args.bank_index), "value_checkpoint": str(args.value_checkpoint)})
    from smagm.features.point_guided.pfgr_lite.checkpoint import load_value_artifact
    from smagm.features.point_guided.pfgr_lite.value_bank import ValueBankReader
    from smagm.features.point_guided.pfgr_lite.value_net import SignedValueNet, evaluate_value

    reader = ValueBankReader(args.bank_index)
    producer, bundle = _producer_and_bundle(args, reader)
    config, _ = _config_for_command(args, stage="S3")
    config = _resolve_cached_value_config(config, bundle)
    # Evaluation is allowed to read a different bank only through an explicit
    # future cross-bank mode.  The current CLI is same-bank R6: bind V to the
    # exact producer, reviewed role manifest, immutable q90 gain scale, and
    # index digest used for fitting before touching any rows.
    expected_roles = reader.role_manifest
    artifact = load_value_artifact(
        args.value_checkpoint,
        expected_producer=producer,
        expected_role_manifest=expected_roles,
        expected_input_variant=None,
        expected_gain_scale_hash=reader.gain_scale.digest,
    )
    fit_identity = artifact.value_fit_identity
    if fit_identity.bank_manifest_hash != reader.manifest_hash:
        raise CLIError(
            "value artifact was fit on a different bank; same-bank evaluation requires an exact manifest digest"
        )
    if artifact.role_manifest is None or expected_roles is None or artifact.role_manifest.digest != expected_roles.digest:
        raise CLIError("value artifact reviewed role manifest does not match the evaluated bank")
    if artifact.config.get("role_manifest_hash") not in {None, expected_roles.digest}:
        raise CLIError("value artifact fit role identity does not match the evaluated bank")
    if artifact.gain_scale.get("digest") != reader.gain_scale.digest or float(artifact.gain_scale.get("scale", 0.0)) != float(reader.gain_scale.scale):
        raise CLIError("value artifact fixed gain-scale provenance does not match the evaluated bank")
    model = SignedValueNet(artifact.value_fit_identity.input_variant)
    model.load_state_dict(dict(artifact.state_dict), strict=True)
    result = evaluate_value(reader, model, input_variant=artifact.value_fit_identity.input_variant, device=args.device, batch_size=max(1, int(args.batch_size or 256)))
    metrics = dict(result.metrics)
    pair_payload = _value_evaluation_pairs(
        reader,
        model,
        input_variant=artifact.value_fit_identity.input_variant,
        device=args.device,
        batch_size=max(1, int(args.batch_size or 256)),
    )
    pairs_path = run_dir / "value_evaluate_pairs.json"
    _write_json(pairs_path, pair_payload)
    metrics.update(
        {
            "artifact": str(args.value_checkpoint),
            "producer_compatibility_hash": producer.compatibility_hash,
            "bank_manifest_hash": reader.manifest_hash,
            "fit_bank_manifest_hash": fit_identity.bank_manifest_hash,
            "evaluated_bank_manifest_hash": reader.manifest_hash,
            "same_bank_scope": True,
            "fit_split_role_hash": artifact.config.get("role_manifest_hash", expected_roles.digest),
            "evaluated_split_role_hash": reader.manifest().split_role_hash,
            "fit_gain_scale_hash": fit_identity.gain_scale_hash,
            "evaluated_gain_scale_hash": reader.gain_scale.digest,
            "role_manifest_digest": expected_roles.digest,
            "paired_ranking_rows": str(pairs_path),
            "paired_ranking_row_count": int(pair_payload["row_count"]),
            "paired_ranking_group_count": int(pair_payload["group_count"]),
            "paired_ranking_schema": pair_payload["schema_version"],
        }
    )
    _write_json(run_dir / "value_evaluate.json", metrics)
    return _publish_receipt(
        args,
        "value-evaluate",
        run_dir,
        counts={"rows": result.row_count, "target_reads": 0, "teacher_calls": 0},
        metrics=metrics,
        artifacts={"value_evaluation_pairs": str(pairs_path)},
        scientific_status="NOT_EVALUATED",
        config_hash=hashlib.sha256(json.dumps(config.as_dict(), sort_keys=True).encode()).hexdigest(),
    )


def _calibrate_command(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = _reserve_run(args, "calibrate")
    if args.dry_manifest:
        return _dry_manifest(args, "calibrate", run_dir, planned={"requires_review_receipt": True, "requires_sealed_evidence": True, "checkpoint": str(args.checkpoint), "value_checkpoint": str(args.value_checkpoint)})
    if args.review_receipt is None:
        raise CLIError("calibration requires an explicit --review-receipt; manifests cannot self-approve an adaptive fit")
    if not args.review_receipt.is_file():
        raise FileNotFoundError(f"review receipt does not exist: {args.review_receipt}")
    if args.evidence is not None and not args.synthetic:
        raise CLIError("external calibration evidence JSON is diagnostic-only; production calibrate must collect sealed S5 traces")
    from dataclasses import replace, replace as dataclass_replace

    from smagm.features.point_guided.pfgr_lite.calibration_runner import CalibrationRunOptions, run_calibration
    from smagm.features.point_guided.pfgr_lite.checkpoint import load_inference_bundle, load_value_artifact, save_inference_bundle

    bundle = load_inference_bundle(args.checkpoint)
    value_artifact = load_value_artifact(
        args.value_checkpoint,
        expected_producer=bundle.producer,
        expected_role_manifest=bundle.role_manifest,
    )
    config, _ = _config_for_command(args, stage="S5")
    requested_teacher_mode = str(getattr(args, "teacher_mode", "exact_footprint"))
    if requested_teacher_mode == "exact_footprint":
        confirmation_mode = "exact"
        confirmation_q_draws = 0
    elif requested_teacher_mode == "iid_fixed_q":
        confirmation_mode = "iid_fixed_q"
        confirmation_q_draws = int(getattr(args, "query_count", config.teacher.q_draws))
    else:
        raise CLIError(f"unknown calibration teacher mode: {requested_teacher_mode!r}")
    options = CalibrationRunOptions(
        confirmation_mode=confirmation_mode,
        confirmation_q_draws=confirmation_q_draws,
        confirmation_seed=args.seed,
        collection_seed=args.seed,
        value_input_variant=value_artifact.value_fit_identity.input_variant,
        max_subjects=args.max_subjects,
        engineering_only=bool(args.synthetic),
    )
    if args.synthetic:
        inputs = _synthetic_inputs(args, config, stage="S5")
    else:
        inputs = _production_inputs(args, config, stage="S5")
    resolved_config = getattr(getattr(inputs, "execution", None), "config", config)
    config = resolved_config
    resolved_config_hash = hashlib.sha256(json.dumps(_jsonable(config.as_dict()), sort_keys=True).encode()).hexdigest()
    review_context = _review_context(
        scope="R7-calibration-cohort",
        config_hash=resolved_config_hash,
        inputs=inputs,
        bundle=bundle,
        args=args,
        policy="adaptive-calibration",
        budget=4,
        value_identity_hash=value_artifact.value_fit_identity.digest,
        split_role="calibration",
    )
    review_receipt = _validate_review_receipt(
        args.review_receipt,
        synthetic=bool(args.synthetic),
        config_hash=resolved_config_hash,
        expected_context=review_context,
        expected_scope="R7-calibration-cohort",
        expected_artifacts={
            "checkpoint_sha256": _sha256(args.checkpoint),
            "value_checkpoint_sha256": _sha256(args.value_checkpoint),
            "role_manifest_digest": getattr(getattr(inputs, "role_manifest", None) or bundle.role_manifest, "digest", None),
            "split_hash": getattr(getattr(inputs, "role_manifest", None) or bundle.role_manifest, "baseline_split_hash", None),
        },
    )
    # The runner consumes the actual fitted V module and its strict identity;
    # calibration must never reconstruct a scorer from a filename or invent a
    # score stream.  ValueArtifact stores tensors only, so hydrate the locked
    # SignedValueNet architecture here before handing off StageInputs.
    from smagm.features.point_guided.pfgr_lite.value_net import SignedValueNet

    value_model = SignedValueNet(input_variant=value_artifact.value_fit_identity.input_variant)
    value_model.load_state_dict(value_artifact.state_dict, strict=True)
    value_model.to(args.device).eval()
    inputs = dataclass_replace(
        inputs,
        metadata={
            **dict(inputs.metadata),
            "value_model": value_model,
            "value_fit_identity": value_artifact.value_fit_identity,
            "gain_scale": float(value_artifact.gain_scale["scale"]),
            "gain_scale_hash": value_artifact.value_fit_identity.gain_scale_hash,
            "gain_scale_provenance": dict(value_artifact.gain_scale),
        },
    )
    collected = run_calibration(inputs, options, run_dir / "calibration")
    calibration = collected["calibration"]
    evidence = collected["calibration_evidence"]
    artifacts: dict[str, str] = dict(collected["artifacts"])
    _write_json(
        run_dir / "calibration.json",
        {
            "calibration": None if calibration is None else asdict(calibration),
            "evidence": None if evidence is None else evidence.as_dict(),
            "value_fit_identity_hash": value_artifact.value_fit_identity.digest,
            "insufficient_data": bool(collected.get("metrics", {}).get("insufficient_data", False)),
            "status": "INCONCLUSIVE" if calibration is None else "SOFTWARE_PASS",
        },
    )
    artifacts["calibration"] = str(run_dir / "calibration.json")
    if calibration is not None and calibration.capability == "adaptive":
        if evidence is None:
            raise CLIError("adaptive calibration must carry a sealed CalibrationEvidence envelope")
        if bundle.role_manifest is not None and bundle.role_manifest.digest != evidence.role_manifest.digest:
            raise CLIError("calibration evidence role manifest does not match producer checkpoint")
        from smagm.features.point_guided.pfgr_lite.policy import load_effective_policy

        resolved_config = getattr(getattr(inputs, "execution", None), "config", config)
        effective_policy = load_effective_policy(
            resolved_config,
            calibration,
            dependencies=bundle.producer,
            capability="adaptive",
            budget=4,
            candidate_chunk_size=config.build_chunk_size,
            random_seed=args.seed,
            value_input_variant=value_artifact.value_fit_identity.input_variant,
            value_fit_identity_hash=value_artifact.value_fit_identity.digest,
            value_fit_identity=value_artifact.value_fit_identity,
            role_manifest_hash=None if evidence.role_manifest is None else evidence.role_manifest.digest,
            gain_scale=float(value_artifact.gain_scale["scale"]),
            gain_scale_hash=value_artifact.value_fit_identity.gain_scale_hash,
            gain_scale_provenance=value_artifact.gain_scale,
        )
        bundle_config = dict(bundle.config)
        bundle_config.update(
            {
                "value_fit_identity_hash": value_artifact.value_fit_identity.digest,
                "gain_scale_hash": value_artifact.value_fit_identity.gain_scale_hash,
                "effective_policy_hash": effective_policy.policy_hash,
            }
        )
        calibrated = replace(
            bundle,
            capability="adaptive",
            calibration=calibration,
            config=bundle_config,
            value_fit_identity=value_artifact.value_fit_identity,
            gain_scale_hash=value_artifact.value_fit_identity.gain_scale_hash,
            effective_policy_hash=effective_policy.policy_hash,
            role_manifest=evidence.role_manifest,
            calibration_evidence=evidence.as_dict(),
            effective_policy=effective_policy.as_dict(),
            gain_scale_provenance=value_artifact.gain_scale,
        )
        checkpoint_out = run_dir / "adaptive.pt"
        save_inference_bundle(checkpoint_out, calibrated)
        # Exercise the strict adaptive artifact decoder immediately; a
        # successful write without this round-trip would not prove nested
        # value/calibration identities agree with the checkpoint envelope.
        load_inference_bundle(checkpoint_out, required_capability="adaptive")
        artifacts["adaptive_checkpoint"] = str(checkpoint_out)
    status = "SOFTWARE_PASS" if calibration is not None and calibration.capability == "adaptive" else "INCONCLUSIVE"
    metrics = dict(collected["metrics"])
    # S5's runner owns one aggregate OperationCounters sink across collection,
    # replay and target-after-trace measurement.  Do not overwrite that
    # measured envelope with the fixture/factory metadata sink (which is a
    # separate input-level counter for ordinary stage receipts).
    input_counts = _counter_receipt(inputs)
    input_counts.pop("operation_counters", None)
    input_counts.pop("operation_counter_schema_version", None)
    metrics.update(input_counts)
    counts = metrics | input_counts
    metrics["review_receipt"] = {"schema_version": review_receipt["schema_version"], "scope": review_receipt["scope"], "decision": review_receipt["decision"], "cohort_hash": review_receipt["cohort_hash"]}
    return _publish_receipt(args, "calibrate", run_dir, status=status, scientific_status="NOT_EVALUATED", counts=counts, metrics=metrics, artifacts=artifacts, config_hash=resolved_config_hash)


def _resume_command(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = _reserve_run(args, "resume")
    if args.dry_manifest:
        return _dry_manifest(args, "resume", run_dir, planned={"resume_checkpoint": str(args.resume_checkpoint), "explicit_stage_restore": True})
    from dataclasses import replace as dataclass_replace

    from smagm.features.point_guided.pfgr_lite.checkpoint import (
        CHECKPOINT_CONFIG_SCHEMA,
        hydrate_inference_model,
        load_resume,
        save_inference_bundle,
        save_resume,
    )
    from smagm.features.point_guided.pfgr_lite.footprint import PFGRQueryLattice
    from smagm.features.point_guided.pfgr_lite.stages import StageExecutionConfig, StageOptions, _input_manifest_hash, run_stage
    resumed = load_resume(args.resume_checkpoint)
    stage = resumed.stage_state.stage
    # ValueNet fitting has its own strict optimizer/RNG/cursor envelope inside
    # ``bank_state.cached_value_fit``.  It does not pass through W3's S0--S2
    # stage runner, but it still uses the same PFGR config identity and the
    # immutable bank/role/producer joins.
    if stage == "value_fit":
        if args.bank_index is None:
            raise CLIError("value-fit resume requires --bank-index for the immutable measured bank")
        config, details = _config_for_command(args, stage="S3")
        config = _resolve_cached_value_config(config, resumed.inference)
        expected = resumed.inference.config.get("pfgr_config") if isinstance(resumed.inference.config, Mapping) else None
        if expected != config.as_dict():
            raise CLIError("resume checkpoint PFGR config does not match the supplied strict execution config")
        cached = resumed.bank_state.get("cached_value_fit") if isinstance(resumed.bank_state, Mapping) else None
        if not isinstance(cached, Mapping):
            raise CLIError("value-fit resume artifact lacks the complete cached fit state")
        from smagm.features.point_guided.pfgr_lite.checkpoint import save_value_artifact
        from smagm.features.point_guided.pfgr_lite.value_bank import ValueBankReader
        from smagm.features.point_guided.pfgr_lite.value_net import fit_value

        reader = ValueBankReader(args.bank_index)
        manifest = reader.manifest()
        producer_hash = resumed.inference.producer.compatibility_hash
        if manifest.producer_compatibility_hash != producer_hash:
            raise CLIError("value-fit resume bank producer identity does not match the inference envelope")
        reader.validate_producer(resumed.inference.producer)
        saved_roles = resumed.inference.role_manifest
        bank_roles = reader.role_manifest
        if saved_roles is not None and bank_roles is not None and saved_roles.digest != bank_roles.digest:
            raise CLIError("value-fit resume role manifest does not match the supplied immutable bank")
        if saved_roles is not None:
            reader.validate_split_role(saved_roles.digest)
        if saved_roles is None and not args.synthetic:
            raise CLIError("production value-fit resume requires a complete training role manifest")
        if not args.synthetic:
            required_paths = {
                "data_root": getattr(args, "data_root", None),
                "split_file": getattr(args, "split_file", None),
                "roles_file": getattr(args, "roles_file", None),
            }
            missing_paths = [f"--{name.replace('_', '-')}" for name, value in required_paths.items() if value is None]
            if missing_paths:
                raise CLIError("production value-fit resume requires the original split/role joins: " + ", ".join(missing_paths))
            if not required_paths["data_root"].is_dir() or not required_paths["split_file"].is_file() or not required_paths["roles_file"].is_file():
                raise CLIError("production value-fit resume split/role/data paths must exist before fitting")
            from smagm.data.brats21_point_guided import load_point_guided_split
            from smagm.features.point_guided.pfgr_lite.types import TrainingRoleManifest

            split = load_point_guided_split(required_paths["split_file"])
            role_payload = _load_json(required_paths["roles_file"])
            role_manifest = TrainingRoleManifest.from_dict(role_payload)
            if role_manifest.baseline_split_hash != split.split_hash:
                raise CLIError("production value-fit resume role manifest baseline split does not match the reviewed split")
            if resumed.inference.split_hash != split.split_hash:
                raise CLIError("production value-fit resume split hash does not match the inference envelope")
            if saved_roles is None or role_manifest.digest != saved_roles.digest:
                raise CLIError("production value-fit resume supplied role manifest does not match the inference envelope")
        saved_bank = resumed.bank_state.get("bank_state")
        if not isinstance(saved_bank, Mapping):
            raise CLIError("value-fit resume artifact lacks bank dependency identity")
        expected_bank = {
            "manifest_hash": reader.manifest_hash,
            "split_role_hash": manifest.split_role_hash,
            "gain_scale_hash": reader.gain_scale.digest,
        }
        for key, value in expected_bank.items():
            if saved_bank.get(key) != value:
                raise CLIError(f"value-fit resume {key} does not match the supplied immutable bank")
        saved_variant = cached.get("input_variant")
        requested_variant = args.value_input if args.value_input is not None else saved_variant
        if requested_variant not in (126, 222, 270, 366):
            raise CLIError("value-fit resume cached input_variant is missing or unsupported")
        if saved_variant != requested_variant:
            raise CLIError("value-fit resume descriptor variant differs from cached optimizer state")
        stage_options = details["execution"].get("stage_options", {})
        epochs = max(1, int(args.epochs or stage_options.get("epochs", 1)))
        batch_size = max(1, int(args.batch_size or stage_options.get("batch_size", 32)))
        fit = fit_value(
            reader,
            config=config.value,
            input_variant=requested_variant,
            epochs=epochs,
            batch_size=batch_size,
            seed=args.seed,
            device=args.device,
            learning_rate=args.learning_rate,
            max_updates=args.max_steps,
            resume=cached,
        )
        role_manifest = resumed.inference.role_manifest or reader.role_manifest
        config_hash = hashlib.sha256(json.dumps(config.as_dict(), sort_keys=True).encode()).hexdigest()
        if bool(fit.complete):
            artifact_path = run_dir / "value.pt"
            save_value_artifact(
                artifact_path,
                fit,
                producer=resumed.inference.producer,
                config=config.value,
                role_manifest=role_manifest,
                stage_provenance=reader.stage_provenance,
            )
            metrics = dict(fit.metrics)
            metrics.update({"artifact": str(artifact_path), "input_variant": requested_variant, "bank_manifest_hash": reader.manifest_hash, "producer_compatibility_hash": producer_hash, "resumed_from": str(args.resume_checkpoint)})
            _write_json(run_dir / "value_fit.json", metrics)
            return _publish_receipt(args, "resume", run_dir, stage=stage, status="SOFTWARE_PASS", counts={"rows": int(metrics.get("row_count", 0)), "updates": int(metrics.get("train_batch_count", 0)), "target_reads": 0, "teacher_calls": 0}, metrics=metrics, artifacts={"value_artifact": str(artifact_path), "resume_source": str(args.resume_checkpoint)}, config_hash=config_hash)
        from smagm.features.point_guided.pfgr_lite.types import StageState

        stage_state = fit.stage_state or StageState(stage="value_fit", substage="interrupted", epoch=0, update=0, completion="pending")
        resume_path = run_dir / "value-resume.pt"
        save_resume(
            resume_path,
            resumed.inference,
            stage_state,
            {},
            {},
            {
                "stage": "value_fit",
                "parent_inference": "inference.pt",
                "producer_compatibility_hash": producer_hash,
                "bank_state": expected_bank,
                "cached_value_fit": fit.resume_state,
            },
        )
        metrics = dict(fit.metrics)
        metrics.update({"fit_complete": False, "resumed_from": str(args.resume_checkpoint), "resume_artifact": str(resume_path), "input_variant": requested_variant})
        _write_json(run_dir / "value_fit_incomplete.json", metrics)
        return _publish_receipt(args, "resume", run_dir, stage=stage, status="INCONCLUSIVE", scientific_status="NOT_EVALUATED", counts={"rows": int(metrics.get("row_count", 0)), "updates": int(metrics.get("train_batch_count", 0)), "target_reads": 0, "teacher_calls": 0}, metrics=metrics, artifacts={"resume_source": str(args.resume_checkpoint), "resume_checkpoint": str(resume_path)}, config_hash=config_hash)

    config, details = _config_for_command(args, stage=stage)
    expected = resumed.inference.config.get("pfgr_config") if isinstance(resumed.inference.config, Mapping) else None
    if not isinstance(expected, Mapping):
        raise CLIError("resume checkpoint is missing its strict PFGR config envelope")
    from smagm.features.point_guided.pfgr_lite.config import PFGRLiteConfig

    _validate_resolved_pfgr_config(config, PFGRLiteConfig.from_dict(expected))
    if stage not in {"S0", "S1", "S2"}:
        raise CLIError(f"resume supports explicit S0/S1/S2 stage replay, not {stage!r}")
    runtime_state = resumed.bank_state.get("stage_runtime") if isinstance(resumed.bank_state, Mapping) else None
    if not isinstance(runtime_state, Mapping):
        if resumed.stage_state.update > 0:
            raise CLIError("resume artifact lacks required stage_runtime snapshots; incomplete optimization state is not resumable")
        runtime_state = None
    if runtime_state is not None:
        required_runtime = {"schema_version", "stage_state", "optimizer_state", "rng_state", "cursor", "parameter_names", "execution_config", "execution_config_hash", "training_config_hash", "producer_compatibility_hash", "split_role_hash", "input_manifest_hash"}
        allowed_runtime = required_runtime | {"continuation"}
        if (set(runtime_state) - allowed_runtime) or (required_runtime - set(runtime_state)) or runtime_state.get("schema_version") != "pfgr-lite-stage-runtime-v1":
            raise CLIError("resume stage_runtime metadata is incomplete or unknown")
    # Hydrate the exact resumed inference envelope through the same W3 factory
    # path used by normal stages.  A fresh external MedicalNet construction
    # would make resume appear valid while silently changing frozen weights.
    resumed_inference_path = run_dir / "resumed_inference.pt"
    save_inference_bundle(resumed_inference_path, resumed.inference)
    setattr(args, "checkpoint", resumed_inference_path)
    inputs = _inputs(args, config, stage=stage)
    saved_roles = resumed.inference.role_manifest
    current_roles = getattr(inputs, "role_manifest", None)
    if saved_roles is not None and current_roles is not None and saved_roles.digest != current_roles.digest:
        raise CLIError("resume supplied role manifest does not match the original checkpoint role manifest")
    if saved_roles is not None and current_roles is None and not args.synthetic:
        raise CLIError("production resume requires the original role manifest join")
    factory_execution = getattr(inputs, "execution", None)
    if not args.synthetic:
        if factory_execution is None:
            raise CLIError("production resume StageInputs must expose factory-resolved execution")
        _validate_resolved_pfgr_config(config, factory_execution.config)
        # The factory necessarily resolves the resumed inference checkpoint as
        # a new ``checkpoint_path``.  That load path is provenance for this
        # invocation, not a new training identity.  First verify all immutable
        # joins against the saved runtime envelope, then retain the original
        # execution sidecar/config for strict hash validation; only the
        # explicitly requested max_updates continuation may differ.
        if runtime_state is not None:
            saved_execution = StageExecutionConfig.from_dict(runtime_state["execution_config"])
            _validate_resolved_pfgr_config(saved_execution.config, factory_execution.config)
            if saved_execution.normalization_hash != factory_execution.normalization_hash:
                raise CLIError("resume normalization identity does not match the saved execution envelope")
            current_manifest_hash = _input_manifest_hash(inputs)
            if runtime_state.get("input_manifest_hash") != current_manifest_hash:
                raise CLIError("resume input manifest does not match the saved observation records")
            current_producer = getattr(inputs, "producer", None)
            current_producer_hash = getattr(current_producer, "compatibility_hash", None)
            if current_producer_hash is None:
                raise CLIError("production resume factory did not expose the hydrated producer identity")
            if runtime_state.get("producer_compatibility_hash") != current_producer_hash:
                raise CLIError("resume producer compatibility identity does not match the saved stage")
            # Hydration must consume the canonical sidecar from the saved
            # inference bundle even though the factory execution receipt has
            # a new parent checkpoint path.
            actual_frontend = _frontend_artifact_for_model(
                getattr(inputs, "model", None),
                resumed.inference.frontend_config,
            )
            if actual_frontend != resumed.inference.frontend_config:
                raise CLIError("resume hydrated frontend configuration differs from the saved inference artifact")
            requested_options = factory_execution.stage_options
            execution_identity = dataclass_replace(saved_execution, stage_options=requested_options)
            # The restored run must execute and serialize the exact saved
            # execution identity (with only the explicitly requested options
            # override).  Keeping the factory's loader-side execution object
            # here would leave checkpoint paths/normalization provenance in
            # ``inputs.execution`` while hashes/details use the saved one.
            factory_execution = execution_identity
            inputs = dataclass_replace(inputs, execution=execution_identity)
            config = execution_identity.config
            details = {**details, "execution": execution_identity.as_dict()}
        else:
            config = factory_execution.config
            details = {**details, "execution": factory_execution.as_dict()}
    else:
        factory_execution = StageExecutionConfig.from_dict(details["execution"])
        inputs = dataclass_replace(inputs, execution=factory_execution)
    if runtime_state is not None:
        _, current_training_hash = _runtime_hashes(details["execution"])
        if runtime_state["training_config_hash"] != current_training_hash:
            raise CLIError("resume training_config_hash mismatch; only an explicit max_updates continuation override is allowed")
    import torch

    model = hydrate_inference_model(resumed.inference, query_lattice_factory=PFGRQueryLattice).to(torch.device(args.device))
    options = getattr(factory_execution, "stage_options", None)
    if options is None:
        options = StageOptions.from_dict(details["execution"].get("stage_options", {}))
    # StageState.update is the committed batch cursor for S0/S1.  Keep the
    # complete ordered sample sequence: W3b's strict restore path validates
    # the cursor against that sequence and skips completed batches itself.
    # The caller may raise ``--max-steps`` explicitly to request additional
    # same-stage work; silently changing the cap here would invalidate the
    # execution identity.
    samples = tuple(getattr(inputs, "samples", ()))
    if not samples:
        raise CLIError("resume cursor is at or beyond the available stage samples; no implicit next stage is selected")
    execution = getattr(inputs, "execution", None)
    if execution is not None:
        execution = dataclass_replace(execution, stage_options=options)
    inputs = dataclass_replace(
        inputs,
        model=model,
        producer=resumed.inference.producer,
        role_manifest=resumed.inference.role_manifest,
        optimizer=None,
        samples=samples,
        execution=execution,
        stage_options=options,
        resume=runtime_state,
    )
    # W3's strict run_stage/_restore_runtime validates and restores optimizer
    # and RNG snapshots before any route or sample work.  Duplicating that
    # restore here would allow a malformed snapshot to perturb process state.
    result = run_stage(stage, config, inputs, run_dir / stage.lower())
    sample = tuple(getattr(inputs, "samples", ()))
    if not sample or not hasattr(model, "encode_observations"):
        raise CLIError("resume stage did not expose an observation context for strict checkpoint publication")
    item = sample[0]
    context = _encode_artifact_context(model, item, args.device)
    stage_metrics = dict(result.metrics)
    source_provenance = getattr(context.producer, "source_provenance", None)
    traversal_count = getattr(source_provenance, "traversal_count", None)
    if isinstance(traversal_count, int) and not isinstance(traversal_count, bool):
        stage_metrics["artifact_context_traversal_count"] = traversal_count
    bundle_config = dict(resumed.inference.config)
    bundle_config["schema_version"] = CHECKPOINT_CONFIG_SCHEMA
    bundle = dataclass_replace(
        resumed.inference,
        state_dict=model.state_dict(),
        producer=context.producer,
        config=bundle_config,
        stage_provenance=result.receipt.stage_provenance,
    )
    inference_path = run_dir / "inference.pt"
    save_inference_bundle(inference_path, bundle)
    resume_path = run_dir / "resume.pt"
    # W3b's resumed StageResult already carries the restored cumulative
    # cursor (its optimizer/update counters start from the loaded snapshot).
    # Adding the parent StageState again would double-count updates.
    cumulative_stage_state = result.stage_state
    runtime_next = dict(
        _stage_runtime(
            result=result,
            execution=details["execution"],
            optimizer=None,
            model=model,
            rng_state=_capture_rng_state(),
            producer_hash=context.producer.compatibility_hash,
            split_role_hash=getattr(getattr(inputs, "role_manifest", None), "digest", None if args.synthetic else ""),
            input_manifest_hash=_input_manifest_hash(inputs),
            sample_order=tuple(item.subject_id for item in samples),
        )
    )
    runtime_next["stage_state"] = asdict(cumulative_stage_state)
    if isinstance(runtime_next.get("cursor"), Mapping):
        cursor_next = dict(runtime_next["cursor"])
        cursor_next["update"] = int(cumulative_stage_state.update)
        cursor_next["batch_index"] = int(cursor_next.get("batch_index", cumulative_stage_state.update))
        runtime_next["cursor"] = cursor_next
    save_resume(
        resume_path,
        bundle,
        cumulative_stage_state,
        runtime_next["optimizer_state"],
        runtime_next["rng_state"],
        {"stage": stage, "resumed_from": str(args.resume_checkpoint), "producer_compatibility_hash": context.producer.compatibility_hash, "stage_runtime": runtime_next},
    )
    summary = {
        "schema_version": "pfgr-lite-resume-summary-v1",
        "status": "SOFTWARE_PASS",
        "stage": result.stage_state.stage,
        "substage": result.stage_state.substage,
        "epoch": cumulative_stage_state.epoch,
        "update": cumulative_stage_state.update,
        "microstep": cumulative_stage_state.microstep,
        "optimizer_groups": list(cumulative_stage_state.optimizer_groups),
        "completion": cumulative_stage_state.completion,
        "producer_compatibility_hash": context.producer.compatibility_hash,
        "split_hash": bundle.split_hash,
        "restored_rng_streams": sorted(resumed.rng_state),
        "optimizer_restored": bool(runtime_next.get("optimizer_state")),
        "implicit_next_stage": False,
        "resumed_from": str(args.resume_checkpoint),
    }
    if isinstance(traversal_count, int) and not isinstance(traversal_count, bool):
        summary["artifact_context_traversal_count"] = traversal_count
    _write_json(run_dir / "resolved_config.json", details["execution"])
    _write_json(run_dir / "stage_state.json", asdict(cumulative_stage_state))
    _write_json(run_dir / "metrics.json", stage_metrics)
    runtime_receipt = {key: value for key, value in runtime_next.items() if key not in {"optimizer_state", "rng_state", "cursor"}}
    cursor_receipt = dict(runtime_next.get("cursor", {})) if isinstance(runtime_next.get("cursor"), Mapping) else {}
    cursor_receipt["route_rng_state"] = sorted(cursor_receipt.get("route_rng_state", {})) if isinstance(cursor_receipt.get("route_rng_state"), Mapping) else "present"
    runtime_receipt["cursor"] = cursor_receipt
    runtime_receipt.update({"optimizer_state_present": bool(runtime_next.get("optimizer_state")), "rng_streams": sorted(runtime_next.get("rng_state", {}))})
    _write_json(run_dir / "stage_runtime.json", runtime_receipt)
    _write_json(run_dir / "resume_summary.json", summary)
    return _publish_receipt(args, "resume", run_dir, stage=stage, counts={"target_reads": 0, "restored_updates": resumed.stage_state.update, "updates": cumulative_stage_state.update}, metrics=summary | {"stage_metrics": stage_metrics}, artifacts={"resume_source": str(args.resume_checkpoint), "inference_checkpoint": str(inference_path), "resume_checkpoint": str(resume_path)}, config_hash=hashlib.sha256(json.dumps(config.as_dict(), sort_keys=True).encode()).hexdigest())


def _smoke_command(args: argparse.Namespace) -> dict[str, Any]:
    # ``smoke`` is intentionally the same concrete S0 stage used by
    # ``static-train``.  Keeping one implementation guarantees that the
    # checkpoint/resume artifacts consumed by the causal runbook are real,
    # while the command remains a bounded engineering pilot (it does not
    # claim that S1/U has run).
    return _stage_command(args, "smoke", stage="S0")


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = _reserve_run(args, "preflight")
    if args.dry_manifest:
        return _dry_manifest(args, "preflight", run_dir, planned={"write_roles": bool(args.write_roles), "target_reads": 0})
    config, details = _config_for_command(args)
    missing = _input_missing(args, require_real=not args.synthetic)
    if missing:
        raise CLIError("preflight failed: " + "; ".join(missing))
    _write_json(run_dir / "resolved_config.json", details["execution"])
    if args.synthetic:
        from smagm.features.point_guided.pfgr_lite.types import TrainingRoleManifest

        roles = TrainingRoleManifest(
            baseline_split_hash="synthetic-split-v1",
            baseline_train_subject_ids=("synthetic-00",),
            baseline_validation_subject_ids=(),
            baseline_test_subject_ids=(),
            producer_fit_subject_ids=("synthetic-00",),
            calibration_fit_subject_ids=(),
            calibration_allowance_subject_ids=(),
            subject_group_ids=(("synthetic-00", "group-00"),),
            engineering_only=True,
        )
        _write_json(run_dir / "roles.json", roles.as_dict())
        _write_json(run_dir / "split.json", {"schema_version": "synthetic-split-v1", "split_hash": "synthetic", "engineering_only": True})
        _write_json(run_dir / "source.json", _source_receipt())
        _write_json(run_dir / "weights.json", {"status": "synthetic_untrained", "source_input_channels": 3, "adapted_input_channels": 3, "input_conv_adapted": False, "official_pretrained_verified": False, "integrity_verified": False, "synthetic_untrained": True})
    else:
        from smagm.data.brats21_point_guided import load_point_guided_split

        split = load_point_guided_split(args.split_file)
        split_payload = split.as_dict() if hasattr(split, "as_dict") else {"split_hash": split.split_hash, "train": list(split.train_subject_ids), "val": list(split.val_subject_ids), "test": list(split.test_subject_ids)}
        _write_json(run_dir / "split.json", split_payload)
        if args.write_roles:
            from smagm.features.point_guided.pfgr_lite.data import build_training_role_manifest

            roles = build_training_role_manifest(split, engineering_only=False)
            _write_json(run_dir / "roles.json", roles.as_dict())
        _write_json(run_dir / "source.json", _source_receipt())
        # Use the same strict MedicalNet loader as the frontend so source
        # channels/adaptation/integrity are evidence, not a filename claim.
        from smagm.features.point_guided.medicalnet_resnet10 import MedicalNetResNet10, load_medicalnet_checkpoint

        frontend_payload = details.get("execution", {}).get("frontend_sidecar", {})
        frontend_values = frontend_payload.get("config", {}) if isinstance(frontend_payload, Mapping) else {}
        backbone = MedicalNetResNet10(in_channels=3)
        provenance = load_medicalnet_checkpoint(
            backbone,
            args.medicalnet_checkpoint,
            expected_sha256=args.medicalnet_sha256,
            require_official_pretrained=bool(frontend_values.get("require_pretrained_backbone", False)),
        )
        _write_json(run_dir / "weights.json", {
            "checkpoint": str(args.medicalnet_checkpoint.resolve()),
            "sha256": provenance.sha256,
            "expected_sha256": args.medicalnet_sha256,
            "source_input_channels": provenance.source_input_channels,
            "adapted_input_channels": provenance.adapted_input_channels,
            "input_conv_adapted": provenance.input_conv_adapted,
            "checkpoint_integrity_verified": provenance.integrity_verified,
            "official_pretrained_verified": provenance.official_pretrained_verified,
            "source_state_dict_key_count": provenance.source_state_dict_key_count,
            "loaded_backbone_key_count": provenance.loaded_backbone_key_count,
            "synthetic_untrained": not provenance.official_pretrained_verified,
        })
    _write_json(run_dir / "environment.json", _environment_receipt(args.device))
    return _publish_receipt(args, "preflight", run_dir, stage="preflight", counts={"target_reads": 0}, config_hash=hashlib.sha256(json.dumps(config.as_dict(), sort_keys=True).encode()).hexdigest())


def _counter_receipt(inputs: Any) -> dict[str, Any]:
    metadata = getattr(inputs, "metadata", {})
    counters = metadata.get("counters") if isinstance(metadata, Mapping) else None
    if counters is None and isinstance(metadata, Mapping):
        counters = metadata.get("data_counters")
    if counters is None:
        return {"observation_reads": None, "target_reads": None, "segmentation_reads": None}
    if hasattr(counters, "as_dict") and callable(counters.as_dict):
        values = counters.as_dict()
    elif isinstance(counters, Mapping):
        values = dict(counters)
    else:
        values = {name: getattr(counters, name) for name in ("observation_reads", "target_reads", "segmentation_reads") if hasattr(counters, name)}
    payload = {
        name: (None if values.get(name) is None else int(values[name]))
        for name in ("observation_reads", "target_reads", "segmentation_reads")
    }
    operation = metadata.get("operation_counters") if isinstance(metadata, Mapping) else None
    if operation is not None:
        if hasattr(operation, "as_dict") and callable(operation.as_dict):
            operation = operation.as_dict()
        if isinstance(operation, Mapping):
            payload["operation_counters"] = {
                str(name): int(value)
                for name, value in operation.items()
                if name != "schema_version" and isinstance(value, int) and not isinstance(value, bool)
            }
            payload["operation_counter_schema_version"] = operation.get("schema_version", "pfgr-lite-operation-counters-v1")
    return payload


def _service_details(command: str, result: Mapping[str, Any], inputs: Any, config: Any) -> tuple[dict[str, Any], dict[str, Any], str | None, str | None]:
    """Extract measured service work and provenance from generated artifacts.

    W5 does not infer work from a requested option.  Services already emit
    detailed rows/metrics; this boundary reads those outputs and publishes
    measured counters, source identity and explicit ``None`` when a service
    did not expose a counter.
    """

    artifacts: dict[str, Any] = {}
    for key, value in result.items():
        if key.endswith("_path") or key.endswith("_checkpoint"):
            artifacts[key] = str(value)
    primary: dict[str, Any] = {}
    primary_path = result.get("benchmark_path") if command == "benchmark" else result.get("metrics_path")
    if primary_path is None and command == "oracle-evaluate":
        primary_path = result.get("output_path")
    if primary_path is not None:
        path = Path(str(primary_path))
        if path.is_file() and path.suffix.lower() == ".json":
            loaded = _load_json(path)
            primary = loaded
    counters = _counter_receipt(inputs)
    operation_calls: dict[str, Any] = {}
    if command == "benchmark" and primary:
        query_calls: dict[str, int] = {}
        decoder_calls: dict[str, int] = {}
        row_path = result.get("rows_path")
        rows: list[Mapping[str, Any]] = []
        if row_path is not None and Path(str(row_path)).is_file():
            for line in Path(str(row_path)).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, Mapping):
                        rows.append(value)
        for row in rows:
            for name, destination in (("query_calls", query_calls), ("decoder_calls", decoder_calls)):
                values = row.get(name)
                if isinstance(values, Mapping):
                    for key, value in values.items():
                        if isinstance(value, int) and not isinstance(value, bool):
                            destination[str(key)] = destination.get(str(key), 0) + int(value)
        operation_calls = {
            "query_calls": query_calls,
            "decoder_calls": decoder_calls,
            "reference_full_write_calls": len(rows) if rows else None,
            "rows_measured": len(rows),
        }
    elif command == "evaluate" and primary:
        route_rows = primary.get("stopping", {})
        operation_calls = {
            "route_decisions": len(route_rows.get("rows", ())) if isinstance(route_rows, Mapping) and isinstance(route_rows.get("rows"), list) else None,
            "action_rows": primary.get("actions", {}).get("count") if isinstance(primary.get("actions"), Mapping) else None,
            "decoder_calls": None,
            "teacher_calls": None,
        }
        input_counters = primary.get("input_data_counters")
        if isinstance(input_counters, Mapping):
            counters.update({name: input_counters.get(name) for name in counters})
    elif command == "oracle-evaluate":
        output_path = result.get("output_path")
        rows = []
        if output_path is not None and Path(str(output_path)).is_file():
            for line in Path(str(output_path)).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, Mapping):
                        rows.append(value)
        operation_calls = {
            "subjects": len(rows),
            "candidate_rows": sum(len(row.get("rows", ())) for row in rows),
            "confirmation_rows": sum(len(row.get("confirmation", ())) for row in rows),
            "teacher_calls": None,
        }
    source_receipt = primary.get("source_receipt") if isinstance(primary.get("source_receipt"), Mapping) else None
    effective_policy_hash: str | None = None
    policy_path = result.get("effective_policy_path")
    if policy_path is not None and Path(str(policy_path)).is_file():
        effective_policy_hash = _sha256(Path(str(policy_path)))
    config_hash = hashlib.sha256(json.dumps(_jsonable(config.as_dict()), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    details = {
        "service_result": _jsonable(dict(result)),
        "service_artifacts": artifacts,
        "service_source_receipt": _jsonable(source_receipt) if source_receipt is not None else None,
        "operation_calls": operation_calls,
        "data_counters": counters,
        "primary_artifact_summary": {
            "schema_version": primary.get("schema_version"),
            "software_status": primary.get("software_status"),
            "scientific_status": primary.get("scientific_status"),
        },
    }
    return details, counters | operation_calls, config_hash, effective_policy_hash


def _service_command(args: argparse.Namespace, command: str) -> dict[str, Any]:
    """Dispatch W2/W3/W5 service calls without cloning their implementations."""

    run_dir = _reserve_run(args, command)
    if args.dry_manifest:
        return _dry_manifest(args, command, run_dir, planned={"command": command, "target_reads": "post-trace only"})
    if command == "benchmark":
        from smagm.features.point_guided.pfgr_lite.benchmark import BenchmarkOptions, run_teacher_benchmark

        config, _ = _config_for_command(args)
        inputs = _inputs(args, config)
        result = run_teacher_benchmark(
            inputs,
            BenchmarkOptions(
                max_subjects=args.max_subjects or 1,
                repeats=args.repeats,
                query_count=args.query_count,
                candidate_count=args.candidate_count,
                max_states=args.max_states,
                teacher_mode=args.teacher_mode,
                seed=args.seed,
                chunk_size=args.decode_chunk_size,
                candidate_chunk_size=args.candidate_chunk_size,
                engineering_only=args.synthetic,
            ),
            run_dir,
        )
    elif command == "evaluate":
        from smagm.features.point_guided.pfgr_lite.experiments import ExperimentOptions, run_evaluation
        from dataclasses import replace as dataclass_replace

        config, _ = _config_for_command(args)
        inputs = _inputs(args, config)
        bundle = None
        artifact = None
        if args.scenario in {"adaptive", "fixed_learned", "parallel_topk"}:
            if args.value_checkpoint is None:
                raise CLIError(f"{args.scenario} evaluation requires --value-checkpoint; learned/adaptive policy must join the exact V artifact")
            from smagm.features.point_guided.pfgr_lite.checkpoint import load_inference_bundle, load_value_artifact
            from smagm.features.point_guided.pfgr_lite.value_net import SignedValueNet

            bundle = load_inference_bundle(args.checkpoint)
            artifact = load_value_artifact(
                args.value_checkpoint,
                expected_producer=bundle.producer,
                expected_role_manifest=bundle.role_manifest,
            )
            value_model = SignedValueNet(input_variant=artifact.value_fit_identity.input_variant)
            value_model.load_state_dict(dict(artifact.state_dict), strict=True)
            value_model.to(args.device).eval()
            if args.scenario == "adaptive" and bundle.capability != "adaptive":
                raise CLIError("adaptive evaluation requires an adaptive checkpoint carrying sealed calibration evidence")
            metadata = dict(getattr(inputs, "metadata", {}) or {})
            metadata.update(
                {
                    "value_model": value_model,
                    "value_fit_identity": artifact.value_fit_identity,
                    "gain_scale": float(artifact.gain_scale["scale"]),
                    "gain_scale_hash": artifact.value_fit_identity.gain_scale_hash,
                    "gain_scale_provenance": dict(artifact.gain_scale),
                    "calibration": bundle.calibration,
                    "effective_policy": None,
                }
            )
            inputs = dataclass_replace(inputs, metadata=metadata)
        else:
            from smagm.features.point_guided.pfgr_lite.checkpoint import load_inference_bundle

            bundle = load_inference_bundle(args.checkpoint)
        resolved_config = getattr(getattr(inputs, "execution", None), "config", config)
        config = resolved_config
        if str(args.split_role) == "test":
            if args.review_receipt is None:
                raise CLIError("held-out test evaluation requires an explicit --review-receipt")
            review_config_hash = hashlib.sha256(json.dumps(_jsonable(config.as_dict()), sort_keys=True).encode()).hexdigest()
            review_context = _review_context(
                scope="R9-final-evaluation",
                config_hash=review_config_hash,
                inputs=inputs,
                bundle=bundle,
                args=args,
                policy=str(args.scenario),
                budget=int(args.budget),
                value_identity_hash=None if artifact is None else artifact.value_fit_identity.digest,
                split_role="test",
            )
            expected_artifacts = {"checkpoint_sha256": _sha256(args.checkpoint)}
            if args.value_checkpoint is not None:
                expected_artifacts["value_checkpoint_sha256"] = _sha256(args.value_checkpoint)
            _validate_review_receipt(
                args.review_receipt,
                synthetic=bool(args.synthetic),
                config_hash=review_config_hash,
                expected_context=review_context,
                expected_scope="R9-final-evaluation",
                expected_artifacts=expected_artifacts,
            )
        result = run_evaluation(
            inputs,
            ExperimentOptions(
                scenario=args.scenario,
                budget=args.budget,
                max_subjects=args.max_subjects or 1,
                seed=args.seed,
                split_role=args.split_role,
                candidate_chunk_size=args.candidate_chunk_size,
                decode_chunk_size=args.decode_chunk_size,
                teacher_mode=args.teacher_mode,
                query_count=args.query_count,
                local_footprint_audit=bool(args.local_footprint_audit),
                engineering_only=args.synthetic,
            ),
            run_dir,
        )
    elif command == "oracle-evaluate":
        from smagm.features.point_guided.pfgr_lite.oracle import OracleOptions, run_oracle_evaluation

        config, _ = _config_for_command(args)
        inputs = _inputs(args, config)
        result = run_oracle_evaluation(
            inputs,
            OracleOptions(
                mode=args.oracle_mode,
                oracle_mode=args.oracle_mode,
                budget=args.budget,
                candidate_count=args.candidate_count,
                query_count=args.query_count,
                max_subjects=args.max_subjects or 1,
                seed=args.seed,
                split_role=args.split_role,
                teacher_mode=args.teacher_mode,
                confirmation_mode=args.confirmation_mode,
                confirmation_query_count=args.confirmation_query_count,
                engineering_only=args.synthetic,
            ),
            run_dir,
        )
    else:
        raise CLIError(f"no service binding implemented for command {command!r}")
    result_metrics = _jsonable(result)
    config = getattr(getattr(inputs, "execution", None), "config", config)
    details, counts, config_hash, effective_policy_hash = _service_details(command, result, inputs, config)
    _write_json(run_dir / "service_receipt.json", details)
    return _publish_receipt(
        args,
        command,
        run_dir,
        metrics=result_metrics | {"service": details},
        counts=counts,
        config_hash=config_hash,
        effective_policy_hash=effective_policy_hash,
        artifacts=details["service_artifacts"],
    )


def _package_command(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = _reserve_run(args, "package")
    if args.dry_manifest:
        return _dry_manifest(args, "package", run_dir, planned={"run_dirs": [str(path) for path in args.run_dir], "scientific_status": "NOT_EVALUATED"})
    from smagm.features.point_guided.pfgr_lite.artifacts import package_evidence

    package_dir = run_dir / "evidence"
    manifest = package_evidence(args.run_dir, package_dir)
    _write_json(run_dir / "manifest.json", manifest)
    return _publish_receipt(args, "package", run_dir, counts=manifest.get("counts", {}), scientific_status="NOT_EVALUATED", evidence_status=manifest.get("evidence_status"), archive=manifest.get("archive"))


def _runbook_check(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = _reserve_run(args, "runbook-check")
    if not args.runbook.is_file():
        raise FileNotFoundError(f"runbook does not exist: {args.runbook}")
    text = args.runbook.read_text(encoding="utf-8")
    required_sections = [f"R{i}" for i in range(11)]
    missing_sections = [name for name in required_sections if not re.search(rf"(?:^|\n)#+[^\n]*{name}(?:[^\n]*)", text, re.IGNORECASE)]
    command_mentions = sorted(set(re.findall(r"(?:python(?:3)?|\$POINT_GUIDED_PYTHON)[^\n`]*-m\s+smagm\.cli\.pfgr_lite\s+([a-z][a-z-]+)", text)))
    missing_commands = [name for name in COMMANDS if name not in command_mentions]
    config_files = sorted(args.config_dir.glob("*.json")) if args.config_dir.is_dir() else []
    config_errors: list[str] = []
    if not args.config_dir.is_dir():
        config_errors.append(f"config directory does not exist: {args.config_dir}")
    elif not config_files:
        config_errors.append(f"config directory contains no JSON configs: {args.config_dir}")
    for path in config_files:
        try:
            _config_document(path)
        except Exception as error:
            config_errors.append(f"{path}: {type(error).__name__}: {error}")
    shell_blocks = re.findall(r"```(?:bash|sh|shell)\s*\n(.*?)```", text, re.IGNORECASE | re.DOTALL)
    shell_errors: list[str] = []
    for index, block in enumerate(shell_blocks):
        # Feed the block on stdin so heredocs and quoted paths are parsed by
        # the same Bash grammar users will execute, without creating a file
        # in an arbitrary temporary directory.
        checked = subprocess.run(["bash", "-n"], input=block, capture_output=True, text=True, check=False)
        if checked.returncode:
            shell_errors.append(f"block {index}: {checked.stderr.strip() or 'bash -n failed'}")
    status = "SOFTWARE_PASS" if not (missing_sections or missing_commands or config_errors or shell_errors) else "SOFTWARE_FAIL"
    payload = {
        "schema_version": CLI_SCHEMA,
        "status": status,
        "scientific_status": "NOT_EVALUATED",
        "runbook": str(args.runbook),
        "config_dir": str(args.config_dir),
        "required_sections": required_sections,
        "missing_sections": missing_sections,
        "command_mentions": command_mentions,
        "missing_commands": missing_commands,
        "config_files": [str(path) for path in config_files],
        "config_errors": config_errors,
        "shell_errors": shell_errors,
        "synthetic_execution": "not executed by runbook-check; use smoke/evaluate --synthetic explicitly",
    }
    _write_json(run_dir / "runbook_check.json", payload)
    return _publish_receipt(args, "runbook-check", run_dir, status=status, metrics=payload, scientific_status="NOT_EVALUATED")


def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    command = args.command
    if command == "preflight":
        return _preflight(args)
    if command == "smoke":
        return _smoke_command(args)
    if command == "static-train":
        return _stage_command(args, command, stage="S0")
    if command == "updater-train":
        return _stage_command(args, command, stage="S1")
    if command == "bank-build":
        return _stage_command(args, command, stage="S2")
    if command == "bank-verify":
        return _bank_verify_command(args)
    if command == "value-fit":
        return _value_fit_command(args)
    if command == "value-evaluate":
        return _value_evaluate_command(args)
    if command == "calibrate":
        return _calibrate_command(args)
    if command == "resume":
        return _resume_command(args)
    if command in {"benchmark", "evaluate", "oracle-evaluate"}:
        return _service_command(args, command)
    if command == "package":
        return _package_command(args)
    if command == "runbook-check":
        return _runbook_check(args)
    # The remaining commands intentionally require their actual typed W3/W4
    # dependencies.  Their dry manifests are still complete and safe; normal
    # invocation reports an actionable blocker instead of a false PASS.
    run_dir = _reserve_run(args, command)
    if args.dry_manifest:
        return _dry_manifest(args, command, run_dir, planned={"command": command, "requires": "verified predecessor receipt"})
    raise CLIError(f"{command} requires an actual predecessor artifact and service integration; use --dry-manifest to inspect requirements")


def main(argv: list[str] | None = None) -> int:
    global _LAST_RESERVED_RUN_DIR
    _LAST_RESERVED_RUN_DIR = None
    args = _parser().parse_args(argv)
    setattr(args, "_argv", list(sys.argv[1:] if argv is None else argv))
    try:
        result = _dispatch(args)
    except Exception as error:
        # If a command reserved an output directory, publish a failure receipt
        # there; parser failures are handled by argparse before this point.
        run_dir = _LAST_RESERVED_RUN_DIR
        if run_dir is not None:
            receipt = _receipt_base(args, args.command, status="SOFTWARE_FAIL", run_dir=run_dir)
            traceback_text = traceback.format_exc()
            receipt.update({"exit_code": 1, "error": f"{type(error).__name__}: {error}", "traceback_path": "traceback.txt"})
            _write_json(run_dir / "receipt.json", receipt)
            (run_dir / "traceback.txt").write_text(traceback_text, encoding="utf-8")
        traceback.print_exc(file=sys.stderr)
        return 1
    print(json.dumps(_jsonable(result), sort_keys=True, indent=2))
    # A completed runbook/config checker can publish a structured
    # ``SOFTWARE_FAIL`` receipt without raising.  Preserve that diagnostic
    # artifact, but expose the failure to ``set -e`` callers and CI.  Scientific
    # ``INCONCLUSIVE`` is intentionally still a successful software exit.
    if isinstance(result, Mapping) and result.get("status") == "SOFTWARE_FAIL":
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
