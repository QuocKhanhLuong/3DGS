"""GPU-only streamed BraTS21 product controller.

The controller owns cohort iteration, immutable per-patient preparation, and
resume bookkeeping. Shared encoder, Gaussian-head, StructuralField, and
optimizer state is promoted atomically between training patients; patient
Gaussian state and evaluator payloads are never promoted. Scientific
representation work remains in the native PyTorch receipt-gated runner; this
module does not add routing or adaptive acquisition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch

from ..data.brats21 import (
    BRATS21_PATIENT_PATTERN,
    BraTS21ValidationError,
    discover_patient,
    extract_axial_plane_at_position,
    npy_bytes,
    validate_patient,
)
from ..data.brats21_sampling import BraTS21SamplingConfig, build_sampling_plan
from ..data.normalization import NormalizationConfig


_ROOT = Path(__file__).resolve().parents[3]
_PRODUCT_SCHEMA = "smagm-brats21-product-experiment-v1"


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _atomic_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", newline="", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_torch(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_metric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _bootstrap_interval(
    values: list[float], *, seed: int, samples: int, confidence: float,
) -> dict[str, object] | None:
    """Return a deterministic patient-bootstrap percentile interval.

    Non-finite values are excluded by the caller and reported through the
    accompanying metric count. A one-patient interval is intentionally
    degenerate rather than presented as population uncertainty.
    """

    if not values:
        return None
    if samples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap samples must be positive and confidence must be in (0,1)")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("bootstrap values must be finite")
    if len(values) == 1:
        low = high = values[0]
    else:
        generator = random.Random(seed)
        means = sorted(
            sum(values[generator.randrange(len(values))] for _ in values) / len(values)
            for _ in range(samples)
        )
        alpha = (1.0 - confidence) / 2.0
        low_index = max(0, min(len(means) - 1, int(alpha * len(means))))
        high_index = max(0, min(len(means) - 1, int((1.0 - alpha) * len(means)) - 1))
        low, high = means[low_index], means[high_index]
    return {
        "method": "deterministic_patient_bootstrap_percentile",
        "confidence": confidence,
        "samples": samples,
        "seed": seed,
        "low": low,
        "high": high,
    }


_PRODUCT_METRIC_FIELDS = (
    "loss", "final_evaluation_loss", "mae", "rmse", "psnr", "ssim", "ncc",
    "gradient_mae", "gradient_rmse", "frequency_error", "edge_f1", "local_contrast_error",
    "data_range", "complete_mae", "complete_rmse", "complete_psnr", "complete_ssim", "complete_ncc",
    "complete_gradient_mae", "complete_gradient_rmse", "complete_edge_f1",
    "complete_local_contrast_error", "complete_frequency_error",
    "supported_fraction", "unsupported_fraction", "evaluable_voxels", "supported_voxels",
    "unsupported_voxels", "distance_to_context_plane_mean_mm", "distance_to_context_plane_max_mm",
    "context_gap_mm", "error_vs_context_gap_mae", "local_observability_mean",
    "roi_voxels", "supported_roi_fraction", "roi_mae", "boundary_band_voxels",
    "supported_boundary_band_fraction", "boundary_band_mae", "tumor_mae", "non_tumor_mae", "roi_contrast_error",
    "runtime_seconds", "inference_wall_time_seconds",
    "per_plane_latency_seconds", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes",
    "anchor_count", "structural_gaussian_count", "volumetric_gaussian_count",
    "propagation_child_count", "cache_size_bytes", "checkpoint_size_bytes",
    "parameter_count", "trainable_parameter_count", "training_step_flops",
)

_PRODUCT_METRIC_STATUS_FIELDS = (
    "metric_scope", "data_range_source", "ssim_window_policy", "complete_metric_status",
    "distance_to_context_plane_status", "context_gap_status", "local_observability_status", "roi_status",
)


def _summary_path_for_record(record: object, *, output_dir: Path) -> Path:
    if not isinstance(record, dict) or not isinstance(record.get("summary"), str):
        raise ValueError("completed product record has no summary path")
    candidate = Path(str(record["summary"]))
    if not candidate.is_absolute():
        candidate = output_dir / candidate
    candidate = candidate.resolve(strict=True)
    try:
        candidate.relative_to(output_dir.resolve())
    except ValueError as error:
        raise ValueError("completed product summary escapes the run directory") from error
    return candidate


def _product_metric_rows(
    *, state: dict[str, Any], output_dir: Path, propagation_variant: str,
) -> list[dict[str, object]]:
    completed = state.get("completed")
    if not isinstance(completed, dict) or not completed:
        raise ValueError("cannot aggregate product metrics without completed patient records")
    variants = ("r0", f"e2_r4_{propagation_variant}")
    rows: list[dict[str, object]] = []
    for completion_key, record in sorted(completed.items()):
        summary_path = _summary_path_for_record(record, output_dir=output_dir)
        summary = _read_json(summary_path)
        patient_pseudonym = summary.get("patient_pseudonymous_id")
        if not isinstance(patient_pseudonym, str) or not patient_pseudonym.startswith("patient-"):
            raise ValueError("product metric aggregation requires a pseudonymous patient identifier")
        split = summary.get("episode_split", summary.get("split"))
        if not isinstance(split, str) or not split:
            split = str(completion_key).split(":", 1)[0]
        evaluations = summary.get("evaluations")
        if not isinstance(evaluations, dict):
            raise ValueError("completed patient summary has no isolated evaluator results")
        for variant in variants:
            variant_report = summary.get(variant)
            evaluation = evaluations.get(variant)
            if not isinstance(variant_report, dict) or not isinstance(evaluation, dict):
                raise ValueError(f"completed patient summary is missing {variant} results")
            metrics = evaluation.get("metrics")
            if not isinstance(metrics, list) or len(metrics) != 1 or not isinstance(metrics[0], dict):
                raise ValueError(f"completed patient evaluator result for {variant} is invalid")
            metric = metrics[0]
            row: dict[str, object] = {
                "patient_pseudonym": patient_pseudonym,
                "split": split,
                "variant": variant,
                "status": "complete",
            }
            for field in _PRODUCT_METRIC_FIELDS:
                source = metric if field in metric else variant_report
                value = source.get(field) if isinstance(source, dict) else None
                if field == "peak_cuda_allocated_bytes":
                    value = summary.get("peak_cuda_allocated_bytes")
                elif field == "peak_cuda_reserved_bytes":
                    value = summary.get("peak_cuda_reserved_bytes")
                elif field == "cache_size_bytes":
                    value = dict(summary.get("resource_metrics", {})).get("cache_size_bytes")
                elif field == "checkpoint_size_bytes":
                    value = dict(summary.get("resource_metrics", {})).get("checkpoint_size_bytes", value)
                row[field] = value if field in ("evaluable_voxels", "supported_voxels", "unsupported_voxels") else _finite_metric(value)
            for field in _PRODUCT_METRIC_STATUS_FIELDS:
                source = metric if field in metric else variant_report
                row[field] = source.get(field) if isinstance(source, dict) else None
            rows.append(row)
    return rows


def _write_product_metric_reports(
    *, state: dict[str, Any], output_dir: Path, output_paths: dict[str, Any], propagation_variant: str,
    aggregation_config: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Write target-free per-patient and aggregate metrics after isolated evaluation."""

    rows = _product_metric_rows(state=state, output_dir=output_dir, propagation_variant=propagation_variant)
    aggregation_config = dict(aggregation_config or {})
    bootstrap_samples = int(aggregation_config.get("bootstrap_samples", 1000))
    bootstrap_confidence = float(aggregation_config.get("confidence", 0.95))
    bootstrap_seed = int(aggregation_config.get("bootstrap_seed", 0))
    missing_policy = str(aggregation_config.get("missing_policy", "omit_non_finite_with_count"))
    if bootstrap_samples <= 0 or not 0.0 < bootstrap_confidence < 1.0:
        raise ValueError("aggregate bootstrap policy must declare positive samples and confidence in (0,1)")
    if missing_policy != "omit_non_finite_with_count":
        raise ValueError("unsupported aggregate missing metric policy")
    fieldnames = (
        "patient_pseudonym", "split", "variant", "status", *_PRODUCT_METRIC_FIELDS, *_PRODUCT_METRIC_STATUS_FIELDS,
    )
    patient_path = output_dir / str(output_paths["patient_metrics"])
    aggregate_path = output_dir / str(output_paths["aggregate_metrics"])
    _atomic_csv(patient_path, fieldnames, rows)
    aggregate: dict[str, object] = {
        "schema": "smagm-brats21-product-aggregate-metrics-v1",
        "claim_scope": "isolated serialized evaluation diagnostics; no scientific or clinical pass",
        "patient_count": len({str(row["patient_pseudonym"]) for row in rows}),
        "row_count": len(rows),
        "aggregation": {
            "scope": "unweighted patient macro-statistics; unsupported pixels are excluded by the evaluator",
            "missing_policy": missing_policy,
            "missing_values_are_not_zero": True,
            "uncertainty": {
                "method": "deterministic_patient_bootstrap_percentile",
                "samples": bootstrap_samples,
                "confidence": bootstrap_confidence,
                "seed": bootstrap_seed,
            },
        },
        "variants": {},
    }
    by_variant: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    for variant, variant_rows in sorted(by_variant.items()):
        variant_summary: dict[str, object] = {
            "patient_count": len({str(row["patient_pseudonym"]) for row in variant_rows}),
            "metrics": {},
        }
        for field in _PRODUCT_METRIC_FIELDS:
            values = [float(row[field]) for row in variant_rows if _finite_metric(row.get(field)) is not None]
            if values:
                ordered = sorted(values)
                middle = len(ordered) // 2
                median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
                variant_summary["metrics"][field] = {
                    "count": len(values), "mean": sum(values) / len(values), "median": median,
                    "minimum": ordered[0], "maximum": ordered[-1],
                    "bootstrap_confidence_interval": _bootstrap_interval(
                        values,
                        seed=bootstrap_seed + sum(ord(char) for char in f"{variant}:{field}"),
                        samples=bootstrap_samples,
                        confidence=bootstrap_confidence,
                    ),
                }
            else:
                variant_summary["metrics"][field] = {
                    "count": 0, "mean": None, "median": None, "minimum": None, "maximum": None,
                    "bootstrap_confidence_interval": None,
                }
        evaluable = sum(int(row["evaluable_voxels"]) for row in variant_rows if isinstance(row["evaluable_voxels"], int))
        supported = sum(int(row["supported_voxels"]) for row in variant_rows if isinstance(row["supported_voxels"], int))
        unsupported = sum(int(row["unsupported_voxels"]) for row in variant_rows if isinstance(row["unsupported_voxels"], int))
        variant_summary["pooled_support"] = {
            "evaluable_voxels": evaluable,
            "supported_voxels": supported,
            "unsupported_voxels": unsupported,
            "supported_fraction": None if evaluable == 0 else supported / evaluable,
            "unsupported_fraction": None if evaluable == 0 else unsupported / evaluable,
        }
        aggregate["variants"][variant] = variant_summary
    _atomic_json(aggregate_path, aggregate)
    return {
        "schema": aggregate["schema"],
        "patient_metrics": str(patient_path.relative_to(output_dir)),
        "aggregate_metrics": str(aggregate_path.relative_to(output_dir)),
        "row_count": len(rows),
    }


def _promote_global_model_checkpoint(
    training_summary_path: Path,
    destination: Path,
    *,
    cohort_hash: str,
    split_hash: str,
    source_patient_pseudonym: str,
    global_update_index: int,
) -> dict[str, object]:
    """Promote only shared trainable state after a patient succeeds.

    The patient checkpoint is checked for the explicit target-exclusion marker;
    patient Gaussian state and evaluator payloads are not copied into the
    cohort checkpoint. Promotion is atomic so a patient-boundary interruption
    leaves either the previous global state or the complete next one.
    """

    summary = _read_json(training_summary_path)
    candidate_reports = [
        (str(key), value)
        for key, value in summary.items()
        if str(key).startswith("e2_r4_") and isinstance(value, dict)
    ]
    if not candidate_reports:
        raise ValueError(f"training summary has no promotable E2/R4 report: {training_summary_path}")
    variant, report = candidate_reports[-1]
    checkpoint_value = report.get("checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise ValueError("training summary has no R4 checkpoint path")
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_absolute():
        checkpoint = training_summary_path.parent / checkpoint
    checkpoint = checkpoint.resolve(strict=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != "smagm-brats21-r4-checkpoint-v1":
        raise ValueError("patient checkpoint is not an R4 checkpoint")
    if payload.get("target_payload_not_in_checkpoint") is not True or payload.get("global_model_eligible") is not True:
        raise ValueError("patient checkpoint is not eligible for global promotion")
    if payload.get("training_updates_applied") is not True:
        raise ValueError("validation-only checkpoint cannot be promoted into the global training state")
    payload_split_hash = payload.get("cohort_split_hash", payload.get("split_hash"))
    summary_split_hash = summary.get("cohort_split_hash", summary.get("split_hash"))
    if payload_split_hash != split_hash or summary_split_hash != split_hash:
        raise ValueError("patient checkpoint split binding does not match the product run")
    if not isinstance(summary.get("manifest_hash"), str) or not isinstance(summary.get("assignment_hash"), str):
        raise ValueError("patient training summary must bind its manifest and assignment before promotion")
    if summary.get("training_updates_applied") is not True:
        raise ValueError("validation-only training summary cannot be promoted into the global training state")
    model_binding_hash = payload.get("model_binding_hash")
    if not isinstance(model_binding_hash, str) or len(model_binding_hash) != 64:
        raise ValueError("patient checkpoint has no valid global model binding")
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, dict):
        raise ValueError("patient checkpoint has no optimizer state for cohort resume")
    required = ("encoder", "gaussian_head", "field")
    if any(key not in payload for key in required):
        raise ValueError("patient checkpoint is missing a shared model state")
    if global_update_index <= 0:
        raise ValueError("global update index must be positive")
    global_payload = {
        "schema": "smagm-brats21-global-training-checkpoint-v1",
        "cohort_hash": cohort_hash,
        "split_hash": split_hash,
        "cohort_split_hash": split_hash,
        "global_update_index": int(global_update_index),
        "source_patient_pseudonym": source_patient_pseudonym,
        "source_variant": variant,
        "source_checkpoint_sha256": _file_hash(checkpoint),
        "source_config_hash": payload.get("config_hash"),
        "source_manifest_hash": summary["manifest_hash"],
        "source_assignment_hash": summary["assignment_hash"],
        "model_binding_hash": model_binding_hash,
        "encoder": payload["encoder"],
        "gaussian_head": payload["gaussian_head"],
        "field": payload["field"],
        "optimizer": optimizer,
        "target_payload_not_in_checkpoint": True,
        "patient_state_not_in_checkpoint": True,
    }
    _atomic_torch(destination, global_payload)
    return {
        "path": str(destination),
        "global_update_index": int(global_update_index),
        "source_patient_pseudonym": source_patient_pseudonym,
        "model_binding_hash": model_binding_hash,
        "source_checkpoint_sha256": global_payload["source_checkpoint_sha256"],
    }


def _promotion_journal(
    *,
    result: dict[str, Any],
    completion_key: str,
    summary_path: Path,
    source_patient_pseudonym: str,
    propagation_variant: str,
    global_update_index: int,
    previous_global_update_index: int,
    previous_global_checkpoint_sha256: str | None,
    cohort_hash: str,
    split_hash: str,
    config_hash: str,
) -> dict[str, Any]:
    """Create the pre-promotion journal record from a target-free checkpoint."""

    summary = _read_json(summary_path)
    if summary.get("training_updates_applied") is not True:
        raise ValueError("only a patient training result with optimizer updates can be promoted")
    variant = f"e2_r4_{propagation_variant}"
    report = summary.get(variant)
    if not isinstance(report, dict) or report.get("training_updates_applied") is not True:
        raise ValueError("promotion journal requires a training-updated E2/R4 report")
    checkpoint_value = report.get("checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise ValueError("promotion journal requires the patient checkpoint path")
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_absolute():
        checkpoint = summary_path.parent / checkpoint
    payload = torch.load(checkpoint.resolve(strict=True), map_location="cpu", weights_only=True)
    model_binding_hash = payload.get("model_binding_hash") if isinstance(payload, dict) else None
    if not isinstance(model_binding_hash, str):
        raise ValueError("promotion journal requires a valid patient model binding hash")
    if not isinstance(summary.get("manifest_hash"), str) or not isinstance(summary.get("assignment_hash"), str):
        raise ValueError("promotion journal requires manifest and assignment bindings")
    return {
        "schema": "smagm-brats21-promotion-journal-v1",
        "completion_key": completion_key,
        "training_summary": str(summary_path.resolve()),
        "result": result,
        "patient_pseudonym": source_patient_pseudonym,
        "source_patient_pseudonym": source_patient_pseudonym,
        "propagation_variant": propagation_variant,
        "global_update_index": global_update_index,
        "previous_global_update_index": previous_global_update_index,
        "previous_global_checkpoint_sha256": previous_global_checkpoint_sha256,
        "cohort_hash": cohort_hash,
        "split_hash": split_hash,
        "config_hash": config_hash,
        "source_manifest_hash": summary["manifest_hash"],
        "source_assignment_hash": summary["assignment_hash"],
        "model_binding_hash": model_binding_hash,
    }


def _global_checkpoint_update_index(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != "smagm-brats21-global-training-checkpoint-v1":
        raise ValueError("global model checkpoint schema is invalid")
    value = payload.get("global_update_index")
    if not isinstance(value, int) or value < 0:
        raise ValueError("global model checkpoint update index is invalid")
    return value


def _global_checkpoint_source_patient(path: Path) -> str | None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != "smagm-brats21-global-training-checkpoint-v1":
        raise ValueError("global model checkpoint schema is invalid")
    source = payload.get("source_patient_pseudonym")
    if source is not None and (not isinstance(source, str) or not source):
        raise ValueError("global model checkpoint source patient is invalid")
    return source


def _reconcile_global_promotion_journal(
    state: dict[str, Any],
    global_checkpoint: Path,
    *,
    cohort_hash: str,
    split_hash: str,
    config_hash: str,
) -> tuple[int, str | None]:
    """Reconcile exactly one interrupted patient-boundary promotion.

    The journal is written before the atomic checkpoint promotion.  Recovery
    accepts only the exact next update whose source patient, manifest,
    assignment, split, and model binding match the journal.  A lone or
    arbitrarily advanced checkpoint is rejected rather than guessed at.
    """

    current_index = int(state.get("global_update_count", 0))
    if current_index < 0:
        raise ValueError("run state global update count is invalid")
    pending = state.get("pending_promotion")
    if pending is not None and not isinstance(pending, dict):
        raise ValueError("run state pending promotion journal is malformed")
    if not global_checkpoint.exists():
        if current_index != 0:
            raise ValueError("run state claims global updates but the global model checkpoint is absent")
        if pending is not None:
            if (
                pending.get("global_update_index") != 1
                or pending.get("previous_global_update_index") != 0
                or pending.get("previous_global_checkpoint_sha256") is not None
                or pending.get("cohort_hash") != cohort_hash
                or pending.get("split_hash") != split_hash
                or pending.get("config_hash") != config_hash
            ):
                raise ValueError("pending promotion journal does not describe the initial global update")
        return current_index, None

    payload = torch.load(global_checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != "smagm-brats21-global-training-checkpoint-v1":
        raise ValueError("global model checkpoint schema is invalid")
    checkpoint_index = payload.get("global_update_index")
    if not isinstance(checkpoint_index, int) or checkpoint_index < 0:
        raise ValueError("global model checkpoint update index is invalid")
    checkpoint_hash = _file_hash(global_checkpoint)
    claimed_hash = state.get("global_checkpoint_sha256")
    if claimed_hash not in (None, checkpoint_hash) and checkpoint_index == current_index:
        raise ValueError("run state global checkpoint hash does not match the checkpoint file")
    if checkpoint_index == current_index:
        if pending is not None:
            raise ValueError("run state retains a pending promotion at an already committed checkpoint index")
        return current_index, checkpoint_hash
    if checkpoint_index != current_index + 1 or pending is None:
        raise ValueError("global checkpoint is not the exact next journaled patient update")
    expected = {
        "global_update_index": checkpoint_index,
        "previous_global_update_index": current_index,
        "previous_global_checkpoint_sha256": claimed_hash,
        "cohort_hash": cohort_hash,
        "split_hash": split_hash,
        "config_hash": config_hash,
    }
    if any(pending.get(key) != value for key, value in expected.items()):
        raise ValueError("pending promotion journal does not match the interrupted global update")
    if payload.get("cohort_hash") != cohort_hash or payload.get("split_hash") != split_hash:
        raise ValueError("recovered global checkpoint cohort binding is invalid")
    for key in ("source_patient_pseudonym", "source_manifest_hash", "source_assignment_hash", "model_binding_hash"):
        if not isinstance(pending.get(key), str) or payload.get(key) != pending.get(key):
            raise ValueError(f"recovered global checkpoint {key} binding is invalid")
    if payload.get("target_payload_not_in_checkpoint") is not True or payload.get("patient_state_not_in_checkpoint") is not True:
        raise ValueError("recovered global checkpoint contains forbidden patient or target state")
    completion_key = pending.get("completion_key")
    result = pending.get("result")
    summary_path = pending.get("training_summary")
    if not isinstance(completion_key, str) or not completion_key or not isinstance(result, dict) or not isinstance(summary_path, str):
        raise ValueError("pending promotion journal has no resumable patient result")
    key_split, separator, key_patient = completion_key.partition(":")
    if not separator or key_patient != pending.get("patient_pseudonym") or result.get("patient_pseudonym") not in (None, key_patient):
        raise ValueError("pending promotion journal completion key is not bound to its source patient")
    summary_file = Path(summary_path).resolve(strict=True)
    if not _training_summary_complete(
        summary_file,
        expected_propagation_variant=str(pending.get("propagation_variant", "p1")),
        require_training_updates=True,
    ):
        raise ValueError("pending promotion journal references an incomplete training summary")
    summary = _read_json(summary_file)
    if summary.get("manifest_hash") != pending.get("source_manifest_hash") or summary.get("assignment_hash") != pending.get("source_assignment_hash"):
        raise ValueError("pending promotion summary binding is inconsistent")
    state.setdefault("completed", {})[completion_key] = result
    state["global_update_count"] = checkpoint_index
    state["global_checkpoint_sha256"] = checkpoint_hash
    state.pop("pending_promotion", None)
    return checkpoint_index, checkpoint_hash


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON config must contain an object: {path}")
    return value


def _resolve_config_path(base: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve(strict=True)


def _resolve_dataset_root(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = _ROOT / candidate
    return candidate.resolve(strict=True)


def _load_product_config(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve(strict=True)
    product = _read_json(path)
    if product.get("schema") != _PRODUCT_SCHEMA:
        raise ValueError(f"product config must use schema {_PRODUCT_SCHEMA}")
    if product.get("stage") != "full":
        raise ValueError("the product controller has one launch mode: stage=full")
    experiment_name = product.get("experiment_name")
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("product config must declare a non-empty experiment_name")
    disk_policy = str(product.get("disk_policy", "advisory")).strip().lower()
    if disk_policy not in ("advisory", "enforced"):
        raise ValueError("product disk_policy must be advisory or enforced")
    if "readiness_prerequisites" in product:
        raise ValueError("smoke/pilot readiness prerequisites are retired; use the full config directly")
    data_path = _resolve_config_path(path.parent, str(product["data_config"]))
    train_path = _resolve_config_path(path.parent, str(product["training_config"]))
    reconstruction_path = _resolve_config_path(path.parent, str(product["reconstruction_config"]))
    evaluation_path = _resolve_config_path(path.parent, str(product["evaluation_config"]))
    data = _read_json(data_path)
    training = _read_json(train_path)
    reconstruction = _read_json(reconstruction_path)
    evaluation = _read_json(evaluation_path)
    if data.get("schema") != "smagm-brats21-data-v2":
        raise ValueError("product data config must use smagm-brats21-data-v2")
    if training.get("schema") != "smagm-brats21-product-train-v1":
        raise ValueError("product training config must use smagm-brats21-product-train-v1")
    if reconstruction.get("schema") != "smagm-brats21-product-reconstruction-v1":
        raise ValueError("product reconstruction config has an unsupported schema")
    if evaluation.get("schema") != "smagm-brats21-product-evaluation-v1":
        raise ValueError("product evaluation config has an unsupported schema")
    aggregation = evaluation.get("aggregation")
    if not isinstance(aggregation, dict):
        raise ValueError("product evaluation config must declare aggregate statistics policy")
    if aggregation.get("scope") != "unweighted_patient_macro":
        raise ValueError("product aggregate statistics must use unweighted patient macro-statistics")
    if aggregation.get("missing_policy") != "omit_non_finite_with_count":
        raise ValueError("product aggregate statistics must declare explicit non-finite missing handling")
    if int(aggregation.get("bootstrap_samples", 0)) <= 0:
        raise ValueError("product aggregate statistics must declare positive bootstrap_samples")
    confidence = float(aggregation.get("confidence", 0.0))
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("product aggregate statistics confidence must be finite and in (0,1)")
    int(aggregation.get("bootstrap_seed", 0))
    product_validation = product.get("validation")
    if not isinstance(product_validation, dict):
        raise ValueError("product config must declare a validation policy")
    validation_cadence = str(product_validation.get("cadence", ""))
    if validation_cadence not in ("disabled", "post_training_patient_disjoint_sweep"):
        raise ValueError("product validation cadence must be disabled or post_training_patient_disjoint_sweep")
    validation_enabled = bool(product_validation.get("enabled", False))
    if validation_enabled != (validation_cadence == "post_training_patient_disjoint_sweep"):
        raise ValueError("product validation enabled flag must agree with its declared cadence")
    if str(product_validation.get("selection_policy", "")) != "fixed_geometry_only_no_checkpoint_selection":
        raise ValueError("product validation must declare fixed geometry-only selection without checkpoint selection")
    for name in ("cohort_hash", "split_hash"):
        declared = product.get(name)
        configured = data.get(name)
        if not isinstance(declared, str) or len(declared) != 64 or any(character not in "0123456789abcdef" for character in declared):
            raise ValueError(f"product config must declare a valid {name}")
        if declared != configured:
            raise ValueError(f"product {name} does not match the referenced data config")
    output_paths = dict(product.get("output_paths", {}))
    required_output_paths = (
        "default_root",
        "run_directory_policy",
        "log_file",
        "state_file",
        "completion_marker",
        "global_checkpoint",
        "patient_metrics",
        "aggregate_metrics",
        "refuse_overwrite_success",
    )
    if any(key not in output_paths for key in required_output_paths):
        raise ValueError(
            "product config output_paths must declare "
            + ", ".join(required_output_paths)
        )
    if output_paths["run_directory_policy"] not in ("fixed", "unique-utc-suffix"):
        raise ValueError("product output_paths.run_directory_policy is unsupported")
    if not all(isinstance(output_paths[key], str) and output_paths[key] for key in (
        "default_root", "log_file", "state_file", "completion_marker", "global_checkpoint",
        "patient_metrics", "aggregate_metrics",
    )):
        raise ValueError("product output path names must be non-empty strings")
    for key in ("default_root", "log_file", "state_file", "completion_marker", "global_checkpoint", "patient_metrics", "aggregate_metrics"):
        output_path = Path(str(output_paths[key]))
        if output_path.is_absolute() or ".." in output_path.parts:
            raise ValueError(f"product output path {key} must remain relative to its run directory")
    declared_files = tuple(str(output_paths[key]) for key in (
        "log_file", "state_file", "completion_marker", "global_checkpoint", "patient_metrics", "aggregate_metrics",
    ))
    if len(set(declared_files)) != len(declared_files):
        raise ValueError("product output files must have distinct relative paths")
    if output_paths["refuse_overwrite_success"] is not True:
        raise ValueError("product output paths must refuse overwrite of successful runs")
    runner = dict(reconstruction.get("runner", {}))
    if runner.get("target_grid", "held_out_target_plane") not in ("held_out_target_plane", "full_source_grid"):
        raise ValueError("product reconstruction runner target_grid must be held_out_target_plane or full_source_grid")
    for name in ("depth_chunk_size", "full_grid_depth_chunk_size"):
        if int(runner.get(name, 1)) <= 0:
            raise ValueError(f"product reconstruction runner {name} must be positive")
    if not isinstance(runner.get("write_nifti", False), bool):
        raise ValueError("product reconstruction runner write_nifti must be boolean")
    chunking = dict(reconstruction.get("chunking", {}))
    if int(chunking.get("full_grid_depth_chunk_size", runner.get("full_grid_depth_chunk_size", 1))) <= 0:
        raise ValueError("product reconstruction full-grid depth chunk size must be positive")
    sampling = BraTS21SamplingConfig(**dict(data["sampling"]))
    NormalizationConfig(**dict(data["normalization"]))
    backend = dict(training.get("training_backend", {}))
    expected_backend = {
        "name": "native", "accelerator": "gpu", "devices": 1, "strategy": "single",
        "precision": "32-true", "accumulate_grad_batches": 1, "activation_checkpointing": False,
    }
    if backend != expected_backend:
        raise ValueError(f"product backend must resolve exactly to native single-GPU float32: {expected_backend}")
    if training.get("device") != "cuda" or training.get("precision") != "float32":
        raise ValueError("product training must request CUDA float32")
    if bool(training.get("training", {}).get("allow_cpu_fallback", False)):
        raise ValueError("product configs must never enable CPU fallback")
    if training.get("encoder_variant") != "e2" or training.get("representation_variant") != "anchor_field":
        raise ValueError("the product controller is locked to E2 + R4")
    if training.get("training", {}).get("gaussian_head_input_adapter") != "anchor_evidence_prefix":
        raise ValueError("the product controller requires the declared anchor_evidence_prefix Gaussian-head adapter")
    propagation_policy = dict(training.get("propagation", {}))
    if float(propagation_policy.get("minimum_evidence_gain", 0.0)) <= 0.0:
        raise ValueError("product propagation must reject zero meaningful evidence gain with a positive threshold")
    if not 0.0 < float(propagation_policy.get("minimum_cross_modality_agreement", 0.0)) <= 1.0:
        raise ValueError("product propagation must declare a positive cross-modality agreement threshold")
    if training.get("t4_routing") is not False:
        raise ValueError("T4 routing must remain explicitly disabled")
    config = {
        "product_path": str(path),
        "product": product,
        "data": data,
        "training": training,
        "reconstruction": reconstruction,
        "evaluation": evaluation,
        "data_path": str(data_path),
        "training_path": str(train_path),
        "reconstruction_path": str(reconstruction_path),
        "evaluation_path": str(evaluation_path),
        "sampling_config_hash": sampling.config_hash,
        "resolved_config_hash": _hash({"product": product, "data": data, "training": training, "reconstruction": reconstruction, "evaluation": evaluation}),
    }
    return config, str(config["resolved_config_hash"])


def _runtime_config(config: dict[str, Any], *, propagation_variant: str, steps: int, wandb_mode: str, split_name: str) -> dict[str, Any]:
    data = config["data"]
    product = config["product"]
    training = json.loads(json.dumps(config["training"]))
    training["schema"] = "smagm-brats21-real-smoke-v1"
    training["execution_mode"] = "product"
    training["experiment_name"] = str(product["experiment_name"])
    training["source_kind"] = "SIMULATED_SPARSE_ACQUISITION"
    training["claim_scope"] = "software-and-execution-evidence-only"
    training["cohort_hash"] = config.get("cohort_hash")
    training["split_hash"] = config.get("split_hash")
    training["cohort_split_hash"] = config.get("split_hash")
    training["episode_split"] = split_name
    training["dataset_root"] = str(data["dataset_root"])
    training["inventory_report"] = str(data.get("inventory_report", ""))
    training["modalities"] = list(data["modalities"])
    training["target_modality"] = str(config["product"].get("target_modality", "flair"))
    training["sampling"] = dict(data["sampling"])
    training["normalization"] = dict(data["normalization"])
    training["propagation_variant"] = propagation_variant
    training["propagation"] = dict(training["propagation"])
    training["propagation"]["variant"] = propagation_variant
    if propagation_variant == "p0":
        training["propagation"]["rounds"] = 0
    training["training"]["steps"] = steps
    training["training"]["allow_cpu_fallback"] = False
    training["inplane_stride_vu"] = list(data["inplane_stride_vu"])
    training["renderer"] = dict(config["reconstruction"].get("renderer", training.get("renderer", {})))
    training["reconstruction"] = dict(config["reconstruction"].get("runner", training.get("reconstruction", {})))
    training["reconstruction"]["full_grid_depth_chunk_size"] = int(
        config["reconstruction"].get("chunking", {}).get(
            "full_grid_depth_chunk_size",
            training["reconstruction"].get("depth_chunk_size", 1),
        )
    )
    training["evaluation"] = json.loads(json.dumps(config["evaluation"]))
    training["output_paths"] = json.loads(json.dumps(product["output_paths"]))
    training["wandb"] = dict(training.get("wandb", {}))
    training["wandb"]["mode"] = wandb_mode
    training["wandb"]["project"] = "smagm-brats21"
    training["wandb"]["group"] = str(product.get("wandb_group", training["wandb"].get("group", "structure-propagation")))
    return training


def _split_for(patient_id: str, fractions: dict[str, float]) -> str:
    pseudonym = hashlib.sha256(f"smagm-brats21-patient-v1:{patient_id}".encode()).hexdigest()
    value = int.from_bytes(hashlib.sha256(pseudonym.encode()).digest()[:8], "big") / float(2**64)
    cumulative = 0.0
    for name in ("train", "validation", "t1_lesion_validation", "t5_final_audit"):
        cumulative += float(fractions.get(name, 0.0))
        if value < cumulative:
            return name
    return "t5_final_audit"


def _cohort_and_split_hashes(config: dict[str, Any]) -> tuple[str, str]:
    data = config["data"]
    inventory_path = Path(str(data.get("inventory_report", "experiments/reports/brats21_dataset_inventory.json")))
    if not inventory_path.is_absolute():
        inventory_path = _ROOT / inventory_path
    inventory = _read_json(inventory_path.resolve(strict=True))
    expected_cohort = str(data.get("cohort_hash") or "")
    actual_cohort = str(inventory.get("cohort_hash") or "")
    if not expected_cohort or expected_cohort != actual_cohort:
        raise ValueError("data config cohort_hash does not match the completed BraTS21 inventory")
    root = _resolve_dataset_root(str(data["dataset_root"]))
    fractions = {str(key): float(value) for key, value in dict(data.get("split_fractions", {})).items()}
    assignments: list[dict[str, str]] = []
    for directory in sorted(item for item in root.iterdir() if item.is_dir()):
        if BRATS21_PATIENT_PATTERN.fullmatch(directory.name) is None:
            continue
        try:
            patient = discover_patient(directory)
            result = validate_patient(patient, require_segmentation=bool(data.get("require_segmentation", True)), include_data=False, include_source_hash=False)
            if result.valid:
                pseudonym = hashlib.sha256(f"smagm-brats21-patient-v1:{patient.patient_id}".encode()).hexdigest()
                assignments.append({"patient_pseudonym": pseudonym, "split": _split_for(patient.patient_id, fractions)})
        except (BraTS21ValidationError, OSError, ValueError):
            continue
    split_payload = {"schema": "smagm-brats21-split-v2", "seed": int(data.get("split_seed", 0)), "fractions": fractions, "assignments": sorted(assignments, key=lambda item: item["patient_pseudonym"])}
    split_hash = _hash(split_payload)
    expected_split = data.get("split_hash")
    if expected_split is not None and str(expected_split) != split_hash:
        raise ValueError("data config split_hash does not match deterministic patient split")
    return actual_cohort, split_hash


def _source_hashes_for_patient(config: dict[str, Any], patient_id: str) -> dict[str, str]:
    """Read precomputed source hashes without opening the patient's NIfTI payloads."""

    inventory_path = Path(str(config["data"].get("inventory_report", "experiments/reports/brats21_dataset_inventory.json")))
    if not inventory_path.is_absolute():
        inventory_path = _ROOT / inventory_path
    inventory = _read_json(inventory_path.resolve(strict=True))
    if inventory.get("schema") != "smagm-brats21-dataset-inventory-v2":
        raise ValueError("product source hashes must come from the v2 BraTS21 inventory")
    expected_cohort = str(config["data"].get("cohort_hash") or "")
    if expected_cohort and str(inventory.get("cohort_hash") or "") != expected_cohort:
        raise ValueError("product source-hash inventory cohort does not match the product config")
    validation_scope = inventory.get("validation_scope")
    all_patient_scope = str(validation_scope.get("all_patients", "")) if isinstance(validation_scope, dict) else ""
    scope_text = all_patient_scope.lower()
    if "full finite-data validation" not in scope_text or "source hashes" not in scope_text:
        raise ValueError("product source-hash inventory is not bound to full finite-value validation")
    for record in inventory.get("valid_patients", []):
        if str(record.get("patient_id")) != patient_id:
            continue
        summaries = record.get("summaries", [])
        hashes = {
            str(summary["suffix"]): str(summary["source_hash"])
            for summary in summaries
            if isinstance(summary, dict) and summary.get("source_hash")
        }
        if not hashes:
            break
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in hashes.values()
        ):
            raise ValueError(f"completed inventory has malformed source hashes for patient {patient_id!r}")
        return hashes
    raise ValueError(f"completed inventory has no source hashes for patient {patient_id!r}")


def _patient_ids(config: dict[str, Any], stage: str, explicit_patient: str | None, limit: int | None, *, split_name: str | None = None) -> tuple[str, ...]:
    root = _resolve_dataset_root(str(config["data"]["dataset_root"]))
    requested_split = str(split_name or config["product"].get("training_split", "train"))
    fractions = {str(k): float(v) for k, v in dict(config["data"].get("split_fractions", {})).items()}
    candidates: list[str] = []
    for directory in sorted(item for item in root.iterdir() if item.is_dir()):
        if BRATS21_PATIENT_PATTERN.fullmatch(directory.name) is None:
            continue
        try:
            patient = discover_patient(directory)
            result = validate_patient(patient, require_segmentation=bool(config["data"].get("require_segmentation", True)), include_data=False, include_source_hash=False)
            if result.valid and _split_for(patient.patient_id, fractions) == requested_split:
                candidates.append(patient.patient_id)
        except (BraTS21ValidationError, OSError, ValueError):
            continue
    if explicit_patient is not None:
        if explicit_patient not in candidates:
            raise ValueError(f"requested patient is not a valid member of the {requested_split} split")
        candidates = [explicit_patient]
    if limit is not None:
        candidates = candidates[:limit]
    if not candidates:
        raise ValueError("no valid patients remain for the requested product stage")
    return tuple(candidates)


def _cuda_preflight() -> dict[str, Any]:
    report: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "not-set"),
    }
    if not report["cuda_available"] or report["device_count"] < 1:
        raise RuntimeError("CUDA is unavailable; the product controller refuses CPU fallback")
    report["devices"] = [
        {"index": index, "name": torch.cuda.get_device_name(index), "total_memory": torch.cuda.get_device_properties(index).total_memory}
        for index in range(int(report["device_count"]))
    ]
    return report


def _disk_report(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path.parent)
    return {"path": str(path), "free_bytes": usage.free, "total_bytes": usage.total, "free_gib": usage.free / 2**30}


def _quarantine_partial_directory(path: Path) -> Path:
    """Preserve an incomplete patient derivative before rebuilding it."""

    if not path.is_dir():
        raise ValueError(f"partial patient derivative is not a directory: {path}")
    for attempt in range(100):
        candidate = path.with_name(f"{path.name}.incomplete-{time.time_ns()}-{attempt}")
        if candidate.exists():
            continue
        path.replace(candidate)
        return candidate
    raise FileExistsError(f"could not quarantine partial patient derivative: {path}")


def _quarantine_partial_file(path: Path) -> Path:
    """Preserve one invalid terminal marker without discarding resumable state."""

    if not path.is_file():
        raise ValueError(f"partial patient derivative is not a file: {path}")
    for attempt in range(100):
        candidate = path.with_name(f"{path.name}.incomplete-{time.time_ns()}-{attempt}")
        if candidate.exists():
            continue
        path.replace(candidate)
        return candidate
    raise FileExistsError(f"could not quarantine partial patient derivative: {path}")


def _resolve_recorded_path(value: object, *, base: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _training_summary_complete(
    path: Path,
    *,
    expected_propagation_variant: str | None = None,
    require_training_updates: bool | None = None,
) -> bool:
    """Validate the immutable patient result before marking it complete."""

    try:
        summary = _read_json(path)
        if summary.get("schema") != "smagm-brats21-real-smoke-v1":
            return False
        if summary.get("scientific_pass_recorded") is not False:
            return False
        if summary.get("target_reveal_barrier_verified") is not True:
            return False
        if expected_propagation_variant is not None:
            if expected_propagation_variant not in ("p0", "p1"):
                return False
            variant = f"e2_r4_{expected_propagation_variant}"
            report = summary.get(variant)
            if not isinstance(report, dict):
                return False
        else:
            reports = [
                (str(key), value)
                for key, value in summary.items()
                if str(key).startswith("e2_r4_") and isinstance(value, dict)
            ]
            if not reports:
                return False
            variant, report = reports[-1]
        if require_training_updates is not None and report.get("training_updates_applied") is not require_training_updates:
            return False
        for key in ("checkpoint", "patient_state"):
            artifact = _resolve_recorded_path(report.get(key), base=path.parent)
            if artifact is None or not artifact.is_file():
                return False
        prediction_package = _resolve_recorded_path(report.get("prediction_package"), base=path.parent)
        if prediction_package is None or not prediction_package.is_dir() or not (prediction_package / "package.json").is_file():
            return False
        from ..evaluation.audit import open_serialized_predictions

        serialized = open_serialized_predictions(prediction_package)
        checkpoint_path = _resolve_recorded_path(report.get("checkpoint"), base=path.parent)
        assert checkpoint_path is not None
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("schema") != "smagm-brats21-r4-checkpoint-v1"
            or checkpoint.get("target_payload_not_in_checkpoint") is not True
            or checkpoint.get("global_model_eligible") is not True
        ):
            return False
        evaluations = summary.get("evaluations")
        audits = summary.get("audits")
        if not isinstance(evaluations, dict) or not isinstance(audits, dict):
            return False
        for name in ("r0", variant):
            evaluation = evaluations.get(name)
            audit = audits.get(name)
            if not isinstance(evaluation, dict) or not isinstance(audit, dict):
                return False
            evaluation_path = _resolve_recorded_path(evaluation.get("path"), base=path.parent)
            audit_path = _resolve_recorded_path(audit.get("path"), base=path.parent)
            if evaluation_path is None or not evaluation_path.is_file() or audit_path is None or not audit_path.is_file():
                return False
            evaluation_payload = _read_json(evaluation_path)
            audit_payload = _read_json(audit_path)
            metrics = evaluation_payload.get("metrics")
            if not isinstance(metrics, list) or not metrics or audit_payload.get("status") != serialized.package.execution_status:
                return False
    except (OSError, EOFError, KeyError, TypeError, ValueError, RuntimeError, pickle.UnpicklingError):
        return False
    return True


def _prepared_bundle_complete(
    path: Path,
    *,
    expected_patient_id: str | None = None,
    expected_split: str | None = None,
    expected_target_modality: str | None = None,
    expected_sampling_protocol_hash: str | None = None,
    expected_source_hashes: dict[str, str] | None = None,
) -> bool:
    """Check a prepared derivative without opening hidden source payloads.

    ``expected_source_hashes`` come from the completed dataset inventory.  The
    product path deliberately does not re-hash source NIfTI files here: doing
    so would read hidden target/evaluator voxels before the receipt barrier.
    Context payload files are independently hashed because they are the only
    source-derived bytes materialized for the training path.
    """

    required = (
        "manifest.json", "evaluator_manifest.json", "assignment.json",
        "split.json", "prepared.json", "hashes.json",
    )
    if not path.is_dir() or any(not (path / name).is_file() for name in required):
        return False
    try:
        prepared_meta = _read_json(path / "prepared.json")
        if prepared_meta.get("schema") != "smagm-brats21-prepared-product-v1":
            return False
        hashes = _read_json(path / "hashes.json")
        if prepared_meta.get("hashes") != hashes:
            return False
        from ..data.brats21_prepare import load_prepared_bundle

        bundle = load_prepared_bundle(path)
        if expected_patient_id is not None and bundle.patient_id != expected_patient_id:
            return False
        if expected_split is not None and bundle.manifest_json.get("split") != expected_split:
            return False
        if expected_target_modality is not None:
            target_reference = bundle.evaluator_json.get("target_reference")
            if not isinstance(target_reference, dict) or target_reference.get("modality_id") != expected_target_modality:
                return False
        if expected_sampling_protocol_hash is not None and prepared_meta.get("sampling_protocol_hash") != expected_sampling_protocol_hash:
            return False
        if prepared_meta.get("manifest_hash") != bundle.manifest.manifest_hash or prepared_meta.get("assignment_hash") != bundle.assignment.assignment_hash:
            return False
        claimed_source_hashes = bundle.manifest_json.get("source_nifti_hashes")
        if expected_source_hashes is not None:
            if not isinstance(claimed_source_hashes, dict) or any(
                claimed_source_hashes.get(key) != value for key, value in expected_source_hashes.items()
            ):
                return False
        if hashes.get("source_nifti") != claimed_source_hashes:
            return False
        if not bundle.target_payload_deferred:
            return False
        if bundle.evaluator_json.get("segmentation_reference") is not None and not bundle.segmentation_payload_deferred:
            return False
        for observation_id in bundle.assignment.context_ids:
            entry = bundle.manifest.metadata(observation_id)
            payload_path = path / entry.relative_path
            if not payload_path.is_file() or _file_hash(payload_path) != bundle.manifest._expected_sha256(observation_id):
                return False
        if bundle.target_payload_deferred:
            if bundle.target_payload_path.exists():
                return False
        elif not bundle.target_payload_path.is_file():
            return False
        if bundle.segmentation_payload_deferred:
            if bundle.segmentation_payload_path is None or bundle.segmentation_payload_path.exists():
                return False
        elif bundle.segmentation_payload_path is not None and not bundle.segmentation_payload_path.is_file():
            return False
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return True


def _runtime_config_file(run_root: Path, config: dict[str, Any], *, propagation_variant: str, steps: int, wandb_mode: str, split_name: str) -> Path:
    path = run_root / "resolved_runtime_config.json"
    _atomic_json(path, _runtime_config(config, propagation_variant=propagation_variant, steps=steps, wandb_mode=wandb_mode, split_name=split_name))
    return path


def _expected_sampling_protocol_hash(
    *, config: dict[str, Any], patient: Any, split_name: str, seed: int,
) -> str:
    validation = validate_patient(
        patient,
        require_segmentation=bool(config["data"].get("require_segmentation", True)),
        include_data=False,
        include_source_hash=False,
    )
    if not validation.valid:
        raise BraTS21ValidationError(f"{patient.patient_id}: {validation.error}")
    summaries = {item.suffix: item for item in validation.summaries}
    plan = build_sampling_plan(
        summaries,
        episode_id=f"brats21-product:{patient.patient_id}:{split_name}:{seed:08d}",
        target_modality=str(config["product"].get("target_modality", "flair")),
        split=split_name,
        seed=seed,
        inplane_stride_vu=tuple(int(value) for value in config["data"].get("inplane_stride_vu", [4, 4])),
        config=BraTS21SamplingConfig(**dict(config["data"]["sampling"])),
    )
    return plan.protocol_hash


def _run_one_patient(
    *, config: dict[str, Any], patient_id: str, run_root: Path, propagation_variant: str, steps: int, wandb_mode: str,
    evaluation_path: Path, seed: int, split_name: str, initial_global_checkpoint: Path | None = None,
    validation_only: bool = False,
) -> dict[str, Any]:
    from .brats21_smoke import run as smoke_run
    from ..data.brats21_prepare import prepare_brats21_product_patient

    pseudonym = hashlib.sha256(f"smagm-brats21-patient-v1:{patient_id}".encode()).hexdigest()[:16]
    train_split = str(config["product"].get("training_split", "train"))
    split_root = run_root if split_name == train_split else run_root / split_name
    patient_root = split_root / "patients" / f"patient-{pseudonym}"
    patient_root.mkdir(parents=True, exist_ok=True)
    summary = patient_root / "summary.json"
    training_summary = patient_root / "training" / "summary.json"
    completion_recovery_path: str | None = None
    training_recovery_path: str | None = None
    if summary.exists():
        marker_valid = False
        recorded_training_summary: Path | None = None
        try:
            marker = _read_json(summary)
            recorded_training_summary = _resolve_recorded_path(marker.get("training_summary"), base=summary.parent)
            marker_valid = (
                marker.get("schema") == "smagm-brats21-product-patient-complete-v1"
                and marker.get("split") == split_name
                and marker.get("patient_pseudonym") == f"patient-{pseudonym}"
                and marker.get("scientific_pass_recorded") is False
                and recorded_training_summary is not None
                and _training_summary_complete(
                    recorded_training_summary,
                    expected_propagation_variant=propagation_variant,
                    require_training_updates=not validation_only,
                )
            )
            if marker_valid:
                prepared_meta_path = patient_root / "prepared" / "prepared.json"
                summary_meta = _read_json(recorded_training_summary)
                prepared_meta = _read_json(prepared_meta_path)
                expected_summary_pseudonym = "patient-" + hashlib.sha256(
                    f"{patient_id}:{summary_meta.get('manifest_hash')}".encode("utf-8")
                ).hexdigest()[:16]
                marker_valid = (
                    summary_meta.get("patient_pseudonymous_id") == expected_summary_pseudonym
                    and summary_meta.get("manifest_hash") == prepared_meta.get("manifest_hash")
                    and summary_meta.get("assignment_hash") == prepared_meta.get("assignment_hash")
                    and summary_meta.get("split_hash") == prepared_meta.get("split_hash")
                    and summary_meta.get("cohort_split_hash") == config.get("split_hash")
                )
        except (OSError, TypeError, ValueError):
            marker_valid = False
        if marker_valid:
            assert recorded_training_summary is not None
            return {
                "patient_pseudonym": f"patient-{pseudonym}",
                "status": "already_complete",
                "summary": str(recorded_training_summary),
            }
        if not summary.is_file():
            raise RuntimeError(f"patient completion marker is not a regular file: {summary}")
        quarantined = _quarantine_partial_file(summary)
        completion_recovery_path = str(quarantined.relative_to(patient_root))
    if training_summary.exists():
        # A process may have completed the immutable patient result just
        # before the cohort-level state update. Reconstitute the small
        # patient completion marker without rerunning or overwriting it.
        if _training_summary_complete(
            training_summary,
            expected_propagation_variant=propagation_variant,
            require_training_updates=not validation_only,
        ):
            prepared_meta = _read_json(patient_root / "prepared" / "prepared.json")
            summary_meta = _read_json(training_summary)
            expected_summary_pseudonym = "patient-" + hashlib.sha256(
                f"{patient_id}:{summary_meta.get('manifest_hash')}".encode("utf-8")
            ).hexdigest()[:16]
            if (
                summary_meta.get("patient_pseudonymous_id") != expected_summary_pseudonym
                or summary_meta.get("manifest_hash") != prepared_meta.get("manifest_hash")
                or summary_meta.get("assignment_hash") != prepared_meta.get("assignment_hash")
                or summary_meta.get("split_hash") != prepared_meta.get("split_hash")
                or summary_meta.get("cohort_split_hash") != config.get("split_hash")
            ):
                raise ValueError("completed patient summary does not bind the expected patient, prepared manifest, assignment, or cohort split")
            _atomic_json(summary, {
                "schema": "smagm-brats21-product-patient-complete-v1",
                "patient_pseudonym": f"patient-{pseudonym}",
                "split": split_name,
                "training_summary": str(training_summary),
                "scientific_pass_recorded": False,
            })
            return {
                "patient_pseudonym": f"patient-{pseudonym}",
                "status": "already_complete",
                "summary": str(training_summary),
                "completion_recovery_path": completion_recovery_path,
            }
        # Preserve only the invalid terminal marker as evidence. The native
        # runner owns R0/R4 substage recovery and may have a valid progress
        # checkpoint in the same training directory; moving the directory
        # wholesale would discard that resumable state.
        quarantined = _quarantine_partial_file(training_summary)
        training_recovery_path = str(quarantined.relative_to(patient_root))
    prepared_root = patient_root / "prepared"
    prepared_recovery_path: str | None = None
    source_root = _resolve_dataset_root(str(config["data"]["dataset_root"]))
    source_patient = discover_patient(source_root / patient_id)
    # Source hashes are read from the completed inventory manifest.  Do not
    # hash source files during product preparation: a full-file pass would
    # open hidden target and evaluator bytes before receipt registration.
    source_hashes = _source_hashes_for_patient(config, patient_id)
    sampling_protocol_hash = _expected_sampling_protocol_hash(
        config=config, patient=source_patient, split_name=split_name, seed=seed,
    )
    if prepared_root.exists() and not _prepared_bundle_complete(
        prepared_root,
        expected_patient_id=patient_id,
        expected_split=split_name,
        expected_target_modality=str(config["product"].get("target_modality", "flair")),
        expected_sampling_protocol_hash=sampling_protocol_hash,
        expected_source_hashes=source_hashes,
    ):
        quarantined = _quarantine_partial_directory(prepared_root)
        prepared_recovery_path = str(quarantined.relative_to(patient_root))
    if not prepared_root.exists():
        prepare_brats21_product_patient(
            source_root=source_root,
            output_dir=prepared_root,
            patient_id=patient_id,
            split=split_name,
            target_modality=str(config["product"].get("target_modality", "flair")),
            seed=seed,
            inplane_stride_vu=tuple(int(value) for value in config["data"].get("inplane_stride_vu", [4, 4])),
            sampling_config=BraTS21SamplingConfig(**dict(config["data"]["sampling"])),
            source_hashes=source_hashes,
            require_segmentation=bool(config["data"].get("require_segmentation", True)),
        )
    evaluator_manifest = _read_json(prepared_root / "evaluator_manifest.json")
    deferred_target_reader = None
    deferred_segmentation_reader = None
    if bool(evaluator_manifest.get("target_payload_deferred", False)):
        target_reference = dict(evaluator_manifest.get("target_reference", {}))
        target_modality = str(target_reference["modality_id"])
        target_path = source_patient.modality_paths[target_modality]
        target_slice_position_index = float(
            target_reference.get("source_slice_position_index", target_reference["source_slice_index"])
        )
        target_stride = tuple(int(value) for value in target_reference["inplane_stride_vu"])
        target_source_hash = str(target_reference["source_nifti_hash"])

        def deferred_target_reader() -> bytes:
            # This callback is passed into EpisodeLedger and is invoked only
            # from reveal_target after prediction receipt registration.
            digest = hashlib.sha256()
            with target_path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != target_source_hash:
                raise RuntimeError("deferred target source hash no longer matches the prepared reference")
            return npy_bytes(
                extract_axial_plane_at_position(
                    target_path,
                    target_slice_position_index,
                    inplane_stride_vu=target_stride,
                    interpolation="linear",
                )
            )

    if bool(evaluator_manifest.get("segmentation_payload_deferred", False)):
        segmentation_reference = dict(evaluator_manifest.get("segmentation_reference", {}))
        segmentation_path = source_patient.segmentation_path
        if segmentation_path is None:
            raise RuntimeError("deferred segmentation reference exists but the source patient has no segmentation")
        segmentation_position = float(segmentation_reference["source_slice_position_index"])
        segmentation_stride = tuple(int(value) for value in segmentation_reference["inplane_stride_vu"])
        segmentation_source_hash = str(segmentation_reference["source_segmentation_hash"])

        def deferred_segmentation_reader() -> bytes:
            digest = hashlib.sha256()
            with segmentation_path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != segmentation_source_hash:
                raise RuntimeError("deferred segmentation source hash no longer matches the prepared reference")
            segmentation = extract_axial_plane_at_position(
                segmentation_path,
                segmentation_position,
                inplane_stride_vu=segmentation_stride,
                interpolation="nearest",
            )
            if not np.all(np.isin(segmentation.astype(np.int64), (0, 1, 2, 4))):
                raise RuntimeError("deferred segmentation plane contains labels outside {0,1,2,4}")
            return npy_bytes(segmentation.astype(np.uint8, copy=False))

    runtime_path = _runtime_config_file(patient_root, config, propagation_variant=propagation_variant, steps=steps, wandb_mode=wandb_mode, split_name=split_name)
    evaluation_runtime_path = patient_root / "resolved_evaluation_config.json"
    evaluation_runtime = json.loads(json.dumps(config["evaluation"]))
    evaluation_runtime.update({
        "schema": "smagm-full-static-evaluation-v1",
        "claim_scope": "serialized-prediction diagnostic only",
        "target_file": "evaluator_targets.pt",
        "target_mode": "external_tensor_file",
        "sealed_audit": False,
        "diagnostic_only": True,
    })
    _atomic_json(evaluation_runtime_path, evaluation_runtime)
    report = smoke_run(
        config_path=runtime_path,
        prepared_dir=prepared_root,
        output_dir=patient_root / "training",
        evaluation_config_path=evaluation_runtime_path,
        allow_cpu_fallback=False,
        steps=steps,
        wandb_mode=wandb_mode,
        deferred_target_reader=deferred_target_reader,
        deferred_segmentation_reader=deferred_segmentation_reader,
        initial_global_checkpoint=initial_global_checkpoint,
        resume=True,
        validation_only=validation_only,
    )
    _atomic_json(summary, {
        "schema": "smagm-brats21-product-patient-complete-v1",
        "patient_pseudonym": f"patient-{pseudonym}",
        "split": split_name,
        "training_summary": str(patient_root / "training" / "summary.json"),
        "scientific_pass_recorded": False,
        "completion_recovery_path": completion_recovery_path,
        "prepared_recovery_path": prepared_recovery_path,
        "training_recovery_path": training_recovery_path,
    })
    return {
        "patient_pseudonym": f"patient-{pseudonym}",
        "split": split_name,
        "status": "complete",
        "summary": str(patient_root / "training" / "summary.json"),
        "report": report,
        "completion_recovery_path": completion_recovery_path,
        "prepared_recovery_path": prepared_recovery_path,
        "training_recovery_path": training_recovery_path,
    }


def run(*, config_path: Path, output_dir: Path, stage: str | None = None, patient_id: str | None = None, patient_limit: int | None = None, resume: str = "auto", wandb_mode: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    config, config_hash = _load_product_config(config_path)
    cohort_hash, split_hash = _cohort_and_split_hashes(config)
    config["cohort_hash"] = cohort_hash
    config["split_hash"] = split_hash
    product = config["product"]
    resolved_stage = stage or str(product.get("stage", "full"))
    if resolved_stage != "full":
        raise ValueError("the product controller has one launch mode: stage=full")
    propagation_variant = "p1"
    steps = int(product["steps"])
    if steps <= 0:
        raise ValueError("product steps must be positive")
    resolved_wandb = str(wandb_mode or product.get("wandb_mode", "disabled"))
    if resolved_wandb not in ("disabled", "offline", "online"):
        raise ValueError("wandb_mode must be disabled, offline, or online")
    output_dir = output_dir.resolve()
    disk = _disk_report(output_dir)
    minimum_free_gib = float(product.get("minimum_free_disk_gib", 50.0))
    disk_policy = str(product.get("disk_policy", "advisory")).strip().lower()
    if disk_policy not in ("advisory", "enforced"):
        raise ValueError("product disk_policy must be advisory or enforced")
    disk["minimum_free_gib"] = minimum_free_gib
    disk["safe"] = float(disk["free_gib"]) >= minimum_free_gib
    disk["policy"] = disk_policy
    disk["guard_enforced"] = disk_policy == "enforced"
    if not disk["safe"]:
        disk["warning"] = (
            f"free disk {disk['free_gib']:.2f} GiB is below the advisory threshold "
            f"{minimum_free_gib:.2f} GiB; the operating system may stop the run if storage is exhausted"
        )
    training_split = str(product.get("training_split", "train"))
    patients = _patient_ids(
        config, resolved_stage, patient_id, patient_limit if patient_limit is not None else product.get("patient_limit"),
        split_name=training_split,
    )
    validation = dict(product.get("validation", {}))
    validation_enabled = bool(validation.get("enabled", False))
    validation_split = str(validation.get("split", "validation"))
    validation_steps = int(validation.get("steps", 1))
    validation_limit = validation.get("patient_limit")
    if validation_enabled:
        if validation_split == training_split:
            raise ValueError("product validation split must be patient-disjoint from the training split")
        if validation_steps <= 0:
            raise ValueError("product validation steps must be positive when validation is enabled")
        if validation_limit is not None and int(validation_limit) <= 0:
            raise ValueError("product validation patient_limit must be positive when validation is enabled")
        validation_patients = _patient_ids(
            config, resolved_stage, None, int(validation_limit) if validation_limit is not None else None,
            split_name=validation_split,
        )
    else:
        validation_patients = ()
    dry = {
        "schema": "smagm-brats21-product-dry-run-v1",
        "stage": resolved_stage,
        "config_hash": config_hash,
        "cohort_hash": cohort_hash,
        "split_hash": split_hash,
        "propagation_variant": propagation_variant,
        "steps": steps,
        "experiment_name": str(product["experiment_name"]),
        "training_split": training_split,
        "patient_count": len(patients),
        "patient_pseudonyms": ["patient-" + hashlib.sha256(f"smagm-brats21-patient-v1:{item}".encode()).hexdigest()[:16] for item in patients],
        "validation": {
            "enabled": validation_enabled,
            "cadence": str(validation.get("cadence", "disabled")),
            "split": validation_split,
            "steps": validation_steps,
            "patient_count": len(validation_patients),
            "patient_pseudonyms": ["patient-" + hashlib.sha256(f"smagm-brats21-patient-v1:{item}".encode()).hexdigest()[:16] for item in validation_patients],
            "selection_scope": "patient-disjoint geometry-only execution sweep; no checkpoint-selection claim",
        },
        "disk": disk,
        "cuda_policy": "required; no CPU fallback",
        "output_paths": json.loads(json.dumps(product["output_paths"])),
        "wandb": {"mode": resolved_wandb, "project": "smagm-brats21", "group": product.get("wandb_group")},
        "runtime_config_hash": _hash(_runtime_config(config, propagation_variant=propagation_variant, steps=steps, wandb_mode=resolved_wandb, split_name=training_split)),
    }
    dry["execution_policy"] = "full-only; smoke and pilot launches are retired"
    if dry_run:
        print(json.dumps(dry, sort_keys=True, indent=2))
        return dry
    if not disk["safe"]:
        message = str(disk["warning"])
        if disk_policy == "enforced":
            raise RuntimeError(
                f"insufficient free disk for {resolved_stage}: "
                f"{disk['free_gib']:.2f} GiB < {minimum_free_gib:.2f} GiB"
            )
        print(f"[disk-warning] {message}", file=sys.stderr, flush=True)
    print(
        f"[experiment] name={product['experiment_name']} stage={resolved_stage} "
        f"propagation={propagation_variant} steps={steps} parameters=reported-per-patient "
        f"training_step_flops=reported-on-first-step",
        flush=True,
    )
    cuda = _cuda_preflight()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = product["output_paths"]
    state_path = output_dir / str(output_paths["state_file"])
    complete_path = output_dir / str(output_paths["completion_marker"])
    if complete_path.exists():
        raise FileExistsError(f"successful product run already exists: {output_dir}")
    if state_path.exists():
        state = _read_json(state_path)
        if state.get("status") == "complete":
            raise FileExistsError(f"successful product run already exists: {output_dir}")
        if resume == "none":
            raise FileExistsError("run state exists; use --resume auto or a new output directory")
        expected_bindings = {
            "stage": resolved_stage,
            "config_hash": config_hash,
            "cohort_hash": cohort_hash,
            "split_hash": split_hash,
            "wandb_mode": resolved_wandb,
        }
        if any(state.get(key) != value for key, value in expected_bindings.items()):
            raise ValueError("existing product run state does not match the resolved config, cohort, split, stage, or W&B mode")
        if not isinstance(state.get("completed"), dict) or not isinstance(state.get("failures"), list):
            raise ValueError("existing product run state is malformed")
    else:
        state = {"schema": "smagm-brats21-product-run-state-v1", "status": "running", "completed": {}, "failures": []}
    global_checkpoint = output_dir / str(output_paths["global_checkpoint"])
    global_update_count, global_checkpoint_sha256 = _reconcile_global_promotion_journal(
        state,
        global_checkpoint,
        cohort_hash=cohort_hash,
        split_hash=split_hash,
        config_hash=config_hash,
    )
    state.update({
        "status": "running", "stage": resolved_stage, "config_hash": config_hash,
        "cohort_hash": cohort_hash, "split_hash": split_hash, "wandb_mode": resolved_wandb,
        "cuda": cuda, "disk": disk, "experiment_name": str(product["experiment_name"]),
        "intended_branch": "feature/structure-constrained-brats-full-pipeline",
        "global_model_checkpoint": str(global_checkpoint), "global_update_count": global_update_count,
        "global_checkpoint_sha256": global_checkpoint_sha256,
        "started_at_unix": state.get("started_at_unix", time.time()),
    })
    _atomic_json(state_path, state)
    evaluation_path = _resolve_config_path(Path(config["product_path"]).parent, str(product["evaluation_config"]))
    for index, item in enumerate(patients):
        pseudonym = "patient-" + hashlib.sha256(f"smagm-brats21-patient-v1:{item}".encode()).hexdigest()[:16]
        completion_key = f"{training_split}:{pseudonym}"
        if completion_key in state["completed"]:
            if global_update_count < index + 1:
                completed_result = state["completed"][completion_key]
                if not isinstance(completed_result, dict) or not isinstance(completed_result.get("summary"), str):
                    raise ValueError("completed patient record has no training summary for global resume")
                state["pending_promotion"] = _promotion_journal(
                    result=completed_result,
                    completion_key=completion_key,
                    summary_path=Path(str(completed_result["summary"])).resolve(strict=True),
                    source_patient_pseudonym=pseudonym,
                    propagation_variant=propagation_variant,
                    global_update_index=global_update_count + 1,
                    previous_global_update_index=global_update_count,
                    previous_global_checkpoint_sha256=global_checkpoint_sha256,
                    cohort_hash=cohort_hash,
                    split_hash=split_hash,
                    config_hash=config_hash,
                )
                _atomic_json(state_path, state)
                promoted = _promote_global_model_checkpoint(
                    Path(str(completed_result["summary"])), global_checkpoint,
                    cohort_hash=cohort_hash, split_hash=split_hash,
                    source_patient_pseudonym=pseudonym, global_update_index=global_update_count + 1,
                )
                global_update_count = int(promoted["global_update_index"])
                global_checkpoint_sha256 = _file_hash(global_checkpoint)
                state["global_update_count"] = global_update_count
                state["global_checkpoint_sha256"] = global_checkpoint_sha256
                state.pop("pending_promotion", None)
                _atomic_json(state_path, state)
            continue
        try:
            result = _run_one_patient(
                config=config, patient_id=item, run_root=output_dir, propagation_variant=propagation_variant,
                steps=steps, wandb_mode=resolved_wandb, evaluation_path=evaluation_path,
                seed=int(config["training"]["seed"]) + index, split_name=training_split,
                initial_global_checkpoint=global_checkpoint if global_checkpoint.exists() else None,
            )
            if (
                result.get("status") == "already_complete"
                and global_checkpoint.exists()
                and _global_checkpoint_source_patient(global_checkpoint) == pseudonym
                and global_update_count == index + 1
            ):
                # The process may have promoted this patient and then stopped
                # before its cohort state marker was written. Reconcile the
                # marker without counting the same patient update twice.
                result["global_model_checkpoint"] = str(global_checkpoint)
                result["global_update_index"] = global_update_count
                global_checkpoint_sha256 = _file_hash(global_checkpoint)
                result["global_checkpoint_sha256"] = global_checkpoint_sha256
                state["completed"][completion_key] = result
                state["global_checkpoint_sha256"] = global_checkpoint_sha256
                _atomic_json(state_path, state)
                continue
            state["pending_promotion"] = _promotion_journal(
                result=result,
                completion_key=completion_key,
                summary_path=Path(str(result["summary"])).resolve(strict=True),
                source_patient_pseudonym=pseudonym,
                propagation_variant=propagation_variant,
                global_update_index=global_update_count + 1,
                previous_global_update_index=global_update_count,
                previous_global_checkpoint_sha256=global_checkpoint_sha256,
                cohort_hash=cohort_hash,
                split_hash=split_hash,
                config_hash=config_hash,
            )
            _atomic_json(state_path, state)
            promoted = _promote_global_model_checkpoint(
                Path(str(result["summary"])), global_checkpoint,
                cohort_hash=cohort_hash, split_hash=split_hash,
                source_patient_pseudonym=pseudonym, global_update_index=global_update_count + 1,
            )
            global_update_count = int(promoted["global_update_index"])
            global_checkpoint_sha256 = _file_hash(global_checkpoint)
            result["global_model_checkpoint"] = str(global_checkpoint)
            result["global_update_index"] = global_update_count
            result["global_checkpoint_sha256"] = global_checkpoint_sha256
            state["completed"][completion_key] = result
            state["global_update_count"] = global_update_count
            state["global_checkpoint_sha256"] = global_checkpoint_sha256
            state.pop("pending_promotion", None)
            _atomic_json(state_path, state)
        except Exception as error:
            failure = {"split": training_split, "patient_pseudonym": pseudonym, "error_type": type(error).__name__, "error": str(error), "patient_index": index}
            state["failures"].append(failure)
            state["status"] = "failed"
            _atomic_json(state_path, state)
            raise
    for index, item in enumerate(validation_patients):
        pseudonym = "patient-" + hashlib.sha256(f"smagm-brats21-patient-v1:{item}".encode()).hexdigest()[:16]
        completion_key = f"{validation_split}:{pseudonym}"
        if completion_key in state["completed"]:
            continue
        try:
            result = _run_one_patient(
                config=config, patient_id=item, run_root=output_dir, propagation_variant="p1",
                steps=validation_steps, wandb_mode=resolved_wandb, evaluation_path=evaluation_path,
                seed=int(config["training"]["seed"]) + 100000 + index, split_name=validation_split,
                initial_global_checkpoint=global_checkpoint if global_checkpoint.exists() else None,
                validation_only=True,
            )
            state["completed"][completion_key] = result
            _atomic_json(state_path, state)
        except Exception as error:
            failure = {"split": validation_split, "patient_pseudonym": pseudonym, "error_type": type(error).__name__, "error": str(error), "patient_index": index}
            state["failures"].append(failure)
            state["status"] = "failed"
            _atomic_json(state_path, state)
            raise
    metrics_report = _write_product_metric_reports(
        state=state,
        output_dir=output_dir,
        output_paths=output_paths,
        propagation_variant=propagation_variant,
        aggregation_config=dict(config["evaluation"].get("aggregation", {})),
    )
    state["metrics"] = metrics_report
    state["status"] = "complete"
    state["finished_at_unix"] = time.time()
    _atomic_json(state_path, state)
    _atomic_json(output_dir / str(output_paths["completion_marker"]), {
        "schema": "smagm-brats21-product-complete-v1", "config_hash": config_hash,
        "training_split": training_split, "patient_count": len(patients),
        "validation_patient_count": len(validation_patients), "state": str(state_path),
        "global_model_checkpoint": str(global_checkpoint), "global_update_count": global_update_count,
        "global_checkpoint_sha256": global_checkpoint_sha256,
        "scientific_pass_recorded": False,
    })
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or validate the GPU-only streamed BraTS21 product pipeline")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("full",), help="Compatibility flag; the product controller only runs the full stage")
    parser.add_argument("--patient-id")
    parser.add_argument("--patient-limit", type=int)
    parser.add_argument("--resume", choices=("auto", "none"), default="auto")
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = run(config_path=args.config, output_dir=args.output_dir, stage=args.stage, patient_id=args.patient_id, patient_limit=args.patient_limit, resume=args.resume, wandb_mode=args.wandb_mode, dry_run=args.dry_run)
    if args.dry_run:
        return
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
