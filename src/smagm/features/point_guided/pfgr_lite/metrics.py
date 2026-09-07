"""Versioned PFGR-Lite metric reductions and scientific decision helpers.

The functions in this module are deliberately tensor-only (apart from JSON
serialisation helpers).  They never load data, call a model, or infer a metric
range from a prediction.  Experiment services provide the declared target,
mask, and fixed normalisation range and retain the paired subject rows before
calling the reducers here.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

METRICS_SCHEMA = "pfgr-lite-metrics-v1"
METRIC_FORMULA_VERSION = "masked-mae-psnr-global-ssim-charbonnier-v1"
SIGNED_IMPROVEMENT_VERSION = "before-minus-after-errors_after-minus-before-quality-v1"
DEFAULT_CHARBONNIER_EPSILON = 1e-3
DEFAULT_DATA_RANGE = 1.0
DEFAULT_SSIM_WINDOW = 11
COMPARISON_OPTIONS_SCHEMA = "pfgr-lite-comparison-options-v1"
SSIM_MASK_VERSION = "center_observed_valid_window_v1"


def _finite_float(name: str, value: object, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else ''}number")
    return result


def _volume(value: Tensor, name: str) -> Tensor:
    """Return one subject's scalar volume as ``[D,H,W]``."""

    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating tensor")
    if value.ndim == 5:
        if value.shape[0] != 1 or value.shape[1] != 1:
            raise ValueError(f"{name} rank-5 form must be [1,1,D,H,W]")
        value = value[0, 0]
    elif value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError(f"{name} rank-4 form must be [1,D,H,W]")
        value = value[0]
    elif value.ndim != 3:
        raise ValueError(f"{name} must be [D,H,W], [1,D,H,W], or [1,1,D,H,W]")
    if value.numel() == 0 or any(int(size) <= 0 for size in value.shape):
        raise ValueError(f"{name} must have positive dimensions")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def _mask(value: Tensor | None, shape: Sequence[int], device: torch.device) -> Tensor:
    if value is None:
        return torch.ones(tuple(int(item) for item in shape), dtype=torch.bool, device=device)
    if not isinstance(value, Tensor):
        raise TypeError("observation_mask must be a tensor or None")
    if value.ndim == 5:
        if value.shape[:2] != (1, 1):
            raise ValueError("observation_mask rank-5 form must be [1,1,D,H,W]")
        value = value[0, 0]
    elif value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError("observation_mask rank-4 form must be [1,D,H,W]")
        value = value[0]
    elif value.ndim != 3:
        raise ValueError("observation_mask must be [D,H,W], [1,D,H,W], or [1,1,D,H,W]")
    if tuple(value.shape) != tuple(int(item) for item in shape):
        raise ValueError("observation_mask shape must match the volume")
    if value.dtype is not torch.bool:
        if not value.is_floating_point() and value.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("observation_mask must be bool or numeric binary")
        if value.numel() == 0 or not bool(torch.isfinite(value).all()):
            raise ValueError("observation_mask must be finite and nonempty")
        if not bool(((value == 0) | (value == 1)).all()):
            raise ValueError("observation_mask must contain only exact 0/1 values")
        value = value.to(dtype=torch.bool)
    value = value.to(device=device)
    if int(value.sum().item()) <= 0:
        raise ValueError("observation_mask must contain at least one valid voxel")
    return value


def _number(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _masked_mean(value: Tensor, mask: Tensor) -> float:
    selected = value[mask]
    if selected.numel() == 0:
        raise ValueError("metric mask has no valid voxels")
    return float(selected.to(dtype=torch.float64).mean().item())


def _global_ssim(
    prediction: Tensor,
    target: Tensor,
    *,
    mask: Tensor,
    data_range: float,
    window: int,
) -> tuple[float | None, str | None, int]:
    """Compute valid-window 3-D SSIM, returning an explicit unavailable reason."""

    if not isinstance(window, int) or isinstance(window, bool) or window <= 0 or window % 2 == 0:
        raise ValueError("ssim_window must be a positive odd integer")
    if min(int(size) for size in prediction.shape) < window:
        return None, "ssim_window_larger_than_volume", 0
    # Valid windows avoid implicit padding.  The mask does not alter local
    # statistics (which would change SSIM's definition); it selects windows
    # whose centre voxel is observed and supplies an explicit denominator.
    pad = 0
    shape = tuple(int(size) for size in prediction.shape)
    pred5 = prediction.to(dtype=torch.float64).reshape(1, 1, *shape)
    target5 = target.to(dtype=torch.float64).reshape(1, 1, *shape)
    kernel = (window, window, window)
    mu_pred = F.avg_pool3d(pred5, kernel, stride=1, padding=pad)
    mu_target = F.avg_pool3d(target5, kernel, stride=1, padding=pad)
    mu_pred_sq = mu_pred.square()
    mu_target_sq = mu_target.square()
    mu_cross = mu_pred * mu_target
    sigma_pred = F.avg_pool3d(pred5.square(), kernel, stride=1, padding=pad) - mu_pred_sq
    sigma_target = F.avg_pool3d(target5.square(), kernel, stride=1, padding=pad) - mu_target_sq
    sigma_cross = F.avg_pool3d(pred5 * target5, kernel, stride=1, padding=pad) - mu_cross
    # Numerical round-off can make a variance tiny negative in FP64; this is
    # a non-semantic guard and is not an epsilon support/pruning rule.
    sigma_pred = sigma_pred.clamp_min(0.0)
    sigma_target = sigma_target.clamp_min(0.0)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = ((2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)) / (
        (mu_pred_sq + mu_target_sq + c1) * (sigma_pred + sigma_target + c2)
    )
    # Centre mask for valid windows.  A window is valid only if its centre is
    # observed, preserving the fixed global mask denominator for point metrics.
    centre_offset = window // 2
    centre_mask = mask[
        centre_offset : shape[0] - centre_offset,
        centre_offset : shape[1] - centre_offset,
        centre_offset : shape[2] - centre_offset,
    ]
    if int(centre_mask.sum().item()) == 0:
        return None, "ssim_mask_has_no_valid_window_centres", 0
    selected = score[0, 0][centre_mask]
    return float(selected.mean().item()), None, int(selected.numel())


def dense_metrics(
    prediction: Tensor,
    target: Tensor,
    observation_mask: Tensor | None = None,
    *,
    data_range: float = DEFAULT_DATA_RANGE,
    charbonnier_epsilon: float = DEFAULT_CHARBONNIER_EPSILON,
    ssim_window: int = DEFAULT_SSIM_WINDOW,
) -> dict[str, Any]:
    """Compute absolute metrics for one subject using fixed declared scaling.

    ``data_range`` is required to be a fixed pipeline/configuration value; it
    is never inferred from the target or predictions.  SSIM is marked
    unavailable when its valid window cannot fit the volume or mask.
    """

    data_range = _finite_float("data_range", data_range, positive=True)
    epsilon = _finite_float("charbonnier_epsilon", charbonnier_epsilon, positive=True)
    pred = _volume(prediction, "prediction")
    truth = _volume(target, "target").to(device=pred.device)
    if pred.shape != truth.shape:
        raise ValueError("prediction and target shapes must match")
    valid = _mask(observation_mask, pred.shape, pred.device)
    error = pred - truth
    abs_error = error.abs()
    mse = _masked_mean(error.square(), valid)
    mae = _masked_mean(abs_error, valid)
    charbonnier = _masked_mean(torch.sqrt(error.square() + epsilon * epsilon), valid)
    if mse == 0.0:
        psnr: float | None = None
        psnr_reason = "zero_mse"
    else:
        psnr = float(10.0 * math.log10(data_range * data_range / mse))
        psnr_reason = None
    ssim, ssim_reason, ssim_count = _global_ssim(
        pred,
        truth,
        mask=valid,
        data_range=data_range,
        window=ssim_window,
    )
    return {
        "schema_version": METRICS_SCHEMA,
        "formula_version": METRIC_FORMULA_VERSION,
        "mae": mae,
        "mse": mse,
        "psnr": psnr,
        "psnr_unavailable_reason": psnr_reason,
        "ssim": ssim,
        "ssim_unavailable_reason": ssim_reason,
        "ssim_valid_window_count": ssim_count,
        "ssim_mask_definition": SSIM_MASK_VERSION,
        "masked_charbonnier": charbonnier,
        "mask_count": int(valid.sum().item()),
        "data_range": data_range,
        "charbonnier_epsilon": epsilon,
        "ssim_window": ssim_window,
        "reduction": "masked_global_mean_fixed_denominator_v1",
    }


def paired_subject_metrics(
    initial_prediction: Tensor,
    final_prediction: Tensor,
    target: Tensor,
    observation_mask: Tensor | None = None,
    *,
    data_range: float = DEFAULT_DATA_RANGE,
    charbonnier_epsilon: float = DEFAULT_CHARBONNIER_EPSILON,
    ssim_window: int = DEFAULT_SSIM_WINDOW,
    subject_id: str | None = None,
    context_id: str | None = None,
    scenario: str | None = None,
    budget: int | None = None,
) -> dict[str, Any]:
    """Return paired initial/final metrics and signed improvements."""

    before = dense_metrics(
        initial_prediction,
        target,
        observation_mask,
        data_range=data_range,
        charbonnier_epsilon=charbonnier_epsilon,
        ssim_window=ssim_window,
    )
    after = dense_metrics(
        final_prediction,
        target,
        observation_mask,
        data_range=data_range,
        charbonnier_epsilon=charbonnier_epsilon,
        ssim_window=ssim_window,
    )
    improvements: dict[str, float | None] = {}
    for name in ("mae", "masked_charbonnier"):
        left, right = _number(before.get(name)), _number(after.get(name))
        improvements[name] = None if left is None or right is None else left - right
    for name in ("psnr", "ssim"):
        left, right = _number(before.get(name)), _number(after.get(name))
        improvements[name] = None if left is None or right is None else right - left
    result: dict[str, Any] = {
        "schema_version": METRICS_SCHEMA,
        "formula_version": METRIC_FORMULA_VERSION,
        "signed_improvement_version": SIGNED_IMPROVEMENT_VERSION,
        "subject_id": subject_id,
        "context_id": context_id,
        "scenario": scenario,
        "budget": budget,
        "before": before,
        "after": after,
        "improvement": improvements,
    }
    return result


# Explicit name used by benchmark tests to make the independent dense path
# obvious.  It intentionally delegates only to tensor metric algebra.
direct_dense_metrics = dense_metrics


def _iter_metric_values(rows: Iterable[Mapping[str, Any]], section: str, key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        section_value = row.get(section, row) if isinstance(row, Mapping) else None
        if not isinstance(section_value, Mapping):
            continue
        number = _number(section_value.get(key))
        if number is not None:
            values.append(number)
    return values


def aggregate_subject_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce paired rows with explicit sum/count/max and preserved K bins."""

    materialised = [dict(row) for row in rows]
    fields = ("mae", "psnr", "ssim", "masked_charbonnier", "mse")
    aggregate: dict[str, Any] = {
        "schema_version": METRICS_SCHEMA,
        "formula_version": METRIC_FORMULA_VERSION,
        "signed_improvement_version": SIGNED_IMPROVEMENT_VERSION,
        "subject_count": len(materialised),
        "sum": {},
        "count": {},
        "max": {},
        "mean": {},
        "k_bins": {str(index): 0 for index in range(5)},
    }
    sections = ("before", "after", "improvement")
    for section in sections:
        aggregate["sum"][section] = {}
        aggregate["count"][section] = {}
        aggregate["max"][section] = {}
        aggregate["mean"][section] = {}
        for key in fields:
            values = _iter_metric_values(materialised, section, key)
            aggregate["sum"][section][key] = float(sum(values))
            aggregate["count"][section][key] = len(values)
            aggregate["max"][section][key] = max(values) if values else None
            aggregate["mean"][section][key] = (
                float(sum(values) / len(values)) if values else None
            )
    for row in materialised:
        budget = row.get("budget")
        if isinstance(budget, int) and 0 <= budget <= 4:
            aggregate["k_bins"][str(budget)] += 1
    aggregate["subjects"] = materialised
    return aggregate


def aggregate_metric_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Alias retaining an explicit name for service callers."""

    return aggregate_subject_metrics(rows)


def _gain_number(row: Mapping[str, Any]) -> float | None:
    for key in ("true_gain", "raw_gain", "gain", "measured_gain"):
        value = _number(row.get(key))
        if value is not None:
            return value
    return None


def action_metric_row(
    row: Mapping[str, Any] | object,
    *,
    numerical_tolerance: float = 1e-10,
    practical_margin: float = 0.0,
    scope: str | None = None,
    selected: bool | None = None,
) -> dict[str, Any]:
    """Normalise one measured action while retaining signed/invalid outcomes."""

    if isinstance(row, Mapping):
        payload = dict(row)
    else:
        payload = {
            name: getattr(row, name)
            for name in (
                "action_id",
                "raw_gain",
                "benefit",
                "harm",
                "q_draws",
                "standard_error",
                "role",
                "measurement_mode",
                "state_version",
                "point_id",
                "context_id",
            )
            if hasattr(row, name)
        }
    tol = _finite_float("numerical_tolerance", numerical_tolerance)
    margin = _finite_float("practical_margin", practical_margin)
    if tol < 0.0 or margin < 0.0:
        raise ValueError("numerical_tolerance and practical_margin must be nonnegative")
    gain = _gain_number(payload)
    if gain is None:
        classification = "unknown"
    elif gain > margin:
        classification = "useful_positive"
    elif gain < -margin:
        classification = "harmful_negative"
    elif abs(gain) <= tol:
        classification = "numerically_neutral"
    else:
        classification = "practically_neutral"
    output = dict(payload)
    output.update(
        {
            "schema_version": METRICS_SCHEMA,
            "true_gain": gain,
            "scope": scope if scope is not None else payload.get("scope", "measured_action"),
            "selected": selected if selected is not None else payload.get("selected"),
            "numerical_tolerance": tol,
            "practical_margin": margin,
            "classification": classification,
        }
    )
    return output


def aggregate_action_metrics(
    rows: Iterable[Mapping[str, Any] | object],
    *,
    numerical_tolerance: float = 1e-10,
    practical_margin: float = 0.0,
) -> dict[str, Any]:
    materialised = [
        action_metric_row(
            row,
            numerical_tolerance=numerical_tolerance,
            practical_margin=practical_margin,
        )
        for row in rows
    ]
    counts = Counter(str(row.get("classification", "unknown")) for row in materialised)
    gains = [float(row["true_gain"]) for row in materialised if _number(row.get("true_gain")) is not None]
    return {
        "schema_version": METRICS_SCHEMA,
        "subject_count": len({str(row.get("subject_id")) for row in materialised if row.get("subject_id") is not None}),
        "action_count": len(materialised),
        "measured_denominator": len(gains),
        "sum_true_gain": float(sum(gains)),
        "mean_true_gain": float(sum(gains) / len(gains)) if gains else None,
        "max_true_gain": max(gains) if gains else None,
        "classification_counts": {key: int(counts.get(key, 0)) for key in (
            "useful_positive",
            "harmful_negative",
            "numerically_neutral",
            "practically_neutral",
            "unknown",
        )},
        "numerical_tolerance": float(numerical_tolerance),
        "practical_margin": float(practical_margin),
        "rows": materialised,
    }


def telescoping_residual(
    executed_gains: Iterable[float],
    initial_loss: float,
    final_loss: float,
) -> float:
    """Return exact signed route residual ``sum(g)-[R0-RK]``."""

    gains = [
        _finite_float("executed gain", value)
        for value in executed_gains
    ]
    initial = _finite_float("initial_loss", initial_loss)
    final = _finite_float("final_loss", final_loss)
    return float(sum(gains) - (initial - final))


def stopping_diagnostics(
    rows: Iterable[Mapping[str, Any]],
    *,
    practical_margin: float,
    numerical_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Classify STOP/selection diagnostics without assuming unmeasured candidates."""

    margin = _finite_float("practical_margin", practical_margin)
    tol = _finite_float("numerical_tolerance", numerical_tolerance)
    if margin < 0.0 or tol < 0.0:
        raise ValueError("margins must be nonnegative")
    counters = Counter()
    coverage = Counter()
    denominators = Counter()
    unknown_reasons = Counter()
    for row in rows:
        stop_code = str(row.get("stop_code", ""))
        scope = str(row.get("candidate_scope", row.get("scope", "selected_only")))
        gain = _gain_number(row)
        best = _number(row.get("best_true_gain"))
        if scope not in {"all", "subset", "selected_only"}:
            scope = "unknown"
        coverage[scope] += 1
        if stop_code == "budget":
            counters["budget_exit"] += 1
        elif stop_code == "low_gain":
            counters["low_gain_exit"] += 1
            if best is not None:
                denominators["false_stop"] += 1
                if best > margin + tol:
                    counters["false_stop"] += 1
            else:
                unknown_reasons["false_stop_unmeasured_candidates"] += 1
        if bool(row.get("selected", False)):
            if gain is not None:
                denominators["harmful_selection"] += 1
                if gain < -margin - tol:
                    counters["harmful_selection"] += 1
            else:
                unknown_reasons["selected_gain_unmeasured"] += 1
        if bool(row.get("continued", False)):
            if best is not None:
                denominators["false_continuation"] += 1
                if best <= margin + tol:
                    counters["false_continuation"] += 1
            else:
                unknown_reasons["continuation_candidates_unmeasured"] += 1
    denominators["stop_events"] = sum(
        counters[key] for key in ("budget_exit", "low_gain_exit")
    )
    denominators["selected_events"] = denominators["harmful_selection"] + unknown_reasons["selected_gain_unmeasured"]
    denominators["continuation_events"] = denominators["false_continuation"] + unknown_reasons["continuation_candidates_unmeasured"]
    return {
        "schema_version": METRICS_SCHEMA,
        "practical_margin": margin,
        "numerical_tolerance": tol,
        "counts": {key: int(counters[key]) for key in (
            "budget_exit",
            "low_gain_exit",
            "false_stop",
            "harmful_selection",
            "false_continuation",
        )},
        "measurement_denominators": {key: int(denominators[key]) for key in (
            "stop_events",
            "selected_events",
            "continuation_events",
            "false_stop",
            "harmful_selection",
            "false_continuation",
        )},
        "unknown_count": int(sum(unknown_reasons.values())),
        "unknown_reasons": {key: int(value) for key, value in sorted(unknown_reasons.items())},
        "candidate_scope_coverage": {key: int(coverage[key]) for key in ("all", "subset", "selected_only", "unknown")},
    }


def headroom_metrics(
    oracle_gain: float | None,
    random_gain: float | None,
    *,
    practical_margin: float,
) -> dict[str, Any]:
    """Compute unclipped correction/selection headroom in raw gain units."""

    margin = _finite_float("practical_margin", practical_margin)
    if margin < 0.0:
        raise ValueError("practical_margin must be nonnegative")
    oracle = _number(oracle_gain)
    random = _number(random_gain)
    if oracle is None:
        return {
            "correction_headroom": None,
            "selection_headroom": None,
            "recovery": None,
            "reason": "oracle_unavailable",
            "practical_margin": margin,
        }
    selection = None if random is None else oracle - random
    return {
        "correction_headroom": oracle,
        "selection_headroom": selection,
        "learned_minus_random": None,
        "reason": None,
        "practical_margin": margin,
    }


@dataclass(frozen=True)
class ComparisonOptions:
    """Strict predeclared reductions for paired scientific comparisons."""

    practical_margin: float = 0.0
    numerical_tolerance: float = 1e-10
    minimum_subjects: int = 32
    confidence_level: float = 0.95
    schema_version: str = COMPARISON_OPTIONS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != COMPARISON_OPTIONS_SCHEMA:
            raise ValueError("unknown ComparisonOptions schema")
        margin = _finite_float("practical_margin", self.practical_margin)
        tolerance = _finite_float("numerical_tolerance", self.numerical_tolerance)
        if margin < 0.0 or tolerance < 0.0:
            raise ValueError("practical_margin and numerical_tolerance must be nonnegative")
        if not isinstance(self.minimum_subjects, int) or isinstance(self.minimum_subjects, bool) or self.minimum_subjects <= 0:
            raise ValueError("minimum_subjects must be positive")
        level = _finite_float("confidence_level", self.confidence_level)
        if not 0.0 < level < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one")

    def as_dict(self) -> dict[str, Any]:
        return {field.name: _jsonable(getattr(self, field.name)) for field in fields(self)}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ComparisonOptions:
        if not isinstance(values, Mapping):
            raise TypeError("ComparisonOptions must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown ComparisonOptions keys: {sorted(unknown)}")
        return cls(**dict(values))


def _comparison_rows(artifact: object, name: str) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    """Extract paired rows and provenance from a JSON/JSONL-like artifact."""

    def _merge_metadata(
        current: Mapping[str, Any],
        incoming: Mapping[str, Any],
    ) -> dict[str, Any]:
        merged = dict(current)
        for key, value in incoming.items():
            if key in merged:
                left = json.dumps(_jsonable(merged[key]), sort_keys=True)
                right = json.dumps(_jsonable(value), sort_keys=True)
                if left != right:
                    raise ValueError(
                        f"{name} provenance mismatch for overlapping field {key!r}"
                    )
            merged[key] = value
        return merged

    metadata: Mapping[str, Any] = {}
    rows: object = artifact
    if isinstance(artifact, Mapping):
        # Public W5 services return a receipt containing artifact paths.  Read
        # those exact files here so comparison is callable on real service
        # output rather than requiring callers to hand-copy rows.
        metrics_path = artifact.get("metrics_path")
        if metrics_path is not None:
            metrics_file = Path(str(metrics_path))
            if metrics_file.exists():
                try:
                    metrics_payload = json.loads(metrics_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(f"{name} metrics artifact is unreadable") from exc
                if isinstance(metrics_payload, Mapping):
                    source = metrics_payload.get("source_receipt", metrics_payload.get("provenance", {}))
                    if isinstance(source, Mapping):
                        metadata = source
        rows_path = artifact.get("paired_subjects_path", artifact.get("output_path"))
        loaded_rows: object | None = None
        if rows_path is not None:
            rows_file = Path(str(rows_path))
            if not rows_file.exists():
                raise ValueError(f"{name} rows artifact does not exist: {rows_file}")
            try:
                text = rows_file.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(f"{name} rows artifact is unreadable") from exc
            if rows_file.suffix.lower() == ".jsonl":
                loaded_rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            else:
                try:
                    loaded_rows = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{name} rows artifact is not valid JSON") from exc
        raw_meta = artifact.get("source_receipt", artifact.get("provenance", {}))
        if isinstance(raw_meta, Mapping):
            metadata = _merge_metadata(metadata, raw_meta)
        rows = loaded_rows if loaded_rows is not None else artifact.get(
            "paired_subjects", artifact.get("rows", artifact.get("subjects", artifact))
        )
        if isinstance(rows, Mapping) and "subjects" in rows:
            rows = rows["subjects"]
        # Oracle JSONL stores provenance on each payload; use the first
        # payload's receipt when the outer service mapping has none.
        if isinstance(rows, Sequence) and rows and isinstance(rows[0], Mapping):
            row_meta = rows[0].get("source_receipt", rows[0].get("provenance", {}))
            if isinstance(row_meta, Mapping):
                # Multi-subject service receipts publish a per-subject
                # initialization map at the outer boundary while each JSONL
                # row retains its scalar context initialization hash.  Verify
                # that the row agrees with the map, then omit the redundant
                # scalar before strict outer-vs-row provenance merging.  A
                # mismatch remains a hard join failure.
                subject_map = metadata.get("subject_initialization_hashes")
                row_initialization = row_meta.get("initialization_hash")
                if isinstance(subject_map, Mapping) and row_initialization is not None:
                    row_subject = rows[0].get("subject_id")
                    expected_initialization = subject_map.get(str(row_subject))
                    if expected_initialization != row_initialization:
                        raise ValueError(
                            f"{name} row initialization_hash conflicts with subject_initialization_hashes"
                        )
                    row_meta = {
                        key: value
                        for key, value in row_meta.items()
                        if key != "initialization_hash"
                    }
                metadata = _merge_metadata(metadata, row_meta)
    if isinstance(rows, Mapping):
        rows = [rows]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError(f"{name} artifact must provide a sequence of paired subject rows")
    result: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"{name} row {index} must be a mapping")
        subject_id = row.get("subject_id")
        if not isinstance(subject_id, str) or not subject_id:
            raise ValueError(f"{name} row {index} has no nonempty subject_id")
        result.append(row)
    if len({str(row["subject_id"]) for row in result}) != len(result):
        raise ValueError(f"{name} artifact contains duplicate subject IDs")
    return result, metadata


def _comparison_gain(row: Mapping[str, Any]) -> float | None:
    improvement = row.get("improvement")
    if isinstance(improvement, Mapping):
        return _number(improvement.get("masked_charbonnier"))
    paired = row.get("paired_metrics")
    if isinstance(paired, Mapping):
        nested = paired.get("improvement")
        if isinstance(nested, Mapping):
            return _number(nested.get("masked_charbonnier"))
    route_gain = row.get("oracle_route_gain")
    if route_gain is not None:
        return _number(route_gain)
    return _number(row.get("masked_charbonnier_gain", row.get("true_gain")))


def _comparison_z0_loss(row: Mapping[str, Any]) -> float | None:
    """Read a baseline Z0 loss from a direct or paired metric row."""

    for key in ("z0_masked_charbonnier", "baseline_loss", "masked_charbonnier"):
        value = _number(row.get(key))
        if value is not None:
            return value
    before = row.get("before")
    if isinstance(before, Mapping):
        value = _number(before.get("masked_charbonnier"))
        if value is not None:
            return value
    return None


def _comparison_z0_identity(row: Mapping[str, Any]) -> str | None:
    value = row.get("z0_digest", row.get("z0_state_digest"))
    if value is None and isinstance(row.get("paired_metrics"), Mapping):
        paired = row["paired_metrics"]
        value = paired.get("z0_digest", paired.get("z0_state_digest"))
    if value is None:
        return None
    text = str(value).strip()
    return text if text and text.lower() not in {"none", "unknown", "unset", "null"} else None


def _join_provenance(named: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if not named:
        raise ValueError("paired artifact provenance is empty")
    required_groups = (
        ("producer_compatibility_hash",),
        ("checkpoint_hash", "initialization_hash"),
        ("baseline_split_hash",),
        ("training_role_manifest_hash",),
        ("split_role",),
        ("normalization_hash",),
        ("mask_definition",),
        ("loss_definition", "label_definition"),
    )
    for group in required_groups:
        # For a multi-subject service run, initialization is intentionally a
        # subject-keyed identity map.  It is an exact alternative to a scalar
        # checkpoint/initialization identity, not a relaxed missing field.
        if group == ("checkpoint_hash", "initialization_hash"):
            map_available = all(
                isinstance(metadata.get("subject_initialization_hashes"), Mapping)
                and bool(metadata.get("subject_initialization_hashes"))
                for metadata in named.values()
            )
            if map_available:
                map_values = {
                    json.dumps(
                        _jsonable(metadata["subject_initialization_hashes"]),
                        sort_keys=True,
                    )
                    for metadata in named.values()
                }
                if len(map_values) != 1:
                    raise ValueError(
                        "paired artifact provenance mismatch for subject_initialization_hashes"
                    )
                continue
        available = [
            key
            for key in group
            if all(
                key in metadata
                and metadata[key] is not None
                and str(metadata[key]).strip()
                and str(metadata[key]).lower() not in {"unknown", "unset", "none", "null"}
                for metadata in named.values()
            )
        ]
        if not available:
            engineering_only = all(
                bool(metadata.get("engineering_only", False))
                for metadata in named.values()
            )
            if engineering_only and group in (
                ("baseline_split_hash",),
                ("training_role_manifest_hash",),
            ):
                # Tiny fixtures are allowed to omit role identities only when
                # every artifact explicitly declares engineering_only.  Keep
                # the gap in the joined receipt; production cannot pass this
                # branch.
                continue
            raise ValueError(
                "paired artifact provenance requires one of " + ", ".join(group)
            )
    keys = {
        "source_hash",
        "checkpoint_hash",
        "initialization_hash",
        "subject_initialization_hashes",
        "producer_compatibility_hash",
        "baseline_split_hash",
        "training_role_manifest_hash",
        "split_role",
        "split_role_hash",
        "normalization_hash",
        "mask_definition",
        "loss_definition",
        "label_definition",
        "data_range",
        "engineering_only",
    }
    joined: dict[str, Any] = {}
    for key in sorted(keys):
        present = [(name, metadata[key]) for name, metadata in named.items() if key in metadata]
        if not present:
            continue
        if len(present) != len(named):
            missing = sorted(set(named) - {name for name, _ in present})
            raise ValueError(f"paired artifact provenance field {key!r} missing from {missing}")
        values = {json.dumps(_jsonable(value), sort_keys=True) for _, value in present}
        if len(values) != 1:
            raise ValueError(f"paired artifact provenance mismatch for {key!r}")
        joined[key] = present[0][1]
    return joined


def compare_paired_artifacts(
    learned: object | None,
    random: object,
    oracle: object,
    *,
    z0: object | None = None,
    options: ComparisonOptions | None = None,
) -> dict[str, Any]:
    """Compare oracle/learned/random paired artifacts with strict joins.

    The reductions are target-after-inference diagnostics only.  Subject IDs
    and every supplied source/checkpoint/split identity must match; candidate
    rows are never treated as independent subjects.  Oracle-gap recovery is
    reported only for subjects whose measured oracle-minus-random gain clears
    the declared practical margin, with no clipping or epsilon denominator.
    """

    resolved = ComparisonOptions() if options is None else options
    if not isinstance(resolved, ComparisonOptions):
        raise TypeError("options must be ComparisonOptions")
    # R4 can be run before a fitted ValueNet artifact exists.  Keep the
    # learned series explicitly unavailable rather than requiring a fabricated
    # placeholder; random/oracle/z0 provenance and paired joins remain strict.
    artifacts = {"random": random, "oracle": oracle}
    if learned is not None:
        artifacts["learned"] = learned
    if z0 is not None:
        artifacts["z0"] = z0
    parsed = {name: _comparison_rows(value, name) for name, value in artifacts.items()}
    rows_by_name = {name: rows for name, (rows, _) in parsed.items()}
    metadata_by_name = {name: metadata for name, (_, metadata) in parsed.items()}
    provenance = _join_provenance(metadata_by_name)
    subject_sets = {name: {str(row["subject_id"]) for row in rows} for name, rows in rows_by_name.items()}
    common = set.intersection(*subject_sets.values()) if subject_sets else set()
    if not common:
        raise ValueError("paired artifacts have no common subject IDs")
    missing = {name: sorted(subject_sets[name] - common) for name in subject_sets if subject_sets[name] - common}
    if missing:
        raise ValueError(f"paired artifacts have nonmatching subject joins: {missing}")
    indexed = {name: {str(row["subject_id"]): row for row in rows} for name, rows in rows_by_name.items()}
    ordered_subjects = sorted(common)
    for subject_id in ordered_subjects:
        identities = {
            name: _comparison_z0_identity(indexed[name][subject_id])
            for name in rows_by_name
        }
        if any(value is None for value in identities.values()):
            missing = sorted(name for name, value in identities.items() if value is None)
            raise ValueError(
                f"paired subject {subject_id!r} is missing per-subject z0 identity in {missing}"
            )
        if len(set(identities.values())) != 1:
            raise ValueError(
                f"paired subject {subject_id!r} has mismatched per-subject z0 identities"
            )
    series: dict[str, list[float]] = {"oracle_vs_z0": [], "oracle_vs_random": [], "learned_vs_random": [], "random_vs_z0": [], "learned_vs_z0": []}
    unknown: dict[str, int] = Counter()
    route_gap_values: list[float] = []
    for subject_id in ordered_subjects:
        gains = {
            name: _comparison_gain(indexed[name][subject_id])
            for name in ("random", "oracle")
        }
        if "learned" in indexed:
            gains["learned"] = _comparison_gain(indexed["learned"][subject_id])
        if z0 is not None:
            gains["z0"] = _comparison_z0_loss(indexed["z0"][subject_id])
        baseline = gains.get("z0", 0.0)
        if baseline is None:
            unknown["z0_gain_unmeasured"] = unknown.get("z0_gain_unmeasured", 0) + 1
            baseline = 0.0
            baseline_available = False
        else:
            baseline_available = True
        if gains["oracle"] is None or not baseline_available:
            unknown["oracle_gain_unmeasured"] = unknown.get("oracle_gain_unmeasured", 0) + 1
        else:
            series["oracle_vs_z0"].append(float(gains["oracle"] - baseline))
        if gains["oracle"] is None or gains["random"] is None:
            unknown["oracle_or_random_gain_unmeasured"] = unknown.get("oracle_or_random_gain_unmeasured", 0) + 1
        else:
            series["oracle_vs_random"].append(float(gains["oracle"] - gains["random"]))
        if "learned" not in gains:
            unknown["learned_artifact_unavailable"] = unknown.get("learned_artifact_unavailable", 0) + 1
        elif gains["learned"] is None or gains["random"] is None:
            unknown["learned_or_random_gain_unmeasured"] = unknown.get("learned_or_random_gain_unmeasured", 0) + 1
        else:
            series["learned_vs_random"].append(float(gains["learned"] - gains["random"]))
        if gains["random"] is not None and baseline_available:
            series["random_vs_z0"].append(float(gains["random"] - baseline))
        if gains.get("learned") is not None and baseline_available:
            series["learned_vs_z0"].append(float(gains["learned"] - baseline))
        if gains["oracle"] is not None and gains.get("learned") is not None and baseline_available:
            route_gap_values.append(float(gains["oracle"] - gains["learned"]))

    decisions = {key: scientific_decision(values, practical_margin=resolved.practical_margin, minimum_subjects=resolved.minimum_subjects, confidence_level=resolved.confidence_level) for key, values in series.items()}
    oracle_z0 = decisions["oracle_vs_z0"]
    oracle_random = decisions["oracle_vs_random"]
    random_z0 = decisions["random_vs_z0"]
    learned_random = decisions["learned_vs_random"]
    learned_z0 = decisions["learned_vs_z0"]
    required_decisions = [oracle_z0, oracle_random, random_z0]
    if "learned" in indexed:
        required_decisions.extend((learned_random, learned_z0))
    if any(item["decision"] == "INCONCLUSIVE" for item in required_decisions):
        r4_branch = "INCONCLUSIVE"
        r4_action = "underpowered_or_unmeasured"
    elif oracle_z0["decision"] == "FAIL":
        r4_branch = "NO_MATERIAL_ORACLE_Z0"
        r4_action = "improve_base_u_or_check_capacity"
    elif random_z0["decision"] == "PASS" and oracle_random["decision"] == "FAIL":
        r4_branch = "RANDOM_APPROX_ORACLE_BOTH_POSITIVE"
        r4_action = "correction_useful_router_optional"
    elif oracle_random["decision"] == "PASS" and learned_random["decision"] == "FAIL":
        r4_branch = "ORACLE_ABOVE_RANDOM_LEARNED_POOR"
        r4_action = "diagnose_labels_v_or_ranking"
    elif learned_random["decision"] == "PASS" and learned_z0["decision"] == "PASS":
        r4_branch = "LEARNED_USEFUL"
        r4_action = "review_calibration_and_holdout_evaluation"
    else:
        r4_branch = "LEARNED_UNAVAILABLE" if "learned" not in indexed else "INCONCLUSIVE"
        r4_action = "await_learned_value_artifact" if "learned" not in indexed else "retain_declared_uncertainty"
    recovery_values: list[float] = []
    recovery_denominator = 0
    for subject_id in ordered_subjects:
        oracle_gain = _comparison_gain(indexed["oracle"][subject_id])
        random_gain = _comparison_gain(indexed["random"][subject_id])
        learned_gain = (
            _comparison_gain(indexed["learned"][subject_id])
            if "learned" in indexed
            else None
        )
        if oracle_gain is None or random_gain is None or learned_gain is None:
            continue
        gap = float(oracle_gain - random_gain)
        if gap > resolved.practical_margin + resolved.numerical_tolerance:
            recovery_denominator += 1
            recovery_values.append(float((learned_gain - random_gain) / gap))
    return {
        "schema_version": METRICS_SCHEMA,
        "comparison_schema_version": COMPARISON_OPTIONS_SCHEMA,
        "software_status": "SOFTWARE_PASS",
        "scientific_status": "INCONCLUSIVE" if r4_branch in {"INCONCLUSIVE", "LEARNED_UNAVAILABLE"} else ("FAIL" if r4_branch == "NO_MATERIAL_ORACLE_Z0" else "PASS"),
        "options": resolved.as_dict(),
        "provenance": provenance,
        "subject_count": len(ordered_subjects),
        "subjects": ordered_subjects,
        "pairwise": {key: {"values": values, "decision": decisions[key]} for key, values in series.items()},
        "headroom": {
            "oracle_vs_z0": series["oracle_vs_z0"],
            "oracle_vs_random": series["oracle_vs_random"],
            "learned_vs_random": series["learned_vs_random"],
            "recovery_values": recovery_values,
            "recovery_denominator": recovery_denominator,
            "recovery_mean": None if not recovery_values else float(sum(recovery_values) / len(recovery_values)),
            "recovery_scope": "oracle_minus_random_above_margin_only",
            "recovery_unknown_reason": None if "learned" in indexed else "learned_artifact_unavailable",
        },
        "stop_aware": {
            "scope": "measured_subject_pairs_only; unmeasured_candidates_unknown",
            # Whole-route oracle-minus-learned differences are not top-1
            # STOP-aware regret; candidate-scope regret is unavailable unless
            # both artifacts expose the same-state proposal bank.
            "route_gap_values": route_gap_values,
            "top1_regret_values": [],
            "top1_regret_unknown_reason": "same_state_candidate_scope_not_supplied",
            "unknown_count": int(sum(unknown.values())),
            "unknown_reasons": dict(sorted(unknown.items())),
        },
        "r4_decision": {"branch": r4_branch, "action": r4_action, "underpowered": r4_branch == "INCONCLUSIVE"},
    }


def scientific_decision(
    paired_differences: Sequence[float],
    *,
    practical_margin: float,
    minimum_subjects: int = 32,
    confidence_level: float = 0.95,
    direction: str = "positive",
    method: str = "normal_approximation",
) -> dict[str, Any]:
    """Return PASS/FAIL/INCONCLUSIVE from a declared paired uncertainty rule."""

    if method != "normal_approximation":
        raise ValueError("only the declared normal_approximation method is supported")
    margin = _finite_float("practical_margin", practical_margin)
    if margin < 0.0:
        raise ValueError("practical_margin must be nonnegative")
    if not isinstance(minimum_subjects, int) or isinstance(minimum_subjects, bool) or minimum_subjects <= 0:
        raise ValueError("minimum_subjects must be positive")
    level = _finite_float("confidence_level", confidence_level)
    if not 0.0 < level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    values = [_finite_float("paired difference", value) for value in paired_differences]
    count = len(values)
    mean = float(sum(values) / count) if values else None
    if count >= 2:
        tensor = torch.tensor(values, dtype=torch.float64)
        standard_error = float(tensor.std(unbiased=True).item() / math.sqrt(count))
    else:
        standard_error = None
    # 1.96 is the declared two-sided 95% normal approximation.  Other levels
    # remain explicit INCONCLUSIVE rather than silently changing quantiles.
    if math.isclose(level, 0.95, rel_tol=0.0, abs_tol=1e-12) and standard_error is not None and mean is not None:
        half_width = 1.96 * standard_error
        lower, upper = mean - half_width, mean + half_width
    else:
        lower = upper = None
    status = "INCONCLUSIVE"
    signed_lower = None if lower is None else (lower if direction == "positive" else -upper)
    signed_upper = None if upper is None else (upper if direction == "positive" else -lower)
    if count >= minimum_subjects and signed_lower is not None and signed_upper is not None:
        if signed_lower > margin:
            status = "PASS"
        elif signed_upper < margin:
            status = "FAIL"
    if direction not in {"positive", "negative"}:
        raise ValueError("direction must be positive or negative")
    return {
        "schema_version": METRICS_SCHEMA,
        "decision": status,
        "method": method,
        "confidence_level": level,
        "minimum_subjects": minimum_subjects,
        "subject_count": count,
        "practical_margin": margin,
        "direction": direction,
        "mean": mean,
        "standard_error": standard_error,
        "ci_lower": signed_lower,
        "ci_upper": signed_upper,
        "underpowered": count < minimum_subjects,
    }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a deterministic JSON object without overwriting an existing file."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite metric artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_jsonable(dict(payload)), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "COMPARISON_OPTIONS_SCHEMA",
    "DEFAULT_CHARBONNIER_EPSILON",
    "DEFAULT_DATA_RANGE",
    "DEFAULT_SSIM_WINDOW",
    "METRICS_SCHEMA",
    "METRIC_FORMULA_VERSION",
    "SIGNED_IMPROVEMENT_VERSION",
    "SSIM_MASK_VERSION",
    "ComparisonOptions",
    "action_metric_row",
    "aggregate_action_metrics",
    "aggregate_metric_rows",
    "aggregate_subject_metrics",
    "compare_paired_artifacts",
    "dense_metrics",
    "direct_dense_metrics",
    "headroom_metrics",
    "paired_subject_metrics",
    "scientific_decision",
    "stopping_diagnostics",
    "telescoping_residual",
    "write_json",
]
