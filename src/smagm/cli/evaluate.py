"""Evaluate serialized predictions; no mutable state or trainer access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from ..evaluation import (
    AuditTarget,
    FreezeRecord,
    ReconstructionMetricConfig,
    evaluate_audit_targets,
    open_serialized_audit_targets,
    open_serialized_predictions,
)


def _load_plan(path: Path) -> tuple[dict[str, object], str]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != "smagm-full-static-evaluation-v1":
        raise ValueError("evaluation plan must use the full-static evaluation schema")
    if not isinstance(plan.get("sealed_audit"), bool) or not isinstance(plan.get("diagnostic_only"), bool):
        raise ValueError("evaluation plan must explicitly declare sealed_audit and diagnostic_only")
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return plan, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_json(payload: object, path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        suffix=".tmp",
        delete=False,
        mode="w",
        encoding="utf-8",
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run(
    *,
    plan_path: Path,
    predictions_dir: Path,
    output_dir: Path,
    self_target_smoke: bool = False,
) -> dict[str, object]:
    plan, plan_hash = _load_plan(plan_path)
    predictions = open_serialized_predictions(predictions_dir)
    target_mode = "self_prediction_smoke" if self_target_smoke else plan.get("target_mode")
    if target_mode == "self_prediction_smoke":
        if plan["sealed_audit"] is not False or plan["diagnostic_only"] is not True:
            raise ValueError("self-prediction smoke must be explicitly diagnostic and unsealed")
        targets = tuple(
            AuditTarget(
                volume.patient_id,
                predictions.package.split_hash,
                volume.modality_id,
                volume.grid,
                volume.intensity.nan_to_num(),
                ~volume.unsupported_mask,
            )
            for volume in predictions.volumes
        )
        diagnostic_only = True
    elif target_mode in ("immutable_tensor_file", "external_tensor_file"):
        if target_mode == "external_tensor_file":
            if plan["sealed_audit"] is not False or plan["diagnostic_only"] is not True:
                raise ValueError("external tensor targets must remain explicitly diagnostic and unsealed")
        else:
            if plan["sealed_audit"] is not True or plan["diagnostic_only"] is not False:
                raise ValueError("immutable audit targets require a sealed non-diagnostic plan")
        target_file = plan.get("target_file")
        if not isinstance(target_file, str) or not target_file:
            raise ValueError(f"{target_mode} mode requires target_file in the frozen plan")
        target_path = Path(target_file)
        if not target_path.is_absolute():
            target_path = plan_path.parent / target_path
        expected_target_hash = plan.get("target_file_sha256")
        if expected_target_hash is not None:
            if not isinstance(expected_target_hash, str) or hashlib.sha256(target_path.read_bytes()).hexdigest() != expected_target_hash:
                raise ValueError("evaluation target file does not match the frozen target_file_sha256")
        targets = open_serialized_audit_targets(target_path)
        diagnostic_only = target_mode == "external_tensor_file"
    else:
        raise ValueError("evaluation plan must explicitly select self_prediction_smoke, external_tensor_file, or immutable_tensor_file")
    freeze = FreezeRecord(
        predictions.package.package_hash,
        predictions.package.config_hash,
        predictions.package.split_hash,
        True,
        True,
    )
    raw_metric_config = plan.get("metric_config", {})
    if not isinstance(raw_metric_config, dict):
        raise ValueError("metric_config must be an object when present")
    metric_config = ReconstructionMetricConfig(
        data_range=None if raw_metric_config.get("data_range") is None else float(raw_metric_config["data_range"]),
        ssim_window_policy=str(raw_metric_config.get("ssim_window_policy", "global")),
        edge_threshold=float(raw_metric_config.get("edge_threshold", 0.05)),
    )
    metrics = evaluate_audit_targets(predictions, targets, freeze_record=freeze, metric_config=metric_config)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "package_hash": predictions.package.package_hash,
        "plan_hash": plan_hash,
        "diagnostic_only": diagnostic_only,
        "target_mode": target_mode,
        "metrics": [metric.__dict__ for metric in metrics],
    }
    _atomic_json(report, output_dir / "evaluation.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate serialized predictions behind the audit barrier")
    parser.add_argument("--plan", type=Path, default=Path("configs/evaluation/full_static_eval.json"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--self-target-smoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run(
        plan_path=args.plan,
        predictions_dir=args.predictions,
        output_dir=args.output_dir,
        self_target_smoke=args.self_target_smoke,
    )
    print(json.dumps(report, sort_keys=True) if args.json else json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
