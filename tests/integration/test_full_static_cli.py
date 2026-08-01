from __future__ import annotations

import json
from pathlib import Path

import torch

from smagm.cli.audit import run as audit_run
from smagm.cli.evaluate import run as evaluate_run
from smagm.cli.full_static_train import run as train_run
from smagm.cli.reconstruct import run as reconstruct_run
from smagm.evaluation import open_serialized_predictions


ROOT = Path(__file__).resolve().parents[2]


def test_full_static_train_reconstruct_evaluate_audit_chain(tmp_path) -> None:
    config = ROOT / "configs" / "experiments" / "full_static_pipeline.json"
    plan = ROOT / "configs" / "evaluation" / "full_static_eval.json"
    train_dir = tmp_path / "train"
    prediction_dir = tmp_path / "predictions"
    report_dir = tmp_path / "report"

    training = train_run(config_path=config, steps=1, output_dir=train_dir, seed=23)
    assert training["t4_routing"] is False
    step = training["steps"][0]
    assert step["event_order"] == ["OPEN_CONTEXT", "COMMIT_TARGET", "REGISTER_PREDICTION", "REVEAL_TARGET"]
    assert step["encoder_gradient_norm"] > 0 and step["field_gradient_norm"] > 0

    reconstruction = reconstruct_run(
        checkpoint_path=train_dir / "checkpoint.pt",
        config_path=config,
        manifest_path=train_dir / "eval_manifest.json",
        output_dir=prediction_dir,
    )
    evaluation = evaluate_run(
        plan_path=plan,
        predictions_dir=prediction_dir,
        output_dir=report_dir,
    )
    audit = audit_run(prediction_dir)
    assert reconstruction["state_version"] == training["state_version"]
    assert evaluation["diagnostic_only"] is True
    assert audit["package_hash"] == reconstruction["package_hash"]
    manifest = json.loads((train_dir / "eval_manifest.json").read_text(encoding="utf-8"))
    assert manifest["contains_target_payloads"] is False

    predictions = open_serialized_predictions(prediction_dir)
    volume = predictions.volumes[0]
    target_path = tmp_path / "audit_targets.pt"
    torch.save({
        "schema": "smagm-audit-targets-v1",
        "targets": [{
            "patient_id": volume.patient_id,
            "split_hash": predictions.package.split_hash,
            "modality_id": volume.modality_id,
            "grid": volume.grid.to_canonical_dict(),
            "values": volume.intensity.nan_to_num(),
            "valid_mask": ~volume.unsupported_mask,
        }],
    }, target_path)
    sealed_plan = json.loads(plan.read_text(encoding="utf-8"))
    sealed_plan.update({
        "sealed_audit": True,
        "diagnostic_only": False,
        "target_mode": "immutable_tensor_file",
        "target_file": target_path.name,
    })
    sealed_plan_path = tmp_path / "sealed_plan.json"
    sealed_plan_path.write_text(json.dumps(sealed_plan), encoding="utf-8")
    sealed = evaluate_run(
        plan_path=sealed_plan_path,
        predictions_dir=prediction_dir,
        output_dir=tmp_path / "sealed_report",
    )
    assert sealed["diagnostic_only"] is False
