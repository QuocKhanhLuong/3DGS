from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from smagm.cli.audit import run as audit_run
from smagm.cli.evaluate import run as evaluate_run
from smagm.cli.full_static_train import run as train_run
from smagm.cli.reconstruct import run as reconstruct_run
from smagm.evaluation import open_serialized_predictions
from smagm.features.encoder import EncoderConfig, EvidenceEncoder
from smagm.state import load_patient_state


ROOT = Path(__file__).resolve().parents[2]


def _field_state_hash(payload: dict[str, torch.Tensor]) -> str:
    content = b"".join(
        name.encode() + value.detach().cpu().contiguous().numpy().tobytes()
        for name, value in payload.items()
    )
    return hashlib.sha256(content).hexdigest()


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

    checkpoint = torch.load(train_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    patient_state = load_patient_state(train_dir / "patient_state.pt")
    assert checkpoint["patient_state_version"] == patient_state.state_version
    assert checkpoint["field_for_patient_state_hash"] == patient_state.field_model_hash
    assert _field_state_hash(checkpoint["field"]) == patient_state.field_model_hash
    encoder = EvidenceEncoder(EncoderConfig(variant="e2"))
    encoder.load_state_dict(checkpoint["encoder"])
    assert encoder.state_hash() == checkpoint["encoder_for_patient_state_hash"]
    assert checkpoint["post_snapshot_optimizer_updates"] == 1
    assert any(
        not torch.equal(checkpoint["field"][name], checkpoint["field_after_training"][name])
        for name in checkpoint["field"]
    )

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
