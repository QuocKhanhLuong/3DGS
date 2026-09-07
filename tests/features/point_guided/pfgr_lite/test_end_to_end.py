from __future__ import annotations

import json
from pathlib import Path

from smagm.cli.pfgr_lite import (
    _config_for_command,
    _parser,
    _review_context,
    _review_context_hash,
    _sha256,
    _synthetic_inputs,
    main,
)


def test_synthetic_command_chain_emits_dependency_manifests_without_fake_success(tmp_path: Path) -> None:
    """Bounded dry chain covers the complete CLI family without training data."""

    output = tmp_path / "runs"
    commands = [
        ("static-train", ["--base", "b2"]),
        ("updater-train", ["--spectral-arm", "u_plus_spectral"]),
        ("bank-build", ["--teacher-mode", "iid_fixed_q"]),
        ("value-fit", ["--bank-index", str(tmp_path / "missing-index.json"), "--value-input", "366"]),
        ("calibrate", ["--checkpoint", str(tmp_path / "producer.pt"), "--value-checkpoint", str(tmp_path / "value.pt")]),
        ("evaluate", ["--checkpoint", str(tmp_path / "calibrated.pt"), "--scenario", "random", "--budget", "1"]),
        ("resume", ["--resume-checkpoint", str(tmp_path / "resume.pt")]),
    ]
    for index, (command, extra) in enumerate(commands):
        name = f"{index}-{command}"
        argv = [
            command,
            "--synthetic",
            "--config",
            "configs/pfgr_lite/synthetic.json",
            "--output-root",
            str(output),
            "--run-name",
            name,
            "--dry-manifest",
            *extra,
        ]
        assert main(argv) == 0
        manifest = json.loads((output / name / "dry_manifest.json").read_text(encoding="utf-8"))
        assert manifest["scientific_status"] == "NOT_EVALUATED"
        assert manifest["status"] in {"DRY_MANIFEST", "BLOCKED"}


def test_synthetic_services_publish_real_static_bank_value_and_resume_artifacts(tmp_path: Path) -> None:
    """Exercise the bounded service chain; only adaptive calibration remains blocked."""

    output = tmp_path / "runs"
    config = "configs/pfgr_lite/synthetic.json"
    assert main(["smoke", "--synthetic", "--config", config, "--output-root", str(output), "--run-name", "smoke", "--max-steps", "1"]) == 0
    smoke = output / "smoke"
    assert (smoke / "inference.pt").is_file()
    assert (smoke / "resume.pt").is_file()

    assert main(["bank-build", "--synthetic", "--config", config, "--output-root", str(output), "--run-name", "bank", "--max-subjects", "1", "--candidate-count", "2", "--query-count", "4"]) == 0
    bank = output / "bank" / "S2" / "bank" / "index.json"
    assert bank.is_file()
    assert main(["bank-verify", "--synthetic", "--config", config, "--bank-index", str(bank), "--checkpoint", str(output / "bank" / "inference.pt"), "--output-root", str(output), "--run-name", "verify"]) == 0

    assert main(["value-fit", "--synthetic", "--config", config, "--bank-index", str(bank), "--checkpoint", str(output / "bank" / "inference.pt"), "--output-root", str(output), "--run-name", "value", "--value-input", "366", "--epochs", "1", "--batch-size", "4"]) == 0
    value = output / "value" / "value.pt"
    assert value.is_file()
    assert main(
        [
            "calibrate",
            "--synthetic",
            "--config",
            config,
            "--checkpoint",
            str(output / "bank" / "inference.pt"),
            "--value-checkpoint",
            str(value),
            "--output-root",
            str(output),
            "--run-name",
            "calibration-review-dry",
            "--max-subjects",
            "2",
            "--dry-manifest",
        ]
    ) == 0
    review_context = json.loads((output / "calibration-review-dry" / "review_context.json").read_text(encoding="utf-8"))
    assert review_context["status"] == "REVIEW_REQUIRED"
    assert review_context["decision_required"] is True
    assert review_context["scientific_status"] == "NOT_EVALUATED"
    assert main(["value-evaluate", "--synthetic", "--config", config, "--bank-index", str(bank), "--checkpoint", str(output / "bank" / "inference.pt"), "--value-checkpoint", str(value), "--output-root", str(output), "--run-name", "value-eval"]) == 0
    value_eval_receipt = json.loads((output / "value-eval" / "receipt.json").read_text(encoding="utf-8"))
    paired_path = output / "value-eval" / "value_evaluate_pairs.json"
    assert paired_path.is_file()
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    assert paired["schema_version"] == "pfgr-lite-value-evaluation-pairs-v1"
    assert paired["same_bank"] is True
    assert paired["row_count"] == value_eval_receipt["metrics"]["paired_ranking_row_count"]
    assert all(row["bank_manifest_hash"] == paired["bank_manifest_hash"] and row["row_key"] for row in paired["rows"])

    assert main(["resume", "--synthetic", "--config", config, "--resume-checkpoint", str(smoke / "resume.pt"), "--output-root", str(output), "--run-name", "resume"]) == 0
    calibration = output / "calibration"
    review_receipt = tmp_path / "review.json"
    review_args = _parser().parse_args(
        [
            "calibrate",
            "--synthetic",
            "--config",
            config,
            "--checkpoint",
            str(output / "bank" / "inference.pt"),
            "--value-checkpoint",
            str(value),
            "--max-subjects",
            "2",
        ]
    )
    review_config, _ = _config_for_command(review_args, stage="S5")
    review_bundle = __import__("smagm.features.point_guided.pfgr_lite.checkpoint", fromlist=["load_inference_bundle"]).load_inference_bundle(output / "bank" / "inference.pt")
    review_inputs = _synthetic_inputs(review_args, review_config, stage="S5")
    review_value = __import__("smagm.features.point_guided.pfgr_lite.checkpoint", fromlist=["load_value_artifact"]).load_value_artifact(value, expected_producer=review_bundle.producer)
    review_config_hash = __import__("hashlib").sha256(json.dumps(review_config.as_dict(), sort_keys=True).encode()).hexdigest()
    review_context = _review_context(
        scope="R7-calibration-cohort",
        config_hash=review_config_hash,
        inputs=review_inputs,
        bundle=review_bundle,
        args=review_args,
        policy="adaptive-calibration",
        budget=4,
        value_identity_hash=review_value.value_fit_identity.digest,
        split_role="calibration",
    )
    review_receipt.write_text(
        json.dumps(
            {
                "schema_version": "pfgr-lite-review-receipt-v1",
                "scope": "R7-calibration-cohort",
                "decision": "ENGINEERING_DIAGNOSTIC",
                "reviewer": "pytest-fixture",
                "created_at": "2026-09-07T00:00:00Z",
                "config_hash": review_config_hash,
                "cohort_hash": _review_context_hash(review_context),
                "artifacts": {
                    "checkpoint_sha256": _sha256(output / "bank" / "inference.pt"),
                    "value_checkpoint_sha256": _sha256(value),
                    "role_manifest_digest": review_bundle.role_manifest.digest,
                    "split_hash": review_bundle.role_manifest.baseline_split_hash,
                },
            }
        ),
        encoding="utf-8",
    )
    assert main(["calibrate", "--synthetic", "--config", config, "--checkpoint", str(output / "bank" / "inference.pt"), "--value-checkpoint", str(value), "--review-receipt", str(review_receipt), "--output-root", str(output), "--run-name", "calibration", "--max-subjects", "2"]) == 0
    receipt = json.loads((calibration / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "INCONCLUSIVE"
    assert receipt["metrics"]["actual_forced_route_collection"] is True
    assert receipt["metrics"]["actual_teacher_measurement"] is True
    assert receipt["metrics"]["insufficient_data"] is True
    assert receipt["metrics"]["capability"] is None
    assert receipt["metrics"]["target_reads"] == 2
    assert receipt["metrics"]["staged_trace_count"] == 2
    assert receipt["metrics"]["replay_count"] == 2
    assert receipt["metrics"]["staged_tensor_bytes"] > 0
    operation_counters = receipt["metrics"]["operation_counters"]
    assert operation_counters["medicalnet_traversals"] == 4
    assert operation_counters["target_validations"] == 2
    assert operation_counters["decoder_calls"] > 0
    assert (calibration / "calibration" / "trace_receipts.json").is_file()
