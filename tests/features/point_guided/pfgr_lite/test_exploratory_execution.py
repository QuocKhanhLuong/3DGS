"""Explicit full-run execution is distinct from a scientific release permit."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from smagm.cli import pfgr_lite as cli
from smagm.features.point_guided.pfgr_lite.artifacts import (
    EvidenceValidationError,
    UnsafeEvidenceError,
    package_evidence,
)


def _request(**changes):
    values = {
        "exploratory_run": True,
        "engineering_only": True,
        "review_receipt": None,
        "synthetic": False,
    }
    return SimpleNamespace(**(values | changes))


def _authorize(args, root, context=None):
    return cli._authorize_execution(
        args,
        run_dir=root,
        scope="R9-final-evaluation",
        config_hash="a" * 64,
        context=context
        or {"selected_subject_ids": ["case-a"], "budget": 4, "seed": 17},
        artifacts={"checkpoint_sha256": "b" * 64},
    )


def test_execution_receipt_binds_workload_without_minting_human_approval(tmp_path):
    args = _request()
    first = _authorize(args, tmp_path)
    assert first["decision"] == "EXPLORATORY_EXECUTION"
    assert first["human_reviewed"] is False
    assert first["authorizes_main"] is False
    assert "reviewer" not in first
    assert first == args._execution_authorization
    assert json.loads((tmp_path / "execution_authorization.json").read_text()) == first
    second = _authorize(
        args, tmp_path, {"selected_subject_ids": ["case-b"], "budget": 4, "seed": 17}
    )
    assert second["cohort_hash"] != first["cohort_hash"]
    assert second["config_hash"] == first["config_hash"]


@pytest.mark.parametrize(
    "changes",
    [
        {"engineering_only": False},
        {"review_receipt": Path("human.json")},
        {"exploratory_run": False},
    ],
)
def test_execution_flag_cannot_silently_replace_other_authority(tmp_path, changes):
    with pytest.raises(cli.CLIError):
        _authorize(_request(**changes), tmp_path)
    assert not (tmp_path / "execution_authorization.json").exists()


def test_ordinary_review_receipt_still_checks_exact_cohort(tmp_path):
    context = {"selected_subject_ids": ["case-a"], "seed": 17}
    path = tmp_path / "review.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": cli.REVIEW_RECEIPT_SCHEMA,
                "scope": "R9-final-evaluation",
                "decision": "APPROVED",
                "reviewer": "test fixture",
                "created_at": "2026-09-12",
                "config_hash": "a" * 64,
                "cohort_hash": cli._review_context_hash(context),
                "artifacts": {"checkpoint_sha256": "b" * 64},
            }
        )
    )
    args = _request(exploratory_run=False, engineering_only=False, review_receipt=path)
    assert _authorize(args, tmp_path, context)["decision"] == "APPROVED"
    with pytest.raises(cli.CLIError, match="cohort_hash"):
        _authorize(args, tmp_path, {"selected_subject_ids": ["other-case"], "seed": 17})


def test_exploratory_receipt_packaging_rejects_false_release_claim(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    payload = _authorize(_request(), run)
    manifest = package_evidence([run], tmp_path / "package")
    assert any(
        Path(row["source_path"]).name == "execution_authorization.json"
        for row in manifest["included"]
    )
    payload["authorizes_main"] = True
    (run / "execution_authorization.json").write_text(json.dumps(payload))
    with pytest.raises(EvidenceValidationError, match="invalid exploratory"):
        package_evidence([run], tmp_path / "reject")


def test_engineering_resume_preserves_producer_config_before_cached_join(
    monkeypatch, tmp_path
):
    from smagm.features.point_guided import PointGuidedConfig
    from smagm.features.point_guided.pfgr_lite import checkpoint
    from smagm.features.point_guided.pfgr_lite.config import (
        PFGRLiteConfig,
        frontend_config_to_dict,
    )

    producer_config = PFGRLiteConfig(engineering_only=False)
    bundle = SimpleNamespace(
        config={
            "pfgr_config": producer_config.as_dict(),
            "frontend_config": frontend_config_to_dict(
                PointGuidedConfig(num_semantic_classes=3)
            ),
        }
    )
    resume_path = tmp_path / "value-resume.pt"
    loaded = []

    def restore(path):
        loaded.append(path)
        return SimpleNamespace(inference=bundle)

    monkeypatch.setattr(checkpoint, "load_resume", restore)
    args = cli._parser().parse_args(
        [
            "resume",
            "--config",
            "configs/pfgr_lite/main.json",
            "--resume-checkpoint",
            str(resume_path),
            "--engineering-only",
            "--device",
            "cpu",
        ]
    )
    config, details = cli._config_for_command(args, stage="S3")
    assert loaded == [resume_path]
    assert config.as_dict() == producer_config.as_dict()
    assert config.engineering_only is False
    assert details["execution"]["stage_options"]["engineering_only"] is True
    assert cli._resolve_cached_value_config(config, bundle) == producer_config


@pytest.mark.parametrize(
    "invalid",
    [
        {"budgets": [[0, 1, 2, 4]]},
        {"cohorts": {"test:4": {"subject_ids": [[1, 2, 3]]}}},
        {"errors": [{"stage": "R9", "status": "failed", "error": [1, 2]}]},
    ],
)
def test_full_pipeline_metadata_vectors_remain_bounded(tmp_path, invalid):
    run = tmp_path / "run"
    run.mkdir()
    payload = {
        "schema_version": "pfgr-lite-pipeline-summary-v1",
        "budgets": [0, 1, 2, 4],
        "cohorts": {
            "validation:4": {"subject_ids": ["val-a"]},
            "test:4": {"subject_ids": ["test-a"]},
        },
        "errors": [{"stage": "R9", "status": "failed", "error": "fixture"}],
    }
    path = run / "pipeline_summary.json"
    path.write_text(json.dumps(payload))
    manifest = package_evidence([run], tmp_path / "good")
    assert any(
        Path(row["source_path"]).name == path.name for row in manifest["included"]
    )
    path.write_text(json.dumps(payload | invalid))
    with pytest.raises(UnsafeEvidenceError):
        package_evidence([run], tmp_path / "bad")


def test_incomplete_value_resume_logs_package_field_names_only(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    fields = ["model_state", "optimizer_state", "cursor", "rng_state"]
    receipt = run / "receipt.json"
    receipt.write_text(json.dumps({"metrics": {"resume_keys": fields}}))
    incomplete = run / "value_fit_incomplete.json"
    incomplete.write_text(json.dumps({"resume_keys": fields, "fit_complete": False}))
    result = package_evidence([run], tmp_path / "good")
    assert {Path(row["source_path"]).name for row in result["included"]} == {
        "receipt.json",
        "value_fit_incomplete.json",
    }
    incomplete.write_text(json.dumps({"resume_keys": [[1, 2, 3]]}))
    with pytest.raises(UnsafeEvidenceError, match="resume field names"):
        package_evidence([run], tmp_path / "bad")


def test_exploratory_calibration_authorizes_before_collection_and_never_exports_adaptive(
    tmp_path, monkeypatch
):
    from smagm.features.point_guided.pfgr_lite import calibration_runner, checkpoint
    from smagm.features.point_guided.pfgr_lite.config import PFGRLiteConfig
    from smagm.features.point_guided.pfgr_lite.value_net import SignedValueNet

    @dataclass
    class Inputs:
        model: object = None
        role_manifest: object = None
        metadata: dict = field(default_factory=dict)

    @dataclass
    class UnexpectedAdaptiveResult:
        # Defense in depth: even a future regression returning adaptive here
        # cannot publish a release from an exploratory command.
        capability: str = "adaptive"

    config = PFGRLiteConfig(engineering_only=True)
    context = {"selected_subject_ids": ["fit", "allowance"], "seed": 17}
    value = SignedValueNet(input_variant=366)
    value_artifact = SimpleNamespace(
        value_fit_identity=SimpleNamespace(
            input_variant=366, digest="v" * 64, gain_scale_hash="g" * 64
        ),
        gain_scale={"scale": 1.0},
        state_dict=value.state_dict(),
    )
    bundle = SimpleNamespace(producer=None, role_manifest=None)
    checkpoint_path = tmp_path / "model.pt"
    value_path = tmp_path / "value.pt"
    checkpoint_path.write_bytes(b"fixture")
    value_path.write_bytes(b"fixture value")
    args = cli._parser().parse_args(
        [
            "calibrate",
            "--checkpoint",
            str(checkpoint_path),
            "--value-checkpoint",
            str(value_path),
            "--engineering-only",
            "--exploratory-run",
            "--device",
            "cpu",
            "--seed",
            "17",
        ]
    )
    monkeypatch.setattr(cli, "_reserve_run", lambda *_: tmp_path)
    monkeypatch.setattr(cli, "_config_for_command", lambda *_a, **_k: (config, {}))
    monkeypatch.setattr(
        cli, "_production_inputs", lambda *_a, **_k: Inputs(model=torch.nn.Linear(1, 1))
    )
    monkeypatch.setattr(cli, "_review_context", lambda **_k: context)
    monkeypatch.setattr(cli, "_counter_receipt", lambda *_: {})
    monkeypatch.setattr(cli, "_publish_receipt", lambda *_a, **kwargs: kwargs)
    monkeypatch.setattr(checkpoint, "load_inference_bundle", lambda *_a, **_k: bundle)
    monkeypatch.setattr(
        checkpoint, "load_value_artifact", lambda *_a, **_k: value_artifact
    )
    exported = []
    monkeypatch.setattr(
        checkpoint, "save_inference_bundle", lambda *a: exported.append(a)
    )

    def collect(_inputs, options, _output_dir):
        assert options.engineering_only is True
        receipt = json.loads((tmp_path / "execution_authorization.json").read_text())
        assert receipt["cohort_hash"] == cli._review_context_hash(context)
        return {
            "calibration": UnexpectedAdaptiveResult(),
            "calibration_evidence": None,
            "artifacts": {},
            "metrics": {},
        }

    monkeypatch.setattr(calibration_runner, "run_calibration", collect)
    result = cli._calibrate_command(args)
    assert result["status"] == "INCONCLUSIVE"
    assert result["metrics"]["adaptive_available"] is False
    assert not exported
    assert not (tmp_path / "adaptive.pt").exists()
    assert (
        json.loads((tmp_path / "calibration.json").read_text())["adaptive_available"]
        is False
    )


@pytest.mark.parametrize(
    "command, flags, module_name, function_name",
    [
        ("benchmark", [], "benchmark", "run_teacher_benchmark"),
        (
            "oracle-evaluate",
            ["--oracle-mode", "sampled_one", "--budget", "1"],
            "oracle",
            "run_oracle_evaluation",
        ),
        (
            "evaluate",
            [
                "--scenario",
                "noop",
                "--budget",
                "0",
                "--split-role",
                "test",
                "--exploratory-run",
            ],
            "experiments",
            "run_evaluation",
        ),
    ],
)
def test_real_data_engineering_flag_reaches_service_options(
    tmp_path,
    monkeypatch,
    command,
    flags,
    module_name,
    function_name,
):
    import importlib

    from smagm.features.point_guided.pfgr_lite import checkpoint
    from smagm.features.point_guided.pfgr_lite.config import PFGRLiteConfig

    producer_path = tmp_path / "producer.pt"
    producer_path.write_bytes(b"fixture producer")
    args = cli._parser().parse_args(
        [
            command,
            "--checkpoint",
            str(producer_path),
            "--engineering-only",
            "--device",
            "cpu",
            *flags,
        ]
    )
    # Hydration preserves the production PFGR config. The execution sidecar
    # must still retain the explicitly requested engineering capability.
    config = PFGRLiteConfig(engineering_only=False)
    monkeypatch.setattr(cli, "_config_for_command", lambda *_a, **_k: (config, {}))
    monkeypatch.setattr(cli, "_inputs", lambda *_: SimpleNamespace())
    monkeypatch.setattr(cli, "_reserve_run", lambda *_: tmp_path)
    monkeypatch.setattr(
        cli,
        "_review_context",
        lambda **_k: {"selected_subject_ids": ["heldout-a"], "seed": args.seed},
    )
    monkeypatch.setattr(
        checkpoint, "load_inference_bundle", lambda *_a, **_k: SimpleNamespace()
    )
    service = importlib.import_module(
        "smagm.features.point_guided.pfgr_lite." + module_name
    )

    class ServiceReached(Exception):
        pass

    def observe(_inputs, options, _run_dir):
        assert options.engineering_only is True
        assert args.synthetic is False
        if command == "evaluate":
            request = json.loads(
                (tmp_path / "execution_authorization.json").read_text()
            )
            assert request["context"]["selected_subject_ids"] == ["heldout-a"]
            assert request["artifacts"]["checkpoint_sha256"] == cli._sha256(
                producer_path
            )
        raise ServiceReached

    monkeypatch.setattr(service, function_name, observe)
    with pytest.raises(ServiceReached):
        cli._service_command(args, command)
