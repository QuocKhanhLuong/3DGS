from __future__ import annotations

import json

import pytest
import torch

from smagm.features.point_guided import PointGuidedConfig
from smagm.features.point_guided.contracts import VolumeGeometry
from smagm.features.point_guided.pfgr_lite.config import (
    PFGRLiteConfig,
    PFGRPolicyConfig,
)
from smagm.features.point_guided.pfgr_lite.data import (
    DataAccessCounters,
    TargetFreeSample,
    build_training_role_manifest,
)
from smagm.features.point_guided.pfgr_lite.experiments import (
    ExperimentOptions,
    _source_receipt,
    run_evaluation,
)
from smagm.features.point_guided.pfgr_lite.metrics import compare_paired_artifacts
from smagm.features.point_guided.pfgr_lite.model import PFGRLiteModel
from smagm.features.point_guided.pfgr_lite.oracle import (
    OracleOptions,
    run_oracle_evaluation,
)
from smagm.features.point_guided.pfgr_lite.stages import (
    StageExecutionConfig,
    StageInputs,
    StageOptions,
)


def _sample() -> TargetFreeSample:
    geometry = VolumeGeometry.from_spacing((3, 3, 3), (1.0, 1.0, 1.0))
    return TargetFreeSample(
        "subject-01",
        torch.zeros((3, 3, 3, 3), dtype=torch.float32),
        torch.ones((1, 3, 3, 3), dtype=torch.bool),
        geometry,
        {},
        "",
        "",
    )


def test_experiment_options_are_strict_and_versioned() -> None:
    options = ExperimentOptions.from_dict({"scenario": "random", "budget": 2, "engineering_only": True})
    assert options.scenario == "random"
    assert options.budget == 2
    with pytest.raises(ValueError, match="unknown ExperimentOptions"):
        ExperimentOptions.from_dict({"not_an_option": 1})
    with pytest.raises(ValueError, match="budget"):
        ExperimentOptions(budget=3)
    assert ExperimentOptions.from_dict({"local_footprint_audit": True}).local_footprint_audit is True


def test_evaluation_routes_before_deferred_target_and_writes_paired_artifacts(tmp_path) -> None:
    order: list[str] = []
    sample = _sample()
    initial = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32)
    final = torch.full_like(initial, 0.5)

    def route(sample, **kwargs):
        assert kwargs.get("target") is None
        order.append("route")
        return {"initial_prediction": initial, "final_prediction": final, "k": 0, "stop_reason": "budget", "decisions": ()}

    def target_provider(subject_id: str):
        assert subject_id == sample.subject_id
        order.append("target")
        return torch.zeros((1, 3, 3, 3), dtype=torch.float32)

    inputs = StageInputs(samples=(sample,), route_builder=route, target_provider=target_provider)
    result = run_evaluation(
        inputs,
        ExperimentOptions(scenario="noop", budget=0, max_subjects=1, engineering_only=True, ssim_window=3),
        tmp_path / "evaluation",
    )
    assert order == ["route", "target"]
    assert result["software_status"] == "SOFTWARE_PASS"
    assert (tmp_path / "evaluation" / "metrics.json").exists()
    assert (tmp_path / "evaluation" / "paired_subjects.jsonl").read_text().count("\n") == 1
    assert (tmp_path / "evaluation" / "action_metrics.jsonl").read_text() == ""
    with pytest.raises(FileExistsError):
        run_evaluation(
            inputs,
            ExperimentOptions(scenario="noop", budget=0, max_subjects=1, engineering_only=True),
            tmp_path / "evaluation",
        )


def test_evaluation_requires_late_target_provider(tmp_path) -> None:
    sample = _sample()
    inputs = StageInputs(
        samples=(sample,),
        route_builder=lambda _sample, **_kwargs: {"initial_prediction": torch.zeros(1, 1, 3, 3, 3), "final_prediction": torch.zeros(1, 1, 3, 3, 3)},
    )
    with pytest.raises(ValueError, match="deferred target_provider"):
        run_evaluation(inputs, ExperimentOptions(engineering_only=True), tmp_path / "missing")


def test_target_read_counter_has_one_owner_even_when_provider_counts(tmp_path) -> None:
    sample = _sample()
    counters = DataAccessCounters()

    def route(_sample, **_kwargs):
        return {
            "initial_prediction": torch.zeros(1, 1, 3, 3, 3),
            "final_prediction": torch.zeros(1, 1, 3, 3, 3),
        }

    def provider(_subject_id: str):
        # Engineering providers may already own the shared counter; the W5
        # service must not add a second increment for the same callback.
        counters.target_reads += 1
        return torch.zeros((1, 3, 3, 3), dtype=torch.float32)

    inputs = StageInputs(
        samples=(sample,),
        route_builder=route,
        target_provider=provider,
        metadata={"counters": counters},
    )
    run_evaluation(
        inputs,
        ExperimentOptions(
            scenario="noop",
            budget=0,
            max_subjects=1,
            engineering_only=True,
            ssim_window=3,
        ),
        tmp_path / "counter-owner",
    )
    assert counters.target_reads == 1


def test_evaluation_restores_module_mode_and_disables_gradients(tmp_path) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU())
    model.train()
    model[1].eval()
    sample = _sample()
    initial = torch.ones((1, 1, 3, 3, 3), requires_grad=True)
    final = torch.zeros_like(initial)

    def route(_sample, **_kwargs):
        assert torch.is_grad_enabled() is False
        assert model.training is False
        return {"initial_prediction": initial, "final_prediction": final}

    inputs = StageInputs(
        samples=(sample,),
        model=model,
        route_builder=route,
        target_provider=lambda _subject_id: torch.zeros((1, 3, 3, 3)),
    )
    run_evaluation(
        inputs,
        ExperimentOptions(
            scenario="noop",
            budget=0,
            max_subjects=1,
            engineering_only=True,
            ssim_window=3,
        ),
        tmp_path / "detached",
    )
    assert model.training is True
    assert model[1].training is False
    assert all(parameter.grad is None for parameter in model.parameters())


def test_source_receipt_rejects_actual_identity_override() -> None:
    options = ExperimentOptions(engineering_only=True)
    inputs = StageInputs(
        metadata={
            "source_receipt": {
                "producer_compatibility_hash": "stale-producer",
                "baseline_split_hash": "stale-split",
            }
        }
    )
    with pytest.raises(ValueError, match="producer_compatibility_hash"):
        _source_receipt(
            inputs,
            options,
            contexts=[
                {
                    "producer_compatibility_hash": "actual-producer",
                    "normalization_hash": "actual-normalization",
                    "initialization_hash": "actual-init",
                }
            ],
        )

    role_manifest = build_training_role_manifest(
        {
            "train_subject_ids": ("train",),
            "val_subject_ids": ("validation",),
            "test_subject_ids": ("test",),
            "split_hash": "actual-split",
        },
        engineering_only=True,
    )
    role_inputs = StageInputs(
        role_manifest=role_manifest,
        metadata={"source_receipt": {"baseline_split_hash": "stale-split"}},
    )
    with pytest.raises(ValueError, match="baseline_split_hash"):
        _source_receipt(role_inputs, options)


def test_source_receipt_keeps_subject_specific_initialization_hashes() -> None:
    receipt = _source_receipt(
        StageInputs(),
        ExperimentOptions(engineering_only=True),
        contexts=[
            {
                "subject_id": "s0",
                "producer_compatibility_hash": "producer",
                "normalization_hash": "normalization",
                "initialization_hash": "init-s0",
            },
            {
                "subject_id": "s1",
                "producer_compatibility_hash": "producer",
                "normalization_hash": "normalization",
                "initialization_hash": "init-s1",
            },
        ],
    )
    assert receipt["initialization_hash"] is None
    assert receipt["subject_initialization_hashes"] == {
        "s0": "init-s0",
        "s1": "init-s1",
    }


def test_parallel_evaluation_measures_frozen_initial_scope(tmp_path) -> None:
    frontend = PointGuidedConfig(
        num_semantic_classes=3,
        num_points=4,
        point_candidate_multiplier=3,
        offset_hidden_channels=12,
        detach_backbone_features=False,
    )
    config = PFGRLiteConfig(
        num_points=4,
        engineering_only=True,
        policy=PFGRPolicyConfig(mode="parallel_topk"),
    )
    model = PFGRLiteModel(config, frontend_config=frontend).eval()
    geometry = VolumeGeometry.from_spacing((9, 9, 9))
    sample = TargetFreeSample(
        "parallel-subject",
        torch.randn((3, 9, 9, 9), dtype=torch.float32),
        torch.ones((1, 9, 9, 9), dtype=torch.bool),
        geometry,
        {},
        "",
        "",
    )

    class Value:
        architecture_hash = "synthetic-value-architecture"
        weights_hash = "synthetic-value-weights"

        def __call__(self, descriptors: torch.Tensor) -> torch.Tensor:
            return torch.ones(
                descriptors.shape[:2],
                dtype=descriptors.dtype,
                device=descriptors.device,
            )

    inputs = StageInputs(
        samples=(sample,),
        model=model,
        execution=StageExecutionConfig(
            config=config,
            frontend_sidecar={},
            normalization={},
            stage_options=StageOptions(stage="S6", engineering_only=True),
        ),
        target_provider=lambda _subject_id: torch.zeros((1, 9, 9, 9)),
        metadata={"value_model": Value()},
    )
    result = run_evaluation(
        inputs,
        ExperimentOptions(
            scenario="parallel_topk",
            budget=2,
            max_subjects=1,
            engineering_only=True,
            ssim_window=3,
        ),
        tmp_path / "parallel",
    )
    assert result["action_count"] == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "parallel" / "action_metrics.jsonl").read_text().splitlines()
        if line
    ]
    measured = [row for row in rows if row["role"] != "parallel_interaction"]
    assert measured
    assert all(row["diagnostic_scope"] == "parallel_initial_state" for row in measured)
    assert all(row["state_version"] == 0 for row in measured)
    metrics = json.loads((tmp_path / "parallel" / "metrics.json").read_text())
    assert metrics["actions"]["action_count"] == 2
    assert metrics["actions"]["measured_denominator"] == 2
    assert metrics["parallel_interactions"]["count"] == 1


def test_actual_two_subject_services_keep_subject_initialization_identity(tmp_path) -> None:
    """A typed model run must retain per-subject, not first-subject, init IDs."""

    frontend = PointGuidedConfig(
        num_semantic_classes=3,
        num_points=4,
        point_candidate_multiplier=3,
        offset_hidden_channels=12,
        detach_backbone_features=False,
    )
    config = PFGRLiteConfig(
        num_points=4,
        engineering_only=True,
        policy=PFGRPolicyConfig(mode="random"),
    )
    model = PFGRLiteModel(config, frontend_config=frontend).eval()
    geometry = VolumeGeometry.from_spacing((9, 9, 9))
    samples = tuple(
        TargetFreeSample(
            f"typed-{index}",
            torch.randn((3, 9, 9, 9), dtype=torch.float32),
            torch.ones((1, 9, 9, 9), dtype=torch.bool),
            geometry,
            {},
            "",
            "",
        )
        for index in range(2)
    )
    execution = StageExecutionConfig(
        config=config,
        frontend_sidecar={},
        normalization={},
        stage_options=StageOptions(stage="S6", engineering_only=True),
    )
    inputs = StageInputs(
        samples=samples,
        model=model,
        execution=execution,
        target_provider=lambda _subject_id: torch.zeros((1, 9, 9, 9)),
    )
    evaluation = run_evaluation(
        inputs,
        ExperimentOptions(
            scenario="random",
            budget=1,
            max_subjects=2,
            engineering_only=True,
            ssim_window=3,
        ),
        tmp_path / "typed-evaluation",
    )
    assert evaluation["software_status"] == "SOFTWARE_PASS"
    init_map = evaluation["source_receipt"]["subject_initialization_hashes"]
    assert set(init_map) == {"typed-0", "typed-1"}
    assert init_map["typed-0"] != init_map["typed-1"]

    oracle = run_oracle_evaluation(
        inputs,
        OracleOptions(
            mode="sampled_one",
            budget=1,
            candidate_count=2,
            max_subjects=2,
            engineering_only=True,
        ),
        tmp_path / "typed-oracle",
    )
    assert oracle["software_status"] == "SOFTWARE_PASS"
    assert oracle["subject_count"] == 2
    oracle_receipt = oracle["source_receipt"]
    assert oracle_receipt["subject_initialization_hashes"] == init_map
    paired = compare_paired_artifacts(None, evaluation, oracle)
    assert paired["subject_count"] == 2
    assert paired["r4_decision"]["branch"] == "INCONCLUSIVE"
