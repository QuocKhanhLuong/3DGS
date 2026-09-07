from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from smagm.features.point_guided import PointGuidedConfig
from smagm.features.point_guided.contracts import VolumeGeometry
from smagm.features.point_guided.pfgr_lite.benchmark import (
    BenchmarkOptions,
    run_teacher_benchmark,
)
from smagm.features.point_guided.pfgr_lite.config import (
    PFGRLiteConfig,
    PFGRPolicyConfig,
)
from smagm.features.point_guided.pfgr_lite.data import TargetFreeSample
from smagm.features.point_guided.pfgr_lite.model import PFGRLiteModel
from smagm.features.point_guided.pfgr_lite.sparse_write import build_footprint
from smagm.features.point_guided.pfgr_lite.stages import (
    StageExecutionConfig,
    StageInputs,
    StageOptions,
)
from tests.features.point_guided.pfgr_lite.test_teacher import _fixture


def _sample() -> TargetFreeSample:
    geometry = VolumeGeometry.from_spacing((5, 5, 5))
    return TargetFreeSample(
        "benchmark-subject",
        torch.zeros((3, 5, 5, 5), dtype=torch.float32),
        torch.ones((1, 5, 5, 5), dtype=torch.bool),
        geometry,
        {},
        "",
        "",
    )


def test_benchmark_options_require_bounded_repeats() -> None:
    assert BenchmarkOptions.from_dict({"repeats": 3}).repeats == 3
    with pytest.raises(ValueError, match="three"):
        BenchmarkOptions(repeats=2)
    with pytest.raises(ValueError, match="unknown BenchmarkOptions"):
        BenchmarkOptions.from_dict({"unknown": 1})


def test_same_work_reference_sparse_parity_and_actual_rows(tmp_path) -> None:
    _, _, lattice, state, action, _, decoder = _fixture(torch.float64)
    decoder.train()
    footprint = build_footprint(lattice, action, chunk_size=11)
    model = SimpleNamespace(decoder=decoder)
    inputs = StageInputs(
        samples=(_sample(),),
        model=model,
        route_builder=lambda _sample, **_kwargs: {"k": 0, "states": (state,), "final_prediction": torch.zeros(1, 1, 5, 5, 5)},
        target_provider=lambda _subject_id: torch.zeros((1, 5, 5, 5), dtype=torch.float64),
        metadata={"lattice": lattice, "benchmark_cases": ({"state": state, "action": action, "voxel_ids_dhw": footprint.voxel_ids_dhw},)},
    )
    result = run_teacher_benchmark(
        inputs,
        BenchmarkOptions(dtype="float64", repeats=3, chunk_size=7, engineering_only=True),
        tmp_path / "benchmark",
    )
    assert result["software_status"] == "SOFTWARE_PASS"
    assert result["row_count"] == 3
    assert result["case_count"] == 1
    assert (tmp_path / "benchmark" / "benchmark.json").exists()
    assert (tmp_path / "benchmark" / "rows.jsonl").read_text().count("\n") == 3
    assert (tmp_path / "benchmark" / "parity.json").read_text().find("PASS") >= 0
    rows = [
        json.loads(line)
        for line in (tmp_path / "benchmark" / "rows.jsonl").read_text().splitlines()
    ]
    assert all(row["sampling_law"] == "iid_fixed_q_plane_mixture_c_over_S_v1" for row in rows)
    assert all(row["query_draws"] == 64 for row in rows)
    assert all(row["gain_error_max"] is not None and row["parity_failure"] is None for row in rows)
    assert decoder.training is True
    assert all(parameter.grad is None for parameter in decoder.parameters())


def test_actual_two_subject_benchmark_retains_global_and_subject_receipts(tmp_path) -> None:
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
            f"benchmark-typed-{index}",
            torch.randn((3, 9, 9, 9), dtype=torch.float32),
            torch.ones((1, 9, 9, 9), dtype=torch.bool),
            geometry,
            {},
            "",
            "",
        )
        for index in range(2)
    )
    config_wrapper = StageExecutionConfig(
        config=config,
        frontend_sidecar={},
        normalization={},
        stage_options=StageOptions(stage="S6", engineering_only=True),
    )
    inputs = StageInputs(
        samples=samples,
        model=model,
        execution=config_wrapper,
        target_provider=lambda _subject_id: torch.zeros((1, 9, 9, 9)),
    )
    result = run_teacher_benchmark(
        inputs,
        BenchmarkOptions(
            max_subjects=2,
            max_states=1,
            candidate_count=1,
            repeats=3,
            query_count=4,
            engineering_only=True,
        ),
        tmp_path / "typed-benchmark",
    )
    assert result["software_status"] == "SOFTWARE_PASS"
    assert result["row_count"] == 6
    receipt = result["source_receipt"]
    assert receipt["producer_compatibility_hash"]
    assert receipt["normalization_hash"]
    assert set(receipt["subject_initialization_hashes"]) == {
        "benchmark-typed-0",
        "benchmark-typed-1",
    }
    assert len(set(receipt["subject_initialization_hashes"].values())) == 2
