from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from smagm.features.point_guided import PointGuidedConfig
from smagm.features.point_guided.contracts import VolumeGeometry
from smagm.features.point_guided.pfgr_lite.config import (
    PFGRLiteConfig,
    PFGRPolicyConfig,
)
from smagm.features.point_guided.pfgr_lite.data import TargetFreeSample
from smagm.features.point_guided.pfgr_lite.experiments import (
    ExperimentOptions,
    _build_lattice,
    _load_policy,
    _selected_actions,
    run_evaluation,
)
from smagm.features.point_guided.pfgr_lite.inference import run_pfgr_inference
from smagm.features.point_guided.pfgr_lite.metrics import dense_metrics
from smagm.features.point_guided.pfgr_lite.model import PFGRLiteModel
from smagm.features.point_guided.pfgr_lite.oracle import OracleContext
from smagm.features.point_guided.pfgr_lite.provenance import tensor_digest
from smagm.features.point_guided.pfgr_lite.sparse_write import (
    make_action_writer,
    make_point_query,
    make_support_legal_mask,
)
from smagm.features.point_guided.pfgr_lite.stages import StageInputs


@pytest.fixture()
def actual_fixture() -> dict[str, object]:
    """One bounded N=4 FP32 subject through the actual PFGR model seams."""

    torch.manual_seed(19)
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
    observations = torch.randn(3, 9, 9, 9, dtype=torch.float32)
    brain_mask = torch.ones(1, 9, 9, 9, dtype=torch.bool)
    sample = TargetFreeSample(
        "acceptance-subject",
        observations,
        brain_mask,
        geometry,
        {},
        "",
        "",
    )
    inputs = StageInputs(
        samples=(sample,),
        model=model,
        config=config,
        target_provider=lambda _subject_id: torch.zeros(1, 9, 9, 9, dtype=torch.float32),
        metadata={"source_receipt": {"engineering_only": True}},
    )
    context = model.encode_observations(observations.unsqueeze(0), brain_mask, geometry)
    options = ExperimentOptions(
        scenario="random",
        budget=2,
        max_subjects=1,
        seed=23,
        split_role="validation",
        teacher_mode="exact_footprint",
        engineering_only=True,
    )
    lattice = _build_lattice(inputs, context, model, config)
    assert lattice is not None
    policy = _load_policy(inputs, context, options, config)
    assert policy is not None
    query = make_point_query()
    writer = make_action_writer(lattice)
    legal_mask = make_support_legal_mask(lattice)
    direct_route = run_pfgr_inference(
        model,
        context,
        policy,
        query=query,
        writer=writer,
        legal_mask=legal_mask,
    )
    direct_prediction = model.decode_final(
        direct_route.final_state,
        context,
        chunk_size=config.decode_chunk_size,
    )
    return {
        "config": config,
        "model": model,
        "sample": sample,
        "inputs": inputs,
        "context": context,
        "options": options,
        "direct_route": direct_route,
        "direct_prediction": direct_prediction,
    }


def _run_public(
    fixture: dict[str, object],
    output_dir: Path,
    *,
    split_role: str,
    target: object,
) -> dict[str, object]:
    inputs = fixture["inputs"]
    assert isinstance(inputs, StageInputs)
    # Keep the same observation/model/checkpoint and vary only the late target
    # provider.  Replacing the provider must not alter the target-free route.
    inputs = StageInputs(
        samples=inputs.samples,
        model=inputs.model,
        config=inputs.config,
        target_provider=lambda _subject_id: target.clone() if isinstance(target, torch.Tensor) else target,
        metadata=inputs.metadata,
    )
    options = fixture["options"]
    assert isinstance(options, ExperimentOptions)
    options = ExperimentOptions.from_dict({**options.as_dict(), "split_role": split_role})
    model = inputs.model
    assert isinstance(model, PFGRLiteModel)
    original_decode = model.decode_final
    captured_final: list[torch.Tensor] = []

    def capture_decode(state: object, context: object, *, chunk_size: int) -> torch.Tensor:
        decoded = original_decode(state, context, chunk_size=chunk_size)
        captured_final.append(decoded.detach().clone())
        return decoded

    # This wrapper forwards the real canonical decode; it records the actual
    # service tensor instead of inferring parity from a scalar loss.
    model.decode_final = capture_decode  # type: ignore[method-assign]
    try:
        run_evaluation(inputs, options, output_dir)
    finally:
        model.decode_final = original_decode  # type: ignore[method-assign]
    row = json.loads((output_dir / "paired_subjects.jsonl").read_text(encoding="utf-8"))
    assert captured_final
    row["_captured_final"] = captured_final[-1]
    return row


def test_public_validation_route_matches_direct_actual_decode(actual_fixture: dict[str, object], tmp_path: Path) -> None:
    """The public service and direct W4 route share actual FP32 K2 actions/prediction."""

    direct_route = actual_fixture["direct_route"]
    direct_prediction = actual_fixture["direct_prediction"]
    context = actual_fixture["context"]
    config = actual_fixture["config"]
    assert isinstance(direct_route, object)
    assert isinstance(direct_prediction, torch.Tensor)
    assert isinstance(context, object)
    assert isinstance(config, PFGRLiteConfig)

    row = _run_public(
        actual_fixture,
        tmp_path / "validation",
        split_role="validation",
        target=torch.zeros(1, 9, 9, 9, dtype=torch.float32),
    )
    assert row["selected_point_ids"]
    assert len(row["selected_point_ids"]) <= 2
    assert row["z0_digest"]
    initial_trace = direct_route.completed_trace  # type: ignore[union-attr]
    assert initial_trace is not None
    initial_prediction = actual_fixture["model"].decode_final(  # type: ignore[union-attr]
        initial_trace.states[0],
        context,  # type: ignore[arg-type]
        chunk_size=config.decode_chunk_size,
    )
    assert row["z0_digest"] == tensor_digest(initial_prediction.detach(), name="z0_prediction")
    direct_actions = _selected_actions(direct_route)
    assert tuple(row["selected_point_ids"]) == tuple(action.point_id for action in direct_actions)
    public_actions = [
        json.loads(line)["action_id"]
        for line in (tmp_path / "validation" / "action_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert tuple(public_actions) == tuple(direct_route.executed_action_ids)
    expected_after = dense_metrics(
        direct_prediction,
        torch.zeros(1, 9, 9, 9, dtype=torch.float32),
    )
    assert row["after"]["masked_charbonnier"] == pytest.approx(  # type: ignore[index]
        expected_after["masked_charbonnier"], abs=1e-6, rel=1e-5
    )
    assert torch.equal(row["_captured_final"], direct_prediction)


def test_validation_and_test_routes_match_and_target_replacement_cannot_change_route(
    actual_fixture: dict[str, object], tmp_path: Path
) -> None:
    zeros = torch.zeros(1, 9, 9, 9, dtype=torch.float32)
    ones = torch.ones_like(zeros)
    validation = _run_public(actual_fixture, tmp_path / "validation", split_role="validation", target=zeros)
    test = _run_public(actual_fixture, tmp_path / "test", split_role="test", target=zeros)
    replaced = _run_public(
        actual_fixture,
        tmp_path / "replaced",
        split_role="validation",
        target={
            "target": ones,
            "segmentation": torch.zeros(9, 9, 9, dtype=torch.int64),
        },
    )

    assert validation["z0_digest"] == test["z0_digest"] == replaced["z0_digest"]
    assert validation["selected_point_ids"] == test["selected_point_ids"] == replaced["selected_point_ids"]
    assert torch.equal(validation["_captured_final"], test["_captured_final"])
    assert torch.equal(validation["_captured_final"], replaced["_captured_final"])
    assert validation["after"] == test["after"]
    assert validation["after"] != replaced["after"]


def test_public_inference_rejects_privileged_oracle_context() -> None:
    observation = type("ObservationMarker", (), {"context_id": "observation-context"})()
    target = type("TargetMarker", (), {"context_id": "target-context"})()
    route = type("RouteMarker", (), {"route_hash": "sealed-route"})()
    oracle_context = OracleContext(observation, target, route, "candidate_subset")

    with pytest.raises(TypeError, match="ObservationContext"):
        run_pfgr_inference(
            object(),
            oracle_context,
            None,  # type: ignore[arg-type]
            query=lambda *_args, **_kwargs: None,
            writer=lambda *_args, **_kwargs: None,
        )


def test_nonengineering_public_evaluation_rejects_callback_route(tmp_path: Path) -> None:
    """Production evaluation must use the sealed W4 route, not a fixture callback."""

    inputs = StageInputs(
        samples=(
            TargetFreeSample(
                "boundary-subject",
                torch.zeros(3, 2, 2, 2),
                torch.ones(1, 2, 2, 2, dtype=torch.bool),
                VolumeGeometry.from_spacing((2, 2, 2)),
                {},
                "",
                "",
            ),
        ),
        route_builder=lambda *_args, **_kwargs: None,
    )
    with pytest.raises(ValueError, match="concrete W4 run_pfgr_inference route"):
        run_evaluation(
            inputs,
            ExperimentOptions(scenario="static", budget=0, engineering_only=False),
            tmp_path / "production-route-rejection",
        )
