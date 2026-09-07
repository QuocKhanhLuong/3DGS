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
from smagm.features.point_guided.pfgr_lite.data import TargetFreeSample
from smagm.features.point_guided.pfgr_lite.experiments import (
    ExperimentOptions,
    _build_lattice,
    _context_for_sample,
    _load_policy,
    _route_for_sample,
)
from smagm.features.point_guided.pfgr_lite.model import PFGRLiteModel
from smagm.features.point_guided.pfgr_lite.oracle import (
    OracleContext,
    OracleOptions,
    OracleResult,
    _oracle_proposals,
    run_oracle_evaluation,
)
from smagm.features.point_guided.pfgr_lite.sparse_write import reference_full_write
from smagm.features.point_guided.pfgr_lite.stages import (
    StageExecutionConfig,
    StageInputs,
    StageOptions,
)


def _sample() -> TargetFreeSample:
    geometry = VolumeGeometry.from_spacing((3, 3, 3))
    return TargetFreeSample(
        "oracle-subject",
        torch.zeros((3, 3, 3, 3), dtype=torch.float32),
        torch.ones((1, 3, 3, 3), dtype=torch.bool),
        geometry,
        {},
        "",
        "",
    )


def test_oracle_options_and_result_are_strict() -> None:
    options = OracleOptions.from_dict({"oracle_mode": "all_exact_one", "budget": 1, "candidate_count": 3, "engineering_only": True})
    assert options.mode == "all_exact_one"
    assert options.oracle_mode == "all_exact_one"
    with pytest.raises(ValueError, match="unknown OracleOptions"):
        OracleOptions.from_dict({"target": True})
    with pytest.raises(ValueError, match="diagnostic"):
        OracleResult("s", "sampled_one", 1, "subset", target_aware=False)
    with pytest.raises(ValueError, match="all_exact_one requires exact"):
        OracleOptions(mode="all_exact_one", teacher_mode="iid_fixed_q", query_count=2)
    with pytest.raises(ValueError, match="independent confirmation"):
        OracleOptions(teacher_mode="iid_fixed_q", query_count=2)
    with pytest.raises(ValueError, match="conflicts"):
        OracleOptions.from_dict({"mode": "greedy", "oracle_mode": "sampled_one"})


def test_oracle_generates_candidates_before_target_and_keeps_scope(tmp_path) -> None:
    order: list[str] = []
    sample = _sample()

    def route(_sample, **_kwargs):
        order.append("route")
        return {"states": ("state-0",), "final_prediction": torch.zeros(1, 1, 3, 3, 3), "k": 0}

    def proposals(*_args, **kwargs):
        order.append("proposals")
        return [{"action_id": f"a-{i}"} for i in range(kwargs["candidate_count"])]

    def target_provider(_subject_id: str):
        order.append("target")
        return torch.zeros((1, 3, 3, 3), dtype=torch.float32)

    def effect(_route, candidates, _target, **_kwargs):
        order.append("measure")
        return [{"action_id": action["action_id"], "raw_gain": (0.3 if action["action_id"] == "a-1" else -0.1)} for action in candidates]

    inputs = StageInputs(
        samples=(sample,),
        route_builder=route,
        proposal_builder=proposals,
        target_provider=target_provider,
        effect_measure=effect,
    )
    result = run_oracle_evaluation(
        inputs,
        OracleOptions(mode="all_exact_one", budget=1, candidate_count=3, engineering_only=True),
        tmp_path / "oracle",
    )
    assert order == ["route", "proposals", "target", "measure"]
    assert result["software_status"] == "SOFTWARE_PASS"
    row = (tmp_path / "oracle" / "privileged_oracle.jsonl").read_text()
    assert "all_candidates" in row
    assert "a-1" in row
    assert "privileged" not in row.lower() or "diagnostic" in row.lower()


def test_oracle_context_cannot_alias_observation_and_target() -> None:
    marker = object()
    with pytest.raises(ValueError, match="separate"):
        OracleContext(marker, marker, object(), "subset")


def test_greedy_oracle_applies_winner_before_next_proposal_bank(tmp_path) -> None:
    order: list[tuple[str, str]] = []
    sample = _sample()

    def route(_sample, **_kwargs):
        return {"states": ("state-0",), "final_prediction": torch.zeros(1, 1, 3, 3, 3), "k": 0}

    def proposals(_route, state, **_kwargs):
        state_name = str(state)
        order.append(("proposals", state_name))
        if state_name == "state-0":
            return [{"action_id": "a0", "state": state_name}, {"action_id": "a1", "state": state_name}]
        return [{"action_id": "b0", "state": state_name}, {"action_id": "b1", "state": state_name}]

    def apply(state, action, **_kwargs):
        action_id = action["action_id"]
        order.append(("apply", action_id))
        return f"{state}->{action_id}"

    def target_provider(_subject_id: str):
        return torch.zeros((1, 3, 3, 3), dtype=torch.float32)

    def effect(_route, candidates, _target, **_kwargs):
        values = {"a0": 0.1, "a1": 0.5, "b0": 0.4, "b1": -0.2}
        return [{"action_id": item["action_id"], "raw_gain": values[item["action_id"]], "state": item["state"]} for item in candidates]

    inputs = StageInputs(
        samples=(sample,),
        route_builder=route,
        proposal_builder=proposals,
        target_provider=target_provider,
        effect_measure=effect,
        metadata={"oracle_apply": apply},
    )
    run_oracle_evaluation(
        inputs,
        OracleOptions(mode="greedy", budget=2, candidate_count=2, engineering_only=True),
        tmp_path / "greedy-oracle",
    )
    payload = (tmp_path / "greedy-oracle" / "privileged_oracle.jsonl").read_text()
    assert '"selected_action_ids": ["a1", "b0"]' in payload
    assert order == [("proposals", "state-0"), ("apply", "a1"), ("proposals", "state-0->a1"), ("apply", "b0")]


def test_typed_oracle_applies_measured_winner_and_decodes_final_state(tmp_path) -> None:
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
    model = PFGRLiteModel(config, frontend_config=frontend)
    model.train()
    geometry = VolumeGeometry.from_spacing((9, 9, 9))
    observations = torch.randn((3, 9, 9, 9), dtype=torch.float32)
    sample = TargetFreeSample(
        "typed-oracle-subject",
        observations,
        torch.ones((1, 9, 9, 9), dtype=torch.bool),
        geometry,
        {},
        "",
        "",
    )
    execution = StageExecutionConfig(
        config=config,
        frontend_sidecar={},
        normalization={},
        stage_options=StageOptions(stage="S6", engineering_only=True),
    )
    base_inputs = StageInputs(
        samples=(sample,),
        model=model,
        execution=execution,
        target_provider=lambda _subject_id: torch.zeros((1, 9, 9, 9), dtype=torch.float32),
    )
    context = _context_for_sample(base_inputs, sample)
    lattice = _build_lattice(base_inputs, context, model, config)
    options = OracleOptions(
        mode="greedy",
        budget=2,
        candidate_count=4,
        teacher_mode="exact_footprint",
        query_count=8,
        engineering_only=True,
    )
    experiment_options = ExperimentOptions(
        scenario="static",
        budget=0,
        max_subjects=1,
        seed=options.seed,
        split_role=options.split_role,
        teacher_mode=options.teacher_mode,
        query_count=options.query_count,
        engineering_only=True,
    )
    policy = _load_policy(base_inputs, context, experiment_options, config)
    route, query, writer = _route_for_sample(
        base_inputs,
        sample,
        context,
        experiment_options,
        config,
        lattice,
        policy,
    )
    initial_state = route.completed_trace.states[0]
    proposal = _oracle_proposals(
        base_inputs,
        context,
        initial_state,
        route,
        query,
        writer,
        lattice,
        options,
        state_index=0,
        policy=policy,
    )
    first_action = proposal.row(0, 0)
    after_state = initial_state.next(
        reference_full_write(lattice, initial_state.planes, first_action)
    )
    target = model.decode_final(after_state, context, chunk_size=1024).detach()
    inputs = StageInputs(
        samples=(sample,),
        model=model,
        execution=execution,
        target_provider=lambda _subject_id: target,
    )
    result = run_oracle_evaluation(inputs, options, tmp_path / "typed-oracle")
    payload = json.loads(
        (tmp_path / "typed-oracle" / "privileged_oracle.jsonl").read_text()
    )
    assert result["software_status"] == "SOFTWARE_PASS"
    assert payload["oracle_final_prediction_decoded"] is True
    assert payload["selected_action_ids"]
    assert payload["oracle_route_gain"] is not None
    assert any(row["state_index"] == 1 and row["state_version"] == 1 for row in payload["rows"])
    assert model.training is True
    assert all(parameter.grad is None for parameter in model.parameters())


def test_typed_sampled_oracle_uses_independent_fixed_q_confirmation(tmp_path) -> None:
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
    sample = TargetFreeSample(
        "iid-oracle-subject",
        torch.randn((3, 9, 9, 9), dtype=torch.float32),
        torch.ones((1, 9, 9, 9), dtype=torch.bool),
        geometry,
        {},
        "",
        "",
    )
    execution = StageExecutionConfig(
        config=config,
        frontend_sidecar={},
        normalization={},
        stage_options=StageOptions(stage="S6", engineering_only=True),
    )
    # Choose a deterministic target equal to one stored initial-state write so
    # at least one screened action has a positive fixed-Q gain and therefore
    # exercises the independent confirmation branch.
    probe_inputs = StageInputs(samples=(sample,), model=model, execution=execution)
    probe_options = ExperimentOptions(
        scenario="static",
        budget=0,
        max_subjects=1,
        engineering_only=True,
    )
    probe_context = _context_for_sample(probe_inputs, sample)
    probe_lattice = _build_lattice(probe_inputs, probe_context, model, config)
    probe_policy = _load_policy(probe_inputs, probe_context, probe_options, config)
    probe_route, probe_query, probe_writer = _route_for_sample(
        probe_inputs,
        sample,
        probe_context,
        probe_options,
        config,
        probe_lattice,
        probe_policy,
    )
    probe_state = probe_route.completed_trace.states[0]
    probe_proposal = _oracle_proposals(
        probe_inputs,
        probe_context,
        probe_state,
        probe_route,
        probe_query,
        probe_writer,
        probe_lattice,
        OracleOptions(mode="sampled_one", budget=1, candidate_count=12, engineering_only=True),
        state_index=0,
        policy=probe_policy,
    )
    probe_action = probe_proposal.row(0, 0)
    target = model.decode_final(
        probe_state.next(reference_full_write(probe_lattice, probe_state.planes, probe_action)),
        probe_context,
        chunk_size=1024,
    ).detach()
    inputs = StageInputs(
        samples=(sample,),
        model=model,
        execution=execution,
        target_provider=lambda _subject_id: target,
    )
    options = OracleOptions(
        mode="sampled_one",
        budget=1,
        candidate_count=12,
        teacher_mode="iid_fixed_q",
        query_count=4,
        confirmation_mode="iid_fixed_q",
        confirmation_query_count=4,
        engineering_only=True,
    )
    result = run_oracle_evaluation(inputs, options, tmp_path / "iid-oracle")
    payload = json.loads(
        (tmp_path / "iid-oracle" / "privileged_oracle.jsonl").read_text()
    )
    assert result["software_status"] == "SOFTWARE_PASS"
    assert payload["rows"]
    assert payload["rows"][0]["sampler_law"] == "iid_fixed_q_plane_mixture_c_over_S_v1"
    assert payload["rows"][0]["q_draws"] == 4
    assert payload["confirmation"]
    confirmation = payload["confirmation"][0]
    assert confirmation["confirmation_mode"] == "iid_fixed_q"
    assert confirmation["confirmation_q_draws"] == 4
    assert confirmation["confirmation_seed"] != payload["rows"][0]["seed"]
    assert "confirmation_discrepancy" in confirmation
