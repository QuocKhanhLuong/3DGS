"""CPU software evidence for the optional retained-pool R4B diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from smagm.features.point_guided.contracts import VolumeGeometry
from smagm.features.point_guided.pfgr_lite.data import TargetFreeSample
from smagm.features.point_guided.pfgr_lite.headroom import (
    HeadroomOptions,
    _DiagnosticTiming,
    _exact_pool_audit_record,
    run_headroom_evaluation,
)
from smagm.features.point_guided.pfgr_lite.provenance import canonical_digest


@pytest.mark.parametrize("value", [None, 0, 1, "false", "true", [], {}])
def test_exact_pool_audit_requires_real_boolean(value):
    with pytest.raises(TypeError, match="exact_pool_audit must be bool"):
        HeadroomOptions(exact_pool_audit=value)


def test_default_off_options_keep_legacy_serialization_and_hash():
    legacy = {
        "schema_version": "pfgr-lite-headroom-options-v1",
        "max_subjects": 4,
        "random_seeds": [17, 29, 41],
        "candidate_count": 32,
        "query_count": 1024,
        "practical_margin": 0.0,
        "split_role": "validation",
        "engineering_only": False,
    }
    for options in (
        HeadroomOptions(),
        HeadroomOptions(exact_pool_audit=False),
        HeadroomOptions(**legacy),
    ):
        assert options.as_dict() == legacy
        assert canonical_digest(options.as_dict()) == canonical_digest(legacy)
    assert HeadroomOptions(exact_pool_audit=True).as_dict() == {
        **legacy,
        "exact_pool_audit": True,
    }


@pytest.fixture
def headroom_fixture(monkeypatch):
    from smagm.features.point_guided.pfgr_lite import experiments, metrics, oracle

    events = []
    marker = {"value": 1.0, "corruption": None}
    geometry = VolumeGeometry.from_spacing((2, 2, 2))
    samples = tuple(
        TargetFreeSample(
            f"subject-{index}",
            torch.zeros(3, 2, 2, 2),
            torch.ones(1, 2, 2, 2, dtype=torch.bool),
            geometry,
            {},
            "",
            "",
        )
        for index in range(4)
    )
    model = torch.nn.Linear(1, 1)
    context = SimpleNamespace(
        context_id="context", producer=SimpleNamespace(compatibility_hash="producer")
    )
    initial = torch.zeros(1, 1, 2, 2, 2)
    state = {"state_digest": "frozen-Z0"}

    def context_for(*args):
        events.append(("context",))
        return context

    def proposals(*args, **kwargs):
        events.append(("proposals",))
        return [
            {
                "action_id": f"action-{index}",
                "point_id": index,
                "legal": True,
                "point_ras_mm": [float(index), 0.0, 1.0],
                "action_digest": f"action-digest-{index}",
                "delta_hash": f"delta-{index}",
            }
            for index in range(32)
        ]

    def advance(_inputs, incoming, _context, _proposal, action, _decision, **kwargs):
        assert incoming == state
        assert kwargs.get("target_context") is None
        events.append(("apply", action["point_id"]))
        return torch.full_like(initial, action["point_id"] + 1.0)

    def target_provider(*args):
        events.append(("target", marker["value"]))
        return torch.full((1, 2, 2, 2), marker["value"])

    def measure(
        _inputs, _route, candidates, _target, _context, _lattice, options, **kwargs
    ):
        assert kwargs["diagnostic_state"] is state
        assert state == {"state_digest": "frozen-Z0"}
        is_audit = options.teacher_mode == "exact_footprint" and len(candidates) == 32
        events.append(("measure", options.teacher_mode, len(candidates)))
        rows = []
        for action in candidates:
            index = action["point_id"]
            # Q screening selects 0, while the true best is 1. A sign reversal
            # at 2 and ties among the other candidates make ranking auditable.
            raw = (
                ([10.0, 9.0, -1.0][index] if index < 3 else 0.0)
                if options.teacher_mode == "iid_fixed_q"
                else ([1.0, 3.0, 2.0][index] if index < 3 else 0.0)
            )
            rows.append(
                {
                    "action_id": action["action_id"],
                    "raw_gain": raw * marker["value"],
                    "action_digest": action["action_digest"],
                    "delta_hash": action["delta_hash"],
                    "state_digest": "frozen-Z0",
                    "target_context_digest": f"target-{marker['value']}",
                    "q_draws": 1024 if options.teacher_mode == "iid_fixed_q" else 0,
                    "footprint_voxels": index + 2,
                }
            )
        if is_audit and marker["corruption"]:
            kind = marker["corruption"]
            if kind == "prediction":
                initial.add_(1)
            elif kind == "missing":
                rows[-1].pop("raw_gain")
            elif kind == "gain":
                rows[0]["raw_gain"] += 0.001
            else:
                rows[0][kind] = "corrupted"
        return rows

    def paired(initial_value, final, target, mask, **kwargs):
        events.append(("metric", kwargs["scenario"]))
        return {
            "true_gain": float(final.mean()) * float(target.mean())
            if not torch.equal(initial_value, final)
            else 0.0
        }

    monkeypatch.setattr(experiments, "_context_for_sample", context_for)
    monkeypatch.setattr(experiments, "_build_lattice", lambda *args: None)
    monkeypatch.setattr(
        experiments, "_load_policy", lambda *args: SimpleNamespace(policy_hash="policy")
    )
    monkeypatch.setattr(
        experiments,
        "_route_for_sample",
        lambda *args: ({"initial_state": state}, None, None),
    )
    monkeypatch.setattr(experiments, "_prediction_for", lambda *args, **kwargs: initial)
    monkeypatch.setattr(oracle, "_oracle_proposals", proposals)
    monkeypatch.setattr(oracle, "_oracle_advance", advance)
    monkeypatch.setattr(oracle, "_measure_candidates", measure)
    monkeypatch.setattr(metrics, "paired_subject_metrics", paired)
    inputs = SimpleNamespace(
        samples=samples,
        model=model,
        target_provider=target_provider,
        metadata={},
        role_manifest=None,
    )

    def run(destination: Path, enabled: bool):
        events.clear()
        result = run_headroom_evaluation(
            inputs,
            HeadroomOptions(engineering_only=True, exact_pool_audit=enabled),
            destination,
        )
        evidence = json.loads(Path(result["evidence_path"]).read_text())
        return evidence, result["headroom_decision"], list(events)

    return run, marker


def test_noisy_winner_audit_reports_regret_without_changing_original_path(
    headroom_fixture, tmp_path
):
    run, _ = headroom_fixture
    off, off_decision, off_events = run(tmp_path / "off", False)
    on, on_decision, on_events = run(tmp_path / "on", True)
    assert off["candidate_pool"] == on["candidate_pool"]
    assert off["candidate_pool_hash"] == on["candidate_pool_hash"]
    assert off["dense_metrics"] == on["dense_metrics"]
    assert [event for event in on_events if event[0] == "apply"] == [
        event for event in off_events if event[0] == "apply"
    ]
    assert [event for event in off_events if event[0] == "measure"] == [
        ("measure", "iid_fixed_q", 32),
        ("measure", "exact_footprint", 1),
    ] * 4
    assert [event for event in on_events if event[0] == "measure"] == [
        ("measure", "iid_fixed_q", 32),
        ("measure", "exact_footprint", 1),
        ("measure", "exact_footprint", 32),
    ] * 4
    identity_envelope = {
        "options_hash",
        "evidence_artifact_hash",
        "evidence_artifact_path",
        "digest",
        "created_at",
    }
    assert {k: v for k, v in off_decision.items() if k not in identity_envelope} == {
        k: v for k, v in on_decision.items() if k not in identity_envelope
    }
    assert on_decision["decision"] == "INCONCLUSIVE"
    assert on_decision["human_reviewed"] is False
    assert on["target_join_calls"] == off["target_join_calls"] == 4
    assert (
        on["frozen_model_digest_before"]
        == on["frozen_model_digest_after"]
        == off["frozen_model_digest_before"]
    )
    for old, current in zip(off["subjects"], on["subjects"]):
        for key in (
            "oracle",
            "confirmation",
            "screening",
            "no_op",
            "random_controls",
            "z0_digest",
            "proposal_freeze",
        ):
            assert old[key] == current[key]
        audit = current["exact_pool_audit"]
        assert audit["exact_best_action_ids"] == ["action-1"]
        assert audit["sampled_selected_action_id"] == "action-0"
        assert audit["exact_best_gain"] == 3.0
        assert audit["sampled_selected_exact_gain"] == 1.0
        assert audit["confirmation_audit_gain_delta"] == 0.0
        assert audit["confirmation_audit_gain_consistent"] is True
        assert audit["top1_regret"] == 2.0
        assert audit["sampled_selected_is_exact_best"] is False
        assert (
            audit["rows"][0]["exact_rank_min"]
            == audit["rows"][0]["exact_rank_max"]
            == 3
        )
        assert (
            audit["sign_agreement"]["counts_sampled_then_exact"]["negative"]["positive"]
            == 1
        )
        assert audit["ranking"]["discordant_pairs"] > 0
        assert audit["pool_coverage"]["finite_exact_candidate_count"] == 32
        assert audit["freeze"]["sealed_values_unchanged"] is True
        assert old["diagnostic_timing"]["phases"]["exact_pool_audit"]["seconds"] is None
        phases = current["diagnostic_timing"]["phases"]
        assert phases["screening"]["query_voxels"] == 32 * 1024
        assert phases["exact_confirmation"]["query_voxels"] == 2
        assert phases["exact_pool_audit"]["query_voxels"] == sum(range(2, 34))
        assert phases["dense_metrics"]["paired_metric_calls"] == 5
        assert phases["dense_metrics"]["dense_metric_evaluations"] == 10
        assert all(row["seconds"] >= 0 for row in phases.values())
        assert current["diagnostic_timing"]["device"] == "cpu"


def test_exact_audit_disagreement_with_same_winner_confirmation_fails(headroom_fixture, tmp_path):
    run, marker = headroom_fixture
    marker["corruption"] = "gain"
    with pytest.raises(ValueError, match="disagrees with the retained winner confirmation"):
        run(tmp_path / "inconsistent-exact", True)


def test_teacher_identity_source_does_not_claim_positional_binding_is_declared():
    from smagm.features.point_guided.pfgr_lite.headroom import _bind_measured_rows

    candidate = {"action_id": "action-0", "point_ras_mm": [0., 0., 0.]}
    positional = _bind_measured_rows([{"raw_gain": 1.0}], [candidate], scope="test", seed=0)
    declared = _bind_measured_rows([{"raw_gain": 1.0, "action_id": "action-0"}], [candidate], scope="test", seed=0)
    assert positional[0]["action_identity_source"] == "positional_binding"
    assert declared[0]["action_identity_source"] == "teacher_declared"


def test_target_substitution_cannot_change_frozen_pool_or_apply_inputs(
    headroom_fixture, tmp_path
):
    run, marker = headroom_fixture
    first, _, _ = run(tmp_path / "first", True)
    marker["value"] = 2.0
    second, _, events = run(tmp_path / "second", True)
    assert first["candidate_pool_hash"] == second["candidate_pool_hash"]
    assert first["subjects"][0]["z0_digest"] == second["subjects"][0]["z0_digest"]
    assert first["subjects"][0]["exact_pool_audit"]["top1_regret"] == 2
    assert second["subjects"][0]["exact_pool_audit"]["top1_regret"] == 4
    first_target = next(
        index for index, event in enumerate(events) if event[0] == "target"
    )
    assert sum(event[0] == "apply" for event in events[:first_target]) == 3
    assert sum(event[0] == "proposals" for event in events[:first_target]) == 1
    assert all(event[0] != "measure" for event in events[:first_target])


@pytest.mark.parametrize(
    "corruption",
    [
        "action_digest",
        "delta_hash",
        "state_digest",
        "target_context_digest",
        "prediction",
    ],
)
def test_audit_rejects_broken_sealed_identity(headroom_fixture, tmp_path, corruption):
    run, marker = headroom_fixture
    marker["corruption"] = corruption
    with pytest.raises(ValueError, match="exact pool audit"):
        run(tmp_path / corruption, True)


def test_missing_audit_gain_does_not_report_pool_best_or_regret(
    headroom_fixture, tmp_path
):
    run, marker = headroom_fixture
    marker["corruption"] = "missing"
    evidence, decision, _ = run(tmp_path / "missing", True)
    audit = evidence["subjects"][0]["exact_pool_audit"]
    assert audit["status"] == "incomplete"
    assert audit["exact_best_gain"] is audit["top1_regret"] is None
    assert audit["observed_best_gain"] == 3.0
    assert audit["pool_coverage"]["finite_exact_candidate_count"] == 31
    assert decision["oracle_gain"] == 1.0


@pytest.mark.parametrize("exact", [[2.0, 2.0, 0.0], [None, None, None]])
def test_audit_ties_and_unavailable_ranking_are_honest(exact):
    candidates = [{"action_id": f"a-{i}"} for i in range(3)]
    rows = [
        {"raw_gain": value, "action_id": f"a-{index}"}
        for index, value in enumerate(exact)
    ]
    result = _exact_pool_audit_record(
        rows,
        [{"raw_gain": 1.0}] * 3,
        candidates,
        winner_index=0,
        proposal_count=64,
        legal_count=60,
        state_digest=None,
    )
    assert result["pool_coverage"]["retained_fraction_of_legal"] == 3 / 60
    assert result["pool_coverage"]["all_legal_candidates_measured"] is False
    assert result["ranking"]["pairwise_order_agreement"] is None
    if exact[0] is not None:
        assert result["exact_best_action_ids"] == ["a-0", "a-1"]
        assert result["top1_regret"] == 0.0
        assert rows[0]["exact_rank_min"] == 1 and rows[0]["exact_rank_max"] == 2
    else:
        assert result["observed_best_gain"] is result["top1_regret"] is None
        assert result["sign_agreement"]["fraction"] is None


def test_timing_null_cpu_and_missing_teacher_counts():
    timer = _DiagnosticTiming(SimpleNamespace(device="cuda:0"))
    with timer.measure("context_lattice"):
        pass
    assert timer.phases["context_lattice"]["seconds"] is None
    assert timer.phases["context_lattice"]["wall_seconds"] >= 0
    assert timer.as_dict()["device"] is None
    timer.observe(torch.zeros(1))
    with timer.measure("dense_metrics"):
        pass
    assert timer.phases["dense_metrics"]["synchronization"] == "cpu_synchronous"
    timer.teacher_counts("screening", [{"raw_gain": 1.0}], exact=False)
    assert timer.phases["screening"]["query_voxels"] is None
    assert timer.phases["exact_pool_audit"]["calls"] == 0
    assert timer.phases["exact_pool_audit"]["seconds"] is None


def test_cuda_timing_synchronizes_actual_observed_device_not_config(monkeypatch):
    # CPU-only simulation of the CUDA synchronization protocol, not GPU evidence.
    class ReportedCudaTensor(torch.Tensor):
        @property
        def device(self):
            return torch.device("cuda:3")

    tensor = torch.zeros(1).as_subclass(ReportedCudaTensor)
    timer = _DiagnosticTiming(SimpleNamespace(device="cpu", observations=tensor))
    syncs = []
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: syncs.append(str(device))
    )
    with timer.measure("screening", candidate_evaluations=32):
        assert syncs == ["cuda:3"]
    assert syncs == ["cuda:3", "cuda:3"]
    assert timer.phases["screening"]["seconds"] >= 0

    def broken(device):
        raise RuntimeError("sync unavailable")

    monkeypatch.setattr(torch.cuda, "synchronize", broken)
    with timer.measure("exact_confirmation"):
        pass
    assert timer.phases["exact_confirmation"]["seconds"] is None
    assert (
        timer.phases["exact_confirmation"]["synchronization"]
        == "unavailable_synchronization_failed"
    )


def test_device_discovered_mid_phase_does_not_claim_synchronized_time(monkeypatch):
    timer = _DiagnosticTiming(torch.zeros(1))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    with timer.measure("context_lattice"):
        timer.devices.add("cuda:1")
    assert timer.phases["context_lattice"]["seconds"] is None
    assert (
        timer.phases["context_lattice"]["synchronization"]
        == "unavailable_device_changed_during_phase"
    )
