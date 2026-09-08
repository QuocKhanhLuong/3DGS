from __future__ import annotations

import hashlib
import json
from dataclasses import replace
import math
from pathlib import Path
import statistics
from types import SimpleNamespace

import pytest
import torch

from smagm.features.point_guided.contracts import VolumeGeometry
from smagm.features.point_guided.pfgr_lite.data import TargetFreeSample
from smagm.features.point_guided.pfgr_lite.headroom import (
    HEADROOM_DECISIONS,
    HeadroomOptions,
    HeadroomDecision,
    _normal_lower_bound,
    require_main_headroom_decision,
    run_headroom_evaluation,
    validate_headroom_decision,
)


def test_scientific_headroom_decisions_are_closed_to_user_contract() -> None:
    assert HEADROOM_DECISIONS == frozenset(
        {
            "HEADROOM_CONFIRMED",
            "CORRECTION_USEFUL_SELECTION_NOT_NEEDED",
            "NO_HEADROOM_OBSERVED",
            "INCONCLUSIVE",
        }
    )


def test_engineering_hydrated_model_cannot_enter_main_headroom_service(tmp_path: Path) -> None:
    inputs = type("Inputs", (), {"model": type("Model", (), {"_engineering_only": True})()})()
    with pytest.raises(ValueError, match="engineering-hydrated model"):
        run_headroom_evaluation(inputs, HeadroomOptions(engineering_only=False), tmp_path / "blocked")


def test_lower_level_headroom_options_allow_reviewed_larger_cohort() -> None:
    options = HeadroomOptions(max_subjects=32)
    assert options.max_subjects == 32


def test_next1_freezes_k0_pool_and_random_controls_before_target_substitution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R4B composes one frozen K0 pool, then joins exactly one substituted target."""

    import smagm.features.point_guided.pfgr_lite.experiments as experiments_module
    import smagm.features.point_guided.pfgr_lite.metrics as metrics_module
    import smagm.features.point_guided.pfgr_lite.oracle as oracle_module

    class ProposalBatch(list):
        proposal_digest = "proposal-batch-v1"

    geometry = VolumeGeometry.from_spacing((2, 2, 2))

    def make_sample(subject_id: str) -> TargetFreeSample:
        return TargetFreeSample(
            subject_id,
            torch.zeros(3, 2, 2, 2),
            torch.ones(1, 2, 2, 2, dtype=torch.bool),
            geometry,
            {},
            "",
            "",
        )

    sample = make_sample("next1-subject")
    context = SimpleNamespace(
        context_id="next1-context",
        producer=SimpleNamespace(compatibility_hash="next1-producer"),
    )
    events: list[tuple[str, object]] = []
    target_marker = {"value": 1.0}

    def context_for_sample(_inputs, _sample):
        events.append(("context", context.context_id))
        return context

    def build_lattice(*_args, **_kwargs):
        return SimpleNamespace()

    def load_policy(*_args, **_kwargs):
        return SimpleNamespace(policy_hash="next1-policy")

    def route_for_sample(*_args, **_kwargs):
        events.append(("route", "k0"))
        return (
            {
                "states": ({"state_digest": "k0"},),
                "initial_prediction": torch.zeros(1, 1, 2, 2, 2),
            },
            SimpleNamespace(),
            SimpleNamespace(),
        )

    def prediction_for(*_args, **_kwargs):
        return torch.zeros(1, 1, 2, 2, 2)

    def proposals(*_args, **_kwargs):
        events.append(("proposals", "before-target"))
        return ProposalBatch(
            {
                "action_id": f"next1-action-{index:02d}",
                "point_id": index,
                "point_ras_mm": [float(index), 1.0, 2.0],
                "delta_hash": f"next1-delta-{index:02d}",
                "action_digest": f"next1-digest-{index:02d}",
                "legal": True,
            }
            for index in range(32)
        )

    def advance(_inputs, _state, _context, _proposal, action, _decision, **kwargs):
        events.append(("advance", kwargs.get("target_context") is None))
        point_id = int(action.get("point_id", 0)) if isinstance(action, dict) else 0
        return torch.full((1, 1, 2, 2, 2), float(point_id + 1))

    def target_provider(_subject_id):
        events.append(("target", target_marker["value"]))
        return torch.full((1, 2, 2, 2), target_marker["value"])

    def measure(_inputs, _route, candidates, _target_context, *_args, **_kwargs):
        events.append(("measure", len(candidates)))
        marker = target_marker["value"]
        rows = []
        for action in candidates:
            point_id = int(action.get("point_id", 0)) if isinstance(action, dict) else 0
            # Candidate zero is the deterministic sealed winner in both target
            # runs; the target marker still changes its measured dense effect.
            gain = marker + (1.0 if point_id == 0 else 0.5)
            rows.append({"action_id": action["action_id"], "raw_gain": gain})
        return rows

    def paired_metrics(initial, final, target, _mask, **_kwargs):
        if torch.equal(initial, final):
            gain = 0.0
        else:
            gain = float(target.mean().item()) + float(final.mean().item())
        return {"true_gain": gain}

    monkeypatch.setattr(experiments_module, "_context_for_sample", context_for_sample)
    monkeypatch.setattr(experiments_module, "_build_lattice", build_lattice)
    monkeypatch.setattr(experiments_module, "_load_policy", load_policy)
    monkeypatch.setattr(experiments_module, "_route_for_sample", route_for_sample)
    monkeypatch.setattr(experiments_module, "_prediction_for", prediction_for)
    monkeypatch.setattr(oracle_module, "_oracle_proposals", proposals)
    monkeypatch.setattr(oracle_module, "_oracle_advance", advance)
    monkeypatch.setattr(oracle_module, "_measure_candidates", measure)
    monkeypatch.setattr(metrics_module, "paired_subject_metrics", paired_metrics)

    from smagm.features.point_guided.pfgr_lite.headroom import run_headroom_evaluation

    inputs = SimpleNamespace(
        samples=(sample,),
        model=None,
        target_provider=target_provider,
        metadata={},
        role_manifest=None,
    )

    # The bounded CLI uses four subjects; this single-subject engineering
    # fixture calls the same typed service with max_subjects=4 by repeating
    # the subject only in the explicit test harness, never in production.
    inputs.samples = tuple(make_sample(f"next1-subject-{index}") for index in range(4))

    def run(marker: float, destination: Path) -> dict[str, object]:
        target_marker["value"] = marker
        events.clear()
        result = run_headroom_evaluation(
            inputs,
            HeadroomOptions(max_subjects=4, engineering_only=True),
            destination,
        )
        evidence = json.loads(Path(result["evidence_path"]).read_text(encoding="utf-8"))
        return evidence

    first = run(1.0, tmp_path / "next1-a")
    second = run(2.0, tmp_path / "next1-b")
    assert first["subject_count"] == second["subject_count"] == 4
    assert first["candidate_pool"] == second["candidate_pool"]
    assert first["dense_metrics"] != second["dense_metrics"]
    for first_subject, second_subject in zip(first["subjects"], second["subjects"]):
        assert first_subject["z0_state_digest"] == second_subject["z0_state_digest"] == "k0"
        assert first_subject["proposal_freeze"]["before_target"] is True
        assert second_subject["proposal_freeze"]["before_target"] is True
        first_controls = [(row["seed"], row["candidate_index"], row["action_id"]) for row in first_subject["random_controls"]]
        second_controls = [(row["seed"], row["candidate_index"], row["action_id"]) for row in second_subject["random_controls"]]
        assert first_controls == second_controls
        assert [row["delta_hash"] for row in first_subject["candidate_pool"]] == [
            row["delta_hash"] for row in second_subject["candidate_pool"]
        ]
        assert first_subject["confirmation"]["rows"]
        assert second_subject["confirmation"]["rows"]
        assert "write_norms" in first_subject and "timing_seconds" in first_subject
    # Every random apply is target-free and precedes the first target callback;
    # the winner is applied later from the same sealed candidate identity.
    first_target = next(index for index, event in enumerate(events) if event[0] == "target")
    assert sum(event[0] == "advance" for event in events[:first_target]) == 3
    assert all(event[0] == "advance" and event[1] is True for event in events if event[0] == "advance")
from smagm.features.point_guided.pfgr_lite.provenance import canonical_digest


_IDENTITY_KEYS = (
    "producer_compatibility_hash",
    "source_manifest_hash",
    "base_checkpoint_hash",
    "updater_checkpoint_hash",
    "split_hash",
    "subject_set_hash",
    "teacher_identity_hash",
    "frozen_model_digest",
)


def _permit_fixture(tmp_path: Path) -> tuple[HeadroomDecision, dict[str, str], Path, Path]:
    options = {
        "schema_version": "pfgr-lite-headroom-options-v1",
        # This fixture models a later reviewed development cohort.  The
        # executable NEXT-1 CLI remains locked to four subjects; a positive
        # MAIN permit must instead bind the actual 32-subject evidence count.
        "max_subjects": 32,
        "random_seeds": [17, 29, 41],
        "candidate_count": 32,
        "query_count": 1024,
        "practical_margin": 0.05,
        "split_role": "validation",
        "engineering_only": False,
    }
    options_hash = canonical_digest(options, prefix="pfgr-lite-headroom-options-v1|")
    subject_ids = [f"validation-{index:02d}" for index in range(32)]
    subjects: list[dict[str, object]] = []
    aggregate_pool: list[dict[str, object]] = []
    winners: list[str] = []
    confirmations: list[str] = []
    for subject_index, subject_id in enumerate(subject_ids):
        candidates = [
            {
                "candidate_index": candidate_index,
                "proposal_position": candidate_index,
                "action_id": f"{subject_id}-action-{candidate_index:02d}",
                "point_id": candidate_index,
                "point_ras_mm": [float(candidate_index), 1.0, 2.0],
                "action_digest": f"digest-{subject_index:02d}-{candidate_index:02d}",
                "delta_hash": f"delta-{subject_index:02d}-{candidate_index:02d}",
                "legal": True,
            }
            for candidate_index in range(32)
        ]
        aggregate_pool.extend({"subject_id": subject_id, **candidate} for candidate in candidates)
        winner = str(candidates[0]["action_id"])
        winners.append(winner)
        confirmations.append(winner)
        screening = [
            {
                "action_id": str(candidate["action_id"]),
                "raw_gain": 0.4 if index == 0 else 0.1,
                "finite_gain": True,
                "candidate_index": index,
                "point_ras_mm": list(candidate["point_ras_mm"]),
                "delta_hash": str(candidate["delta_hash"]),
                "action_digest": str(candidate["action_digest"]),
            }
            for index, candidate in enumerate(candidates)
        ]
        confirmation_row = {
            "action_id": winner,
            "candidate_index": 0,
            "point_ras_mm": list(candidates[0]["point_ras_mm"]),
            "action_digest": str(candidates[0]["action_digest"]),
            "delta_hash": str(candidates[0]["delta_hash"]),
            "raw_gain": 0.4,
            "finite_gain": True,
            "q_draws": 0,
            "footprint_voxels": 16,
            "mask_count": 32,
        }
        controls = [
            {
                "seed": seed,
                "candidate_index": index + 1,
                "action_id": str(candidates[index + 1]["action_id"]),
                "screening_gain": 0.1,
                "gain": 0.1,
                "prediction_available": True,
                "metric": {"raw_gain": 0.1},
            }
            for index, seed in enumerate((17, 29, 41))
        ]
        subjects.append(
            {
                "subject_id": subject_id,
                "candidate_pool": candidates,
                "candidate_pool_hash": canonical_digest(candidates, prefix="pfgr-lite-headroom-candidate-pool-v1|"),
                "screening": {
                    "teacher_mode": "iid_fixed_q",
                    "configured_query_count": 1024,
                    "query_count": 1024,
                    "actual_query_count": 1024,
                    "mask_denominator": 32,
                    "rows": screening,
                },
                "confirmation": {
                    "teacher_mode": "exact_footprint",
                    "configured_query_count": 1024,
                    "query_count": 16,
                    "actual_query_count": 16,
                    "rows": [confirmation_row],
                    "winner_action_id": winner,
                    "confirmation_action_id": winner,
                    "same_winner": True,
                },
                "random_controls": controls,
                "no_op": {"gain": 0.0, "metric": {"raw_gain": 0.0}},
                "oracle": {
                    "screening_gain": 0.4,
                    "confirmation_gain": 0.4,
                    "gain": 0.4,
                    "metric": {"raw_gain": 0.4},
                    "query_count": 16,
                    "mask_denominator": 32,
                    "winner_action_id": winner,
                    "confirmation_action_id": winner,
                    "same_winner": True,
                    "prediction_available": True,
                },
                "oracle_minus_random": 0.3,
                "proposal_freeze": {"before_target": True, "proposal_digest": f"proposal-{subject_id}"},
                "write_norms": {"available": True, "write_saturation": {"status": "not_applicable"}},
            }
        )

    random_values = [0.1] * 32
    oracle_values = [0.4] * 32
    z0_values = [0.0] * 32
    oracle_z0 = [0.4] * 32
    oracle_random = [0.3] * 32
    subject_set_hash = canonical_digest(tuple(subject_ids), prefix="pfgr-lite-headroom-subjects-v1|")
    candidate_pool_hash = canonical_digest(aggregate_pool, prefix="pfgr-lite-headroom-candidate-pool-v1|")
    evidence = {
        "schema_version": "pfgr-lite-headroom-evidence-v1",
        "privileged": True,
        "target_dependent": True,
        "options_hash": options_hash,
        "options": options,
        "subject_ids": subject_ids,
        "subject_count": 32,
        "candidate_count": 32,
        "query_count": 1024,
        "candidate_pool_hash": candidate_pool_hash,
        "candidate_pool": aggregate_pool,
        "subjects": subjects,
        "confirmation_gain_tolerance": 1e-6,
        "winner_action_id": canonical_digest(tuple(winners), prefix="pfgr-lite-headroom-winners-v1|"),
        "confirmation_action_id": canonical_digest(tuple(confirmations), prefix="pfgr-lite-headroom-confirmations-v1|"),
        "winner_confirmation_match": True,
        "proposals_frozen_before_target": True,
        "dense_metrics": {
            "z0": z0_values,
            "random": random_values,
            "oracle": oracle_values,
            "oracle_minus_z0": oracle_z0,
            "oracle_minus_random": oracle_random,
        },
        "independent_subject_count": 32,
        "uncertainty": 0.0,
        "confidence_intervals": {
            "method": "normal_approximation_z_1.96",
            "level": 0.95,
            "oracle_minus_z0_lower": _normal_lower_bound(oracle_z0),
            "oracle_minus_random_lower": _normal_lower_bound(oracle_random),
        },
        "precision": {"finite_dense_rows": True, "independent_subjects": 32, "minimum_for_main": 32},
        "frozen_model_digest_before": "model-v1",
        "frozen_model_digest_after": "model-v1",
        "frozen_model_unchanged": True,
        "write_saturation": {"status": "not_applicable"},
        "saturation_definition": "not_applicable:additive_compact_writer_no_clipping_v1",
    }
    expected = {key: f"{key}-v1" for key in _IDENTITY_KEYS}
    expected["subject_set_hash"] = subject_set_hash
    # The live model digest is a mandatory identity and uses the canonical
    # module-state algorithm rather than the field-name fixture token.
    expected["frozen_model_digest"] = "model-v1"
    evidence_path = tmp_path / "headroom-evidence.json"
    evidence_path.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
    evidence_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    review = {
        "schema_version": "pfgr-lite-headroom-review-v1",
        "decision": "APPROVED",
        "reviewer": "Astra",
        "evidence_artifact_hash": evidence_hash,
        "options_hash": options_hash,
        "identity_envelope": expected,
    }
    review_path = tmp_path / "headroom-review.json"
    review_path.write_text(json.dumps(review, sort_keys=True), encoding="utf-8")
    review_hash = hashlib.sha256(review_path.read_bytes()).hexdigest()
    decision = HeadroomDecision(
        producer_compatibility_hash=expected["producer_compatibility_hash"],
        source_manifest_hash=expected["source_manifest_hash"],
        base_checkpoint_hash=expected["base_checkpoint_hash"],
        updater_checkpoint_hash=expected["updater_checkpoint_hash"],
        split_hash=expected["split_hash"],
        subject_set_hash=subject_set_hash,
        teacher_identity_hash=expected["teacher_identity_hash"],
        candidate_pool_hash=candidate_pool_hash,
        winner_action_id=evidence["winner_action_id"],
        confirmation_action_id=evidence["confirmation_action_id"],
        winner_confirmation_match=True,
        proposals_frozen_before_target=True,
        no_op_gain=0.0,
        random_gain=0.1,
        oracle_gain=0.4,
        oracle_random_gain=0.3,
        practical_margin=0.05,
        uncertainty=0.0,
        candidate_count=32,
        query_count=1024,
        subject_count=32,
        seeds=(17, 29, 41),
        delta_hashes=tuple(f"delta-{index:02d}-00" for index in range(32)),
        saturation_definition="not_applicable:additive_compact_writer_no_clipping_v1",
        decision="HEADROOM_CONFIRMED",
        scientific_status="CONFIRMED",
        human_reviewed=True,
        reviewer="Astra",
        created_at="2026-09-08T00:00:00Z",
        evidence_artifact_hash=evidence_hash,
        source_provenance_status="CLEAN",
        precise=True,
        engineering_only=False,
        correction_only=False,
        evidence_artifact_path=str(evidence_path),
        review_artifact_hash=review_hash,
        review_artifact_path=str(review_path),
        options_hash=options_hash,
        frozen_model_digest="model-v1",
    )
    return decision, expected, evidence_path, review_path


def test_main_gate_accepts_one_bound_positive_permit(tmp_path: Path) -> None:
    decision, expected, _, _ = _permit_fixture(tmp_path)
    accepted = require_main_headroom_decision(decision, expected=expected)
    assert accepted.decision == "HEADROOM_CONFIRMED"


def test_main_gate_denies_removed_exact_confirmations_even_after_rehash(tmp_path: Path) -> None:
    """Artifact/hash edits must not turn a missing confirmation into MAIN evidence."""

    decision, expected, evidence_path, review_path = _permit_fixture(tmp_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    for subject in evidence["subjects"]:
        subject["confirmation"]["rows"] = []
        subject["confirmation"]["query_count"] = 0
        subject["confirmation"]["actual_query_count"] = 0
    evidence_path.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
    evidence_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["evidence_artifact_hash"] = evidence_hash
    review_path.write_text(json.dumps(review, sort_keys=True), encoding="utf-8")
    rehashed = replace(
        decision,
        evidence_artifact_hash=evidence_hash,
        review_artifact_hash=hashlib.sha256(review_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="confirmation rows"):
        require_main_headroom_decision(rehashed, expected=expected)


@pytest.mark.parametrize("mutation", ["action", "delta", "pool", "confirmation_index"])
def test_main_gate_denies_retained_confirmation_identity_or_pool_mutation(tmp_path: Path, mutation: str) -> None:
    decision, expected, evidence_path, review_path = _permit_fixture(tmp_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if mutation == "action":
        evidence["subjects"][0]["confirmation"]["rows"][0]["action_id"] = "tampered-action"
    elif mutation == "delta":
        evidence["subjects"][0]["confirmation"]["rows"][0]["delta_hash"] = "tampered-delta"
    elif mutation == "confirmation_index":
        # The retained confirmation row is a singleton measurement, but its
        # candidate_index must remain the original sealed-pool position.
        evidence["subjects"][0]["confirmation"]["rows"][0]["candidate_index"] = 1
    else:
        # Remove one retained pool row and update both artifact/decision hashes;
        # the per-subject sealed-pool reconciliation must still reject it.
        evidence["candidate_pool"] = evidence["candidate_pool"][:-1]
        evidence["candidate_pool_hash"] = canonical_digest(
            evidence["candidate_pool"], prefix="pfgr-lite-headroom-candidate-pool-v1|"
        )
    evidence_path.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
    evidence_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["evidence_artifact_hash"] = evidence_hash
    review_path.write_text(json.dumps(review, sort_keys=True), encoding="utf-8")
    updates: dict[str, object] = {
        "evidence_artifact_hash": evidence_hash,
        "review_artifact_hash": hashlib.sha256(review_path.read_bytes()).hexdigest(),
    }
    if mutation == "pool":
        updates["candidate_pool_hash"] = evidence["candidate_pool_hash"]
    rehashed = replace(decision, **updates)
    with pytest.raises(ValueError, match="(?:action|delta|pool|candidate)"):
        require_main_headroom_decision(rehashed, expected=expected)


@pytest.mark.parametrize("mutation", ["freeze", "random", "ranking", "confirmation_gain"])
def test_main_gate_reconciles_cross_record_target_boundary_joins(tmp_path: Path, mutation: str) -> None:
    """Rehashing one record cannot bypass the sealed R4B composition joins."""

    decision, expected, evidence_path, review_path = _permit_fixture(tmp_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    subject = evidence["subjects"][0]
    if mutation == "freeze":
        subject["proposal_freeze"]["before_target"] = False
    elif mutation == "random":
        subject["random_controls"][0]["action_id"] = "outside-sealed-pool"
    elif mutation == "ranking":
        # Candidate zero remains the retained winner while a different
        # screened candidate is made strictly better; the validator must
        # recompute ranking instead of trusting oracle.winner_action_id.
        subject["screening"]["rows"][1]["raw_gain"] = 0.9
        for control in subject["random_controls"]:
            if control["candidate_index"] == 1:
                control["screening_gain"] = 0.9
    else:
        subject["confirmation"]["rows"][0]["raw_gain"] = 0.8
        subject["oracle"]["confirmation_gain"] = 0.8
    evidence_path.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
    evidence_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["evidence_artifact_hash"] = evidence_hash
    review_path.write_text(json.dumps(review, sort_keys=True), encoding="utf-8")
    rehashed = replace(
        decision,
        evidence_artifact_hash=evidence_hash,
        review_artifact_hash=hashlib.sha256(review_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="(?:frozen|outside|ranking|reconcile|confirmation)"):
        require_main_headroom_decision(rehashed, expected=expected)


def test_main_gate_denies_missing_or_updater_only_decision(tmp_path: Path) -> None:
    decision, expected, _, _ = _permit_fixture(tmp_path)
    with pytest.raises(ValueError, match="accepted HeadroomDecision"):
        require_main_headroom_decision(None, expected=expected)
    with pytest.raises(ValueError, match="does not authorize"):
        require_main_headroom_decision(replace(decision, decision="INCONCLUSIVE", scientific_status="INCONCLUSIVE"), expected=expected)


def test_main_gate_requires_exact_zero_noop_per_subject_and_decision(tmp_path: Path) -> None:
    decision, expected, evidence_path, review_path = _permit_fixture(tmp_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["subjects"][0]["no_op"]["gain"] = 1e-3
    evidence_path.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
    evidence_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["evidence_artifact_hash"] = evidence_hash
    review_path.write_text(json.dumps(review, sort_keys=True), encoding="utf-8")
    mutated = replace(
        decision,
        evidence_artifact_hash=evidence_hash,
        review_artifact_hash=hashlib.sha256(review_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="no-op"):
        require_main_headroom_decision(mutated, expected=expected)


@pytest.mark.parametrize("field", ["producer_compatibility_hash", "source_manifest_hash", "frozen_model_digest"])
def test_main_gate_denies_stale_identity_join(tmp_path: Path, field: str) -> None:
    decision, expected, _, _ = _permit_fixture(tmp_path)
    stale = dict(expected)
    stale[field] = f"stale-{field}"
    with pytest.raises(ValueError, match="mismatch"):
        require_main_headroom_decision(decision, expected=stale)


def test_main_gate_denies_correction_only_and_wide_ci(tmp_path: Path) -> None:
    decision, expected, evidence_path, review_path = _permit_fixture(tmp_path)
    with pytest.raises(ValueError, match="correction-only"):
        require_main_headroom_decision(replace(decision, correction_only=True), expected=expected)

    # Make the retained paired subject rows genuinely noisy: the mean remains
    # positive, but alternating large effects widen the independent-subject CI
    # beyond the frozen practical margin.  This exercises recomputation from
    # rows rather than trusting a hand-edited summary bound.
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    noisy_deltas: list[float] = []
    for subject_index, subject in enumerate(evidence["subjects"]):
        noisy_gain = 1.5 if subject_index < 16 else -0.9
        noisy_deltas.append(noisy_gain)
        winner_id = subject["oracle"]["winner_action_id"]
        for row in subject["screening"]["rows"]:
            # Keep the originally sealed winner ranked first even when the
            # noisy paired effect is negative, so this test reaches the CI
            # denial rather than the independent ranking guard.
            row["raw_gain"] = noisy_gain if row["action_id"] == winner_id else noisy_gain - 0.1
        subject["confirmation"]["rows"][0]["raw_gain"] = noisy_gain
        for control in subject["random_controls"]:
            control["gain"] = 0.0
            control["screening_gain"] = noisy_gain - 0.1
            control["metric"]["raw_gain"] = 0.0
        subject["oracle"]["screening_gain"] = noisy_gain
        subject["oracle"]["confirmation_gain"] = noisy_gain
        subject["oracle"]["gain"] = noisy_gain
        subject["oracle"]["metric"]["raw_gain"] = noisy_gain
        subject["oracle_minus_random"] = noisy_gain
    mean_gain = statistics.fmean(noisy_deltas)
    uncertainty = statistics.stdev(noisy_deltas) / math.sqrt(len(noisy_deltas))
    lower_bound = _normal_lower_bound(noisy_deltas)
    evidence["dense_metrics"] = {
        "z0": [0.0] * len(noisy_deltas),
        "random": [0.0] * len(noisy_deltas),
        "oracle": noisy_deltas,
        "oracle_minus_z0": noisy_deltas,
        "oracle_minus_random": noisy_deltas,
    }
    evidence["uncertainty"] = uncertainty
    evidence["confidence_intervals"]["oracle_minus_z0_lower"] = lower_bound
    evidence["confidence_intervals"]["oracle_minus_random_lower"] = lower_bound
    evidence_path.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
    evidence_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["evidence_artifact_hash"] = evidence_hash
    review_path.write_text(json.dumps(review, sort_keys=True), encoding="utf-8")
    review_hash = hashlib.sha256(review_path.read_bytes()).hexdigest()
    wide = replace(
        decision,
        evidence_artifact_hash=evidence_hash,
        review_artifact_hash=review_hash,
        random_gain=0.0,
        oracle_gain=mean_gain,
        oracle_random_gain=mean_gain,
        uncertainty=uncertainty,
    )
    assert lower_bound < decision.practical_margin
    with pytest.raises(ValueError, match="confidence bounds"):
        require_main_headroom_decision(wide, expected=expected)


def test_main_gate_denies_arbitrary_review_and_cohort_mismatch(tmp_path: Path) -> None:
    decision, expected, _, review_path = _permit_fixture(tmp_path)
    review_path.write_text(json.dumps({"decision": "APPROVED"}), encoding="utf-8")
    review_hash = hashlib.sha256(review_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="review artifact schema"):
        require_main_headroom_decision(replace(decision, review_artifact_hash=review_hash), expected=expected)
    wrong_cohort = dict(expected)
    wrong_cohort["subject_set_hash"] = "wrong-cohort"
    with pytest.raises(ValueError, match="mismatch"):
        require_main_headroom_decision(decision, expected=wrong_cohort)


def test_headroom_schema_rejects_unknown_decision_enum() -> None:
    values = {
        "producer_compatibility_hash": "p",
        "source_manifest_hash": "s",
        "base_checkpoint_hash": "b",
        "updater_checkpoint_hash": "u",
        "split_hash": "split",
        "subject_set_hash": "subjects",
        "teacher_identity_hash": "teacher",
        "candidate_pool_hash": "pool",
        "winner_action_id": "winner",
        "confirmation_action_id": "confirmation",
        "winner_confirmation_match": True,
        "proposals_frozen_before_target": True,
        "no_op_gain": 0.0,
        "random_gain": 0.0,
        "oracle_gain": 0.0,
        "oracle_random_gain": 0.0,
        "practical_margin": 0.0,
        "uncertainty": 0.0,
        "candidate_count": 32,
        "query_count": 1024,
        "subject_count": 32,
        "seeds": (17, 29, 41),
        "delta_hashes": ("delta",),
        "saturation_definition": "not_applicable",
        "decision": "ACCEPTED",
        "scientific_status": "CONFIRMED",
        "human_reviewed": False,
        "reviewer": None,
        "created_at": "now",
        "evidence_artifact_hash": "evidence",
        "source_provenance_status": "CLEAN",
        "precise": True,
    }
    with pytest.raises(ValueError, match="unknown HeadroomDecision decision enum"):
        HeadroomDecision.from_dict(values)
