from __future__ import annotations

import pytest
import torch

from smagm.features.point_guided.pfgr_lite.config import PFGRLiteConfig, PFGRPolicyConfig
from smagm.features.point_guided.pfgr_lite.action_proposal import ACTION_GENERATOR_VERSION
from smagm.features.point_guided.pfgr_lite.calibration import CalibrationEvidence, CalibrationWinner, TraceReceipt, attach_calibration_evidence, calibration_evidence, fit_calibration
from smagm.features.point_guided.pfgr_lite.policy import (
    EffectivePolicy,
    load_effective_policy,
    select_or_stop,
)
from smagm.features.point_guided.pfgr_lite.provenance import ProducerCompatibility, ValueFitIdentity
from smagm.features.point_guided.pfgr_lite.types import ActionProposalBatch, GainCalibration, TrainingRoleManifest
from smagm.features.point_guided.pfgr_lite.provenance import canonical_digest


def _producer() -> ProducerCompatibility:
    return ProducerCompatibility(
        observation_normalization_hash="norm",
        geometry_query_version_hash="query",
        medicalnet_provenance_hash="medicalnet",
        frozen_bn_hash="bn",
        static_head_hash="static",
        semantic_head_hash="semantic",
        point_refiner_hash="points",
        spectral_projector_hash="spectral",
        state_initializer_hash="state",
        updater_hash="updater",
        decoder_hash="decoder",
        writer_hash="writer",
        candidate_geometry_hash="candidate",
        label_definition_hash="label",
    )


def _proposals() -> ActionProposalBatch:
    producer_hash = _producer().digest
    return ActionProposalBatch(
        context_id="ctx",
        state_version=0,
        state_digest="state",
        producer_compatibility_hash=producer_hash,
        point_ids=torch.tensor([[3, 1, 7]], dtype=torch.long),
        points_ras_mm=torch.zeros(1, 3, 3),
        o270=torch.zeros(1, 3, 270),
        v126=torch.zeros(1, 3, 126),
        delta=torch.tensor([[[0.1] + [0.0] * 95, [0.05] + [0.0] * 95, [0.02] + [0.0] * 95]]),
        legal=torch.ones(1, 3, dtype=torch.bool),
        updater_producer_hash="updater",
        writer_hash="writer",
        query_hash="query",
        geometry_hash="geometry",
        point_identity_hash="point",
    )


def _adaptive_policy(*, margin: float = 0.0, budget: int = 4) -> EffectivePolicy:
    producer = _producer()
    config = PFGRLiteConfig(engineering_only=True)
    value_identity = ValueFitIdentity(
        input_variant=366,
        architecture_hash="arch",
        weights_hash="weights",
        fit_config_hash="fit-config",
        bank_manifest_hash="bank",
        gain_scale_hash="scale",
    )
    fit: list[CalibrationWinner] = []
    allowance: list[CalibrationWinner] = []
    for role, start, rows in (("calibration_fit", 0, fit), ("calibration_allowance", 100, allowance)):
        for index in range(64):
            subject = f"subject-{start + index // 2:03d}"
            rows.append(
                CalibrationWinner(
                    subject_id=subject,
                    action_id=f"action-{role}-{index}",
                    proposal_digest=f"proposal-{role}-{index}",
                    action_digest=f"digest-{role}-{index}",
                    raw_score=float(index + 1),
                    measured_gain=float(index + 1),
                    producer_compatibility_hash=producer.digest,
                    value_fit_identity_hash=value_identity.digest,
                    gain_scale_hash="scale",
                    policy_hash=canonical_digest(config.policy, prefix="pfgr-lite-policy-config-v1|"),
                    writer_hash="writer",
                    query_hash="query",
                    proposal_generator_hash=ACTION_GENERATOR_VERSION,
                    role=role,
                    measurement_role="exact_footprint",
                    state_version=index % 4,
                )
            )
    producer_subjects = tuple(f"producer-{index:03d}" for index in range(32))
    fit_subjects = tuple(sorted({row.subject_id for row in fit}))
    allowance_subjects = tuple(sorted({row.subject_id for row in allowance}))
    roles = TrainingRoleManifest(
        baseline_split_hash="split",
        baseline_train_subject_ids=producer_subjects + fit_subjects + allowance_subjects,
        baseline_validation_subject_ids=("validation",),
        baseline_test_subject_ids=("test",),
        producer_fit_subject_ids=producer_subjects,
        calibration_fit_subject_ids=fit_subjects,
        calibration_allowance_subject_ids=allowance_subjects,
        subject_group_ids=tuple((subject, subject) for subject in producer_subjects + fit_subjects + allowance_subjects + ("validation", "test")),
        engineering_only=False,
    )
    all_rows = fit + allowance
    receipts = tuple(
        TraceReceipt(
            trace_hash=f"trace-{index:03d}",
            context_id=f"context-{index:03d}",
            state_versions=(0, 1, 2, 3, 4),
            proposal_digests=tuple(row.proposal_digest for row in all_rows[index * 4 : index * 4 + 4]),
            action_digests=tuple(row.action_digest for row in all_rows[index * 4 : index * 4 + 4]),
        )
        for index in range(len(all_rows) // 4)
    )
    evidence = CalibrationEvidence(
        baseline_split_hash="split",
        producer_fit_subjects=producer_subjects,
        fit_subjects=fit_subjects,
        allowance_subjects=allowance_subjects,
        completed_trace_hashes=tuple(receipt.trace_hash for receipt in receipts),
        completed_trace_receipts=receipts,
        winner_bindings=tuple((row.subject_id, row.action_id, row.proposal_digest, row.action_digest, row.state_version) for row in all_rows),
        winner_confirmations=tuple((row.subject_id, row.action_id, row.proposal_digest, row.action_digest, row.measurement_role, 0, None, None, f"confirm-{row.action_id}") for row in all_rows),
        producer_compatibility_hash=producer.digest,
        value_fit_identity_hash=value_identity.digest,
        gain_scale_hash="scale",
        policy_hash=canonical_digest(config.policy, prefix="pfgr-lite-policy-config-v1|"),
        writer_hash="writer",
        query_hash="query",
        proposal_generator_hash=ACTION_GENERATOR_VERSION,
        config_hash=canonical_digest(config.as_dict(), prefix="pfgr-lite-calibration-config-v1|"),
        role_manifest=roles,
        confirmation_mode="exact",
        confirmation_independence_hash="confirm",
        fit_role_hash="fit-role",
        allowance_role_hash="allowance-role",
    )
    calibration = attach_calibration_evidence(
        GainCalibration(
            a=1.0,
            b=0.0,
            allowance=0.0,
            fit_role_hash="fit-role",
            allowance_role_hash="allowance-role",
            fit_count=64,
            allowance_count=64,
            producer_compatibility_hash=producer.digest,
            value_fit_identity_hash=value_identity.digest,
            gain_scale_hash="scale",
            capability="adaptive",
        ),
        evidence,
    )
    policy = load_effective_policy(
        config,
        calibration,
        dependencies=producer,
        capability="adaptive",
        budget=budget,
        value_fit_identity=value_identity,
        gain_scale=1.0,
        gain_scale_hash="scale",
        role_manifest_hash=evidence.role_manifest.digest,
    )
    if margin == 0.0:
        return policy
    return EffectivePolicy(
        mode=policy.mode,
        budget=policy.budget,
        quality_margin=margin,
        compute_cost=policy.compute_cost,
        producer_compatibility_hash=policy.producer_compatibility_hash,
        calibration=policy.calibration,
        value_fit_identity=value_identity,
        gain_scale=policy.gain_scale,
        gain_scale_hash=policy.gain_scale_hash,
        capability="adaptive",
    )


def test_loader_requires_calibration_for_adaptive_and_rejects_legacy_fields() -> None:
    producer = _producer()
    with pytest.raises(ValueError, match="calibration"):
        load_effective_policy(PFGRLiteConfig(), None, dependencies=producer, capability="adaptive")
    with pytest.raises(ValueError, match="unknown PFGR config keys"):
        PFGRLiteConfig.from_dict({**PFGRLiteConfig().as_dict(), "lambda_step": 0.025})


def test_selector_uses_lowest_point_id_tie_and_strict_zero_stop() -> None:
    proposals = _proposals()
    policy = _adaptive_policy()
    tied = select_or_stop(proposals, torch.tensor([[1.0, 1.0, 0.2]]), policy, step=0)
    assert tied.stop_code == "continue"
    assert tied.selected_point_id == 1
    stopped = select_or_stop(proposals, torch.tensor([[-1.0, -1.0, -0.2]]), policy, step=0)
    assert stopped.stop_code == "low_gain"
    assert stopped.selected_point_id == -1
    assert stopped.proposal_digest == proposals.proposal_digest


def test_selector_applies_fixed_gain_scale_before_affine_calibration() -> None:
    proposals = _proposals()
    base = _adaptive_policy()
    scaled = EffectivePolicy(
        mode=base.mode,
        budget=base.budget,
        quality_margin=base.quality_margin,
        compute_cost=base.compute_cost,
        producer_compatibility_hash=base.producer_compatibility_hash,
        calibration=base.calibration,
        value_fit_identity=base.value_fit_identity,
        gain_scale=2.0,
        gain_scale_hash=base.gain_scale_hash,
        engineering_only=base.engineering_only,
        capability="adaptive",
    )
    decision = select_or_stop(proposals, torch.tensor([[1.0, 0.5, 0.25]]), scaled, step=0)
    assert decision.stop_code == "continue"
    assert decision.raw_value == pytest.approx(2.0, abs=1e-12)


def test_forced_diagnostic_can_select_negative_but_budget_zero_does_no_work() -> None:
    producer = _producer()
    config = PFGRLiteConfig(policy=PFGRPolicyConfig(mode="forced_diagnostic"))
    policy = load_effective_policy(config, None, dependencies=producer, capability="forced_diagnostic", budget=1)
    proposals = _proposals()
    forced = select_or_stop(proposals, torch.tensor([[-1.0, -2.0, -3.0]]), policy, step=0)
    assert forced.stop_code == "continue"
    assert forced.selected_point_id == 3
    policy0 = load_effective_policy(config, None, dependencies=producer, capability="forced_diagnostic", budget=0)
    zero = select_or_stop(None, None, policy0, step=0)
    assert zero.stop_code == "budget"
    assert zero.proposal_digest == ""


def test_random_policy_is_seeded_and_nonfinite_scores_fail() -> None:
    producer = _producer()
    config = PFGRLiteConfig(policy=PFGRPolicyConfig(mode="random"))
    first_policy = load_effective_policy(config, None, dependencies=producer, capability="forced_diagnostic", budget=1, random_seed=17)
    second_policy = load_effective_policy(config, None, dependencies=producer, capability="forced_diagnostic", budget=1, random_seed=17)
    proposals = _proposals()
    first = select_or_stop(proposals, torch.zeros(1, 3), first_policy, step=0)
    second = select_or_stop(proposals, torch.zeros(1, 3), second_policy, step=0)
    assert first.selected_point_id == second.selected_point_id
    with pytest.raises(ValueError, match="nonfinite"):
        select_or_stop(proposals, torch.tensor([[float("nan"), 0.0, 0.0]]), first_policy, step=0)


def test_adaptive_loader_binds_v_scale_and_role_identity() -> None:
    policy = _adaptive_policy()
    evidence = policy.calibration
    assert evidence is not None
    envelope = calibration_evidence(evidence)
    assert envelope is not None
    producer = _producer()
    with pytest.raises(ValueError, match="ValueFitIdentity|exact"):
        load_effective_policy(
            PFGRLiteConfig(engineering_only=True),
            evidence,
            dependencies=producer,
            capability="adaptive",
            value_fit_identity=ValueFitIdentity(input_variant=366, architecture_hash="other-arch", weights_hash="weights", fit_config_hash="fit-config", bank_manifest_hash="bank", gain_scale_hash="scale"),
            gain_scale=1.0,
            gain_scale_hash="scale",
            role_manifest_hash=envelope.role_manifest.digest,
        )
    with pytest.raises(ValueError, match="gain-scale"):
        load_effective_policy(
            PFGRLiteConfig(engineering_only=True),
            evidence,
            dependencies=producer,
            capability="adaptive",
            value_fit_identity=policy.value_fit_identity,
            gain_scale=2.0,
            gain_scale_hash="other-scale",
            role_manifest_hash=envelope.role_manifest.digest,
        )
    with pytest.raises(ValueError, match="role|TrainingRoleManifest"):
        load_effective_policy(
            PFGRLiteConfig(engineering_only=True),
            evidence,
            dependencies=producer,
            capability="adaptive",
            value_fit_identity=policy.value_fit_identity,
            gain_scale=1.0,
            gain_scale_hash="scale",
        )
