from __future__ import annotations

import pytest
import torch

from smagm.features.point_guided import PointGuidedConfig
from smagm.features.point_guided.pfgr_lite.action_proposal import ACTION_GENERATOR_VERSION
from smagm.features.point_guided.pfgr_lite.config import PFGRLiteConfig, PFGRPolicyConfig
from smagm.features.point_guided.pfgr_lite.inference import run_pfgr_inference
from smagm.features.point_guided.pfgr_lite.calibration import CalibrationEvidence, CalibrationWinner, TraceReceipt, attach_calibration_evidence, calibration_evidence, fit_calibration, trace_receipt_from_route
from smagm.features.point_guided.pfgr_lite.model import PFGRLiteModel
from smagm.features.point_guided.pfgr_lite.policy import EffectivePolicy, load_effective_policy
from smagm.features.point_guided.pfgr_lite.provenance import ValueFitIdentity, canonical_digest
from smagm.features.point_guided.state_init import DynamicTriPlanes
from smagm.features.point_guided.state_init import DynamicTriPlanes
from smagm.features.point_guided.pfgr_lite.types import GainCalibration, TrainingRoleManifest


def _frontend_config() -> PointGuidedConfig:
    return PointGuidedConfig(
        num_semantic_classes=3,
        num_points=4,
        point_candidate_multiplier=3,
        offset_hidden_channels=12,
        detach_backbone_features=False,
    )


@pytest.fixture(scope="module")
def model_context():
    model = PFGRLiteModel(PFGRLiteConfig(num_points=4, engineering_only=True), frontend_config=_frontend_config()).eval()
    context = model.encode_observations(torch.randn(1, 3, 9, 9, 9), None, (1.0, 1.0, 1.0))
    return model, context


def _query(state: DynamicTriPlanes, points: torch.Tensor, geometry: object) -> torch.Tensor:
    del geometry
    return torch.cat(
        [
            getattr(state, name).mean(dim=(-2, -1)).unsqueeze(1).expand(-1, points.shape[1], -1)
            for name in ("xy", "xz", "yz")
        ],
        dim=-1,
    )


def _writer(state, context, action):
    del context
    split = action.delta.reshape(1, 3, 32)
    return DynamicTriPlanes(
        xy=state.planes.xy + split[:, 0, :, None, None],
        xz=state.planes.xz + split[:, 1, :, None, None],
        yz=state.planes.yz + split[:, 2, :, None, None],
    )


def _adaptive_calibration(context, config):
    value_identity = ValueFitIdentity(
        input_variant=366,
        architecture_hash="arch",
        weights_hash="weights",
        fit_config_hash="fit-config",
        bank_manifest_hash="bank",
        gain_scale_hash="scale",
    )
    fit = []
    allowance = []
    producer = context.producer.compatibility_hash
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
                    producer_compatibility_hash=producer,
                    value_fit_identity_hash=value_identity.digest,
                    gain_scale_hash="scale",
                    policy_hash=canonical_digest(config.policy, prefix="pfgr-lite-policy-config-v1|"),
                    writer_hash=context.producer.compatibility.writer_hash,
                    query_hash=context.producer.compatibility.geometry_query_version_hash,
                    proposal_generator_hash=ACTION_GENERATOR_VERSION,
                    role=role,
                    measurement_role="exact_footprint",
                    state_version=index % 4,
                )
            )
    producer_subjects = tuple(f"producer-{index:03d}" for index in range(32))
    fit_subjects = tuple(sorted({row.subject_id for row in fit}))
    allowance_subjects = tuple(sorted({row.subject_id for row in allowance}))
    manifest = TrainingRoleManifest(
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
        producer_compatibility_hash=producer,
        value_fit_identity_hash=value_identity.digest,
        gain_scale_hash="scale",
        policy_hash=canonical_digest(config.policy, prefix="pfgr-lite-policy-config-v1|"),
        writer_hash=context.producer.compatibility.writer_hash,
        query_hash=context.producer.compatibility.geometry_query_version_hash,
        proposal_generator_hash=ACTION_GENERATOR_VERSION,
        config_hash=canonical_digest(config.as_dict(), prefix="pfgr-lite-calibration-config-v1|"),
        role_manifest=manifest,
        confirmation_mode="exact",
        confirmation_independence_hash="confirm",
        fit_role_hash="fit-role",
        allowance_role_hash="allowance-role",
    )
    calibration = GainCalibration(
        a=1.0,
        b=0.0,
        allowance=0.0,
        fit_role_hash="fit-role",
        allowance_role_hash="allowance-role",
        fit_count=64,
        allowance_count=64,
        producer_compatibility_hash=producer,
        value_fit_identity_hash=value_identity.digest,
        gain_scale_hash="scale",
        capability="adaptive",
    )
    return attach_calibration_evidence(calibration, evidence), value_identity


def test_k0_has_no_proposal_or_value_work(model_context) -> None:
    model, context = model_context
    policy = load_effective_policy(
        PFGRLiteConfig(policy=PFGRPolicyConfig(mode="random"), num_points=4, engineering_only=True),
        None,
        dependencies=context.producer,
        capability="forced_diagnostic",
        budget=0,
    )

    def fail(*args, **kwargs):
        raise AssertionError("K0 must not call proposal/value/query/writer")

    result = run_pfgr_inference(model, context, policy, query=fail, writer=fail, value_model=fail)
    assert result.k == 0
    assert result.stop_reason == "budget"
    assert result.counters.candidate_proposals == 0
    assert result.counters.value_evaluations == 0


def test_random_route_is_deterministic_and_reaches_budget(model_context) -> None:
    model, context = model_context
    config = PFGRLiteConfig(policy=PFGRPolicyConfig(mode="random"), num_points=4, engineering_only=True)
    first_policy = load_effective_policy(config, None, dependencies=context.producer, capability="forced_diagnostic", budget=2, random_seed=19)
    second_policy = load_effective_policy(config, None, dependencies=context.producer, capability="forced_diagnostic", budget=2, random_seed=19)
    legal = torch.ones(1, 4, dtype=torch.bool)
    first = run_pfgr_inference(model, context, first_policy, query=_query, writer=_writer, legal_mask=legal)
    second = run_pfgr_inference(model, context, second_policy, query=_query, writer=_writer, legal_mask=legal)
    assert first.k == second.k == 2
    assert first.stop_reason == second.stop_reason == "budget"
    assert first.executed_action_ids == second.executed_action_ids
    assert torch.allclose(first.final_state.planes.xy, second.final_state.planes.xy, atol=1e-6, rtol=1e-5)


def test_adaptive_route_uses_value_scores_and_never_reads_target(model_context) -> None:
    model, context = model_context
    config = PFGRLiteConfig(num_points=4, engineering_only=True)
    calibration, value_identity = _adaptive_calibration(context, config)
    policy = load_effective_policy(
        config,
        calibration,
        dependencies=context.producer,
        capability="adaptive",
        budget=1,
        value_fit_identity=value_identity,
        gain_scale=1.0,
        gain_scale_hash="scale",
        role_manifest_hash=calibration_evidence(calibration).role_manifest.digest,
    )

    class Value:
        architecture_hash = "arch"
        weights_hash = "weights"

        def __call__(self, descriptors: torch.Tensor) -> torch.Tensor:
            assert descriptors.shape[-1] == 366
            return torch.ones(descriptors.shape[:2], dtype=descriptors.dtype, device=descriptors.device)

    value = Value()

    result = run_pfgr_inference(model, context, policy, query=_query, writer=_writer, value_model=value, legal_mask=torch.ones(1, 4, dtype=torch.bool))
    assert result.k == 1
    assert result.counters.value_evaluations == 4
    assert result.decisions[0].proposal_digest
    assert result.decisions[0].action_digest


def test_parallel_mode_uses_one_frozen_bank_with_explicit_compound_writer(model_context) -> None:
    model, context = model_context
    config = PFGRLiteConfig(policy=PFGRPolicyConfig(mode="parallel_topk"), num_points=4, engineering_only=True)
    policy = load_effective_policy(config, None, dependencies=context.producer, capability="forced_diagnostic", budget=1)
    def compound(state, context, actions):
        outputs = []
        current = state
        for action in actions:
            outputs.append(_writer(current, context, action))
            current = type(state).next(current, outputs[-1])
        return outputs

    result = run_pfgr_inference(
        model,
        context,
        policy,
        query=_query,
        writer=_writer,
        value_model=lambda x: torch.ones(x.shape[:2]),
        legal_mask=torch.ones(1, 4, dtype=torch.bool),
        compound_writer=compound,
    )
    assert result.k == 1
    assert result.completed_trace is None
    assert result.parallel_trace is not None
    assert len(result.parallel_trace.selected_action_digests) == 1


def test_forced_route_retains_typed_terminal_no_legal_assessment(model_context) -> None:
    model, context = model_context
    policy = load_effective_policy(
        PFGRLiteConfig(policy=PFGRPolicyConfig(mode="forced_diagnostic"), num_points=4, engineering_only=True),
        None,
        dependencies=context.producer,
        capability="forced_diagnostic",
        budget=4,
    )
    result = run_pfgr_inference(
        model,
        context,
        policy,
        query=_query,
        writer=_writer,
        legal_mask=torch.zeros(1, 4, dtype=torch.bool),
    )
    assert result.stop_reason == "no_legal_action"
    assert result.k == 0
    assert result.completed_trace is not None
    assert result.terminal_proposals is not None
    receipt = trace_receipt_from_route(result)
    assert receipt.state_versions == (0,)
    assert receipt.terminal_stop_code == "no_legal_action"
