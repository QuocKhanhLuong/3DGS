from __future__ import annotations

import pytest
import torch

from smagm.features.point_guided import PointGuidedConfig
from smagm.features.point_guided.pfgr_lite.calibration import CalibrationEvidence, CalibrationWinner, ForcedCalibrationTrace, TraceReceipt, fit_calibration
from smagm.features.point_guided.pfgr_lite.action_proposal import ACTION_GENERATOR_VERSION
from smagm.features.point_guided.pfgr_lite.config import PFGRLiteConfig, PFGRPolicyConfig
from smagm.features.point_guided.pfgr_lite.inference import run_pfgr_inference
from smagm.features.point_guided.pfgr_lite.model import PFGRLiteModel
from smagm.features.point_guided.pfgr_lite.policy import load_effective_policy
from smagm.features.point_guided.pfgr_lite.provenance import canonical_digest
from smagm.features.point_guided.pfgr_lite.types import DynamicTriPlanes, TrainingRoleManifest


def _records(role: str, start: int, *, measurement_role: str = "exact_footprint") -> list[CalibrationWinner]:
    rows: list[CalibrationWinner] = []
    for index in range(64):
        subject = f"subject-{start + index // 2:03d}"
        raw = float(index + 1) / 10.0
        gain = 2.0 * raw + 1.0
        rows.append(
            CalibrationWinner(
                subject_id=subject,
                action_id=f"action-{role}-{index}",
                proposal_digest=f"proposal-{role}-{index}",
                action_digest=f"action-digest-{role}-{index}",
                raw_score=raw,
                measured_gain=gain,
                producer_compatibility_hash="producer",
                value_fit_identity_hash="value-fit",
                gain_scale_hash="scale",
                policy_hash=canonical_digest(PFGRLiteConfig().policy, prefix="pfgr-lite-policy-config-v1|"),
                writer_hash="writer",
                query_hash="query",
                proposal_generator_hash=ACTION_GENERATOR_VERSION,
                role=role,
                measurement_role=measurement_role,
                state_version=index % 4,
            )
        )
    return rows


def _evidence(config: PFGRLiteConfig, fit: list[CalibrationWinner], allowance: list[CalibrationWinner], *, synthetic: bool = True) -> CalibrationEvidence:
    fit_subjects = tuple(sorted({row.subject_id for row in fit}))
    allowance_subjects = tuple(sorted({row.subject_id for row in allowance}))
    producer_subjects = tuple(f"producer-{index:03d}" for index in range(32))
    baseline_train = producer_subjects + fit_subjects + allowance_subjects
    role_manifest = TrainingRoleManifest(
        baseline_split_hash="baseline-split",
        baseline_train_subject_ids=baseline_train,
        baseline_validation_subject_ids=("validation-000",),
        baseline_test_subject_ids=("test-000",),
        producer_fit_subject_ids=producer_subjects,
        calibration_fit_subject_ids=fit_subjects,
        calibration_allowance_subject_ids=allowance_subjects,
        subject_group_ids=tuple((subject, subject) for subject in baseline_train + ("validation-000", "test-000")),
        engineering_only=True,
    )
    identities = fit[0]
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
    return CalibrationEvidence(
        baseline_split_hash="baseline-split",
        producer_fit_subjects=producer_subjects,
        fit_subjects=fit_subjects,
        allowance_subjects=allowance_subjects,
        completed_trace_hashes=tuple(receipt.trace_hash for receipt in receipts),
        completed_trace_receipts=receipts,
        winner_bindings=tuple((row.subject_id, row.action_id, row.proposal_digest, row.action_digest, row.state_version) for row in all_rows),
        producer_compatibility_hash=identities.producer_compatibility_hash,
        value_fit_identity_hash=identities.value_fit_identity_hash,
        gain_scale_hash=identities.gain_scale_hash,
        policy_hash=canonical_digest(config.policy, prefix="pfgr-lite-policy-config-v1|"),
        writer_hash=identities.writer_hash,
        query_hash=identities.query_hash,
        proposal_generator_hash=identities.proposal_generator_hash,
        config_hash=canonical_digest(config.as_dict(), prefix="pfgr-lite-calibration-config-v1|"),
        role_manifest=role_manifest,
        confirmation_mode="exact",
        confirmation_independence_hash="independent-confirmation",
        synthetic=synthetic,
    )


def test_fit_positive_affine_and_pooled_higher_allowance() -> None:
    fit = _records("calibration_fit", 0)
    allowance = _records("calibration_allowance", 100)
    for index, row in enumerate(allowance):
        allowance[index] = CalibrationWinner(
            **{
                **row.__dict__,
                "measured_gain": row.measured_gain - float(index % 5) / 10.0,
            }
        )
    config = PFGRLiteConfig()
    calibration = fit_calibration(fit, allowance, config, evidence=_evidence(config, fit, allowance))
    assert calibration.capability == "diagnostic"
    assert calibration.a == pytest.approx(2.0, abs=1e-12)
    assert calibration.b == pytest.approx(1.0, abs=1e-12)
    assert calibration.allowance == pytest.approx(0.4, abs=1e-12)
    assert calibration.fit_count == calibration.allowance_count == 64
    assert calibration.fit_role_hash and calibration.allowance_role_hash


def test_calibration_requires_disjoint_adequate_non_screening_roles() -> None:
    with pytest.raises(ValueError, match="envelope"):
        fit_calibration(_records("calibration_fit", 0)[:10], _records("calibration_allowance", 100), PFGRLiteConfig())
    overlap = _records("calibration_allowance", 0)
    fit = _records("calibration_fit", 0)
    with pytest.raises(ValueError, match="membership|disjoint|evidence"):
        fit_calibration(fit, overlap, PFGRLiteConfig(), evidence=_evidence(PFGRLiteConfig(), fit, _records("calibration_allowance", 100)))
    with pytest.raises(ValueError, match="exact or independent"):
        _records("calibration_allowance", 100, measurement_role="screening")


def test_calibration_rejects_degenerate_or_mismatched_identities() -> None:
    fit = _records("calibration_fit", 0)
    allowance = _records("calibration_allowance", 100)
    config = PFGRLiteConfig()
    evidence = _evidence(config, fit, allowance)
    for index, row in enumerate(fit):
        fit[index] = CalibrationWinner(**{**row.__dict__, "raw_score": 1.0})
    with pytest.raises(ValueError, match="degenerate"):
        fit_calibration(fit, allowance, config, evidence=evidence)
    mismatched = allowance
    mismatched[0] = CalibrationWinner(**{**mismatched[0].__dict__, "query_hash": "other-query"})
    with pytest.raises(ValueError, match="identity mismatch"):
        fit_calibration(_records("calibration_fit", 0), mismatched, config, evidence=evidence)


def test_duplicate_rows_and_synthetic_evidence_never_establish_adaptive_release() -> None:
    config = PFGRLiteConfig()
    fit = _records("calibration_fit", 0)
    allowance = _records("calibration_allowance", 100)
    duplicate = fit[0]
    fit[1] = CalibrationWinner(**{**duplicate.__dict__, "action_id": duplicate.action_id, "proposal_digest": duplicate.proposal_digest, "action_digest": duplicate.action_digest})
    with pytest.raises(ValueError, match="unique"):
        fit_calibration(fit, allowance, config, evidence=_evidence(config, fit, allowance))
    fit = _records("calibration_fit", 0)
    allowance = _records("calibration_allowance", 100)
    calibration = fit_calibration(fit, allowance, config, evidence=_evidence(config, fit, allowance, synthetic=True))
    assert calibration.capability == "diagnostic"


def test_forced_trace_requires_explicit_subject_context_receipt() -> None:
    torch.manual_seed(31)
    frontend = PointGuidedConfig(
        num_semantic_classes=3,
        num_points=4,
        point_candidate_multiplier=3,
        offset_hidden_channels=12,
        detach_backbone_features=False,
    )
    model = PFGRLiteModel(PFGRLiteConfig(num_points=4, engineering_only=True, policy=PFGRPolicyConfig(mode="forced_diagnostic")), frontend_config=frontend).eval()
    context = model.encode_observations(torch.randn(1, 3, 9, 9, 9), None, (1.0, 1.0, 1.0))
    policy = load_effective_policy(
        PFGRLiteConfig(num_points=4, engineering_only=True, policy=PFGRPolicyConfig(mode="forced_diagnostic")),
        None,
        dependencies=context.producer,
        capability="forced_diagnostic",
        budget=4,
    )

    def query(state, points, feature_geometry):
        del state, feature_geometry
        return torch.zeros(points.shape[0], points.shape[1], 96, dtype=points.dtype, device=points.device)

    def writer(state, context, action):
        del context, action
        return DynamicTriPlanes(xy=state.planes.xy, xz=state.planes.xz, yz=state.planes.yz)

    route = run_pfgr_inference(model, context, policy, query=query, writer=writer, legal_mask=torch.zeros(1, 4, dtype=torch.bool))
    assert route.stop_reason == "no_legal_action"
    geometry_hash = canonical_digest(
        {"shape_dhw": context.geometry.shape_dhw, "voxel_to_ras_mm": context.geometry.voxel_to_ras_mm},
        prefix="pfgr-lite-subject-geometry-v1|",
    )
    binding_payload = {
        "schema_version": "pfgr-lite-subject-context-binding-v1",
        "subject_id": "subject-actual",
        "observation_record_id": "record-actual",
        "context_id": context.context_id,
        "geometry_hash": geometry_hash,
        "normalization_hash": context.producer.compatibility.observation_normalization_hash,
    }
    binding = dict(binding_payload)
    binding["binding_digest"] = canonical_digest(binding_payload, prefix="pfgr-lite-subject-context-binding-v1|")
    wrapped = ForcedCalibrationTrace(context, route, policy, "subject-actual", binding)
    assert wrapped.receipt.state_versions == (0,)
    bad_binding = dict(binding)
    bad_binding["subject_id"] = "subject-replaced"
    with pytest.raises(ValueError, match="digest|subject"):
        ForcedCalibrationTrace(context, route, policy, "subject-actual", bad_binding)
