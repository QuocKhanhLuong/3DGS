from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from smagm.features.point_guided.pfgr_lite import (
    ActionProposalBatch,
    CompletedBehaviorTrace,
    Decision,
    EffectTeacherConfig,
    GainCalibration,
    PFGRLiteConfig,
    PFGRPolicyConfig,
    StaticSynthesisConfig,
    ValueFitIdentity,
    PFGRState,
    ProducerCompatibility,
)
from smagm.features.point_guided.state_init import DynamicTriPlanes


def test_locked_pfgr_configuration_defaults_and_schema() -> None:
    config = PFGRLiteConfig()
    assert config.schema_version == "pfgr-lite-config-v1"
    assert config.candidate_count == 2048
    assert config.state_channels == 32
    assert config.correction_channels == 96
    assert config.write_scale == pytest.approx(0.1)
    assert config.policy.budgets == (0, 1, 2, 4)
    assert config.static.variant == "b2_ordered_multiscale_v1"
    assert config.value.input_variants == (126, 222, 270, 366)
    assert config.teacher.mode == "iid_fixed_q"
    assert config.teacher.q_draws == 1024


@pytest.mark.parametrize(
    ("factory", "message"),
    (
        (lambda: PFGRPolicyConfig(candidate_count=4), "candidate_count"),
        (lambda: PFGRLiteConfig(write_scale=0.2), "write_scale"),
        (lambda: EffectTeacherConfig(mode="iid_fixed_q", q_draws=1), "q_draws"),
        (lambda: StaticSynthesisConfig(variant="unknown"), "unknown"),
        (lambda: PFGRPolicyConfig(mode="future"), "unknown"),
        (lambda: StaticSynthesisConfig(architecture="b0_legacy_v1"), "architecture"),
        (lambda: PFGRLiteConfig(num_points=4), "engineering_only"),
    ),
)
def test_pfgr_contract_rejects_unlocked_values(factory, message: str) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        factory()


def test_action_proposal_batch_detects_tensor_mutation() -> None:
    dtype = torch.float64
    batch = ActionProposalBatch(
        context_id="ctx",
        producer_compatibility_hash="producer",
        state_version=0,
        state_digest="state",
        point_ids=torch.tensor([[0]], dtype=torch.long),
        points_ras_mm=torch.zeros(1, 1, 3, dtype=dtype),
        o270=torch.zeros(1, 1, 270, dtype=dtype),
        v126=torch.zeros(1, 1, 126, dtype=dtype),
        delta=torch.zeros(1, 1, 96, dtype=dtype),
        legal=torch.ones(1, 1, dtype=torch.bool),
        updater_producer_hash="producer",
        writer_hash="writer",
        query_hash="query",
        geometry_hash="geometry",
        point_identity_hash="point",
    )
    batch.delta[0, 0, 0] = 1.0
    with pytest.raises(RuntimeError, match="mutation"):
        batch.validate_integrity()


def test_calibration_and_value_identity_are_distinct() -> None:
    value = ValueFitIdentity(input_variant=366, architecture_hash="a", weights_hash="w", fit_config_hash="f", bank_manifest_hash="b", gain_scale_hash="s")
    calibration = GainCalibration(a=1.0, b=0.0, allowance=0.0, producer_compatibility_hash="p", value_fit_identity_hash=value.digest, gain_scale_hash="s")
    assert value.digest != calibration.version
    assert calibration.allowance >= 0.0


def _producer() -> ProducerCompatibility:
    return ProducerCompatibility(
        observation_normalization_hash="norm",
        geometry_query_version_hash="geometry",
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


def _proposal_batch(state_digest: str = "state") -> ActionProposalBatch:
    dtype = torch.float64
    return ActionProposalBatch(
        context_id="ctx",
        producer_compatibility_hash="producer",
        state_version=0,
        state_digest=state_digest,
        point_ids=torch.tensor([[0]], dtype=torch.long),
        points_ras_mm=torch.zeros(1, 1, 3, dtype=dtype),
        o270=torch.zeros(1, 1, 270, dtype=dtype),
        v126=torch.zeros(1, 1, 126, dtype=dtype),
        delta=torch.zeros(1, 1, 96, dtype=dtype),
        legal=torch.ones(1, 1, dtype=torch.bool),
        updater_producer_hash="producer",
        writer_hash="writer",
        query_hash="query",
        geometry_hash="geometry",
        point_identity_hash="point",
    )


def test_action_identity_covers_metadata_and_stored_delta() -> None:
    batch = _proposal_batch()
    row = batch.row(0, 0)
    assert row.action_digest
    assert row.context_id == batch.context_id
    assert row.state_digest == batch.state_digest
    row.validate_integrity()
    with torch.no_grad():
        row.delta[0] = 0.5
    with pytest.raises(RuntimeError, match="ActionProposal mutation"):
        row.validate_integrity()
    with torch.no_grad():
        batch.delta[0, 0, 0] = 0.25
    with pytest.raises(RuntimeError, match="mutation"):
        batch.validate_integrity()
    changed = _proposal_batch(state_digest="state-changed")
    assert changed.proposal_digest != _proposal_batch().proposal_digest


def test_action_batch_rejects_bool_ids_duplicates_and_nonbinary_legal() -> None:
    with pytest.raises(TypeError, match="bool"):
        ActionProposalBatch(
            context_id="ctx",
            producer_compatibility_hash="producer",
            state_version=0,
            state_digest="state",
            point_ids=torch.tensor([[True]], dtype=torch.bool),
            points_ras_mm=torch.zeros(1, 1, 3),
            o270=torch.zeros(1, 1, 270),
            v126=torch.zeros(1, 1, 126),
            delta=torch.zeros(1, 1, 96),
            legal=torch.ones(1, 1, dtype=torch.bool),
            updater_producer_hash="producer",
            writer_hash="writer",
            query_hash="query",
            geometry_hash="geometry",
            point_identity_hash="point",
        )
    with pytest.raises(ValueError, match="unique"):
        ActionProposalBatch(
            context_id="ctx",
            producer_compatibility_hash="producer",
            state_version=0,
            state_digest="state",
            point_ids=torch.tensor([[0, 0]], dtype=torch.long),
            points_ras_mm=torch.zeros(1, 2, 3),
            o270=torch.zeros(1, 2, 270),
            v126=torch.zeros(1, 2, 126),
            delta=torch.zeros(1, 2, 96),
            legal=torch.ones(1, 2, dtype=torch.bool),
            updater_producer_hash="producer",
            writer_hash="writer",
            query_hash="query",
            geometry_hash="geometry",
            point_identity_hash="point",
        )
    with pytest.raises(ValueError, match="0/1"):
        ActionProposalBatch(
            context_id="ctx",
            producer_compatibility_hash="producer",
            state_version=0,
            state_digest="state",
            point_ids=torch.tensor([[0]], dtype=torch.long),
            points_ras_mm=torch.zeros(1, 1, 3),
            o270=torch.zeros(1, 1, 270),
            v126=torch.zeros(1, 1, 126),
            delta=torch.zeros(1, 1, 96),
            legal=torch.tensor([[2]], dtype=torch.int64),
            updater_producer_hash="producer",
            writer_hash="writer",
            query_hash="query",
            geometry_hash="geometry",
            point_identity_hash="point",
        )


def test_k0_trace_is_coherent_and_trace_hash_covers_decisions() -> None:
    planes = DynamicTriPlanes(
        xy=torch.zeros(1, 32, 2, 2, dtype=torch.float64),
        xz=torch.zeros(1, 32, 2, 2, dtype=torch.float64),
        yz=torch.zeros(1, 32, 2, 2, dtype=torch.float64),
    )
    state = PFGRState(planes=planes, context_id="ctx", producer=_producer(), role="deployment")
    trace = CompletedBehaviorTrace(context_id="ctx", states=(state,))
    assert trace.route_hash
    with pytest.raises(ValueError, match="initial"):
        CompletedBehaviorTrace(context_id="ctx")
    proposal = _proposal_batch(state_digest=state.state_digest)
    next_state = state.next(DynamicTriPlanes(xy=planes.xy.clone(), xz=planes.xz.clone(), yz=planes.yz.clone()))
    decision = Decision(selected_point_id=0, policy_hash="policy", step=0)
    complete = CompletedBehaviorTrace(
        context_id="ctx",
        states=(state, next_state),
        proposals=(proposal,),
        decisions=(decision,),
    )
    assert complete.route_hash != trace.route_hash
    with pytest.raises(ValueError, match="route_hash"):
        CompletedBehaviorTrace(
            context_id="ctx",
            states=(state, next_state),
            proposals=(proposal,),
            decisions=(decision,),
            route_hash="0" * 64,
        )


def test_state_without_producer_is_rejected_at_constructor_boundary() -> None:
    planes = DynamicTriPlanes(
        xy=torch.zeros(1, 32, 2, 2),
        xz=torch.zeros(1, 32, 2, 2),
        yz=torch.zeros(1, 32, 2, 2),
    )
    with pytest.raises(TypeError, match="producer"):
        PFGRState(planes=planes, context_id="ctx", producer=None)


def test_target_bearing_modules_stay_lazy_in_fresh_process() -> None:
    root = Path(__file__).resolve().parents[4]
    code = """
import sys
import smagm.features.point_guided
for name in ('smagm.features.point_guided.training_objective', 'smagm.features.point_guided.reward_supervision', 'smagm.features.point_guided.oracle'):
    assert name not in sys.modules, name
from smagm.features.point_guided import PointGuidedMRIModel
for name in ('smagm.features.point_guided.training_objective', 'smagm.features.point_guided.reward_supervision', 'smagm.features.point_guided.oracle'):
    assert name not in sys.modules, name
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_frontend_configuration_cannot_be_silently_ignored() -> None:
    from smagm.features.point_guided.pfgr_lite import PFGRLiteModel

    with pytest.raises(TypeError, match="point_guided"):
        PFGRLiteModel(PFGRLiteConfig(point_guided={"num_points": 2048}))
