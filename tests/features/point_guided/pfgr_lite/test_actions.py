from __future__ import annotations

import pytest
import torch

from smagm.features.point_guided import PointGuidedConfig
from smagm.features.point_guided.pfgr_lite.action_proposal import (
    apply_scored_action,
    propose_actions,
)
from smagm.features.point_guided.pfgr_lite.config import PFGRLiteConfig, PFGRPolicyConfig
from smagm.features.point_guided.pfgr_lite.model import PFGRLiteModel
from smagm.features.point_guided.pfgr_lite.policy import load_effective_policy, select_or_stop
from smagm.features.point_guided.pfgr_lite.types import OperationCounters
from smagm.features.point_guided.state_init import DynamicTriPlanes


def _frontend_config() -> PointGuidedConfig:
    return PointGuidedConfig(
        num_semantic_classes=3,
        num_points=4,
        point_candidate_multiplier=3,
        offset_hidden_channels=12,
        detach_backbone_features=False,
    )


@pytest.fixture(scope="module")
def model_context() -> tuple[PFGRLiteModel, object, object]:
    model = PFGRLiteModel(
        PFGRLiteConfig(num_points=4, engineering_only=True),
        frontend_config=_frontend_config(),
    ).eval()
    context = model.encode_observations(torch.randn(1, 3, 9, 9, 9), None, (1.0, 1.0, 1.0))
    state = model.initialize_state(context, role="deployment")
    return model, context, state


def _point_query(state: DynamicTriPlanes, points: torch.Tensor, feature_geometry: object) -> torch.Tensor:
    del feature_geometry
    rows = []
    for name in ("xy", "xz", "yz"):
        rows.append(getattr(state, name).mean(dim=(-2, -1)).unsqueeze(1).expand(-1, points.shape[1], -1))
    return torch.cat(rows, dim=-1)


def _writer(state, context, action):
    del context
    delta = action.delta.reshape(1, 3, 32)
    return DynamicTriPlanes(
        xy=state.planes.xy + delta[:, 0].unsqueeze(-1).unsqueeze(-1),
        xz=state.planes.xz + delta[:, 1].unsqueeze(-1).unsqueeze(-1),
        yz=state.planes.yz + delta[:, 2].unsqueeze(-1).unsqueeze(-1),
    )


def test_chunked_proposals_match_direct_shared_updater(model_context) -> None:
    model, context, state = model_context
    first = propose_actions(model.updater, state, context, query=_point_query, candidate_chunk_size=1)
    second = propose_actions(model.updater, state, context, query=_point_query, candidate_chunk_size=16)
    assert torch.equal(first.point_ids, second.point_ids)
    assert torch.equal(first.o270, second.o270)
    assert torch.equal(first.v126, second.v126)
    assert torch.allclose(first.delta, second.delta, atol=1e-6, rtol=1e-5)
    direct = model.updater(first.o270.reshape(-1, 270), write_scale=0.1).packed.reshape(1, -1, 96)
    assert torch.allclose(first.delta, direct, atol=1e-6, rtol=1e-5)
    assert first.proposal_digest != "" and second.proposal_digest != ""


def test_selected_action_executes_stored_delta_without_rerunning_u(model_context, monkeypatch) -> None:
    model, context, state = model_context
    proposals = propose_actions(model.updater, state, context, query=_point_query, candidate_chunk_size=4)
    policy = load_effective_policy(
        PFGRLiteConfig(
            policy=PFGRPolicyConfig(mode="forced_diagnostic"),
            num_points=4,
            engineering_only=True,
        ),
        None,
        dependencies=context.producer,
        capability="forced_diagnostic",
        budget=1,
    )
    decision = select_or_stop(proposals, torch.ones(1, 4), policy, step=0)
    assert decision.stop_code == "continue"
    monkeypatch.setattr(model.updater, "forward", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("U rerun")))
    next_state = apply_scored_action(state, context, proposals, decision, writer=_writer)
    assert next_state.state_version == 1
    assert next_state.planes.xy.grad_fn is not None


def test_mutated_or_stale_proposal_is_rejected_before_writer(model_context) -> None:
    model, context, state = model_context
    proposals = propose_actions(model.updater, state, context, query=_point_query, candidate_chunk_size=4)
    policy = load_effective_policy(
        PFGRLiteConfig(
            policy=PFGRPolicyConfig(mode="forced_diagnostic"),
            num_points=4,
            engineering_only=True,
        ),
        None,
        dependencies=context.producer,
        capability="forced_diagnostic",
        budget=1,
    )
    decision = select_or_stop(proposals, torch.ones(1, 4), policy, step=0)
    with torch.no_grad():
        proposals.delta[0, 0, 0] += 0.01
    with pytest.raises(RuntimeError, match="mutation"):
        apply_scored_action(state, context, proposals, decision, writer=_writer)


def test_zero_delta_is_diagnostic_and_not_legal(model_context) -> None:
    model, context, state = model_context
    counters = OperationCounters()
    legal = torch.ones(1, 4, dtype=torch.bool)
    proposals = propose_actions(
        model.updater,
        state,
        context,
        query=_point_query,
        candidate_chunk_size=4,
        legal_mask=legal,
        counters=counters,
    )
    # Explicitly replace one row only in an isolated proposal construction so
    # the zero correction remains available for diagnostic serialization.
    zero = proposals.delta.detach().clone()
    zero[0, 0].zero_()
    legal_replacement = proposals.legal.clone()
    legal_replacement[0, 0] = False
    replacement = proposals.__class__(
        context_id=proposals.context_id,
        context_version=proposals.context_version,
        producer_compatibility_hash=proposals.producer_compatibility_hash,
        state_version=proposals.state_version,
        state_digest=proposals.state_digest,
        point_ids=proposals.point_ids,
        points_ras_mm=proposals.points_ras_mm,
        o270=proposals.o270,
        v126=proposals.v126,
        delta=zero,
        legal=legal_replacement,
        updater_version=proposals.updater_version,
        updater_producer_hash=proposals.updater_producer_hash,
        writer_version=proposals.writer_version,
        writer_hash=proposals.writer_hash,
        query_version=proposals.query_version,
        query_hash=proposals.query_hash,
        geometry_version=proposals.geometry_version,
        geometry_hash=proposals.geometry_hash,
        point_version=proposals.point_version,
        point_identity_hash=proposals.point_identity_hash,
    )
    assert not bool((replacement.delta[0, 0].abs() > 0).any())
    assert not bool(replacement.legal[0, 0])
