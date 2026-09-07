from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from smagm.features.point_guided import PointGuidedConfig
from smagm.features.point_guided.contracts import VolumeGeometry
from smagm.features.point_guided.decoder import ImplicitTriPlaneDecoder
from smagm.features.point_guided.pfgr_lite import teacher as teacher_module
from smagm.features.point_guided.pfgr_lite.config import (
    EffectTeacherConfig,
    PFGRLiteConfig,
    PFGRPolicyConfig,
)
from smagm.features.point_guided.pfgr_lite.footprint import PFGRQueryLattice
from smagm.features.point_guided.pfgr_lite.inference import run_pfgr_inference
from smagm.features.point_guided.pfgr_lite.model import PFGRLiteModel
from smagm.features.point_guided.pfgr_lite.policy import load_effective_policy
from smagm.features.point_guided.pfgr_lite.provenance import (
    ProducerCompatibility,
    canonical_digest,
    module_state_digest,
)
from smagm.features.point_guided.pfgr_lite.sparse_write import (
    make_action_writer,
    make_point_query,
    make_support_legal_mask,
    reference_full_write,
)
from smagm.features.point_guided.pfgr_lite.teacher import (
    DiagnosticGainResult,
    ValidatedTargetContext,
    clear_target_validation_stats,
    clear_teacher_cache,
    measure_actions,
    measure_diagnostic_actions,
    target_validation_stats,
    validate_target,
)
from smagm.features.point_guided.pfgr_lite.types import (
    ActionProposal,
    CompletedBehaviorTrace,
    OperationCounters,
    PFGRState,
)
from smagm.features.point_guided.spectral_query import FeatureGridGeometry
from smagm.features.point_guided.state_init import DynamicTriPlanes


def _fixture(dtype: torch.dtype = torch.float64):
    output = VolumeGeometry.from_spacing((5, 5, 5), (1.0, 1.0, 1.0))
    feature_volume = VolumeGeometry.from_spacing((3, 3, 3), (2.0, 2.0, 2.0))
    geometry = FeatureGridGeometry(
        output,
        feature_volume,
        "conv1_pre_maxpool",
        (2.0, 2.0, 2.0),
        (0.0, 0.0, 0.0),
        ("synthetic",),
    )
    lattice = PFGRQueryLattice.build(
        output, geometry, query_dtype=dtype, build_chunk_size=9
    )
    generator = torch.Generator().manual_seed(52)
    planes = DynamicTriPlanes(
        torch.randn((1, 32, 3, 3), dtype=dtype, generator=generator),
        torch.randn((1, 32, 3, 3), dtype=dtype, generator=generator),
        torch.randn((1, 32, 3, 3), dtype=dtype, generator=generator),
    )
    decoder = ImplicitTriPlaneDecoder().to(dtype=dtype)
    hashes = {
        name: "x" * 64
        for name in (
            "observation_normalization_hash",
            "geometry_query_version_hash",
            "medicalnet_provenance_hash",
            "frozen_bn_hash",
            "static_head_hash",
            "semantic_head_hash",
            "point_refiner_hash",
            "spectral_projector_hash",
            "state_initializer_hash",
            "updater_hash",
            "decoder_hash",
            "writer_hash",
            "candidate_geometry_hash",
            "label_definition_hash",
        )
    }
    teacher_definition = EffectTeacherConfig()
    hashes["decoder_hash"] = module_state_digest(decoder)
    hashes["writer_hash"] = canonical_digest("compact-writeback-4mm-v1")
    hashes["label_definition_hash"] = canonical_digest(
        {
            "definition": teacher_definition.label_definition,
            "rho": teacher_definition.rho,
            "epsilon": teacher_definition.epsilon,
            "mask_definition": teacher_definition.mask_definition,
            "global_mask_denominator": "sum(mask)>0_fixed_subject_v1",
        },
        prefix="pfgr-lite-label-definition-v1|",
    )
    producer = ProducerCompatibility(**hashes)
    state = PFGRState(planes, "ctx", producer=producer)
    point = torch.tensor((2.0, 2.0, 2.0), dtype=dtype)
    action = ActionProposal(
        context_id="ctx",
        context_version="pfgr-lite-types-v1",
        producer_compatibility_hash=producer.digest,
        state_version=0,
        state_digest=state.state_digest,
        point_id=0,
        point_ras_mm=point,
        o270=torch.zeros((270,), dtype=dtype),
        v126=torch.zeros((126,), dtype=dtype),
        delta=torch.randn((96,), dtype=dtype, generator=generator),
        legal=True,
        updater_version="u-v1",
        updater_producer_hash=producer.updater_hash,
        writer_version="compact-writeback-4mm-v1",
        writer_hash=hashes["writer_hash"],
        query_version="pfgr-lite-query-lattice-v1",
        query_hash=lattice.geometry_hash,
        geometry_version="g-v1",
        geometry_hash=lattice.geometry_hash,
        point_version="p-v1",
        point_identity_hash="p-hash",
        action_id="action-0",
    )
    trace = CompletedBehaviorTrace("ctx", states=(state,))
    return output, geometry, lattice, state, action, trace, decoder


def test_exact_teacher_gain_matches_full_dense_reference() -> None:
    output, geometry, lattice, state, action, trace, decoder = _fixture()
    target = torch.randn((1, *output.shape_dhw), dtype=torch.float64)
    mask = torch.ones_like(target, dtype=torch.bool)
    target_context = validate_target(
        "ctx",
        target,
        mask,
        output_geometry=output,
        feature_geometry=geometry,
        lattice=lattice,
        engineering_only=True,
    )
    label = measure_actions(
        trace,
        [action],
        target_context,
        decoder,
        EffectTeacherConfig(mode="exact_footprint"),
        chunk_size=5,
    )[0]
    ids = torch.tensor(
        [
            (d, h, w)
            for d in range(output.shape_dhw[0])
            for h in range(output.shape_dhw[1])
            for w in range(output.shape_dhw[2])
        ],
        dtype=torch.long,
    )
    before = decoder.mlp(lattice.query(state.planes, ids, chunk_size=5)).reshape(-1)
    after_state = reference_full_write(lattice, state.planes, action)
    after = decoder.mlp(lattice.query(after_state, ids, chunk_size=5)).reshape(-1)
    error_before = torch.sqrt((before - target.reshape(-1)).square() + 1e-6)
    error_after = torch.sqrt((after - target.reshape(-1)).square() + 1e-6)
    dense_gain = float((error_before - error_after).mean().item())
    assert label.raw_gain == pytest.approx(dense_gain, abs=1e-10, rel=1e-9)
    assert label.raw_gain == pytest.approx(label.benefit - label.harm, abs=1e-12)


def test_diagnostic_teacher_binds_real_state_without_completed_trace() -> None:
    output, geometry, lattice, state, action, _trace, decoder = _fixture()
    target_context = validate_target(
        "ctx",
        torch.randn((1, *output.shape_dhw), dtype=torch.float64),
        None,
        output_geometry=output,
        feature_geometry=geometry,
        lattice=lattice,
        engineering_only=True,
    )
    diagnostic = measure_diagnostic_actions(
        state,
        [action],
        target_context,
        decoder,
        EffectTeacherConfig(mode="exact_footprint"),
        lattice=lattice,
        chunk_size=5,
    )
    assert len(diagnostic) == 1
    result = diagnostic[0]
    assert isinstance(result, DiagnosticGainResult)
    assert result.scope == "oracle_state"
    assert result.privileged is True
    assert result.state is state and result.action is action and result.proposal is action
    assert result.state_version == state.state_version
    assert result.label.raw_gain == pytest.approx(
        measure_actions(
            _trace,
            [action],
            target_context,
            decoder,
            EffectTeacherConfig(mode="exact_footprint"),
            lattice=lattice,
            chunk_size=5,
        )[0].raw_gain,
        abs=1e-10,
        rel=1e-9,
    )
    payload = result.as_dict()
    assert payload["schema_version"] == "pfgr-lite-diagnostic-gain-v1"
    assert "target_data" not in str(payload).lower()


def test_diagnostic_teacher_rejects_route_bound_target_context() -> None:
    output, geometry, lattice, state, action, _trace, decoder = _fixture()
    target_context = validate_target(
        "ctx",
        torch.zeros((1, *output.shape_dhw), dtype=torch.float64),
        None,
        output_geometry=output,
        feature_geometry=geometry,
        lattice=lattice,
        trace_route_hash="sealed-route",
        engineering_only=True,
    )
    with pytest.raises(ValueError, match="route-unbound"):
        measure_diagnostic_actions(
            state,
            [action],
            target_context,
            decoder,
            EffectTeacherConfig(mode="exact_footprint"),
            lattice=lattice,
            chunk_size=5,
        )


def test_diagnostic_teacher_accepts_state_version_transition_identity() -> None:
    output, geometry, lattice, state, action, _trace, decoder = _fixture()
    next_state = state.next(state.planes)
    next_action = replace(
        action,
        state_version=next_state.state_version,
        state_digest=next_state.state_digest,
        action_digest=None,
    )
    target_context = validate_target(
        "ctx",
        torch.zeros((1, *output.shape_dhw), dtype=torch.float64),
        None,
        output_geometry=output,
        feature_geometry=geometry,
        lattice=lattice,
        engineering_only=True,
    )
    result = measure_diagnostic_actions(
        next_state,
        [next_action],
        target_context,
        decoder,
        EffectTeacherConfig(mode="exact_footprint"),
        lattice=lattice,
        chunk_size=5,
    )[0]
    assert result.state_version == 1
    assert result.state_digest == next_state.state_digest
    assert result.action.state_version == 1


def test_fixed_q_keeps_complete_support_law_and_duplicate_draws() -> None:
    _, geometry, lattice, _, action, trace, decoder = _fixture()
    target_context = validate_target(
        "ctx",
        torch.zeros((1, 5, 5, 5), dtype=torch.float64),
        torch.zeros((1, 5, 5, 5), dtype=torch.bool).logical_not(),
        output_geometry=lattice.output_geometry,
        feature_geometry=geometry,
        lattice=lattice,
        engineering_only=True,
    )
    config = EffectTeacherConfig(mode="iid_fixed_q", q_draws=32)
    clear_teacher_cache()
    first = measure_actions(
        trace, [action], target_context, decoder, config, seed=17, chunk_size=7
    )[0]
    second = measure_actions(
        trace, [action], target_context, decoder, config, seed=17, chunk_size=7
    )[0]
    assert first == second
    assert first.role == "iid_fixed_q"
    assert first.q_draws == 32
    assert first.seed == second.seed
    assert first.sampler_law == "iid_fixed_q_plane_mixture_c_over_S_v1"
    assert first.standard_error is not None and first.standard_error >= 0.0
    assert first.raw_gain == pytest.approx(first.benefit - first.harm, abs=1e-12)


def test_fixed_q_uses_global_mask_denominator_without_mask_rejection() -> None:
    output, geometry, lattice, _, action, trace, decoder = _fixture(torch.float32)
    mask = torch.zeros((1, *output.shape_dhw), dtype=torch.bool)
    mask[:, 2, 2, 2] = True
    context = validate_target(
        "ctx",
        torch.zeros_like(mask, dtype=torch.float32),
        mask,
        output_geometry=output,
        feature_geometry=geometry,
        lattice=lattice,
        engineering_only=True,
    )
    counters = OperationCounters()
    label = measure_actions(
        trace,
        [action],
        context,
        decoder.float(),
        EffectTeacherConfig(mode="iid_fixed_q", q_draws=16),
        seed=3,
        counters=counters,
    )[0]
    assert label.q_draws == 16
    assert label.valid_masked_contributions <= label.q_draws
    assert counters.sampled_draws == 16
    assert label.mask_count == 1


def test_target_context_is_detached_and_mutation_guarded() -> None:
    output, geometry, lattice, *_ = _fixture()
    target = torch.zeros(
        (1, *output.shape_dhw), dtype=torch.float64, requires_grad=True
    )
    context = validate_target(
        "ctx",
        target,
        None,
        output_geometry=output,
        feature_geometry=geometry,
        lattice=lattice,
        engineering_only=True,
    )
    assert not context.target.requires_grad
    context.target[0, 0, 0, 0] = 1.0
    with pytest.raises(RuntimeError, match="mutation detected"):
        context.validate_integrity()


def test_target_hot_guard_skips_rehash_and_explicit_audit_catches_unsafe_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, geometry, lattice, _, action, trace, decoder = _fixture()
    context = validate_target(
        "ctx",
        torch.zeros((1, 5, 5, 5), dtype=torch.float64),
        None,
        output_geometry=lattice.output_geometry,
        feature_geometry=geometry,
        lattice=lattice,
        engineering_only=True,
    )
    clear_teacher_cache()
    clear_target_validation_stats()
    original_digest = teacher_module._tensor_digest
    digest_calls = 0

    def spy_digest(value: torch.Tensor) -> str:
        nonlocal digest_calls
        digest_calls += 1
        return original_digest(value)

    monkeypatch.setattr(teacher_module, "_tensor_digest", spy_digest)
    config = EffectTeacherConfig(mode="exact_footprint")
    measure_actions(trace, [action], context, decoder, config)
    measure_actions(trace, [action], context, decoder, config)
    assert digest_calls == 0
    assert target_validation_stats() == {"hot_checks": 2, "full_audits": 0}
    # ``.data`` bypasses PyTorch's version counter; the normal hot path is
    # documented not to claim detection, while an explicit full audit catches
    # the changed detached bytes before they can be used for a bank.
    context.target.data[0, 0, 0, 0] = 1.0
    with pytest.raises(RuntimeError, match="mutation detected"):
        context.validate_integrity(full_audit=True)
    assert target_validation_stats()["full_audits"] == 1


def test_target_context_rejects_empty_mask_and_shape_mismatch() -> None:
    output, _, _, *_ = _fixture()
    with pytest.raises(ValueError, match="at least one valid"):
        validate_target(
            "ctx",
            torch.zeros((1, 5, 5, 5)),
            torch.zeros((1, 5, 5, 5), dtype=torch.bool),
            engineering_only=True,
        )
    with pytest.raises(ValueError, match="shape"):
        validate_target(
            "ctx",
            torch.zeros((1, 4, 5, 5)),
            torch.ones((1, 5, 5, 5), dtype=torch.bool),
            output_geometry=output,
            engineering_only=True,
        )


def test_production_target_requires_explicit_observation_binding() -> None:
    with pytest.raises(ValueError, match="observation_context"):
        validate_target(
            "ctx",
            torch.zeros((1, 5, 5, 5)),
            None,
        )
    context = validate_target(
        "ctx",
        torch.zeros((1, 5, 5, 5)),
        None,
        engineering_only=True,
    )
    assert context.engineering_only is True
    assert context.observation_context_id is None


def test_direct_production_target_context_requires_complete_binding() -> None:
    target = torch.zeros((1, 5, 5, 5), dtype=torch.float64)
    mask = torch.ones_like(target, dtype=torch.bool)
    with pytest.raises(ValueError, match="ObservationContext binding"):
        ValidatedTargetContext(
            completed_context_id="ctx",
            target=target,
            observation_mask=mask,
            mask_count=int(mask.sum().item()),
            provenance="post_trace_target_provider_v1",
        )


def test_bound_target_rejects_replaced_mask_or_geometry() -> None:
    frontend_config = PointGuidedConfig(
        num_semantic_classes=3,
        num_points=4,
        point_candidate_multiplier=3,
        offset_hidden_channels=12,
        detach_backbone_features=False,
    )
    model = PFGRLiteModel(
        PFGRLiteConfig(num_points=4, engineering_only=True),
        frontend_config=frontend_config,
    ).eval()
    observations = torch.randn((1, 3, 9, 9, 9), dtype=torch.float32)
    observation_context = model.encode_observations(observations, None, (1.0, 1.0, 1.0))
    target = torch.zeros((1, 9, 9, 9), dtype=torch.float32)
    bound = validate_target(
        observation_context.context_id,
        target,
        observation_context.observation_mask,
        observation_context=observation_context,
    )
    assert bound.engineering_only is False
    assert bound.observation_context_id == observation_context.context_id
    replacement = observation_context.observation_mask.clone()
    replacement.reshape(-1)[0] = ~replacement.reshape(-1)[0]
    with pytest.raises(ValueError, match="does not match observation context mask"):
        validate_target(
            observation_context.context_id,
            target,
            replacement,
            observation_context=observation_context,
        )
    wrong_geometry = VolumeGeometry.from_spacing((9, 9, 9), (1.1, 1.0, 1.0))
    with pytest.raises(ValueError, match="does not match observation context"):
        validate_target(
            observation_context.context_id,
            target,
            None,
            output_geometry=wrong_geometry,
            observation_context=observation_context,
        )


def test_actual_model_w4_route_reaches_point_query_teacher() -> None:
    frontend_config = PointGuidedConfig(
        num_semantic_classes=3,
        num_points=4,
        point_candidate_multiplier=3,
        offset_hidden_channels=12,
        detach_backbone_features=False,
    )
    model = PFGRLiteModel(
        PFGRLiteConfig(num_points=4, engineering_only=True),
        frontend_config=frontend_config,
    ).eval()
    observation_context = model.encode_observations(
        torch.randn((1, 3, 9, 9, 9), dtype=torch.float32), None, (1.0, 1.0, 1.0)
    )
    lattice = PFGRQueryLattice.build(
        observation_context.geometry,
        observation_context.feature_geometry,
        query_dtype=torch.float32,
        build_chunk_size=37,
    )
    policy_config = PFGRLiteConfig(
        policy=PFGRPolicyConfig(mode="random"),
        num_points=4,
        engineering_only=True,
    )
    policy = load_effective_policy(
        policy_config,
        None,
        dependencies=observation_context.producer,
        capability="forced_diagnostic",
        budget=2,
        random_seed=19,
    )
    route = run_pfgr_inference(
        model,
        observation_context,
        policy,
        query=make_point_query(),
        writer=make_action_writer(lattice),
        legal_mask=make_support_legal_mask(lattice),
    )
    assert route.k == 2
    target_context = validate_target(
        observation_context.context_id,
        torch.randn((1, 9, 9, 9), dtype=torch.float32),
        None,
        observation_context=observation_context,
        completed_trace=route.completed_trace,
        lattice=lattice,
    )
    actions: list[ActionProposal] = []
    for proposal, decision in zip(
        route.completed_trace.proposals, route.completed_trace.decisions
    ):
        location = (proposal.point_ids[0] == decision.selected_point_id).nonzero(
            as_tuple=False
        )[0, 0]
        actions.append(proposal.row(0, int(location.item())))
    labels = measure_actions(
        route.completed_trace,
        actions,
        target_context,
        model.decoder,
        EffectTeacherConfig(mode="exact_footprint"),
        lattice=lattice,
        chunk_size=37,
        candidate_chunk_size=2,
        observation_context=observation_context,
    )
    assert len(labels) == route.k
    assert all(torch.isfinite(torch.tensor(label.raw_gain)) for label in labels)


def test_teacher_cache_hits_then_invalidates_on_decoder_mutation() -> None:
    _, geometry, lattice, _, action, trace, decoder = _fixture()
    context = validate_target(
        "ctx",
        torch.zeros((1, 5, 5, 5), dtype=torch.float64),
        None,
        output_geometry=lattice.output_geometry,
        feature_geometry=geometry,
        lattice=lattice,
        engineering_only=True,
    )
    clear_teacher_cache()
    first_counters = OperationCounters()
    measure_actions(
        trace,
        [action],
        context,
        decoder,
        EffectTeacherConfig(mode="exact_footprint"),
        counters=first_counters,
    )
    assert first_counters.cache_misses == 1 and first_counters.cache_hits == 0
    second_counters = OperationCounters()
    measure_actions(
        trace,
        [action],
        context,
        decoder,
        EffectTeacherConfig(mode="exact_footprint"),
        counters=second_counters,
    )
    assert second_counters.cache_hits == 1 and second_counters.cache_misses == 0
    with torch.no_grad():
        decoder.mlp[0].weight[0, 0] += 0.125
    third_counters = OperationCounters()
    with pytest.raises(ValueError, match="decoder weights"):
        measure_actions(
            trace,
            [action],
            context,
            decoder,
            EffectTeacherConfig(mode="exact_footprint"),
            counters=third_counters,
        )


def test_measure_actions_rejects_context_or_trace_identity_mismatch() -> None:
    _, geometry, lattice, _, action, trace, decoder = _fixture()
    wrong_context = validate_target(
        "other",
        torch.zeros((1, 5, 5, 5), dtype=torch.float64),
        None,
        output_geometry=lattice.output_geometry,
        feature_geometry=geometry,
        lattice=lattice,
        engineering_only=True,
    )
    with pytest.raises(ValueError, match="IDs"):
        measure_actions(
            trace,
            [action],
            wrong_context,
            decoder,
            EffectTeacherConfig(mode="exact_footprint"),
        )
    unsealed = replace(trace, sealed=True)
    object.__setattr__(unsealed, "sealed", False)
    with pytest.raises(ValueError, match="sealed"):
        measure_actions(
            unsealed,
            [action],
            wrong_context,
            decoder,
            EffectTeacherConfig(mode="exact_footprint"),
        )
    bad_updater = replace(action, updater_producer_hash="stale", action_digest=None)
    with pytest.raises(ValueError, match="updater producer"):
        measure_actions(
            trace,
            [bad_updater],
            validate_target(
                "ctx",
                torch.zeros((1, 5, 5, 5), dtype=torch.float64),
                None,
                output_geometry=lattice.output_geometry,
                feature_geometry=geometry,
                lattice=lattice,
                engineering_only=True,
            ),
            decoder,
            EffectTeacherConfig(mode="exact_footprint"),
        )


def test_exact_teacher_large_union_uses_streaming_cache_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, geometry, lattice, _, action, trace, decoder = _fixture()
    context = validate_target(
        "ctx",
        torch.zeros((1, *output.shape_dhw), dtype=torch.float64),
        None,
        output_geometry=output,
        feature_geometry=geometry,
        lattice=lattice,
        engineering_only=True,
    )
    clear_teacher_cache()
    monkeypatch.setattr(teacher_module, "PROBE_CACHE_MAX_ROWS", 1)
    counters = OperationCounters()
    label = measure_actions(
        trace,
        [action],
        context,
        decoder,
        EffectTeacherConfig(mode="exact_footprint"),
        chunk_size=7,
        counters=counters,
    )[0]
    assert label.role == "exact_footprint"
    assert counters.cache_misses == 1 and counters.cache_hits == 0
    assert counters.before_decoder_outputs == counters.after_decoder_outputs
    assert counters.before_decoder_outputs == counters.unique_decoded_queries
    assert teacher_module.teacher_cache_stats()["entries"] == 0


def test_candidate_chunk_batches_nonlinear_decoder() -> None:
    _, geometry, lattice, _, action, trace, decoder = _fixture()
    action_two = replace(
        action,
        action_id="action-1",
        point_ras_mm=torch.tensor((1.0, 1.0, 1.0), dtype=action.point_ras_mm.dtype),
        delta=action.delta * 0.5,
        action_digest=None,
    )
    context = validate_target(
        "ctx",
        torch.zeros((1, 5, 5, 5), dtype=torch.float64),
        None,
        output_geometry=lattice.output_geometry,
        feature_geometry=geometry,
        lattice=lattice,
        engineering_only=True,
    )
    config = EffectTeacherConfig(mode="exact_footprint")
    clear_teacher_cache()
    serial_counters = OperationCounters()
    serial = measure_actions(
        trace,
        [action, action_two],
        context,
        decoder,
        config,
        chunk_size=7,
        candidate_chunk_size=1,
        counters=serial_counters,
    )
    clear_teacher_cache()
    batched_counters = OperationCounters()
    batched = measure_actions(
        trace,
        [action, action_two],
        context,
        decoder,
        config,
        chunk_size=7,
        candidate_chunk_size=2,
        counters=batched_counters,
    )
    assert [label.raw_gain for label in batched] == pytest.approx(
        [label.raw_gain for label in serial], abs=1e-10, rel=1e-9
    )
    assert batched_counters.before_decoder_outputs == serial_counters.before_decoder_outputs
    assert batched_counters.after_decoder_outputs == serial_counters.after_decoder_outputs
    # Both candidates are decoded through the same bounded nonlinear batch;
    # decoder work is no greater than serial reference work.
    assert batched_counters.cache_misses == 2
    assert batched_counters.decoder_calls < serial_counters.decoder_calls
    assert batched_counters.bytes_copied > 0
    clear_teacher_cache()
    chunk4 = measure_actions(
        trace,
        [action, action_two],
        context,
        decoder,
        config,
        chunk_size=7,
        candidate_chunk_size=4,
    )
    assert [label.raw_gain for label in chunk4] == pytest.approx(
        [label.raw_gain for label in serial], abs=1e-10, rel=1e-9
    )

    fixed_config = EffectTeacherConfig(mode="iid_fixed_q", q_draws=32)
    clear_teacher_cache()
    fixed_serial = measure_actions(
        trace,
        [action, action_two],
        context,
        decoder,
        fixed_config,
        chunk_size=7,
        candidate_chunk_size=1,
        seed=19,
    )
    clear_teacher_cache()
    fixed_batched = measure_actions(
        trace,
        [action, action_two],
        context,
        decoder,
        fixed_config,
        chunk_size=7,
        candidate_chunk_size=2,
        seed=19,
    )
    assert [label.raw_gain for label in fixed_batched] == pytest.approx(
        [label.raw_gain for label in fixed_serial], abs=1e-10, rel=1e-9
    )
