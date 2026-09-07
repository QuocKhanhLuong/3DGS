from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_teacher import _fixture
from torch import nn

from smagm.features.point_guided.contracts import VolumeGeometry
from smagm.features.point_guided.pfgr_lite.config import EffectTeacherConfig
from smagm.features.point_guided.pfgr_lite.footprint import PFGRQueryLattice
from smagm.features.point_guided.pfgr_lite.provenance import (
    canonical_digest,
    module_state_digest,
)
from smagm.features.point_guided.pfgr_lite.sparse_write import (
    build_footprint,
    query_write_delta,
    reference_full_write,
)
from smagm.features.point_guided.pfgr_lite.teacher import (
    _sample_iid_voxels,
    clear_teacher_cache,
    measure_actions,
    validate_target,
)
from smagm.features.point_guided.pfgr_lite.types import (
    ActionProposal,
    CompletedBehaviorTrace,
    OperationCounters,
    PFGRState,
    ProducerCompatibility,
)
from smagm.features.point_guided.spectral_query import FeatureGridGeometry
from smagm.features.point_guided.state_init import DynamicTriPlanes


class _FirstChannelDecoder(nn.Module):
    """Responsive scalar decoder used to make sign/support failures visible."""

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.mlp = nn.Linear(96, 1, bias=False, dtype=dtype)
        with torch.no_grad():
            self.mlp.weight.zero_()
            self.mlp.weight[0, 0] = 1.0


def _producer_for(
    decoder: nn.Module, teacher: EffectTeacherConfig
) -> ProducerCompatibility:
    """Build a complete producer envelope with actual decoder/label hashes."""

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
            "candidate_geometry_hash",
        )
    }
    hashes["decoder_hash"] = module_state_digest(decoder)
    hashes["writer_hash"] = canonical_digest("compact-writeback-4mm-v1")
    hashes["label_definition_hash"] = canonical_digest(
        {
            "definition": teacher.label_definition,
            "rho": teacher.rho,
            "epsilon": teacher.epsilon,
            "mask_definition": teacher.mask_definition,
            "global_mask_denominator": "sum(mask)>0_fixed_subject_v1",
        },
        prefix="pfgr-lite-label-definition-v1|",
    )
    return ProducerCompatibility(**hashes)


def _full_ids(shape: tuple[int, int, int]) -> torch.Tensor:
    depth, height, width = shape
    flat = torch.arange(depth * height * width, dtype=torch.long)
    return torch.stack(
        (
            flat // (height * width),
            (flat % (height * width)) // width,
            flat % width,
        ),
        dim=-1,
    )


def _local_global_fixture():
    """Construct the locked <=0.1 local-positive/global-negative witness."""

    dtype = torch.float64
    geometry = VolumeGeometry.from_spacing((33, 9, 9), (1.0, 1.0, 1.0))
    feature = FeatureGridGeometry(
        geometry,
        geometry,
        "conv1_pre_maxpool",
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
        ("synthetic",),
    )
    lattice = PFGRQueryLattice.build(
        geometry, feature, query_dtype=dtype, build_chunk_size=31
    )
    state_planes = DynamicTriPlanes(
        torch.zeros((1, 32, 9, 9), dtype=dtype),
        torch.zeros((1, 32, 33, 9), dtype=dtype),
        torch.zeros((1, 32, 33, 9), dtype=dtype),
    )
    decoder = _FirstChannelDecoder(dtype)
    teacher = EffectTeacherConfig(mode="exact_footprint")
    producer = _producer_for(decoder, teacher)
    state = PFGRState(state_planes, "ctx-local-global", producer=producer)
    point = torch.tensor((4.0, 4.0, 16.0), dtype=dtype)
    delta = torch.zeros((96,), dtype=dtype)
    delta[0] = 0.1
    action = ActionProposal(
        context_id=state.context_id,
        context_version="pfgr-lite-types-v1",
        producer_compatibility_hash=producer.digest,
        state_version=state.state_version,
        state_digest=state.state_digest,
        point_id=0,
        point_ras_mm=point,
        o270=torch.zeros((270,), dtype=dtype),
        v126=torch.zeros((126,), dtype=dtype),
        delta=delta,
        legal=True,
        updater_version="update-net-270-128-96-v1",
        updater_producer_hash=producer.updater_hash,
        writer_version="compact-writeback-4mm-v1",
        writer_hash=producer.writer_hash,
        query_version=lattice.query_version,
        query_hash=lattice.geometry_hash,
        geometry_version="pfgr-lite-static-geometry-v1",
        geometry_hash=lattice.geometry_hash,
        point_version="point-candidate-geometry-v1",
        point_identity_hash="point-identity-local-global",
        action_id="action-local-global",
    )
    trace = CompletedBehaviorTrace(state.context_id, states=(state,))
    after = reference_full_write(lattice, state_planes, action)
    ids = _full_ids(geometry.shape_dhw)
    before_values = lattice.query(state_planes, ids, chunk_size=37)[:, 0]
    after_values = lattice.query(after, ids, chunk_size=37)[:, 0]
    sphere = (ids.to(dtype) - torch.tensor((16.0, 4.0, 4.0), dtype=dtype)).square().sum(
        dim=1
    ) <= 16.0
    target = torch.zeros((1, *geometry.shape_dhw), dtype=dtype)
    target.reshape(-1)[sphere] = after_values[sphere]
    return (
        geometry,
        feature,
        lattice,
        state,
        action,
        trace,
        decoder,
        target,
        sphere,
        before_values,
        after_values,
    )


def test_noop_gain_is_exactly_zero() -> None:
    output, geometry, lattice, _, action, trace, decoder = _fixture()
    noop = replace(action, delta=torch.zeros_like(action.delta), action_digest=None)
    target_context = validate_target(
        "ctx",
        torch.zeros((1, *output.shape_dhw), dtype=torch.float64),
        None,
        output_geometry=output,
        feature_geometry=geometry,
        lattice=lattice,
        engineering_only=True,
    )
    label = measure_actions(
        trace,
        [noop],
        target_context,
        decoder,
        EffectTeacherConfig(mode="exact_footprint"),
    )[0]
    assert label.raw_gain == 0.0
    assert label.benefit == 0.0
    assert label.harm == 0.0


def test_harmful_gain_is_signed_and_not_clipped() -> None:
    output, geometry, lattice, original_state, action, _, _ = _fixture()
    zero_planes = DynamicTriPlanes(
        torch.zeros_like(original_state.planes.xy),
        torch.zeros_like(original_state.planes.xz),
        torch.zeros_like(original_state.planes.yz),
    )
    decoder = _FirstChannelDecoder(torch.float64)
    producer = replace(
        original_state.producer,
        decoder_hash=module_state_digest(decoder),
    )
    state = PFGRState(zero_planes, "ctx", producer=producer)
    harmful = replace(
        action,
        producer_compatibility_hash=producer.digest,
        state_digest=state.state_digest,
        delta=torch.ones_like(action.delta),
        action_digest=None,
    )
    from smagm.features.point_guided.pfgr_lite.types import CompletedBehaviorTrace

    trace = CompletedBehaviorTrace("ctx", states=(state,))
    target_context = validate_target(
        "ctx",
        torch.zeros((1, *output.shape_dhw), dtype=torch.float64),
        None,
        output_geometry=output,
        feature_geometry=geometry,
        lattice=lattice,
        engineering_only=True,
    )
    label = measure_actions(
        trace,
        [harmful],
        target_context,
        decoder,
        EffectTeacherConfig(mode="exact_footprint"),
    )[0]
    assert label.raw_gain < 0.0
    assert label.harm > label.benefit
    assert label.raw_gain == pytest.approx(label.benefit - label.harm, abs=1e-12)


def test_signed_gain_telescopes_for_sequential_reference_writes() -> None:
    output, _, lattice, state, action, _, decoder = _fixture()
    all_ids = torch.tensor(
        [
            (d, h, w)
            for d in range(output.shape_dhw[0])
            for h in range(output.shape_dhw[1])
            for w in range(output.shape_dhw[2])
        ],
        dtype=torch.long,
    )
    target = torch.randn((1, *output.shape_dhw), dtype=torch.float64)
    before_prediction = decoder.mlp(
        lattice.query(state.planes, all_ids, chunk_size=8)
    ).reshape(-1)
    state_after = reference_full_write(lattice, state.planes, action)
    after_prediction = decoder.mlp(
        lattice.query(state_after, all_ids, chunk_size=8)
    ).reshape(-1)
    # One action's global Charbonnier difference is exactly the route delta;
    # composing two actions preserves the same telescoping identity.
    action2 = replace(
        action,
        action_id="action-1",
        point_id=1,
        delta=-action.delta,
        action_digest=None,
    )
    state_after2 = reference_full_write(lattice, state_after, action2)
    final_prediction = decoder.mlp(
        lattice.query(state_after2, all_ids, chunk_size=8)
    ).reshape(-1)
    def rho(values: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(values.square() + 1e-6)
    first_gain = (
        rho(before_prediction - target.reshape(-1))
        - rho(after_prediction - target.reshape(-1))
    ).mean()
    second_gain = (
        rho(after_prediction - target.reshape(-1))
        - rho(final_prediction - target.reshape(-1))
    ).mean()
    cumulative = first_gain + second_gain
    total = (
        rho(before_prediction - target.reshape(-1))
        - rho(final_prediction - target.reshape(-1))
    ).mean()
    assert torch.allclose(cumulative, total, atol=1e-10, rtol=1e-9)


def test_fixed_q_plane_mixture_matches_enumerated_probability_law() -> None:
    _, _, lattice, _, action, _, _ = _fixture()
    footprint = build_footprint(lattice, action, chunk_size=8)
    draws, probabilities, planes = _sample_iid_voxels(
        footprint, q_draws=20_000, seed=11
    )
    counts = torch.tensor(
        [int(value.numel()) for value in footprint._pfgr_plane_voxel_linear],
        dtype=torch.float64,
    )
    expected_plane = counts / counts.sum()
    empirical_plane = torch.stack(
        [(planes == index).to(torch.float64).mean() for index in range(3)]
    )
    assert torch.allclose(empirical_plane, expected_plane, atol=0.02, rtol=0.0)
    assert bool((probabilities > 0.0).all())
    assert int(torch.unique(draws).numel()) <= draws.numel()


def test_teacher_counters_report_real_work_and_cache_reuse() -> None:
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
    counters = OperationCounters()
    labels = measure_actions(
        trace,
        [action],
        context,
        decoder,
        EffectTeacherConfig(mode="iid_fixed_q", q_draws=8),
        counters=counters,
        chunk_size=4,
    )
    assert labels[0].q_draws == 8
    assert counters.candidate_labels == 1
    assert counters.target_validations == 0
    assert counters.sampled_draws == 8
    assert (
        counters.before_decoder_outputs
        == counters.after_decoder_outputs
        == counters.unique_decoded_queries
    )
    assert counters.decoder_calls == counters.mlp_calls
    assert counters.cache_misses == 1


def test_same_action_local_positive_but_global_footprint_negative() -> None:
    (
        geometry,
        feature,
        lattice,
        _state,
        action,
        trace,
        decoder,
        target,
        sphere,
        before_values,
        after_values,
    ) = _local_global_fixture()
    context = validate_target(
        trace.context_id,
        target,
        None,
        output_geometry=geometry,
        feature_geometry=feature,
        lattice=lattice,
        completed_trace=trace,
        engineering_only=True,
    )
    label = measure_actions(
        trace,
        [action],
        context,
        decoder,
        EffectTeacherConfig(mode="exact_footprint"),
        chunk_size=37,
    )[0]
    def rho(value: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(value.square() + 1e-6)
    local_gain = (
        rho(before_values[sphere] - target.reshape(-1)[sphere])
        - rho(after_values[sphere] - target.reshape(-1)[sphere])
    ).mean()
    assert float(local_gain) > 0.0
    assert label.raw_gain < 0.0
    assert label.harm > label.benefit


def test_teacher_labels_telescope_actual_sequential_states() -> None:
    output, geometry, lattice, state0, action1, _, decoder = _fixture()
    target = torch.randn((1, *output.shape_dhw), dtype=torch.float64)
    context = validate_target(
        "ctx",
        target,
        None,
        output_geometry=output,
        feature_geometry=geometry,
        lattice=lattice,
        engineering_only=True,
    )
    state1_planes = reference_full_write(lattice, state0.planes, action1)
    state1 = PFGRState(state1_planes, "ctx", producer=state0.producer)
    action2 = replace(
        action1,
        action_id="action-sequential-2",
        point_id=1,
        state_version=state1.state_version,
        state_digest=state1.state_digest,
        delta=-action1.delta,
        action_digest=None,
    )
    label1 = measure_actions(
        CompletedBehaviorTrace("ctx", states=(state0,)),
        [action1],
        context,
        decoder,
        EffectTeacherConfig(mode="exact_footprint"),
        chunk_size=5,
    )[0]
    label2 = measure_actions(
        CompletedBehaviorTrace("ctx", states=(state1,)),
        [action2],
        context,
        decoder,
        EffectTeacherConfig(mode="exact_footprint"),
        chunk_size=5,
    )[0]
    ids = _full_ids(output.shape_dhw)
    def rho(value: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(value.square() + 1e-6)
    prediction0 = decoder.mlp(lattice.query(state0.planes, ids, chunk_size=5)).reshape(
        -1
    )
    state2 = reference_full_write(lattice, state1.planes, action2)
    prediction2 = decoder.mlp(lattice.query(state2, ids, chunk_size=5)).reshape(-1)
    dense_total = (
        rho(prediction0 - target.reshape(-1)) - rho(prediction2 - target.reshape(-1))
    ).mean()
    assert label1.raw_gain + label2.raw_gain == pytest.approx(
        float(dense_total.detach()), abs=1e-10, rel=1e-9
    )


def test_fixed_q_exact_expectation_from_enumerated_mixture_law() -> None:
    output, geometry, lattice, state, action, _trace, decoder = _fixture()
    target = torch.randn((1, *output.shape_dhw), dtype=torch.float64)
    context = validate_target(
        "ctx",
        target,
        None,
        output_geometry=output,
        feature_geometry=geometry,
        lattice=lattice,
        engineering_only=True,
    )
    footprint = build_footprint(lattice, action, chunk_size=5)
    ids = footprint.voxel_ids_dhw
    prediction_before = decoder.mlp(
        lattice.query(state.planes, ids, chunk_size=5)
    ).reshape(-1)
    prediction_after = decoder.mlp(
        lattice.query(state.planes, ids, chunk_size=5)
        + query_write_delta(lattice, footprint, ids, action.delta, chunk_size=5)
    ).reshape(-1)
    target_rows = context.gather_target(ids, dtype=torch.float64)
    mask_rows = context.gather_mask(ids)
    differences = torch.sqrt(
        (prediction_before - target_rows).square() + 1e-6
    ) - torch.sqrt((prediction_after - target_rows).square() + 1e-6)
    exact_gain = float(
        (differences * mask_rows.to(torch.float64)).sum().item() / context.M
    )
    plane_rows = tuple(footprint._pfgr_plane_voxel_linear)
    counts = [int(rows.numel()) for rows in plane_rows]
    total = float(sum(counts))
    union = footprint._pfgr_union_linear
    multiplicity = footprint.multiplicity.to(torch.float64)
    linear = (
        ids[:, 0] * output.shape_dhw[1] * output.shape_dhw[2]
        + ids[:, 1] * output.shape_dhw[2]
        + ids[:, 2]
    )
    position = torch.searchsorted(union, linear)
    probabilities = multiplicity[position] / total
    contribution = (
        differences * mask_rows.to(torch.float64) / (float(context.M) * probabilities)
    )
    expectation = sum(
        float((contribution[torch.isin(linear, rows)] * (1.0 / total)).sum().item())
        for rows in plane_rows
        if rows.numel()
    )
    assert expectation == pytest.approx(exact_gain, abs=1e-10, rel=1e-9)
    assert bool((probabilities > 0).all())


def test_fixed_q_sampler_allows_empty_plane_with_positive_union_support() -> None:
    footprint = SimpleNamespace(
        _pfgr_plane_voxel_linear=(
            torch.empty((0,), dtype=torch.long),
            torch.tensor((0, 1), dtype=torch.long),
            torch.tensor((1, 2), dtype=torch.long),
        ),
        _pfgr_union_linear=torch.tensor((0, 1, 2), dtype=torch.long),
        multiplicity=torch.tensor((1, 2, 1), dtype=torch.long),
    )
    draws, probabilities, planes = _sample_iid_voxels(footprint, q_draws=128, seed=7)
    assert draws.shape == probabilities.shape == planes.shape == (128,)
    assert bool((planes != 0).all())
    assert bool((probabilities > 0).all())
