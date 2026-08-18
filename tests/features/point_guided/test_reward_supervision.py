"""Focused Gate-E E2--E4 tests for compact counterfactual supervision."""

from __future__ import annotations

import copy

import pytest
import torch

from smagm.features.point_guided.contracts import VolumeGeometry
from smagm.features.point_guided.decoder import ImplicitTriPlaneDecoder
from smagm.features.point_guided.reward import GateBDescriptorContext
from smagm.features.point_guided.reward_supervision import (
    CounterfactualConfig,
    build_local_support_samples,
    build_spill_samples,
    counterfactual_reward_supervision,
    pairwise_reward_ranking_loss,
    sample_counterfactual_candidates,
    spill_aware_reward_target,
)
from smagm.features.point_guided.sampling import ras_mm_to_voxel_dhw
from smagm.features.point_guided.spectral_query import FeatureGridGeometry
from smagm.features.point_guided.state_init import DynamicTriPlanes
from smagm.features.point_guided.trajectory import AdaptiveRewardCostTrajectory
from smagm.features.point_guided.trajectory_cost import TrajectoryConfig, route_utility


def _feature_geometry(shape_dhw: tuple[int, int, int] = (11, 11, 11)) -> FeatureGridGeometry:
    source = VolumeGeometry.from_spacing(shape_dhw, (1.0, 1.0, 1.0))
    return FeatureGridGeometry(
        source_geometry=source,
        feature_geometry=source,
        tap="conv1_pre_maxpool",
        feature_to_source_scale_dhw=(1.0, 1.0, 1.0),
        feature_to_source_offset_dhw=(0.0, 0.0, 0.0),
        operator_chain=("synthetic_identity_feature_grid",),
    )


def _state(
    geometry: FeatureGridGeometry,
    *,
    requires_grad: bool = False,
) -> DynamicTriPlanes:
    depth, height, width = geometry.shape_dhw
    generator = torch.Generator().manual_seed(13)
    planes = (
        torch.randn(1, 32, height, width, generator=generator),
        torch.randn(1, 32, depth, width, generator=generator),
        torch.randn(1, 32, depth, height, generator=generator),
    )
    if requires_grad:
        planes = tuple(plane.requires_grad_() for plane in planes)
    return DynamicTriPlanes(xy=planes[0], xz=planes[1], yz=planes[2])


def _trajectory(
    *,
    lambda_travel: float = 0.1,
    lambda_overlap: float = 0.2,
    lambda_step: float = 0.3,
) -> AdaptiveRewardCostTrajectory:
    return AdaptiveRewardCostTrajectory(
        TrajectoryConfig(
            lambda_travel=lambda_travel,
            lambda_overlap=lambda_overlap,
            lambda_step=lambda_step,
            k_max=1,
            selection_temperature=0.7,
            write_scale=0.1,
        )
    ).eval()


def _counterfactual_inputs(
    *,
    requires_grad: bool = False,
) -> tuple[
    DynamicTriPlanes,
    FeatureGridGeometry,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    GateBDescriptorContext,
    torch.Tensor,
]:
    geometry = _feature_geometry()
    state = _state(geometry, requires_grad=requires_grad)
    feature_dhw = torch.tensor(
        [[[5.0, 5.0, 5.0], [4.0, 5.0, 5.0], [5.0, 4.0, 5.0], [5.0, 5.0, 4.0], [2.0, 2.0, 2.0]]]
    )
    points = geometry.feature_dhw_to_ras_mm(feature_dhw)
    point_semantic = torch.tensor(
        [[[0.2, 0.3, 0.5], [0.3, 0.4, 0.3], [0.4, 0.4, 0.2], [0.5, 0.2, 0.3], [0.6, 0.2, 0.2]]]
    )
    f_spec = torch.zeros(1, points.shape[1], 168)
    reliability = torch.full((1, points.shape[1], 3), 1.0 / 3.0)
    descriptors = GateBDescriptorContext(
        q_xy=torch.zeros(1, points.shape[1], 24),
        q_xz=torch.zeros(1, points.shape[1], 24),
        q_yz=torch.zeros(1, points.shape[1], 24),
    )
    target = torch.zeros(1, 1, *geometry.source_geometry.shape_dhw)
    return state, geometry, points, point_semantic, f_spec, descriptors, target


def _counterfactual_result(
    trajectory: AdaptiveRewardCostTrajectory,
    decoder: ImplicitTriPlaneDecoder,
    state: DynamicTriPlanes,
    geometry: FeatureGridGeometry,
    points: torch.Tensor,
    point_semantic: torch.Tensor,
    f_spec: torch.Tensor,
    descriptors: GateBDescriptorContext,
    target: torch.Tensor,
    *,
    generator: torch.Generator,
    reward_ranking_weight: float = 0.0,
    reward_ranking_min_target_gap: float = 0.001,
):
    return counterfactual_reward_supervision(
        trajectory,
        decoder,
        state,
        points,
        point_semantic,
        f_spec,
        torch.full((1, points.shape[1], 3), 1.0 / 3.0),
        descriptors,
        geometry,
        geometry.source_geometry,
        target,
        selected_indices=torch.tensor([0], dtype=torch.long),
        config=CounterfactualConfig(
            counterfactual_candidates=3,
            high_candidate_count=1,
            random_candidate_count=1,
            spill_weight_beta=0.75,
            spill_sample_count=3,
            reward_ranking_weight=reward_ranking_weight,
            reward_ranking_min_target_gap=reward_ranking_min_target_gap,
        ),
        generator=generator,
    )


def test_counterfactual_candidate_mix_keeps_selected_high_and_seeded_random_without_duplicates() -> None:
    predicted_reward = torch.tensor(
        [
            [0.11, 0.99, 0.88, 0.77, 0.66, 0.55, 0.44, 0.33],
            [0.71, 0.61, 0.51, 0.41, 0.31, 0.21, 0.91, 0.81],
        ]
    )
    selected = torch.tensor([1, 6], dtype=torch.long)
    config = CounterfactualConfig(
        counterfactual_candidates=5,
        high_candidate_count=2,
        random_candidate_count=2,
        spill_sample_count=0,
    )

    first = sample_counterfactual_candidates(
        predicted_reward,
        selected,
        config,
        generator=torch.Generator().manual_seed(2026),
    )
    repeat = sample_counterfactual_candidates(
        predicted_reward,
        selected,
        config,
        generator=torch.Generator().manual_seed(2026),
    )

    assert torch.equal(first.indices, repeat.indices)
    assert torch.equal(first.selected_mask, repeat.selected_mask)
    assert first.indices.shape == (2, 5)
    assert first.indices[first.selected_mask].tolist() == selected.tolist()
    assert first.indices[first.high_reward_mask].reshape(2, 2).tolist() == [[2, 3], [7, 0]]
    assert first.random_mask.sum(dim=1).tolist() == [2, 2]
    for row in first.indices:
        assert torch.unique(row).numel() == row.numel()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"high_candidate_count": 0}, "high_candidate_count"),
        ({"random_candidate_count": 0}, "random_candidate_count"),
        ({"spill_sample_count": 1}, "XY/XZ/YZ"),
        ({"spill_sample_count": 2}, "XY/XZ/YZ"),
    ],
)
def test_counterfactual_config_keeps_the_locked_candidate_mix_and_nonzero_three_fibre_spill(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CounterfactualConfig(**kwargs)

    # The explicit zero-spill synthetic/ablation mode remains legal.
    assert CounterfactualConfig(spill_sample_count=0).spill_sample_count == 0


def test_spill_aware_target_is_detached_and_only_beta_penalizes_spill_regression() -> None:
    local_before = torch.tensor([[4.0, 4.0]], requires_grad=True)
    local_after = torch.tensor([[2.0, 1.0]], requires_grad=True)
    spill_before = torch.tensor([[1.0, 0.0]], requires_grad=True)
    spill_after = torch.tensor([[5.0, 0.0]], requires_grad=True)

    no_penalty = spill_aware_reward_target(
        local_before,
        local_after,
        spill_before,
        spill_after,
        spill_weight_beta=0.0,
    )
    penalized = spill_aware_reward_target(
        local_before,
        local_after,
        spill_before,
        spill_after,
        spill_weight_beta=0.5,
    )

    assert not no_penalty.requires_grad
    assert not penalized.requires_grad
    assert torch.allclose(no_penalty, torch.tensor([[2.0 / 4.001, 3.0 / 4.001]]))
    assert torch.allclose(penalized, torch.tensor([[0.0, 3.0 / 4.001]]))
    assert local_before.grad is None
    assert local_after.grad is None
    assert spill_before.grad is None
    assert spill_after.grad is None


def test_local_support_is_physical_four_mm_and_spill_keeps_only_collateral_fibres() -> None:
    geometry = _feature_geometry((13, 13, 13))
    centre = geometry.feature_dhw_to_ras_mm(torch.tensor([[[6.0, 6.0, 6.0]]]))
    local = build_local_support_samples(centre[:, 0], geometry.source_geometry)
    local_valid = local.valid_mask.squeeze(-1)
    local_distance = torch.linalg.vector_norm(local.points_ras_mm - centre, dim=-1)

    assert bool(local_valid.any())
    assert bool((local_distance[local_valid] <= 4.0 + 1e-6).all())
    assert bool((local_distance[local_valid] >= 0.0).all())

    spill = build_spill_samples(
        centre[:, 0],
        geometry.source_geometry,
        sample_count=6,
        generator=torch.Generator().manual_seed(12),
    )
    assert spill.fiber_ids is not None
    spill_valid = spill.valid_mask.squeeze(-1)
    spill_distance = torch.linalg.vector_norm(spill.points_ras_mm - centre, dim=-1)
    assert bool((spill_distance[spill_valid] > 4.0 + 1e-6).all())
    assert set(spill.fiber_ids[spill_valid].tolist()) == {0, 1, 2}

    source_dhw = ras_mm_to_voxel_dhw(centre, geometry.source_geometry)
    spill_dhw = ras_mm_to_voxel_dhw(spill.points_ras_mm, geometry.source_geometry)
    for fiber_id, matching_axes in ((0, (1, 2)), (1, (0, 2)), (2, (0, 1))):
        mask = spill_valid & (spill.fiber_ids == fiber_id)
        assert bool(mask.any())
        assert torch.allclose(
            spill_dhw[..., list(matching_axes)][mask],
            source_dhw.expand_as(spill_dhw)[..., list(matching_axes)][mask],
            atol=1e-5,
        )


def test_counterfactual_measurement_never_calls_full_decoder_or_mutates_live_state_and_isolates_target_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, geometry, points, point_semantic, f_spec, descriptors, target = _counterfactual_inputs(requires_grad=True)
    trajectory = _trajectory()
    decoder = ImplicitTriPlaneDecoder().eval()
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.zero_()
    before = tuple(plane.detach().clone() for plane in (state.xy, state.xz, state.yz))
    full_decode_calls = [0]

    def forbidden_full_decode(*_args: object, **_kwargs: object) -> torch.Tensor:
        full_decode_calls[0] += 1
        raise AssertionError("counterfactual candidates must not decode a full volume")

    monkeypatch.setattr(decoder, "forward", forbidden_full_decode)
    result = _counterfactual_result(
        trajectory,
        decoder,
        state,
        geometry,
        points,
        point_semantic,
        f_spec,
        descriptors,
        target,
        generator=torch.Generator().manual_seed(17),
    )

    assert full_decode_calls == [0]
    assert result.candidates.candidate_count == 3
    assert not result.reward_target.requires_grad
    assert result.valid_count == 3
    for original, actual in zip(before, (state.xy, state.xz, state.yz)):
        assert torch.equal(actual.detach(), original)

    result.loss.backward()
    assert any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0.0) for parameter in trajectory.reward_net.parameters())
    assert all(parameter.grad is None for parameter in trajectory.update_net.parameters())
    assert all(parameter.grad is None for parameter in decoder.parameters())
    assert state.xy.grad is None
    assert state.xz.grad is None
    assert state.yz.grad is None


def test_measured_reward_target_is_independent_of_gate_c_routing_cost_coefficients() -> None:
    state, geometry, points, point_semantic, f_spec, descriptors, target = _counterfactual_inputs()
    first_trajectory = _trajectory(lambda_travel=0.01, lambda_overlap=0.02, lambda_step=0.03)
    second_trajectory = _trajectory(lambda_travel=9.0, lambda_overlap=8.0, lambda_step=7.0)
    second_trajectory.load_state_dict(copy.deepcopy(first_trajectory.state_dict()))
    decoder = ImplicitTriPlaneDecoder().eval()
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.zero_()

    routing_reward = torch.tensor([[0.9, 0.7]])
    routing_travel = torch.tensor([[0.1, 0.6]])
    routing_overlap = torch.tensor([[0.2, 0.4]])
    first_utility = route_utility(routing_reward, routing_travel, routing_overlap, first_trajectory.config)
    second_utility = route_utility(routing_reward, routing_travel, routing_overlap, second_trajectory.config)
    assert not torch.equal(first_utility, second_utility)

    first = _counterfactual_result(
        first_trajectory,
        decoder,
        state,
        geometry,
        points,
        point_semantic,
        f_spec,
        descriptors,
        target,
        generator=torch.Generator().manual_seed(71),
    )
    second = _counterfactual_result(
        second_trajectory,
        decoder,
        state,
        geometry,
        points,
        point_semantic,
        f_spec,
        descriptors,
        target,
        generator=torch.Generator().manual_seed(71),
    )

    assert torch.equal(first.candidates.indices, second.candidates.indices)
    assert torch.equal(first.reward_prediction, second.reward_prediction)
    assert torch.equal(first.local_before, second.local_before)
    assert torch.equal(first.local_after, second.local_after)
    assert torch.equal(first.spill_before, second.spill_before)
    assert torch.equal(first.spill_after, second.spill_after)
    assert torch.equal(first.reward_target, second.reward_target)


def test_pairwise_reward_ranking_perfect_order_and_scale_is_zero() -> None:
    target = torch.tensor([[0.01, 0.02, 0.04]])
    prediction = target.clone().requires_grad_()
    result = pairwise_reward_ranking_loss(prediction, target, torch.ones_like(target, dtype=torch.bool), min_target_gap=0.001)

    assert result.informative_pair_count == 3
    assert result.violation_count == 0
    torch.testing.assert_close(result.loss, torch.zeros(()))


def test_pairwise_reward_ranking_wrong_order_has_finite_corrective_gradient() -> None:
    target = torch.tensor([[0.01, 0.02]])
    prediction = torch.tensor([[0.03, 0.01]], requires_grad=True)
    result = pairwise_reward_ranking_loss(prediction, target, torch.ones_like(target, dtype=torch.bool), min_target_gap=0.001)

    assert bool(result.loss > 0.0)
    result.loss.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())
    assert prediction.grad[0, 0] > 0.0
    assert prediction.grad[0, 1] < 0.0


def test_pairwise_reward_ranking_penalizes_compressed_but_correct_gap() -> None:
    target = torch.tensor([[0.00, 0.02]])
    prediction = torch.tensor([[0.010, 0.015]])
    result = pairwise_reward_ranking_loss(prediction, target, torch.ones_like(target, dtype=torch.bool), min_target_gap=0.001)

    assert bool(result.loss > 0.0)
    torch.testing.assert_close(result.loss, torch.tensor(0.015))


def test_pairwise_reward_ranking_excludes_target_gaps_below_threshold() -> None:
    target = torch.tensor([[0.0100, 0.0105]])
    prediction = torch.tensor([[0.4, 0.1]], requires_grad=True)
    result = pairwise_reward_ranking_loss(prediction, target, torch.ones_like(target, dtype=torch.bool), min_target_gap=0.001)

    assert result.valid_pair_count == 1
    assert result.informative_pair_count == 0
    torch.testing.assert_close(result.loss, prediction.sum() * 0.0)
    result.loss.backward()
    assert prediction.grad is not None and bool(torch.isfinite(prediction.grad).all())
    assert bool((prediction.grad == 0.0).all())


def test_pairwise_reward_ranking_honors_invalid_candidates() -> None:
    target = torch.tensor([[0.00, 0.02, 0.04]])
    prediction = torch.tensor([[0.04, 0.99, 0.00]])
    valid = torch.tensor([[True, False, True]])
    result = pairwise_reward_ranking_loss(prediction, target, valid, min_target_gap=0.001)

    assert result.valid_pair_count == 1
    assert result.informative_pair_count == 1
    assert bool(result.loss > 0.0)


def test_pairwise_reward_ranking_isolates_subject_rows() -> None:
    target = torch.tensor([[0.00, 0.02], [0.01, 0.03]])
    prediction = torch.tensor([[0.02, 0.01], [0.03, 0.01]])
    valid = torch.ones_like(target, dtype=torch.bool)
    result = pairwise_reward_ranking_loss(prediction, target, valid, min_target_gap=0.001)

    assert result.valid_pair_count == 2
    assert result.informative_pair_count == 2


def test_pairwise_reward_ranking_all_equal_targets_is_finite_differentiable_zero() -> None:
    target = torch.full((2, 4), 0.01)
    prediction = torch.randn(2, 4, requires_grad=True)
    result = pairwise_reward_ranking_loss(prediction, target, torch.ones_like(target, dtype=torch.bool), min_target_gap=0.001)

    assert result.informative_pair_count == 0
    assert bool(torch.isfinite(result.loss))
    result.loss.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())
    assert bool((prediction.grad == 0.0).all())


def test_ranking_disabled_reproduces_absolute_reward_loss_and_composition() -> None:
    state, geometry, points, point_semantic, f_spec, descriptors, target = _counterfactual_inputs()
    trajectory = _trajectory()
    decoder = ImplicitTriPlaneDecoder().eval()
    result = _counterfactual_result(
        trajectory,
        decoder,
        state,
        geometry,
        points,
        point_semantic,
        f_spec,
        descriptors,
        target,
        generator=torch.Generator().manual_seed(101),
        reward_ranking_weight=0.0,
    )
    expected_absolute = torch.where(
        result.valid_mask,
        torch.nn.functional.smooth_l1_loss(result.reward_prediction, result.reward_target, reduction="none"),
        torch.zeros_like(result.reward_prediction),
    ).sum() / result.valid_mask.sum().to(dtype=result.reward_prediction.dtype)

    torch.testing.assert_close(result.absolute_loss, expected_absolute)
    torch.testing.assert_close(result.ranking_weighted_loss, result.ranking_loss * 0.0)
    torch.testing.assert_close(result.loss, expected_absolute)

    enabled = _counterfactual_result(
        trajectory,
        decoder,
        state,
        geometry,
        points,
        point_semantic,
        f_spec,
        descriptors,
        target,
        generator=torch.Generator().manual_seed(101),
        reward_ranking_weight=0.1,
    )
    torch.testing.assert_close(enabled.loss, enabled.absolute_loss + enabled.ranking_weighted_loss)
    assert enabled.informative_pair_count >= 0
    assert 0.0 <= enabled.ranking_violation_fraction <= 1.0
    assert enabled.mean_target_pair_gap >= 0.0


def test_pairwise_reward_ranking_gradient_is_finite_and_nonzero_for_wrong_order() -> None:
    target = torch.tensor([[0.01, 0.02, 0.03]])
    prediction = torch.tensor([[0.03, 0.01, 0.02]], requires_grad=True)
    result = pairwise_reward_ranking_loss(prediction, target, torch.ones_like(target, dtype=torch.bool), min_target_gap=0.001)
    result.loss.backward()

    assert bool(torch.isfinite(result.loss))
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())
    assert bool(prediction.grad.abs().sum() > 0.0)
