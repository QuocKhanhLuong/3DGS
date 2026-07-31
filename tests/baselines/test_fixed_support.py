import torch
import pytest

from smagm.baselines.fixed_gaussian import FixedGaussianHeadConfig, RawFixedGaussianOutput, construct_fixed_gaussians
from smagm.baselines.fixed_support import FixedSupportBatch, FixedSupportConfig, sample_fixed_supports
from smagm.contracts.coordinates import PhysicalPlane, SourceAffineTransform, SourceConvention
from smagm.features.contracts import EncoderFeatureMaps, FeatureGridToPlaneTransform


def _plane(**changes: object) -> PhysicalPlane:
    fields: dict[str, object] = {
        "pixel_center_origin_ras_mm": (10.0, 20.0, 30.0),
        "axis_u_ras": (1.0, 0.0, 0.0),
        "axis_v_ras": (0.0, 1.0, 0.0),
        "spacing_uv_mm": (2.0, 3.0),
        "thickness_mm": 4.0,
        "shape_hw": (8, 8),
        "signed_normal_ras": (0.0, 0.0, 1.0),
        "observation_id": "obs-1",
    }
    fields.update(changes)
    return PhysicalPlane(**fields)  # type: ignore[arg-type]


def _features_for_plane(plane: PhysicalPlane) -> EncoderFeatureMaps:
    return EncoderFeatureMaps(
        structural=torch.arange(16, dtype=torch.float64).reshape(1, 1, 4, 4),
        appearance=torch.ones((1, 1, 4, 4), dtype=torch.float64),
        reliability=torch.ones((1, 1, 4, 4), dtype=torch.float64),
        grid_to_planes=(FeatureGridToPlaneTransform((8, 8), (4, 4), (2, 2), input_plane=plane),),
        modality_ids=("mri",),
    )


def _source_affine_plane() -> PhysicalPlane:
    return _plane(
        source_transform=SourceAffineTransform(
            ((2.0, 0.0, 0.0, 10.0), (0.0, 3.0, 0.0, 20.0), (0.0, 0.0, 4.0, 30.0), (0.0, 0.0, 0.0, 1.0)),
            SourceConvention.CANONICAL_RAS,
        )
    )


def _rotated_plane() -> PhysicalPlane:
    return _plane(axis_u_ras=(0.0, 1.0, 0.0), axis_v_ras=(-1.0, 0.0, 0.0))


def _translated_plane() -> PhysicalPlane:
    return _plane(pixel_center_origin_ras_mm=(11.0, 20.0, 30.0))


def _legacy_plane_definition() -> PhysicalPlane:
    return PhysicalPlane(
        pixel_center_origin_ras_mm=(10.0, 20.0, 30.0),
        axis_u_ras=(1.0, 0.0, 0.0),
        axis_v_ras=(0.0, 1.0, 0.0),
        spacing_uv_mm=(2.0, 3.0),
        thickness_mm=4.0,
        shape_hw=(8, 8),
        signed_normal_ras=(0.0, 0.0, 1.0),
        observation_id="obs-1",
    )


def test_stride_transform_maps_supports_to_expected_world_centres() -> None:
    plane = _plane()
    structural = torch.arange(16, dtype=torch.float64).reshape(1, 1, 4, 4)
    appearance = torch.ones((1, 1, 4, 4), dtype=torch.float64)
    reliability = torch.ones((1, 1, 4, 4), dtype=torch.float64)
    features = EncoderFeatureMaps(
        structural=structural,
        appearance=appearance,
        reliability=reliability,
        grid_to_planes=(FeatureGridToPlaneTransform(
            input_shape_hw=(8, 8),
            feature_shape_hw=(4, 4),
            stride_vu=(2, 2),
            input_plane=plane,
        ),),
        modality_ids=("mri",),
    )
    supports = sample_fixed_supports(
        features,
        plane,
        config=FixedSupportConfig(step_vu=(2, 2)),
    )
    assert supports.feature_indices_vu.tolist() == [[0, 0], [0, 2], [2, 0], [2, 2]]
    expected = torch.tensor(
        [
            [11.0, 21.5, 30.0],
            [19.0, 21.5, 30.0],
            [11.0, 33.5, 30.0],
            [19.0, 33.5, 30.0],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(supports.centers_ras_mm, expected)
    assert supports.observation_ids == ("obs-1",) * 4
    assert supports.feature_vectors.shape == (4, 3)


def test_support_topology_is_value_independent() -> None:
    plane = _plane()
    transform = FeatureGridToPlaneTransform((8, 8), (4, 4), (2, 2), input_plane=plane)
    config = FixedSupportConfig(step_vu=(1, 2), border_vu=(1, 0), max_points=5)
    first = EncoderFeatureMaps(
        structural=torch.randn((1, 2, 4, 4)),
        appearance=torch.randn((1, 1, 4, 4)),
        reliability=torch.ones((1, 1, 4, 4)),
        grid_to_planes=(transform,),
        modality_ids=("mri",),
    )
    second = EncoderFeatureMaps(
        structural=torch.randn((1, 2, 4, 4)) * 100.0,
        appearance=torch.randn((1, 1, 4, 4)) * 100.0,
        reliability=torch.ones((1, 1, 4, 4)),
        grid_to_planes=(transform,),
        modality_ids=("mri",),
    )
    a = sample_fixed_supports(first, plane, observation_id="obs-1", config=config)
    b = sample_fixed_supports(second, plane, observation_id="obs-1", config=config)
    assert torch.equal(a.feature_indices_vu, b.feature_indices_vu)
    assert torch.allclose(a.centers_ras_mm, b.centers_ras_mm)


def test_valid_feature_mask_is_the_only_eligibility_signal() -> None:
    plane = _plane()
    transform = FeatureGridToPlaneTransform((8, 8), (4, 4), (2, 2), input_plane=plane)
    valid = torch.zeros((1, 1, 4, 4), dtype=torch.bool)
    valid[0, 0, 0, 0] = True
    valid[0, 0, 0, 2] = True
    valid[0, 0, 2, 1] = True
    features = EncoderFeatureMaps(
        structural=torch.arange(16, dtype=torch.float64).reshape(1, 1, 4, 4),
        appearance=torch.ones((1, 1, 4, 4), dtype=torch.float64),
        reliability=torch.tensor(
            [[[[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]]]],
            dtype=torch.float64,
        ),
        grid_to_planes=(transform,),
        modality_ids=("mri",),
        valid_feature_mask=valid,
    )
    supports = sample_fixed_supports(features, plane, config=FixedSupportConfig(step_vu=(1, 1)))
    assert supports.feature_indices_vu.tolist() == [[0, 0], [0, 2], [2, 1]]
    assert torch.equal(supports.reliability[:, 0], torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64))

    with pytest.raises(ValueError, match="value-independent"):
        FixedSupportConfig(minimum_reliability=0.1)


def test_half_pixel_feature_centres_and_plane_basis_are_preserved() -> None:
    plane = PhysicalPlane(
        pixel_center_origin_ras_mm=(10.0, 20.0, 30.0),
        axis_u_ras=(0.0, 1.0, 0.0),
        axis_v_ras=(-1.0, 0.0, 0.0),
        spacing_uv_mm=(2.0, 3.0),
        thickness_mm=4.0,
        shape_hw=(8, 8),
        signed_normal_ras=(0.0, 0.0, 1.0),
        observation_id="rotated-obs",
    )
    features = EncoderFeatureMaps(
        structural=torch.zeros((1, 1, 4, 4), dtype=torch.float64),
        appearance=torch.zeros((1, 1, 4, 4), dtype=torch.float64),
        reliability=torch.ones((1, 1, 4, 4), dtype=torch.float64),
        grid_to_planes=(FeatureGridToPlaneTransform((8, 8), (4, 4), (2, 2), (0.5, 0.5), input_plane=plane),),
        modality_ids=("mri",),
    )
    supports = sample_fixed_supports(features, plane, config=FixedSupportConfig(step_vu=(3, 3)))
    assert supports.feature_indices_vu.tolist() == [[0, 0], [0, 3], [3, 0], [3, 3]]
    expected = torch.tensor(
        [[8.5, 21.0, 30.0], [8.5, 33.0, 30.0], [-9.5, 21.0, 30.0], [-9.5, 33.0, 30.0]],
        dtype=torch.float64,
    )
    assert torch.allclose(supports.centers_ras_mm, expected)
    expected_basis = torch.tensor(((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), dtype=torch.float64)
    assert torch.equal(supports.support_basis_ras, expected_basis.expand(supports.count, -1, -1))


def test_fixed_support_batch_rejects_missing_or_invalid_ras_bases() -> None:
    common = {
        "centers_ras_mm": torch.zeros((1, 3), dtype=torch.float64),
        "feature_vectors": torch.zeros((1, 2), dtype=torch.float64),
        "feature_indices_vu": torch.zeros((1, 2), dtype=torch.long),
        "reliability": torch.ones((1, 1), dtype=torch.float64),
        "observation_ids": ("obs",),
        "source_plane_hashes": ("0" * 64,),
        "batch_index": 0,
    }
    with pytest.raises(TypeError, match="support_basis_ras"):
        FixedSupportBatch(**common)  # type: ignore[call-arg]

    invalid_bases = (
        None,
        torch.eye(3, dtype=torch.float64).reshape(1, 3, 3) * float("nan"),
        torch.eye(3, dtype=torch.float64).reshape(1, 1, 3, 3),
        torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float64),
    )
    for basis in invalid_bases:
        with pytest.raises(ValueError, match="support_basis_ras"):
            FixedSupportBatch(**common, support_basis_ras=basis)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="source_plane_hashes"):
        FixedSupportBatch(
            **{key: value for key, value in common.items() if key != "source_plane_hashes"},
            support_basis_ras=torch.eye(3, dtype=torch.float64).reshape(1, 3, 3),
        )
    with pytest.raises(ValueError, match="source_plane_hashes"):
        FixedSupportBatch(
            **{**common, "source_plane_hashes": ("not-a-canonical-plane-hash",)},
            support_basis_ras=torch.eye(3, dtype=torch.float64).reshape(1, 3, 3),
        )


@pytest.mark.parametrize("mismatched_plane", [_translated_plane(), _rotated_plane(), _source_affine_plane()])
def test_support_sampling_rejects_same_shape_plane_identity_mismatches(mismatched_plane: PhysicalPlane) -> None:
    bound_plane = _plane()
    with pytest.raises(ValueError, match="exactly match"):
        sample_fixed_supports(_features_for_plane(bound_plane), mismatched_plane)


def test_support_sampling_rejects_observation_id_spoof_and_accepts_canonical_equivalent_plane() -> None:
    bound_plane = _plane()
    features = _features_for_plane(bound_plane)
    with pytest.raises(ValueError, match="exactly match"):
        sample_fixed_supports(features, _plane(observation_id="spoofed-observation"))
    with pytest.raises(ValueError, match="observation_id override"):
        sample_fixed_supports(features, bound_plane, observation_id="spoofed-observation")

    equivalent_plane = _legacy_plane_definition()
    supports = sample_fixed_supports(features, equivalent_plane, config=FixedSupportConfig(step_vu=(3, 3)))
    assert supports.observation_ids == ("obs-1",) * supports.count
    assert supports.source_plane_hashes == (features.grid_to_planes[0].source_plane_hash,) * supports.count


def test_support_plane_provenance_survives_into_gaussian_primitive_ids() -> None:
    plane = _plane()
    supports = sample_fixed_supports(_features_for_plane(plane), plane, config=FixedSupportConfig(step_vu=(3, 3)))
    raw = RawFixedGaussianOutput(
        center_offset_raw=torch.zeros((supports.count, 3), dtype=torch.float64),
        covariance_raw=torch.zeros((supports.count, 6), dtype=torch.float64),
        log_amplitude_raw=torch.zeros((supports.count, 1), dtype=torch.float64),
        appearance_raw=torch.zeros((supports.count, 1), dtype=torch.float64),
    )
    gaussians = construct_fixed_gaussians(supports, raw, config=FixedGaussianHeadConfig(input_dim=supports.feature_vectors.shape[1]))
    expected = tuple(
        f"fixed:{plane_hash}:{observation_id}:{v}:{u}"
        for plane_hash, observation_id, (v, u) in zip(
            supports.source_plane_hashes, supports.observation_ids, supports.feature_indices_vu.tolist()
        )
    )
    assert gaussians.primitive_id == expected
