import torch

from smagm.baselines.fixed_support import FixedSupportConfig, sample_fixed_supports
from smagm.contracts.coordinates import PhysicalPlane
from smagm.features.contracts import EncoderFeatureMaps, FeatureGridToPlaneTransform


def _plane() -> PhysicalPlane:
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
    structural = torch.arange(16, dtype=torch.float64).reshape(1, 1, 4, 4)
    appearance = torch.ones((1, 1, 4, 4), dtype=torch.float64)
    reliability = torch.ones((1, 1, 4, 4), dtype=torch.float64)
    features = EncoderFeatureMaps(
        structural=structural,
        appearance=appearance,
        reliability=reliability,
        grid_to_plane=FeatureGridToPlaneTransform(
            input_shape_hw=(8, 8),
            feature_shape_hw=(4, 4),
            stride_vu=(2, 2),
            offset_vu_input_pixels=(0.0, 0.0),
        ),
        modality_ids=("mri",),
    )
    supports = sample_fixed_supports(
        features,
        _plane(),
        config=FixedSupportConfig(step_vu=(2, 2)),
    )
    assert supports.feature_indices_vu.tolist() == [[0, 0], [0, 2], [2, 0], [2, 2]]
    expected = torch.tensor(
        [
            [10.0, 20.0, 30.0],
            [18.0, 20.0, 30.0],
            [10.0, 32.0, 30.0],
            [18.0, 32.0, 30.0],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(supports.centers_ras_mm, expected)
    assert supports.observation_ids == ("obs-1",) * 4
    assert supports.feature_vectors.shape == (4, 3)


def test_support_topology_is_value_independent() -> None:
    transform = FeatureGridToPlaneTransform((8, 8), (4, 4), (2, 2), (0.0, 0.0))
    config = FixedSupportConfig(step_vu=(1, 2), border_vu=(1, 0), max_points=5)
    first = EncoderFeatureMaps(
        structural=torch.randn((1, 2, 4, 4)),
        appearance=torch.randn((1, 1, 4, 4)),
        reliability=torch.ones((1, 1, 4, 4)),
        grid_to_plane=transform,
    )
    second = EncoderFeatureMaps(
        structural=torch.randn((1, 2, 4, 4)) * 100.0,
        appearance=torch.randn((1, 1, 4, 4)) * 100.0,
        reliability=torch.ones((1, 1, 4, 4)),
        grid_to_plane=transform,
    )
    a = sample_fixed_supports(first, _plane(), observation_id="obs-1", config=config)
    b = sample_fixed_supports(second, _plane(), observation_id="obs-1", config=config)
    assert torch.equal(a.feature_indices_vu, b.feature_indices_vu)
    assert torch.allclose(a.centers_ras_mm, b.centers_ras_mm)
