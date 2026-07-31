"""Blocking T1-B encoder and common-contract tests."""

from __future__ import annotations

import pytest
import torch

from smagm.baselines.fixed_support import FixedSupportConfig, sample_fixed_supports
from smagm.contracts.coordinates import PhysicalPlane
from smagm.features.encoder import EncoderConfig, EvidenceEncoder


def _plane(shape_hw: tuple[int, int], *, observation_id: str, rotated: bool = False) -> PhysicalPlane:
    if rotated:
        axis_u, axis_v, normal = (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)
    else:
        axis_u, axis_v, normal = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    return PhysicalPlane(
        pixel_center_origin_ras_mm=(3.0 if rotated else 0.0, 4.0 if rotated else 0.0, 5.0 if rotated else 0.0),
        axis_u_ras=axis_u,
        axis_v_ras=axis_v,
        spacing_uv_mm=(1.2, 1.7),
        thickness_mm=1.0,
        shape_hw=shape_hw,
        signed_normal_ras=normal,
        observation_id=observation_id,
    )


def _images(batch: int, shape_hw: tuple[int, int], *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    torch.manual_seed(31)
    return torch.randn((batch, 1, *shape_hw), dtype=dtype)


@pytest.mark.parametrize("variant", ("e0", "e1", "e2"))
def test_variants_share_exact_output_contract_and_are_finite(variant: str) -> None:
    shape = (31, 29)
    encoder = EvidenceEncoder(EncoderConfig(variant=variant, output_stride=2))
    features = encoder(_images(1, shape), _plane(shape, observation_id="obs"), "mri")
    assert features.structural.shape == (1, 16, 16, 15)
    assert features.appearance.shape == (1, 8, 16, 15)
    assert features.reliability.shape == (1, 1, 16, 15)
    assert features.valid_feature_mask.shape == (1, 1, 16, 15)
    assert bool(torch.isfinite(features.concatenated()).all())
    assert bool(((features.reliability >= 0.0) & (features.reliability <= 1.0)).all())
    assert not bool(features.valid_feature_mask[0, 0, -1].any())
    assert not bool(features.valid_feature_mask[0, 0, :, -1].any())


@pytest.mark.parametrize("stride", (1, 2, 4))
def test_encoder_stride_contract_uses_ceil_shape_and_bound_transform(stride: int) -> None:
    shape = (9, 11)
    encoder = EvidenceEncoder(EncoderConfig(variant="e1", output_stride=stride))
    features = encoder(_images(1, shape), _plane(shape, observation_id=f"stride-{stride}"), "mri")
    expected = tuple((length + stride - 1) // stride for length in shape)
    assert features.feature_shape_hw == expected
    assert features.grid_to_planes[0].output_stride == stride
    assert features.grid_to_planes[0].feature_shape_hw == expected


def test_parameter_report_is_explicit_and_e0_has_no_hidden_learned_encoder() -> None:
    e0 = EvidenceEncoder(EncoderConfig(variant="e0"))
    e1 = EvidenceEncoder(EncoderConfig(variant="e1"))
    e2 = EvidenceEncoder(EncoderConfig(variant="e2"))
    assert e0.parameter_report.parameter_count == 0
    assert e0.parameter_report.adapter_operation_count > 0
    assert e1.parameter_report.parameter_count > 0
    assert e2.parameter_report.parameter_count > e1.parameter_report.parameter_count


def test_support_indices_are_identical_across_variants_and_ignore_reliability_values() -> None:
    shape = (17, 19)
    plane = _plane(shape, observation_id="same")
    outputs = [EvidenceEncoder(EncoderConfig(variant=variant))(_images(1, shape), plane, "mri") for variant in ("e0", "e1", "e2")]
    indices = [
        sample_fixed_supports(features, plane, config=FixedSupportConfig(step_vu=(3, 3))).feature_indices_vu
        for features in outputs
    ]
    assert torch.equal(indices[0], indices[1]) and torch.equal(indices[0], indices[2])
    altered = outputs[1].reliability.detach().clone()
    altered.zero_()
    altered_features = type(outputs[1])(
        structural=outputs[1].structural,
        appearance=outputs[1].appearance,
        reliability=altered,
        grid_to_planes=outputs[1].grid_to_planes,
        modality_ids=outputs[1].modality_ids,
        valid_feature_mask=outputs[1].valid_feature_mask,
    )
    assert torch.equal(
        indices[1],
        sample_fixed_supports(altered_features, plane, config=FixedSupportConfig(step_vu=(3, 3))).feature_indices_vu,
    )


def test_batch_geometry_is_per_item_and_single_plane_cannot_broadcast() -> None:
    shape = (13, 15)
    planes = (_plane(shape, observation_id="left"), _plane(shape, observation_id="right", rotated=True))
    image = _images(2, shape)
    encoder = EvidenceEncoder(EncoderConfig(variant="e1"))
    features = encoder(image, planes, ("mri-a", "mri-b"))
    assert tuple(transform.input_plane.observation_id for transform in features.grid_to_planes) == ("left", "right")
    first = sample_fixed_supports(features, planes[0], batch_index=0, config=FixedSupportConfig(step_vu=(4, 4)))
    second = sample_fixed_supports(features, planes[1], batch_index=1, config=FixedSupportConfig(step_vu=(4, 4)))
    assert first.observation_ids[0] == "left" and second.observation_ids[0] == "right"
    assert first.source_plane_hashes[0] != second.source_plane_hashes[0]
    assert not torch.allclose(first.centers_ras_mm, second.centers_ras_mm)
    with pytest.raises(ValueError, match="batch of one"):
        encoder(image, planes[0], ("mri-a", "mri-b"))


@pytest.mark.parametrize("variant", ("e0", "e1", "e2"))
def test_constant_images_and_encoder_path_have_finite_gradients(variant: str) -> None:
    shape = (17, 19)
    image = torch.ones((1, 1, *shape), requires_grad=True)
    encoder = EvidenceEncoder(EncoderConfig(variant=variant))
    features = encoder(image, _plane(shape, observation_id="constant"), "mri")
    loss = features.structural.square().mean() + features.appearance.square().mean() + features.reliability.mean()
    loss.backward()
    assert bool(torch.isfinite(image.grad).all())
    assert all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in encoder.parameters())
