"""Geometry, analytic masking, and local-to-RAS covariance tests."""

from __future__ import annotations

import pytest
import torch

from smagm.baselines.fixed_gaussian import (
    FixedGaussianHeadConfig,
    RawFixedGaussianOutput,
    _local_factor_to_ras,
    construct_fixed_gaussians,
)
from smagm.baselines.fixed_support import FixedSupportBatch
from smagm.contracts.coordinates import PhysicalPlane
from smagm.features.analytic import analytic_feature_bank
from smagm.features.contracts import FeatureGridToPlaneTransform


def _plane(shape_hw: tuple[int, int], *, rotated: bool = False, observation_id: str = "obs") -> PhysicalPlane:
    if rotated:
        basis = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    else:
        basis = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return PhysicalPlane(
        pixel_center_origin_ras_mm=(1.0, 2.0, 3.0),
        axis_u_ras=basis[0], axis_v_ras=basis[1], signed_normal_ras=basis[2],
        spacing_uv_mm=(1.0, 1.0), thickness_mm=1.0, shape_hw=shape_hw,
        observation_id=observation_id,
    )


@pytest.mark.parametrize("stride", (1, 2, 4))
def test_feature_to_ras_to_grid_round_trip(stride: int) -> None:
    shape = (9, 11)
    feature_shape = tuple((length + stride - 1) // stride for length in shape)
    transform = FeatureGridToPlaneTransform(shape, feature_shape, (stride, stride), input_plane=_plane(shape))
    v = torch.tensor([0.25, feature_shape[0] - 1.25], dtype=torch.float64)
    u = torch.tensor([0.75, feature_shape[1] - 1.5], dtype=torch.float64)
    ras = transform.ras_mm_from_feature_vu(v, u)
    grid = transform.grid_sample_coordinates(ras)
    recovered_u = ((grid[:, 0] + 1.0) * feature_shape[1] / 2.0) - 0.5
    recovered_v = ((grid[:, 1] + 1.0) * feature_shape[0] / 2.0) - 0.5
    assert torch.allclose(recovered_v, v, atol=1e-12)
    assert torch.allclose(recovered_u, u, atol=1e-12)


def test_invalid_hole_erodes_legal_analytic_evidence_value_independently() -> None:
    shape = (25, 27)
    valid = torch.ones((1, 1, *shape), dtype=torch.bool)
    valid[:, :, 10:13, 11:14] = False
    first = torch.ones((1, 1, *shape))
    second = first.clone()
    second[~valid] = 999.0
    out_a = analytic_feature_bank(first, valid_mask=valid)
    out_b = analytic_feature_bank(second, valid_mask=valid)
    assert torch.equal(out_a.valid_mask, out_b.valid_mask)
    assert torch.equal(out_a.tensor, out_b.tensor)
    assert not bool(out_a.valid_mask[:, :, 6:17, 7:18].any())
    assert torch.equal(out_a.tensor[:, :, ~out_a.valid_mask[0, 0]], torch.zeros_like(out_a.tensor[:, :, ~out_a.valid_mask[0, 0]]))


def test_all_valid_analytic_topology_is_preserved() -> None:
    out = analytic_feature_bank(torch.randn((2, 1, 9, 11)))
    assert bool(out.valid_mask.all())


def _basis(rotation: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return rotation.to(dtype=dtype).reshape(1, 3, 3)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_identity_and_rotated_local_covariance_frames(dtype: torch.dtype) -> None:
    local_factor = torch.tensor([[[2.0, 0.0, 0.0], [0.4, 3.0, 0.0], [0.2, -0.3, 4.0]]], dtype=dtype)
    identity = _basis(torch.eye(3), dtype)
    rotated = _basis(torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]), dtype)
    epsilon = 1e-5
    identity_factor = _local_factor_to_ras(local_factor, identity, epsilon)
    rotated_factor = _local_factor_to_ras(local_factor, rotated, epsilon)
    sigma_local = local_factor @ local_factor.transpose(-1, -2)
    assert torch.allclose(identity_factor @ identity_factor.transpose(-1, -2), sigma_local + epsilon * torch.eye(3, dtype=dtype), atol=2e-5)
    expected_rotated = rotated.transpose(-1, -2) @ sigma_local @ rotated + epsilon * torch.eye(3, dtype=dtype)
    assert torch.allclose(rotated_factor @ rotated_factor.transpose(-1, -2), expected_rotated, atol=2e-5)


def _support(plane: PhysicalPlane, dtype: torch.dtype) -> FixedSupportBatch:
    basis = torch.tensor((plane.axis_u_ras, plane.axis_v_ras, plane.signed_normal_ras), dtype=dtype).reshape(1, 3, 3)
    return FixedSupportBatch(
        centers_ras_mm=torch.tensor((plane.pixel_center_origin_ras_mm,), dtype=dtype),
        feature_vectors=torch.ones((1, 25), dtype=dtype),
        feature_indices_vu=torch.zeros((1, 2), dtype=torch.long),
        reliability=torch.ones((1, 1), dtype=dtype),
        observation_ids=(plane.observation_id or "obs",),
        source_plane_hashes=(__import__("hashlib").sha256(plane.canonical_json().encode()).hexdigest(),),
        batch_index=0,
        support_basis_ras=basis,
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_fixed_gaussian_local_geometry_and_raw_covariance_gradients(dtype: torch.dtype) -> None:
    local_raw = torch.tensor([[0.0, 0.2, -0.1, 0.3, -0.2, 0.4]], dtype=dtype, requires_grad=True)
    raw = RawFixedGaussianOutput(
        center_offset_raw=torch.zeros((1, 3), dtype=dtype, requires_grad=True),
        covariance_raw=local_raw,
        log_amplitude_raw=torch.zeros((1, 1), dtype=dtype, requires_grad=True),
        appearance_raw=torch.zeros((1, 1), dtype=dtype, requires_grad=True),
    )
    config = FixedGaussianHeadConfig(input_dim=25, appearance_channels=1, covariance_epsilon=1e-5)
    first = construct_fixed_gaussians(_support(_plane((5, 5), observation_id="a"), dtype), raw, config=config)
    second = construct_fixed_gaussians(_support(_plane((5, 5), rotated=True, observation_id="b"), dtype), raw, config=config)
    first_local = first.covariance()[0]
    second_local = second.covariance()[0]
    b = torch.tensor((
        (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)
    ), dtype=dtype)
    assert torch.allclose(b @ second_local @ b.transpose(0, 1), first_local, atol=3e-5)
    loss = first.covariance().sum() + second.covariance().sum()
    loss.backward()
    assert local_raw.grad is not None and bool(torch.isfinite(local_raw.grad).all())
