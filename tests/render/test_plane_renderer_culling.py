"""Equivalence gates for conservative renderer culling and telemetry."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as functional

from smagm.contracts.coordinates import PhysicalPlane
from smagm.gaussians import GaussianBatch
from smagm.renderer import (
    RenderConfig,
    RenderResult,
    SlabProfile,
    render_plane,
    render_plane_brute_force_reference,
)


def _plane(*, shape: tuple[int, int] = (16, 18)) -> PhysicalPlane:
    return PhysicalPlane(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0),
        1.5,
        shape,
        (0.0, 0.0, 1.0),
    )


def _config() -> RenderConfig:
    return RenderConfig(
        support_epsilon=1e-12,
        pixel_chunk_size=7,
        gaussian_chunk_size=2,
        profile=SlabProfile.box(3),
        tile_shape_hw=(4, 5),
    )


def _raw_inputs(*, requires_grad: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch.float64
    centers = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [7.0, 2.0, 0.2],
            [14.0, 3.0, -0.1],
            [3.0, 10.0, 0.3],
            [11.0, 11.0, 0.0],
            [16.0, 14.0, -0.2],
            [70.0, 2.0, 0.0],
            [3.0, 5.0, 60.0],
            [80.0, 80.0, 80.0],
        ],
        dtype=dtype,
    )
    factors = torch.eye(3, dtype=dtype).expand(centers.shape[0], -1, -1).clone() * 0.08
    log_amplitude = torch.tensor([[0.0], [-0.2], [0.1], [-0.1], [0.2], [-0.3], [0.0], [0.0], [0.0]], dtype=dtype)
    appearance = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]], dtype=dtype)
    appearance_valid = torch.ones_like(appearance, dtype=torch.bool)
    appearance_valid[1, 0] = False

    def _leaf(value: torch.Tensor) -> torch.Tensor:
        return value.requires_grad_(requires_grad)

    return _leaf(centers), _leaf(factors), _leaf(log_amplitude), _leaf(appearance), appearance_valid


def _batch(inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> GaussianBatch:
    centers, factors, log_amplitude, appearance, appearance_valid = inputs
    return GaussianBatch(centers, factors, log_amplitude, appearance, appearance_valid)


def _assert_equivalent(optimized: RenderResult, reference: RenderResult) -> None:
    assert torch.allclose(optimized.intensity, reference.intensity, rtol=1e-10, atol=1e-12, equal_nan=True)
    assert torch.allclose(optimized.support_mass, reference.support_mass, rtol=1e-10, atol=1e-12)
    assert torch.allclose(optimized.supported_psf_mass, reference.supported_psf_mass, rtol=0.0, atol=0.0)
    assert torch.equal(optimized.unsupported_mask, reference.unsupported_mask)


def _render_loss(result: RenderResult) -> torch.Tensor:
    return torch.nan_to_num(result.intensity, nan=0.0).square().mean() + 0.25 * result.support_mass.square().mean()


def test_culled_render_matches_explicit_brute_force_for_outputs_and_appearance_validity() -> None:
    inputs = _raw_inputs(requires_grad=False)
    optimized = render_plane(_batch(inputs), _plane(), config=_config())
    reference = render_plane_brute_force_reference(_batch(inputs), _plane(), config=_config())

    _assert_equivalent(optimized, reference)
    assert reference.pixel_gaussian_candidate_pairs == 16 * 18 * 9
    # The invalid appearance Gaussian is never evaluated by the optimized path,
    # while the explicit reference intentionally counts every batch Gaussian.
    assert optimized.pixel_gaussian_candidate_pairs <= reference.pixel_gaussian_candidate_pairs - (16 * 18)
    assert optimized.pixel_gaussian_candidate_pairs > 0


def test_culled_render_matches_brute_force_gaussian_parameter_gradients() -> None:
    optimized_inputs = _raw_inputs(requires_grad=True)
    reference_inputs = _raw_inputs(requires_grad=True)
    optimized = render_plane(_batch(optimized_inputs), _plane(), config=_config())
    reference = render_plane_brute_force_reference(_batch(reference_inputs), _plane(), config=_config())

    _assert_equivalent(optimized, reference)
    optimized_gradients = torch.autograd.grad(_render_loss(optimized), optimized_inputs[:4])
    reference_gradients = torch.autograd.grad(_render_loss(reference), reference_inputs[:4])

    for parameter_name, optimized_gradient, reference_gradient in zip(
        ("centers", "covariance_factor", "log_support_amplitude", "appearance"),
        optimized_gradients,
        reference_gradients,
    ):
        assert torch.isfinite(optimized_gradient).all()
        assert torch.isfinite(reference_gradient).all()
        assert torch.allclose(optimized_gradient, reference_gradient, rtol=1e-9, atol=1e-11), parameter_name
        assert optimized_gradient.abs().max().item() > 0.0, parameter_name


class _StructuralFieldLike(nn.Module):
    """Small differentiable upstream producer of legal Gaussian tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.latent = nn.Parameter(torch.tensor(
            [[-0.4, 0.2, 0.1, 0.3], [0.1, -0.2, 0.5, -0.3], [0.4, 0.3, -0.1, 0.2], [-0.1, 0.5, 0.2, -0.4], [0.2, -0.3, 0.4, 0.1]],
            dtype=torch.float64,
        ))
        self.geometry = nn.Linear(4, 9, dtype=torch.float64)
        self.amplitude = nn.Linear(4, 1, dtype=torch.float64)
        self.appearance = nn.Linear(4, 1, dtype=torch.float64)

    def forward(self) -> GaussianBatch:
        hidden = torch.tanh(self.latent)
        geometry = self.geometry(hidden)
        base_centers = hidden.new_tensor([[1.0, 1.0, 0.0], [7.0, 3.0, 0.1], [3.0, 10.0, -0.2], [12.0, 12.0, 0.0], [90.0, 90.0, 90.0]])
        centers = base_centers + 0.03 * torch.tanh(geometry[:, :3])
        diagonal = 0.07 + 0.02 * functional.softplus(geometry[:, 3:6])
        lower = 0.01 * torch.tanh(geometry[:, 6:9])
        zeros = torch.zeros_like(diagonal[:, 0])
        rows = (
            torch.stack((diagonal[:, 0], zeros, zeros), dim=1),
            torch.stack((lower[:, 0], diagonal[:, 1], zeros), dim=1),
            torch.stack((lower[:, 1], lower[:, 2], diagonal[:, 2]), dim=1),
        )
        factors = torch.stack(rows, dim=1)
        log_amplitude = -0.2 + 0.1 * torch.tanh(self.amplitude(hidden))
        appearance = 2.0 + self.appearance(hidden)
        return GaussianBatch(centers, factors, log_amplitude, appearance, torch.ones_like(appearance, dtype=torch.bool))


def test_culled_render_preserves_differentiable_structural_field_like_gradients() -> None:
    torch.manual_seed(20260802)
    optimized_field = _StructuralFieldLike()
    reference_field = copy.deepcopy(optimized_field)
    plane = _plane(shape=(14, 16))
    config = _config()

    optimized = render_plane(optimized_field(), plane, config=config)
    reference = render_plane_brute_force_reference(reference_field(), plane, config=config)
    _assert_equivalent(optimized, reference)
    optimized_parameters = tuple(optimized_field.parameters())
    reference_parameters = tuple(reference_field.parameters())
    optimized_gradients = torch.autograd.grad(_render_loss(optimized), optimized_parameters)
    reference_gradients = torch.autograd.grad(_render_loss(reference), reference_parameters)

    for optimized_gradient, reference_gradient in zip(optimized_gradients, reference_gradients):
        assert torch.isfinite(optimized_gradient).all()
        assert torch.isfinite(reference_gradient).all()
        assert torch.allclose(optimized_gradient, reference_gradient, rtol=1e-9, atol=1e-11)
        assert optimized_gradient.abs().max().item() > 0.0


def test_manual_render_result_construction_keeps_default_candidate_pair_diagnostic() -> None:
    result = RenderResult(
        torch.zeros((1, 1), dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.bool),
    )
    assert result.pixel_gaussian_candidate_pairs == 0
