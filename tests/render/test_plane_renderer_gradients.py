"""Float64 autograd gates for all continuous Gaussian rendering parameters."""

from __future__ import annotations

import unittest

import torch

from smagm.contracts.coordinates import PhysicalPlane
from smagm.gaussians import GaussianBatch
from smagm.renderer import RenderConfig, render_plane


_PLANE = PhysicalPlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.7, 0.9), 1.0, (2, 2), (0.0, 0.0, 1.0))


def _reference_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch.float64
    return (
        torch.tensor([[0.3, 0.6, 0.2], [1.1, 0.9, -0.1]], dtype=dtype, requires_grad=True),
        torch.tensor([[[1.0, 0.0, 0.0], [0.1, 0.9, 0.0], [0.0, -0.1, 0.8]], [[0.8, 0.0, 0.0], [-0.1, 1.1, 0.0], [0.1, 0.2, 0.7]]], dtype=dtype, requires_grad=True),
        torch.tensor([[0.1], [-0.2]], dtype=dtype, requires_grad=True),
        torch.tensor([[1.0], [3.0]], dtype=dtype, requires_grad=True),
    )


def _render_flattened(center: torch.Tensor, factor: torch.Tensor, log_amplitude: torch.Tensor, appearance: torch.Tensor) -> torch.Tensor:
    # Gradcheck perturbs every raw input component.  Project the parameter to
    # the public lower-triangular factor domain before validation so unused
    # upper entries remain mathematically and contractually zero.
    factor = torch.tril(factor)
    gaussians = GaussianBatch(center, factor, log_amplitude, appearance, torch.ones_like(appearance, dtype=torch.bool))
    result = render_plane(gaussians, _PLANE, config=RenderConfig(support_epsilon=1e-12))
    # Include support to make common-mode log-amplitude derivatives observable.
    return torch.cat((result.intensity.reshape(-1), result.support_mass.reshape(-1)))


class PlaneRendererGradientTests(unittest.TestCase):
    def test_float64_gradcheck_covers_center_factor_log_amplitude_and_appearance(self) -> None:
        inputs = _reference_inputs()
        self.assertTrue(torch.autograd.gradcheck(_render_flattened, inputs, eps=1e-6, atol=1e-5, rtol=1e-4, fast_mode=False))

    def test_all_continuous_inputs_receive_finite_non_null_gradients(self) -> None:
        inputs = _reference_inputs()
        output = _render_flattened(*inputs)
        gradients = torch.autograd.grad(output.square().sum(), inputs)

        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().max().item(), 0.0)
