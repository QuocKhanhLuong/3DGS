"""Chunking parity contracts for the CPU reference renderer."""

from __future__ import annotations

import unittest

import torch

from smagm.contracts.coordinates import PhysicalPlane
from smagm.gaussians import GaussianBatch
from smagm.renderer import RenderConfig, SlabProfile, render_plane


def _plane() -> PhysicalPlane:
    return PhysicalPlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.8, 1.2), 1.5, (5, 6), (0.0, 0.0, 1.0))


def _inputs(*, requires_grad: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch.float64
    centers = torch.tensor([[0.2, 0.1, 0.0], [1.9, 2.1, 0.4], [3.4, 1.2, -0.2]], dtype=dtype, requires_grad=requires_grad)
    factors = torch.tensor([[[1.1, 0.0, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 0.8]], [[0.7, 0.0, 0.0], [-0.2, 1.3, 0.0], [0.1, 0.2, 1.0]], [[1.3, 0.0, 0.0], [0.0, 0.8, 0.0], [-0.1, 0.2, 0.6]]], dtype=dtype, requires_grad=requires_grad)
    log_amplitude = torch.tensor([[0.0], [-0.3], [0.2]], dtype=dtype, requires_grad=requires_grad)
    appearance = torch.tensor([[1.0], [3.0], [5.0]], dtype=dtype, requires_grad=requires_grad)
    return centers, factors, log_amplitude, appearance


def _render(inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], config: RenderConfig):
    centers, factors, log_amplitude, appearance = inputs
    batch = GaussianBatch(centers, factors, log_amplitude, appearance, torch.ones_like(appearance, dtype=torch.bool))
    return render_plane(batch, _plane(), config=config)


class PlaneRendererChunkingTests(unittest.TestCase):
    def test_pixel_and_gaussian_chunking_match_unchunked_forward_reference(self) -> None:
        inputs = _inputs(requires_grad=False)
        unchunked = _render(inputs, RenderConfig(profile=SlabProfile.box(5)))
        chunked = _render(inputs, RenderConfig(pixel_chunk_size=4, gaussian_chunk_size=1, profile=SlabProfile.box(5)))

        self.assertTrue(torch.allclose(chunked.intensity, unchunked.intensity, rtol=1e-12, atol=1e-12, equal_nan=True))
        self.assertTrue(torch.allclose(chunked.support_mass, unchunked.support_mass, rtol=1e-12, atol=1e-12))
        self.assertTrue(torch.equal(chunked.supported_psf_mass, unchunked.supported_psf_mass))
        self.assertTrue(torch.equal(chunked.unsupported_mask, unchunked.unsupported_mask))

    def test_chunked_and_unchunked_gradients_match_without_graph_breaks(self) -> None:
        plain_inputs = _inputs(requires_grad=True)
        chunked_inputs = _inputs(requires_grad=True)
        plain = _render(plain_inputs, RenderConfig(profile=SlabProfile.box(3)))
        chunked = _render(chunked_inputs, RenderConfig(pixel_chunk_size=3, gaussian_chunk_size=2, profile=SlabProfile.box(3)))
        plain_loss = (plain.intensity.square().mean() + plain.support_mass.mean())
        chunked_loss = (chunked.intensity.square().mean() + chunked.support_mass.mean())
        plain_gradients = torch.autograd.grad(plain_loss, plain_inputs)
        chunked_gradients = torch.autograd.grad(chunked_loss, chunked_inputs)

        for plain_gradient, chunked_gradient in zip(plain_gradients, chunked_gradients):
            self.assertIsNotNone(plain_gradient)
            self.assertIsNotNone(chunked_gradient)
            self.assertTrue(torch.isfinite(plain_gradient).all())
            self.assertTrue(torch.isfinite(chunked_gradient).all())
            self.assertTrue(torch.allclose(plain_gradient, chunked_gradient, rtol=1e-10, atol=1e-11))
