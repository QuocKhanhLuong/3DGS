"""Independent analytic and finite-slab reference checks for the T0 renderer."""

from __future__ import annotations

import math
import unittest

import torch

from smagm.contracts.coordinates import PhysicalPlane
from smagm.gaussians import GaussianBatch
from smagm.renderer import RenderConfig, SlabProfile, render_plane


def _plane(*, thickness: float = 1.0, shape: tuple[int, int] = (3, 4)) -> PhysicalPlane:
    return PhysicalPlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0), thickness, shape, (0.0, 0.0, 1.0))


def _batch(centers: torch.Tensor, factors: torch.Tensor, amplitudes: torch.Tensor, appearance: torch.Tensor, valid: torch.Tensor | None = None) -> GaussianBatch:
    return GaussianBatch(centers, factors, amplitudes, appearance, valid if valid is not None else torch.ones_like(appearance, dtype=torch.bool))


class PlaneRendererAnalyticTests(unittest.TestCase):
    def test_single_isotropic_gaussian_matches_closed_form_thin_plane_support_and_intensity(self) -> None:
        dtype = torch.float64
        plane = _plane()
        center = torch.tensor([[1.0, 1.0, 0.5]], dtype=dtype)
        sigma = 2.0
        log_amplitude = math.log(1.7)
        gaussians = _batch(center, torch.eye(3, dtype=dtype).unsqueeze(0) * sigma, torch.tensor([[log_amplitude]], dtype=dtype), torch.tensor([[5.25]], dtype=dtype))

        result = render_plane(gaussians, plane)
        v, u = torch.meshgrid(torch.arange(3, dtype=dtype), torch.arange(4, dtype=dtype), indexing="ij")
        squared_distance = (u - 1.0).square() + (v - 1.0).square() + 0.5**2
        # GaussianBatch defines Sigma = L L^T + covariance_epsilon I.
        expected_support = math.exp(log_amplitude) * torch.exp(-0.5 * squared_distance / (sigma**2 + 1e-8))

        self.assertTrue(torch.allclose(result.support_mass, expected_support, rtol=1e-10, atol=1e-12))
        self.assertTrue(torch.allclose(result.intensity, torch.full((3, 4), 5.25, dtype=dtype), rtol=0, atol=1e-12))
        self.assertTrue(torch.equal(result.supported_psf_mass, torch.ones_like(result.support_mass)))
        self.assertFalse(result.unsupported_mask.any())

    def test_rotated_anisotropic_mixture_matches_independent_quadratic_form_reference(self) -> None:
        dtype = torch.float64
        plane = _plane(shape=(2, 3))
        centers = torch.tensor([[0.5, 0.25, -0.1], [2.0, 1.5, 0.6]], dtype=dtype)
        factors = torch.tensor([[[1.3, 0.0, 0.0], [0.4, 0.7, 0.0], [-0.2, 0.3, 1.1]], [[0.8, 0.0, 0.0], [0.1, 1.4, 0.0], [0.2, -0.1, 0.9]]], dtype=dtype)
        log_amplitudes = torch.tensor([[math.log(1.2)], [math.log(0.7)]], dtype=dtype)
        appearance = torch.tensor([[2.0], [9.0]], dtype=dtype)
        batch = _batch(centers, factors, log_amplitudes, appearance)
        result = render_plane(batch, plane)

        v, u = torch.meshgrid(torch.arange(2, dtype=dtype), torch.arange(3, dtype=dtype), indexing="ij")
        points = torch.stack((u, v, torch.zeros_like(u)), dim=-1).reshape(-1, 3)
        covariance = factors @ factors.transpose(-1, -2) + 1e-8 * torch.eye(3, dtype=dtype)
        differences = points[:, None, :] - centers[None, :, :]
        quadratic = torch.einsum("pgi,gij,pgj->pg", differences, torch.linalg.inv(covariance), differences)
        density = torch.exp(log_amplitudes[:, 0])[None, :] * torch.exp(-0.5 * quadratic)
        expected_support = density.sum(dim=1)
        expected_intensity = (density * appearance[:, 0]).sum(dim=1) / expected_support

        self.assertTrue(torch.allclose(result.support_mass.reshape(-1), expected_support, rtol=1e-10, atol=1e-12))
        self.assertTrue(torch.allclose(result.intensity.reshape(-1), expected_intensity, rtol=1e-10, atol=1e-12))

    def test_delta_profile_exactly_equals_default_thin_path(self) -> None:
        dtype = torch.float64
        batch = _batch(torch.tensor([[0.5, 0.5, 0.2]], dtype=dtype), torch.eye(3, dtype=dtype).unsqueeze(0), torch.zeros((1, 1), dtype=dtype), torch.tensor([[3.0]], dtype=dtype))
        plane = _plane(thickness=3.0)
        default = render_plane(batch, plane)
        delta = render_plane(batch, plane, config=RenderConfig(profile=SlabProfile.delta()))

        self.assertTrue(torch.equal(default.intensity, delta.intensity))
        self.assertTrue(torch.equal(default.support_mass, delta.support_mass))
        self.assertTrue(torch.equal(default.supported_psf_mass, delta.supported_psf_mass))
        self.assertTrue(torch.equal(default.unsupported_mask, delta.unsupported_mask))

    def test_box_and_discrete_slab_profiles_normalize_and_converge(self) -> None:
        dtype = torch.float64
        plane = _plane(thickness=2.0, shape=(2, 2))
        batch = _batch(torch.tensor([[0.5, 0.5, 0.4]], dtype=dtype), torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.6]]], dtype=dtype), torch.zeros((1, 1), dtype=dtype), torch.tensor([[2.0]], dtype=dtype))
        box_8 = render_plane(batch, plane, config=RenderConfig(profile=SlabProfile.box(8)))
        box_64 = render_plane(batch, plane, config=RenderConfig(profile=SlabProfile.box(64)))
        box_512 = render_plane(batch, plane, config=RenderConfig(profile=SlabProfile.box(512)))
        discrete = SlabProfile.discrete((-0.5, 0.0, 0.5), (2.0, 4.0, 2.0))

        self.assertAlmostEqual(sum(discrete.weights), 1.0)
        self.assertLess(torch.max(torch.abs(box_64.support_mass - box_512.support_mass)).item(), torch.max(torch.abs(box_8.support_mass - box_512.support_mass)).item())
        self.assertTrue(torch.allclose(box_64.intensity, torch.full_like(box_64.intensity, 2.0)))

    def test_slab_psf_averages_normalized_latent_intensity_at_each_depth(self) -> None:
        dtype = torch.float64
        plane = _plane(thickness=2.0, shape=(1, 1))
        centers = torch.tensor([[0.0, 0.0, -0.8], [0.0, 0.0, 0.9]], dtype=dtype)
        factors = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.35]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.55]],
            ],
            dtype=dtype,
        )
        log_amplitudes = torch.tensor([[0.0], [math.log(1.7)]], dtype=dtype)
        appearance = torch.tensor([[1.0], [9.0]], dtype=dtype)
        profile = SlabProfile.discrete((-0.5, 0.5), (0.25, 0.75))
        batch = _batch(centers, factors, log_amplitudes, appearance)
        result = render_plane(batch, plane, config=RenderConfig(profile=profile))

        sample_points = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=dtype)
        covariance = factors @ factors.transpose(-1, -2) + 1e-8 * torch.eye(3, dtype=dtype)
        differences = sample_points[:, None, :] - centers[None, :, :]
        quadratic = torch.einsum(
            "sgi,gij,sgj->sg",
            differences,
            torch.linalg.inv(covariance),
            differences,
        )
        density = torch.exp(log_amplitudes[:, 0])[None, :] * torch.exp(-0.5 * quadratic)
        local_support = density.sum(dim=1)
        local_intensity = (density * appearance[:, 0]).sum(dim=1) / local_support
        psf_weights = torch.tensor(profile.weights, dtype=dtype)
        expected = (psf_weights * local_intensity).sum()
        density_weighted_alternative = (
            psf_weights * (density * appearance[:, 0]).sum(dim=1)
        ).sum() / (psf_weights * local_support).sum()

        self.assertTrue(torch.allclose(result.intensity[0, 0], expected, rtol=1e-11, atol=1e-12))
        self.assertTrue(torch.allclose(result.support_mass[0, 0], (psf_weights * local_support).sum(), rtol=1e-11, atol=1e-12))
        self.assertGreater(abs(expected.item() - density_weighted_alternative.item()), 0.1)

    def test_slab_unsupported_status_uses_weighted_psf_coverage(self) -> None:
        dtype = torch.float64
        plane = _plane(thickness=2.0, shape=(1, 1))
        batch = _batch(
            torch.tensor([[0.0, 0.0, -1.0]], dtype=dtype),
            torch.tensor(
                [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.05]]],
                dtype=dtype,
            ),
            torch.zeros((1, 1), dtype=dtype),
            torch.tensor([[7.0]], dtype=dtype),
        )
        tiny_tail = render_plane(
            batch,
            plane,
            config=RenderConfig(
                profile=SlabProfile.discrete((-0.5, 0.5), (0.9999, 0.0001))
            ),
        )
        refined_tiny_tail = render_plane(
            batch,
            plane,
            config=RenderConfig(
                profile=SlabProfile.discrete(
                    (-0.5, 0.25, 0.5),
                    (0.9999, 0.00005, 0.00005),
                )
            ),
        )
        half_missing = render_plane(
            batch,
            plane,
            config=RenderConfig(
                profile=SlabProfile.discrete((-0.5, 0.5), (0.5, 0.5))
            ),
        )

        self.assertAlmostEqual(tiny_tail.supported_psf_mass.item(), 0.9999)
        self.assertFalse(tiny_tail.unsupported_mask.item())
        self.assertAlmostEqual(tiny_tail.intensity.item(), 7.0)
        self.assertAlmostEqual(refined_tiny_tail.supported_psf_mass.item(), 0.9999)
        self.assertEqual(
            tiny_tail.unsupported_mask.item(),
            refined_tiny_tail.unsupported_mask.item(),
        )
        self.assertAlmostEqual(half_missing.supported_psf_mass.item(), 0.5)
        self.assertTrue(half_missing.unsupported_mask.item())
        self.assertTrue(torch.isnan(half_missing.intensity).item())

    def test_numeric_render_policies_reject_nonfinite_or_out_of_slab_values(self) -> None:
        for epsilon in (math.nan, math.inf):
            with self.subTest(epsilon=epsilon):
                with self.assertRaises(ValueError):
                    RenderConfig(support_epsilon=epsilon)
        with self.assertRaises(ValueError):
            SlabProfile.discrete((0.6,), (1.0,))
        with self.assertRaises(ValueError):
            SlabProfile.discrete((0.0, 0.1), (1e308, 1e308))
        float32_batch = _batch(
            torch.zeros((1, 3), dtype=torch.float32),
            torch.eye(3, dtype=torch.float32).unsqueeze(0),
            torch.zeros((1, 1), dtype=torch.float32),
            torch.ones((1, 1), dtype=torch.float32),
        )
        with self.assertRaisesRegex(ValueError, "representable"):
            render_plane(
                float32_batch,
                _plane(shape=(1, 1)),
                config=RenderConfig(support_epsilon=1e300),
            )

    def test_unsupported_and_tiny_support_are_explicit_and_numerically_safe(self) -> None:
        dtype = torch.float64
        plane = _plane(shape=(1, 1))
        factor = torch.eye(3, dtype=dtype).unsqueeze(0)
        no_contributor = _batch(torch.zeros((1, 3), dtype=dtype), factor, torch.zeros((1, 1), dtype=dtype), torch.tensor([[7.0]], dtype=dtype), torch.tensor([[False]]))
        unsupported = render_plane(no_contributor, plane)
        self.assertTrue(unsupported.unsupported_mask.item())
        self.assertEqual(unsupported.support_mass.item(), 0.0)
        self.assertEqual(unsupported.supported_psf_mass.item(), 0.0)
        self.assertTrue(torch.isnan(unsupported.intensity).item())

        tiny = _batch(torch.zeros((1, 3), dtype=dtype), factor, torch.tensor([[math.log(1e-20)]], dtype=dtype), torch.tensor([[7.0]], dtype=dtype))
        supported = render_plane(tiny, plane, config=RenderConfig(support_epsilon=1e-30))
        self.assertFalse(supported.unsupported_mask.item())
        self.assertTrue(torch.isfinite(supported.intensity).item())
        self.assertAlmostEqual(supported.intensity.item(), 7.0)
