"""Affine-equivariance and deterministic coordinate behavior for rendering."""

from __future__ import annotations

import unittest

import torch

from smagm.contracts.coordinates import (
    PhysicalPlane,
    SourceAffineTransform,
    SourceConvention,
)
from smagm.gaussians import GaussianBatch
from smagm.renderer import render_plane


def _batch(centers: torch.Tensor) -> GaussianBatch:
    dtype = centers.dtype
    return GaussianBatch(
        centers,
        torch.eye(3, dtype=dtype).expand(2, -1, -1).clone(),
        torch.tensor([[0.0], [-0.2]], dtype=dtype),
        torch.tensor([[1.0], [4.0]], dtype=dtype),
        torch.ones((2, 1), dtype=torch.bool),
    )


class PlaneRendererCoordinateTests(unittest.TestCase):
    def test_equivalent_rigid_coordinate_frames_render_identical_images(self) -> None:
        dtype = torch.float64
        original_plane = PhysicalPlane((1.0, 2.0, 3.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.7, 1.1), 1.0, (3, 4), (0.0, 0.0, 1.0))
        original_centers = torch.tensor([[1.8, 3.2, 3.1], [2.9, 2.4, 2.6]], dtype=dtype)
        original = render_plane(_batch(original_centers), original_plane)

        # x' = Qx + t: a 90 degree RAS rotation around superior plus translation.
        rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
        translation = torch.tensor([10.0, -4.0, 7.0], dtype=dtype)
        transform = lambda vector: tuple((rotation @ torch.tensor(vector, dtype=dtype) + translation).tolist())
        rotate_axis = lambda vector: tuple((rotation @ torch.tensor(vector, dtype=dtype)).tolist())
        transformed_plane = PhysicalPlane(
            transform(original_plane.pixel_center_origin_ras_mm),
            rotate_axis(original_plane.axis_u_ras),
            rotate_axis(original_plane.axis_v_ras),
            original_plane.spacing_uv_mm,
            original_plane.thickness_mm,
            original_plane.shape_hw,
            rotate_axis(original_plane.signed_normal_ras),
        )
        transformed = render_plane(_batch((original_centers @ rotation.T) + translation), transformed_plane)

        self.assertTrue(torch.allclose(original.intensity, transformed.intensity, rtol=1e-11, atol=1e-12, equal_nan=True))
        self.assertTrue(torch.allclose(original.support_mass, transformed.support_mass, rtol=1e-11, atol=1e-12))
        self.assertTrue(torch.equal(original.supported_psf_mass, transformed.supported_psf_mass))
        self.assertTrue(torch.equal(original.unsupported_mask, transformed.unsupported_mask))

    def test_reference_cpu_render_is_deterministic_for_a_recorded_seed(self) -> None:
        torch.manual_seed(20260729)
        first_centers = torch.randn((2, 3), dtype=torch.float64)
        torch.manual_seed(20260729)
        second_centers = torch.randn((2, 3), dtype=torch.float64)
        plane = PhysicalPlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0), 1.0, (4, 5), (0.0, 0.0, 1.0))

        first = render_plane(_batch(first_centers), plane)
        second = render_plane(_batch(second_centers), plane)
        self.assertTrue(torch.equal(first.intensity, second.intensity))
        self.assertTrue(torch.equal(first.support_mass, second.support_mass))
        self.assertTrue(torch.equal(first.unsupported_mask, second.unsupported_mask))

    def test_dicom_lps_provenance_renders_in_the_same_canonical_ras_frame(self) -> None:
        dtype = torch.float64
        source = SourceAffineTransform(
            (
                (-1.0, 0.0, 0.0, 0.0),
                (0.0, -1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            SourceConvention.DICOM_LPS,
        )
        canonical = PhysicalPlane(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0),
            1.0,
            (2, 3),
            (0.0, 0.0, 1.0),
        )
        with_provenance = PhysicalPlane(
            canonical.pixel_center_origin_ras_mm,
            canonical.axis_u_ras,
            canonical.axis_v_ras,
            canonical.spacing_uv_mm,
            canonical.thickness_mm,
            canonical.shape_hw,
            canonical.signed_normal_ras,
            source_transform=source,
        )
        centers = torch.tensor([[0.5, 0.5, 0.2], [1.5, 0.75, -0.3]], dtype=dtype)
        plain = render_plane(_batch(centers), canonical)
        converted = render_plane(_batch(centers), with_provenance)
        self.assertTrue(torch.equal(plain.intensity, converted.intensity))
        self.assertTrue(torch.equal(plain.support_mass, converted.support_mass))
