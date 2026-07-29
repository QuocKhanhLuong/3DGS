"""Blocking tensor-contract checks for the differentiable Gaussian state."""

from __future__ import annotations

import math
import unittest

import torch

from smagm.contracts.coordinates import PhysicalPlane
from smagm.gaussians import GaussianBatch
from smagm.renderer import render_plane


def _valid_batch(**changes: object) -> GaussianBatch:
    dtype = torch.float64
    fields: dict[str, object] = {
        "centers_ras_mm": torch.zeros((2, 3), dtype=dtype),
        "covariance_factor": torch.eye(3, dtype=dtype).expand(2, -1, -1).clone(),
        "log_support_amplitude": torch.zeros((2, 1), dtype=dtype),
        "appearance": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype),
        "appearance_valid": torch.tensor([[True, True], [True, False]]),
        "primitive_kind": ("structural", "volumetric"),
        "primitive_id": ("g0", "g1"),
    }
    fields.update(changes)
    return GaussianBatch(**fields)  # type: ignore[arg-type]


class GaussianBatchContractTests(unittest.TestCase):
    def test_valid_batch_preserves_shapes_dtypes_and_spd_covariance(self) -> None:
        batch = _valid_batch()
        covariance = batch.covariance()

        self.assertEqual(batch.count, 2)
        self.assertEqual(batch.appearance_channels, 2)
        self.assertEqual(covariance.shape, (2, 3, 3))
        self.assertEqual(covariance.dtype, torch.float64)
        self.assertTrue(torch.all(torch.linalg.eigvalsh(covariance) > 0))
        self.assertTrue(torch.allclose(covariance, covariance.transpose(-1, -2)))

    def test_shapes_dtypes_finiteness_triangularity_and_positive_diagonal_are_blocking(self) -> None:
        invalid = (
            {"centers_ras_mm": torch.zeros((2, 2), dtype=torch.float64)},
            {"covariance_factor": torch.eye(3, dtype=torch.float64).unsqueeze(0)},
            {"log_support_amplitude": torch.zeros((2,), dtype=torch.float64)},
            {"appearance": torch.ones((2, 0), dtype=torch.float64), "appearance_valid": torch.zeros((2, 0), dtype=torch.bool)},
            {"appearance_valid": torch.ones((2, 2), dtype=torch.float64)},
            {"appearance": torch.tensor([[float("nan"), 2.0], [3.0, 4.0]], dtype=torch.float64)},
            {"log_support_amplitude": torch.full((2, 1), 1e4, dtype=torch.float64)},
            {"covariance_epsilon": math.inf},
            {"centers_ras_mm": torch.full((2, 3), 1e300, dtype=torch.float64)},
            {"covariance_factor": torch.full((2, 3, 3), 1e200, dtype=torch.float64).tril()},
            {"centers_ras_mm": torch.zeros((2, 3), dtype=torch.float16), "covariance_factor": torch.eye(3, dtype=torch.float16).expand(2, -1, -1).clone(), "log_support_amplitude": torch.zeros((2, 1), dtype=torch.float16), "appearance": torch.ones((2, 2), dtype=torch.float16)},
            {"covariance_factor": torch.tensor([[[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float64)},
            {"covariance_factor": torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float64)},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises((TypeError, ValueError)):
                    _valid_batch(**changes)

    def test_in_place_tensor_mutation_is_caught_by_forward_revalidation(self) -> None:
        batch = _valid_batch()
        batch.covariance_factor[0, 0, 1] = 0.5
        with self.assertRaisesRegex(ValueError, "lower triangular"):
            batch.validate()
        plane = PhysicalPlane(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0),
            1.0,
            (1, 1),
            (0.0, 0.0, 1.0),
        )
        with self.assertRaisesRegex(ValueError, "lower triangular"):
            render_plane(batch, plane)

    def test_modality_validity_masks_invalid_appearance_channels(self) -> None:
        batch = _valid_batch()
        self.assertEqual(batch.appearance_valid.dtype, torch.bool)
        self.assertFalse(batch.appearance_valid[1, 1].item())
        with self.assertRaises(ValueError):
            _valid_batch(appearance_valid=torch.tensor([[True], [True]], dtype=torch.bool))
