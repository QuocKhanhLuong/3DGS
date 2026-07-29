"""Blocking physical-coordinate and source-affine contracts for T0."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import math
import unittest

from smagm.contracts.coordinates import (
    PhysicalPlane,
    SourceAffineTransform,
    SourceConvention,
    TargetGrid,
)


def _identity_affine() -> tuple[tuple[float, ...], ...]:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _canonical_source() -> SourceAffineTransform:
    return SourceAffineTransform(_identity_affine(), SourceConvention.CANONICAL_RAS)


def _plane(source: SourceAffineTransform | None = None, **changes: object) -> PhysicalPlane:
    fields: dict[str, object] = {
        "pixel_center_origin_ras_mm": (0.0, 0.0, 0.0),
        "axis_u_ras": (1.0, 0.0, 0.0),
        "axis_v_ras": (0.0, 1.0, 0.0),
        "spacing_uv_mm": (1.0, 1.0),
        "thickness_mm": 1.0,
        "shape_hw": (3, 4),
        "signed_normal_ras": (0.0, 0.0, 1.0),
        "source_transform": source,
        "observation_id": "synthetic-plane",
    }
    fields.update(changes)
    return PhysicalPlane(**fields)  # type: ignore[arg-type]


class CoordinateContractTests(unittest.TestCase):
    def test_ras_identity_affine_preserves_landmarks(self) -> None:
        source = _canonical_source()

        self.assertEqual(source.origin_ras_mm, (0.0, 0.0, 0.0))
        self.assertEqual(source.axis_u_step_ras_mm, (1.0, 0.0, 0.0))
        self.assertEqual(source.axis_v_step_ras_mm, (0.0, 1.0, 0.0))
        self.assertEqual(source.signed_slice_axis_ras, (0.0, 0.0, 1.0))
        self.assertEqual(source.plane_index_to_ras_mm, _identity_affine())

    def test_dicom_lps_landmarks_are_canonicalized_to_ras(self) -> None:
        source = SourceAffineTransform(
            ((2.0, 0.0, 0.0, 10.0), (0.0, 3.0, 0.0, 20.0), (0.0, 0.0, 4.0, 30.0), (0.0, 0.0, 0.0, 1.0)),
            SourceConvention.DICOM_LPS,
        )

        self.assertEqual(source.origin_ras_mm, (-10.0, -20.0, 30.0))
        self.assertEqual(source.axis_u_step_ras_mm, (-2.0, 0.0, 0.0))
        self.assertEqual(source.axis_v_step_ras_mm, (0.0, -3.0, 0.0))
        self.assertEqual(source.signed_slice_axis_ras, (0.0, 0.0, 1.0))
        self.assertEqual(source.plane_index_to_ras_mm[3], (0.0, 0.0, 0.0, 1.0))

    def test_pixel_centres_use_tensor_v_u_order(self) -> None:
        plane = _plane(
            pixel_center_origin_ras_mm=(10.0, 20.0, 30.0),
            spacing_uv_mm=(2.0, 3.0),
        )

        self.assertEqual(plane.world_from_vu(0, 0), (10.0, 20.0, 30.0))
        self.assertEqual(plane.world_from_vu(2, 3), (16.0, 26.0, 30.0))

    def test_target_grid_explicitly_maps_d_h_w_as_w_h_d_homogeneous_index(self) -> None:
        grid = TargetGrid(
            ((2.0, 0.0, 0.0, 10.0), (0.0, 3.0, 0.0, 20.0), (0.0, 0.0, 4.0, 30.0), (0.0, 0.0, 0.0, 1.0)),
            (5, 6, 7),
            modality_ids=("t1",),
        )

        self.assertEqual(grid.world_from_dhw(4, 5, 6), (22.0, 35.0, 46.0))

    def test_malformed_nonfinite_and_singular_affines_fail(self) -> None:
        malformed = ((1.0, 0.0, 0.0, 0.0),) * 3
        singular = ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        nonfinite = ((math.nan, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))

        for affine in (malformed, singular, nonfinite):
            with self.subTest(affine=affine):
                with self.assertRaises(ValueError):
                    SourceAffineTransform(affine, SourceConvention.CANONICAL_RAS)

    def test_affine_rank_check_is_scale_independent(self) -> None:
        small_but_valid = (
            (1e-4, 0.0, 0.0, 0.0),
            (0.0, 2e-4, 0.0, 0.0),
            (0.0, 0.0, 3e-4, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        SourceAffineTransform(small_but_valid, SourceConvention.CANONICAL_RAS)
        ill_conditioned = (
            (1.0, 1.0, 0.0, 0.0),
            (0.0, 1e-12, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        with self.assertRaisesRegex(ValueError, "ill-conditioned"):
            SourceAffineTransform(ill_conditioned, SourceConvention.CANONICAL_RAS)

    def test_affine_rank_check_accepts_finite_extreme_scale(self) -> None:
        extreme_but_valid = (
            (1e308, 0.0, 0.0, 0.0),
            (0.0, 5e307, 0.0, 0.0),
            (0.0, 0.0, 2e307, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )

        source = SourceAffineTransform(
            extreme_but_valid,
            SourceConvention.CANONICAL_RAS,
        )
        self.assertEqual(source.axis_u_step_ras_mm, (1e308, 0.0, 0.0))

    def test_source_origin_axis_spacing_and_independent_signed_normal_must_agree(self) -> None:
        source = _canonical_source()
        invalid_variants = (
            {"pixel_center_origin_ras_mm": (0.1, 0.0, 0.0)},
            {"axis_u_ras": (0.0, 1.0, 0.0), "axis_v_ras": (-1.0, 0.0, 0.0), "signed_normal_ras": (0.0, 0.0, 1.0)},
            {"spacing_uv_mm": (2.0, 1.0)},
            {"signed_normal_ras": (0.0, 0.0, -1.0)},
        )
        for changes in invalid_variants:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    _plane(source, **changes)

        independently_flipped_slice_axis = SourceAffineTransform(
            (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, -1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            SourceConvention.CANONICAL_RAS,
        )
        with self.assertRaisesRegex(ValueError, "signed slice axis"):
            _plane(independently_flipped_slice_axis)

    def test_left_handed_nifti_slice_axis_is_preserved(self) -> None:
        left_handed = SourceAffineTransform(
            (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, -1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            SourceConvention.NIFTI_RAS,
        )
        plane = _plane(left_handed, signed_normal_ras=(0.0, 0.0, -1.0))
        self.assertEqual(plane.signed_normal_ras, (0.0, 0.0, -1.0))

    def test_records_are_frozen_and_serialize_canonically(self) -> None:
        source = _canonical_source()
        plane = _plane(source)
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            plane.shape_hw = (99, 99)  # type: ignore[misc]

        expected = {
            "axis_u_ras": [1.0, 0.0, 0.0],
            "axis_v_ras": [0.0, 1.0, 0.0],
            "coordinate_system": "RAS",
            "distance_unit": "mm",
            "observation_id": "synthetic-plane",
            "pixel_index_order": ["v", "u"],
            "pixel_center_origin_ras_mm": [0.0, 0.0, 0.0],
            "shape_hw": [3, 4],
            "signed_normal_ras": [0.0, 0.0, 1.0],
            "spacing_uv_mm": [1.0, 1.0],
            "thickness_mm": 1.0,
            "source_transform": {
                "convention": "CANONICAL_RAS",
                "distance_unit": "mm",
                "index_order": ["u", "v", "slice"],
                "plane_index_to_source_mm": [list(row) for row in _identity_affine()],
            },
        }
        self.assertEqual(json.loads(plane.canonical_json()), expected)
        self.assertEqual(plane.canonical_json(), plane.canonical_json())
