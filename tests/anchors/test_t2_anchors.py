from __future__ import annotations

import hashlib

import pytest
import torch

from smagm.anchors import (
    AggregationConfig,
    AnchorBootstrapConfig,
    CachedPlaneEvidence,
    CandidateSelectionConfig,
    ConsolidationConfig,
    bootstrap_anchors,
    lift_candidates,
    physical_nms,
    select_structural_candidates,
)
from smagm.contracts.coordinates import PhysicalPlane
from smagm.features.contracts import EncoderFeatureMaps, FeatureGridToPlaneTransform


def _features(observation_id: str, *, z_mm: float = 0.0, rotated: bool = False) -> EncoderFeatureMaps:
    axis_u = (0.0, 1.0, 0.0) if rotated else (1.0, 0.0, 0.0)
    axis_v = (-1.0, 0.0, 0.0) if rotated else (0.0, 1.0, 0.0)
    plane = PhysicalPlane(
        pixel_center_origin_ras_mm=(0.0, 0.0, z_mm), axis_u_ras=axis_u, axis_v_ras=axis_v,
        spacing_uv_mm=(2.0, 3.0), thickness_mm=1.0, shape_hw=(5, 5),
        signed_normal_ras=(0.0, 0.0, 1.0), observation_id=observation_id,
    )
    transform = FeatureGridToPlaneTransform((5, 5), (5, 5), input_plane=plane)
    structural = torch.zeros(1, 2, 5, 5)
    structural[0, :, 1, 2] = 3.0
    structural[0, :, 3, 4] = 3.0
    appearance = torch.ones(1, 2, 5, 5)
    reliability = torch.full((1, 1, 5, 5), 0.5)
    mask = torch.ones(1, 1, 5, 5, dtype=torch.bool)
    mask[0, 0, 0, 0] = False
    return EncoderFeatureMaps(structural, appearance, reliability, (transform,), ("t1",), mask)


def _cached(observation_id: str, *, z_mm: float = 0.0, rotated: bool = False) -> CachedPlaneEvidence:
    features = _features(observation_id, z_mm=z_mm, rotated=rotated)
    return CachedPlaneEvidence(
        observation_id, "t1", features, hashlib.sha256(observation_id.encode()).hexdigest(),
        "synthetic-canonical-ras-v1",
        normalized_image=torch.ones(1, 1, 5, 5), valid_image_mask=torch.ones(1, 1, 5, 5, dtype=torch.bool),
    )


def test_candidates_are_deterministic_exclude_invalid_and_lift_with_physical_geometry() -> None:
    features = _features("obs", rotated=True)
    config = CandidateSelectionConfig(maximum_candidates=2)
    first = select_structural_candidates(features, config=config)
    second = select_structural_candidates(features, config=config)
    assert torch.equal(first.feature_indices_vu, second.feature_indices_vu)
    assert first.candidate_ids == ("obs:v1:u2", "obs:v3:u4")
    lifted = lift_candidates(first)
    assert torch.allclose(lifted.centers_ras_mm[0], torch.tensor((-3.0, 4.0, 0.0)))


def test_physical_nms_uses_millimetres_not_pixel_indices() -> None:
    candidates = lift_candidates(select_structural_candidates(_features("obs"), config=CandidateSelectionConfig(maximum_candidates=2)))
    assert physical_nms(candidates, radius_mm=1.0).centers_ras_mm.shape[0] == 2
    assert physical_nms(candidates, radius_mm=20.0).centers_ras_mm.shape[0] == 1


def test_bootstrap_is_context_only_patient_bound_and_retains_modality_slots() -> None:
    config = AnchorBootstrapConfig(
        candidate=CandidateSelectionConfig(maximum_candidates=2),
        consolidation=ConsolidationConfig(nms_radius_mm=0.5, merge_radius_mm=1.5, maximum_component_diameter_mm=2.0),
        aggregation=AggregationConfig(maximum_plane_distance_mm=4.0),
    )
    anchors = bootstrap_anchors((_cached("a", z_mm=0.0), _cached("b", z_mm=1.0)), patient_id="patient", modality_ids=("t1",), config=config)
    assert anchors.patient_id == "patient"
    assert anchors.count == 2
    assert anchors.appearance_valid.all()
    assert all(set(ids) == {"a", "b"} for ids in anchors.geometry.contributing_observation_ids)


def test_bootstrap_rejects_any_non_context_evidence() -> None:
    item = _cached("obs")
    object.__setattr__(item, "context_only", False)
    try:
        bootstrap_anchors((item,), patient_id="patient", modality_ids=("t1",))
    except PermissionError:
        pass
    else:
        raise AssertionError("target-derived evidence was accepted")


def test_bootstrap_rejects_undeclared_or_mismatched_registration() -> None:
    item = _cached("a")
    object.__setattr__(item, "registration_id", "")
    with pytest.raises(ValueError, match="registration identity"):
        item.__post_init__()
    first = _cached("a")
    second = _cached("b")
    object.__setattr__(second, "registration_id", "different-registration")
    with pytest.raises(PermissionError, match="common registration"):
        bootstrap_anchors((first, second), patient_id="patient", modality_ids=("t1",))
