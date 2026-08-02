from __future__ import annotations

import hashlib
import math
from pathlib import Path
import struct

import numpy as np
import pytest
import torch

from smagm.anchors import AnchorBatch, AnchorGeometryBatch
from smagm.anchors.contracts import anchor_evidence_hash
from smagm.contracts.coordinates import PhysicalPlane, TargetGrid
from smagm.contracts.outputs import VolumeReconstruction, volume_output_hash
from smagm.evaluation import AuditTarget, FreezeRecord, ReconstructionMetricConfig, compute_reconstruction_metrics, evaluate_audit_targets, open_serialized_audit_targets, open_serialized_predictions, paired_patient_summary
from smagm.reconstruction import (
    build_reconstruction_package,
    export_reconstruction_package,
    reconstruct_plane,
    reconstruct_volume,
)
from smagm.state import build_initial_patient_state


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _state():
    geometry = AnchorGeometryBatch(
        ("a",), torch.zeros(1, 3), torch.eye(3).unsqueeze(0), torch.tensor([[True, True, True]]),
        torch.tensor([[4.0, 4.0, 4.0]]), torch.ones(1, 1), torch.zeros(1, 1),
        (("obs",),), ((_digest("plane"),),), (_digest("anchor"),),
    )
    evidence = torch.ones(1, 4); appearance = torch.tensor([[2.0]]); valid = torch.ones(1, 1, dtype=torch.bool)
    observability = torch.tensor([[1.0, 1.0, 0.0]])
    evidence_hash = anchor_evidence_hash(patient_id="p", geometry=geometry, evidence=evidence, appearance=appearance, appearance_valid=valid, observability=observability)
    anchors = AnchorBatch("p", geometry, evidence, appearance, valid, observability, ("t1",), evidence_hash)
    return build_initial_patient_state(
        patient_id="p", manifest_hash=_digest("manifest"), config_hash=_digest("config"),
        context_observation_ids=("obs",), cache_key_hashes=(_digest("cache"),), anchors=anchors,
        field_config_hash=_digest("field-config"), field_model_hash=_digest("field-model"),
    )


def _grid() -> TargetGrid:
    return TargetGrid(
        ((1.0, 0.0, 0.0, -2.0), (0.0, 1.0, 0.0, -2.0), (0.0, 0.0, 1.5, -1.5), (0.0, 0.0, 0.0, 1.0)),
        (3, 5, 5), ("t1",), (_digest("normalization"),),
    )


def test_arbitrary_plane_reconstruction_preserves_geometry_and_explicit_support() -> None:
    plane = PhysicalPlane(
        pixel_center_origin_ras_mm=(-2.0, -2.0, 0.0), axis_u_ras=(1.0, 0.0, 0.0), axis_v_ras=(0.0, 1.0, 0.0),
        spacing_uv_mm=(1.0, 1.0), thickness_mm=1.0, shape_hw=(5, 5), signed_normal_ras=(0.0, 0.0, 1.0),
    )
    output = reconstruct_plane(_state(), plane, modality_id="t1")
    assert output.plane.canonical_json() == plane.canonical_json()
    assert len(output.artifact_hash) == 64
    assert torch.isfinite(output.intensity[~output.unsupported_mask]).all()
    assert torch.isnan(output.intensity[output.unsupported_mask]).all()


def test_chunked_and_unchunked_volume_are_identical() -> None:
    state = _state(); grid = _grid()
    first = reconstruct_volume(state, grid, modality_id="t1", depth_chunk_size=1)
    second = reconstruct_volume(state, grid, modality_id="t1", depth_chunk_size=3)
    assert torch.allclose(first.intensity, second.intensity, equal_nan=True)
    assert torch.equal(first.unsupported_mask, second.unsupported_mask)
    assert first.grid.canonical_json() == grid.canonical_json()


def test_export_round_trip_preserves_affine_hashes_and_prevents_overwrite(tmp_path) -> None:
    volume = reconstruct_volume(_state(), _grid(), modality_id="t1")
    package = build_reconstruction_package(
        (volume,), repository_commit=hashlib.sha1(b"commit").hexdigest(), config_hash=_digest("config"),
        manifest_hash=_digest("manifest"), split_hash=_digest("split"), assignment_hash=_digest("assignment"),
        encoder_identity="e2", field_identity="anchor-field", gaussian_identity="dual-bank",
        propagation_identity="p0", environment_hash=_digest("environment"),
    )
    directory = export_reconstruction_package(package, (volume,), tmp_path / "prediction")
    predictions = open_serialized_predictions(directory)
    restored = predictions.volumes[0]
    assert restored.grid.canonical_json() == volume.grid.canonical_json()
    assert restored.artifact_hash == volume.artifact_hash
    assert not restored.intensity.requires_grad
    nii = directory / "volume_t1.nii"
    raw = nii.read_bytes()
    assert struct.unpack_from("<I", raw, 0)[0] == 348
    assert struct.unpack_from("<4f", raw, 280) == pytest.approx(tuple(_grid().index_to_ras_mm[0]))
    serialized = np.frombuffer(raw, dtype="<f4", offset=352).reshape((5, 5, 3))
    assert np.allclose(serialized, volume.intensity.permute(2, 1, 0).numpy(), equal_nan=True)
    with pytest.raises(FileExistsError):
        export_reconstruction_package(package, (volume,), directory)


def test_isolated_audit_requires_matching_freeze_record_and_grid(tmp_path) -> None:
    volume = reconstruct_volume(_state(), _grid(), modality_id="t1")
    package = build_reconstruction_package(
        (volume,), repository_commit=hashlib.sha1(b"commit").hexdigest(), config_hash=_digest("config"),
        manifest_hash=_digest("manifest"), split_hash=_digest("split"), assignment_hash=_digest("assignment"),
        encoder_identity="e2", field_identity="anchor-field", gaussian_identity="dual-bank",
        propagation_identity="p0", environment_hash=_digest("environment"),
    )
    predictions = open_serialized_predictions(export_reconstruction_package(package, (volume,), tmp_path / "prediction"))
    target = AuditTarget("p", _digest("split"), "t1", _grid(), volume.intensity.nan_to_num(), ~volume.unsupported_mask)
    with pytest.raises(PermissionError):
        evaluate_audit_targets(predictions, (target,), freeze_record=FreezeRecord(package.package_hash, package.config_hash, package.split_hash, False, True))
    metrics = evaluate_audit_targets(predictions, (target,), freeze_record=FreezeRecord(package.package_hash, package.config_hash, package.split_hash, True, True))
    assert metrics[0].mae == pytest.approx(0.0)
    assert metrics[0].supported_fraction == pytest.approx(1.0)


def test_evaluator_uses_declared_data_range_and_support_conditioned_metrics() -> None:
    volume = reconstruct_volume(_state(), _grid(), modality_id="t1")
    target = volume.intensity.nan_to_num() + 0.1
    metrics = compute_reconstruction_metrics(
        volume,
        target,
        torch.ones_like(target, dtype=torch.bool),
        metric_config=ReconstructionMetricConfig(data_range=1.0, ssim_window_policy="global", edge_threshold=0.05),
    )
    assert metrics.data_range == pytest.approx(1.0)
    assert metrics.data_range_source == "declared"
    assert metrics.ssim_window_policy == "global"
    assert metrics.supported_fraction == pytest.approx(1.0)
    assert metrics.unsupported_fraction == pytest.approx(0.0)
    assert metrics.complete_metric_status == "COMPUTED_ALL_DECLARED_TARGET_VOXELS"
    assert metrics.complete_mae == pytest.approx(metrics.mae)
    assert metrics.gradient_rmse >= 0.0
    assert metrics.local_contrast_error >= 0.0


def test_evaluator_does_not_zero_fill_unsupported_voxels() -> None:
    volume = reconstruct_volume(_state(), _grid(), modality_id="t1")
    unsupported = torch.zeros_like(volume.unsupported_mask)
    unsupported[0, 0, 0] = True
    intensity = volume.intensity.clone()
    support_mass = volume.support_mass.clone()
    uncertainty = volume.support_uncertainty.clone()
    intensity[unsupported] = torch.nan
    uncertainty[unsupported] = torch.nan
    partial = VolumeReconstruction(
        patient_id=volume.patient_id,
        modality_id=volume.modality_id,
        grid=volume.grid,
        intensity=intensity,
        support_mass=support_mass,
        unsupported_mask=unsupported,
        support_uncertainty=uncertainty,
        depth_chunk_size=volume.depth_chunk_size,
        renderer_config_hash=volume.renderer_config_hash,
        patient_state_version=volume.patient_state_version,
        artifact_hash=volume_output_hash(
            patient_id=volume.patient_id,
            modality_id=volume.modality_id,
            grid=volume.grid,
            depth_chunk_size=volume.depth_chunk_size,
            renderer_config_hash=volume.renderer_config_hash,
            patient_state_version=volume.patient_state_version,
            intensity=intensity,
            support_mass=support_mass,
            unsupported_mask=unsupported,
            support_uncertainty=uncertainty,
        ),
    )
    metrics = compute_reconstruction_metrics(
        partial,
        volume.intensity.nan_to_num(),
        torch.ones_like(volume.intensity, dtype=torch.bool),
        metric_config=ReconstructionMetricConfig(data_range=1.0),
    )
    assert metrics.supported_fraction < 1.0
    assert metrics.unsupported_fraction > 0.0
    assert metrics.complete_metric_status == "NOT_COMPUTED_UNSUPPORTED_PIXELS"
    assert math.isnan(metrics.complete_mae)
    assert math.isnan(metrics.complete_ssim)


def test_evaluator_reports_serialized_geometry_and_observability_strata() -> None:
    volume = reconstruct_volume(_state(), _grid(), modality_id="t1")
    plane_kwargs = dict(
        axis_u_ras=(1.0, 0.0, 0.0), axis_v_ras=(0.0, 1.0, 0.0), spacing_uv_mm=(1.0, 1.0),
        thickness_mm=1.0, shape_hw=(5, 5), signed_normal_ras=(0.0, 0.0, 1.0),
    )
    context_planes = (
        PhysicalPlane(pixel_center_origin_ras_mm=(-2.0, -2.0, -2.0), **plane_kwargs),
        PhysicalPlane(pixel_center_origin_ras_mm=(-2.0, -2.0, 2.0), **plane_kwargs),
    )
    target = volume.intensity.nan_to_num() + 0.1
    observability = torch.linspace(0.0, 1.0, target.numel()).reshape_as(target)
    metrics = compute_reconstruction_metrics(
        volume,
        target,
        torch.ones_like(target, dtype=torch.bool),
        metric_config=ReconstructionMetricConfig(data_range=1.0),
        context_planes=context_planes,
        context_gap_mm=4.0,
        local_observability=observability,
    )
    assert metrics.distance_to_context_plane_status == "COMPUTED_SUPPORT_CONDITIONED"
    assert metrics.distance_to_context_plane_mean_mm >= 0.0
    assert metrics.distance_to_context_plane_max_mm >= metrics.distance_to_context_plane_mean_mm
    assert metrics.distance_to_context_plane_strata
    assert metrics.context_gap_status == "COMPUTED_SUPPORT_CONDITIONED"
    assert metrics.context_gap_mm == pytest.approx(4.0)
    assert metrics.error_vs_context_gap_mae == pytest.approx(metrics.mae)
    assert metrics.local_observability_status == "COMPUTED_SUPPORT_CONDITIONED"
    assert metrics.local_observability_mean == pytest.approx(float(observability.mean()))
    assert metrics.error_vs_local_observability_strata


def test_evaluator_reports_segmentation_roi_and_boundary_metrics() -> None:
    volume = reconstruct_volume(_state(), _grid(), modality_id="t1")
    prediction = VolumeReconstruction(
        volume.patient_id, volume.modality_id, volume.grid,
        volume.intensity.nan_to_num(), volume.support_mass,
        torch.zeros_like(volume.unsupported_mask), volume.support_uncertainty,
        volume.depth_chunk_size, volume.renderer_config_hash,
        volume.patient_state_version, volume.artifact_hash,
    )
    target = prediction.intensity + 0.1
    segmentation = torch.zeros_like(target, dtype=torch.uint8)
    segmentation[:, 2:4, 2:4] = 1
    metrics = compute_reconstruction_metrics(
        prediction,
        target,
        torch.ones_like(target, dtype=torch.bool),
        metric_config=ReconstructionMetricConfig(data_range=1.0),
        segmentation=segmentation,
    )
    assert metrics.roi_status == "COMPUTED"
    assert metrics.roi_voxels == 12
    assert metrics.supported_roi_fraction == pytest.approx(1.0)
    assert metrics.roi_mae == pytest.approx(0.1)
    assert metrics.tumor_mae == pytest.approx(0.1)
    assert metrics.non_tumor_mae == pytest.approx(0.1)
    assert metrics.boundary_band_voxels > 0
    assert metrics.supported_boundary_band_fraction == pytest.approx(1.0)
    assert metrics.boundary_band_mae == pytest.approx(0.1)


def test_patient_paired_statistics_are_deterministic() -> None:
    first = {"a": 1.0, "b": 2.0, "c": 3.0}; second = {"a": 0.5, "b": 1.0, "c": 2.5}
    assert paired_patient_summary(first, second, seed=7) == paired_patient_summary(first, second, seed=7)


def test_export_provenance_rejects_placeholder_commit_and_corrupt_volume(tmp_path) -> None:
    volume = reconstruct_volume(_state(), _grid(), modality_id="t1")
    common = dict(
        config_hash=_digest("config"), manifest_hash=_digest("manifest"), split_hash=_digest("split"),
        assignment_hash=_digest("assignment"), encoder_identity="e2", field_identity="field",
        gaussian_identity="memory", propagation_identity="p0", environment_hash=_digest("environment"),
    )
    with pytest.raises(ValueError, match="exact repository commit"):
        build_reconstruction_package((volume,), repository_commit="0" * 40, **common)
    package = build_reconstruction_package((volume,), repository_commit=hashlib.sha1(b"commit").hexdigest(), **common)
    directory = export_reconstruction_package(package, (volume,), tmp_path / "corrupt")
    artifact = directory / "volume_t1.pt"
    artifact.write_bytes(artifact.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        open_serialized_predictions(directory)


def test_safe_audit_target_round_trip_preserves_grid_and_mask(tmp_path) -> None:
    grid = _grid()
    context_plane = PhysicalPlane(
        pixel_center_origin_ras_mm=(-2.0, -2.0, -2.0), axis_u_ras=(1.0, 0.0, 0.0),
        axis_v_ras=(0.0, 1.0, 0.0), spacing_uv_mm=(1.0, 1.0), thickness_mm=1.0,
        shape_hw=(5, 5), signed_normal_ras=(0.0, 0.0, 1.0),
    )
    segmentation = torch.zeros(grid.shape_dhw, dtype=torch.uint8)
    segmentation[:, 2, 2] = 1
    path = tmp_path / "targets.pt"
    torch.save({
        "schema": "smagm-audit-targets-v1",
        "targets": [{
            "patient_id": "p", "split_hash": _digest("split"), "modality_id": "t1",
            "grid": grid.to_canonical_dict(), "values": torch.ones(grid.shape_dhw),
            "valid_mask": torch.ones(grid.shape_dhw, dtype=torch.bool),
            "context_planes": [context_plane.to_canonical_dict()],
            "context_gap_mm": 4.0,
            "local_observability": torch.full(grid.shape_dhw, 0.5),
        }],
        "segmentation": segmentation,
        "segmentation_evaluator_only": True,
    }, path)
    target = open_serialized_audit_targets(path)[0]
    assert target.grid.canonical_json() == grid.canonical_json()
    assert target.valid_mask.dtype is torch.bool
    assert target.context_planes[0].canonical_json() == context_plane.canonical_json()
    assert target.context_gap_mm == pytest.approx(4.0)
    assert torch.equal(target.local_observability, torch.full(grid.shape_dhw, 0.5))
    assert torch.equal(target.segmentation, segmentation)


def test_isolated_audit_owner_imports_no_training_state_or_reconstruction_generation() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "smagm" / "evaluation"
    source = (root / "audit.py").read_text(encoding="utf-8")
    forbidden = ("..training", "..state", "..reconstruction", "..features", "..anchors", "..memory")
    assert [name for name in forbidden if name in source] == []
