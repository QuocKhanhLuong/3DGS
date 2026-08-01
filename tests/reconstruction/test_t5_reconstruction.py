from __future__ import annotations

import hashlib
from pathlib import Path
import struct

import pytest
import torch

from smagm.anchors import AnchorBatch, AnchorGeometryBatch
from smagm.anchors.contracts import anchor_evidence_hash
from smagm.contracts.coordinates import PhysicalPlane, TargetGrid
from smagm.evaluation import AuditTarget, FreezeRecord, evaluate_audit_targets, open_serialized_audit_targets, open_serialized_predictions, paired_patient_summary
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
    path = tmp_path / "targets.pt"
    torch.save({
        "schema": "smagm-audit-targets-v1",
        "targets": [{
            "patient_id": "p", "split_hash": _digest("split"), "modality_id": "t1",
            "grid": grid.to_canonical_dict(), "values": torch.ones(grid.shape_dhw),
            "valid_mask": torch.ones(grid.shape_dhw, dtype=torch.bool),
        }],
    }, path)
    target = open_serialized_audit_targets(path)[0]
    assert target.grid.canonical_json() == grid.canonical_json()
    assert target.valid_mask.dtype is torch.bool


def test_isolated_audit_owner_imports_no_training_state_or_reconstruction_generation() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "smagm" / "evaluation"
    source = (root / "audit.py").read_text(encoding="utf-8")
    forbidden = ("..training", "..state", "..reconstruction", "..features", "..anchors", "..memory")
    assert [name for name in forbidden if name in source] == []
