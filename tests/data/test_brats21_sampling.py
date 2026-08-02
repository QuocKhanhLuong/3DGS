from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from smagm.cli.brats21_product import _prepared_bundle_complete
from smagm.data.brats21 import BRATS21_MODALITIES, BraTS21VolumeSummary
from smagm.data.brats21 import canonical_hash
from smagm.data.brats21_prepare import load_prepared_bundle, prepare_brats21_product_patient
from smagm.data.brats21_sampling import BraTS21SamplingConfig, build_sampling_plan, physical_slice_positions


def _summary(modality: str) -> BraTS21VolumeSummary:
    affine = ((0.0, -2.0, 0.0, 10.0), (1.0, 0.0, 0.0, -20.0), (0.0, 0.0, 1.5, 5.0), (0.0, 0.0, 0.0, 1.0))
    return BraTS21VolumeSummary(
        modality, (8, 10, 31), (1.0, 2.0, 1.5), affine, ("L", "P", "S"),
        "float32", (0.0, 1.0), None, hashlib.sha256(modality.encode()).hexdigest(),
    )


def _summaries() -> dict[str, BraTS21VolumeSummary]:
    return {modality: _summary(modality) for modality in BRATS21_MODALITIES}


def test_aligned_quantile_selection_is_deterministic_and_physical() -> None:
    summaries = _summaries()
    config = BraTS21SamplingConfig()
    first = build_sampling_plan(summaries, episode_id="episode-a", target_modality="flair", seed=17, config=config)
    second = build_sampling_plan(summaries, episode_id="episode-a", target_modality="flair", seed=17, config=config)
    assert first.protocol_hash == second.protocol_hash
    assert len(first.context) == 20
    assert len({item.physical_position_mm for item in first.context}) == 5
    assert first.target.physical_position_mm not in {item.physical_position_mm for item in first.context}
    assert first.target.physical_position_mm == pytest.approx(
        first.target.plane.source_transform.origin_ras_mm[2] - summaries["flair"].affine[2][3]  # type: ignore[union-attr]
    )
    positions = physical_slice_positions(summaries["flair"])
    context_positions = sorted({item.physical_position_mm for item in first.context})
    midpoint_gaps = {
        (left + right) / 2.0
        for left, right in zip(context_positions, context_positions[1:])
    }
    assert any(first.target.physical_position_mm == pytest.approx(value) for value in midpoint_gaps)
    expected_fractional_index = np.interp(
        first.target.physical_position_mm,
        np.asarray(positions),
        np.arange(len(positions), dtype=np.float64),
    )
    assert first.target.source_slice_position_index == pytest.approx(expected_fractional_index)


def test_train_jitter_is_seeded_and_validation_is_not_jittered() -> None:
    summaries = _summaries()
    train = BraTS21SamplingConfig(train_jitter_fraction=0.15)
    first = build_sampling_plan(summaries, episode_id="episode-a", split="train", seed=9, config=train)
    second = build_sampling_plan(summaries, episode_id="episode-a", split="train", seed=9, config=train)
    other = build_sampling_plan(summaries, episode_id="episode-a", split="train", seed=10, config=train)
    assert first.protocol_hash == second.protocol_hash
    assert first.protocol_hash != other.protocol_hash
    validation = build_sampling_plan(summaries, episode_id="episode-a", split="validation", seed=9, config=train)
    assert tuple(item.source_slice_index for item in validation.context) == tuple(
        item.source_slice_index for item in build_sampling_plan(summaries, episode_id="episode-a", split="validation", seed=10, config=train).context
    )


def test_staggered_protocol_preserves_legal_roles_and_no_target_overlap() -> None:
    summaries = _summaries()
    plan = build_sampling_plan(
        summaries,
        episode_id="episode-staggered",
        target_modality="t2",
        seed=3,
        config=BraTS21SamplingConfig(modality_alignment="staggered"),
    )
    assert len(plan.context) == 20
    assert all(item.role == "context" for item in plan.context)
    assert plan.target.modality_id == "t2"
    assert plan.target.physical_position_mm not in {item.physical_position_mm for item in plan.context}
    target_context_positions = sorted({
        item.physical_position_mm for item in plan.context if item.modality_id == plan.target_modality
    })
    assert any(
        left < plan.target.physical_position_mm < right
        for left, right in zip(target_context_positions, target_context_positions[1:])
    )


def test_product_preparation_materializes_only_declared_planes_and_binds_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nib = pytest.importorskip("nibabel")
    source = tmp_path / "source"
    patient = source / "BraTS2021_00000"
    patient.mkdir(parents=True)
    shape = (8, 9, 31)
    affine = np.array(((0.0, -2.0, 0.0, 10.0), (1.0, 0.0, 0.0, -20.0), (0.0, 0.0, 1.5, 5.0), (0.0, 0.0, 0.0, 1.0)))
    for index, modality in enumerate((*BRATS21_MODALITIES, "seg")):
        values = np.zeros(shape, dtype=np.uint8) if modality == "seg" else (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + index)
        nib.save(nib.Nifti1Image(values, affine), patient / f"BraTS2021_00000_{modality}.nii.gz")
    source_hashes = {
        modality: hashlib.sha256((patient / f"BraTS2021_00000_{modality}.nii.gz").read_bytes()).hexdigest()
        for modality in (*BRATS21_MODALITIES, "seg")
    }
    import smagm.data.brats21_prepare as prepare_module

    extracted: list[tuple[str, int]] = []
    real_extract = prepare_module.extract_axial_plane

    def record_context_extraction(path: Path, slice_index: int, *, inplane_stride_vu: tuple[int, int]):
        extracted.append((path.name, int(slice_index)))
        return real_extract(path, slice_index, inplane_stride_vu=inplane_stride_vu)

    monkeypatch.setattr(
        prepare_module,
        "_sha256_file",
        lambda path: pytest.fail(f"product preparation re-hashed a source/evaluator file: {path}"),
    )
    monkeypatch.setattr(prepare_module, "extract_axial_plane", record_context_extraction)
    prepared = prepare_brats21_product_patient(
        source_root=source,
        output_dir=tmp_path / "prepared",
        patient_id="BraTS2021_00000",
        inplane_stride_vu=(2, 2),
        sampling_config=BraTS21SamplingConfig(),
        source_hashes=source_hashes,
    )
    bundle = load_prepared_bundle(tmp_path / "prepared")
    assert prepared["context_count"] == 20
    target_reference = bundle.evaluator_json["target_reference"]
    assert len(extracted) == 20
    assert all(not name.endswith("_seg.nii.gz") for name, _ in extracted)
    assert ("BraTS2021_00000_flair.nii.gz", int(target_reference["source_slice_index"])) not in extracted
    assert len(bundle.assignment.context_ids) == 20
    assert bundle.target_id not in bundle.assignment.context_ids
    assert bundle.target_payload_deferred
    assert not bundle.target_payload_path.exists()
    assert bundle.segmentation_payload_deferred
    assert not bundle.segmentation_payload_path.exists()
    assert all(item.modality_id in BRATS21_MODALITIES for item in bundle.manifest.entries)
    assert isinstance(target_reference["source_slice_position_index"], float)
    assert target_reference["source_shape_xyz"] == [8, 9, 31]
    assert target_reference["source_slice_position_index"] != target_reference["source_slice_index"]
    segmentation_reference = bundle.evaluator_json["segmentation_reference"]
    assert segmentation_reference["source_shape_xyz"] == [8, 9, 31]
    assert len(segmentation_reference["reference_sha256"]) == 64
    assert _prepared_bundle_complete(
        tmp_path / "prepared",
        expected_patient_id="BraTS2021_00000",
        expected_split="validation",
        expected_target_modality="flair",
        expected_sampling_protocol_hash=prepared["sampling_protocol_hash"],
        expected_source_hashes=source_hashes,
    )
    context_path = next((tmp_path / "prepared" / "payloads" / "context").iterdir())
    context_path.write_bytes(context_path.read_bytes() + b"corrupt")
    assert not _prepared_bundle_complete(
        tmp_path / "prepared",
        expected_patient_id="BraTS2021_00000",
        expected_split="validation",
        expected_target_modality="flair",
        expected_sampling_protocol_hash=prepared["sampling_protocol_hash"],
        expected_source_hashes=source_hashes,
    )


def test_prepared_bundle_rejects_cross_file_target_plane_tamper(tmp_path: Path) -> None:
    nib = pytest.importorskip("nibabel")
    source = tmp_path / "source-plane-tamper"
    patient = source / "BraTS2021_00003"
    patient.mkdir(parents=True)
    shape = (8, 9, 31)
    affine = np.eye(4, dtype=np.float64)
    for index, modality in enumerate((*BRATS21_MODALITIES, "seg")):
        values = np.zeros(shape, dtype=np.uint8) if modality == "seg" else (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + index)
        nib.save(nib.Nifti1Image(values, affine), patient / f"BraTS2021_00003_{modality}.nii.gz")
    source_hashes = {
        modality: hashlib.sha256((patient / f"BraTS2021_00003_{modality}.nii.gz").read_bytes()).hexdigest()
        for modality in (*BRATS21_MODALITIES, "seg")
    }
    prepared_root = tmp_path / "prepared-plane-tamper"
    prepare_brats21_product_patient(
        source_root=source,
        output_dir=prepared_root,
        patient_id="BraTS2021_00003",
        inplane_stride_vu=(2, 2),
        sampling_config=BraTS21SamplingConfig(),
        source_hashes=source_hashes,
    )
    evaluator_path = prepared_root / "evaluator_manifest.json"
    hashes_path = prepared_root / "hashes.json"
    prepared_path = prepared_root / "prepared.json"
    evaluator = json.loads(evaluator_path.read_text(encoding="utf-8"))
    evaluator["target_plane"] = dict(evaluator["target_plane"])
    origin = list(evaluator["target_plane"]["pixel_center_origin_ras_mm"])
    origin[0] += 0.5
    evaluator["target_plane"]["pixel_center_origin_ras_mm"] = origin
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    hashes["evaluator_manifest"] = canonical_hash(evaluator)
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    prepared["evaluator_manifest_hash"] = hashes["evaluator_manifest"]
    prepared["hashes"] = hashes
    evaluator_path.write_text(json.dumps(evaluator), encoding="utf-8")
    hashes_path.write_text(json.dumps(hashes), encoding="utf-8")
    prepared_path.write_text(json.dumps(prepared), encoding="utf-8")
    with pytest.raises(ValueError, match="target plane"):
        load_prepared_bundle(prepared_root)


def test_product_preparation_keeps_optional_segmentation_absent(tmp_path: Path) -> None:
    nib = pytest.importorskip("nibabel")
    source = tmp_path / "source-no-seg"
    patient = source / "BraTS2021_00001"
    patient.mkdir(parents=True)
    shape = (8, 9, 31)
    affine = np.eye(4, dtype=np.float64)
    for index, modality in enumerate(BRATS21_MODALITIES):
        values = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + index)
        nib.save(nib.Nifti1Image(values, affine), patient / f"BraTS2021_00001_{modality}.nii.gz")
    source_hashes = {
        modality: hashlib.sha256((patient / f"BraTS2021_00001_{modality}.nii.gz").read_bytes()).hexdigest()
        for modality in BRATS21_MODALITIES
    }
    prepare_brats21_product_patient(
        source_root=source,
        output_dir=tmp_path / "prepared-no-seg",
        patient_id="BraTS2021_00001",
        inplane_stride_vu=(2, 2),
        sampling_config=BraTS21SamplingConfig(),
        source_hashes=source_hashes,
        require_segmentation=False,
    )
    bundle = load_prepared_bundle(tmp_path / "prepared-no-seg")
    assert not bundle.segmentation_payload_deferred
    assert bundle.segmentation_payload_path is None


def test_product_preparation_rejects_malformed_inventory_hash(tmp_path: Path) -> None:
    nib = pytest.importorskip("nibabel")
    source = tmp_path / "source-mutated"
    patient = source / "BraTS2021_00002"
    patient.mkdir(parents=True)
    shape = (8, 9, 31)
    affine = np.eye(4, dtype=np.float64)
    for index, modality in enumerate(BRATS21_MODALITIES):
        values = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + index)
        nib.save(nib.Nifti1Image(values, affine), patient / f"BraTS2021_00002_{modality}.nii.gz")
    source_hashes = {
        modality: hashlib.sha256((patient / f"BraTS2021_00002_{modality}.nii.gz").read_bytes()).hexdigest()
        for modality in BRATS21_MODALITIES
    }
    malformed_hashes = dict(source_hashes)
    malformed_hashes["t1"] = "0" * 63
    with pytest.raises(ValueError, match="malformed source hashes"):
        prepare_brats21_product_patient(
            source_root=source,
            output_dir=tmp_path / "prepared-mutated",
            patient_id="BraTS2021_00002",
            inplane_stride_vu=(2, 2),
            sampling_config=BraTS21SamplingConfig(),
            source_hashes=malformed_hashes,
            require_segmentation=False,
        )
