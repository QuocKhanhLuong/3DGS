from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from smagm.features.point_guided.contracts import VolumeGeometry
from smagm.features.point_guided.pfgr_lite.data import (
    DataAccessCounters,
    TargetFreeSample,
    build_training_role_manifest,
    defer_supervision,
    load_observation_sample,
    normalization_identity,
)


def _sample(subject: str = "subject-01") -> TargetFreeSample:
    geometry = VolumeGeometry.from_spacing((2, 2, 2), (1.0, 1.5, 2.0))
    return TargetFreeSample(
        subject,
        torch.zeros(3, 2, 2, 2, dtype=torch.float32),
        torch.ones(1, 2, 2, 2, dtype=torch.bool),
        geometry,
        {"policy": "masked_zscore", "mask_version": "v1"},
        "",
        "",
    )


def test_observation_loader_passes_all_target_flags_false(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = _sample()
    seen: dict[str, object] = {}

    def fake_loader(subject, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            subject_id=sample.subject_id,
            observations=sample.observations,
            brain_mask=sample.brain_mask,
            geometry=sample.geometry,
            normalization_metadata=sample.normalization_metadata,
            source_paths={},
            target=None,
            segmentation=None,
        )

    monkeypatch.setattr("smagm.data.brats21_point_guided.load_point_guided_subject", fake_loader)
    counters = DataAccessCounters()
    loaded = load_observation_sample("subject-01", counters=counters)
    assert loaded.target_free
    assert seen["require_target"] is False
    assert seen["load_target"] is False
    assert seen["require_segmentation"] is False
    assert seen["load_segmentation"] is False
    assert counters.observation_reads == 1


def test_deferred_supervision_is_late_and_one_shot() -> None:
    sample = _sample()
    calls: list[str] = []

    def provider(subject_id: str) -> torch.Tensor:
        calls.append(subject_id)
        return torch.ones(1, 2, 2, 2)

    callback = defer_supervision(sample, provider, engineering_only=True)
    assert calls == []
    context = callback(prediction=torch.zeros(1, 1, 2, 2, 2))
    assert calls == [sample.subject_id]
    assert torch.equal(context.target, torch.ones_like(context.target))
    with pytest.raises(RuntimeError, match="consumed only once"):
        callback(prediction=torch.zeros(1, 1, 2, 2, 2))


def test_deferred_semantic_supervision_loads_segmentation_only_after_prediction() -> None:
    sample = _sample()
    calls: list[str] = []
    loaded = SimpleNamespace(
        subject_id=sample.subject_id,
        observations=sample.observations,
        brain_mask=sample.brain_mask,
        geometry=sample.geometry,
        normalization_metadata={},
        source_paths={},
        target=torch.ones(1, 2, 2, 2),
        segmentation=torch.tensor([[[0, 1], [2, 4]], [[0, 1], [2, 4]]], dtype=torch.int64),
    )
    counters = DataAccessCounters()

    def provider(subject_id: str):
        calls.append(subject_id)
        return loaded

    callback = defer_supervision(
        sample,
        provider,
        counters=counters,
        engineering_only=True,
        include_segmentation=True,
    )
    assert calls == []
    joined = callback(prediction=torch.zeros(1, 1, 2, 2, 2))
    assert calls == [sample.subject_id]
    assert set(joined) == {"target_context", "semantic_target"}
    assert tuple(joined["semantic_target"].shape) == (1, 2, 2, 2)
    assert set(torch.unique(joined["semantic_target"]).tolist()) == {0, 1, 2}
    assert counters.target_reads == 1
    assert counters.segmentation_reads == 1


def test_deferred_supervision_rejects_geometry_mismatch() -> None:
    sample = _sample()
    wrong_geometry = VolumeGeometry.from_spacing((2, 2, 2), (1.0, 1.0, 1.0))
    context = SimpleNamespace(
        context_id="ctx",
        geometry=wrong_geometry,
        producer=SimpleNamespace(observation_normalization_hash=sample.normalization_hash),
    )
    callback = defer_supervision(sample, lambda _sid: torch.zeros(1, 2, 2, 2), engineering_only=True)
    with pytest.raises(ValueError, match="geometry"):
        callback(completed_context=context, prediction=torch.zeros(1, 1, 2, 2, 2))


def test_role_manifest_preserves_baseline_and_engineering_small_fixture() -> None:
    baseline = {
        "train": ["s3", "s1", "s2"],
        "val": ["v1"],
        "test": ["t1"],
        "split_hash": "baseline-hash",
    }
    manifest = build_training_role_manifest(baseline, engineering_only=True)
    assert set(manifest.baseline_train_subject_ids) == {"s1", "s2", "s3"}
    assert set(manifest.producer_fit_subject_ids) == {"s1", "s2", "s3"}
    assert manifest.calibration_fit_subject_ids == ()
    assert manifest.calibration_allowance_subject_ids == ()
    assert set(manifest.producer_fit_subject_ids) | set(manifest.calibration_fit_subject_ids) | set(manifest.calibration_allowance_subject_ids) == {"s1", "s2", "s3"}
    assert manifest.digest
    with pytest.raises(ValueError, match="crosses baseline splits"):
        build_training_role_manifest(baseline, related_groups={"s1": "linked", "v1": "linked"}, engineering_only=True)


def _write_nifti_subject(root: Path, subject_id: str, *, target_value: float = 4.0) -> Path:
    import nibabel as nib
    import numpy as np

    subject = root / subject_id
    subject.mkdir()
    affine = np.asarray(
        ((1.1, 0.1, 0.0, 12.0), (0.0, 1.4, 0.1, -4.0), (0.0, 0.0, 1.7, 8.0), (0.0, 0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    for modality, value in (("t1", 1.0), ("t2", 2.0), ("flair", 3.0), ("t1ce", target_value)):
        nib.save(nib.Nifti1Image(np.full((9, 9, 9), value, dtype=np.float32), affine), str(subject / f"{subject_id}_{modality}.nii.gz"))
    return subject


def test_nifti_adapter_preserves_affine_and_deferred_target_identity(tmp_path: Path) -> None:
    """Exercise the real NIfTI adapter through PFGR encode/decode and late join."""

    from smagm.data.brats21_point_guided import load_point_guided_subject
    from smagm.features.point_guided.pfgr_lite.config import PFGRLiteConfig
    from smagm.features.point_guided.pfgr_lite.footprint import PFGRQueryLattice
    from smagm.features.point_guided.pfgr_lite.model import PFGRLiteModel
    from smagm.features.point_guided.pfgr_lite.provenance import canonical_digest

    subject = _write_nifti_subject(tmp_path, "BraTS2021_00001")
    other = _write_nifti_subject(tmp_path, "BraTS2021_00002", target_value=7.0)
    recipe = {"normalization_policy": "masked_zscore"}
    sample = load_observation_sample(subject, normalization_config=recipe)
    assert sample.geometry.__class__.__name__ == "VolumeGeometry"
    assert sample.geometry.voxel_to_ras_mm[0][1] == pytest.approx(0.1, abs=1e-6)
    recipe_hash = normalization_identity(config=recipe)
    producer_hash = canonical_digest(recipe_hash, prefix="pfgr-lite-observation-normalization-v1|")
    config = PFGRLiteConfig(
        num_points=4,
        engineering_only=True,
        build_chunk_size=64,
        decode_chunk_size=64,
        observation_normalization=recipe_hash,
    )

    class LatticeFactory:
        def build(self, **kwargs):
            return PFGRQueryLattice.build(**kwargs)

    model = PFGRLiteModel(config, query_lattice_factory=LatticeFactory())
    sample = replace(
        sample,
        normalization_hash=producer_hash,
        normalization_metadata={**dict(sample.normalization_metadata), "producer_normalization_hash": producer_hash},
    )
    context = model.encode_observations(sample.observations.unsqueeze(0), sample.brain_mask, sample.geometry)
    prediction = model.decode_final(model.initialize_state(context), context, chunk_size=64)
    calls: list[str] = []

    def provider(subject_id: str):
        calls.append(subject_id)
        return load_point_guided_subject(
            subject,
            require_target=True,
            load_target=True,
            require_segmentation=False,
            load_segmentation=False,
            normalization_policy="masked_zscore",
        )

    callback = defer_supervision(sample, provider, counters=DataAccessCounters())
    assert calls == []
    joined = callback(completed_context=context, prediction=prediction)
    assert tuple(joined.target.shape) == (1, 9, 9, 9)
    assert calls == [sample.subject_id]

    # An arbitrary marker must not trigger a production target read.
    calls.clear()
    guarded = defer_supervision(sample, provider, counters=DataAccessCounters())
    with pytest.raises(TypeError, match="ObservationContext"):
        guarded(completed_context=object(), prediction=object())
    assert calls == []

    # Same-shape/same-affine data from another subject is rejected at the
    # observation/source identity boundary, even if its target is valid.
    def wrong_provider(subject_id: str):
        del subject_id
        return load_point_guided_subject(
            other,
            require_target=True,
            load_target=True,
            require_segmentation=False,
            load_segmentation=False,
            normalization_policy="masked_zscore",
        )

    wrong = defer_supervision(sample, wrong_provider)
    with pytest.raises(ValueError, match="subject"):
        wrong(completed_context=context, prediction=prediction)
