from __future__ import annotations

import hashlib
from io import BytesIO

import numpy as np
import pytest
import torch

from smagm.contracts.coordinates import PhysicalPlane
from smagm.contracts.observation import AvailabilityObservationMeta, SparseAvailabilityManifest
from smagm.data.episodes import EpisodeSamplingConfig, build_episode_schedule
from smagm.data.io import DecoderConfig, decode_observation
from smagm.data.normalization import NormalizationConfig, apply_preprocessing, fit_preprocessing


def _plane(observation_id: str, shape: tuple[int, int] = (7, 6)) -> PhysicalPlane:
    return PhysicalPlane(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0),
        1.0,
        shape,
        (0.0, 0.0, 1.0),
        observation_id=observation_id,
    )


def _metadata(observation_id: str, modality: str = "T2") -> AvailabilityObservationMeta:
    return AvailabilityObservationMeta(
        observation_id=observation_id,
        patient_id="patient-a",
        split="train",
        relative_path=f"{observation_id}.npy",
        modality_id=modality,
        plane=_plane(observation_id),
        is_synthetic=True,
    )


def _payload(value: float, *, nonfinite: bool = False) -> bytes:
    array = np.full((7, 6), value, dtype=np.float32)
    if nonfinite:
        array[0, 0] = np.nan
    buffer = BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def test_decoder_accepts_bytes_only_preserves_plane_and_masks_nonfinite() -> None:
    meta = _metadata("context")
    decoded = decode_observation(_payload(3.0, nonfinite=True), meta)
    assert decoded.image.shape == (1, 7, 6)
    assert decoded.valid_mask.sum().item() == 41
    assert decoded.image[0, 0, 0].item() == 0.0
    assert decoded.metadata.plane.canonical_json() == meta.plane.canonical_json()
    with pytest.raises(TypeError, match="bytes"):
        decode_observation("context.npy", meta)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-finite"):
        decode_observation(_payload(3.0, nonfinite=True), meta, config=DecoderConfig(nonfinite_policy="reject"))


def test_context_only_normalization_rejects_undeclared_inputs_and_freezes_target_transform() -> None:
    context = decode_observation(_payload(2.0), _metadata("context"))
    target = decode_observation(_payload(9.0), _metadata("target"))
    with pytest.raises(ValueError, match="complete declared context"):
        fit_preprocessing((context, target), context_ids=("context",))
    record = fit_preprocessing((context,), context_ids=("context",))
    normalized_target = apply_preprocessing(record, target)
    assert record.fitted_from_context_ids == ("context",)
    assert torch.allclose(normalized_target.image, torch.full_like(normalized_target.image, 7e6))


def test_unseen_target_modality_is_explicitly_rejected_or_identity_transformed() -> None:
    context = decode_observation(_payload(2.0), _metadata("context", "T2"))
    target = decode_observation(_payload(9.0), _metadata("target", "FLAIR"))
    strict = fit_preprocessing((context,), context_ids=("context",))
    with pytest.raises(ValueError, match="no context-derived"):
        apply_preprocessing(strict, target)
    identity = fit_preprocessing(
        (context,),
        context_ids=("context",),
        config=NormalizationConfig(unseen_modality_policy="identity"),
    )
    assert torch.equal(apply_preprocessing(identity, target).image, target.image)


def test_episode_schedule_is_deterministic_metadata_only_and_patient_bound() -> None:
    entries = tuple(_metadata(f"obs-{index}") for index in range(4))
    digests = {entry.observation_id: hashlib.sha256(entry.observation_id.encode()).hexdigest() for entry in entries}
    manifest = SparseAvailabilityManifest(entries, manifest_id="m", integrity_digests=digests)
    config = EpisodeSamplingConfig(context_count=2, target_count=1, episode_count=3, seed=41)
    first = build_episode_schedule(manifest, patient_id="patient-a", config=config)
    second = build_episode_schedule(manifest, patient_id="patient-a", config=config)
    assert first.schedule_hash == second.schedule_hash
    assert [item.assignment_hash for item in first.assignments] == [item.assignment_hash for item in second.assignments]
    assert all(set(item.context_ids).isdisjoint(item.target_ids) for item in first.assignments)
