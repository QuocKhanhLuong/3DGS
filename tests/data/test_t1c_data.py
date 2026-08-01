from __future__ import annotations

import hashlib
from io import BytesIO

import numpy as np
import pytest
import torch

from smagm.contracts.coordinates import PhysicalPlane
from smagm.contracts.observation import AvailabilityObservationMeta, SparseAvailabilityManifest
from smagm.data.episodes import (
    EpisodeSamplingConfig,
    EpisodeSamplingError,
    EpisodeSamplingFailureReason,
    ModalityEpisodePolicy,
    build_episode_schedule,
)
from smagm.data.io import DecoderConfig, decode_observation
from smagm.data.normalization import (
    DegenerateNormalizationError,
    FrozenPopulationStatistic,
    NormalizationConfig,
    apply_preprocessing,
    fit_preprocessing,
)


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


def _array_payload(values: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.save(buffer, values.astype(np.float32), allow_pickle=False)
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
    assert torch.allclose(normalized_target.image, torch.full_like(normalized_target.image, 7.0))
    assert torch.isfinite(normalized_target.image).all()
    assert normalized_target.image.abs().max().item() < 10.0


def test_unseen_target_modality_is_explicitly_rejected() -> None:
    context = decode_observation(_payload(2.0), _metadata("context", "T2"))
    target = decode_observation(_payload(9.0), _metadata("target", "FLAIR"))
    strict = fit_preprocessing((context,), context_ids=("context",))
    with pytest.raises(ValueError, match="no context-derived"):
        apply_preprocessing(strict, target)
    with pytest.raises(ValueError, match="rejects unseen"):
        NormalizationConfig(unseen_modality_policy="identity")  # type: ignore[arg-type]


def test_degenerate_context_scale_policy_is_explicit_and_hashable() -> None:
    context = decode_observation(_payload(2.0), _metadata("context"))
    identity = fit_preprocessing((context,), context_ids=("context",))
    parameter = identity.modality_parameters[0]
    assert parameter.center == pytest.approx(2.0)
    assert parameter.scale == 1.0
    assert parameter.fallback_reason == "CONTEXT_SCALE_BELOW_MINIMUM"
    assert len(identity.record_hash) == 64
    with pytest.raises(DegenerateNormalizationError, match="CONTEXT_SCALE_BELOW_MINIMUM"):
        fit_preprocessing(
            (context,),
            context_ids=("context",),
            config=NormalizationConfig(degenerate_scale_policy="reject_episode"),
        )
    frozen = fit_preprocessing(
        (context,),
        context_ids=("context",),
        config=NormalizationConfig(
            degenerate_scale_policy="frozen_population_scale",
            frozen_population_statistics={"T2": FrozenPopulationStatistic(3.0, 2.0, "a" * 64)},
        ),
    )
    assert frozen.modality_parameters[0].scale == 2.0
    assert frozen.modality_parameters[0].source_statistic_hash == "a" * 64


def test_ordinary_context_uses_measured_scale_and_stays_finite() -> None:
    values = np.arange(42, dtype=np.float32).reshape(7, 6)
    context = decode_observation(_array_payload(values), _metadata("context"))
    record = fit_preprocessing((context,), context_ids=("context",))
    parameter = record.modality_parameters[0]
    normalized = apply_preprocessing(record, context)
    assert parameter.scale > record.minimum_context_scale
    assert parameter.fallback_reason is None
    assert torch.isfinite(normalized.image).all()
    assert normalized.image.abs().max().item() < 3.0


def test_low_variance_context_uses_the_same_declared_identity_fallback() -> None:
    values = np.full((7, 6), 4.0, dtype=np.float32)
    values[0, 0] = 4.00001
    context = decode_observation(_array_payload(values), _metadata("context"))
    record = fit_preprocessing((context,), context_ids=("context",))
    parameter = record.modality_parameters[0]
    assert parameter.scale == 1.0
    assert parameter.fallback_reason == "CONTEXT_SCALE_BELOW_MINIMUM"


def test_input_content_hash_binds_payload_mask_and_preprocessing_record() -> None:
    first = decode_observation(_payload(2.0), _metadata("context"))
    second = decode_observation(_payload(3.0), _metadata("context"))
    first_record = fit_preprocessing((first,), context_ids=("context",))
    second_record = fit_preprocessing((second,), context_ids=("context",))
    assert apply_preprocessing(first_record, first).input_content_hash != apply_preprocessing(second_record, second).input_content_hash
    masked = decode_observation(_payload(2.0, nonfinite=True), _metadata("context"))
    masked_record = fit_preprocessing((masked,), context_ids=("context",))
    assert apply_preprocessing(first_record, first).input_content_hash != apply_preprocessing(masked_record, masked).input_content_hash
    identity_record = fit_preprocessing(
        (first,), context_ids=("context",), config=NormalizationConfig(policy="identity")
    )
    assert apply_preprocessing(first_record, first).input_content_hash != apply_preprocessing(identity_record, first).input_content_hash


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


def test_modality_policy_requires_a_same_modality_context_and_rejects_multi_target() -> None:
    entries = (_metadata("t2", "T2"), _metadata("flair", "FLAIR"))
    manifest = SparseAvailabilityManifest(
        entries,
        manifest_id="m-modalities",
        integrity_digests={entry.observation_id: hashlib.sha256(entry.observation_id.encode()).hexdigest() for entry in entries},
    )
    with pytest.raises(EpisodeSamplingError) as exc_info:
        build_episode_schedule(
            manifest,
            patient_id="patient-a",
            config=EpisodeSamplingConfig(context_count=1, seed=2, modality_policy=ModalityEpisodePolicy()),
        )
    assert exc_info.value.reason is EpisodeSamplingFailureReason.MISSING_CONTEXT_MODALITY
    with pytest.raises(ValueError, match="requires same_modality"):
        ModalityEpisodePolicy(same_modality_context_required=False)
    with pytest.raises(ValueError, match="exactly one target"):
        EpisodeSamplingConfig(context_count=1, target_count=2)
