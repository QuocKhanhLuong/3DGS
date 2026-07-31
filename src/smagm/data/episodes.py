"""Deterministic, metadata-only episode assignment schedules.

The sampler operates on the sealed sparse manifest only.  It makes modality
requirements explicit before any payload is opened, so a target can never be
selected merely because it happened to occur after an ID shuffle.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import random
from typing import Literal

from ..contracts.episode import EpisodeAssignment
from ..contracts.observation import AvailabilityObservationMeta, SparseAvailabilityManifest


class EpisodeSamplingFailureReason(str, Enum):
    """Typed reasons why a manifest cannot yield a legal T1-C episode."""

    INSUFFICIENT_OBSERVATIONS = "INSUFFICIENT_OBSERVATIONS"
    NO_LEGAL_TARGET = "NO_LEGAL_TARGET"
    MISSING_CONTEXT_MODALITY = "MISSING_CONTEXT_MODALITY"
    INVALID_MANIFEST_BINDING = "INVALID_MANIFEST_BINDING"


class EpisodeSamplingError(ValueError):
    """A deterministic episode failure with a machine-readable reason."""

    def __init__(self, reason: EpisodeSamplingFailureReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class ModalityEpisodePolicy:
    """Legality policy for target modalities in a sparse episode."""

    same_modality_context_required: bool = True
    unseen_target_modality_policy: Literal["reject"] = "reject"

    def __post_init__(self) -> None:
        if not isinstance(self.same_modality_context_required, bool):
            raise TypeError("same_modality_context_required must be bool")
        if self.unseen_target_modality_policy != "reject":
            raise ValueError("the T1-C reference rejects unseen target modalities")

    def to_dict(self) -> dict[str, object]:
        return {
            "same_modality_context_required": self.same_modality_context_required,
            "unseen_target_modality_policy": self.unseen_target_modality_policy,
        }


@dataclass(frozen=True)
class EpisodeSamplingConfig:
    context_count: int
    target_count: int = 1
    episode_count: int = 1
    seed: int = 0
    modality_policy: ModalityEpisodePolicy = ModalityEpisodePolicy()

    def __post_init__(self) -> None:
        if any(not isinstance(value, int) or value <= 0 for value in (self.context_count, self.target_count, self.episode_count)):
            raise ValueError("context_count, target_count, and episode_count must be positive integers")
        if self.target_count != 1:
            raise ValueError("the T1-C reference supports exactly one target per episode")
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(self.modality_policy, ModalityEpisodePolicy):
            raise TypeError("modality_policy must be a ModalityEpisodePolicy")

    def to_dict(self) -> dict[str, object]:
        return {
            "context_count": self.context_count,
            "episode_count": self.episode_count,
            "modality_policy": self.modality_policy.to_dict(),
            "seed": self.seed,
            "target_count": self.target_count,
        }


@dataclass(frozen=True)
class EpisodeSchedule:
    assignments: tuple[EpisodeAssignment, ...]
    schedule_hash: str


def _ranked(entries: tuple[AvailabilityObservationMeta, ...], *, seed_payload: bytes) -> tuple[AvailabilityObservationMeta, ...]:
    """Return a deterministic random order without using pixels or paths."""

    local_seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
    ordered = list(entries)
    random.Random(local_seed).shuffle(ordered)
    return tuple(ordered)


def _legal_assignment_entries(
    entries: tuple[AvailabilityObservationMeta, ...],
    *,
    config: EpisodeSamplingConfig,
    seed_payload: bytes,
) -> tuple[tuple[AvailabilityObservationMeta, ...], AvailabilityObservationMeta]:
    """Choose a target only after proving a compatible context set exists."""

    if len(entries) < config.context_count + config.target_count:
        raise EpisodeSamplingError(
            EpisodeSamplingFailureReason.INSUFFICIENT_OBSERVATIONS,
            "patient has too few legal observations for the requested episode",
        )
    ranked = _ranked(entries, seed_payload=seed_payload)
    saw_candidate_without_modality_context = False
    for target in ranked:
        remaining = tuple(entry for entry in ranked if entry.observation_id != target.observation_id)
        same_modality = tuple(entry for entry in remaining if entry.modality_id == target.modality_id)
        if config.modality_policy.same_modality_context_required and not same_modality:
            saw_candidate_without_modality_context = True
            continue
        context: list[AvailabilityObservationMeta] = []
        if config.modality_policy.same_modality_context_required:
            context.append(same_modality[0])
        context.extend(entry for entry in remaining if entry.observation_id not in {item.observation_id for item in context})
        if len(context) >= config.context_count:
            return tuple(context[: config.context_count]), target
    reason = (
        EpisodeSamplingFailureReason.MISSING_CONTEXT_MODALITY
        if saw_candidate_without_modality_context and config.modality_policy.same_modality_context_required
        else EpisodeSamplingFailureReason.NO_LEGAL_TARGET
    )
    raise EpisodeSamplingError(reason, "no target has a legal context assignment under the modality policy")


def build_episode_schedule(
    manifest: SparseAvailabilityManifest,
    *,
    patient_id: str,
    config: EpisodeSamplingConfig,
) -> EpisodeSchedule:
    """Build a deterministic legal schedule from manifest metadata, never pixels."""

    if not isinstance(manifest, SparseAvailabilityManifest) or not isinstance(config, EpisodeSamplingConfig):
        raise TypeError("manifest and config must use T1-C contracts")
    entries = tuple(sorted((entry for entry in manifest.entries if entry.patient_id == patient_id), key=lambda item: item.observation_id))
    if not entries:
        raise EpisodeSamplingError(
            EpisodeSamplingFailureReason.INVALID_MANIFEST_BINDING,
            "patient_id has no observations in the sealed sparse manifest",
        )
    assignments: list[EpisodeAssignment] = []
    for index in range(config.episode_count):
        payload = f"{manifest.manifest_hash}:{patient_id}:{config.seed}:{index}".encode("utf-8")
        context, target = _legal_assignment_entries(entries, config=config, seed_payload=payload)
        assignments.append(
            EpisodeAssignment.create(
                manifest,
                episode_id=f"{patient_id}-episode-{index:06d}",
                patient_id=patient_id,
                context_ids=tuple(item.observation_id for item in context),
                target_ids=(target.observation_id,),
            )
        )
    schedule_hash = hashlib.sha256(
        json.dumps(
            {
                "assignment_hashes": [item.assignment_hash for item in assignments],
                "config": config.to_dict(),
                "manifest_hash": manifest.manifest_hash,
                "patient_id": patient_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return EpisodeSchedule(tuple(assignments), schedule_hash)
