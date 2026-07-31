"""Deterministic, metadata-only episode assignment schedules."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random

from ..contracts.episode import EpisodeAssignment
from ..contracts.observation import SparseAvailabilityManifest


@dataclass(frozen=True)
class EpisodeSamplingConfig:
    context_count: int
    target_count: int
    episode_count: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if any(not isinstance(value, int) or value <= 0 for value in (self.context_count, self.target_count, self.episode_count)):
            raise ValueError("context_count, target_count, and episode_count must be positive integers")
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")


@dataclass(frozen=True)
class EpisodeSchedule:
    assignments: tuple[EpisodeAssignment, ...]
    schedule_hash: str


def build_episode_schedule(
    manifest: SparseAvailabilityManifest,
    *,
    patient_id: str,
    config: EpisodeSamplingConfig,
) -> EpisodeSchedule:
    """Build a deterministic schedule from IDs and metadata, never pixels."""

    if not isinstance(manifest, SparseAvailabilityManifest) or not isinstance(config, EpisodeSamplingConfig):
        raise TypeError("manifest and config must use T1-C contracts")
    identifiers = sorted(entry.observation_id for entry in manifest.entries if entry.patient_id == patient_id)
    required = config.context_count + config.target_count
    if len(identifiers) < required:
        raise ValueError("patient has too few legal observations for the requested episode")
    assignments: list[EpisodeAssignment] = []
    for index in range(config.episode_count):
        seed_payload = f"{manifest.manifest_hash}:{patient_id}:{config.seed}:{index}".encode("utf-8")
        local_seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
        shuffled = list(identifiers)
        random.Random(local_seed).shuffle(shuffled)
        assignments.append(
            EpisodeAssignment.create(
                manifest,
                episode_id=f"{patient_id}-episode-{index:06d}",
                patient_id=patient_id,
                context_ids=shuffled[: config.context_count],
                target_ids=shuffled[config.context_count : required],
            )
        )
    payload = {
        "assignment_hashes": [item.assignment_hash for item in assignments],
        "config": config.__dict__,
        "manifest_hash": manifest.manifest_hash,
        "patient_id": patient_id,
    }
    schedule_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return EpisodeSchedule(tuple(assignments), schedule_hash)
