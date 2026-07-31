"""Matched E0/E1/E2 assignment ownership for T1-C attribution."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts.observation import SparseAvailabilityManifest
from ..data.episodes import EpisodeSamplingConfig, EpisodeSchedule, build_episode_schedule


@dataclass(frozen=True)
class MatchedVariantSchedule:
    variants: tuple[str, ...]
    episode_schedule: EpisodeSchedule

    def __post_init__(self) -> None:
        if self.variants != ("e0", "e1", "e2"):
            raise ValueError("the locked T1-C matched schedule must contain e0, e1, and e2")

    def for_variant(self, variant: str) -> EpisodeSchedule:
        if variant not in self.variants:
            raise KeyError(f"unknown matched variant: {variant}")
        return self.episode_schedule


def build_matched_variant_schedule(
    manifest: SparseAvailabilityManifest,
    *,
    patient_id: str,
    config: EpisodeSamplingConfig,
) -> MatchedVariantSchedule:
    """Return one immutable assignment schedule shared by all T1 variants."""

    return MatchedVariantSchedule(
        variants=("e0", "e1", "e2"),
        episode_schedule=build_episode_schedule(manifest, patient_id=patient_id, config=config),
    )
