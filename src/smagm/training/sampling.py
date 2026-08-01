"""Matched E0/E1/E2 assignment ownership for T1-C attribution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

from ..contracts.observation import SparseAvailabilityManifest
from ..data.episodes import EpisodeSamplingConfig, EpisodeSchedule, build_episode_schedule


def _canonicalize_and_freeze(value: object) -> object:
    """Make the identity payload recursively immutable after canonical JSON validation."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _canonicalize_and_freeze(item) for key, item in sorted(value.items())})
    if isinstance(value, (tuple, list)):
        return tuple(_canonicalize_and_freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"matched experiment conditions must be canonical JSON values, not {type(value).__name__}")


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


@dataclass(frozen=True)
class MatchedExperimentIdentity:
    """Hash of every T1-C condition shared by E0, E1, and E2."""

    manifest_hash: str
    split_registry_hash: str
    assignment_schedule_hash: str
    modality_mapping_hash: str
    shared_conditions: Mapping[str, object]
    identity_hash: str

    def __post_init__(self) -> None:
        for name in ("manifest_hash", "split_registry_hash", "assignment_schedule_hash", "modality_mapping_hash", "identity_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a SHA-256 digest")
        frozen = _canonicalize_and_freeze(self.shared_conditions)
        if not isinstance(frozen, MappingProxyType):
            raise TypeError("shared_conditions must be a mapping")
        object.__setattr__(self, "shared_conditions", frozen)

    @classmethod
    def from_resolved_conditions(
        cls,
        *,
        manifest_hash: str,
        split_registry_hash: str,
        assignment_schedule_hash: str,
        modality_mapping_hash: str,
        shared_conditions: Mapping[str, object],
    ) -> "MatchedExperimentIdentity":
        payload = {
            "assignment_schedule_hash": assignment_schedule_hash,
            "manifest_hash": manifest_hash,
            "modality_mapping_hash": modality_mapping_hash,
            "shared_conditions": json.loads(json.dumps(shared_conditions, sort_keys=True, separators=(",", ":"))),
            "split_registry_hash": split_registry_hash,
        }
        identity_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return cls(
            manifest_hash=manifest_hash,
            split_registry_hash=split_registry_hash,
            assignment_schedule_hash=assignment_schedule_hash,
            modality_mapping_hash=modality_mapping_hash,
            shared_conditions=shared_conditions,
            identity_hash=identity_hash,
        )


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
