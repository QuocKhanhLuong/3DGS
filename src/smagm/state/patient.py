"""Immutable patient-specific state distinct from global trainable parameters."""

from __future__ import annotations

from dataclasses import dataclass
import re

from ..anchors import AnchorBatch
from ..memory import GaussianMemory
from .versioning import state_version_hash


@dataclass(frozen=True)
class PatientState:
    patient_id: str
    manifest_hash: str
    config_hash: str
    context_observation_ids: tuple[str, ...]
    cache_key_hashes: tuple[str, ...]
    anchors: AnchorBatch
    memory: GaussianMemory
    field_config_hash: str
    field_model_hash: str
    update_round: int
    parent_state_version: str | None
    state_version: str

    def __post_init__(self) -> None:
        if not self.patient_id or self.anchors.patient_id != self.patient_id or not self.context_observation_ids:
            raise ValueError("patient state requires matching patient and non-empty context identity")
        if self.update_round < 0 or len(self.cache_key_hashes) != len(self.context_observation_ids):
            raise ValueError("patient state update/cache metadata is invalid")
        for value in (self.manifest_hash, self.config_hash, *self.cache_key_hashes, self.field_config_hash, self.field_model_hash):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("patient-state identities must be SHA-256 digests")
        expected = patient_state_version(
            patient_id=self.patient_id, manifest_hash=self.manifest_hash, config_hash=self.config_hash,
            context_observation_ids=self.context_observation_ids, cache_key_hashes=self.cache_key_hashes,
            anchor_evidence_hash=self.anchors.evidence_hash, memory_hash=self.memory.memory_hash,
            field_config_hash=self.field_config_hash, field_model_hash=self.field_model_hash,
            update_round=self.update_round, parent_state_version=self.parent_state_version,
        )
        if self.state_version != expected:
            raise ValueError("state_version does not bind the exact patient state")


def patient_state_version(**payload: object) -> str:
    return state_version_hash(dict(payload))
