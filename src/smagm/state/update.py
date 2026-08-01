"""Immutable accepted patient-state update transactions."""

from __future__ import annotations

from ..memory import GaussianMemory
from .patient import PatientState, patient_state_version


def apply_memory_update(state: PatientState, memory: GaussianMemory) -> PatientState:
    if memory.memory_hash == state.memory.memory_hash:
        return state
    update_round = state.update_round + 1
    payload = dict(
        patient_id=state.patient_id, manifest_hash=state.manifest_hash, config_hash=state.config_hash,
        context_observation_ids=state.context_observation_ids, cache_key_hashes=state.cache_key_hashes,
        anchor_evidence_hash=state.anchors.evidence_hash, memory_hash=memory.memory_hash,
        field_config_hash=state.field_config_hash, field_model_hash=state.field_model_hash,
        update_round=update_round, parent_state_version=state.state_version,
    )
    return PatientState(
        patient_id=state.patient_id, manifest_hash=state.manifest_hash, config_hash=state.config_hash,
        context_observation_ids=state.context_observation_ids, cache_key_hashes=state.cache_key_hashes,
        anchors=state.anchors, memory=memory, field_config_hash=state.field_config_hash,
        field_model_hash=state.field_model_hash, update_round=update_round,
        parent_state_version=state.state_version, state_version=patient_state_version(**payload),
    )
