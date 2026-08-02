"""Build the immutable T2 initial state from legal context anchors."""

from __future__ import annotations

import torch

from ..anchors import AnchorBatch
from ..gaussians import GaussianBatch
from ..memory import SeedMemoryConfig, initialize_seed_memory
from .patient import PatientState, patient_state_version


def build_initial_patient_state(
    *, patient_id: str, manifest_hash: str, config_hash: str,
    context_observation_ids: tuple[str, ...], cache_key_hashes: tuple[str, ...],
    anchors: AnchorBatch, field_config_hash: str, field_model_hash: str,
    memory_config: SeedMemoryConfig | None = None,
    field_values: torch.Tensor | None = None,
    volumetric_gaussians: GaussianBatch | None = None,
) -> PatientState:
    memory = initialize_seed_memory(
        anchors,
        config=memory_config,
        field_values=field_values,
        volumetric_gaussians=volumetric_gaussians,
    )
    payload = dict(
        patient_id=patient_id, manifest_hash=manifest_hash, config_hash=config_hash,
        context_observation_ids=context_observation_ids, cache_key_hashes=cache_key_hashes,
        anchor_evidence_hash=anchors.evidence_hash, memory_hash=memory.memory_hash,
        field_config_hash=field_config_hash, field_model_hash=field_model_hash,
        update_round=0, parent_state_version=None,
    )
    return PatientState(
        patient_id=patient_id, manifest_hash=manifest_hash, config_hash=config_hash,
        context_observation_ids=context_observation_ids, cache_key_hashes=cache_key_hashes,
        anchors=anchors, memory=memory, field_config_hash=field_config_hash,
        field_model_hash=field_model_hash, update_round=0, parent_state_version=None,
        state_version=patient_state_version(**payload),
    )
