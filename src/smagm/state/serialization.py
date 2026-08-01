"""Safe tensor-only patient-state schema and exact round-trip helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..anchors import AnchorBatch, AnchorGeometryBatch
from ..gaussians import AmplitudeGaugePolicy, restore_gauge_fixed_gaussian_batch
from ..memory import GaussianMemory, GaussianMemoryBank, PrimitiveKind, PrimitiveObservability
from .patient import PatientState


STATE_SCHEMA = "smagm-static-patient-state-v1"


def _frozen(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().contiguous()


def _observability_payload(value: PrimitiveObservability) -> dict[str, torch.Tensor]:
    return {name: _frozen(getattr(value, name)) for name in value.__dataclass_fields__}


def _bank_payload(bank: GaussianMemoryBank) -> dict[str, Any]:
    gaussian = bank.gaussians
    return {
        "kind": bank.kind.value, "anchor_ids": bank.anchor_ids,
        "parent_primitive_ids": bank.parent_primitive_ids, "provenance_hashes": bank.provenance_hashes,
        "observability": _observability_payload(bank.observability),
        "gaussian": {
            "centers_ras_mm": _frozen(gaussian.centers_ras_mm),
            "covariance_factor": _frozen(gaussian.covariance_factor),
            "log_support_amplitude": _frozen(gaussian.log_support_amplitude),
            "appearance": _frozen(gaussian.appearance),
            "appearance_valid": _frozen(gaussian.appearance_valid),
            "covariance_epsilon": gaussian.covariance_epsilon,
            "primitive_kind": gaussian.primitive_kind, "primitive_id": gaussian.primitive_id,
            "gauge_policy": gaussian.gauge_policy.value, "gauge_config_hash": gaussian.gauge_config_hash,
        },
    }


def patient_state_payload(state: PatientState) -> dict[str, Any]:
    geometry = state.anchors.geometry
    return {
        "schema": STATE_SCHEMA,
        "patient": {
            "patient_id": state.patient_id, "manifest_hash": state.manifest_hash,
            "config_hash": state.config_hash, "context_observation_ids": state.context_observation_ids,
            "cache_key_hashes": state.cache_key_hashes, "field_config_hash": state.field_config_hash,
            "field_model_hash": state.field_model_hash, "update_round": state.update_round,
            "parent_state_version": state.parent_state_version, "state_version": state.state_version,
        },
        "anchors": {
            "patient_id": state.anchors.patient_id, "anchor_ids": geometry.anchor_ids,
            "centers_ras_mm": _frozen(geometry.centers_ras_mm), "frame_axes_ras": _frozen(geometry.frame_axes_ras),
            "frame_validity": _frozen(geometry.frame_validity), "support_scales_mm": _frozen(geometry.support_scales_mm),
            "geometry_confidence": _frozen(geometry.geometry_confidence), "disagreement": _frozen(geometry.disagreement),
            "contributing_observation_ids": geometry.contributing_observation_ids,
            "contributing_plane_hashes": geometry.contributing_plane_hashes,
            "provenance_hashes": geometry.provenance_hashes,
            "evidence": _frozen(state.anchors.evidence), "appearance": _frozen(state.anchors.appearance),
            "appearance_valid": _frozen(state.anchors.appearance_valid), "observability": _frozen(state.anchors.observability),
            "modality_ids": state.anchors.modality_ids, "evidence_hash": state.anchors.evidence_hash,
        },
        "memory": {
            "structural": _bank_payload(state.memory.structural),
            "volumetric": _bank_payload(state.memory.volumetric),
            "modality_ids": state.memory.modality_ids, "memory_hash": state.memory.memory_hash,
        },
    }


def _restore_bank(payload: dict[str, Any]) -> GaussianMemoryBank:
    gaussian = payload["gaussian"]
    batch = restore_gauge_fixed_gaussian_batch(
        centers_ras_mm=gaussian["centers_ras_mm"], covariance_factor=gaussian["covariance_factor"],
        log_support_amplitude=gaussian["log_support_amplitude"], appearance=gaussian["appearance"],
        appearance_valid=gaussian["appearance_valid"], covariance_epsilon=float(gaussian["covariance_epsilon"]),
        primitive_kind=tuple(gaussian["primitive_kind"]), primitive_id=tuple(gaussian["primitive_id"]),
        gauge_policy=AmplitudeGaugePolicy(gaussian["gauge_policy"]), gauge_config_hash=gaussian["gauge_config_hash"],
    )
    observability = PrimitiveObservability(**payload["observability"])
    return GaussianMemoryBank(
        PrimitiveKind(payload["kind"]), batch, tuple(payload["anchor_ids"]),
        tuple(payload["parent_primitive_ids"]), tuple(payload["provenance_hashes"]), observability,
    )


def patient_state_from_payload(payload: dict[str, Any]) -> PatientState:
    if not isinstance(payload, dict) or payload.get("schema") != STATE_SCHEMA:
        raise ValueError("patient-state payload has an unsupported schema")
    raw_anchor = payload["anchors"]
    geometry = AnchorGeometryBatch(
        tuple(raw_anchor["anchor_ids"]), raw_anchor["centers_ras_mm"], raw_anchor["frame_axes_ras"],
        raw_anchor["frame_validity"], raw_anchor["support_scales_mm"], raw_anchor["geometry_confidence"],
        raw_anchor["disagreement"], tuple(tuple(v) for v in raw_anchor["contributing_observation_ids"]),
        tuple(tuple(v) for v in raw_anchor["contributing_plane_hashes"]), tuple(raw_anchor["provenance_hashes"]),
    )
    anchors = AnchorBatch(
        raw_anchor["patient_id"], geometry, raw_anchor["evidence"], raw_anchor["appearance"],
        raw_anchor["appearance_valid"], raw_anchor["observability"], tuple(raw_anchor["modality_ids"]),
        raw_anchor["evidence_hash"],
    )
    raw_memory = payload["memory"]
    memory = GaussianMemory(
        _restore_bank(raw_memory["structural"]), _restore_bank(raw_memory["volumetric"]),
        tuple(raw_memory["modality_ids"]), raw_memory["memory_hash"],
    )
    patient = payload["patient"]
    return PatientState(
        patient_id=patient["patient_id"], manifest_hash=patient["manifest_hash"], config_hash=patient["config_hash"],
        context_observation_ids=tuple(patient["context_observation_ids"]), cache_key_hashes=tuple(patient["cache_key_hashes"]),
        anchors=anchors, memory=memory, field_config_hash=patient["field_config_hash"], field_model_hash=patient["field_model_hash"],
        update_round=int(patient["update_round"]), parent_state_version=patient["parent_state_version"], state_version=patient["state_version"],
    )


def save_patient_state(state: PatientState, path: str | Path) -> Path:
    destination = Path(path)
    if not destination.parent.exists():
        raise FileNotFoundError("patient-state parent directory does not exist")
    if destination.exists():
        raise FileExistsError("immutable patient-state output already exists")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        torch.save(patient_state_payload(state), temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_patient_state(path: str | Path) -> PatientState:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return patient_state_from_payload(payload)
