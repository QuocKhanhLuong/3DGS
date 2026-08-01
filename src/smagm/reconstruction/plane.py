"""Modality-aware arbitrary physical-plane reconstruction from frozen state."""

from __future__ import annotations

import hashlib

import torch

from ..contracts.coordinates import PhysicalPlane
from ..contracts.outputs import PlaneReconstruction, plane_output_hash
from ..gaussians import AmplitudeGaugePolicy, restore_gauge_fixed_gaussian_batch
from ..renderer import RenderConfig, render_plane
from ..state import PatientState
from .uncertainty import support_uncertainty


def combined_memory_gaussians(state: PatientState):
    """Combine dual memory banks without reapplying the amplitude gauge."""

    banks = (state.memory.structural.gaussians, state.memory.volumetric.gaussians)
    reference = banks[0]
    if any(bank.gauge_policy is AmplitudeGaugePolicy.LEGACY_RAW for bank in banks):
        raise ValueError("patient memory requires explicit Phase-1 amplitude-gauge provenance")
    if any(bank.gauge_policy != reference.gauge_policy or bank.gauge_config_hash != reference.gauge_config_hash for bank in banks[1:]):
        raise ValueError("patient memory banks must share one amplitude-gauge identity")
    if any(bank.covariance_epsilon != reference.covariance_epsilon for bank in banks[1:]):
        raise ValueError("patient memory banks must share covariance_epsilon")
    return restore_gauge_fixed_gaussian_batch(
        centers_ras_mm=torch.cat([bank.centers_ras_mm for bank in banks]),
        covariance_factor=torch.cat([bank.covariance_factor for bank in banks]),
        log_support_amplitude=torch.cat([bank.log_support_amplitude for bank in banks]),
        appearance=torch.cat([bank.appearance for bank in banks]),
        appearance_valid=torch.cat([bank.appearance_valid for bank in banks]),
        covariance_epsilon=reference.covariance_epsilon,
        primitive_kind=tuple(v for bank in banks for v in (bank.primitive_kind or ())),
        primitive_id=tuple(v for bank in banks for v in (bank.primitive_id or ())),
        gauge_policy=reference.gauge_policy,
        gauge_config_hash=reference.gauge_config_hash or "",
    )


def reconstruct_plane(
    state: PatientState, plane: PhysicalPlane, *, modality_id: str, render_config: RenderConfig | None = None,
) -> PlaneReconstruction:
    if modality_id not in state.memory.modality_ids:
        raise KeyError("requested modality is absent from patient-state mapping")
    render_config = render_config or RenderConfig()
    channel = state.memory.modality_ids.index(modality_id)
    gaussians = combined_memory_gaussians(state)
    rendered = render_plane(gaussians, plane, appearance_channel=channel, config=render_config)
    propagation_uncertainty = float(torch.cat((
        state.memory.structural.observability.uncertainty,
        state.memory.volumetric.observability.uncertainty,
    )).mean().detach())
    renderer_config_hash = hashlib.sha256(render_config.renderer_version.encode()).hexdigest()
    uncertainty = support_uncertainty(rendered.support_mass, rendered.unsupported_mask, propagation_uncertainty=propagation_uncertainty)
    artifact_hash = plane_output_hash(
        patient_id=state.patient_id, modality_id=modality_id, plane=plane,
        intensity=rendered.intensity, support_mass=rendered.support_mass,
        unsupported_mask=rendered.unsupported_mask, support_uncertainty=uncertainty,
        renderer_version=render_config.renderer_version, renderer_config_hash=renderer_config_hash,
        patient_state_version=state.state_version,
    )
    return PlaneReconstruction(
        patient_id=state.patient_id, modality_id=modality_id, plane=plane,
        intensity=rendered.intensity, support_mass=rendered.support_mass,
        unsupported_mask=rendered.unsupported_mask,
        support_uncertainty=uncertainty, renderer_version=render_config.renderer_version,
        renderer_config_hash=renderer_config_hash, patient_state_version=state.state_version,
        artifact_hash=artifact_hash,
    )
