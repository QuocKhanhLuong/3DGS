"""Legal static T2/T3 episode bridge built on the hardened T1-C context path."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import torch

from ..anchors import AnchorBootstrapConfig, CachedPlaneEvidence, bootstrap_anchors
from ..baselines import RepresentationPlan, RepresentationVariant, resolve_representation_plan
from ..baselines.fixed_gaussian import FixedGaussianHead
from ..contracts.episode import EpisodeAssignment, EpisodeController, EpisodeLedger, FrozenPatientState
from ..features.encoder import EvidenceEncoder
from ..fields import GlobalStructuralField, SharedStructuralField, query_structural_field
from ..losses.reconstruction import ReconstructionLossResult, reconstruction_loss
from ..memory import PropagationConfig, PropagationTransaction, SeedMemoryConfig, propagate_memory
from ..reconstruction.plane import combined_memory_gaussians
from ..renderer import RenderResult
from ..state import PatientState, apply_memory_update, build_initial_patient_state
from .episode import ContextOnlyEpisodeStep, LegalEpisodeConfig, build_context_only_episode_step


@dataclass(frozen=True)
class StaticEpisodeResult:
    patient_state: PatientState
    context_step: ContextOnlyEpisodeStep
    prediction: RenderResult
    target: torch.Tensor
    target_valid_mask: torch.Tensor
    loss: ReconstructionLossResult
    propagation_transactions: tuple[PropagationTransaction, ...]
    receipt_hash: str
    audit_hash: str
    representation_plan: RepresentationPlan


def build_static_episode_step(
    *, ledger: EpisodeLedger, assignment: EpisodeAssignment, target_id: str,
    encoder: EvidenceEncoder, gaussian_head: FixedGaussianHead,
    config: LegalEpisodeConfig, patient_id: str, manifest_hash: str,
    patient_config_hash: str,
    field_model: SharedStructuralField | None = None,
    field_config_hash: str | None = None,
    global_field_model: GlobalStructuralField | None = None,
    representation_variant: str | RepresentationVariant = RepresentationVariant.ANCHOR_FIELD,
    field_maximum_neighbors: int = 8,
    registration_id: str = "manifest-canonical-ras-v1",
    bootstrap_config: AnchorBootstrapConfig | None = None,
    seed_memory_config: SeedMemoryConfig | None = None,
    propagation_config: PropagationConfig | None = None,
    patient_bounds_min_ras_mm: torch.Tensor | None = None,
    patient_bounds_max_ras_mm: torch.Tensor | None = None,
) -> StaticEpisodeResult:
    """Build all static state before target metadata/payload reveal.

    The target is touched only after the T0.5 commit/receipt barrier. The
    trainable field is supplied by the caller and never registered in state.
    """

    propagation_config = propagation_config or PropagationConfig(variant="p0")
    plan = resolve_representation_plan(
        representation_variant,
        propagation_variant=propagation_config.variant,
    )
    if plan.variant not in (
        RepresentationVariant.DIRECT_ANCHOR_GAUSSIAN,
        RepresentationVariant.ANCHOR_FIELD,
        RepresentationVariant.GLOBAL_FIELD,
    ):
        raise ValueError("the T2/T3 static state builder accepts only R3, R4, or R5")
    if not isinstance(field_maximum_neighbors, int) or field_maximum_neighbors <= 0:
        raise ValueError("field_maximum_neighbors must be a positive integer")
    if not isinstance(registration_id, str) or not registration_id:
        raise ValueError("static aggregation requires an explicit registration identity")
    if plan.variant is RepresentationVariant.ANCHOR_FIELD:
        if field_model is None or global_field_model is not None:
            raise ValueError("R4 requires exactly the shared local StructuralField")
    elif plan.variant is RepresentationVariant.GLOBAL_FIELD:
        if global_field_model is None or field_model is not None:
            raise ValueError("R5 requires exactly the global coordinate field")
    elif field_model is not None or global_field_model is not None:
        raise ValueError("R3 removes all structural-field modules")

    context_step = build_context_only_episode_step(
        ledger=ledger, assignment=assignment, encoder=encoder, gaussian_head=gaussian_head, config=config
    )
    cached = tuple(
        CachedPlaneEvidence(
            item.observation_id, item.modality_id, item.features, item.cache_key_hash,
            registration_id,
            normalized_image=item.normalized_image, valid_image_mask=item.valid_mask,
        )
        for item in context_step.context_evidence
    )
    modality_ids = tuple(sorted({item.modality_id for item in cached}))
    anchors = bootstrap_anchors(cached, patient_id=patient_id, modality_ids=modality_ids, config=bootstrap_config)
    field_values: torch.Tensor | None
    if plan.variant is RepresentationVariant.ANCHOR_FIELD:
        assert field_model is not None
        if field_config_hash != field_model.config.config_hash:
            raise ValueError("field_config_hash must bind the exact local-field config")
        model = field_model
        field_output = query_structural_field(
            model, anchors, anchors.centers_ras_mm,
            maximum_neighbors=field_maximum_neighbors,
        )
        if not bool(field_output.supported.all()):
            raise RuntimeError("seed anchors must be supported by their local StructuralField")
        field_values = field_output.value
        resolved_field_config_hash = model.config.config_hash
    elif plan.variant is RepresentationVariant.GLOBAL_FIELD:
        assert global_field_model is not None
        pooled = anchors.evidence.mean(dim=0)
        field_values = global_field_model(anchors.centers_ras_mm, pooled)
        resolved_field_config_hash = global_field_model.config.config_hash
        model = global_field_model
    else:
        field_values = None
        resolved_field_config_hash = hashlib.sha256(b"no-structural-field").hexdigest()
        model = None
    model_hash = hashlib.sha256(b"no-structural-field").hexdigest() if model is None else hashlib.sha256(
        b"".join(name.encode() + value.detach().cpu().contiguous().numpy().tobytes() for name, value in model.state_dict().items())
    ).hexdigest()
    state = build_initial_patient_state(
        patient_id=patient_id, manifest_hash=manifest_hash, config_hash=patient_config_hash,
        context_observation_ids=context_step.context_ids,
        cache_key_hashes=context_step.feature_cache_key_hashes,
        anchors=anchors, field_config_hash=resolved_field_config_hash, field_model_hash=model_hash,
        memory_config=seed_memory_config, field_values=field_values,
    )
    device = state.memory.structural.gaussians.centers_ras_mm.device
    lower = patient_bounds_min_ras_mm if patient_bounds_min_ras_mm is not None else state.anchors.centers_ras_mm.min(dim=0).values - state.anchors.support_scales_mm.max(dim=0).values
    upper = patient_bounds_max_ras_mm if patient_bounds_max_ras_mm is not None else state.anchors.centers_ras_mm.max(dim=0).values + state.anchors.support_scales_mm.max(dim=0).values
    propagated_memory, transactions = propagate_memory(
        state.memory, state.anchors, config=propagation_config,
        bounds_min_ras_mm=lower.to(device=device), bounds_max_ras_mm=upper.to(device=device),
    )
    if propagated_memory.memory_hash != state.memory.memory_hash:
        state = apply_memory_update(state, propagated_memory)
    target_metadata = ledger.metadata(target_id)
    target_modality = target_metadata.modality_id
    if target_modality not in modality_ids:
        raise ValueError("static reference requires a legal context modality for the target")
    frozen = FrozenPatientState.create(ledger=ledger, gaussians=combined_memory_gaussians(state), upstream_state_hash=state.state_version)
    ledger.expose_target_metadata(target_id)
    commit = ledger.commit_target(target_id, frozen.state_version)
    prediction, receipt = EpisodeController().render_and_register(
        ledger=ledger, commit_capability=commit, frozen_state=frozen,
        appearance_channel=modality_ids.index(target_modality), render_config=config.renderer,
    )
    target_payload = ledger.reveal_target(target_id, receipt)
    from ..data.io import decode_observation
    from ..data.normalization import apply_preprocessing
    target_decoded = decode_observation(target_payload, ledger.metadata(target_id), config=config.decoder)
    target_normalized = apply_preprocessing(context_step.preprocessing, target_decoded)
    target = target_normalized.image[0].to(device=device, dtype=prediction.intensity.dtype)
    target_valid = target_normalized.valid_mask[0].to(device=device)
    loss = reconstruction_loss(prediction, target, target_valid, config=config.reconstruction_loss)
    receipt_payload = json.dumps(
        ledger.prediction_records[-1].to_canonical_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )
    receipt_hash = hashlib.sha256(receipt_payload.encode("utf-8")).hexdigest()
    return StaticEpisodeResult(
        patient_state=state, context_step=context_step, prediction=prediction, target=target,
        target_valid_mask=target_valid, loss=loss, propagation_transactions=transactions,
        receipt_hash=receipt_hash, audit_hash=ledger.audit_hash, representation_plan=plan,
    )
