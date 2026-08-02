"""Legal static T2/T3 episode bridge built on the hardened T1-C context path."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import torch

from ..anchors import AnchorBootstrapConfig, CachedPlaneEvidence, bootstrap_anchors
from ..baselines import RepresentationPlan, RepresentationVariant, resolve_representation_plan
from ..baselines.fixed_gaussian import FixedGaussianHead, construct_fixed_gaussians
from ..baselines.fixed_support import FixedSupportBatch
from ..contracts.episode import EpisodeAssignment, EpisodeController, EpisodeLedger, FrozenPatientState
from ..features.encoder import EvidenceEncoder
from ..fields import GlobalStructuralField, SharedStructuralField, query_structural_field
from ..gaussians import restore_gauge_fixed_gaussian_batch
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


def _static_modality_order(
    config: LegalEpisodeConfig,
    gaussian_head: FixedGaussianHead,
    context_modalities: set[str],
) -> tuple[str, ...]:
    """Return the exact memory-channel order declared by the episode mapping.

    Static memory stores one compact appearance column per Gaussian-head output
    channel.  Therefore the mapping must be a bijection onto ``0..M-1`` rather
    than being reconstructed from sorted modality names.
    """

    mapping = dict(config.modality_to_appearance_channel or {})
    ordered = tuple(name for name, _ in sorted(mapping.items(), key=lambda item: (item[1], item[0])))
    channels = tuple(mapping[name] for name in ordered)
    expected = tuple(range(gaussian_head.config.appearance_channels))
    if channels != expected:
        raise ValueError(
            "static appearance mapping must cover every Gaussian-head channel exactly once and contiguously"
        )
    if not context_modalities or not context_modalities.issubset(mapping):
        raise ValueError("every static context modality must have an explicit appearance-channel mapping")
    for modality_id in context_modalities:
        config.appearance_channel_for(
            modality_id,
            available_channels=gaussian_head.config.appearance_channels,
        )
    return ordered


def _head_volumetric_gaussians(
    anchors: "AnchorBatch", gaussian_head: FixedGaussianHead, *, input_adapter: str,
) -> "GaussianBatch":
    """Construct the R4 volumetric bank from context-only anchor evidence.

    Anchor selection and physical consolidation are discrete, but the compact
    evidence at the selected anchors remains connected to the encoder graph.
    The explicit prefix adapter is part of the product contract because the
    shared T1 head currently declares a 25-channel input while R4 aggregation
    exposes a 52-channel mean/dispersion evidence vector.
    """

    from ..anchors import AnchorBatch
    from ..gaussians import GaussianBatch

    if input_adapter != "anchor_evidence_prefix":
        raise ValueError("the maintained static path supports only anchor_evidence_prefix Gaussian-head input")
    if not isinstance(anchors, AnchorBatch):
        raise TypeError("anchors must be an AnchorBatch")
    input_dim = gaussian_head.config.input_dim
    if anchors.evidence.shape[1] < input_dim:
        raise ValueError(
            f"Gaussian-head input adapter requires {input_dim} anchor evidence channels; "
            f"only {anchors.evidence.shape[1]} are available"
        )
    if gaussian_head.config.appearance_channels != anchors.appearance.shape[1]:
        raise ValueError("Gaussian-head appearance channels must match anchor modality channels")
    observation_ids: list[str] = []
    plane_hashes: list[str] = []
    for anchor_id, observations, planes in zip(
        anchors.anchor_ids,
        anchors.geometry.contributing_observation_ids,
        anchors.geometry.contributing_plane_hashes,
    ):
        if not observations or not planes:
            raise ValueError(f"anchor {anchor_id} has incomplete legal provenance for Gaussian-head construction")
        observation_ids.append(observations[0])
        plane_hashes.append(planes[0])
    supports = FixedSupportBatch(
        centers_ras_mm=anchors.centers_ras_mm,
        feature_vectors=anchors.evidence[:, :input_dim],
        feature_indices_vu=torch.zeros((anchors.count, 2), dtype=torch.long, device=anchors.evidence.device),
        reliability=anchors.geometry.geometry_confidence.clamp(0.0, 1.0),
        observation_ids=tuple(observation_ids),
        source_plane_hashes=tuple(plane_hashes),
        batch_index=0,
        # FixedSupportBatch uses row basis (t1, t2, n), matching the physical
        # local-frame offset convention used by the Gaussian head.
        support_basis_ras=anchors.frame_axes_ras,
    )
    predicted = construct_fixed_gaussians(
        supports,
        gaussian_head(supports.feature_vectors),
        config=gaussian_head.config,
    )
    primitive_ids = tuple(f"volumetric:{anchor_id}" for anchor_id in anchors.anchor_ids)
    return restore_gauge_fixed_gaussian_batch(
        centers_ras_mm=predicted.centers_ras_mm,
        covariance_factor=predicted.covariance_factor,
        log_support_amplitude=predicted.log_support_amplitude,
        appearance=predicted.appearance,
        # Preserve the modality-validity mask from legal context evidence;
        # the head may emit a value for every configured channel, but an
        # unseen modality is not silently promoted to observed appearance.
        appearance_valid=anchors.appearance_valid,
        # The fixed-head converter keeps its Cholesky epsilon in the factor
        # and returns a tiny residual epsilon.  Dual-bank rendering requires
        # the same runtime residual policy as the anchor-created structural
        # bank, whose GaussianBatch default is 1e-8.
        covariance_epsilon=1e-8,
        primitive_kind=("VOLUMETRIC",) * anchors.count,
        primitive_id=primitive_ids,
        gauge_policy=predicted.gauge_policy,
        gauge_config_hash=predicted.gauge_config_hash or "",
    )


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
    source_affine_ras_from_index: torch.Tensor | None = None,
    source_shape_xyz: tuple[int, int, int] | None = None,
    gaussian_head_input_adapter: str = "anchor_evidence_prefix",
) -> StaticEpisodeResult:
    """Build all static state before target metadata/payload reveal.

    The target is touched only after the T0.5 commit/receipt barrier. The
    trainable field is supplied by the caller and never registered in state.
    """

    if target_id not in assignment.target_ids:
        raise PermissionError("target_id must be assigned as a target")
    if len(assignment.target_ids) != 1:
        raise ValueError("the static reference supports exactly one target per episode")
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
    context_modalities = {item.modality_id for item in cached}
    modality_ids = _static_modality_order(config, gaussian_head, context_modalities)
    anchors = bootstrap_anchors(cached, patient_id=patient_id, modality_ids=modality_ids, config=bootstrap_config)
    field_values: torch.Tensor | None
    supported_anchor_mask: torch.Tensor | None = None
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
        supported_anchor_mask = field_output.supported
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
    volumetric_gaussians = _head_volumetric_gaussians(
        anchors, gaussian_head, input_adapter=gaussian_head_input_adapter,
    )
    model_hash = hashlib.sha256(b"no-structural-field").hexdigest() if model is None else hashlib.sha256(
        b"".join(name.encode() + value.detach().cpu().contiguous().numpy().tobytes() for name, value in model.state_dict().items())
    ).hexdigest()
    state = build_initial_patient_state(
        patient_id=patient_id, manifest_hash=manifest_hash, config_hash=patient_config_hash,
        context_observation_ids=context_step.context_ids,
        cache_key_hashes=context_step.feature_cache_key_hashes,
        anchors=anchors, field_config_hash=resolved_field_config_hash, field_model_hash=model_hash,
        memory_config=seed_memory_config, field_values=field_values,
        volumetric_gaussians=volumetric_gaussians,
    )
    device = state.memory.structural.gaussians.centers_ras_mm.device
    lower = patient_bounds_min_ras_mm if patient_bounds_min_ras_mm is not None else state.anchors.centers_ras_mm.min(dim=0).values - state.anchors.support_scales_mm.max(dim=0).values
    upper = patient_bounds_max_ras_mm if patient_bounds_max_ras_mm is not None else state.anchors.centers_ras_mm.max(dim=0).values + state.anchors.support_scales_mm.max(dim=0).values
    propagated_memory, transactions = propagate_memory(
        state.memory, state.anchors, config=propagation_config,
        bounds_min_ras_mm=lower.to(device=device), bounds_max_ras_mm=upper.to(device=device),
        supported_anchor_mask=supported_anchor_mask,
        source_affine_ras_from_index=source_affine_ras_from_index,
        source_shape_xyz=source_shape_xyz,
    )
    if propagated_memory.memory_hash != state.memory.memory_hash:
        state = apply_memory_update(state, propagated_memory)
    target_metadata = ledger.metadata(target_id)
    target_modality = target_metadata.modality_id
    if target_modality not in context_modalities:
        raise ValueError("static reference requires a legal context observation for the target modality")
    target_appearance_channel = config.appearance_channel_for(
        target_modality,
        available_channels=state.memory.structural.gaussians.appearance_channels,
    )
    frozen = FrozenPatientState.create(ledger=ledger, gaussians=combined_memory_gaussians(state), upstream_state_hash=state.state_version)
    ledger.expose_target_metadata(target_id)
    commit = ledger.commit_target(target_id, frozen.state_version)
    prediction, receipt = EpisodeController().render_and_register(
        ledger=ledger, commit_capability=commit, frozen_state=frozen,
        appearance_channel=target_appearance_channel, render_config=config.renderer,
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
