"""One fail-closed legal episode dispatcher for R0--R5 and P0/P1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import torch

from ..anchors import AnchorBootstrapConfig
from ..baselines import (
    FreeGaussianState,
    RepresentationPlan,
    RepresentationVariant,
    SparseInterpolationConfig,
    construct_sparse_interpolation_gaussians,
    resolve_representation_plan,
)
from ..baselines.fixed_gaussian import FixedGaussianHead
from ..contracts.coordinates import PhysicalPlane
from ..contracts.episode import EpisodeAssignment, EpisodeController, EpisodeLedger, FrozenPatientState
from ..data.io import decode_observation
from ..data.normalization import apply_preprocessing, fit_preprocessing
from ..features.encoder import EvidenceEncoder
from ..fields import GlobalStructuralField, SharedStructuralField
from ..losses.reconstruction import ReconstructionLossResult, reconstruction_loss
from ..memory import PropagationConfig, SeedMemoryConfig
from ..renderer import RenderResult
from ..state import PatientState
from .episode import LegalEpisodeConfig, build_legal_episode_step
from .static import build_static_episode_step


def _hash(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ContextImageEvidence:
    """Preprocessed context payload without an encoder feature allocation."""

    observation_id: str
    modality_id: str
    plane: PhysicalPlane
    normalized_image: torch.Tensor  # [1,1,H,W]
    valid_mask: torch.Tensor  # [1,1,H,W], bool

    def __post_init__(self) -> None:
        if self.normalized_image.ndim != 4 or self.normalized_image.shape[:2] != (1, 1):
            raise ValueError("context image evidence must have shape [1,1,H,W]")
        if not isinstance(self.plane, PhysicalPlane) or tuple(self.plane.shape_hw) != tuple(self.normalized_image.shape[-2:]):
            raise ValueError("context image evidence plane must match image geometry")
        if self.valid_mask.shape != self.normalized_image.shape or self.valid_mask.dtype is not torch.bool:
            raise ValueError("context image valid mask must be bool and match image")


@dataclass(frozen=True)
class RepresentationEpisodeResult:
    representation_plan: RepresentationPlan
    state_version: str
    patient_state: PatientState | None
    free_gaussian_state: FreeGaussianState | None
    prediction: RenderResult
    target: torch.Tensor
    target_valid_mask: torch.Tensor
    loss: ReconstructionLossResult
    context_ids: tuple[str, ...]
    receipt_hash: str
    audit_hash: str


def _direct_context_baseline(
    *,
    ledger: EpisodeLedger,
    assignment: EpisodeAssignment,
    target_id: str,
    config: LegalEpisodeConfig,
    plan: RepresentationPlan,
    interpolation_config: SparseInterpolationConfig | None,
) -> RepresentationEpisodeResult:
    decoded_context = []
    for observation_id in assignment.context_ids:
        payload = ledger.open_context(observation_id)
        decoded_context.append(decode_observation(payload, ledger.metadata(observation_id), config=config.decoder))
    preprocessing = fit_preprocessing(decoded_context, context_ids=assignment.context_ids, config=config.normalization)
    evidence = []
    for decoded in decoded_context:
        normalized = apply_preprocessing(preprocessing, decoded)
        evidence.append(ContextImageEvidence(
            decoded.observation_id,
            decoded.modality_id,
            decoded.metadata.plane,
            normalized.image.unsqueeze(0).to(dtype=torch.float32),
            normalized.valid_mask.unsqueeze(0),
        ))
    modality_ids = tuple(sorted({item.modality_id for item in evidence}))
    target_metadata = ledger.metadata(target_id)
    if target_metadata.modality_id not in modality_ids:
        raise ValueError("baseline target modality requires legal same-modality context")
    seed = construct_sparse_interpolation_gaussians(
        evidence,
        modality_ids=modality_ids,
        config=interpolation_config,
    )
    free_state: FreeGaussianState | None = None
    if plan.variant is RepresentationVariant.FREE_GAUSSIAN:
        free_state = FreeGaussianState(seed)
        gaussians = free_state()
        representation_hash = free_state.state_hash
    else:
        gaussians = seed
        representation_hash = _hash({"primitive_ids": seed.primitive_id, "variant": plan.variant.value})
    frozen = FrozenPatientState.create(
        ledger=ledger,
        gaussians=gaussians,
        upstream_state_hash=_hash({
            "assignment_hash": assignment.assignment_hash,
            "plan_hash": plan.plan_hash,
            "preprocessing_hash": preprocessing.record_hash,
            "representation_hash": representation_hash,
        }),
    )
    ledger.expose_target_metadata(target_id)
    commit = ledger.commit_target(target_id, frozen.state_version)
    prediction, receipt = EpisodeController().render_and_register(
        ledger=ledger,
        commit_capability=commit,
        frozen_state=frozen,
        appearance_channel=modality_ids.index(target_metadata.modality_id),
        render_config=config.renderer,
    )
    payload = ledger.reveal_target(target_id, receipt)
    target_decoded = decode_observation(payload, ledger.metadata(target_id), config=config.decoder)
    normalized_target = apply_preprocessing(preprocessing, target_decoded)
    target = normalized_target.image[0].to(dtype=prediction.intensity.dtype, device=prediction.intensity.device)
    target_valid = normalized_target.valid_mask[0].to(device=prediction.intensity.device)
    loss = reconstruction_loss(prediction, target, target_valid, config=config.reconstruction_loss)
    return RepresentationEpisodeResult(
        plan,
        frozen.state_version,
        None,
        free_state,
        prediction,
        target,
        target_valid,
        loss,
        assignment.context_ids,
        _hash(ledger.prediction_records[-1].to_canonical_dict()),
        ledger.audit_hash,
    )


def build_representation_episode_step(
    *,
    ledger: EpisodeLedger,
    assignment: EpisodeAssignment,
    target_id: str,
    representation_variant: str | RepresentationVariant,
    propagation_variant: str = "p0",
    config: LegalEpisodeConfig | None = None,
    encoder: EvidenceEncoder | None = None,
    gaussian_head: FixedGaussianHead | None = None,
    local_field: SharedStructuralField | None = None,
    global_field: GlobalStructuralField | None = None,
    field_maximum_neighbors: int = 8,
    registration_id: str = "manifest-canonical-ras-v1",
    bootstrap_config: AnchorBootstrapConfig | None = None,
    seed_memory_config: SeedMemoryConfig | None = None,
    propagation_config: PropagationConfig | None = None,
    interpolation_config: SparseInterpolationConfig | None = None,
) -> RepresentationEpisodeResult:
    """Run one receipt-gated episode while constructing only selected modules."""

    config = config or LegalEpisodeConfig()
    plan = resolve_representation_plan(representation_variant, propagation_variant=propagation_variant)
    if target_id not in assignment.target_ids:
        raise PermissionError("target_id must be assigned as a target")
    if plan.variant in (RepresentationVariant.INTERPOLATION, RepresentationVariant.FREE_GAUSSIAN):
        if any(module is not None for module in (encoder, gaussian_head, local_field, global_field)):
            raise ValueError("R0/R2 reject encoder, Gaussian-head, anchor-field, and global-field modules")
        return _direct_context_baseline(
            ledger=ledger,
            assignment=assignment,
            target_id=target_id,
            config=config,
            plan=plan,
            interpolation_config=interpolation_config,
        )
    if encoder is None or gaussian_head is None:
        raise ValueError("R1/R3/R4/R5 require the declared evidence encoder and common Gaussian head contract")
    if plan.variant is RepresentationVariant.FIXED_SUPPORT_GAUSSIAN:
        if local_field is not None or global_field is not None:
            raise ValueError("R1 removes anchor and field modules")
        result = build_legal_episode_step(
            ledger=ledger,
            assignment=assignment,
            target_id=target_id,
            encoder=encoder,
            gaussian_head=gaussian_head,
            config=config,
        )
        return RepresentationEpisodeResult(
            plan,
            result.state_version,
            None,
            None,
            result.prediction,
            result.target,
            result.target_valid_mask,
            result.loss,
            result.context_ids,
            result.receipt_record_hash,
            result.audit_hash,
        )
    if propagation_config is None:
        propagation_config = PropagationConfig(variant=propagation_variant)
    elif propagation_config.variant != propagation_variant:
        raise ValueError("propagation config and selected propagation switch disagree")
    static = build_static_episode_step(
        ledger=ledger,
        assignment=assignment,
        target_id=target_id,
        encoder=encoder,
        gaussian_head=gaussian_head,
        config=config,
        patient_id=assignment.patient_id,
        manifest_hash=ledger.manifest_hash,
        patient_config_hash=_hash({"episode_config": config.config_hash, "plan": plan.plan_hash}),
        field_model=local_field,
        field_config_hash=None if local_field is None else local_field.config.config_hash,
        global_field_model=global_field,
        representation_variant=plan.variant,
        field_maximum_neighbors=field_maximum_neighbors,
        registration_id=registration_id,
        bootstrap_config=bootstrap_config,
        seed_memory_config=seed_memory_config,
        propagation_config=propagation_config,
    )
    return RepresentationEpisodeResult(
        plan,
        static.patient_state.state_version,
        static.patient_state,
        None,
        static.prediction,
        static.target,
        static.target_valid_mask,
        static.loss,
        static.context_step.context_ids,
        static.receipt_hash,
        static.audit_hash,
    )


__all__ = [
    "ContextImageEvidence",
    "RepresentationEpisodeResult",
    "build_representation_episode_step",
]
