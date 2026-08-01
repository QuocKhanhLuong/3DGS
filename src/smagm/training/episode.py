"""Legal context-only and context-to-target orchestration for T1-C."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from types import MappingProxyType
from typing import Iterable, Mapping

import torch

from ..baselines.fixed_gaussian import FixedGaussianHead, construct_fixed_gaussians
from ..baselines.fixed_support import FixedSupportBatch, FixedSupportConfig, sample_fixed_supports
from ..contracts.coordinates import PhysicalPlane
from ..contracts.episode import EpisodeAssignment, EpisodeController, EpisodeLedger, FrozenPatientState
from ..data.episodes import EpisodeSamplingError, EpisodeSamplingFailureReason
from ..data.io import DecoderConfig, decode_observation
from ..data.normalization import NormalizationConfig, PreprocessingRecord, apply_preprocessing, fit_preprocessing
from ..features.cache import FeatureCache, FeatureCacheKey
from ..features.contracts import EncoderFeatureMaps
from ..features.encoder import EvidenceEncoder
from ..losses.reconstruction import ReconstructionLossConfig, ReconstructionLossResult, reconstruction_loss
from ..renderer import RenderConfig, RenderResult


def _json_hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LegalEpisodeConfig:
    """The shared, immutable context-to-target contract for one T1-C run."""

    decoder: DecoderConfig = DecoderConfig()
    normalization: NormalizationConfig = NormalizationConfig()
    supports: FixedSupportConfig = FixedSupportConfig()
    renderer: RenderConfig = RenderConfig()
    reconstruction_loss: ReconstructionLossConfig = ReconstructionLossConfig()
    modality_to_appearance_channel: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        mapping = {"synthetic-mri": 0} if self.modality_to_appearance_channel is None else dict(self.modality_to_appearance_channel)
        if not mapping:
            raise ValueError("modality_to_appearance_channel must declare at least one modality")
        for modality_id, channel in mapping.items():
            if not isinstance(modality_id, str) or not modality_id or not isinstance(channel, int) or channel < 0:
                raise ValueError("modality_to_appearance_channel must map named modalities to non-negative integers")
        object.__setattr__(self, "modality_to_appearance_channel", MappingProxyType(dict(sorted(mapping.items()))))

    @property
    def modality_mapping_hash(self) -> str:
        return _json_hash(dict(self.modality_to_appearance_channel or {}))

    def appearance_channel_for(self, modality_id: str, *, available_channels: int) -> int:
        mapping = self.modality_to_appearance_channel or {}
        if modality_id not in mapping:
            raise ValueError(f"no appearance channel is declared for modality {modality_id!r}")
        channel = mapping[modality_id]
        if channel >= available_channels:
            raise ValueError(
                f"appearance channel {channel} for modality {modality_id!r} is outside the Gaussian-head contract"
            )
        return channel

    @property
    def config_hash(self) -> str:
        return _json_hash(
            {
                "decoder": self.decoder.__dict__,
                "modality_to_appearance_channel": dict(self.modality_to_appearance_channel or {}),
                "normalization": self.normalization.to_dict(),
                "reconstruction_loss": self.reconstruction_loss.__dict__,
                "renderer_version": self.renderer.renderer_version,
                "supports": {
                    "border_vu": self.supports.border_vu,
                    "max_points": self.supports.max_points,
                    "step_vu": self.supports.step_vu,
                },
            }
        )


@dataclass(frozen=True)
class ContextEvidence:
    """Context-only evidence retained only for the current optimizer step."""

    observation_id: str
    modality_id: str
    plane: PhysicalPlane
    normalized_image: torch.Tensor  # [1, 1, H, W]
    valid_mask: torch.Tensor  # [1, 1, H, W], bool
    features: EncoderFeatureMaps
    cache_key_hash: str

    def __post_init__(self) -> None:
        if self.normalized_image.ndim != 4 or self.normalized_image.shape[0:2] != (1, 1):
            raise ValueError("context evidence image must have shape [1, 1, H, W]")
        if self.valid_mask.shape != self.normalized_image.shape or self.valid_mask.dtype is not torch.bool:
            raise ValueError("context evidence valid mask must match the normalized image")
        if self.features.batch_size != 1 or self.features.modality_ids != (self.modality_id,):
            raise ValueError("context evidence features must bind exactly the context modality")


@dataclass(frozen=True)
class ContextOnlyEpisodeStep:
    """Warm-up handoff that contains no target metadata, payload, or loss."""

    assignment_hash: str
    preprocessing: PreprocessingRecord
    context_evidence: tuple[ContextEvidence, ...]
    context_ids: tuple[str, ...]
    feature_cache_key_hashes: tuple[str, ...]
    cache_bytes: int
    support_count: int
    support_topology_hash: str
    encoder_runtime_seconds: float

    def __post_init__(self) -> None:
        if not self.context_evidence or not self.context_ids or self.cache_bytes <= 0 or self.support_count <= 0:
            raise ValueError("context-only episode requires non-empty legal context evidence")


@dataclass(frozen=True)
class LegalEpisodeStep:
    """Live post-reveal training handoff; target never appears in frozen state."""

    assignment_hash: str
    state_version: str
    target_id: str
    prediction: RenderResult
    target: torch.Tensor
    target_valid_mask: torch.Tensor
    loss: ReconstructionLossResult
    receipt_record_hash: str
    audit_hash: str
    preprocessing: PreprocessingRecord
    context_evidence: tuple[ContextEvidence, ...]
    context_ids: tuple[str, ...]
    feature_cache_key_hashes: tuple[str, ...]
    cache_bytes: int
    support_count: int
    support_topology_hash: str
    encoder_runtime_seconds: float

    def __post_init__(self) -> None:
        if self.target.ndim != 2 or self.target_valid_mask.shape != self.target.shape:
            raise ValueError("legal target handoff must contain [H, W] tensors")
        if self.target_valid_mask.dtype is not torch.bool:
            raise TypeError("target_valid_mask must be bool")
        if self.loss.status != "OK":
            raise ValueError("trainer handoff cannot contain a skipped reconstruction loss")
        if not self.context_evidence or not self.context_ids or self.target_id in self.context_ids:
            raise ValueError("target must remain distinct from non-empty context evidence")
        if self.cache_bytes <= 0 or self.support_count <= 0:
            raise ValueError("episode state requires non-empty cache and supports")


def _validate_inputs(
    ledger: EpisodeLedger,
    assignment: EpisodeAssignment,
    encoder: EvidenceEncoder,
    gaussian_head: FixedGaussianHead,
) -> None:
    if not isinstance(ledger, EpisodeLedger) or not isinstance(assignment, EpisodeAssignment):
        raise TypeError("ledger and assignment must use T0.5 contracts")
    if ledger.assignment_hash != assignment.assignment_hash:
        raise ValueError("assignment does not match the episode ledger")
    if not isinstance(encoder, EvidenceEncoder) or not isinstance(gaussian_head, FixedGaussianHead):
        raise TypeError("encoder and gaussian_head must use T1-B contracts")


def _combine_supports(items: Iterable[FixedSupportBatch]) -> FixedSupportBatch:
    batches = tuple(items)
    if not batches:
        raise ValueError("at least one context support batch is required")
    return FixedSupportBatch(
        centers_ras_mm=torch.cat([item.centers_ras_mm for item in batches], dim=0),
        feature_vectors=torch.cat([item.feature_vectors for item in batches], dim=0),
        feature_indices_vu=torch.cat([item.feature_indices_vu for item in batches], dim=0),
        reliability=torch.cat([item.reliability for item in batches], dim=0),
        observation_ids=tuple(value for item in batches for value in item.observation_ids),
        source_plane_hashes=tuple(value for item in batches for value in item.source_plane_hashes),
        batch_index=0,
        support_basis_ras=torch.cat([item.support_basis_ras for item in batches], dim=0),
    )


def _cache_key_hash(key: FeatureCacheKey) -> str:
    return _json_hash(
        {
            "canonical_source_plane_hash": key.canonical_source_plane_hash,
            "dtype": key.dtype,
            "encoder_configuration_hash": key.encoder_configuration_hash,
            "encoder_state_hash": key.encoder_state_hash,
            "encoder_variant": key.encoder_variant,
            "feature_grid_transform_hash": key.feature_grid_transform.transform_hash,
            "input_content_hash": key.input_content_hash,
            "input_preprocessing_hash": key.input_preprocessing_hash,
            "observation_id": key.observation_id,
            "output_channel_contract": key.output_channel_contract,
            "output_stride": key.output_stride,
            "valid_feature_mask_hash": key.valid_feature_mask_hash,
        }
    )


def _feature_bytes(features: EncoderFeatureMaps) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in (features.structural, features.appearance, features.reliability, features.valid_feature_mask)
    )


def _support_topology_hash(supports: FixedSupportBatch) -> str:
    return _json_hash(
        {
            "feature_indices_vu": supports.feature_indices_vu.detach().cpu().tolist(),
            "observation_ids": supports.observation_ids,
            "source_plane_hashes": supports.source_plane_hashes,
            "support_basis_ras": supports.support_basis_ras.detach().cpu().tolist(),
        }
    )


def _build_context_evidence(
    *,
    ledger: EpisodeLedger,
    assignment: EpisodeAssignment,
    encoder: EvidenceEncoder,
    gaussian_head: FixedGaussianHead,
    config: LegalEpisodeConfig,
) -> tuple[PreprocessingRecord, tuple[ContextEvidence, ...], tuple[FixedSupportBatch, ...], tuple[str, ...], int, float]:
    """The one shared context-only path used by warm-up and reconstruction."""

    head_parameter = next(gaussian_head.parameters())
    device, dtype = head_parameter.device, head_parameter.dtype
    decoded_context = []
    for observation_id in assignment.context_ids:
        metadata = ledger.metadata(observation_id)
        payload = ledger.open_context(observation_id)
        decoded_context.append(decode_observation(payload, metadata, config=config.decoder))
    preprocessing = fit_preprocessing(decoded_context, context_ids=assignment.context_ids, config=config.normalization)

    cache = FeatureCache()
    evidence: list[ContextEvidence] = []
    support_batches: list[FixedSupportBatch] = []
    cache_key_hashes: list[str] = []
    cache_bytes = 0
    encoder_runtime_seconds = 0.0
    encoder_state_hash = encoder.state_hash()
    for decoded in decoded_context:
        config.appearance_channel_for(decoded.modality_id, available_channels=gaussian_head.config.appearance_channels)
        normalized = apply_preprocessing(preprocessing, decoded)
        image = normalized.image.unsqueeze(0).to(device=device, dtype=dtype)
        valid_mask = normalized.valid_mask.unsqueeze(0).to(device=device)
        start = time.perf_counter()
        features = encoder(image, decoded.metadata.plane, decoded.modality_id, valid_mask)
        encoder_runtime_seconds += time.perf_counter() - start
        key = FeatureCacheKey.from_features(
            features,
            batch_index=0,
            encoder_variant=encoder.config.variant,
            encoder_configuration_hash=encoder.config.config_hash,
            encoder_state_hash=encoder_state_hash,
            input_preprocessing_hash=normalized.preprocessing_hash,
            input_content_hash=normalized.input_content_hash,
        )
        cache.put(key, features, target_derived=False)
        cached = cache.get(key)
        support_batches.append(sample_fixed_supports(cached, decoded.metadata.plane, config=config.supports))
        cache_hash = _cache_key_hash(key)
        evidence.append(ContextEvidence(decoded.observation_id, decoded.modality_id, decoded.metadata.plane, image, valid_mask, cached, cache_hash))
        cache_key_hashes.append(cache_hash)
        cache_bytes += _feature_bytes(cached)
    return preprocessing, tuple(evidence), tuple(support_batches), tuple(cache_key_hashes), cache_bytes, encoder_runtime_seconds


def build_context_only_episode_step(
    *,
    ledger: EpisodeLedger,
    assignment: EpisodeAssignment,
    encoder: EvidenceEncoder,
    gaussian_head: FixedGaussianHead,
    config: LegalEpisodeConfig | None = None,
) -> ContextOnlyEpisodeStep:
    """Build legal context evidence for auxiliary warm-up without target access."""

    _validate_inputs(ledger, assignment, encoder, gaussian_head)
    config = config or LegalEpisodeConfig()
    preprocessing, evidence, supports, key_hashes, cache_bytes, encoder_runtime_seconds = _build_context_evidence(
        ledger=ledger, assignment=assignment, encoder=encoder, gaussian_head=gaussian_head, config=config
    )
    combined = _combine_supports(supports)
    return ContextOnlyEpisodeStep(
        assignment_hash=assignment.assignment_hash,
        preprocessing=preprocessing,
        context_evidence=evidence,
        context_ids=assignment.context_ids,
        feature_cache_key_hashes=key_hashes,
        cache_bytes=cache_bytes,
        support_count=combined.count,
        support_topology_hash=_support_topology_hash(combined),
        encoder_runtime_seconds=encoder_runtime_seconds,
    )


def build_legal_episode_step(
    *,
    ledger: EpisodeLedger,
    assignment: EpisodeAssignment,
    target_id: str,
    encoder: EvidenceEncoder,
    gaussian_head: FixedGaussianHead,
    config: LegalEpisodeConfig | None = None,
    controller: EpisodeController | None = None,
) -> LegalEpisodeStep:
    """Freeze context state, render/register, then reveal one legal target."""

    _validate_inputs(ledger, assignment, encoder, gaussian_head)
    if target_id not in assignment.target_ids:
        raise PermissionError("target_id must be assigned as a target")
    if len(assignment.target_ids) != 1:
        raise ValueError("the T1-C reference supports exactly one target per episode")
    config = config or LegalEpisodeConfig()
    controller = controller or EpisodeController()
    head_parameter = next(gaussian_head.parameters())
    device, dtype = head_parameter.device, head_parameter.dtype
    target_metadata = ledger.metadata(target_id)  # public manifest geometry only
    context_modalities = {ledger.metadata(observation_id).modality_id for observation_id in assignment.context_ids}
    if target_metadata.modality_id not in context_modalities:
        raise EpisodeSamplingError(
            EpisodeSamplingFailureReason.MISSING_CONTEXT_MODALITY,
            "the T1-C target modality requires one declared same-modality context observation",
        )
    target_appearance_channel = config.appearance_channel_for(
        target_metadata.modality_id, available_channels=gaussian_head.config.appearance_channels
    )
    preprocessing, evidence, support_batches, cache_key_hashes, cache_bytes, encoder_runtime_seconds = _build_context_evidence(
        ledger=ledger, assignment=assignment, encoder=encoder, gaussian_head=gaussian_head, config=config
    )
    supports = _combine_supports(support_batches)
    if gaussian_head.config.input_dim != supports.feature_vectors.shape[1]:
        raise ValueError("Gaussian head input contract disagrees with encoded feature channels")
    gaussians = construct_fixed_gaussians(supports, gaussian_head(supports.feature_vectors), config=gaussian_head.config)
    frozen = FrozenPatientState.create(
        ledger=ledger,
        gaussians=gaussians,
        upstream_state_hash=_json_hash(
            {
                "assignment_hash": assignment.assignment_hash,
                "cache_key_hashes": cache_key_hashes,
                "episode_config_hash": config.config_hash,
                "preprocessing_record_hash": preprocessing.record_hash,
            }
        ),
    )

    ledger.expose_target_metadata(target_id)
    commit = ledger.commit_target(target_id, frozen.state_version)
    prediction, receipt = controller.render_and_register(
        ledger=ledger,
        commit_capability=commit,
        frozen_state=frozen,
        appearance_channel=target_appearance_channel,
        render_config=config.renderer,
    )
    target_payload = ledger.reveal_target(target_id, receipt)
    target_decoded = decode_observation(target_payload, ledger.metadata(target_id), config=config.decoder)
    normalized_target = apply_preprocessing(preprocessing, target_decoded)
    target = normalized_target.image[0].to(device=device, dtype=dtype)
    target_valid = normalized_target.valid_mask[0].to(device=device)
    loss = reconstruction_loss(prediction, target, target_valid, config=config.reconstruction_loss)
    if not ledger.prediction_records:
        raise RuntimeError("legal target reveal completed without a prediction record")
    return LegalEpisodeStep(
        assignment_hash=assignment.assignment_hash,
        state_version=frozen.state_version,
        target_id=target_id,
        prediction=prediction,
        target=target,
        target_valid_mask=target_valid,
        loss=loss,
        receipt_record_hash=_json_hash(ledger.prediction_records[-1].to_canonical_dict()),
        audit_hash=ledger.audit_hash,
        preprocessing=preprocessing,
        context_evidence=evidence,
        context_ids=assignment.context_ids,
        feature_cache_key_hashes=cache_key_hashes,
        cache_bytes=cache_bytes,
        support_count=supports.count,
        support_topology_hash=_support_topology_hash(supports),
        encoder_runtime_seconds=encoder_runtime_seconds,
    )
