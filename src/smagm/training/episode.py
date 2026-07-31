"""Legal context-to-target orchestration for the fixed-topology T1-C path."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

import torch

from ..baselines.fixed_gaussian import FixedGaussianHead, construct_fixed_gaussians
from ..baselines.fixed_support import FixedSupportBatch, FixedSupportConfig, sample_fixed_supports
from ..contracts.episode import EpisodeAssignment, EpisodeController, EpisodeLedger, FrozenPatientState
from ..data.io import DecoderConfig, decode_observation
from ..data.normalization import (
    NormalizationConfig,
    PreprocessingRecord,
    apply_preprocessing,
    fit_preprocessing,
)
from ..features.cache import FeatureCache, FeatureCacheKey
from ..features.encoder import EvidenceEncoder
from ..losses.reconstruction import ReconstructionLossConfig, ReconstructionLossResult, reconstruction_loss
from ..renderer import RenderConfig, RenderResult


def _json_hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LegalEpisodeConfig:
    decoder: DecoderConfig = DecoderConfig()
    normalization: NormalizationConfig = NormalizationConfig()
    supports: FixedSupportConfig = FixedSupportConfig()
    renderer: RenderConfig = RenderConfig()
    reconstruction_loss: ReconstructionLossConfig = ReconstructionLossConfig()
    appearance_channel: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.appearance_channel, int) or self.appearance_channel < 0:
            raise ValueError("appearance_channel must be a non-negative integer")

    @property
    def config_hash(self) -> str:
        return _json_hash(
            {
                "appearance_channel": self.appearance_channel,
                "decoder": self.decoder.__dict__,
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
class LegalEpisodeStep:
    """Live post-reveal training handoff; target never appears in patient state."""

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
    context_ids: tuple[str, ...]
    feature_cache_key_hashes: tuple[str, ...]
    cache_bytes: int
    support_count: int

    def __post_init__(self) -> None:
        if self.target.ndim != 2 or self.target_valid_mask.shape != self.target.shape:
            raise ValueError("legal target handoff must contain [H, W] tensors")
        if self.target_valid_mask.dtype is not torch.bool:
            raise TypeError("target_valid_mask must be bool")
        if self.loss.status != "OK":
            raise ValueError("trainer handoff cannot contain a skipped reconstruction loss")
        if not self.context_ids or self.target_id in self.context_ids:
            raise ValueError("target must remain distinct from context IDs")
        if self.cache_bytes <= 0 or self.support_count <= 0:
            raise ValueError("episode state requires non-empty cache and supports")


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
            "input_preprocessing_hash": key.input_preprocessing_hash,
            "observation_id": key.observation_id,
            "output_channel_contract": key.output_channel_contract,
            "valid_feature_mask_hash": key.valid_feature_mask_hash,
        }
    )


def _feature_bytes(features: object) -> int:
    tensors = (
        features.structural,
        features.appearance,
        features.reliability,
        features.valid_feature_mask,
    )
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


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
    """Build context-only state, render/register, then reveal and score target."""

    if not isinstance(ledger, EpisodeLedger) or not isinstance(assignment, EpisodeAssignment):
        raise TypeError("ledger and assignment must use T0.5 contracts")
    if ledger.assignment_hash != assignment.assignment_hash:
        raise ValueError("assignment does not match the episode ledger")
    if target_id not in assignment.target_ids:
        raise PermissionError("target_id must be assigned as a target")
    if not isinstance(encoder, EvidenceEncoder) or not isinstance(gaussian_head, FixedGaussianHead):
        raise TypeError("encoder and gaussian_head must use T1-B contracts")
    config = config or LegalEpisodeConfig()
    controller = controller or EpisodeController()
    head_parameter = next(gaussian_head.parameters())
    device, dtype = head_parameter.device, head_parameter.dtype

    decoded_context = []
    for observation_id in assignment.context_ids:
        metadata = ledger.metadata(observation_id)
        payload = ledger.open_context(observation_id)
        decoded_context.append(decode_observation(payload, metadata, config=config.decoder))
    preprocessing = fit_preprocessing(
        decoded_context,
        context_ids=assignment.context_ids,
        config=config.normalization,
    )

    cache = FeatureCache()
    support_batches: list[FixedSupportBatch] = []
    cache_key_hashes: list[str] = []
    cache_bytes = 0
    encoder_state_hash = encoder.state_hash()
    for decoded in decoded_context:
        normalized = apply_preprocessing(preprocessing, decoded)
        image = normalized.image.unsqueeze(0).to(device=device, dtype=dtype)
        valid_mask = normalized.valid_mask.unsqueeze(0).to(device=device)
        features = encoder(image, decoded.metadata.plane, decoded.modality_id, valid_mask)
        key = FeatureCacheKey.from_features(
            features,
            batch_index=0,
            encoder_variant=encoder.config.variant,
            encoder_configuration_hash=encoder.config.config_hash,
            encoder_state_hash=encoder_state_hash,
            input_preprocessing_hash=normalized.preprocessing_hash,
        )
        cache.put(key, features, target_derived=False)
        cached = cache.get(key)
        support_batches.append(
            sample_fixed_supports(cached, decoded.metadata.plane, config=config.supports)
        )
        cache_key_hashes.append(_cache_key_hash(key))
        cache_bytes += _feature_bytes(cached)

    supports = _combine_supports(support_batches)
    if gaussian_head.config.input_dim != supports.feature_vectors.shape[1]:
        raise ValueError("Gaussian head input contract disagrees with encoded feature channels")
    raw = gaussian_head(supports.feature_vectors)
    gaussians = construct_fixed_gaussians(supports, raw, config=gaussian_head.config)
    upstream_state_hash = _json_hash(
        {
            "assignment_hash": assignment.assignment_hash,
            "cache_key_hashes": cache_key_hashes,
            "episode_config_hash": config.config_hash,
            "preprocessing_parameters_hash": preprocessing.parameters_hash,
        }
    )
    frozen = FrozenPatientState.create(
        ledger=ledger,
        gaussians=gaussians,
        upstream_state_hash=upstream_state_hash,
    )

    # Geometry is public manifest metadata; target payload remains unopened.
    ledger.expose_target_metadata(target_id)
    commit = ledger.commit_target(target_id, frozen.state_version)
    prediction, receipt = controller.render_and_register(
        ledger=ledger,
        commit_capability=commit,
        frozen_state=frozen,
        appearance_channel=config.appearance_channel,
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
    receipt_record_hash = _json_hash(ledger.prediction_records[-1].to_canonical_dict())
    return LegalEpisodeStep(
        assignment_hash=assignment.assignment_hash,
        state_version=frozen.state_version,
        target_id=target_id,
        prediction=prediction,
        target=target,
        target_valid_mask=target_valid,
        loss=loss,
        receipt_record_hash=receipt_record_hash,
        audit_hash=ledger.audit_hash,
        preprocessing=preprocessing,
        context_ids=assignment.context_ids,
        feature_cache_key_hashes=tuple(cache_key_hashes),
        cache_bytes=cache_bytes,
        support_count=supports.count,
    )
