"""Fail-closed in-memory feature cache for legal context evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch

from .contracts import EncoderFeatureMaps, FeatureGridToPlaneTransform


class FeatureCacheMismatchError(KeyError):
    """Raised for a cache miss or any key mismatch."""


@dataclass(frozen=True)
class FeatureCacheKey:
    """All provenance needed to identify one legal context feature tensor."""

    observation_id: str
    canonical_source_plane_hash: str
    encoder_variant: str
    encoder_configuration_hash: str
    encoder_state_hash: str
    input_preprocessing_hash: str
    feature_grid_transform: FeatureGridToPlaneTransform
    valid_feature_mask_hash: str
    dtype: str
    output_channel_contract: tuple[int, int, int]

    def __post_init__(self) -> None:
        if not self.observation_id or self.encoder_variant not in ("e0", "e1", "e2"):
            raise ValueError("observation_id and encoder_variant are required")
        for name in (
            "canonical_source_plane_hash",
            "encoder_configuration_hash",
            "encoder_state_hash",
            "input_preprocessing_hash",
            "valid_feature_mask_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty hash or identity")
        if not isinstance(self.feature_grid_transform, FeatureGridToPlaneTransform):
            raise TypeError("feature_grid_transform must be a FeatureGridToPlaneTransform")
        if self.feature_grid_transform.input_plane is None:
            raise ValueError("feature_grid_transform must bind a source plane")
        if self.feature_grid_transform.input_plane.observation_id != self.observation_id:
            raise ValueError("cache observation_id must match the transform-bound plane")
        if self.feature_grid_transform.source_plane_hash != self.canonical_source_plane_hash:
            raise ValueError("cache source-plane hash must match the transform-bound plane")
        if self.dtype not in ("torch.float32", "torch.float64"):
            raise ValueError("cache dtype must be torch.float32 or torch.float64")
        if len(self.output_channel_contract) != 3 or any(int(value) <= 0 for value in self.output_channel_contract):
            raise ValueError("output_channel_contract must contain three positive channel counts")

    @classmethod
    def from_features(
        cls,
        features: EncoderFeatureMaps,
        *,
        batch_index: int,
        encoder_variant: str,
        encoder_configuration_hash: str,
        encoder_state_hash: str,
        input_preprocessing_hash: str,
    ) -> "FeatureCacheKey":
        if not 0 <= batch_index < features.batch_size:
            raise IndexError("batch_index is outside feature batch")
        transform = features.grid_to_planes[batch_index]
        plane = transform.input_plane
        assert plane is not None
        mask = features.valid_feature_mask[batch_index, 0]
        mask_hash = hashlib.sha256(mask.to(dtype=torch.uint8).detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        return cls(
            observation_id=plane.observation_id or "",
            canonical_source_plane_hash=transform.source_plane_hash,
            encoder_variant=encoder_variant,
            encoder_configuration_hash=encoder_configuration_hash,
            encoder_state_hash=encoder_state_hash,
            input_preprocessing_hash=input_preprocessing_hash,
            feature_grid_transform=transform,
            valid_feature_mask_hash=mask_hash,
            dtype=str(features.structural.dtype),
            output_channel_contract=(features.structural.shape[1], features.appearance.shape[1], features.reliability.shape[1]),
        )


@dataclass(frozen=True)
class CachedFeatureMaps:
    key: FeatureCacheKey
    features: EncoderFeatureMaps


class FeatureCache:
    """Small exact-key cache; target-derived insertion is prohibited."""

    def __init__(self) -> None:
        self._items: dict[FeatureCacheKey, CachedFeatureMaps] = {}

    def put(self, key: FeatureCacheKey, features: EncoderFeatureMaps, *, target_derived: bool = False) -> None:
        if target_derived:
            raise ValueError("target-derived feature caching is forbidden before legal target reveal")
        if not isinstance(key, FeatureCacheKey) or not isinstance(features, EncoderFeatureMaps):
            raise TypeError("cache requires a FeatureCacheKey and EncoderFeatureMaps")
        if features.batch_size != 1:
            raise ValueError("cache values must be one explicitly keyed feature batch item")
        actual = FeatureCacheKey.from_features(
            features,
            batch_index=0,
            encoder_variant=key.encoder_variant,
            encoder_configuration_hash=key.encoder_configuration_hash,
            encoder_state_hash=key.encoder_state_hash,
            input_preprocessing_hash=key.input_preprocessing_hash,
        )
        if actual != key:
            raise FeatureCacheMismatchError("cache key does not match feature provenance or output contract")
        self._items[key] = CachedFeatureMaps(key=key, features=features)

    def get(self, key: FeatureCacheKey) -> EncoderFeatureMaps:
        if key not in self._items:
            raise FeatureCacheMismatchError("no exact feature-cache entry exists for the requested key")
        return self._items[key].features

    def __len__(self) -> int:
        return len(self._items)
