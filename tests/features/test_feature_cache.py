"""Fail-closed feature cache provenance tests."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from smagm.contracts.coordinates import PhysicalPlane
from smagm.features.cache import FeatureCache, FeatureCacheKey, FeatureCacheMismatchError
from smagm.features.encoder import EncoderConfig, EvidenceEncoder


def _plane(shape_hw: tuple[int, int], observation_id: str = "obs") -> PhysicalPlane:
    return PhysicalPlane(
        pixel_center_origin_ras_mm=(0.0, 0.0, 0.0), axis_u_ras=(1.0, 0.0, 0.0),
        axis_v_ras=(0.0, 1.0, 0.0), spacing_uv_mm=(1.0, 1.0), thickness_mm=1.0,
        shape_hw=shape_hw, signed_normal_ras=(0.0, 0.0, 1.0), observation_id=observation_id,
    )


def _features(dtype: torch.dtype = torch.float32, observation_id: str = "obs"):
    shape = (9, 11)
    encoder = EvidenceEncoder(EncoderConfig(variant="e1"))
    image = torch.randn((1, 1, *shape), dtype=dtype)
    features = encoder(image, _plane(shape, observation_id), "mri")
    key = FeatureCacheKey.from_features(
        features,
        batch_index=0,
        encoder_variant="e1",
        encoder_configuration_hash=encoder.config.config_hash,
        encoder_state_hash=encoder.state_hash(),
        input_preprocessing_hash="b" * 64,
        input_content_hash="a" * 64,
    )
    return encoder, features, key


def test_exact_matching_cache_retrieval_and_target_rejection() -> None:
    _, features, key = _features()
    cache = FeatureCache()
    cache.put(key, features)
    assert cache.get(key) is features
    assert len(cache) == 1
    with pytest.raises(ValueError, match="target-derived"):
        cache.put(key, features, target_derived=True)


@pytest.mark.parametrize(
    "field",
    (
        "encoder_configuration_hash",
        "encoder_state_hash",
        "input_preprocessing_hash",
        "input_content_hash",
        "dtype",
        "valid_feature_mask_hash",
    ),
)
def test_mismatched_cache_metadata_fails_closed(field: str) -> None:
    _, features, key = _features()
    cache = FeatureCache()
    cache.put(key, features)
    replacement = "c" * 64
    if field == "dtype":
        replacement = "torch.float64"
    mismatch = replace(key, **{field: replacement})
    with pytest.raises(FeatureCacheMismatchError):
        cache.get(mismatch)


def test_output_stride_mismatch_is_rejected_before_cache_lookup() -> None:
    _, features, key = _features()
    with pytest.raises(ValueError, match="output_stride"):
        replace(key, output_stride=2)


def test_mismatched_plane_and_transform_fail_closed() -> None:
    _, features, key = _features(observation_id="obs-a")
    _, other_features, other_key = _features(observation_id="obs-b")
    cache = FeatureCache()
    cache.put(key, features)
    with pytest.raises(FeatureCacheMismatchError):
        cache.get(other_key)
    with pytest.raises(ValueError):
        replace(key, feature_grid_transform=other_key.feature_grid_transform)


def test_input_identity_distinguishes_payload_and_mask_content() -> None:
    _, features, key = _features()
    cache = FeatureCache()
    cache.put(key, features)
    with pytest.raises(FeatureCacheMismatchError):
        cache.get(replace(key, input_content_hash="b" * 64))
    with pytest.raises(FeatureCacheMismatchError):
        cache.get(replace(key, valid_feature_mask_hash="c" * 64))


def test_cached_feature_value_mutation_fails_closed_without_detaching_values() -> None:
    _, features, key = _features()
    cache = FeatureCache()
    cache.put(key, features)
    with torch.no_grad():
        features.structural.add_(1.0)
    with pytest.raises(FeatureCacheMismatchError, match="values changed"):
        cache.get(key)
