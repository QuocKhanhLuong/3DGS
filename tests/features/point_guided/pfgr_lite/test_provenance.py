from __future__ import annotations

import torch

from smagm.features.point_guided.pfgr_lite import (
    PFGRLiteConfig,
    PFGRLiteModel,
    ProducerCompatibility,
    ValueFitIdentity,
    batchnorm_state_digest,
    canonical_digest,
    module_state_digest,
    tensor_digest,
)
from smagm.features.point_guided import PointGuidedConfig


def test_tensor_and_module_hashes_include_dtype_shape_and_bytes() -> None:
    value = torch.tensor([1.0, 2.0], dtype=torch.float32)
    assert tensor_digest(value) != tensor_digest(value.to(torch.float64))
    module = torch.nn.BatchNorm3d(2)
    before = batchnorm_state_digest(module)
    module.running_mean[0] = 1.0
    assert batchnorm_state_digest(module) != before
    assert module_state_digest(module) != canonical_digest("module")


def test_producer_compatibility_excludes_value_fit_identity() -> None:
    producer = ProducerCompatibility(static_head_hash="static-a")
    changed_v = ValueFitIdentity(input_variant=126, architecture_hash="v1", weights_hash="w1", fit_config_hash="f1", bank_manifest_hash="b1", gain_scale_hash="s1")
    changed_v2 = ValueFitIdentity(input_variant=126, architecture_hash="v2", weights_hash="w2", fit_config_hash="f2", bank_manifest_hash="b1", gain_scale_hash="s1")
    assert producer.digest == ProducerCompatibility(static_head_hash="static-a").digest
    assert changed_v.digest != changed_v2.digest
    assert producer.digest != changed_v.digest


def test_context_provenance_reports_one_traversal_and_synthetic_weights() -> None:
    frontend = PointGuidedConfig(num_semantic_classes=3, num_points=2, point_candidate_multiplier=2, offset_hidden_channels=12)
    model = PFGRLiteModel(PFGRLiteConfig(), frontend_config=frontend).eval()
    context = model.encode_observations(torch.randn(1, 3, 7, 7, 7), None, (1.0, 1.0, 1.0))
    assert context.producer.source_provenance.synthetic_untrained
    assert context.producer.source_provenance.traversal_count == 1
    assert context.producer.compatibility.digest == context.producer.digest

