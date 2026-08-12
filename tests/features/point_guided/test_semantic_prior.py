"""CPU-only invariants for the locked MedicalNet semantic prior."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from smagm.features.point_guided.config import PointGuidedConfig
from smagm.features.point_guided.contracts import (
    COARSE_SEMANTIC_CLASS_NAMES,
    NUM_COARSE_SEMANTIC_CLASSES,
)
from smagm.features.point_guided.medicalnet_resnet10 import (
    MedicalNetCheckpointError,
    MedicalNetFeatures,
    MedicalNetResNet10,
    _load_checkpoint_object,
    sha256_file,
)
from smagm.features.point_guided import medicalnet_resnet10 as medicalnet_module
from smagm.features.point_guided.semantic_prior import SemanticPrior


def _config(**overrides: object) -> PointGuidedConfig:
    return PointGuidedConfig(num_semantic_classes=3, **overrides)


def test_backbone_exposes_typed_intermediate_features_at_locked_boundaries() -> None:
    backbone = MedicalNetResNet10(in_channels=3).eval()
    volumes = torch.randn(1, 3, 9, 11, 13)
    state_dict_keys_before = tuple(backbone.state_dict())
    maxpool_inputs: list[torch.Tensor] = []

    def capture_maxpool_input(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        maxpool_inputs.append(inputs[0].clone())

    hook = backbone.maxpool.register_forward_pre_hook(capture_maxpool_input)
    try:
        with torch.no_grad():
            features = backbone.forward_intermediate_features(volumes)
    finally:
        hook.remove()

    assert isinstance(features, MedicalNetFeatures)
    assert isinstance(features.shallow, torch.Tensor)
    assert isinstance(features.layer1, torch.Tensor)
    assert isinstance(features.deep, torch.Tensor)
    assert features.shallow.shape == (1, 64, 5, 6, 7)
    assert features.layer1.shape == (1, 64, 3, 3, 4)
    assert features.deep.shape == (1, 512, 2, 2, 2)
    for shallow_size, layer1_size, deep_size in zip(
        features.shallow.shape[-3:],
        features.layer1.shape[-3:],
        features.deep.shape[-3:],
    ):
        assert shallow_size > layer1_size > deep_size

    assert len(maxpool_inputs) == 1
    torch.testing.assert_close(features.shallow, maxpool_inputs[0], rtol=0.0, atol=0.0)
    with torch.no_grad():
        expected_shallow = backbone.relu(backbone.bn1(backbone.conv1(volumes)))
        expected_layer1 = backbone.layer1(backbone.maxpool(expected_shallow))
        expected_deep = backbone.layer4(backbone.layer3(backbone.layer2(expected_layer1)))
        legacy_deep = backbone.forward_features(volumes)
        module_deep = backbone(volumes)

    torch.testing.assert_close(features.shallow, expected_shallow, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(features.layer1, expected_layer1, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(features.deep, expected_deep, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(features.deep, legacy_deep, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(features.deep, module_deep, rtol=1e-6, atol=1e-6)
    assert tuple(backbone.state_dict()) == state_dict_keys_before


def test_intermediate_feature_api_uses_one_backbone_traversal() -> None:
    backbone = MedicalNetResNet10(in_channels=3).eval()
    volumes = torch.randn(1, 3, 9, 11, 13)
    module_names = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4")
    calls = {name: 0 for name in module_names}

    def counter(name: str):
        def count_call(*_args: object) -> None:
            calls[name] += 1

        return count_call

    hooks = [getattr(backbone, name).register_forward_hook(counter(name)) for name in module_names]
    try:
        with torch.no_grad():
            backbone.forward_intermediate_features(volumes)
    finally:
        for hook in hooks:
            hook.remove()

    assert calls == {name: 1 for name in module_names}


def test_semantic_prior_uses_the_shared_deep_feature_without_changing_logits() -> None:
    prior = SemanticPrior(_config()).eval()
    volumes = torch.randn(1, 3, 9, 11, 13)

    with torch.no_grad():
        shared_features = prior.extract_intermediate_features(volumes)
        compatible_deep = prior.extract_features(volumes)

    torch.testing.assert_close(shared_features.deep, compatible_deep, rtol=1e-6, atol=1e-6)

    layer4_outputs: list[torch.Tensor] = []
    head_inputs: list[torch.Tensor] = []

    def capture_layer4_output(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        layer4_outputs.append(output.clone())

    def capture_head_input(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        head_inputs.append(inputs[0].clone())

    layer4_hook = prior.backbone.layer4.register_forward_hook(capture_layer4_output)
    head_hook = prior.semantic_head.register_forward_pre_hook(capture_head_input)
    try:
        with torch.no_grad():
            logits = prior.forward_logits(volumes)
    finally:
        layer4_hook.remove()
        head_hook.remove()

    assert logits.shape == (1, 3, 9, 11, 13)
    assert len(layer4_outputs) == 1
    assert len(head_inputs) == 1
    torch.testing.assert_close(head_inputs[0], layer4_outputs[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(head_inputs[0], shared_features.deep, rtol=1e-6, atol=1e-6)


def test_phase_two_feature_control_defaults_and_validation() -> None:
    config = _config()

    assert config.freeze_coarse_backbone is True
    assert config.detach_backbone_features is True
    assert config.spectral_tap == "conv1_pre_maxpool"

    with pytest.raises(ValueError, match="spectral_tap"):
        _config(spectral_tap="layer2")
    with pytest.raises(ValueError, match="detach_backbone_features"):
        _config(detach_backbone_features="true")
    with pytest.raises(ValueError, match="freeze_coarse_backbone"):
        _config(freeze_coarse_backbone="frozen")


def test_phase_three_locks_the_ordered_production_semantic_contract() -> None:
    config = _config()
    prior = SemanticPrior(config)

    assert COARSE_SEMANTIC_CLASS_NAMES == (
        "normal brain",
        "edema",
        "tumor-core candidate",
    )
    assert config.num_semantic_classes == NUM_COARSE_SEMANTIC_CLASSES == 3
    assert prior.semantic_head.out_channels == NUM_COARSE_SEMANTIC_CLASSES
    assert prior.semantic_head.weight.shape[0] == NUM_COARSE_SEMANTIC_CLASSES


@pytest.mark.parametrize("invalid_count", (2, 4, True, False))
def test_phase_three_rejects_nonproduction_semantic_counts(invalid_count: int) -> None:
    with pytest.raises(ValueError, match="exactly 3"):
        PointGuidedConfig(num_semantic_classes=invalid_count)


@pytest.mark.parametrize(
    ("spectral_tap", "feature_attribute"),
    (("conv1_pre_maxpool", "shallow"), ("layer1", "layer1")),
)
def test_selected_spectral_feature_matches_the_configured_shared_tap(
    spectral_tap: str,
    feature_attribute: str,
) -> None:
    prior = SemanticPrior(_config(spectral_tap=spectral_tap)).eval()
    volumes = torch.randn(1, 3, 9, 11, 13)

    with torch.no_grad():
        features = prior.extract_intermediate_features(volumes)
        selected = prior.select_spectral_feature(features)

    expected = getattr(features, feature_attribute)
    assert selected is not expected
    torch.testing.assert_close(selected, expected, rtol=0.0, atol=0.0)

    with pytest.raises(TypeError, match="MedicalNetFeatures"):
        prior.select_spectral_feature(torch.zeros(1))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("freeze_backbone", "detach_feature", "expects_selected_grad"),
    ((True, True, False), (True, False, True), (False, True, False), (False, False, True)),
)
def test_phase_two_freeze_and_detach_combinations_are_explicit(
    freeze_backbone: bool,
    detach_feature: bool,
    expects_selected_grad: bool,
) -> None:
    prior = SemanticPrior(
        _config(
            freeze_coarse_backbone=freeze_backbone,
            detach_backbone_features=detach_feature,
        )
    ).eval()
    volumes = torch.randn(1, 3, 9, 11, 13, requires_grad=True)

    features = prior.extract_intermediate_features(volumes)
    selected = prior.select_spectral_feature(features)

    assert all(parameter.requires_grad is not freeze_backbone for parameter in prior.backbone.parameters())
    assert selected.requires_grad is expects_selected_grad
    if detach_feature:
        assert selected is not features.shallow
        assert selected.grad_fn is None
    else:
        assert selected is features.shallow


def test_detached_selected_feature_blocks_backbone_but_not_downstream_or_semantic_gradients() -> None:
    prior = SemanticPrior(
        _config(freeze_coarse_backbone=False, detach_backbone_features=True)
    ).eval()
    volumes = torch.randn(1, 3, 9, 11, 13, requires_grad=True)
    features = prior.extract_intermediate_features(volumes)
    selected = prior.select_spectral_feature(features)
    downstream_probe = torch.nn.Conv3d(selected.shape[1], 1, kernel_size=1)

    assert features.shallow.requires_grad
    assert features.deep.requires_grad
    assert not selected.requires_grad
    assert selected.grad_fn is None

    downstream_probe(selected).square().mean().backward()
    assert downstream_probe.weight.grad is not None
    assert all(parameter.grad is None for parameter in prior.backbone.parameters())
    assert volumes.grad is None

    prior.zero_grad(set_to_none=True)
    semantic_loss = prior.head(features.deep).square().mean()
    semantic_loss.backward()
    assert prior.semantic_head.weight.grad is not None
    assert prior.backbone.conv1.weight.grad is not None
    assert volumes.grad is not None


def test_attached_selected_feature_reaches_a_fine_tuned_backbone() -> None:
    prior = SemanticPrior(
        _config(
            freeze_coarse_backbone=False,
            detach_backbone_features=False,
            spectral_tap="layer1",
        )
    ).eval()
    volumes = torch.randn(1, 3, 9, 11, 13, requires_grad=True)
    selected = prior.select_spectral_feature(prior.extract_intermediate_features(volumes))

    assert selected.requires_grad
    selected.square().mean().backward()

    assert prior.backbone.conv1.weight.grad is not None
    assert volumes.grad is not None


def test_attached_frozen_selected_feature_preserves_input_grad_without_backbone_grad() -> None:
    prior = SemanticPrior(
        _config(freeze_coarse_backbone=True, detach_backbone_features=False)
    ).eval()
    volumes = torch.randn(1, 3, 9, 11, 13, requires_grad=True)
    selected = prior.select_spectral_feature(prior.extract_intermediate_features(volumes))

    assert selected.requires_grad
    selected.square().mean().backward()

    assert volumes.grad is not None
    assert all(parameter.grad is None for parameter in prior.backbone.parameters())


@pytest.mark.parametrize(
    "overrides",
    (
        {"detach_backbone_features": False},
        {"spectral_tap": "layer1"},
        {"detach_backbone_features": False, "spectral_tap": "layer1"},
    ),
)
def test_spectral_controls_do_not_change_semantic_logits(
    overrides: dict[str, object],
) -> None:
    baseline = SemanticPrior(_config()).eval()
    variant = SemanticPrior(_config(**overrides)).eval()
    variant.load_state_dict(baseline.state_dict())
    volumes = torch.randn(1, 3, 9, 11, 13)

    with torch.no_grad():
        baseline_logits = baseline.forward_logits(volumes)
        variant_logits = variant.forward_logits(volumes)

    assert tuple(variant.state_dict()) == tuple(baseline.state_dict())
    torch.testing.assert_close(variant_logits, baseline_logits, rtol=0.0, atol=0.0)


def test_shared_semantic_and_selected_feature_path_uses_one_backbone_traversal() -> None:
    prior = SemanticPrior(_config()).eval()
    volumes = torch.randn(1, 3, 9, 11, 13)
    module_names = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4")
    calls = {name: 0 for name in module_names}

    def counter(name: str):
        def count_call(*_args: object) -> None:
            calls[name] += 1

        return count_call

    hooks = [
        getattr(prior.backbone, name).register_forward_hook(counter(name)) for name in module_names
    ]
    try:
        with torch.no_grad():
            features = prior.extract_intermediate_features(volumes)
            selected = prior.select_spectral_feature(features)
            semantic_logits = F.interpolate(
                prior.head(features.deep),
                size=volumes.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
    finally:
        for hook in hooks:
            hook.remove()

    assert selected.shape == features.shallow.shape
    assert semantic_logits.shape == (1, 3, 9, 11, 13)
    assert calls == {name: 1 for name in module_names}


def test_semantic_prior_returns_full_resolution_probability_volume() -> None:
    prior = SemanticPrior(_config()).eval()
    volumes = torch.randn(1, 3, 9, 11, 13)

    with torch.no_grad():
        probabilities = prior(volumes)

    assert probabilities.shape == (1, 3, 9, 11, 13)
    assert bool((probabilities >= 0.0).all())
    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones_like(probabilities[:, 0]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_semantic_prior_freezes_backbone_and_keeps_head_trainable() -> None:
    prior = SemanticPrior(_config())

    assert prior.backbone_is_frozen
    assert not prior.pretrained_loaded
    assert all(not parameter.requires_grad for parameter in prior.backbone.parameters())
    assert all(parameter.requires_grad for parameter in prior.semantic_head.parameters())

    prior.train()
    assert prior.training
    assert not prior.backbone.training
    assert prior.semantic_head.training

    batch_norm_state = [
        (
            module,
            module.running_mean.clone(),
            module.running_var.clone(),
            module.num_batches_tracked.clone(),
        )
        for module in prior.backbone.modules()
        if isinstance(module, torch.nn.BatchNorm3d)
    ]
    with torch.no_grad():
        prior(torch.randn(1, 3, 9, 11, 13))

    assert all(not module.training for module, *_state in batch_norm_state)
    for module, running_mean, running_var, batches_tracked in batch_norm_state:
        assert torch.equal(module.running_mean, running_mean)
        assert torch.equal(module.running_var, running_var)
        assert torch.equal(module.num_batches_tracked, batches_tracked)

    prior.zero_grad(set_to_none=True)
    logits = prior.forward_logits(torch.randn(1, 3, 9, 11, 13))
    logits.square().mean().backward()
    assert prior.semantic_head.weight.grad is not None
    assert all(parameter.grad is None for parameter in prior.backbone.parameters())


def test_checkpoint_adapts_stem_and_records_provenance(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "medicalnet_resnet10.pt"
    source_backbone = MedicalNetResNet10(in_channels=1)
    with torch.no_grad():
        source_backbone.conv1.weight.copy_(
            torch.arange(source_backbone.conv1.weight.numel(), dtype=torch.float32).reshape_as(
                source_backbone.conv1.weight
            )
        )
    torch.save({"state_dict": source_backbone.state_dict()}, checkpoint_path)
    expected_sha256 = sha256_file(checkpoint_path)

    prior = SemanticPrior(
        _config(
            medicalnet_checkpoint_path=checkpoint_path,
            medicalnet_checkpoint_sha256=expected_sha256,
        )
    )

    expected_weight = source_backbone.conv1.weight.repeat(1, 3, 1, 1, 1) / 3.0
    assert torch.equal(prior.backbone.conv1.weight, expected_weight)
    assert prior.checkpoint_loaded
    assert not prior.pretrained_loaded
    assert prior.checkpoint_provenance is not None
    assert prior.backbone_provenance is prior.checkpoint_provenance
    assert prior.checkpoint_provenance.sha256 == expected_sha256
    assert prior.checkpoint_provenance.source_input_channels == 1
    assert prior.checkpoint_provenance.adapted_input_channels == 3
    assert prior.checkpoint_provenance.input_conv_adapted
    assert prior.checkpoint_provenance.integrity_verified
    assert not prior.checkpoint_provenance.official_pretrained_verified


def test_checkpoint_hash_and_load_bind_to_an_immutable_byte_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "medicalnet_resnet10.pt"
    torch.save({"state_dict": MedicalNetResNet10(in_channels=1).state_dict()}, checkpoint_path)
    expected_sha256 = sha256_file(checkpoint_path)
    original_open = Path.open
    opens: list[Path] = []

    def tracked_open(path: Path, *args: object, **kwargs: object):
        if path == checkpoint_path:
            opens.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    prior = SemanticPrior(
        _config(
            medicalnet_checkpoint_path=checkpoint_path,
            medicalnet_checkpoint_sha256=expected_sha256,
        )
    )

    assert prior.checkpoint_loaded
    assert opens == [checkpoint_path]


def test_checkpoint_snapshot_is_not_affected_by_an_in_place_rewrite_after_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "medicalnet_resnet10.pt"
    torch.save({"state_dict": MedicalNetResNet10(in_channels=1).state_dict()}, checkpoint_path)
    expected_sha256 = sha256_file(checkpoint_path)
    original_hash_stream = medicalnet_module._sha256_stream

    def hash_then_rewrite(stream):
        digest = original_hash_stream(stream)
        torch.save({"state_dict": {"not_a_resnet10_key": torch.zeros(1)}}, checkpoint_path)
        return digest

    monkeypatch.setattr(medicalnet_module, "_sha256_stream", hash_then_rewrite)
    prior = SemanticPrior(
        _config(
            medicalnet_checkpoint_path=checkpoint_path,
            medicalnet_checkpoint_sha256=expected_sha256,
        )
    )

    assert prior.checkpoint_loaded


def test_checkpoint_rejects_sha_mismatch_and_missing_path(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "minimal.pt"
    torch.save({"state_dict": {"conv1.weight": torch.zeros(64, 1, 7, 7, 7)}}, checkpoint_path)

    with pytest.raises(MedicalNetCheckpointError, match="SHA-256 mismatch"):
        SemanticPrior(
            _config(
                medicalnet_checkpoint_path=checkpoint_path,
                medicalnet_checkpoint_sha256="0" * 64,
            )
        )

    with pytest.raises(FileNotFoundError, match="Automatic downloads are disabled"):
        SemanticPrior(_config(medicalnet_checkpoint_path=tmp_path / "does-not-exist.pt"))


def test_checkpoint_rejects_incomplete_backbone_state_dict(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "incomplete.pt"
    torch.save({"state_dict": {"conv1.weight": torch.zeros(64, 1, 7, 7, 7)}}, checkpoint_path)

    with pytest.raises(MedicalNetCheckpointError, match="missing keys"):
        SemanticPrior(_config(medicalnet_checkpoint_path=checkpoint_path))


def test_pretrained_mode_requires_an_approved_official_one_channel_digest(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "local_resnet10.pt"
    torch.save({"state_dict": MedicalNetResNet10(in_channels=1).state_dict()}, checkpoint_path)

    with pytest.raises(MedicalNetCheckpointError, match="official pretrained MedicalNet"):
        SemanticPrior(
            _config(
                medicalnet_checkpoint_path=checkpoint_path,
                medicalnet_checkpoint_sha256=sha256_file(checkpoint_path),
                require_pretrained_backbone=True,
            )
        )


def test_checkpoint_loader_refuses_unsafe_torch_load_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "unsupported.pt"
    checkpoint_path.write_bytes(b"not read because torch.load is mocked")
    calls: list[dict[str, object]] = []

    def unsupported_weights_only(*args: object, **kwargs: object) -> object:
        calls.append(dict(kwargs))
        raise TypeError("weights_only is unsupported")

    monkeypatch.setattr(torch, "load", unsupported_weights_only)
    with pytest.raises(MedicalNetCheckpointError, match="refusing unsafe checkpoint deserialization"):
        _load_checkpoint_object(checkpoint_path)
    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_backbone_layout_matches_an_independent_resnet10_manifest() -> None:
    """Check the upstream ResNet10 topology without generating a fixture from it."""

    backbone = MedicalNetResNet10(in_channels=1)
    assert backbone.layers == (1, 1, 1, 1)
    assert backbone.shortcut_type == "B"
    assert backbone.conv1.weight.shape == (64, 1, 7, 7, 7)
    assert backbone.layer1[0].conv1.weight.shape == (64, 64, 3, 3, 3)
    assert backbone.layer2[0].conv1.weight.shape == (128, 64, 3, 3, 3)
    assert backbone.layer2[0].downsample[0].weight.shape == (128, 64, 1, 1, 1)
    assert backbone.layer3[0].conv1.dilation == (2, 2, 2)
    assert backbone.layer3[0].conv1.stride == (1, 1, 1)
    assert backbone.layer4[0].conv1.dilation == (4, 4, 4)
    assert backbone.layer4[0].conv1.stride == (1, 1, 1)
