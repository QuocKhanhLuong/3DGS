"""CPU-only invariants for the locked MedicalNet semantic prior."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from smagm.features.point_guided.config import PointGuidedConfig
from smagm.features.point_guided.medicalnet_resnet10 import (
    MedicalNetCheckpointError,
    MedicalNetResNet10,
    _load_checkpoint_object,
    sha256_file,
)
from smagm.features.point_guided import medicalnet_resnet10 as medicalnet_module
from smagm.features.point_guided.semantic_prior import SemanticPrior


def _config(**overrides: object) -> PointGuidedConfig:
    return PointGuidedConfig(num_semantic_classes=3, **overrides)


def test_semantic_prior_returns_full_resolution_probability_volume() -> None:
    prior = SemanticPrior(_config()).eval()
    volumes = torch.randn(1, 3, 9, 11, 13)

    with torch.no_grad():
        probabilities = prior(volumes)

    assert probabilities.shape == (1, 3, 9, 11, 13)
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
