"""Coarse semantic prior built on the locked MedicalNet ResNet10 backbone."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import PointGuidedConfig
from .medicalnet_resnet10 import (
    MedicalNetCheckpointProvenance,
    MedicalNetResNet10,
    load_medicalnet_checkpoint,
)

__all__ = ["MedicalNetSemanticPrior", "SemanticPrior"]


class SemanticPrior(nn.Module):
    """Return full-resolution coarse semantic probabilities for T1/T2/FLAIR.

    The MedicalNet feature extractor is frozen by default according to
    :class:`PointGuidedConfig`.  Its only added learned component is the
    ``Conv3d(512, K, 1)`` semantic head; no decoder or reconstruction volume is
    present in this module.
    """

    def __init__(self, config: PointGuidedConfig) -> None:
        super().__init__()
        if not isinstance(config, PointGuidedConfig):
            raise TypeError("config must be a PointGuidedConfig")

        self.config = config
        self.backbone = MedicalNetResNet10(in_channels=3)
        self.semantic_head = nn.Conv3d(
            self.backbone.feature_channels,
            config.num_semantic_classes,
            kernel_size=1,
        )
        self.checkpoint_provenance: MedicalNetCheckpointProvenance | None = None
        # ``checkpoint_loaded`` means only that a local tensor checkpoint
        # passed strict shape validation. ``pretrained_loaded`` is reserved
        # for an allowlisted official upstream digest and stays false for an
        # arbitrary user-provided state dict.
        self.checkpoint_loaded = False
        self.pretrained_loaded = False

        if config.medicalnet_checkpoint_path is not None:
            self.checkpoint_provenance = load_medicalnet_checkpoint(
                self.backbone,
                config.medicalnet_checkpoint_path,
                expected_sha256=config.medicalnet_checkpoint_sha256,
                require_official_pretrained=config.require_pretrained_backbone,
            )
            self.checkpoint_loaded = True
            self.pretrained_loaded = self.checkpoint_provenance.official_pretrained_verified

        self._backbone_frozen = False
        self.set_backbone_frozen(config.freeze_coarse_backbone)

    @property
    def backbone_provenance(self) -> MedicalNetCheckpointProvenance | None:
        """Alias that makes checkpoint evidence explicit to callers."""

        return self.checkpoint_provenance

    @property
    def backbone_is_frozen(self) -> bool:
        """Whether backbone parameters and batch-normalization state are frozen."""

        return self._backbone_frozen

    @property
    def head(self) -> nn.Conv3d:
        """Read-only convenience alias for the minimal semantic head."""

        return self.semantic_head

    def set_backbone_frozen(self, frozen: bool) -> None:
        """Set the explicit frozen policy without changing the semantic head."""

        self._backbone_frozen = bool(frozen)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not self._backbone_frozen)
        if self._backbone_frozen:
            # Frozen BatchNorm layers must not update running statistics even
            # while the semantic head is trained.
            self.backbone.eval()
        else:
            self.backbone.train(self.training)

    def train(self, mode: bool = True) -> SemanticPrior:
        """Preserve frozen-backbone evaluation mode during head-only training."""

        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    @staticmethod
    def _validate_input(volumes: Tensor) -> None:
        if not isinstance(volumes, Tensor):
            raise TypeError("semantic prior expects a torch.Tensor input")
        if volumes.ndim != 5:
            raise ValueError("semantic prior input must have shape [B, 3, D, H, W]")
        if volumes.shape[1] != 3:
            raise ValueError(
                "semantic prior input channels are locked to [T1, T2, FLAIR] (three channels)"
            )
        if not volumes.is_floating_point() or not bool(torch.isfinite(volumes).all()):
            raise ValueError("semantic prior input must contain finite floating-point values")

    def extract_features(self, volumes: Tensor) -> Tensor:
        """Expose stride-eight backbone features without exposing a decoder."""

        self._validate_input(volumes)
        if self._backbone_frozen:
            with torch.no_grad():
                return self.backbone.forward_features(volumes)
        return self.backbone.forward_features(volumes)

    def forward_logits(self, volumes: Tensor) -> Tensor:
        """Return full-resolution semantic logits from the minimal head."""

        features = self.extract_features(volumes)
        coarse_logits = self.semantic_head(features)
        return F.interpolate(
            coarse_logits,
            size=volumes.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )

    def forward(self, volumes: Tensor) -> Tensor:
        """Return ``[B, K, D, H, W]`` soft semantic probabilities."""

        return F.softmax(self.forward_logits(volumes), dim=1)


# More specific spelling retained as a public alias for frontend composition.
MedicalNetSemanticPrior = SemanticPrior
