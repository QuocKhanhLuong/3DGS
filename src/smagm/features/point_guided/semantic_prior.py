"""Coarse semantic prior built on the locked MedicalNet ResNet10 backbone."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import PointGuidedConfig
from .medicalnet_resnet10 import (
    MedicalNetCheckpointProvenance,
    MedicalNetFeatures,
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

    @property
    def selected_spectral_feature_channels(self) -> int:
        """Return the actual channel count of the configured shared tap.

        This intentionally reads the instantiated MedicalNet modules instead
        of encoding the current 64-channel coincidence in frontend
        composition.  A checkpoint changes values but never this structural
        contract.
        """

        if self.config.spectral_tap == "conv1_pre_maxpool":
            return int(self.backbone.conv1.out_channels)
        if self.config.spectral_tap == "layer1":
            if not self.backbone.layer1:
                raise RuntimeError("MedicalNet layer1 must contain a final block")
            final_block = self.backbone.layer1[-1]
            final_conv = getattr(final_block, "conv2", None)
            if not isinstance(final_conv, nn.Conv3d):
                raise RuntimeError("MedicalNet layer1 final block must expose Conv3d conv2")
            return int(final_conv.out_channels)
        raise RuntimeError(f"unsupported spectral_tap: {self.config.spectral_tap!r}")

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

    def extract_intermediate_features(self, volumes: Tensor) -> MedicalNetFeatures:
        """Expose shared backbone maps from one pass without exposing a decoder.

        Freezing disables gradients for backbone parameters and fixes its batch
        normalization state, but does not implicitly detach its output maps.
        The configured selected-feature boundary owns that separate decision.
        """

        self._validate_input(volumes)
        return self.backbone.forward_intermediate_features(volumes)

    def extract_features(self, volumes: Tensor) -> Tensor:
        """Expose the compatible final semantic feature from the shared maps."""

        return self.extract_intermediate_features(volumes).deep

    def select_spectral_feature(self, features: MedicalNetFeatures) -> Tensor:
        """Select the configured shared branch feature at the explicit detach boundary.

        This method consumes an already-computed :class:`MedicalNetFeatures`, so
        selecting a feature never traverses MedicalNet a second time.  It never
        modifies the shared maps or the deep semantic feature.
        """

        if not isinstance(features, MedicalNetFeatures):
            raise TypeError("features must be a MedicalNetFeatures instance")

        if self.config.spectral_tap == "conv1_pre_maxpool":
            selected = features.shallow
        elif self.config.spectral_tap == "layer1":
            selected = features.layer1
        else:  # Defensive fail-closed guard for a malformed externally mutated config.
            raise RuntimeError(f"unsupported spectral_tap: {self.config.spectral_tap!r}")

        if self.config.detach_backbone_features:
            return selected.detach()
        return selected

    @staticmethod
    def _output_shape_dhw(output_spatial_shape: Sequence[int]) -> tuple[int, int, int]:
        shape = tuple(output_spatial_shape)
        if len(shape) != 3 or any(
            not isinstance(length, int) or isinstance(length, bool) or length <= 0
            for length in shape
        ):
            raise ValueError("output_spatial_shape must contain three positive DHW integers")
        return shape  # type: ignore[return-value]

    def forward_logits_from_intermediate_features(
        self,
        features: MedicalNetFeatures,
        *,
        output_spatial_shape: Sequence[int],
    ) -> Tensor:
        """Apply the semantic head to an already-computed shared deep map.

        Keeping this operation separate from feature extraction allows frontend
        composition to feed the semantic and static-base-plane branches from
        exactly one MedicalNet traversal.
        """

        if not isinstance(features, MedicalNetFeatures):
            raise TypeError("features must be a MedicalNetFeatures instance")
        deep = features.deep
        if not isinstance(deep, Tensor) or deep.ndim != 5:
            raise ValueError("features.deep must be a rank-5 torch.Tensor")
        if deep.shape[1] != self.semantic_head.in_channels:
            raise ValueError("features.deep channels must match the semantic head input")
        if not deep.is_floating_point() or not bool(torch.isfinite(deep).all()):
            raise ValueError("features.deep must contain finite floating-point values")
        return F.interpolate(
            self.semantic_head(deep),
            size=self._output_shape_dhw(output_spatial_shape),
            mode="trilinear",
            align_corners=False,
        )

    def forward_from_intermediate_features(
        self,
        features: MedicalNetFeatures,
        *,
        output_spatial_shape: Sequence[int],
    ) -> Tensor:
        """Return full-resolution probabilities from one shared feature bundle."""

        return F.softmax(
            self.forward_logits_from_intermediate_features(
                features,
                output_spatial_shape=output_spatial_shape,
            ),
            dim=1,
        )

    def forward_logits(self, volumes: Tensor) -> Tensor:
        """Return full-resolution semantic logits from the minimal head."""

        features = self.extract_intermediate_features(volumes)
        return self.forward_logits_from_intermediate_features(
            features,
            output_spatial_shape=volumes.shape[-3:],
        )

    def forward(self, volumes: Tensor) -> Tensor:
        """Return ``[B, K, D, H, W]`` soft semantic probabilities."""

        features = self.extract_intermediate_features(volumes)
        return self.forward_from_intermediate_features(
            features,
            output_spatial_shape=volumes.shape[-3:],
        )


# More specific spelling retained as a public alias for frontend composition.
MedicalNetSemanticPrior = SemanticPrior
