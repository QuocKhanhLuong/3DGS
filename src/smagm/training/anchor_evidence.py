"""Typed trainable projection from compact anchors to the fixed Gaussian head."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

import torch
from torch import nn

from ..anchors import AnchorBatch


@dataclass(frozen=True)
class AnchorEvidenceProjectorConfig:
    """Declared full-evidence to Gaussian-head feature mapping dimensions."""

    evidence_dim: int
    head_input_dim: int
    bias: bool = True

    def __post_init__(self) -> None:
        for name in ("evidence_dim", "head_input_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.bias, bool):
            raise TypeError("bias must be bool")

    @property
    def config_hash(self) -> str:
        payload = {
            "schema": "smagm-anchor-evidence-projector-config-v1",
            "evidence_dim": self.evidence_dim,
            "head_input_dim": self.head_input_dim,
            "bias": self.bias,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AnchorEvidenceProjectorReport:
    """Stable parameter accounting for later product telemetry."""

    config_hash: str
    evidence_dim: int
    head_input_dim: int
    parameter_count: int
    trainable_parameter_count: int


@dataclass(frozen=True)
class ProjectedAnchorEvidence:
    """Head-ready evidence while retaining anchor legality and modality metadata."""

    anchor_ids: tuple[str, ...]
    feature_vectors: torch.Tensor  # [N, Gaussian-head input channels]
    modality_ids: tuple[str, ...]
    appearance_valid: torch.Tensor  # [N, M], bool; copied only by reference
    source_evidence_hash: str
    projector_config_hash: str
    contributing_observation_ids: tuple[tuple[str, ...], ...]
    contributing_plane_hashes: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        count = len(self.anchor_ids)
        if count <= 0 or len(set(self.anchor_ids)) != count:
            raise ValueError("projected anchor evidence requires unique non-empty anchor IDs")
        if (
            not isinstance(self.feature_vectors, torch.Tensor)
            or self.feature_vectors.ndim != 2
            or self.feature_vectors.shape[0] != count
            or self.feature_vectors.shape[1] <= 0
            or not bool(torch.isfinite(self.feature_vectors).all())
        ):
            raise ValueError("projected feature_vectors must be finite with shape [N, C]")
        if (
            not isinstance(self.appearance_valid, torch.Tensor)
            or self.appearance_valid.dtype is not torch.bool
            or self.appearance_valid.ndim != 2
            or self.appearance_valid.shape[0] != count
            or self.appearance_valid.shape[1] != len(self.modality_ids)
        ):
            raise ValueError("appearance_valid must preserve one bool column per modality")
        if len(self.modality_ids) == 0 or len(set(self.modality_ids)) != len(self.modality_ids):
            raise ValueError("modality_ids must be unique and non-empty")
        if any(len(values) != count for values in (
            self.contributing_observation_ids,
            self.contributing_plane_hashes,
        )):
            raise ValueError("projected anchor provenance must match anchor count")
        for digest in (self.source_evidence_hash, self.projector_config_hash):
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("projected anchor evidence hashes must be SHA-256 digests")


class AnchorEvidenceProjector(nn.Module):
    """Learn a per-anchor map from all compact evidence channels to head inputs.

    This module intentionally has no topology, aggregation, or modality-routing
    role.  It preserves the legal anchor metadata verbatim and only projects the
    already context-derived compact evidence tensor.
    """

    def __init__(self, config: AnchorEvidenceProjectorConfig) -> None:
        super().__init__()
        if not isinstance(config, AnchorEvidenceProjectorConfig):
            raise TypeError("config must be an AnchorEvidenceProjectorConfig")
        self.config = config
        self.projection = nn.Linear(
            config.evidence_dim,
            config.head_input_dim,
            bias=config.bias,
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @property
    def parameter_report(self) -> AnchorEvidenceProjectorReport:
        return AnchorEvidenceProjectorReport(
            config_hash=self.config.config_hash,
            evidence_dim=self.config.evidence_dim,
            head_input_dim=self.config.head_input_dim,
            parameter_count=self.parameter_count,
            trainable_parameter_count=self.trainable_parameter_count,
        )

    def forward(self, anchors: AnchorBatch) -> ProjectedAnchorEvidence:
        if not isinstance(anchors, AnchorBatch):
            raise TypeError("anchor evidence projection requires an AnchorBatch")
        if anchors.evidence.shape[1] != self.config.evidence_dim:
            raise ValueError(
                "anchor evidence channels disagree with the declared projector input: "
                f"expected {self.config.evidence_dim}, got {anchors.evidence.shape[1]}"
            )
        if anchors.evidence.dtype not in (torch.float32, torch.float64):
            raise TypeError("anchor evidence projection requires float32 or float64 evidence")
        parameter = self.projection.weight
        if parameter.device != anchors.evidence.device or parameter.dtype != anchors.evidence.dtype:
            raise ValueError("anchor evidence and projector parameters must share device and dtype")
        feature_vectors = self.projection(anchors.evidence)
        return ProjectedAnchorEvidence(
            anchor_ids=anchors.anchor_ids,
            feature_vectors=feature_vectors,
            modality_ids=anchors.modality_ids,
            appearance_valid=anchors.appearance_valid,
            source_evidence_hash=anchors.evidence_hash,
            projector_config_hash=self.config.config_hash,
            contributing_observation_ids=anchors.geometry.contributing_observation_ids,
            contributing_plane_hashes=anchors.geometry.contributing_plane_hashes,
        )
