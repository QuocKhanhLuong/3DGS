"""Canonical provenance and compatibility identities for PFGR-Lite.

The PFGR implementation deliberately keeps model-producer compatibility
separate from value-model fitting/calibration identity.  The helpers in this
module are dependency free (apart from PyTorch) and are safe to import from
target-free inference code.  They hash canonical JSON and tensor bytes rather
than object representations or filesystem paths so a worker cannot silently
make a stale bank appear compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch
from torch import Tensor, nn


SCHEMA_VERSION = "pfgr-lite-provenance-v1"
"""Version of the canonical provenance envelope."""


def _jsonable(value: Any) -> Any:
    """Convert supported values to deterministic JSON-compatible objects."""

    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not torch.isfinite(torch.tensor(value, dtype=torch.float64)):
            raise ValueError("provenance cannot encode nonfinite floats")
        # repr is stable across supported Python versions for finite doubles.
        return repr(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"unsupported provenance value type: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return a sorted, compact JSON representation of ``value``."""

    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_digest(value: Any, *, prefix: str = "") -> str:
    """SHA-256 hash canonical JSON metadata."""

    payload = f"{prefix}{canonical_json(value)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tensor_digest(value: Tensor, *, name: str = "tensor") -> str:
    """Hash dtype, shape, and canonical contiguous tensor bytes.

    Tensor values are moved to CPU only for hashing; this helper never mutates
    or detaches the caller's graph-connected tensor.  The ``detach`` operation
    is intentionally restricted to metadata hashing and must not be used by
    state initialization or training code.
    """

    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point() and value.dtype not in (torch.bool, torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        raise TypeError(f"{name} has unsupported dtype {value.dtype}")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    cpu = value.detach().contiguous().cpu()
    header = canonical_json({"name": name, "dtype": str(cpu.dtype), "shape": tuple(cpu.shape)}).encode("utf-8")
    return hashlib.sha256(header + b"\0" + cpu.numpy().tobytes()).hexdigest()


def module_state_digest(module: nn.Module, *, include_buffers: bool = True) -> str:
    """Hash a module's named parameters and (optionally) buffers."""

    if not isinstance(module, nn.Module):
        raise TypeError("module must be an nn.Module")
    entries: list[tuple[str, str]] = []
    for name, parameter in sorted(module.named_parameters(), key=lambda item: item[0]):
        entries.append((f"parameter:{name}", tensor_digest(parameter, name=name)))
    if include_buffers:
        for name, buffer in sorted(module.named_buffers(), key=lambda item: item[0]):
            entries.append((f"buffer:{name}", tensor_digest(buffer, name=name)))
    return canonical_digest(entries, prefix="pfgr-lite-module-state-v1|")


def module_parameter_digest(module: nn.Module) -> str:
    """Hash only trainable/frozen parameter tensors, excluding buffers."""

    return module_state_digest(module, include_buffers=False)


def batchnorm_state_digest(module: nn.Module) -> str:
    """Hash BatchNorm affine and running-state tensors in a module."""

    if not isinstance(module, nn.Module):
        raise TypeError("module must be an nn.Module")
    entries: list[tuple[str, str]] = []
    for module_name, child in module.named_modules():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            for name, value in (
                ("weight", child.weight),
                ("bias", child.bias),
                ("running_mean", child.running_mean),
                ("running_var", child.running_var),
                ("num_batches_tracked", child.num_batches_tracked),
            ):
                if value is not None:
                    entries.append((f"{module_name}.{name}", tensor_digest(value, name=f"{module_name}.{name}")))
    return canonical_digest(entries, prefix="pfgr-lite-bn-state-v1|")


def best_effort_git_head(repository_root: str | Path | None = None) -> str | None:
    """Return a source SHA for reproduction metadata, never compatibility."""

    try:
        result = subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=None if repository_root is None else Path(repository_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return result or None


@dataclass(frozen=True)
class SourceProvenance:
    """Human/reproduction provenance, intentionally not producer compatibility."""

    schema_version: str = SCHEMA_VERSION
    source_sha: str | None = None
    config_sha: str | None = None
    implementation_version: str = "pfgr-lite-v1"
    model_family: str = "MedicalNet_ResNet10"
    source_input_channels: int = 3
    adapted_input_channels: int = 3
    input_conv_adapted: bool = False
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    checkpoint_integrity_verified: bool = False
    source_state_dict_key_count: int = 0
    loaded_backbone_key_count: int = 0
    adaptation_digest: str | None = None
    parameter_hash: str | None = None
    frozen_bn_hash: str | None = None
    official_pretrained_verified: bool = False
    synthetic_untrained: bool = True
    traversal_count: int = 0
    details: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"source provenance schema must be {SCHEMA_VERSION!r}")
        if self.source_input_channels not in (1, 3):
            raise ValueError("source_input_channels must be 1 or 3")
        if self.adapted_input_channels not in (1, 3):
            raise ValueError("adapted_input_channels must be 1 or 3")
        if self.input_conv_adapted and (self.source_input_channels != 1 or self.adapted_input_channels != 3):
            raise ValueError("input_conv_adapted requires a one-channel source adapted to three channels")
        if not isinstance(self.input_conv_adapted, bool) or not isinstance(self.official_pretrained_verified, bool):
            raise TypeError("input_conv_adapted and official_pretrained_verified must be bool")
        if not isinstance(self.checkpoint_integrity_verified, bool):
            raise TypeError("checkpoint_integrity_verified must be bool")
        if not isinstance(self.synthetic_untrained, bool):
            raise TypeError("synthetic_untrained must be bool")
        for name in ("source_state_dict_key_count", "loaded_backbone_key_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.adaptation_digest is not None and (not isinstance(self.adaptation_digest, str) or not self.adaptation_digest):
            raise ValueError("adaptation_digest must be a nonempty string or None")
        if not isinstance(self.traversal_count, int) or isinstance(self.traversal_count, bool) or self.traversal_count < 0:
            raise ValueError("traversal_count must be a nonnegative integer")
        if self.official_pretrained_verified and self.synthetic_untrained:
            raise ValueError("official pretrained provenance cannot be synthetic_untrained")

    @property
    def digest(self) -> str:
        return canonical_digest(self, prefix="pfgr-lite-source-provenance-v1|")

    @property
    def sha256(self) -> str | None:
        """Checkpoint field name used by ``MedicalNetCheckpointProvenance``."""

        return self.checkpoint_sha256

    @property
    def integrity_verified(self) -> bool:
        """Whether the configured checkpoint digest was actually verified."""

        return self.checkpoint_integrity_verified

    def as_dict(self) -> dict[str, Any]:
        payload = _jsonable(self)
        payload["sha256"] = self.checkpoint_sha256
        payload["integrity_verified"] = self.checkpoint_integrity_verified
        return payload


@dataclass(frozen=True)
class ProducerCompatibility:
    """Exact producer identity used by state, bank, and label matching.

    Value-network architecture/weights, fitting settings, calibration, policy,
    CLI, metrics, and a bare repository SHA are deliberately absent.  Source
    SHA/config are carried by :class:`SourceProvenance` instead.
    """

    schema_version: str = "pfgr-lite-producer-compat-v1"
    observation_normalization_hash: str = "unknown"
    geometry_query_version_hash: str = "unknown"
    medicalnet_provenance_hash: str = "unknown"
    frozen_bn_hash: str = "unknown"
    static_head_hash: str = "unknown"
    semantic_head_hash: str = "unknown"
    point_refiner_hash: str = "unknown"
    spectral_projector_hash: str = "unknown"
    state_initializer_hash: str = "unknown"
    updater_hash: str = "unknown"
    decoder_hash: str = "unknown"
    writer_hash: str = "unknown"
    candidate_geometry_hash: str = "unknown"
    label_definition_hash: str = "unknown"
    source_version: str = "pfgr-lite-v1"
    component_versions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != "pfgr-lite-producer-compat-v1":
            raise ValueError("unknown ProducerCompatibility schema_version")
        for name, value in self.__dict__.items():
            if name.endswith("_hash"):
                if not isinstance(value, str) or not value or value.lower() in {"unknown", "unset", "none", "null"}:
                    raise ValueError(f"{name} must be a complete non-sentinel hash")
        if not isinstance(self.source_version, str) or not self.source_version:
            raise ValueError("source_version must be a nonempty string")
        if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in self.component_versions):
            raise ValueError("component_versions must contain nonempty string pairs")

    @property
    def digest(self) -> str:
        # Keep this identity strictly scoped to state/proposal/label
        # producers.  Diagnostic component metadata may mention source or V
        # settings, but repository Git SHA and value-fit identities belong to
        # their separate provenance envelopes and must not stale a bank.
        scoped_versions = tuple(
            (key, value)
            for key, value in self.component_versions
            if key.lower() not in {
                "git",
                "git_sha",
                "source_sha",
                "config_sha",
                "repository",
                "value",
                "value_model",
                "v",
            }
        )
        payload = {
            "schema_version": self.schema_version,
            "observation_normalization_hash": self.observation_normalization_hash,
            "geometry_query_version_hash": self.geometry_query_version_hash,
            "medicalnet_provenance_hash": self.medicalnet_provenance_hash,
            "frozen_bn_hash": self.frozen_bn_hash,
            "static_head_hash": self.static_head_hash,
            "semantic_head_hash": self.semantic_head_hash,
            "point_refiner_hash": self.point_refiner_hash,
            "spectral_projector_hash": self.spectral_projector_hash,
            "state_initializer_hash": self.state_initializer_hash,
            "updater_hash": self.updater_hash,
            "decoder_hash": self.decoder_hash,
            "writer_hash": self.writer_hash,
            "candidate_geometry_hash": self.candidate_geometry_hash,
            "label_definition_hash": self.label_definition_hash,
            "source_version": self.source_version,
            "component_versions": scoped_versions,
        }
        return canonical_digest(payload, prefix="pfgr-lite-producer-compat-v1|")

    def matches(self, other: "ProducerCompatibility") -> bool:
        return isinstance(other, ProducerCompatibility) and self.digest == other.digest

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class ValueFitIdentity:
    """Identity for one V architecture/fit, separate from producer identity."""

    schema_version: str = "pfgr-lite-value-fit-v1"
    input_variant: int = 366
    architecture_hash: str = "unknown"
    weights_hash: str = "unknown"
    fit_config_hash: str = "unknown"
    bank_manifest_hash: str = "unknown"
    gain_scale_hash: str = "unknown"

    def __post_init__(self) -> None:
        if self.schema_version != "pfgr-lite-value-fit-v1":
            raise ValueError("unknown ValueFitIdentity schema_version")
        if self.input_variant not in (126, 222, 270, 366):
            raise ValueError("input_variant must be one of 126, 222, 270, 366")
        for name in ("architecture_hash", "weights_hash", "fit_config_hash", "bank_manifest_hash", "gain_scale_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.lower() in {"unknown", "unset", "none", "null"}:
                raise ValueError(f"{name} must be a complete non-sentinel hash")

    @property
    def digest(self) -> str:
        return canonical_digest(self, prefix="pfgr-lite-value-fit-v1|")


@dataclass(frozen=True)
class CalibrationIdentity:
    """Exact calibration binding for a producer and one fitted V."""

    producer_compatibility_hash: str
    value_fit_identity_hash: str
    version: str = "pfgr-lite-calibration-identity-v1"

    def __post_init__(self) -> None:
        if self.version != "pfgr-lite-calibration-identity-v1":
            raise ValueError("unknown calibration identity version")
        for name in ("producer_compatibility_hash", "value_fit_identity_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.lower() in {"unknown", "unset", "none", "null"}:
                raise ValueError("calibration identity hashes must be complete non-sentinel hashes")

    @property
    def digest(self) -> str:
        return canonical_digest(self, prefix="pfgr-lite-calibration-identity-v1|")


def source_provenance_from_semantic_prior(prior: nn.Module, *, source_sha: str | None = None, config_sha: str | None = None) -> SourceProvenance:
    """Build an honest provenance record from the existing semantic prior."""

    checkpoint = getattr(prior, "backbone_provenance", None)
    backbone = getattr(prior, "backbone", None)
    if checkpoint is None:
        # Checkpoint-less construction is an explicitly synthetic arm.  The
        # live backbone channel count is useful only to describe that local
        # initialization; it must never be presented as checkpoint evidence.
        source_channels = int(getattr(backbone, "in_channels", 3))
        adapted_channels = source_channels
        adapted = False
        official = False
        integrity_verified = False
    else:
        # A configured checkpoint is the sole authority for source/adaptation
        # channel provenance.  Do not silently substitute the instantiated
        # three-channel stem when a malformed checkpoint receipt omits fields.
        required_fields = (
            "source_input_channels",
            "adapted_input_channels",
            "input_conv_adapted",
            "official_pretrained_verified",
            "integrity_verified",
            "sha256",
        )
        missing = [name for name in required_fields if not hasattr(checkpoint, name)]
        if missing:
            raise ValueError(
                "checkpoint provenance is incomplete; missing " + ", ".join(missing)
            )
        source_channels = int(checkpoint.source_input_channels)
        adapted_channels = int(checkpoint.adapted_input_channels)
        adapted = checkpoint.input_conv_adapted
        official = checkpoint.official_pretrained_verified
        integrity_verified = checkpoint.integrity_verified
        if not isinstance(adapted, bool) or not isinstance(official, bool) or not isinstance(integrity_verified, bool):
            raise TypeError("checkpoint provenance boolean fields must be bool")
        if not isinstance(checkpoint.sha256, str) or not checkpoint.sha256:
            raise ValueError("checkpoint provenance sha256 must be complete")
    adaptation_digest = canonical_digest(
        {
            "algorithm": "repeat_divide_mean_stem_v1" if adapted else "identity_stem_v1",
            "source_input_channels": source_channels,
            "adapted_input_channels": adapted_channels,
            "input_conv_adapted": adapted,
            "checkpoint_sha256": None if checkpoint is None else checkpoint.sha256,
        },
        prefix="pfgr-lite-input-conv-adaptation-v1|",
    )
    return SourceProvenance(
        source_sha=source_sha,
        config_sha=config_sha,
        source_input_channels=source_channels,
        adapted_input_channels=adapted_channels,
        input_conv_adapted=adapted,
        checkpoint_path=getattr(checkpoint, "checkpoint_path", None),
        checkpoint_sha256=None if checkpoint is None else checkpoint.sha256,
        checkpoint_integrity_verified=integrity_verified,
        source_state_dict_key_count=int(getattr(checkpoint, "source_state_dict_key_count", 0)) if checkpoint is not None else 0,
        loaded_backbone_key_count=int(getattr(checkpoint, "loaded_backbone_key_count", 0)) if checkpoint is not None else 0,
        adaptation_digest=adaptation_digest,
        parameter_hash=module_parameter_digest(backbone) if isinstance(backbone, nn.Module) else None,
        frozen_bn_hash=batchnorm_state_digest(backbone) if isinstance(backbone, nn.Module) else None,
        official_pretrained_verified=official,
        # An arbitrary local/synthetic checkpoint is not official pretrained
        # evidence.  Keep it labelled synthetic/untrained unless an approved
        # official digest was actually verified.
        synthetic_untrained=not official,
        traversal_count=0,
    )


__all__ = [
    "CalibrationIdentity",
    "ProducerCompatibility",
    "SCHEMA_VERSION",
    "SourceProvenance",
    "ValueFitIdentity",
    "batchnorm_state_digest",
    "best_effort_git_head",
    "canonical_digest",
    "canonical_json",
    "module_parameter_digest",
    "module_state_digest",
    "source_provenance_from_semantic_prior",
    "tensor_digest",
]
