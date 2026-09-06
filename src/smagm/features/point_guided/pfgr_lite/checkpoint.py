"""Strict PFGR-Lite inference and resume serialization.

Artifacts are plain tensor/mapping payloads loaded with ``weights_only=True``
where supported.  Legacy checkpoints require an explicitly named adapter and
are never routed through the PFGR loader implicitly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Callable

import torch
from torch import Tensor
import numpy as np

from .calibration import CalibrationEvidence, attach_calibration_evidence, calibration_evidence
from .config import PFGRLiteConfig, frontend_config_from_dict, frontend_config_to_dict
from .provenance import ProducerCompatibility, SourceProvenance, batchnorm_state_digest, canonical_digest, module_parameter_digest, module_state_digest, source_provenance_from_semantic_prior, tensor_digest
from .types import (
    RESUME_SCHEMA,
    GainCalibration,
    InferenceBundle,
    ResumeState,
    StageState,
    ProducerDependencies,
    TrainingRoleManifest,
)
from .provenance import ValueFitIdentity


CHECKPOINT_FORMAT = "pfgr-lite-checkpoint-payload-v1"
RESUME_FORMAT = "pfgr-lite-resume-payload-v1"
CHECKPOINT_CONFIG_SCHEMA = "pfgr-lite-checkpoint-config-v1"
VALUE_ARTIFACT_FORMAT = "point-guided-pfgr-lite-value-v1"
VALUE_ARTIFACT_SCHEMA = VALUE_ARTIFACT_FORMAT
_GAIN_SCALE_KEYS = {
    "schema_version",
    "scale",
    "quantile",
    "method",
    "floor",
    "floor_applied",
    "training_role",
    "training_row_hash",
    "training_row_count",
}
_VALUE_FIT_CONFIG_SCHEMA = "pfgr-lite-value-fit-config-v1"
_VALUE_COMPLETION_SCHEMA = "pfgr-lite-value-completion-v1"


def _validate_value_fit_envelopes(
    *,
    identity: ValueFitIdentity,
    producer: ProducerDependencies,
    role_manifest: TrainingRoleManifest | None,
    config: Mapping[str, Any],
    completion: Mapping[str, Any],
    stage_provenance: Mapping[str, Any] | None,
) -> None:
    """Validate W3 fit options/completion when an artifact is release-grade.

    Engineering fixtures may retain a deliberately tiny opaque config.  Any
    artifact marked with the versioned fit envelope (and every non-synthetic
    source) is checked against the actual V identity, bank, scale, producer,
    role and completion state on both save and load.
    """

    production = not producer.source_provenance.synthetic_untrained
    if production and (
        not producer.source_provenance.official_pretrained_verified
        or not producer.source_provenance.checkpoint_integrity_verified
        or role_manifest is None
        or stage_provenance is None
    ):
        raise ValueError("production value artifacts require verified source, role and stage provenance")
    if config.get("schema_version") == _VALUE_FIT_CONFIG_SCHEMA:
        required = {
            "schema_version",
            "model_architecture",
            "options",
            "bank_manifest_hash",
            "gain_scale_hash",
            "producer_compatibility_hash",
            "role_manifest_hash",
            "producer_config",
        }
        if set(config) != required:
            raise ValueError("value fit config keys are incomplete or unknown")
        if config["model_architecture"] != identity.architecture_hash:
            raise ValueError("value fit config architecture does not match ValueFitIdentity")
        if config["bank_manifest_hash"] != identity.bank_manifest_hash:
            raise ValueError("value fit config bank identity does not match ValueFitIdentity")
        if config["gain_scale_hash"] != identity.gain_scale_hash:
            raise ValueError("value fit config scale identity does not match ValueFitIdentity")
        if config["producer_compatibility_hash"] != producer.compatibility_hash:
            raise ValueError("value fit config producer identity does not match producer")
        role_hash = None if role_manifest is None else role_manifest.digest
        if config["role_manifest_hash"] != role_hash:
            raise ValueError("value fit config role identity does not match role manifest")
        options = config["options"]
        if not isinstance(options, Mapping):
            raise TypeError("value fit config options must be a mapping")
        option_keys = {"batch_size", "seed", "learning_rate", "weight_decay", "loss", "shuffle", "robust_ablation", "optimizer"}
        if set(options) != option_keys:
            raise ValueError("value fit config options are incomplete or unknown")
        if canonical_digest(
            {
                "schema_version": _VALUE_FIT_CONFIG_SCHEMA,
                "model_architecture": identity.architecture_hash,
                "options": dict(options),
            },
            prefix="pfgr-lite-value-fit-config-v1|",
        ) != identity.fit_config_hash:
            raise ValueError("value fit config hash does not match ValueFitIdentity")
    elif production:
        raise ValueError("production value artifacts require the versioned value fit config envelope")
    if completion.get("schema_version") == _VALUE_COMPLETION_SCHEMA:
        required = {
            "schema_version",
            "complete",
            "stage",
            "bank_manifest_hash",
            "fit_config_hash",
            "producer_compatibility_hash",
            "gain_scale_hash",
        }
        if set(completion) != required:
            raise ValueError("value completion keys are incomplete or unknown")
        if completion["complete"] is not True or completion["stage"] != "value_fit":
            raise ValueError("value completion receipt must record completed value_fit")
        if completion["bank_manifest_hash"] != identity.bank_manifest_hash or completion["fit_config_hash"] != identity.fit_config_hash or completion["gain_scale_hash"] != identity.gain_scale_hash or completion["producer_compatibility_hash"] != producer.compatibility_hash:
            raise ValueError("value completion receipt identity does not match artifact")
    elif production:
        raise ValueError("production value artifacts require the versioned completion receipt")


def _complete_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.lower() in {"unknown", "unset", "none", "null"}:
        raise ValueError(f"{name} must be a complete non-sentinel string")
    return value


def _optional_complete_text(value: object, name: str) -> str | None:
    """Nullable identity used by explicit pre-V/static checkpoint stages."""

    if value is None:
        return None
    return _complete_text(value, name)


def _reject_secret_keys(key: object) -> None:
    if not isinstance(key, str):
        raise TypeError("serialized mapping keys must be strings")
    lower = key.lower()
    if any(token in lower for token in ("target", "oracle", "segmentation", "teacher")):
        raise ValueError("target/oracle/teacher state is forbidden in PFGR artifacts")


def _safe_value(value: Any, *, path: str) -> Any:
    """Convert resume metadata to a ``weights_only``-safe payload."""

    if isinstance(value, Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"nonfinite tensor at {path}")
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(f"object NumPy arrays are unsupported at {path}")
        return {
            "__pfgr_numpy_array__": True,
            "dtype": str(value.dtype),
            "shape": tuple(int(item) for item in value.shape),
            "data": torch.as_tensor(value.copy()),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise TypeError(f"serialized mapping keys at {path} must be strings or integer optimizer IDs")
            if isinstance(key, str):
                _reject_secret_keys(key)
            result[key] = _safe_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, tuple):
        return tuple(_safe_value(item, path=f"{path}[]") for item in value)
    if isinstance(value, list):
        return [_safe_value(item, path=f"{path}[]") for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not torch.isfinite(torch.tensor(value, dtype=torch.float64)):
            raise ValueError(f"nonfinite float at {path}")
        return value
    raise TypeError(f"unsupported checkpoint value at {path}: {type(value).__name__}")


def _restore_value(value: Any, *, path: str) -> Any:
    """Restore safe NumPy RNG array wrappers after weights-only loading."""

    if isinstance(value, Mapping):
        if value.get("__pfgr_numpy_array__") is True:
            if set(value) != {"__pfgr_numpy_array__", "dtype", "shape", "data"} or not isinstance(value["data"], Tensor):
                raise ValueError(f"malformed NumPy array wrapper at {path}")
            dtype = np.dtype(value["dtype"])
            shape = tuple(int(item) for item in value["shape"])
            array = value["data"].detach().cpu().numpy().astype(dtype, copy=False)
            if tuple(array.shape) != shape:
                raise ValueError(f"NumPy array shape mismatch at {path}")
            return array.copy()
        return {key: _restore_value(item, path=f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_restore_value(item, path=f"{path}[]") for item in value)
    if isinstance(value, list):
        return [_restore_value(item, path=f"{path}[]") for item in value]
    return value


def _validate_loaded_value(value: Any, *, path: str) -> None:
    """Recheck recursively loaded resume metadata before exposing it."""

    if isinstance(value, Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"nonfinite tensor at {path}")
        return
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(f"object NumPy arrays are unsupported at {path}")
        if value.dtype.kind in {"f", "c"} and not bool(np.isfinite(value).all()):
            raise ValueError(f"nonfinite NumPy array at {path}")
        return
    if isinstance(value, Mapping):
        if value.get("__pfgr_numpy_array__") is True:
            if set(value) != {"__pfgr_numpy_array__", "dtype", "shape", "data"}:
                raise ValueError(f"malformed NumPy array wrapper at {path}")
            if not isinstance(value["data"], Tensor):
                raise TypeError(f"NumPy array wrapper data at {path} must be a tensor")
            return
        for key, item in value.items():
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise TypeError(f"serialized mapping keys at {path} must be strings or integer optimizer IDs")
            if isinstance(key, str):
                _reject_secret_keys(key)
            _validate_loaded_value(item, path=f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _validate_loaded_value(item, path=f"{path}[]")
        return
    if value is None or isinstance(value, (str, int, bool)):
        return
    if isinstance(value, float) and torch.isfinite(torch.tensor(value, dtype=torch.float64)):
        return
    raise TypeError(f"unsupported loaded resume value at {path}: {type(value).__name__}")


def _state_dict_digest(state_dict: Mapping[str, Tensor]) -> str:
    return canonical_digest(
        [(name, tensor_digest(state_dict[name], name=name)) for name in sorted(state_dict)],
        prefix="pfgr-lite-checkpoint-state-v1|",
    )


def _module_state_digest_from_mapping(state_dict: Mapping[str, Tensor]) -> str:
    """Recompute the W3 ``module_state_digest`` for a tensor-only V state.

    ValueNet has parameters only (no running-state buffers).  Keeping this
    tiny reconstruction local avoids importing the value fitting module into
    the checkpoint boundary while still detecting a replaced V tensor.
    """

    entries = [
        (f"parameter:{name}", tensor_digest(state_dict[name], name=name))
        for name in sorted(state_dict)
    ]
    return canonical_digest(entries, prefix="pfgr-lite-module-state-v1|")


def _strict_value_state(state_dict: Mapping[str, Tensor], identity: ValueFitIdentity) -> None:
    """Construct the locked W3 head and perform an exact state load.

    The import is deliberately local: checkpoint metadata remains usable in a
    target-free process without allocating a ValueNet unless a V artifact is
    actually being validated.
    """

    from .value_net import SignedValueNet

    model = SignedValueNet(identity.input_variant)
    try:
        model.load_state_dict(dict(state_dict), strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("value artifact state_dict is not the locked SignedValueNet architecture") from exc
    if model.architecture_hash != identity.architecture_hash:
        raise ValueError("value artifact architecture identity does not match SignedValueNet")
    if module_state_digest(model) != identity.weights_hash:
        raise ValueError("value artifact V weights do not match ValueFitIdentity")


def _source_from_dict(values: Mapping[str, Any]) -> SourceProvenance:
    allowed = {field.name for field in fields(SourceProvenance)}
    unknown = set(values) - allowed - {"sha256", "integrity_verified"}
    if unknown:
        raise ValueError(f"unknown source provenance keys: {sorted(unknown)}")
    parsed = {key: values[key] for key in allowed if key in values}
    if "details" in parsed:
        parsed["details"] = tuple(tuple(item) for item in parsed["details"])
    return SourceProvenance(**parsed)


def _compatibility_from_dict(values: Mapping[str, Any]) -> ProducerCompatibility:
    allowed = {field.name for field in fields(ProducerCompatibility)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown producer compatibility keys: {sorted(unknown)}")
    parsed = dict(values)
    if "component_versions" in parsed:
        parsed["component_versions"] = tuple(tuple(item) for item in parsed["component_versions"])
    return ProducerCompatibility(**parsed)


def _producer_from_dict(values: Mapping[str, Any]) -> ProducerDependencies:
    allowed = {field.name for field in fields(ProducerDependencies)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown producer dependency keys: {sorted(unknown)}")
    parsed = dict(values)
    if not isinstance(parsed.get("compatibility"), Mapping) or not isinstance(parsed.get("source_provenance"), Mapping):
        raise TypeError("serialized producer requires compatibility and source_provenance mappings")
    parsed["compatibility"] = _compatibility_from_dict(parsed["compatibility"])
    parsed["source_provenance"] = _source_from_dict(parsed["source_provenance"])
    return ProducerDependencies(**parsed)


def _calibration_from_dict(values: Mapping[str, Any]) -> GainCalibration:
    allowed = {field.name for field in fields(GainCalibration)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown calibration keys: {sorted(unknown)}")
    return GainCalibration(**dict(values))


def _normalise_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a strict PFGR + legacy-frontend sidecar envelope."""

    if not isinstance(config, Mapping):
        raise TypeError("checkpoint config must be a mapping")
    required = {
        "schema_version",
        "pfgr_config",
        "frontend_config",
        "stage",
        "split_roles",
        "value_fit_identity_hash",
        "gain_scale_hash",
        "effective_policy_hash",
    }
    unknown = set(config) - required
    missing = required - set(config)
    if unknown or missing:
        raise ValueError(f"checkpoint config keys are incomplete or unknown: unknown={sorted(unknown)}, missing={sorted(missing)}")
    if config.get("schema_version") != CHECKPOINT_CONFIG_SCHEMA:
        raise ValueError("unknown checkpoint config schema")
    pfgr_raw = config.get("pfgr_config")
    frontend_raw = config.get("frontend_config")
    if not isinstance(pfgr_raw, Mapping) or not isinstance(frontend_raw, Mapping):
        raise TypeError("checkpoint config requires pfgr_config and frontend_config mappings")
    parsed_pfgr = PFGRLiteConfig.from_dict(pfgr_raw)
    parsed_frontend = frontend_config_from_dict(frontend_raw)
    stage = _complete_text(config.get("stage"), "stage")
    roles = config.get("split_roles")
    if not isinstance(roles, Mapping):
        raise TypeError("split_roles must be a mapping")
    expected_roles = {"producer_fit", "calibration_fit", "calibration_allowance"}
    if set(roles) != expected_roles:
        raise ValueError("split_roles must name producer_fit, calibration_fit, and calibration_allowance")
    role_values = {name: _complete_text(roles[name], f"split_roles.{name}") for name in expected_roles}
    return {
        "schema_version": CHECKPOINT_CONFIG_SCHEMA,
        "pfgr_config": parsed_pfgr.as_dict(),
        "frontend_config": frontend_config_to_dict(parsed_frontend),
        "stage": stage,
        "split_roles": role_values,
        # S0/pre-V and static snapshots carry explicit nulls.  Learned policy
        # loaders/checkpoint capabilities enforce concrete identities later;
        # no caller is asked to invent placeholder hashes before V exists.
        "value_fit_identity_hash": _optional_complete_text(config.get("value_fit_identity_hash"), "value_fit_identity_hash"),
        "gain_scale_hash": _optional_complete_text(config.get("gain_scale_hash"), "gain_scale_hash"),
        "effective_policy_hash": _optional_complete_text(config.get("effective_policy_hash"), "effective_policy_hash"),
    }


@dataclass(frozen=True)
class ValueArtifact:
    """Strict value-only fit artifact returned by :func:`load_value_artifact`.

    The payload contains only V parameters; producer tensors never get
    duplicated here.  Producer/source/role/config records are immutable
    identity envelopes used to join the artifact with an inference snapshot.
    """

    state_dict: Mapping[str, Tensor]
    value_fit_identity: ValueFitIdentity
    gain_scale: Mapping[str, Any]
    producer: ProducerDependencies
    source_provenance: SourceProvenance
    config: Mapping[str, Any]
    role_manifest: TrainingRoleManifest | None
    completion: Mapping[str, Any]
    stage_provenance: Mapping[str, Any] | None = None
    schema_version: str = VALUE_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != VALUE_ARTIFACT_SCHEMA:
            raise ValueError("unknown value artifact schema")
        if not isinstance(self.value_fit_identity, ValueFitIdentity):
            raise TypeError("value artifact requires ValueFitIdentity")
        if not isinstance(self.producer, ProducerDependencies):
            raise TypeError("value artifact requires ProducerDependencies")
        if not isinstance(self.source_provenance, SourceProvenance):
            raise TypeError("value artifact requires SourceProvenance")
        if not isinstance(self.config, Mapping) or not self.config:
            raise ValueError("value artifact config must be a nonempty mapping")
        if self.role_manifest is not None and not isinstance(self.role_manifest, TrainingRoleManifest):
            raise TypeError("role_manifest must be TrainingRoleManifest or None")
        if not isinstance(self.completion, Mapping) or not self.completion:
            raise ValueError("value artifact completion must be a nonempty mapping")
        if self.stage_provenance is not None and not isinstance(self.stage_provenance, Mapping):
            raise TypeError("stage_provenance must be a mapping or None")
        if not isinstance(self.state_dict, Mapping) or not self.state_dict:
            raise ValueError("value artifact state_dict must be nonempty")
        for name, value in self.state_dict.items():
            if not isinstance(name, str) or not name:
                raise TypeError("value artifact state keys must be nonempty strings")
            if any(token in name.lower() for token in ("backbone", "frontend", "decoder", "updater", "target", "oracle", "teacher")):
                raise ValueError("value artifact may contain only V tensors; producer tensors are forbidden")
            if not isinstance(value, Tensor) or value.numel() == 0:
                raise TypeError("value artifact state values must be nonempty tensors")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError("value artifact state tensors must be finite")
        _strict_value_state(self.state_dict, self.value_fit_identity)
        if not isinstance(self.gain_scale, Mapping):
            raise TypeError("gain_scale must be a mapping")
        scale = dict(self.gain_scale)
        if set(scale) != _GAIN_SCALE_KEYS | {"digest"}:
            raise ValueError("gain_scale keys are incomplete or unknown")
        if scale.get("schema_version") != "point-guided-pfgr-lite-gain-scale-v1":
            raise ValueError("unknown gain-scale provenance schema")
        digest = scale.pop("digest", None)
        if not isinstance(digest, str) or not digest:
            raise ValueError("gain_scale must carry its canonical digest")
        if canonical_digest(scale, prefix="pfgr-lite-gain-scale-v1|") != digest:
            raise ValueError("gain scale digest mismatch")
        if digest != self.value_fit_identity.gain_scale_hash:
            raise ValueError("gain scale does not match ValueFitIdentity")
        _validate_value_fit_envelopes(
            identity=self.value_fit_identity,
            producer=self.producer,
            role_manifest=self.role_manifest,
            config=self.config,
            completion=self.completion,
            stage_provenance=self.stage_provenance,
        )

    @property
    def weights_hash(self) -> str:
        return self.value_fit_identity.weights_hash

    def as_dict(self) -> dict[str, Any]:
        producer = {
            field.name: (
                self.producer.compatibility.as_dict()
                if field.name == "compatibility"
                else self.producer.source_provenance.as_dict()
                if field.name == "source_provenance"
                else getattr(self.producer, field.name)
            )
            for field in fields(ProducerDependencies)
        }
        return {
            "schema_version": self.schema_version,
            "value_fit_identity": {field.name: getattr(self.value_fit_identity, field.name) for field in fields(ValueFitIdentity)},
            "gain_scale": dict(self.gain_scale),
            "producer": producer,
            "source_provenance": self.source_provenance.as_dict(),
            "config": dict(self.config),
            "role_manifest": None if self.role_manifest is None else self.role_manifest.as_dict(),
            "completion": dict(self.completion),
            "stage_provenance": None if self.stage_provenance is None else dict(self.stage_provenance),
        }


def _validate_stage_provenance(
    value: Mapping[str, Any],
    *,
    producer: ProducerDependencies,
    role_manifest: TrainingRoleManifest | None,
    engineering_only: bool,
    _nested: bool = False,
) -> dict[str, Any]:
    """Validate the canonical producer-stage/spectral receipt envelope."""

    required = {
        "schema_version",
        "stage",
        "spectral_arm",
        "completed",
        "producer_compatibility_hash",
        "projector_before_hash",
        "projector_after_hash",
        "projector_gradient_evidence",
        "projector_update_evidence",
        "initialization_id",
        "checkpoint_id",
        "source_id",
        "split_role_hash",
        "role_manifest_digest",
        "verified_prior_receipt",
        "verified_prior_receipt_hash",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("stage_provenance keys are incomplete or unknown")
    if value["schema_version"] != "pfgr-lite-producer-stage-v1":
        raise ValueError("unknown producer-stage provenance schema")
    if value["stage"] != "updater":
        raise ValueError("producer-stage provenance stage must be updater")
    if value["spectral_arm"] not in {"u_plus_spectral", "verified_prior"}:
        raise ValueError("unknown producer-stage spectral arm")
    if not isinstance(value["completed"], bool):
        raise TypeError("producer-stage completed must be bool")
    if not isinstance(value["producer_compatibility_hash"], str) or value["producer_compatibility_hash"] != producer.compatibility_hash:
        raise ValueError("producer-stage producer identity does not match current producer")
    for name in ("projector_before_hash", "projector_after_hash", "initialization_id", "checkpoint_id", "source_id", "split_role_hash", "role_manifest_digest"):
        _complete_text(value[name], f"stage_provenance.{name}")
    if role_manifest is not None and value["role_manifest_digest"] != role_manifest.digest:
        raise ValueError("stage-provenance role manifest identity mismatch")
    if not isinstance(value["projector_gradient_evidence"], Mapping) or set(value["projector_gradient_evidence"]) != {"l2_norm_max", "nonzero_steps", "measured_steps"}:
        raise ValueError("projector_gradient_evidence keys are incomplete or unknown")
    gradient = value["projector_gradient_evidence"]
    if not math.isfinite(float(gradient["l2_norm_max"])) or float(gradient["l2_norm_max"]) < 0.0:
        raise ValueError("projector gradient norm must be finite and nonnegative")
    for name in ("nonzero_steps", "measured_steps"):
        if not isinstance(gradient[name], int) or isinstance(gradient[name], bool) or gradient[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not isinstance(value["projector_update_evidence"], Mapping) or set(value["projector_update_evidence"]) != {"changed_parameter_count", "optimizer_steps"}:
        raise ValueError("projector_update_evidence keys are incomplete or unknown")
    update = value["projector_update_evidence"]
    for name in ("changed_parameter_count", "optimizer_steps"):
        if not isinstance(update[name], int) or isinstance(update[name], bool) or update[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    prior = value["verified_prior_receipt"]
    prior_hash = value["verified_prior_receipt_hash"]
    if value["spectral_arm"] == "u_plus_spectral":
        if prior is not None or prior_hash is not None:
            raise ValueError("u_plus_spectral stage may not carry a verified-prior receipt")
        if value["completed"] and (float(gradient["l2_norm_max"]) <= 0.0 or gradient["nonzero_steps"] <= 0 or gradient["measured_steps"] <= 0 or update["changed_parameter_count"] <= 0 or update["optimizer_steps"] <= 0 or value["projector_before_hash"] == value["projector_after_hash"]):
            raise ValueError("completed u_plus_spectral stage lacks measured projector updates")
        if value["projector_after_hash"] != producer.compatibility.spectral_projector_hash:
            raise ValueError("stage projector_after_hash does not match current producer")
    else:
        if not isinstance(prior, Mapping) or not isinstance(prior_hash, str) or not prior_hash:
            raise ValueError("verified_prior requires a complete original stage receipt and hash")
        nested = _validate_stage_provenance(prior, producer=producer, role_manifest=role_manifest, engineering_only=engineering_only, _nested=True)
        if canonical_digest(nested, prefix="pfgr-lite-producer-stage-v1|") != prior_hash:
            raise ValueError("verified_prior receipt hash mismatch")
        if nested["spectral_arm"] != "u_plus_spectral" or nested["projector_after_hash"] != value["projector_after_hash"]:
            raise ValueError("verified_prior receipt does not bind the final projector")
    if not value["completed"] and not engineering_only:
        raise ValueError("incomplete producer stage cannot enter a production artifact")
    return dict(value)


def _bundle_metadata(bundle: InferenceBundle, state_dict: Mapping[str, Tensor]) -> dict[str, Any]:
    if not isinstance(bundle, InferenceBundle):
        raise TypeError("bundle must be InferenceBundle")
    config = _normalise_config(bundle.config)
    producer = {
        "compatibility": bundle.producer.compatibility.as_dict(),
        "source_provenance": bundle.producer.source_provenance.as_dict(),
        **{
            field.name: getattr(bundle.producer, field.name)
            for field in fields(ProducerDependencies)
            if field.name not in {"compatibility", "source_provenance"}
        },
    }
    calibration = None if bundle.calibration is None else asdict(bundle.calibration)
    evidence = None
    if bundle.calibration is not None:
        attached = calibration_evidence(bundle.calibration)
        evidence = None if attached is None else attached.as_dict()
    if bundle.calibration_evidence is not None:
        if evidence is not None and bundle.calibration_evidence != evidence:
            raise ValueError("bundle calibration_evidence does not match attached calibration evidence")
        evidence = dict(bundle.calibration_evidence)
    frontend_config = bundle.frontend_config if bundle.frontend_config is not None else config["frontend_config"]
    if frontend_config != config["frontend_config"]:
        raise ValueError("bundle frontend_config does not match strict config sidecar")
    value_fit = None if bundle.value_fit_identity is None else {field.name: getattr(bundle.value_fit_identity, field.name) for field in fields(ValueFitIdentity)}
    role_manifest = None if bundle.role_manifest is None else bundle.role_manifest.as_dict()
    if bundle.role_manifest is not None and bundle.split_hash is not None and bundle.role_manifest.baseline_split_hash != bundle.split_hash:
        raise ValueError("role manifest baseline split must match inference split_hash")
    gain_scale_hash = bundle.gain_scale_hash or config["gain_scale_hash"]
    effective_policy_hash = bundle.effective_policy_hash or config["effective_policy_hash"]
    if bundle.capability == "adaptive":
        pfgr = PFGRLiteConfig.from_dict(config["pfgr_config"])
        source = bundle.producer.source_provenance
        if pfgr.engineering_only:
            raise ValueError("engineering-only PFGR configuration cannot publish adaptive release")
        if source.synthetic_untrained or not source.official_pretrained_verified or not source.checkpoint_integrity_verified:
            raise ValueError("adaptive checkpoint requires verified official MedicalNet source provenance")
        if attached is None or not attached.deployment_ready:
            raise ValueError("adaptive checkpoint requires deployment-ready calibration evidence")
        if bundle.value_fit_identity is None or not gain_scale_hash or not effective_policy_hash or bundle.role_manifest is None:
            raise ValueError("adaptive checkpoint requires complete value/scale/policy/role identities")
        if attached.value_fit_identity_hash != bundle.value_fit_identity.digest or attached.gain_scale_hash != gain_scale_hash:
            raise ValueError("adaptive checkpoint value/scale identities do not match calibration evidence")
        if bundle.stage_provenance is None:
            raise ValueError("adaptive checkpoint requires producer-stage provenance")
        _validate_stage_provenance(
            bundle.stage_provenance,
            producer=bundle.producer,
            role_manifest=bundle.role_manifest,
            engineering_only=bundle.role_manifest.engineering_only,
        )
    return {
        "schema_version": bundle.schema_version,
        "capability": bundle.capability,
        "split_hash": bundle.split_hash,
        "config": config,
        "producer": producer,
        "calibration": calibration,
        "calibration_evidence": evidence,
        "frontend_config": frontend_config,
        "value_fit_identity": value_fit,
        "gain_scale_hash": gain_scale_hash,
        "effective_policy_hash": effective_policy_hash,
        "role_manifest": role_manifest,
        "stage_provenance": bundle.stage_provenance,
        "effective_policy": bundle.effective_policy,
        "gain_scale_provenance": bundle.gain_scale_provenance,
        "state_dict_digest": _state_dict_digest(state_dict),
    }


def _decode_bundle_payload(payload: Mapping[str, Any]) -> InferenceBundle:
    expected = {"format", "metadata_json", "state_dict"}
    unknown = set(payload) - expected
    if unknown:
        raise ValueError(f"unknown inference payload keys: {sorted(unknown)}")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("payload is not a PFGR-Lite inference artifact")
    metadata_raw = payload.get("metadata_json")
    if not isinstance(metadata_raw, str):
        raise TypeError("inference metadata_json must be a string")
    try:
        metadata = json.loads(metadata_raw)
    except json.JSONDecodeError as error:
        raise ValueError("inference metadata is not valid JSON") from error
    if not isinstance(metadata, Mapping):
        raise TypeError("inference metadata must decode to a mapping")
    required = {
        "schema_version",
        "capability",
        "split_hash",
        "config",
        "producer",
        "calibration",
        "calibration_evidence",
        "frontend_config",
        "value_fit_identity",
        "gain_scale_hash",
        "effective_policy_hash",
        "role_manifest",
        "stage_provenance",
        "effective_policy",
        "gain_scale_provenance",
        "state_dict_digest",
    }
    if set(metadata) != required:
        raise ValueError("inference metadata keys are incomplete or unknown")
    state_raw = payload.get("state_dict")
    if not isinstance(state_raw, Mapping):
        raise TypeError("inference state_dict must be a mapping")
    state: dict[str, Tensor] = {}
    for name, value in state_raw.items():
        _reject_secret_keys(name)
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise TypeError("inference state_dict must map string names to tensors")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("inference state_dict tensors must be finite")
        state[name] = value.detach().cpu().clone()
    if _state_dict_digest(state) != metadata["state_dict_digest"]:
        raise ValueError("inference state_dict digest mismatch")
    config = _normalise_config(metadata["config"])
    producer = _producer_from_dict(metadata["producer"])
    parsed_pfgr = PFGRLiteConfig.from_dict(config["pfgr_config"])
    if metadata["capability"] == "adaptive":
        source = producer.source_provenance
        if parsed_pfgr.engineering_only:
            raise ValueError("engineering-only PFGR configuration cannot load adaptive release")
        if source.synthetic_untrained or not source.official_pretrained_verified or not source.checkpoint_integrity_verified:
            raise ValueError("adaptive artifact requires verified official MedicalNet source provenance")
    calibration_raw = metadata["calibration"]
    calibration = None if calibration_raw is None else _calibration_from_dict(calibration_raw)
    evidence_raw = metadata["calibration_evidence"]
    evidence: CalibrationEvidence | None = None
    if evidence_raw is not None:
        if calibration is None:
            raise ValueError("calibration evidence cannot exist without calibration")
        evidence = CalibrationEvidence.from_dict(evidence_raw)
        attach_calibration_evidence(calibration, evidence)
    value_fit_raw = metadata["value_fit_identity"]
    value_fit = None
    if value_fit_raw is not None:
        if not isinstance(value_fit_raw, Mapping):
            raise TypeError("value_fit_identity must be a mapping or None")
        value_fit = ValueFitIdentity(**dict(value_fit_raw))
    role_raw = metadata["role_manifest"]
    role_manifest = None if role_raw is None else TrainingRoleManifest.from_dict(role_raw)
    frontend_config = metadata["frontend_config"]
    if not isinstance(frontend_config, Mapping):
        raise TypeError("frontend_config metadata must be a mapping")
    # Validate the sidecar independently before handing it to callers.
    frontend_config_from_dict(frontend_config)
    if frontend_config != config["frontend_config"]:
        raise ValueError("frontend_config metadata does not match config sidecar")
    for name in ("gain_scale_hash", "effective_policy_hash"):
        if metadata[name] != config[name]:
            raise ValueError(f"{name} metadata does not match config sidecar")
    stage_provenance = metadata["stage_provenance"]
    if stage_provenance is not None and not isinstance(stage_provenance, Mapping):
        raise TypeError("stage_provenance metadata must be a mapping or None")
    if stage_provenance is not None:
        stage_provenance = _validate_stage_provenance(
            stage_provenance,
            producer=producer,
            role_manifest=role_manifest,
            engineering_only=(role_manifest is None or role_manifest.engineering_only),
        )
    elif metadata["capability"] == "adaptive":
        raise ValueError("adaptive inference artifact requires producer-stage provenance")
    effective_policy = metadata["effective_policy"]
    if effective_policy is not None and not isinstance(effective_policy, Mapping):
        raise TypeError("effective_policy metadata must be a mapping or None")
    gain_scale_provenance = metadata["gain_scale_provenance"]
    if gain_scale_provenance is not None and not isinstance(gain_scale_provenance, Mapping):
        raise TypeError("gain_scale_provenance metadata must be a mapping or None")
    if metadata["capability"] == "adaptive":
        if calibration is None or evidence is None or not evidence.deployment_ready:
            raise ValueError("adaptive inference artifact requires deployment-ready calibration evidence")
        if role_manifest is None or value_fit is None or not metadata["gain_scale_hash"] or not metadata["effective_policy_hash"]:
            raise ValueError("adaptive inference artifact requires complete value/scale/policy/role identities")
        if evidence.value_fit_identity_hash != value_fit.digest:
            raise ValueError("adaptive artifact ValueFitIdentity does not match calibration evidence")
        if evidence.gain_scale_hash != metadata["gain_scale_hash"]:
            raise ValueError("adaptive artifact gain-scale identity does not match calibration evidence")
        if effective_policy is None or effective_policy.get("policy_hash") != metadata["effective_policy_hash"]:
            raise ValueError("adaptive artifact effective policy identity is missing or stale")
    return InferenceBundle(
        state_dict=state,
        producer=producer,
        config=config,
        capability=metadata["capability"],
        calibration=calibration,
        split_hash=metadata["split_hash"],
        schema_version=metadata["schema_version"],
        frontend_config=frontend_config,
        value_fit_identity=value_fit,
        gain_scale_hash=metadata["gain_scale_hash"],
        effective_policy_hash=metadata["effective_policy_hash"],
        role_manifest=role_manifest,
        stage_provenance=stage_provenance,
        calibration_evidence=None if evidence_raw is None else evidence.as_dict(),
        effective_policy=effective_policy,
        gain_scale_provenance=gain_scale_provenance,
    )


def _atomic_save(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing PFGR artifact: {path}")
    if not path.parent.exists():
        raise FileNotFoundError(f"artifact parent directory does not exist: {path.parent}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        # ``replace`` would silently clobber a concurrently published
        # artifact after the initial exists() check.  A same-directory hard
        # link is an atomic create-if-absent publication primitive.
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite existing PFGR artifact: {path}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    """Validate a metadata mapping before embedding it in canonical JSON."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    safe = _safe_value(value, path=name)
    if not isinstance(safe, Mapping):  # pragma: no cover - defensive
        raise TypeError(f"{name} must be a mapping")
    # ``_safe_value`` permits tensors for resume payloads; value artifact
    # metadata is JSON-only so reject those explicitly here.
    def _plain(item: Any, path: str) -> Any:
        if isinstance(item, Tensor):
            raise TypeError(f"{name} metadata cannot contain tensors at {path}")
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise TypeError(f"{name} metadata keys must be strings at {path}")
            return {key: _plain(child, f"{path}.{key}") for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [_plain(child, f"{path}[]") for child in item]
        if item is None or isinstance(item, (str, int, bool)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{name} metadata contains a nonfinite float at {path}")
            return item
        raise TypeError(f"unsupported {name} metadata value {type(item).__name__} at {path}")

    return _plain(safe, name)


def _value_identity_from_object(value_fit: Any) -> ValueFitIdentity:
    identity = getattr(value_fit, "identity", getattr(value_fit, "fit_identity", None))
    if not isinstance(identity, ValueFitIdentity):
        raise TypeError("value fit must expose a complete ValueFitIdentity")
    return identity


def _value_state_from_object(value_fit: Any) -> dict[str, Tensor]:
    model = getattr(value_fit, "model", None)
    if model is None:
        model = getattr(value_fit, "value_net", None)
    if model is not None and hasattr(model, "state_dict"):
        raw = model.state_dict()
    elif isinstance(value_fit, Mapping):
        raw = value_fit.get("state_dict", value_fit.get("model_state_dict"))
    else:
        raw = None
    if not isinstance(raw, Mapping):
        raise TypeError("value fit must expose a tensor state_dict")
    result: dict[str, Tensor] = {}
    for name, tensor in raw.items():
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            raise TypeError("value fit state_dict must map string names to tensors")
        result[name] = tensor.detach().cpu().clone()
    return result


def save_value_artifact(
    path: str | Path,
    value_fit: Any,
    *,
    producer: ProducerDependencies,
    config: Mapping[str, Any] | Any,
    role_manifest: TrainingRoleManifest | None = None,
    source_provenance: SourceProvenance | None = None,
    stage_provenance: Mapping[str, Any] | None = None,
    completion: Mapping[str, Any] | None = None,
) -> None:
    """Publish one strict V-only fit artifact.

    ``value_fit`` accepts W3a's ``ValueFitResult`` protocol (``model``,
    ``identity`` and ``gain_scale`` attributes) or an equivalent mapping.  No
    producer model state is accepted or copied into this payload.
    """

    if not isinstance(producer, ProducerDependencies):
        raise TypeError("producer must be ProducerDependencies")
    identity = _value_identity_from_object(value_fit)
    state = _value_state_from_object(value_fit)
    if _module_state_digest_from_mapping(state) != identity.weights_hash:
        raise ValueError("value fit state does not match ValueFitIdentity.weights_hash")
    scale_object = getattr(value_fit, "gain_scale", None)
    if scale_object is None and isinstance(value_fit, Mapping):
        scale_object = value_fit.get("gain_scale")
    if hasattr(scale_object, "as_dict"):
        scale = dict(scale_object.as_dict())
    elif isinstance(scale_object, Mapping):
        scale = dict(scale_object)
    else:
        raise TypeError("value fit must expose GainScale.as_dict()")
    scale_copy = dict(scale)
    scale_digest = scale_copy.get("digest")
    if not isinstance(scale_digest, str) or not scale_digest:
        scale_digest = canonical_digest({key: value for key, value in scale_copy.items() if key != "digest"}, prefix="pfgr-lite-gain-scale-v1|")
        scale_copy["digest"] = scale_digest
    if scale_digest != identity.gain_scale_hash:
        raise ValueError("value fit gain scale does not match ValueFitIdentity")
    source = producer.source_provenance if source_provenance is None else source_provenance
    if not isinstance(source, SourceProvenance):
        raise TypeError("source_provenance must be SourceProvenance")
    if source.digest != producer.source_provenance.digest:
        raise ValueError("value artifact source provenance must match producer")
    role = role_manifest
    if role is not None and not isinstance(role, TrainingRoleManifest):
        raise TypeError("role_manifest must be TrainingRoleManifest or None")
    completion_map = dict(completion) if completion is not None else {
        "complete": bool(getattr(value_fit, "complete", False)),
        "stage": "value_fit",
        "bank_manifest_hash": identity.bank_manifest_hash,
    }
    if not isinstance(config, Mapping) and hasattr(config, "as_dict"):
        config = config.as_dict()
    config_map = _json_mapping(config, name="config")
    fit_options = getattr(value_fit, "fit_options", None)
    if fit_options is not None and hasattr(fit_options, "as_dict") and config_map.get("schema_version") != _VALUE_FIT_CONFIG_SCHEMA:
        raw_options = dict(fit_options.as_dict())
        options = {
            "batch_size": raw_options["batch_size"],
            "seed": raw_options["seed"],
            "learning_rate": raw_options["learning_rate"],
            "weight_decay": raw_options["weight_decay"],
            "loss": raw_options["loss"],
            "shuffle": raw_options["shuffle"],
            "robust_ablation": raw_options["robust_ablation"],
            "optimizer": {
                "class": "torch.optim.Adam",
                "groups": [{
                    "name": "value_net",
                    "lr": float(raw_options["learning_rate"]),
                    "weight_decay": float(raw_options["weight_decay"]),
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "amsgrad": False,
                    "maximize": False,
                }],
            },
        }
        config_map = {
            "schema_version": _VALUE_FIT_CONFIG_SCHEMA,
            "model_architecture": identity.architecture_hash,
            "options": options,
            "bank_manifest_hash": identity.bank_manifest_hash,
            "gain_scale_hash": identity.gain_scale_hash,
            "producer_compatibility_hash": producer.compatibility_hash,
            "role_manifest_hash": None if role is None else role.digest,
            "producer_config": config_map,
        }
        if completion is None:
            completion_map = {
                "schema_version": _VALUE_COMPLETION_SCHEMA,
                "complete": bool(getattr(value_fit, "complete", False)),
                "stage": "value_fit",
                "bank_manifest_hash": identity.bank_manifest_hash,
                "fit_config_hash": identity.fit_config_hash,
                "producer_compatibility_hash": producer.compatibility_hash,
                "gain_scale_hash": identity.gain_scale_hash,
            }
    completion_map = _json_mapping(completion_map, name="completion")
    stage_map = None if stage_provenance is None else _json_mapping(stage_provenance, name="stage_provenance")
    if stage_map is not None:
        stage_map = _validate_stage_provenance(
            stage_map,
            producer=producer,
            role_manifest=role,
            engineering_only=(role is None or role.engineering_only),
        )
    # Validate the exact same envelope at publication time that a later load
    # will reconstruct; this prevents malformed architecture/scale metadata
    # from ever becoming an apparently valid artifact.
    artifact = ValueArtifact(
        state_dict=state,
        value_fit_identity=identity,
        gain_scale=scale_copy,
        producer=producer,
        source_provenance=source,
        config=config_map,
        role_manifest=role,
        completion=completion_map,
        stage_provenance=stage_map,
    )
    metadata = {
        "schema_version": VALUE_ARTIFACT_SCHEMA,
        "input_variant": identity.input_variant,
        "value_fit_identity": {field.name: getattr(identity, field.name) for field in fields(ValueFitIdentity)},
        "gain_scale": scale_copy,
        "producer": {
            field.name: (
                producer.compatibility.as_dict()
                if field.name == "compatibility"
                else producer.source_provenance.as_dict()
                if field.name == "source_provenance"
                else getattr(producer, field.name)
            )
            for field in fields(ProducerDependencies)
        },
        "source_provenance": source.as_dict(),
        "config": config_map,
        "role_manifest": None if role is None else role.as_dict(),
        "completion": completion_map,
        "stage_provenance": stage_map,
        "state_dict_digest": _state_dict_digest(artifact.state_dict),
    }
    payload = {
        "format": VALUE_ARTIFACT_FORMAT,
        "metadata_json": json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False),
        "state_dict": state,
    }
    _atomic_save(Path(path), payload)


def _decode_value_payload(payload: Mapping[str, Any]) -> ValueArtifact:
    expected = {"format", "metadata_json", "state_dict"}
    if set(payload) != expected or payload.get("format") != VALUE_ARTIFACT_FORMAT:
        raise ValueError("unknown or incomplete value artifact payload")
    raw_metadata = payload.get("metadata_json")
    if not isinstance(raw_metadata, str):
        raise TypeError("value artifact metadata_json must be a string")
    try:
        metadata = json.loads(raw_metadata)
    except json.JSONDecodeError as exc:
        raise ValueError("value artifact metadata is not valid JSON") from exc
    required = {"schema_version", "input_variant", "value_fit_identity", "gain_scale", "producer", "source_provenance", "config", "role_manifest", "completion", "stage_provenance", "state_dict_digest"}
    if not isinstance(metadata, Mapping) or set(metadata) != required:
        raise ValueError("value artifact metadata keys are incomplete or unknown")
    if metadata["schema_version"] != VALUE_ARTIFACT_SCHEMA:
        raise ValueError("unknown value artifact schema")
    identity_raw = metadata["value_fit_identity"]
    if not isinstance(identity_raw, Mapping):
        raise TypeError("value_fit_identity must be a mapping")
    identity = ValueFitIdentity(**dict(identity_raw))
    if metadata["input_variant"] != identity.input_variant:
        raise ValueError("value artifact input_variant does not match ValueFitIdentity")
    state_raw = payload["state_dict"]
    if not isinstance(state_raw, Mapping):
        raise TypeError("value artifact state_dict must be a mapping")
    state: dict[str, Tensor] = {}
    for name, value in state_raw.items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise TypeError("value artifact state_dict must map string names to tensors")
        state[name] = value.detach().cpu().clone()
    if _state_dict_digest(state) != metadata["state_dict_digest"]:
        raise ValueError("value artifact state_dict digest mismatch")
    producer_raw = metadata["producer"]
    if not isinstance(producer_raw, Mapping):
        raise TypeError("value artifact producer must be a mapping")
    producer = _producer_from_dict(producer_raw)
    source_raw = metadata["source_provenance"]
    if not isinstance(source_raw, Mapping):
        raise TypeError("value artifact source provenance must be a mapping")
    source = _source_from_dict(source_raw)
    role_raw = metadata["role_manifest"]
    role = None if role_raw is None else TrainingRoleManifest.from_dict(role_raw)
    config = _json_mapping(metadata["config"], name="config")
    completion = _json_mapping(metadata["completion"], name="completion")
    stage = None if metadata["stage_provenance"] is None else _json_mapping(metadata["stage_provenance"], name="stage_provenance")
    if stage is not None:
        stage = _validate_stage_provenance(
            stage,
            producer=producer,
            role_manifest=role,
            engineering_only=(role is None or role.engineering_only),
        )
    artifact = ValueArtifact(
        state_dict=state,
        value_fit_identity=identity,
        gain_scale=dict(metadata["gain_scale"]),
        producer=producer,
        source_provenance=source,
        config=config,
        role_manifest=role,
        completion=completion,
        stage_provenance=stage,
    )
    if artifact.source_provenance.digest != artifact.producer.source_provenance.digest:
        raise ValueError("value artifact source/producer identity mismatch")
    return artifact


def load_value_artifact(
    path: str | Path,
    *,
    expected_producer: ProducerDependencies | ProducerCompatibility | None = None,
    expected_role_manifest: TrainingRoleManifest | None = None,
    expected_input_variant: int | None = None,
    expected_gain_scale_hash: str | None = None,
) -> ValueArtifact:
    """Load and strictly validate a V-only artifact with explicit joins."""

    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    try:
        payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("PFGR value artifacts require torch.load(weights_only=True)") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("value artifact payload must be a mapping")
    artifact = _decode_value_payload(payload)
    if expected_producer is not None:
        expected_hash = expected_producer.compatibility_hash if isinstance(expected_producer, ProducerDependencies) else expected_producer.digest
        if artifact.producer.compatibility_hash != expected_hash:
            raise ValueError("value artifact producer compatibility mismatch")
    if expected_role_manifest is not None:
        if not isinstance(expected_role_manifest, TrainingRoleManifest) or artifact.role_manifest is None or artifact.role_manifest.digest != expected_role_manifest.digest:
            raise ValueError("value artifact role manifest mismatch")
    if expected_input_variant is not None and artifact.value_fit_identity.input_variant != expected_input_variant:
        raise ValueError("value artifact input variant mismatch")
    if expected_gain_scale_hash is not None and artifact.value_fit_identity.gain_scale_hash != expected_gain_scale_hash:
        raise ValueError("value artifact gain scale mismatch")
    return artifact


def hydrate_inference_model(
    bundle: InferenceBundle,
    *,
    model_factory: Callable[..., Any] | None = None,
    query_lattice_factory: Any | None = None,
) -> Any:
    """Construct PFGRLiteModel from the strict sidecar and load exact state.

    The default import is local to keep checkpoint imports target/teacher free.
    A custom factory is accepted for tests, but it must expose the same strict
    ``(config, frontend_config=..., query_lattice_factory=...)`` constructor
    seam when a canonical W2 query lattice is supplied.  The lattice factory
    is intentionally an explicit injection: W4 never fabricates query
    geometry, and callers that need ``decode_final`` must provide the same
    canonical factory used by route execution.
    """

    if not isinstance(bundle, InferenceBundle):
        raise TypeError("bundle must be InferenceBundle")
    if bundle.frontend_config is None:
        raise ValueError("model hydration requires the serialized frontend_config sidecar")
    pfgr_config = PFGRLiteConfig.from_dict(bundle.config["pfgr_config"])
    frontend_config = frontend_config_from_dict(bundle.frontend_config)
    if model_factory is None:
        from .model import PFGRLiteModel

        model_factory = PFGRLiteModel
    if query_lattice_factory is None:
        model = model_factory(pfgr_config, frontend_config=frontend_config)
    else:
        try:
            model = model_factory(
                pfgr_config,
                frontend_config=frontend_config,
                query_lattice_factory=query_lattice_factory,
            )
        except TypeError as exc:
            raise TypeError(
                "model_factory must accept the explicit query_lattice_factory injection"
            ) from exc
    if not hasattr(model, "load_state_dict") or not hasattr(model, "state_dict"):
        raise TypeError("model_factory must return a torch module with state_dict/load_state_dict")
    model.load_state_dict(dict(bundle.state_dict), strict=True)
    compatibility = bundle.producer.compatibility
    frontend = getattr(model, "frontend", None)
    prior = getattr(frontend, "semantic_prior", None)
    checks = (
        ("static_head_hash", getattr(model, "static_head", None)),
        ("state_initializer_hash", getattr(getattr(model, "static_head", None), "final_projection", None)),
        ("updater_hash", getattr(model, "updater", None)),
        ("decoder_hash", getattr(model, "decoder", None)),
        ("semantic_head_hash", getattr(prior, "semantic_head", None)),
        ("point_refiner_hash", getattr(frontend, "point_refiner", None)),
        ("spectral_projector_hash", getattr(frontend, "spectral_anchor_builder", None)),
    )
    for name, component in checks:
        expected = getattr(compatibility, name)
        if component is not None and module_state_digest(component) != expected:
            raise ValueError(f"hydrated model {name} does not match producer compatibility")
    backbone = getattr(prior, "backbone", None)
    if backbone is not None and batchnorm_state_digest(backbone) != compatibility.frozen_bn_hash:
        raise ValueError("hydrated model frozen BatchNorm state does not match producer compatibility")
    if backbone is not None:
        source = bundle.producer.source_provenance
        actual_source = source_provenance_from_semantic_prior(prior)
        for name in (
            "source_input_channels",
            "adapted_input_channels",
            "input_conv_adapted",
            "checkpoint_sha256",
            "checkpoint_integrity_verified",
            "adaptation_digest",
            "parameter_hash",
            "frozen_bn_hash",
            "official_pretrained_verified",
            "synthetic_untrained",
        ):
            if getattr(actual_source, name) != getattr(source, name):
                raise ValueError(f"hydrated model source provenance {name} does not match bundle")
        if source.parameter_hash is not None and module_parameter_digest(backbone) != source.parameter_hash:
            raise ValueError("hydrated model MedicalNet parameters do not match source provenance")
        actual_medicalnet_hash = canonical_digest(
            {
                "checkpoint_sha256": source.checkpoint_sha256,
                "checkpoint_integrity_verified": source.checkpoint_integrity_verified,
                "adaptation_digest": source.adaptation_digest,
                "source_input_channels": source.source_input_channels,
                "adapted_input_channels": source.adapted_input_channels,
                "input_conv_adapted": source.input_conv_adapted,
                "parameter_hash": source.parameter_hash,
            },
            prefix="pfgr-lite-medicalnet-producer-v1|",
        )
        if actual_medicalnet_hash != compatibility.medicalnet_provenance_hash:
            raise ValueError("hydrated model MedicalNet provenance does not match producer compatibility")
    # Recompute the remaining producer identities whose values are determined
    # by the strict model/config sidecar rather than one subject's geometry.
    model_config = getattr(model, "config", None)
    if isinstance(model_config, PFGRLiteConfig):
        expected_normalization = canonical_digest(
            model_config.observation_normalization,
            prefix="pfgr-lite-observation-normalization-v1|",
        )
        if compatibility.observation_normalization_hash != expected_normalization:
            raise ValueError("hydrated model observation-normalization identity does not match producer")
        expected_query = canonical_digest(
            "pfgr-lite-query-lattice-v1",
            prefix="pfgr-lite-geometry-version|",
        )
        if compatibility.geometry_query_version_hash != expected_query:
            raise ValueError("hydrated model query algorithm identity does not match producer")
        expected_writer = canonical_digest("compact-writeback-4mm-v1")
        if compatibility.writer_hash != expected_writer:
            raise ValueError("hydrated model writer identity does not match producer")
        expected_candidate = canonical_digest(
            {
                "candidate_count": model_config.candidate_count,
                "num_points": model_config.num_points,
                "point_candidate_multiplier": getattr(frontend_config, "point_candidate_multiplier", None),
                "directional_offsets_mm": tuple(getattr(frontend_config, "directional_offsets_mm", ())),
                "support_radius_mm": model_config.support_radius_mm,
                "max_displacement_mm": model_config.max_displacement_mm,
                "algorithm_version": "point-candidate-geometry-v1",
            },
            prefix="pfgr-lite-candidate-geometry-v1|",
        )
        if compatibility.candidate_geometry_hash != expected_candidate:
            raise ValueError("hydrated model candidate geometry identity does not match producer")
        expected_label = canonical_digest(
            {
                "definition": model_config.teacher.label_definition,
                "rho": model_config.teacher.rho,
                "epsilon": model_config.teacher.epsilon,
                "mask_definition": model_config.teacher.mask_definition,
                "global_mask_denominator": "sum(mask)>0_fixed_subject_v1",
            },
            prefix="pfgr-lite-label-definition-v1|",
        )
        if compatibility.label_definition_hash != expected_label:
            raise ValueError("hydrated model label-definition identity does not match producer")
    return model


def save_inference_bundle(path: str | Path, bundle: InferenceBundle) -> None:
    """Atomically write a strict target-free inference artifact."""

    if not isinstance(bundle, InferenceBundle):
        raise TypeError("bundle must be InferenceBundle")
    state = {name: value.detach().cpu().clone() for name, value in bundle.state_dict.items()}
    payload = {
        "format": CHECKPOINT_FORMAT,
        "metadata_json": json.dumps(
            _bundle_metadata(bundle, state),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        "state_dict": state,
    }
    _atomic_save(Path(path), payload)


def load_inference_bundle(
    path: str | Path,
    *,
    expected_split_hash: str | None = None,
    required_capability: str | None = None,
) -> InferenceBundle:
    """Load and recompute every strict inference identity."""

    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    try:
        payload = torch.load(artifact, map_location="cpu", weights_only=True)
    except TypeError as exc:
        # Never fall back to unrestricted pickle loading at this production
        # boundary.  Old runtimes must upgrade or use the explicit legacy
        # adapter API instead.
        raise RuntimeError("PFGR inference artifacts require torch.load(weights_only=True)") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("inference payload must be a mapping")
    bundle = _decode_bundle_payload(payload)
    if expected_split_hash is not None and bundle.split_hash != expected_split_hash:
        raise ValueError("inference split hash mismatch")
    if expected_split_hash is not None and bundle.role_manifest is not None and bundle.role_manifest.baseline_split_hash != expected_split_hash:
        raise ValueError("training role manifest baseline split mismatch")
    if required_capability is not None:
        levels = {"static": 0, "forced_diagnostic": 1, "adaptive": 2}
        if required_capability not in levels:
            raise ValueError("unknown required inference capability")
        if bundle.capability not in levels or levels[bundle.capability] < levels[required_capability]:
            raise ValueError("inference artifact capability is insufficient")
    return bundle


def load_legacy_inference_bundle(
    path: str | Path,
    *,
    adapter: Callable[[Mapping[str, Any]], InferenceBundle] | None = None,
) -> InferenceBundle:
    """Explicit legacy-only loader; no implicit PFGR reinterpretation."""

    if adapter is None:
        raise RuntimeError("legacy inference requires an explicit adapter; PFGR loading is strict")
    artifact = Path(path)
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("legacy payload must be a mapping for the explicit adapter")
    result = adapter(payload)
    if not isinstance(result, InferenceBundle):
        raise TypeError("legacy adapter must return InferenceBundle")
    return result


def _stage_payload(stage_state: StageState) -> dict[str, Any]:
    if not isinstance(stage_state, StageState):
        raise TypeError("stage_state must be StageState")
    return asdict(stage_state)


def save_resume(
    path: str | Path,
    bundle: InferenceBundle,
    stage_state: StageState,
    optimizer_state: Mapping[str, Any],
    rng_state: Mapping[str, Any],
    bank_state: Mapping[str, Any],
) -> None:
    """Write a strict, atomic resume artifact with complete runtime state."""

    if not isinstance(bundle, InferenceBundle) or not isinstance(stage_state, StageState):
        raise TypeError("resume requires InferenceBundle and StageState")
    for name, value in (("optimizer_state", optimizer_state), ("rng_state", rng_state), ("bank_state", bank_state)):
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping")
    state = {name: value.detach().cpu().clone() for name, value in bundle.state_dict.items()}
    inference_payload = {
        "format": CHECKPOINT_FORMAT,
        "metadata_json": json.dumps(
            _bundle_metadata(bundle, state),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        "state_dict": state,
    }
    payload = {
        "format": RESUME_FORMAT,
        "protocol": RESUME_SCHEMA,
        "inference": inference_payload,
        "stage_state": _stage_payload(stage_state),
        "optimizer_state": _safe_value(optimizer_state, path="optimizer_state"),
        "rng_state": _safe_value(rng_state, path="rng_state"),
        "bank_state": _safe_value(bank_state, path="bank_state"),
    }
    _atomic_save(Path(path), payload)


def load_resume(path: str | Path, *, expected_protocol: str = RESUME_SCHEMA) -> ResumeState:
    """Load strict resume state; partial/unknown protocols are rejected."""

    if expected_protocol != RESUME_SCHEMA:
        raise ValueError("expected_protocol must be the current PFGR resume schema")
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    try:
        payload = torch.load(artifact, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("PFGR resume artifacts require torch.load(weights_only=True)") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("resume payload must be a mapping")
    expected = {"format", "protocol", "inference", "stage_state", "optimizer_state", "rng_state", "bank_state"}
    unknown = set(payload) - expected
    if unknown or set(payload) != expected:
        raise ValueError("resume payload keys are incomplete or unknown")
    if payload["format"] != RESUME_FORMAT or payload["protocol"] != expected_protocol:
        raise ValueError("unknown or mismatched PFGR resume protocol")
    inference_raw = payload["inference"]
    if not isinstance(inference_raw, Mapping):
        raise TypeError("resume inference payload must be a mapping")
    inference = _decode_bundle_payload(inference_raw)
    stage_raw = payload["stage_state"]
    if not isinstance(stage_raw, Mapping):
        raise TypeError("resume stage_state must be a mapping")
    stage_allowed = {field.name for field in fields(StageState)}
    if set(stage_raw) - stage_allowed:
        raise ValueError("unknown stage_state keys")
    stage = StageState(**dict(stage_raw))
    maps: dict[str, Mapping[str, Any]] = {}
    for name in ("optimizer_state", "rng_state", "bank_state"):
        value = payload[name]
        if not isinstance(value, Mapping):
            raise TypeError(f"resume {name} must be a mapping")
        _validate_loaded_value(value, path=name)
        maps[name] = _restore_value(value, path=name)
    bank = maps["bank_state"]
    if "producer_compatibility_hash" in bank and bank["producer_compatibility_hash"] != inference.producer.compatibility_hash:
        raise ValueError("resume bank producer identity does not match inference bundle")
    if "split_role_hash" in bank and inference.role_manifest is not None and bank["split_role_hash"] != inference.role_manifest.digest:
        raise ValueError("resume bank split-role identity does not match inference role manifest")
    if "gain_scale_hash" in bank and inference.gain_scale_hash and bank["gain_scale_hash"] != inference.gain_scale_hash:
        raise ValueError("resume bank gain-scale identity does not match inference bundle")
    return ResumeState(
        inference=inference,
        stage_state=stage,
        optimizer_state=maps["optimizer_state"],
        rng_state=maps["rng_state"],
        bank_state=maps["bank_state"],
        protocol=expected_protocol,
    )


def restore_rng_state(rng_state: Mapping[str, Any]) -> None:
    """Explicitly restore Python, NumPy, and Torch RNG snapshots.

    Restoration is opt-in so merely inspecting a checkpoint cannot mutate
    process-global randomness.  Missing streams are left untouched; supplied
    streams are type-checked and restored exactly.
    """

    if not isinstance(rng_state, Mapping):
        raise TypeError("rng_state must be a mapping")
    _validate_loaded_value(rng_state, path="rng_state")
    if "python" in rng_state:
        value = rng_state["python"]
        if not isinstance(value, tuple):
            raise TypeError("python RNG state must be a tuple")
        random.setstate(value)
    if "numpy" in rng_state:
        value = rng_state["numpy"]
        if not isinstance(value, tuple) or len(value) != 5:
            raise TypeError("NumPy RNG state must be a five-element tuple")
        np.random.set_state(value)
    if "torch_cpu" in rng_state:
        value = rng_state["torch_cpu"]
        if not isinstance(value, Tensor) or value.dtype != torch.uint8:
            raise TypeError("torch_cpu RNG state must be a uint8 tensor")
        torch.set_rng_state(value.detach().cpu())
    if "torch_cuda" in rng_state:
        value = rng_state["torch_cuda"]
        if not isinstance(value, (tuple, list)) or not torch.cuda.is_available():
            raise RuntimeError("torch_cuda RNG state requires CUDA and a sequence of uint8 tensors")
        if any(not isinstance(item, Tensor) or item.dtype != torch.uint8 for item in value):
            raise TypeError("torch_cuda RNG state must contain uint8 tensors")
        torch.cuda.set_rng_state_all([item.detach().cpu() for item in value])


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_CONFIG_SCHEMA",
    "RESUME_FORMAT",
    "VALUE_ARTIFACT_FORMAT",
    "VALUE_ARTIFACT_SCHEMA",
    "ValueArtifact",
    "load_inference_bundle",
    "load_legacy_inference_bundle",
    "load_resume",
    "load_value_artifact",
    "hydrate_inference_model",
    "restore_rng_state",
    "save_inference_bundle",
    "save_resume",
    "save_value_artifact",
]
