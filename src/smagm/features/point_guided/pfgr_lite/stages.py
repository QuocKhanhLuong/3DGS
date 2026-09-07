"""Bounded PFGR-Lite staged services (S0--S6).

The stage runner is intentionally an orchestration layer around explicit
callables.  It owns optimizer groups, deterministic route sampling, late
target joins, and immutable receipts, while W1/W2/W4 retain the actual model,
query, writer, teacher, policy, and checkpoint contracts.  This keeps the
module useful for small CPU engineering fixtures without creating a second
frontend or a competing shared declaration.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import Tensor, nn

from .config import PFGRLiteConfig, ValueModelConfig
from .data import TargetFreeSample, bind_observation_context, build_training_role_manifest, defer_supervision, load_observation_sample, normalization_identity
from .objectives import static_objective, updater_objective
from .provenance import ProducerCompatibility, canonical_digest, module_state_digest
from .types import CompletedBehaviorTrace, Decision, OperationCounters, ProducerDependencies, StageState, TrainingRoleManifest


STAGE_OPTIONS_SCHEMA = "pfgr-lite-stage-options-v1"
EXECUTION_CONFIG_SCHEMA = "pfgr-lite-execution-config-v1"
STAGE_RECEIPT_SCHEMA = "pfgr-lite-stage-receipt-v1"
PRODUCER_STAGE_SCHEMA = "pfgr-lite-producer-stage-v1"
STAGE_RUNTIME_SCHEMA = "pfgr-lite-stage-runtime-v1"
STAGES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6")
ARMS = ("u_only", "u_plus_spectral")


def _positive_int(name: str, value: object, *, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds bound {maximum}")
    return int(value)


def _nonnegative_int(name: str, value: object, *, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds bound {maximum}")
    return int(value)


def _finite_nonnegative(name: str, value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _jsonable(value.as_dict())
    if hasattr(value, "to_metadata") and callable(value.to_metadata):
        return _jsonable(value.to_metadata())
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Tensor):
        return {"dtype": str(value.dtype), "shape": tuple(value.shape), "sha256": canonical_digest(value.detach().cpu().numpy().tobytes().hex())}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("receipt cannot contain nonfinite values")
        return value
    return str(value)


def _invoke(callable_obj: Callable[..., Any], *positional: Any, **keyword: Any) -> Any:
    """Invoke a seam with the richest compatible keyword subset.

    W2/W4 can evolve their adapters independently; signature inspection keeps
    this runner strict about values while avoiding positional argument hacks.
    Internal TypeErrors from the callable itself are not swallowed when its
    signature is inspectable.
    """

    if not callable(callable_obj):
        raise TypeError("injected service must be callable")
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return callable_obj(*positional, **keyword)
    parameters = signature.parameters
    accepts_var_kw = any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values())
    positional_names = [
        name
        for name, item in parameters.items()
        if item.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    positional_count = min(len(positional), len(positional_names))
    consumed_names = set(positional_names[:positional_count])
    if accepts_var_kw:
        resolved_kw = {name: value for name, value in keyword.items() if name not in consumed_names}
    else:
        resolved_kw = {
            name: value
            for name, value in keyword.items()
            if name in parameters and name not in consumed_names and parameters[name].kind != inspect.Parameter.POSITIONAL_ONLY
        }
    return callable_obj(*positional[:positional_count], **resolved_kw)


def _stage_name(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("stage must be a string")
    canonical = value.upper()
    if canonical not in STAGES:
        raise ValueError(f"unknown PFGR stage {value!r}; expected one of {STAGES}")
    return canonical


@dataclass(frozen=True)
class StageOptions:
    """Strict bounded operational options for one PFGR stage."""

    stage: str = "S0"
    arm: Literal["u_only", "u_plus_spectral"] = "u_plus_spectral"
    seed: int = 0
    epochs: int = 1
    batch_size: int = 1
    max_updates: int | None = None
    device: str = "cpu"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    loss: Literal["charbonnier", "charbonnier_ssim_gradient"] = "charbonnier_ssim_gradient"
    query_chunk_size: int = 1024
    candidate_chunk_size: int = 1
    accumulation_steps: int = 1
    delta_weight: float = 0.0
    query_mode: Literal["exact_dense", "iid_fixed_q"] = "exact_dense"
    candidate_count: int = 32
    max_states_per_subject: int = 3
    candidates_per_state: int = 32
    output_format: Literal["json"] = "json"
    semantic_objective: bool = False
    ssim_weight: float = 0.2
    gradient_weight: float = 0.1
    semantic_weight: float = 0.2
    # S2 measurement budget sidecar.  ``None`` inherits the resolved teacher
    # config; exact-dense always resolves to Q=0 and fixed-Q requires Q>=2.
    teacher_q_draws: int | None = None
    engineering_only: bool = False
    schema_version: str = STAGE_OPTIONS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != STAGE_OPTIONS_SCHEMA:
            raise ValueError("unknown StageOptions schema")
        object.__setattr__(self, "stage", _stage_name(self.stage))
        if self.arm not in ARMS:
            raise ValueError("arm must be u_only or u_plus_spectral")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        _positive_int("epochs", self.epochs, maximum=10000)
        _positive_int("batch_size", self.batch_size, maximum=1_000_000)
        if self.max_updates is not None:
            _nonnegative_int("max_updates", self.max_updates, maximum=10_000_000)
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a nonempty string")
        if not math.isfinite(float(self.learning_rate)) or float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(float(self.weight_decay)) or float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.loss not in ("charbonnier", "charbonnier_ssim_gradient"):
            raise ValueError("loss must be charbonnier or charbonnier_ssim_gradient")
        _positive_int("query_chunk_size", self.query_chunk_size, maximum=1_000_000)
        _positive_int("candidate_chunk_size", self.candidate_chunk_size, maximum=1_000_000)
        _positive_int("accumulation_steps", self.accumulation_steps, maximum=1_000_000)
        if self.accumulation_steps != 1:
            raise ValueError("accumulation_steps other than 1 is not implemented by bounded stage services")
        object.__setattr__(self, "delta_weight", _finite_nonnegative("delta_weight", self.delta_weight))
        if self.query_mode not in ("exact_dense", "iid_fixed_q"):
            raise ValueError("query_mode must be exact_dense or iid_fixed_q")
        if self.teacher_q_draws is not None:
            _nonnegative_int("teacher_q_draws", self.teacher_q_draws, maximum=1_000_000)
            if self.query_mode == "exact_dense" and self.teacher_q_draws != 0:
                raise ValueError("exact_dense teacher_q_draws must be zero")
            if self.query_mode == "iid_fixed_q" and self.teacher_q_draws < 2:
                raise ValueError("iid_fixed_q teacher_q_draws must be at least 2")
        _positive_int("candidate_count", self.candidate_count, maximum=32)
        _positive_int("max_states_per_subject", self.max_states_per_subject, maximum=3)
        _positive_int("candidates_per_state", self.candidates_per_state, maximum=32)
        if self.output_format != "json":
            raise ValueError("only json stage receipts are supported")
        if not isinstance(self.semantic_objective, bool) or not isinstance(self.engineering_only, bool):
            raise TypeError("semantic_objective and engineering_only must be bool")
        for name in ("ssim_weight", "gradient_weight", "semantic_weight"):
            object.__setattr__(self, name, _finite_nonnegative(name, getattr(self, name)))

    def as_dict(self) -> dict[str, Any]:
        return {field.name: _jsonable(getattr(self, field.name)) for field in fields(self)}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "StageOptions":
        if not isinstance(values, Mapping):
            raise TypeError("StageOptions must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown StageOptions keys: {sorted(unknown)}")
        return cls(**dict(values))


@dataclass(frozen=True)
class StageExecutionConfig:
    """Strict execution wrapper around W1's PFGRLiteConfig.

    ``PFGRLiteConfig`` remains authoritative for protocol dimensions.  This
    sidecar carries the frontend serialization and measured normalization
    fields used by data/provenance joins; it is intentionally not a checkpoint
    envelope.
    """

    config: PFGRLiteConfig
    frontend_sidecar: Mapping[str, Any]
    normalization: Mapping[str, Any]
    stage_options: StageOptions
    schema_version: str = EXECUTION_CONFIG_SCHEMA
    normalization_hash: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != EXECUTION_CONFIG_SCHEMA:
            raise ValueError("unknown StageExecutionConfig schema")
        if not isinstance(self.config, PFGRLiteConfig):
            raise TypeError("execution config.config must be PFGRLiteConfig")
        if not isinstance(self.frontend_sidecar, Mapping):
            raise TypeError("frontend_sidecar must be a mapping")
        sidecar = dict(self.frontend_sidecar)
        # A full serialized PointGuidedConfig is validated by its owning W1
        # parser.  The factory-only sidecar permits only provenance/identity
        # metadata; silently retaining arbitrary keys would make execution
        # configuration non-reproducible.
        if "config" in sidecar:
            from .config import frontend_config_from_dict

            frontend_config_from_dict(sidecar)
        else:
            allowed_sidecar = {
                "schema_version",
                "checkpoint_path",
                "checkpoint_id",
                "medicalnet_checkpoint_path",
                "medicalnet_checkpoint_sha256",
                "frontend_config_hash",
                "model_config_hash",
                "model_family",
                "query_lattice_version",
                "query_lattice_hash",
                "writer_version",
                "writer_hash",
                "source_id",
            }
            unknown_sidecar = set(sidecar) - allowed_sidecar
            if unknown_sidecar:
                raise ValueError(f"unknown frontend sidecar keys: {sorted(unknown_sidecar)}")
        if not isinstance(self.normalization, Mapping):
            raise TypeError("normalization must be a mapping")
        if not isinstance(self.stage_options, StageOptions):
            raise TypeError("stage_options must be StageOptions")
        norm = dict(self.normalization)
        allowed = {
            "policy",
            "normalization_policy",
            "brain_mask_threshold",
            "normalization_epsilon",
            "lower_percentile",
            "upper_percentile",
            "mask_version",
            "range",
            "identity",
            "recipe_identity",
            "producer_identity",
            "schema_version",
        }
        unknown = set(norm) - allowed
        if unknown:
            raise ValueError(f"unknown normalization keys: {sorted(unknown)}")
        measured = dict(norm)
        measured.setdefault("policy", measured.get("normalization_policy", "masked_zscore"))
        measured.setdefault("schema_version", "pfgr-observation-normalization-v1")
        fields_for_hash = {name: measured[name] for name in ("brain_mask_threshold", "normalization_epsilon", "normalization_policy", "lower_percentile", "upper_percentile", "mask_version", "range") if name in measured}
        if "policy" in measured and "normalization_policy" not in fields_for_hash:
            fields_for_hash["normalization_policy"] = measured["policy"]
        # Keep both identities explicit: the recipe hash is derived from the
        # actual loader fields, while PFGR ProducerCompatibility hashes that
        # resolved recipe identifier once more as its producer-bound value.
        # This mirrors PFGRLiteModel._producer_dependencies exactly and avoids
        # binding a bank to an unrelated arbitrary policy string.
        measured_hash = normalization_identity(config=fields_for_hash)
        producer_hash = canonical_digest(measured_hash, prefix="pfgr-lite-observation-normalization-v1|")
        declared = norm.get("identity")
        if declared is not None and declared not in {measured_hash, producer_hash}:
            raise ValueError("normalization identity does not match measured fields")
        norm.setdefault("recipe_identity", measured_hash)
        norm.setdefault("producer_identity", producer_hash)
        object.__setattr__(self, "normalization_hash", producer_hash)
        object.__setattr__(self, "frontend_sidecar", MappingProxyType(sidecar))
        object.__setattr__(self, "normalization", MappingProxyType(norm))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pfgr_config": self.config.as_dict(),
            "frontend_sidecar": _jsonable(self.frontend_sidecar),
            "normalization": _jsonable(self.normalization),
            "normalization_hash": self.normalization_hash,
            "stage_options": self.stage_options.as_dict(),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "StageExecutionConfig":
        if not isinstance(values, Mapping):
            raise TypeError("StageExecutionConfig must be a mapping")
        allowed = {"schema_version", "pfgr_config", "frontend_sidecar", "normalization", "stage_options", "normalization_hash"}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown StageExecutionConfig keys: {sorted(unknown)}")
        if values.get("schema_version", EXECUTION_CONFIG_SCHEMA) != EXECUTION_CONFIG_SCHEMA:
            raise ValueError("unknown StageExecutionConfig schema")
        config = values.get("pfgr_config")
        if not isinstance(config, PFGRLiteConfig):
            config = PFGRLiteConfig.from_dict(config or {})
        options = values.get("stage_options")
        if not isinstance(options, StageOptions):
            options = StageOptions.from_dict(options or {})
        result = cls(
            config=config,
            frontend_sidecar=values.get("frontend_sidecar", {}),
            normalization=values.get("normalization", {}),
            stage_options=options,
        )
        declared_hash = values.get("normalization_hash")
        if declared_hash is not None and declared_hash != result.normalization_hash:
            raise ValueError("normalization_hash does not match resolved recipe")
        return result


@dataclass(frozen=True)
class StageInputs:
    """Explicit dependency-injection bundle consumed by :func:`run_stage`."""

    samples: Sequence[TargetFreeSample] = ()
    model: object | None = None
    producer: ProducerDependencies | object | None = None
    execution: StageExecutionConfig | None = None
    config: PFGRLiteConfig | None = None
    stage_options: StageOptions | None = None
    frontend_sidecar: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    normalization: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    role_manifest: TrainingRoleManifest | None = None
    static_step: Callable[..., Any] | None = None
    route_builder: Callable[..., Any] | None = None
    behavior_builder: Callable[..., Any] | None = None
    proposal_builder: Callable[..., Any] | None = None
    effect_measure: Callable[..., Any] | None = None
    target_provider: Callable[..., Any] | None = None
    bank_writer: Callable[..., Any] | None = None
    value_fitter: Callable[..., Any] | None = None
    calibration_fitter: Callable[..., Any] | None = None
    evaluator: Callable[..., Any] | None = None
    query: Callable[..., Any] | None = None
    writer: Callable[..., Any] | None = None
    decoder: object | None = None
    optimizer: torch.optim.Optimizer | None = None
    resume: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.samples, Sequence):
            raise TypeError("StageInputs.samples must be a sequence")
        for sample in self.samples:
            if not isinstance(sample, TargetFreeSample):
                raise TypeError("StageInputs samples must be TargetFreeSample values")
        if self.execution is not None and not isinstance(self.execution, StageExecutionConfig):
            raise TypeError("execution must be StageExecutionConfig")
        if self.config is not None and not isinstance(self.config, PFGRLiteConfig):
            raise TypeError("config must be PFGRLiteConfig")
        if self.stage_options is not None and not isinstance(self.stage_options, StageOptions):
            raise TypeError("stage_options must be StageOptions")
        if self.role_manifest is not None and not isinstance(self.role_manifest, TrainingRoleManifest):
            raise TypeError("role_manifest must be TrainingRoleManifest")
        if self.optimizer is not None and not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch optimizer")
        metadata = dict(self.metadata) if isinstance(self.metadata, Mapping) else {}
        if "counters" not in metadata:
            # Every stage gets a real counter object so receipts never infer
            # target/observation I/O from the number of preconstructed
            # samples.  ``build_stage_inputs`` passes this same object to the
            # legacy loader; direct engineering fixtures use it for deferred
            # target joins as well.
            from .data import DataAccessCounters

            metadata["counters"] = DataAccessCounters()
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


@dataclass(frozen=True)
class StageReceipt:
    """Stable on-disk receipt; tensors are represented only by hashes."""

    stage: str
    status: Literal["complete", "engineering_only", "failed"]
    optimizer_groups: tuple[str, ...] = ()
    subjects: int = 0
    route_updates: int = 0
    gradient_steps: int = 0
    target_reads: int = 0
    observation_reads: int = 0
    metrics: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    stage_provenance: Mapping[str, Any] | None = None
    error: str | None = None
    schema_version: str = STAGE_RECEIPT_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        if self.schema_version != STAGE_RECEIPT_SCHEMA:
            raise ValueError("unknown StageReceipt schema")
        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "status": self.status,
            "optimizer_groups": list(self.optimizer_groups),
            "subjects": self.subjects,
            "route_updates": self.route_updates,
            "gradient_steps": self.gradient_steps,
            "target_reads": self.target_reads,
            "observation_reads": self.observation_reads,
            "metrics": _jsonable(self.metrics),
            "stage_provenance": _jsonable(self.stage_provenance),
            "error": self.error,
        }


@dataclass(frozen=True)
class StageResult:
    """Returned stage artifact with W1 StageState compatibility properties."""

    stage_state: StageState
    receipt: StageReceipt
    output_dir: Path
    runtime_state: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def stage(self) -> str:
        return self.stage_state.stage

    @property
    def substage(self) -> str:
        return self.stage_state.substage

    @property
    def epoch(self) -> int:
        return self.stage_state.epoch

    @property
    def update(self) -> int:
        return self.stage_state.update

    @property
    def microstep(self) -> int:
        return self.stage_state.microstep

    @property
    def optimizer_groups(self) -> tuple[str, ...]:
        return self.stage_state.optimizer_groups

    @property
    def completion(self) -> str:
        return self.stage_state.completion

    @property
    def metrics(self) -> Mapping[str, Any]:
        return self.receipt.metrics

    def as_stage_state(self) -> StageState:
        return self.stage_state


def _module(model: object | None, names: Sequence[str]) -> nn.Module | None:
    if model is None:
        return None
    for name in names:
        value: Any = model
        for part in name.split("."):
            value = getattr(value, part, None)
            if value is None:
                break
        if isinstance(value, nn.Module):
            return value
    return None


def _producer_hash(producer: object | None, model: object | None = None) -> str:
    if producer is not None:
        for name in ("compatibility_hash", "digest"):
            value = getattr(producer, name, None)
            if isinstance(value, str) and value:
                return value
        compatibility = getattr(producer, "compatibility", None)
        value = getattr(compatibility, "digest", None)
        if isinstance(value, str) and value:
            return value
    value = getattr(getattr(model, "producer_dependencies", None), "compatibility_hash", None)
    if isinstance(value, str) and value:
        return value
    # PFGRLiteModel keeps producer construction private because the final
    # compatibility envelope is geometry/context-bound.  Its private helper
    # nevertheless derives compatibility fields solely from module/config
    # state (the geometry object contributes only the locked version), so a
    # resume may validate the identity before the next one-traversal route.
    producer_builder = getattr(model, "_producer_dependencies", None)
    backbone = getattr(getattr(model, "frontend", None), "semantic_prior", None)
    backbone = getattr(backbone, "backbone", None)
    if callable(producer_builder) and isinstance(backbone, nn.Module):
        try:
            from smagm.features.point_guided.contracts import VolumeGeometry
            from .static_geometry import derive_multiscale_feature_geometry

            geometry = VolumeGeometry.from_spacing((9, 9, 9))
            geometries = derive_multiscale_feature_geometry(backbone, geometry)
            candidate = producer_builder(geometries=geometries, traversal_count=1)
            value = getattr(candidate, "compatibility_hash", None)
            if isinstance(value, str) and value:
                return value
        except Exception:
            # A model with an incomplete synthetic frontend still receives
            # the explicit engineering fallback below; production callers
            # fail closed in _restore_runtime when no current producer exists.
            pass
    return canonical_digest("engineering-producer", prefix="pfgr-lite-engineering-producer-v1|")


def _producer_hash_after_updates(
    inputs: StageInputs,
    *,
    observed_producer: object | None,
    spectral: nn.Module | None,
    updater: nn.Module | None,
) -> str:
    """Refresh compatibility fields after an optimizer step.

    Observation contexts intentionally retain their pre-step producer digest
    and therefore become stale after S1.  The stage receipt must bind the
    *post*-step hashes rather than reusing that old context identity.  This
    helper preserves every unrelated compatibility field and only replaces
    the modules that S1 can update (U and the shared spectral projector).
    """

    base = observed_producer or inputs.producer
    if base is None:
        base = getattr(inputs.model, "producer_dependencies", None)
    compatibility = getattr(base, "compatibility", base)
    if isinstance(compatibility, ProducerCompatibility):
        updates: dict[str, str] = {}
        if spectral is not None:
            updates["spectral_projector_hash"] = module_state_digest(spectral)
        if updater is not None:
            updates["updater_hash"] = module_state_digest(updater)
        if updates:
            compatibility = replace(compatibility, **updates)
            return compatibility.digest
    return _producer_hash(base, inputs.model)


def _producer_hash_after_static_updates(
    inputs: StageInputs,
    *,
    observed_producer: object | None,
    static_head: nn.Module | None,
    decoder: nn.Module | None,
    semantic_head: nn.Module | None,
) -> str:
    """Refresh the producer identity after S0's authorized optimizer step.

    S0 updates the static synthesis/state initializer and decoder (and the
    optional semantic head).  The completed target-free context still carries
    the pre-step compatibility envelope, so runtime state must bind a
    post-step digest rather than making that stale context appear resumable.
    """

    base = observed_producer or inputs.producer
    if base is None:
        base = getattr(inputs.model, "producer_dependencies", None)
    compatibility = getattr(base, "compatibility", base)
    if isinstance(compatibility, ProducerCompatibility):
        updates: dict[str, str] = {}
        if static_head is not None:
            updates["static_head_hash"] = module_state_digest(static_head)
            final_projection = getattr(static_head, "final_projection", None)
            if isinstance(final_projection, nn.Module):
                updates["state_initializer_hash"] = module_state_digest(final_projection)
        if decoder is not None:
            updates["decoder_hash"] = module_state_digest(decoder)
        if semantic_head is not None:
            updates["semantic_head_hash"] = module_state_digest(semantic_head)
        if updates:
            return replace(compatibility, **updates).digest
    return _producer_hash(base, inputs.model)


def _freeze_model(model: object | None, trainable: Sequence[nn.Module]) -> dict[int, bool]:
    states: dict[int, bool] = {}
    if not isinstance(model, nn.Module):
        return states
    trainable_ids = {id(parameter) for module in trainable for parameter in module.parameters()}
    for parameter in model.parameters():
        states[id(parameter)] = bool(parameter.requires_grad)
        parameter.requires_grad_(id(parameter) in trainable_ids)
    return states


def _module_parameters(modules: Sequence[nn.Module]) -> list[nn.Parameter]:
    result: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                result.append(parameter)
                seen.add(id(parameter))
    return result


def _optimizer_for(inputs: StageInputs, modules: Sequence[nn.Module], options: StageOptions) -> torch.optim.Optimizer | None:
    expected_ids = {id(parameter) for parameter in _module_parameters(modules)}
    if inputs.optimizer is not None:
        actual_ids = {id(parameter) for group in inputs.optimizer.param_groups for parameter in group["params"]}
        if actual_ids != expected_ids:
            raise ValueError("provided optimizer parameter ownership does not match stage trainable modules")
        return inputs.optimizer
    if not expected_ids:
        return None
    return torch.optim.Adam(_module_parameters(modules), lr=float(options.learning_rate), weight_decay=float(options.weight_decay))


def _resume_value(resume: object | None, name: str, default: Any = None) -> Any:
    if resume is None:
        return default
    if isinstance(resume, Mapping):
        return resume.get(name, default)
    return getattr(resume, name, default)


def _parameter_names(modules: Sequence[nn.Module]) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[int] = set()
    for module_index, module in enumerate(modules):
        for name, parameter in module.named_parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            names.append(f"module{module_index}.{name}")
    return tuple(names)


def _rng_snapshot(local_rng: random.Random | None = None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().detach().cpu().clone(),
    }
    try:
        import numpy as np

        snapshot["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        snapshot["torch_cuda"] = tuple(item.detach().cpu().clone() for item in torch.cuda.get_rng_state_all())
    return snapshot


def _clone_runtime_value(value: Any) -> Any:
    """Clone runtime snapshots without retaining graph/device references."""

    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _clone_runtime_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_runtime_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_runtime_value(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.copy()
    except ImportError:
        pass
    return copy.deepcopy(value)


def _full_execution_identity(execution: StageExecutionConfig) -> str:
    """Hash the complete execution envelope, including the stop cap."""

    return canonical_digest(execution.as_dict(), prefix="pfgr-lite-execution-config-v1|")


def _execution_identity(execution: StageExecutionConfig) -> str:
    """Hash immutable execution semantics while permitting a larger resume cap."""

    payload = execution.as_dict()
    stage_options = dict(payload.get("stage_options", {}))
    # ``max_updates`` is a caller's bounded stop target, not a producer/model
    # identity.  A resumed invocation may raise that cap after restoring its
    # cursor; all other strict option fields remain part of the identity.
    stage_options["max_updates"] = None
    payload["stage_options"] = stage_options
    return canonical_digest(payload, prefix="pfgr-lite-training-config-v1|")


def _stage_state_payload(
    stage: str,
    execution: StageExecutionConfig,
    inputs: StageInputs,
    *,
    epoch: int,
    update: int,
    microstep: int,
    optimizer_groups: Sequence[str],
) -> dict[str, Any]:
    options = execution.stage_options
    if stage in {"S0", "S1"} and inputs.samples:
        batches = max(1, math.ceil(len(inputs.samples) / options.batch_size))
        expected = options.epochs * batches
        completion = "complete" if int(update) >= expected else "pending"
    else:
        completion = "complete"
    return {
        "stage": stage,
        "substage": "complete",
        "epoch": int(epoch),
        "update": int(update),
        "microstep": int(microstep),
        "optimizer_groups": tuple(str(name) for name in optimizer_groups),
        "completion": completion,
        "version": "pfgr-lite-stage-state-v1",
    }


def _require_runtime_mapping(resume: object) -> Mapping[str, Any]:
    if not isinstance(resume, Mapping):
        raise TypeError("resume runtime state must be a mapping")
    required = {
        "schema_version",
        "stage_state",
        "optimizer_state",
        "rng_state",
        "cursor",
        "parameter_names",
        "execution_config_hash",
        "training_config_hash",
        "producer_compatibility_hash",
        "split_role_hash",
        "execution_config",
        "input_manifest_hash",
    }
    optional = {"continuation"}
    missing = required - set(resume)
    unknown = set(resume) - required - optional
    if missing:
        raise ValueError(f"resume runtime state missing required fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"resume runtime state has unknown fields: {sorted(unknown)}")
    if resume["schema_version"] != STAGE_RUNTIME_SCHEMA:
        raise ValueError("unknown stage runtime schema")
    return resume


def _restore_runtime(
    inputs: StageInputs,
    execution: StageExecutionConfig,
    optimizer: torch.optim.Optimizer | None,
    modules: Sequence[nn.Module],
    *,
    stage: str,
) -> tuple[int, int, int, int, tuple[str, ...]]:
    """Restore strict runtime state before any stage sampling/route work."""

    resume = inputs.resume
    if resume is None:
        return 0, 0, 0, 0, tuple(sample.subject_id for sample in inputs.samples)
    resume = _require_runtime_mapping(resume)
    resume_stage = resume["stage_state"]
    if not isinstance(resume_stage, Mapping):
        raise TypeError("resume stage_state must be a mapping")
    stage_keys = {
        "stage",
        "substage",
        "epoch",
        "update",
        "microstep",
        "optimizer_groups",
        "completion",
        "version",
    }
    if set(resume_stage) != stage_keys:
        raise ValueError("resume stage_state keys are incomplete or unknown")
    if resume_stage["version"] != "pfgr-lite-stage-state-v1":
        raise ValueError("unknown resume StageState version")
    resume_stage_name = resume_stage["stage"]
    if str(resume_stage_name).upper() != stage:
        raise ValueError("resume stage does not match requested stage")
    try:
        StageState(
            stage=str(resume_stage_name),
            substage=str(resume_stage["substage"]),
            epoch=int(resume_stage["epoch"]),
            update=int(resume_stage["update"]),
            microstep=int(resume_stage["microstep"]),
            optimizer_groups=tuple(str(item) for item in resume_stage["optimizer_groups"]),
            completion=str(resume_stage["completion"]),
            version=str(resume_stage["version"]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("resume stage_state is invalid") from exc
    expected_names = _parameter_names(modules)
    saved_names = resume["parameter_names"]
    if not isinstance(saved_names, (tuple, list)) or tuple(saved_names) != expected_names:
        raise ValueError("resume optimizer parameter ownership does not match stage modules")
    saved_execution_hash = resume["execution_config_hash"]
    saved_training_hash = resume["training_config_hash"]
    current_execution_hash = _full_execution_identity(execution)
    current_training_hash = _execution_identity(execution)
    if not isinstance(saved_execution_hash, str) or saved_execution_hash != current_execution_hash:
        # A changed bounded stop cap is the sole permitted continuation
        # override; the immutable training identity must still match.
        if not isinstance(saved_training_hash, str) or saved_training_hash != current_training_hash:
            raise ValueError("resume execution/training configuration identity does not match")
        saved_config = resume.get("execution_config")
        old_options = saved_config.get("stage_options", {}) if isinstance(saved_config, Mapping) else {}
        old_cap = old_options.get("max_updates") if isinstance(old_options, Mapping) else None
        new_cap = execution.stage_options.max_updates
        if old_cap == new_cap:
            raise ValueError("resume execution configuration identity does not match")
    elif not isinstance(saved_training_hash, str) or saved_training_hash != current_training_hash:
        raise ValueError("resume training configuration identity does not match")
    saved_producer = resume["producer_compatibility_hash"]
    if saved_producer is not None and not isinstance(saved_producer, str):
        raise TypeError("resume producer_compatibility_hash must be a string or null")
    current_source = inputs.producer or getattr(inputs.model, "producer_dependencies", None)
    if saved_producer is not None:
        if current_source is None and not execution.stage_options.engineering_only:
            raise ValueError("production resume requires current producer compatibility dependencies")
        current_producer = _producer_hash(current_source, inputs.model)
        if saved_producer != current_producer:
            raise ValueError("resume producer compatibility identity does not match")
    saved_split = resume["split_role_hash"]
    if saved_split is not None and not isinstance(saved_split, str):
        raise TypeError("resume split_role_hash must be a string or null")
    current_split = inputs.role_manifest.digest if inputs.role_manifest is not None else None
    if saved_split is not None and current_split is None:
        raise ValueError("resume split-role identity requires the current role manifest")
    if saved_split is not None and current_split is not None and saved_split != current_split:
        raise ValueError("resume split-role identity does not match")
    saved_input_manifest = resume["input_manifest_hash"]
    if not isinstance(saved_input_manifest, str) or saved_input_manifest != _input_manifest_hash(inputs):
        raise ValueError("resume input observation identity does not match StageInputs")
    # Validate the complete cursor before touching optimizer or process RNG;
    # malformed snapshots must be side-effect free.
    cursor = resume["cursor"]
    if not isinstance(cursor, Mapping):
        raise TypeError("resume cursor must be a mapping")
    cursor_keys = {"epoch", "batch_index", "update", "microstep", "sample_order", "route_rng_state"}
    if set(cursor) != cursor_keys:
        raise ValueError("resume cursor keys are incomplete or unknown")
    epoch = int(cursor["epoch"])
    batch_index = int(cursor["batch_index"])
    updates = int(cursor["update"])
    measured = updates
    order_value = cursor["sample_order"]
    if not isinstance(order_value, (tuple, list)):
        raise TypeError("resume sample_order must be a sequence")
    order = tuple(str(item) for item in order_value)
    if order != tuple(sample.subject_id for sample in inputs.samples):
        raise ValueError("resume sample order does not match StageInputs")
    microstep = int(cursor["microstep"])
    if epoch < 0 or batch_index < 0 or updates < 0 or measured < 0 or microstep < 0:
        raise ValueError("resume cursor values must be nonnegative")
    route_state = cursor["route_rng_state"]
    if route_state is not None:
        route_state = _tupleize_rng_state(route_state)
        try:
            random.Random().setstate(route_state)
        except (TypeError, ValueError) as exc:
            raise ValueError("resume route_rng_state is invalid") from exc
    optimizer_state = resume["optimizer_state"]
    if not isinstance(optimizer_state, Mapping):
        raise TypeError("resume optimizer_state must be a mapping")
    rng_state = resume["rng_state"]
    if not isinstance(rng_state, Mapping):
        raise TypeError("resume rng_state must be a mapping")
    if not execution.stage_options.engineering_only and updates > 0:
        if saved_producer is None or saved_split is None:
            raise ValueError("production resume after updates requires producer and split-role identities")
        if not optimizer_state or not rng_state:
            raise ValueError("production resume after updates requires optimizer and RNG snapshots")
    if optimizer_state:
        if optimizer is None:
            raise ValueError("resume contains optimizer state but stage has no optimizer")
        optimizer.load_state_dict(_clone_runtime_value(dict(optimizer_state)))
    if rng_state:
        from .checkpoint import restore_rng_state

        restore_rng_state(rng_state)
    return epoch, batch_index, updates, measured, order


def _runtime_state(
    inputs: StageInputs,
    execution: StageExecutionConfig,
    optimizer: torch.optim.Optimizer | None,
    modules: Sequence[nn.Module],
    *,
    stage: str,
    epoch: int,
    batch_index: int,
    update: int,
    measured_steps: int,
    local_rng: random.Random | None = None,
    producer_hash: str | None = None,
    optimizer_groups: Sequence[str] | None = None,
) -> dict[str, Any]:
    del measured_steps  # runtime cursor tracks optimizer updates exactly
    groups = tuple(optimizer_groups or ())
    update_state = _stage_state_payload(stage, execution, inputs, epoch=epoch, update=update, microstep=0, optimizer_groups=groups)
    full_hash = _full_execution_identity(execution)
    training_hash = _execution_identity(execution)
    runtime: dict[str, Any] = {
        "schema_version": STAGE_RUNTIME_SCHEMA,
        "stage_state": update_state,
        "optimizer_state": {} if optimizer is None else _clone_runtime_value(optimizer.state_dict()),
        "rng_state": _clone_runtime_value(_rng_snapshot()),
        "cursor": {
            "epoch": int(epoch),
            "batch_index": int(batch_index),
            "update": int(update),
            "microstep": 0,
            "sample_order": tuple(sample.subject_id for sample in inputs.samples),
            "route_rng_state": None if local_rng is None else _clone_runtime_value(local_rng.getstate()),
        },
        "parameter_names": _parameter_names(modules),
        "execution_config_hash": full_hash,
        "training_config_hash": training_hash,
        "producer_compatibility_hash": producer_hash or _producer_hash(inputs.producer, inputs.model),
        "split_role_hash": inputs.role_manifest.digest if inputs.role_manifest is not None else None,
        "execution_config": execution.as_dict(),
        "input_manifest_hash": _input_manifest_hash(inputs),
    }
    resume = inputs.resume
    if isinstance(resume, Mapping):
        old_hash = resume.get("execution_config_hash")
        old_training = resume.get("training_config_hash")
        if isinstance(old_hash, str) and old_hash != full_hash and old_training == training_hash:
            old_config = resume.get("execution_config")
            old_options = old_config.get("stage_options", {}) if isinstance(old_config, Mapping) else {}
            old_cap = old_options.get("max_updates") if isinstance(old_options, Mapping) else None
            runtime["continuation"] = {
                "resumed_from_execution_config_hash": old_hash,
                "execution_config_hash": full_hash,
                "training_config_hash": training_hash,
                "max_updates_override": {"previous": old_cap, "requested": execution.stage_options.max_updates},
            }
    return runtime


def _restore_requires_grad(model: object | None, states: Mapping[int, bool]) -> None:
    if not isinstance(model, nn.Module):
        return
    for parameter in model.parameters():
        if id(parameter) in states:
            parameter.requires_grad_(states[id(parameter)])


def _target_context_for(
    sample: TargetFreeSample,
    result: object,
    inputs: StageInputs,
    *,
    engineering_only: bool = False,
    include_segmentation: bool = False,
) -> object:
    if isinstance(result, Mapping):
        direct = result.get("target_context")
        completed = result.get("context")
        prediction = result.get("prediction", result.get("predictions"))
        trace = result.get("trace")
    else:
        direct = getattr(result, "target_context", None)
        completed = getattr(result, "context", None)
        prediction = getattr(result, "prediction", getattr(result, "predictions", None))
        trace = result
    if direct is not None and not include_segmentation:
        return direct
    if inputs.target_provider is None:
        if direct is not None:
            return direct
        raise ValueError("stage result has no target context and no deferred target_provider")
    counter = inputs.metadata.get("counters") if isinstance(inputs.metadata, Mapping) else None
    callback = defer_supervision(
        sample,
        inputs.target_provider,
        counters=counter,
        engineering_only=engineering_only,
        include_segmentation=include_segmentation,
    )
    joined = callback(completed_context=completed, prediction=prediction, trace=trace)
    if include_segmentation and direct is not None and isinstance(joined, Mapping):
        # Preserve any explicitly returned semantic metadata while retaining
        # the validated late target/segmentation join.
        merged = dict(joined)
        if isinstance(direct, Mapping):
            if "semantic_target" in direct and "semantic_target" not in merged:
                merged["semantic_target"] = direct["semantic_target"]
        return merged
    return joined


def _access_count(inputs: StageInputs, name: str, default: int = 0) -> int:
    counter = inputs.metadata.get("counters") if isinstance(inputs.metadata, Mapping) else None
    value = getattr(counter, name, None) if counter is not None else None
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else int(default)


def _operation_counters(inputs: StageInputs) -> OperationCounters:
    """Return the explicit W2/W4 counter sink for one stage invocation."""

    candidate = inputs.metadata.get("operation_counters") if isinstance(inputs.metadata, Mapping) else None
    if candidate is None:
        return OperationCounters()
    if not isinstance(candidate, OperationCounters):
        raise TypeError("metadata['operation_counters'] must be OperationCounters")
    return candidate


def _counter_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, int | str]:
    result: dict[str, int | str] = {"schema_version": str(after.get("schema_version", "pfgr-lite-operation-counters-v1"))}
    for name, value in after.items():
        if name == "schema_version":
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            prior = before.get(name, 0)
            result[name] = int(value - prior) if isinstance(prior, int) and not isinstance(prior, bool) else int(value)
    return result


def _input_manifest_hash(inputs: StageInputs) -> str:
    """Bind resume state to immutable observation content, not only IDs."""

    return canonical_digest(
        {
            "schema_version": "pfgr-lite-input-manifest-v1",
            "samples": [
                {"subject_id": sample.subject_id, "observation_record_id": sample.observation_record_id}
                for sample in inputs.samples
            ],
        },
        prefix="pfgr-lite-input-manifest-v1|",
    )


def _tupleize_rng_state(value: object) -> object:
    if isinstance(value, list):
        return tuple(_tupleize_rng_state(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_tupleize_rng_state(item) for item in value)
    return value


def _default_static_step(sample: TargetFreeSample, *, model: object, options: StageOptions, inputs: StageInputs) -> Mapping[str, Any]:
    """Run the real model's static encode/decode path for one sample."""

    encode = getattr(model, "encode_observations", None)
    initialize = getattr(model, "initialize_state", None)
    decode = getattr(model, "decode_final", None)
    if not callable(encode) or not callable(initialize) or not callable(decode):
        raise ValueError("default S0 requires PFGRLiteModel encode_observations/initialize_state/decode_final")
    observations = sample.observations.unsqueeze(0).to(device=options.device)
    mask = sample.brain_mask.to(device=observations.device)
    context = encode(observations, mask, sample.geometry)
    state = initialize(context, role="training_behavior")
    prediction = decode(state, context, chunk_size=options.query_chunk_size)
    # ``ObservationContext.frontend.s_coarse`` is the one-traversal semantic
    # probability map.  Keep it graph-connected for the explicit S0 semantic
    # arm; the target/label join remains deferred until after this prediction.
    semantic_probabilities = getattr(getattr(context, "frontend", None), "s_coarse", None)
    return {
        "prediction": prediction,
        "context": context,
        "semantic_probabilities": semantic_probabilities,
        "subject_context_binding": bind_observation_context(sample, context),
    }


def _default_random_route(
    sample: TargetFreeSample,
    *,
    model: object,
    options: StageOptions,
    inputs: StageInputs,
    k: int,
    seed: int,
    counters: OperationCounters | None = None,
) -> Mapping[str, Any]:
    """Execute a complete target-free random route using W2/W4 injections."""

    encode = getattr(model, "encode_observations", None)
    initialize = getattr(model, "initialize_state", None)
    if not callable(encode) or not callable(initialize):
        raise ValueError("default S1 route requires PFGRLiteModel encode_observations/initialize_state")
    if inputs.query is None or inputs.writer is None:
        raise ValueError("default S1 route requires explicit W2 query and writer injections")
    from .action_proposal import apply_scored_action, propose_actions
    # W4's canonical point-query adapter carries an explicit identity.  Tiny
    # engineering callers often inject the same tensor-producing callable as
    # a lambda (or a DynamicStatePointQuery wrapper) without those attributes;
    # bind the point-query identity at this seam rather than allowing
    # propose_actions' geometry fallback to disagree with StoredActionWriter.
    # The wrapper is intentionally output-preserving and never detaches the
    # queried state, so gradients through U/query/write remain live.
    query = inputs.query
    if not all(isinstance(getattr(query, name, None), str) and getattr(query, name) for name in ("query_version", "query_hash")):
        from .sparse_write import POINT_QUERY_HASH, POINT_QUERY_VERSION

        original_query = query

        class _BoundPointQuery:
            query_version = POINT_QUERY_VERSION
            query_hash = POINT_QUERY_HASH

            def __call__(self, state: object, points: Tensor, feature_geometry: object) -> Tensor:
                value = original_query(state, points, feature_geometry)
                packed = getattr(value, "packed", value)
                if not isinstance(packed, Tensor):
                    raise TypeError("injected point query must return a tensor or DynamicPointSamples.packed")
                return packed

        query = _BoundPointQuery()

    # W2's compact writer exposes the only supported candidate-legality
    # contract.  It is observation/geometry-only and must be applied before a
    # zero-initialized U is allowed to emit an explicit zero write.  Direct
    # engineering fixtures may omit the adapter; in that case proposal
    # legality remains the nonzero-delta diagnostic emitted by W4, but no
    # all-ones fallback is ever synthesized.
    support_legal_mask = inputs.metadata.get("support_legal_mask") if isinstance(inputs.metadata, Mapping) else None
    if support_legal_mask is not None and not callable(support_legal_mask) and not isinstance(support_legal_mask, Tensor):
        raise TypeError("metadata['support_legal_mask'] must be a callable or tensor")

    observations = sample.observations.unsqueeze(0).to(device=options.device)
    context = encode(observations, sample.brain_mask.to(device=observations.device), sample.geometry)
    state = initialize(context, role="training_behavior")
    decode = getattr(model, "decode_final", None)
    if not callable(decode):
        raise ValueError("default S1 route requires PFGRLiteModel.decode_final")
    # The initial baseline is measured once on the same target-free route but
    # detached from the updater objective; final/intermediate predictions below
    # remain differentiable through U and the frozen writer/decoder path.
    with torch.no_grad():
        initial_prediction = _prediction_from(decode(state, context, chunk_size=options.query_chunk_size))
    rng = random.Random(seed)
    states: list[Any] = [state]
    proposals: list[Any] = []
    decisions: list[Decision] = []
    predictions: list[Tensor] = []
    deltas: list[Tensor] = []
    for step in range(k):
        support_legal: Tensor | None = None
        if callable(support_legal_mask):
            support_legal = support_legal_mask(state, context, context.frontend.refined_points_ras_mm)
            if not isinstance(support_legal, Tensor):
                raise TypeError("support_legal_mask must return a tensor")
            if support_legal.shape != (1, context.frontend.refined_points_ras_mm.shape[1]):
                raise ValueError("support_legal_mask must return [B,N] for the current context")
            if support_legal.dtype != torch.bool:
                if support_legal.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                    raise TypeError("support_legal_mask must be bool or integer 0/1")
                if not bool(((support_legal == 0) | (support_legal == 1)).all()):
                    raise ValueError("support_legal_mask integer values must be exact 0/1")
                support_legal = support_legal.to(dtype=torch.bool)
            if support_legal.device != context.frontend.refined_points_ras_mm.device:
                support_legal = support_legal.to(device=context.frontend.refined_points_ras_mm.device)
        elif isinstance(support_legal_mask, Tensor):
            support_legal = support_legal_mask
            expected_shape = (1, context.frontend.refined_points_ras_mm.shape[1])
            if support_legal.shape != expected_shape:
                raise ValueError("support_legal_mask tensor must match current [B,N] points")
            if support_legal.dtype != torch.bool:
                if support_legal.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                    raise TypeError("support_legal_mask must be bool or integer 0/1")
                if not bool(((support_legal == 0) | (support_legal == 1)).all()):
                    raise ValueError("support_legal_mask integer values must be exact 0/1")
                support_legal = support_legal.to(dtype=torch.bool)
            if support_legal.device != context.frontend.refined_points_ras_mm.device:
                support_legal = support_legal.to(device=context.frontend.refined_points_ras_mm.device)
        proposal = propose_actions(
            getattr(model, "updater", getattr(model, "update_net", None)),
            state,
            context,
            query=query,
            candidate_chunk_size=options.candidate_chunk_size,
            counters=counters,
            legal_mask=support_legal,
            write_scale=0.1,
        )
        if support_legal is not None:
            # ``propose_actions`` normally excludes exact-zero deltas.  S1's
            # explicit zero-U capability control may execute a zero write, but
            # only at nodes retained by the canonical writer-support mask.
            proposal = replace(proposal, legal=support_legal.to(device=proposal.legal.device), proposal_digest=None)
        legal_locations = proposal.legal.nonzero(as_tuple=False)
        if legal_locations.numel() == 0:
            raise RuntimeError("S1 route has no writer-support eligible candidate")
        choice = legal_locations[rng.randrange(int(legal_locations.shape[0]))]
        row = proposal.row(int(choice[0]), int(choice[1]))
        decision = Decision(
            selected_point_id=int(row.point_id),
            proposal_digest=proposal.proposal_digest,
            action_digest=row.action_digest,
            active=True,
            raw_value=0.0,
            calibrated_value=0.0,
            conservative_value=0.0,
            allowance=0.0,
            quality_margin=0.0,
            compute_cost=0.0,
            policy_hash=canonical_digest("random-stage-policy", prefix="pfgr-lite-stage-policy-v1|"),
            stop_code="continue",
            step=step,
        )
        next_state = apply_scored_action(state, context, proposal, decision, writer=inputs.writer, counters=counters)
        prediction = decode(next_state, context, chunk_size=options.query_chunk_size)
        states.append(next_state)
        proposals.append(proposal)
        decisions.append(decision)
        predictions.append(prediction)
        deltas.append(row.delta)
        state = next_state
    trace = CompletedBehaviorTrace(context.context_id, states=tuple(states), proposals=tuple(proposals), decisions=tuple(decisions))
    return {
        "predictions": tuple(predictions),
        "initial_prediction": initial_prediction,
        "states": tuple(states),
        "proposals": tuple(proposals),
        "decisions": tuple(decisions),
        "deltas": tuple(deltas),
        "trace": trace,
        "context": context,
        "subject_context_binding": bind_observation_context(sample, context),
    }


def _prediction_from(result: object) -> Tensor:
    value: Any = result
    if isinstance(result, Mapping):
        value = result.get("prediction", result.get("predictions", result.get("output")))
    else:
        value = getattr(result, "prediction", getattr(result, "predictions", result))
    if isinstance(value, (tuple, list)):
        value = value[-1]
    if not isinstance(value, Tensor):
        raise TypeError("stage callback must return a prediction tensor")
    if value.ndim == 4:
        value = value.unsqueeze(1)
    if value.ndim != 5 or value.shape[1] != 1 or not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError("prediction must be finite floating [B,1,D,H,W]")
    return value


def _route_predictions(result: object) -> tuple[Tensor, ...]:
    value: Any = result
    if isinstance(result, Mapping):
        value = result.get("predictions", result.get("route_predictions", result.get("outputs")))
    else:
        value = getattr(result, "predictions", getattr(result, "route_predictions", getattr(result, "outputs", None)))
    if value is None:
        value = getattr(result, "prediction", result if isinstance(result, Tensor) else None)
    if isinstance(value, Tensor):
        value = (value,)
    if value is None:
        raise TypeError("route callback must expose predictions")
    return tuple(_prediction_from(item) for item in value)


def _route_k(result: object) -> int:
    predictions = _route_predictions(result)
    if len(predictions) not in (1, 2, 4):
        raise ValueError("S1 routes must contain K in {1,2,4} post-write predictions")
    return len(predictions)


def _paired_dense_metrics(
    initial_prediction: object,
    final_prediction: object,
    target_context: object,
    *,
    sample: TargetFreeSample,
    context: object | None,
    budget: int,
    charbonnier_epsilon: float,
) -> dict[str, Any]:
    """Measure same-route D(Z0)/D(ZK) metrics after a validated target join."""

    initial = _prediction_from(initial_prediction).detach()
    final = _prediction_from(final_prediction).detach()
    owned = target_context
    if isinstance(owned, Mapping) and "target_context" in owned:
        owned = owned.get("target_context")
    target = owned.get("target") if isinstance(owned, Mapping) else getattr(owned, "target", None)
    mask = owned.get("observation_mask") if isinstance(owned, Mapping) else getattr(owned, "observation_mask", None)
    if not isinstance(target, Tensor):
        raise TypeError("completed target context must expose a target tensor for paired metrics")
    from .metrics import paired_subject_metrics

    with torch.no_grad():
        return paired_subject_metrics(
            initial,
            final,
            target.detach(),
            None if mask is None else mask.detach() if isinstance(mask, Tensor) else mask,
            data_range=1.0,
            charbonnier_epsilon=float(charbonnier_epsilon),
            ssim_window=11,
            subject_id=sample.subject_id,
            context_id=getattr(context, "context_id", None),
            scenario="S1_random_route",
            budget=int(budget),
        )


def _stage_provenance(
    inputs: StageInputs,
    options: StageOptions,
    *,
    gradient_norm: float,
    nonzero_steps: int,
    measured_steps: int,
    optimizer_steps: int,
    changed: int,
    before: str,
    after: str,
    completed: bool,
    producer_hash: str | None = None,
) -> dict[str, Any] | None:
    if options.stage not in {"S1", "S2", "S4"}:
        return None
    producer_hash = producer_hash or _producer_hash(inputs.producer, inputs.model)
    if options.arm == "u_only":
        if not inputs.metadata.get("verified_prior_receipt") or not inputs.metadata.get("verified_prior_receipt_hash"):
            # A cold U-only control is an engineering diagnostic, not a
            # canonical producer receipt that could unlock MAIN banking.
            return None
        spectral_arm = "verified_prior"
    else:
        spectral_arm = "u_plus_spectral"
    # Stage receipts are consumed as immutable producer provenance by S2/W5.
    # Never emit the historical ``none`` checkpoint sentinel: production
    # stages must carry the actual loaded checkpoint/source identity, while a
    # synthetic engineering receipt receives an explicit diagnostic identity.
    metadata = inputs.metadata if isinstance(inputs.metadata, Mapping) else {}
    initialization_id = metadata.get("initialization_id", "stage-initialization")
    checkpoint_id = metadata.get("checkpoint_id")
    source_id = metadata.get("source_id")
    if checkpoint_id is None or str(checkpoint_id).strip().lower() in {"", "none", "null", "unknown", "unset"}:
        if not options.engineering_only:
            raise ValueError("production stage provenance requires actual checkpoint_id from a loaded checkpoint")
        checkpoint_id = "engineering-initialization"
    if source_id is None or str(source_id).strip().lower() in {"", "none", "null", "unknown", "unset"}:
        if not options.engineering_only:
            raise ValueError("production stage provenance requires actual source_id")
        source_id = "engineering-source"
    if initialization_id is None or str(initialization_id).strip().lower() in {"", "none", "null", "unknown", "unset"}:
        if not options.engineering_only:
            raise ValueError("production stage provenance requires actual initialization_id")
        initialization_id = "engineering-initialization"
    split_role_hash = inputs.role_manifest.digest if inputs.role_manifest is not None else str(metadata.get("split_role_hash", "engineering"))
    role_manifest_digest = inputs.role_manifest.digest if inputs.role_manifest is not None else "engineering-role"
    if not options.engineering_only and inputs.role_manifest is None:
        raise ValueError("production stage provenance requires the reviewed TrainingRoleManifest")
    return {
        "schema_version": PRODUCER_STAGE_SCHEMA,
        "stage": "updater",
        "spectral_arm": spectral_arm,
        "completed": bool(completed),
        "producer_compatibility_hash": producer_hash,
        "projector_before_hash": before,
        "projector_after_hash": after,
        "projector_gradient_evidence": {
            "l2_norm_max": float(gradient_norm),
            "nonzero_steps": int(nonzero_steps),
            "measured_steps": int(measured_steps),
        },
        "projector_update_evidence": {
            "changed_parameter_count": int(changed),
            "optimizer_steps": int(optimizer_steps),
        },
        "initialization_id": str(initialization_id),
        "checkpoint_id": str(checkpoint_id),
        "source_id": str(source_id),
        "split_role_hash": split_role_hash,
        "role_manifest_digest": role_manifest_digest,
        "verified_prior_receipt": metadata.get("verified_prior_receipt"),
        "verified_prior_receipt_hash": metadata.get("verified_prior_receipt_hash"),
    }


def _write_receipt(output_dir: Path, receipt: StageReceipt, execution: StageExecutionConfig) -> None:
    path = output_dir / "stage_receipt.json"
    payload = receipt.as_dict() | {"execution": execution.as_dict()}
    temporary = output_dir / ".stage_receipt.json.tmp"
    temporary.write_text(json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_stage_history(output_dir: Path, stage: str, records: Sequence[Mapping[str, Any]]) -> str:
    """Persist bounded numeric per-update history without replacing a parent run."""

    path = output_dir / "stage_history.jsonl"
    temporary = output_dir / ".stage_history.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {
                "schema_version": "pfgr-lite-stage-history-v1",
                "stage": stage,
                "scope": "segment",
                "record": _jsonable(record),
            }
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path.name


def _stage_state_progress(
    stage: str,
    options: StageOptions,
    inputs: StageInputs,
    updates: int,
    groups: tuple[str, ...],
    *,
    completion_override: str | None = None,
    substage_override: str | None = None,
) -> StageState:
    """Derive resumable epoch/completion from bounded optimizer progress."""

    if stage in {"S0", "S1"} and inputs.samples:
        batches_per_epoch = max(1, math.ceil(len(inputs.samples) / options.batch_size))
        expected_updates = options.epochs * batches_per_epoch
        complete = int(updates) >= expected_updates
        epoch = min(options.epochs, int(updates) // batches_per_epoch)
        # A partial final batch is still in the current epoch; retaining the
        # floor makes resume state explicit rather than falsely reporting the
        # configured epoch count after ``max_updates`` cuts it short.
    else:
        complete = True
        epoch = options.epochs
    if completion_override is not None:
        if completion_override not in {"pending", "complete"}:
            raise ValueError("completion_override must be pending or complete")
        complete = completion_override == "complete"
    substage = substage_override or ("value_fit" if stage in {"S3", "S4"} and not complete else "complete")
    return StageState(
        stage=stage,
        substage=substage,
        epoch=int(epoch),
        update=int(updates),
        microstep=0,
        optimizer_groups=tuple(groups),
        completion="complete" if complete else "pending",
    )


def _resolve_execution(stage: str, config: object, inputs: StageInputs) -> StageExecutionConfig:
    if inputs.execution is not None:
        execution = inputs.execution
        if execution.stage_options.stage != stage:
            raise ValueError("StageOptions stage does not match run_stage stage")
        if config is not execution.config and config != execution.config:
            # The production factory resolves the concrete observation recipe
            # into ``PFGRLiteConfig.observation_normalization``.  CLI callers
            # may still hold W1's historical schema-label default; accept that
            # one narrow normalization-only substitution after comparing every
            # other authoritative PFGR field.  Any explicit non-default
            # policy/hash mismatch remains fail-closed.
            if not isinstance(config, PFGRLiteConfig) or config.observation_normalization != "pfgr-observation-normalization-v1" or replace(config, observation_normalization=execution.config.observation_normalization) != execution.config:
                raise ValueError("run_stage config does not match StageExecutionConfig")
        if execution.stage_options.semantic_objective and stage != "S0":
            raise ValueError("semantic_objective is supported only by S0 static objective")
        if stage != "S2" and execution.stage_options.query_mode != "exact_dense":
            raise ValueError("query_mode='iid_fixed_q' is supported only for S2 teacher measurement; S0/S1 use exact dense reconstruction and S3-S6 do not execute voxel query losses")
        if stage != "S2" and execution.stage_options.teacher_q_draws is not None:
            raise ValueError("teacher_q_draws is supported only for S2 teacher measurement")
        return execution
    if isinstance(config, Mapping):
        config = PFGRLiteConfig.from_dict(config)
    if not isinstance(config, PFGRLiteConfig):
        raise TypeError("config must be PFGRLiteConfig or strict mapping")
    options = inputs.stage_options or StageOptions(stage=stage, engineering_only=config.engineering_only)
    if options.stage != stage:
        raise ValueError("StageOptions stage does not match run_stage stage")
    if options.semantic_objective and stage != "S0":
        raise ValueError("semantic_objective is supported only by S0 static objective")
    if stage != "S2" and options.query_mode != "exact_dense":
        raise ValueError("query_mode='iid_fixed_q' is supported only for S2 teacher measurement; S0/S1 use exact dense reconstruction and S3-S6 do not execute voxel query losses")
    if stage != "S2" and options.teacher_q_draws is not None:
        raise ValueError("teacher_q_draws is supported only for S2 teacher measurement")
    return StageExecutionConfig(
        config=config,
        frontend_sidecar=inputs.frontend_sidecar,
        normalization=inputs.normalization,
        stage_options=options,
    )


def _run_s0(inputs: StageInputs, execution: StageExecutionConfig) -> tuple[dict[str, Any], tuple[str, ...], int, int, int, dict[str, Any]]:
    options = execution.stage_options
    static_step = inputs.static_step
    if static_step is None:
        if inputs.model is None:
            raise ValueError("S0 requires a PFGR model or explicit static_step(sample, ...) callable")
        static_step = lambda sample, **_kwargs: _default_static_step(sample, model=inputs.model, options=execution.stage_options, inputs=inputs)
    static_names_and_modules: tuple[tuple[str, nn.Module | None], ...] = (
        ("static_head", _module(inputs.model, ("static_head",))),
        ("base_plane_projector", _module(inputs.model, ("frontend.base_plane_projector", "base_plane_projector"))),
        ("decoder", _module(inputs.model, ("decoder", "implicit_decoder"))),
    )
    semantic = _module(inputs.model, ("frontend.semantic_prior.semantic_head", "semantic_head")) if execution.stage_options.semantic_objective else None
    if execution.stage_options.semantic_objective:
        if semantic is None:
            raise ValueError("semantic_objective=True requires the existing semantic head")
        static_names_and_modules = static_names_and_modules + (("semantic_head", semantic),)
    static_modules = [module for _, module in static_names_and_modules if module is not None]
    requires = _freeze_model(inputs.model, static_modules)
    optimizer = _optimizer_for(inputs, static_modules, execution.stage_options)
    start_epoch, start_batch, updates, measured_steps, _ = _restore_runtime(inputs, execution, optimizer, static_modules, stage="S0")
    total_loss = torch.tensor(0.0)
    processed_subjects: set[str] = set()
    observed_producer: object | None = None
    gradient_evidence: dict[str, dict[str, float | int]] = {
        name: {"l2_norm_max": 0.0, "nonzero_steps": 0, "measured_steps": 0} for name, module in static_names_and_modules if module is not None
    }
    history_records: list[dict[str, Any]] = []
    trainable_before: dict[str, tuple[Tensor, ...]] = {
        name: tuple(parameter.detach().clone() for parameter in module.parameters())
        for name, module in static_names_and_modules
        if module is not None
    }
    before_hashes: dict[str, str] = {}
    trainable_module_ids = {id(module) for module in static_modules}
    frozen_modules = {
        "medicalnet_backbone": _module(inputs.model, ("frontend.semantic_prior.backbone",)),
        "semantic_head": _module(inputs.model, ("frontend.semantic_prior.semantic_head",)),
        "point_refiner": _module(inputs.model, ("frontend.point_refiner",)),
        "spectral_projector": _module(inputs.model, ("frontend.spectral_anchor_builder",)),
    }
    frozen_modules = {name: module for name, module in frozen_modules.items() if module is not None and id(module) not in trainable_module_ids}
    for name, module in frozen_modules.items():
        if module is not None:
            before_hashes[name] = module_state_digest(module)
    try:
        if not inputs.samples:
            raise ValueError("S0 requires at least one target-free sample")
        limit = execution.stage_options.max_updates
        cursor_epoch = start_epoch
        cursor_batch = start_batch
        batches_per_epoch = max(1, math.ceil(len(inputs.samples) / options.batch_size))
        effective_epochs = execution.stage_options.epochs
        if limit is not None and updates < limit:
            effective_epochs = max(effective_epochs, start_epoch + math.ceil((limit - updates) / batches_per_epoch))
        for epoch in range(start_epoch, effective_epochs):
            for batch_index, start in enumerate(range(0, len(inputs.samples), execution.stage_options.batch_size)):
                if epoch == start_epoch and batch_index < start_batch:
                    continue
                if limit is not None and updates >= limit:
                    break
                batch = inputs.samples[start : start + execution.stage_options.batch_size]
                losses: list[Tensor] = []
                for sample in batch:
                    result = _invoke(static_step, sample, sample=sample, options=options, config=execution.config)
                    prediction = _prediction_from(result)
                    context = result.get("context") if isinstance(result, Mapping) else getattr(result, "context", None)
                    if context is not None:
                        observed_producer = getattr(context, "producer", None)
                    target_context = _target_context_for(
                        sample,
                        result,
                        inputs,
                        engineering_only=options.engineering_only,
                        include_segmentation=options.semantic_objective,
                    )
                    processed_subjects.add(sample.subject_id)
                    objective_input: Any = result if options.semantic_objective and isinstance(result, Mapping) else prediction
                    losses.append(static_objective(objective_input, target_context, config=options))
                loss = torch.stack(losses).mean()
                total_loss = loss.detach()
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    batch_gradient_norms: dict[str, float] = {}
                    for name, module in static_names_and_modules:
                        if module is None:
                            continue
                        entry = gradient_evidence[name]
                        entry["measured_steps"] = int(entry["measured_steps"]) + 1
                        squares = [parameter.grad.detach().square().sum() for parameter in module.parameters() if parameter.grad is not None]
                        norm = float(torch.sqrt(torch.stack(squares).sum()).item()) if squares else 0.0
                        batch_gradient_norms[name] = norm
                        entry["l2_norm_max"] = max(float(entry["l2_norm_max"]), norm)
                        if norm > 0.0:
                            entry["nonzero_steps"] = int(entry["nonzero_steps"]) + 1
                    optimizer.step()
                    updates += 1
                    measured_steps += 1
                    history_records.append(
                        {
                            "epoch": int(epoch),
                            "update": int(updates),
                            "subject_ids": tuple(sample.subject_id for sample in batch),
                            "objective": float(loss.detach().item()),
                            "module_gradient_l2": dict(sorted(batch_gradient_norms.items())),
                        }
                    )
                    cursor_epoch = epoch + (1 if batch_index + 1 >= batches_per_epoch else 0)
                    cursor_batch = 0 if batch_index + 1 >= batches_per_epoch else batch_index + 1
            if limit is not None and updates >= limit:
                break
        groups = tuple(name for name, module in static_names_and_modules if module is not None)
        update_evidence: dict[str, dict[str, int]] = {}
        for name, module in static_names_and_modules:
            if module is None:
                continue
            update_evidence[name] = {"changed_parameter_count": 0, "optimizer_steps": updates}
        for name, module in static_names_and_modules:
            if module is None:
                continue
            snapshots = trainable_before[name]
            update_evidence[name]["changed_parameter_count"] = sum(
                int(not torch.equal(before, parameter.detach()))
                for before, parameter in zip(snapshots, module.parameters())
            )
        after_hashes = {name: module_state_digest(module) for name, module in frozen_modules.items() if module is not None}
        semantic_head = _module(inputs.model, ("frontend.semantic_prior.semantic_head", "semantic_head")) if options.semantic_objective else None
        refreshed_hash = _producer_hash_after_static_updates(
            inputs,
            observed_producer=observed_producer,
            static_head=_module(inputs.model, ("static_head",)),
            decoder=_module(inputs.model, ("decoder", "implicit_decoder")),
            semantic_head=semantic_head,
        )
        runtime = _runtime_state(
            inputs,
            execution,
            optimizer,
            static_modules,
            stage="S0",
            epoch=cursor_epoch,
            batch_index=cursor_batch,
            update=updates,
            measured_steps=measured_steps,
            producer_hash=refreshed_hash,
            optimizer_groups=groups,
        )
        return ({"loss": float(total_loss.item()), "subjects": len(processed_subjects), "authorized_modules": list(groups), "loss_components": {"charbonnier": 1.0, "ssim": options.ssim_weight, "gradient": options.gradient_weight}, "query_mode": "exact_dense", "query_scope": "S0_full_volume_objective", "gradient_evidence": gradient_evidence, "update_evidence": update_evidence, "history": history_records, "history_scope": "segment", "history_parent": "prior_runtime" if inputs.resume is not None else None, "frozen_hashes": {name: {"before": before_hashes[name], "after": after_hashes.get(name), "unchanged": before_hashes[name] == after_hashes.get(name)} for name in before_hashes}}, groups, measured_steps, updates, len(processed_subjects), runtime)
    finally:
        _restore_requires_grad(inputs.model, requires)


def _run_s1(inputs: StageInputs, execution: StageExecutionConfig) -> tuple[dict[str, Any], tuple[str, ...], int, int, int, dict[str, Any] | None]:
    options = execution.stage_options
    route_builder = inputs.route_builder
    if route_builder is None:
        if inputs.model is None:
            raise ValueError("S1 requires a PFGR model or explicit target-free route_builder")
        route_builder = lambda sample, **kwargs: _default_random_route(
            sample,
            model=inputs.model,
            options=options,
            inputs=inputs,
            k=kwargs.get("k", 1),
            seed=kwargs.get("seed", options.seed),
            counters=kwargs.get("counters"),
        )
    updater = _module(inputs.model, ("updater", "update_net"))
    spectral = _module(inputs.model, ("frontend.spectral_anchor_builder", "spectral_projector", "spectral_anchor_builder"))
    if options.arm == "u_plus_spectral" and spectral is None and not options.engineering_only:
        raise ValueError("u_plus_spectral S1 requires the existing shared spectral projector")
    trainable = [module for module in (updater, spectral if options.arm == "u_plus_spectral" else None) if module is not None]
    requires = _freeze_model(inputs.model, trainable)
    frozen_specs: dict[str, nn.Module | None] = {
        "decoder": _module(inputs.model, ("decoder", "implicit_decoder")),
        "static_head": _module(inputs.model, ("static_head",)),
        "base_plane_projector": _module(inputs.model, ("frontend.base_plane_projector", "base_plane_projector")),
        "semantic_head": _module(inputs.model, ("frontend.semantic_prior.semantic_head", "semantic_head")),
        "point_refiner": _module(inputs.model, ("frontend.point_refiner",)),
        "medicalnet_backbone": _module(inputs.model, ("frontend.semantic_prior.backbone",)),
    }
    # The explicit spectral arm owns the shared projector; in U-only control
    # it is frozen and therefore included in this unchanged-hash evidence.
    if options.arm != "u_plus_spectral":
        frozen_specs["spectral_projector"] = spectral
    frozen_modules = {name: module for name, module in frozen_specs.items() if module is not None}
    frozen_modes = {id(module): bool(module.training) for module in frozen_modules.values()}
    frozen_before_hashes = {name: module_state_digest(module) for name, module in frozen_modules.items()}
    for module in frozen_modules.values():
        module.eval()
    optimizer = _optimizer_for(inputs, trainable, options)
    operation_counters = _operation_counters(inputs)
    operation_counters_before = operation_counters.as_dict()
    rng = random.Random(options.seed)
    start_epoch, start_batch, optimizer_steps, measured_steps, _ = _restore_runtime(inputs, execution, optimizer, trainable, stage="S1")
    resume_cursor = _resume_value(inputs.resume, "cursor", {}) or {}
    local_route_state = resume_cursor.get("route_rng_state") if isinstance(resume_cursor, Mapping) else None
    if local_route_state is not None:
        local_route_state = _tupleize_rng_state(local_route_state)
        if not isinstance(local_route_state, tuple):
            raise TypeError("resume route_rng_state must be a tuple")
        rng.setstate(local_route_state)
    updates_limit = options.max_updates
    route_count = 0
    sampled_k_counts = {1: 0, 2: 0, 4: 0}
    grad_norm_max = 0.0
    nonzero_steps = 0
    updater_grad_norm_sum = 0.0
    updater_grad_norm_max = 0.0
    updater_nonzero_steps = 0
    updater_measured_steps = 0
    spectral_grad_norm_sum = 0.0
    history_records: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    paired_unmeasured: list[dict[str, Any]] = []
    observed_producer_hash: str | None = None
    observed_producer: object | None = None
    before_hash = module_state_digest(spectral) if spectral is not None else "none"
    changed_projector = 0
    changed_total = 0
    try:
        if not inputs.samples:
            raise ValueError("S1 requires at least one target-free sample")
        if updates_limit is not None and updates_limit <= 0:
            raise ValueError("S1 requires at least one bounded optimizer update")
        objective = None
        cursor_epoch = start_epoch
        cursor_batch = start_batch
        batches_per_epoch = max(1, math.ceil(len(inputs.samples) / options.batch_size))
        effective_epochs = options.epochs
        if options.max_updates is not None and optimizer_steps < options.max_updates:
            effective_epochs = max(effective_epochs, start_epoch + math.ceil((options.max_updates - optimizer_steps) / batches_per_epoch))
        for epoch in range(start_epoch, effective_epochs):
            for batch_index, start in enumerate(range(0, len(inputs.samples), options.batch_size)):
                if epoch == start_epoch and batch_index < start_batch:
                    continue
                if options.max_updates is not None and optimizer_steps >= options.max_updates:
                    break
                batch = inputs.samples[start : start + options.batch_size]
                losses: list[Tensor] = []
                batch_k_values: list[int] = []
                for sample in batch:
                    k = rng.choice((1, 2, 4))
                    batch_k_values.append(k)
                    sampled_k_counts[k] += 1
                    route_seed = rng.randrange(0, 2**63 - 1)
                    result = _invoke(
                        route_builder,
                        sample,
                        sample=sample,
                        k=k,
                        seed=route_seed,
                        options=options,
                        config=execution.config,
                        counters=operation_counters,
                    )
                    context = result.get("context") if isinstance(result, Mapping) else getattr(result, "context", None)
                    if context is not None:
                        observed_producer = getattr(context, "producer", None)
                        observed_producer_hash = getattr(observed_producer, "compatibility_hash", None)
                    if _route_k(result) != k:
                        raise ValueError("route_builder returned a route with K different from the sampled K")
                    target_context = _target_context_for(sample, result, inputs, engineering_only=execution.stage_options.engineering_only)
                    initial_prediction = result.get("initial_prediction") if isinstance(result, Mapping) else getattr(result, "initial_prediction", None)
                    route_predictions = _route_predictions(result)
                    if initial_prediction is None:
                        if not options.engineering_only:
                            raise ValueError("production S1 route must expose same-forward initial_prediction")
                        paired_unmeasured.append({"subject_id": sample.subject_id, "reason": "initial_prediction_unavailable", "budget": int(k)})
                    else:
                        pair = _paired_dense_metrics(
                            initial_prediction,
                            route_predictions[-1],
                            target_context,
                            sample=sample,
                            context=context,
                            budget=k,
                            charbonnier_epsilon=execution.config.teacher.epsilon,
                        )
                        pair["stage_update"] = int(optimizer_steps + 1)
                        pair["k"] = int(k)
                        paired_rows.append(pair)
                    losses.append(updater_objective(result, target_context, config=options))
                    route_count += 1
                objective = torch.stack(losses).mean()
                if optimizer is None:
                    continue
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                measured_steps += 1
                updater_norm = 0.0
                if updater is not None:
                    updater_squares = [parameter.grad.detach().square().sum() for parameter in updater.parameters() if parameter.grad is not None]
                    updater_norm = float(torch.sqrt(torch.stack(updater_squares).sum()).item()) if updater_squares else 0.0
                    updater_grad_norm_sum += updater_norm
                    updater_grad_norm_max = max(updater_grad_norm_max, updater_norm)
                    updater_measured_steps += 1
                    if updater_norm > 0.0:
                        updater_nonzero_steps += 1
                if spectral is not None:
                    squares = [parameter.grad.detach().square().sum() for parameter in spectral.parameters() if parameter.grad is not None]
                    norm = float(torch.sqrt(torch.stack(squares).sum()).item()) if squares else 0.0
                    grad_norm_max = max(grad_norm_max, norm)
                    spectral_grad_norm_sum += norm
                    if norm > 0.0:
                        nonzero_steps += 1
                before_projector = [parameter.detach().clone() for parameter in spectral.parameters()] if spectral is not None else []
                before_parameters = [parameter.detach().clone() for module in trainable for parameter in module.parameters()]
                optimizer.step()
                after_parameters = [parameter.detach() for module in trainable for parameter in module.parameters()]
                changed_total += sum(int(not torch.equal(before, after)) for before, after in zip(before_parameters, after_parameters))
                if spectral is not None:
                    changed_projector += sum(int(not torch.equal(before, after)) for before, after in zip(before_projector, spectral.parameters()))
                optimizer_steps += 1
                history_records.append(
                    {
                        "epoch": int(epoch),
                        "update": int(optimizer_steps),
                        "subject_ids": tuple(sample.subject_id for sample in batch),
                        "k": tuple(batch_k_values),
                        "objective": float(objective.detach().item()),
                        "module_gradient_l2": {
                            "updater": float(updater_norm),
                            "spectral_projector": float(norm if spectral is not None else 0.0),
                        },
                    }
                )
                cursor_epoch = epoch + (1 if batch_index + 1 >= batches_per_epoch else 0)
                cursor_batch = 0 if batch_index + 1 >= batches_per_epoch else batch_index + 1
            if options.max_updates is not None and optimizer_steps >= options.max_updates:
                break
        if objective is None:
            raise ValueError("S1 produced no optimizer batch")
        after_hash = module_state_digest(spectral) if spectral is not None else "none"
        frozen_after_hashes = {name: module_state_digest(module) for name, module in frozen_modules.items()}
        frozen_hashes = {
            name: {
                "before": frozen_before_hashes[name],
                "after": frozen_after_hashes[name],
                "unchanged": frozen_before_hashes[name] == frozen_after_hashes[name],
            }
            for name in frozen_before_hashes
        }
        changed_frozen = [name for name, value in frozen_hashes.items() if not value["unchanged"]]
        if changed_frozen:
            raise RuntimeError(f"S1 frozen module state changed: {sorted(changed_frozen)}")
        completed = optimizer is not None and optimizer_steps > 0
        refreshed_hash = _producer_hash_after_updates(
            inputs,
            observed_producer=observed_producer,
            spectral=spectral,
            updater=updater,
        )
        provenance = _stage_provenance(inputs, options, gradient_norm=grad_norm_max, nonzero_steps=nonzero_steps, measured_steps=measured_steps, optimizer_steps=optimizer_steps, changed=changed_projector, before=before_hash, after=after_hash, completed=completed, producer_hash=refreshed_hash)
        if options.arm == "u_plus_spectral" and (not completed or nonzero_steps <= 0 or changed_projector <= 0) and not options.engineering_only:
            raise RuntimeError("u_plus_spectral S1 requires positive measured spectral gradients and projector updates")
        paired_aggregate = None
        if paired_rows:
            from .metrics import aggregate_subject_metrics

            paired_aggregate = aggregate_subject_metrics(paired_rows)
        metrics = {
            "loss": float(objective.detach().item()),
            "route_count": route_count,
            "sampled_k_counts": {str(k): int(v) for k, v in sampled_k_counts.items()},
            "spectral_gradient_l2_max": grad_norm_max,
            "spectral_gradient_l2_sum": spectral_grad_norm_sum,
            "spectral_gradient_l2_measured_steps": measured_steps if spectral is not None else 0,
            "spectral_nonzero_steps": nonzero_steps,
            "updater_gradient_l2_sum": updater_grad_norm_sum,
            "updater_gradient_l2_max": updater_grad_norm_max,
            "updater_gradient_l2_measured_steps": updater_measured_steps,
            "updater_gradient_nonzero_steps": updater_nonzero_steps,
            "gradient_evidence": {
                "updater": {
                    "l2_norm_sum": updater_grad_norm_sum,
                    "l2_norm_max": updater_grad_norm_max,
                    "measured_steps": updater_measured_steps,
                    "nonzero_steps": updater_nonzero_steps,
                },
                "spectral_projector": {
                    "l2_norm_sum": spectral_grad_norm_sum,
                    "l2_norm_max": grad_norm_max,
                    "measured_steps": measured_steps if spectral is not None else 0,
                    "nonzero_steps": nonzero_steps,
                },
            },
            "measured_steps": measured_steps,
            "optimizer_steps": optimizer_steps,
            "changed_parameter_count": changed_projector,
            "changed_parameter_count_total": changed_total,
            "epochs": options.epochs,
            "batch_size": options.batch_size,
            "query_mode": "exact_dense",
            "query_scope": "S1_full_volume_objective_per_route_state",
            "operation_counters": _counter_delta(operation_counters_before, operation_counters.as_dict()),
            "operation_counter_scope": "S1_target_free_route_and_write_v1",
            "history": history_records,
            "history_scope": "segment",
            "history_parent": "prior_runtime" if inputs.resume is not None else None,
            "paired_dense_metrics": paired_rows,
            "paired_dense_metrics_aggregate": paired_aggregate,
            "paired_dense_metrics_measured_count": len(paired_rows),
            "paired_dense_metrics_unmeasured": paired_unmeasured,
            "paired_dense_metrics_scope": "same_forward_detached_z0_vs_final_target_after_inference_v1",
            "frozen_hashes": frozen_hashes,
        }
        groups = tuple("updater" if module is updater else "spectral_projector" for module in trainable)
        runtime = _runtime_state(
            inputs,
            execution,
            optimizer,
            trainable,
            stage="S1",
            epoch=cursor_epoch,
            batch_index=cursor_batch,
            update=optimizer_steps,
            measured_steps=measured_steps,
            local_rng=rng,
            producer_hash=refreshed_hash,
            optimizer_groups=groups,
        )
        return metrics, groups, measured_steps, optimizer_steps, len(inputs.samples), provenance, runtime
    finally:
        _restore_requires_grad(inputs.model, requires)
        for module in frozen_modules.values():
            module.train(frozen_modes[id(module)])


def _candidate_value(item: object) -> tuple[str, str, float, object]:
    if isinstance(item, Mapping):
        action_id = str(item.get("action_id", item.get("id", "")))
        stratum = str(item.get("stratum", "uniform"))
        score = float(item.get("score", item.get("predicted_score", 0.0)))
    else:
        action_id = str(getattr(item, "action_id", getattr(item, "id", "")))
        stratum = str(getattr(item, "stratum", "uniform"))
        score = float(getattr(item, "score", getattr(item, "predicted_score", 0.0)))
    if not action_id:
        raise ValueError("candidate rows require an action_id")
    if not math.isfinite(score):
        raise ValueError("candidate score must be finite")
    return action_id, stratum, score, item


def _candidate_feature(item: object) -> tuple[float, ...] | None:
    """Return a bounded observation-only feature for diverse S2 sampling."""

    source: object = item.get("action", item) if isinstance(item, Mapping) else item
    point = item.get("point_ras_mm") if isinstance(item, Mapping) else None
    if point is None:
        point = getattr(source, "point_ras_mm", None)
    semantic = item.get("semantic", item.get("semantic3")) if isinstance(item, Mapping) else None
    if semantic is None:
        v126 = item.get("v126") if isinstance(item, Mapping) else None
        if v126 is None:
            v126 = getattr(source, "v126", None)
        if isinstance(v126, Tensor) and v126.numel() >= 99:
            semantic = v126.reshape(-1)[96:99]
    values: list[float] = []
    for value in (point, semantic):
        if isinstance(value, Tensor):
            flat = value.detach().cpu().reshape(-1)
            if flat.numel() > 16:
                flat = flat[:16]
            if not bool(torch.isfinite(flat).all()):
                return None
            values.extend(float(item_value) for item_value in flat.tolist())
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) > 16:
                value = value[:16]
            try:
                parsed = [float(item_value) for item_value in value]
            except (TypeError, ValueError):
                return None
            if not all(math.isfinite(item_value) for item_value in parsed):
                return None
            values.extend(parsed)
    return tuple(values) if values else None


def _select_diverse(
    pool: Sequence[object],
    *,
    count: int,
    selected: Sequence[object],
) -> list[object]:
    """Greedy deterministic farthest-point selection with explicit ties."""

    if count <= 0:
        return []
    parsed = [_candidate_value(item) for item in pool]
    unique: dict[str, object] = {}
    for action_id, _stratum, _score, item in parsed:
        unique.setdefault(action_id, item)
    candidates = list(unique.values())
    feature_rows = {id(item): _candidate_feature(item) for item in candidates}
    all_features = [feature for feature in feature_rows.values() if feature is not None]
    if all_features:
        width = max(len(feature) for feature in all_features)
        mins = [min(feature[index] if index < len(feature) else 0.0 for feature in all_features) for index in range(width)]
        spans = [max(feature[index] if index < len(feature) else 0.0 for feature in all_features) - mins[index] for index in range(width)]

        def normalized(item: object) -> tuple[float, ...] | None:
            feature = feature_rows.get(id(item))
            if feature is None:
                feature = _candidate_feature(item)
            if feature is None:
                return None
            return tuple(((feature[index] if index < len(feature) else 0.0) - mins[index]) / spans[index] if spans[index] > 0.0 else 0.0 for index in range(width))

    else:
        normalized = lambda _item: None  # type: ignore[assignment]
    chosen: list[object] = []
    used = {_candidate_value(item)[0] for item in selected}
    available = [item for item in candidates if _candidate_value(item)[0] not in used]
    while available and len(chosen) < count:
        reference = [normalized(item) for item in (*selected, *chosen)]
        reference = [item for item in reference if item is not None]
        scored: list[tuple[float, float, str, object]] = []
        for item in available:
            feature = normalized(item)
            distance = 0.0
            if feature is not None and reference:
                distance = min(sum((feature[index] - other[index]) ** 2 for index in range(min(len(feature), len(other)))) for other in reference)
            action_id, _stratum, score, _ = _candidate_value(item)
            scored.append((distance, score, action_id, item))
        # Maximise separation, then use score and action ID as explicit ties.
        _, _, selected_id, selected_item = max(scored, key=lambda row: (row[0], row[1], -len(row[2]), row[2]))
        chosen.append(selected_item)
        available = [item for item in available if _candidate_value(item)[0] != selected_id]
    return chosen


def _validate_observation_only_candidate(item: object) -> None:
    """Reject privileged/target-aware rows before mixed S2 selection.

    S2 has no diagnostic-bank mode of its own.  A caller that wants to retain
    oracle or hard-mining rows must route them through an explicitly separate
    privileged service; silently letting them enter the MAIN mixed pool would
    leak target information into ValueNet training.
    """

    if isinstance(item, Mapping):
        lowered = {str(key).lower(): value for key, value in item.items()}
        flagged = any(
            bool(lowered.get(key))
            for key in ("target_aware", "uses_target", "oracle", "teacher_label", "privileged", "hard_mining")
        )
        stratum = str(lowered.get("stratum", "")).lower()
    else:
        flagged = any(bool(getattr(item, key, False)) for key in ("target_aware", "uses_target", "oracle", "teacher_label", "privileged", "hard_mining"))
        stratum = str(getattr(item, "stratum", "")).lower()
    flagged = flagged or any(token in stratum for token in ("target", "oracle", "teacher", "privileged", "hard_mining"))
    if flagged:
        raise ValueError("S2 candidate pool contains target-aware/privileged rows; use a separate diagnostic partition")


def _select_candidates(candidates: Sequence[object], *, count: int, seed: int) -> list[object]:
    parsed = [_candidate_value(item) for item in candidates]
    by_stratum: dict[str, list[tuple[str, str, float, object]]] = {}
    for row in parsed:
        by_stratum.setdefault(row[1], []).append(row)
    # The three MAIN strata are fixed when present; small engineering fixtures
    # may provide fewer rows and receive deterministic uniform fill.
    quotas = {"uniform": min(16, count), "frozen_v_high_score": min(8, max(0, count - min(16, count))), "predicted_semantic_spatial": min(8, max(0, count - min(24, count)))}
    rng = random.Random(seed)
    selected: list[object] = []
    seen: set[str] = set()
    # The uniform stratum is sampled from *all* legal rows, independently of
    # any optional model-derived strata.  A row may also carry a high-score or
    # semantic/spatial tag; action-id deduplication below keeps one selection.
    unique_uniform: dict[str, object] = {}
    for action_id, _stratum, _score, item in parsed:
        prior = unique_uniform.get(action_id)
        if prior is None or _stratum == "uniform" and _candidate_value(prior)[1] != "uniform":
            unique_uniform[action_id] = item
    all_uniform = sorted(unique_uniform.values(), key=lambda item: _candidate_value(item)[0])
    uniform_quota = quotas["uniform"]
    if uniform_quota:
        uniform_rows = rng.sample(all_uniform, min(uniform_quota, len(all_uniform)))
    else:
        uniform_rows = []
    for item in uniform_rows:
        action_id = _candidate_value(item)[0]
        if action_id not in seen:
            selected.append(item)
            seen.add(action_id)
    for stratum in ("frozen_v_high_score", "predicted_semantic_spatial"):
        pool = by_stratum.get(stratum, [])
        if stratum == "frozen_v_high_score":
            pool = sorted(pool, key=lambda row: (-row[2], row[0]))
        else:
            pool = [row for row in pool]
        pool = [row for row in pool if row[0] not in seen]
        diverse = _select_diverse([row[3] for row in pool], count=quotas[stratum], selected=selected) if stratum == "predicted_semantic_spatial" else [row[3] for row in pool[: quotas[stratum]]]
        for item in diverse:
            row = _candidate_value(item)
            if row[0] not in seen:
                selected.append(item)
                seen.add(row[0])
    unique_remaining: dict[str, object] = {}
    for _action_id, _stratum, _score, item in parsed:
        unique_remaining.setdefault(_action_id, item)
    remaining = [(_candidate_value(item)[0], item) for item in unique_remaining.values() if _candidate_value(item)[0] not in seen]
    rng.shuffle(remaining)
    for _action_id, item in remaining:
        if len(selected) >= count:
            break
        selected.append(item)
        seen.add(_candidate_value(item)[0])
    return selected[:count]


SELECTED_REPLAY_SCHEMA = "pfgr-lite-selected-replay-v1"


def _write_selected_replay(
    output_dir: Path,
    sample: TargetFreeSample,
    context: object | None,
    state: object,
    *,
    state_index: int,
    prefix_state_digests: Sequence[str],
) -> str:
    """Write one bounded metadata-only replay record for a selected state."""

    context_id = str(getattr(context, "context_id", "")) if context is not None else ""
    producer = getattr(context, "producer", None) if context is not None else None
    producer_hash = getattr(producer, "compatibility_hash", None)
    if producer_hash is None:
        producer_hash = getattr(getattr(producer, "compatibility", None), "digest", "")
    state_version = getattr(state, "state_version", getattr(state, "version", 0))
    state_digest = getattr(state, "state_digest", "")
    if not isinstance(state_digest, str) or not state_digest:
        state_digest = canonical_digest(
            {"subject_id": sample.subject_id, "state_index": int(state_index), "state_version": state_version},
            prefix="pfgr-lite-engineering-state-v1|",
        )
    feature_geometry = getattr(context, "feature_geometry", None) if context is not None else None
    feature_geometry_hash = canonical_digest(_jsonable(feature_geometry), prefix="pfgr-lite-feature-geometry-replay-v1|")
    route_prefix_hash = canonical_digest(tuple(prefix_state_digests), prefix="pfgr-lite-route-prefix-replay-v1|")
    payload = {
        "schema_version": SELECTED_REPLAY_SCHEMA,
        "snapshot_kind": "metadata_only",
        "subject_id": sample.subject_id,
        "state_index": int(state_index),
        "state_version": int(state_version) if isinstance(state_version, int) and not isinstance(state_version, bool) else str(state_version),
        "state_digest": state_digest,
        "context_id": context_id,
        "producer_compatibility_hash": str(producer_hash),
        "geometry_hash": sample.geometry_hash,
        "feature_geometry_hash": feature_geometry_hash,
        "normalization_hash": sample.normalization_hash,
        "route_prefix_hash": route_prefix_hash,
        "tensor_payload": "omitted",
        "raw_target_payload": "omitted",
    }
    identity = canonical_digest(payload, prefix="pfgr-lite-selected-replay-id-v1|")
    safe_subject = "".join(char if char.isalnum() or char in "-_" else "_" for char in sample.subject_id) or "subject"
    replay_dir = output_dir / "selected_replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    path = replay_dir / f"{safe_subject}-s{int(state_index)}-{identity[:16]}.json"
    # The stage output directory is reserved by run_stage; exclusive creation
    # protects this record if a caller races within the same destination.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    return str(path.relative_to(output_dir))


def _invoke_frozen(callable_obj: Callable[..., Any], *positional: Any, **keyword: Any) -> Any:
    """Invoke an S2 service without constructing producer autograd graphs."""

    with torch.no_grad():
        return _invoke(callable_obj, *positional, **keyword)


def _s2_teacher_config(config: PFGRLiteConfig, options: StageOptions) -> object:
    """Resolve S2's explicit exact-vs-fixed-Q measurement envelope."""

    requested_mode = "exact_footprint" if options.query_mode == "exact_dense" else "iid_fixed_q"
    teacher_config = config.teacher
    # StageOptions is an explicit sidecar: it may override only the sampling
    # budget, never the fixed rho/epsilon/mask teacher semantics in PFGRLiteConfig.
    if requested_mode == "exact_footprint":
        effective_q = 0
    else:
        effective_q = teacher_config.q_draws if options.teacher_q_draws is None else options.teacher_q_draws
        if not isinstance(effective_q, int) or isinstance(effective_q, bool) or effective_q < 2:
            raise ValueError("iid_fixed_q S2 requires effective teacher_q_draws >= 2")
    if teacher_config.mode != requested_mode or teacher_config.q_draws != effective_q:
        teacher_config = replace(teacher_config, mode=requested_mode, q_draws=effective_q)
    return teacher_config


def _run_s2(inputs: StageInputs, execution: StageExecutionConfig, output_dir: Path) -> tuple[dict[str, Any], tuple[str, ...], int, int, int]:
    """Run S2 with producer modules frozen and in evaluation mode."""

    model = inputs.model if isinstance(inputs.model, nn.Module) else None
    requires = _freeze_model(model, ()) if model is not None else {}
    modes = {id(module): bool(module.training) for module in model.modules()} if model is not None else {}
    before_hash = module_state_digest(model) if model is not None else None
    if model is not None:
        model.eval()
    try:
        result = _run_s2_impl(inputs, execution, output_dir)
        if model is None:
            return result
        metrics, groups, gradients, updates, subjects = result
        after_hash = module_state_digest(model)
        metrics = dict(metrics)
        metrics.update(
            {
                "frozen_model_hash_before": before_hash,
                "frozen_model_hash_after": after_hash,
                "frozen_model_unchanged": before_hash == after_hash,
                "producer_eval_mode": True,
            }
        )
        if before_hash != after_hash:
            raise RuntimeError("S2 producer parameters changed during frozen collection")
        return metrics, groups, gradients, updates, subjects
    finally:
        if model is not None:
            _restore_requires_grad(model, requires)
            for module in model.modules():
                module.train(modes[id(module)])


def _run_s2_impl(inputs: StageInputs, execution: StageExecutionConfig, output_dir: Path) -> tuple[dict[str, Any], tuple[str, ...], int, int, int]:
    options = execution.stage_options
    stage_provenance = inputs.metadata.get("stage_provenance") if isinstance(inputs.metadata, Mapping) else None
    if not options.engineering_only and not isinstance(stage_provenance, Mapping):
        raise ValueError("production S2 requires the verified completed producer-stage provenance receipt")
    if isinstance(stage_provenance, Mapping) and not options.engineering_only:
        if stage_provenance.get("completed") is not True:
            raise ValueError("production S2 stage provenance must record completed=true")
    route_builder = inputs.behavior_builder or inputs.route_builder
    if route_builder is None:
        if inputs.model is None:
            raise ValueError("S2 requires a PFGR model or target-free behavior_builder/route_builder")
        route_builder = lambda sample, **kwargs: _default_random_route(
            sample,
            model=inputs.model,
            options=options,
            inputs=inputs,
            k=4,
            seed=kwargs.get("seed", options.seed),
            counters=kwargs.get("counters"),
        )
    proposal_builder = inputs.proposal_builder
    if proposal_builder is None:
        def proposal_builder(trace: object, state: object, **_kwargs: object) -> list[object]:
            values = getattr(trace, "proposals", None)
            if isinstance(trace, Mapping):
                values = trace.get("proposals")
            result: list[object] = []
            state_version = getattr(state, "state_version", getattr(state, "version", 0))
            frozen_scorer = inputs.metadata.get("frozen_value_scorer") if isinstance(inputs.metadata, Mapping) else None
            for proposal in tuple(values or ()):
                if getattr(proposal, "state_version", state_version) != state_version:
                    continue
                for batch_index in range(proposal.point_ids.shape[0]):
                    for point_index in range(proposal.point_ids.shape[1]):
                        legal = getattr(proposal, "legal", None)
                        if isinstance(legal, Tensor) and legal.ndim >= 2 and not bool(legal[batch_index, point_index]):
                            continue
                        row = proposal.row(batch_index, point_index)
                        predicted_score = float(getattr(row, "predicted_value", getattr(row, "raw_value", 0.0)))
                        if not math.isfinite(predicted_score):
                            predicted_score = 0.0
                        frozen_score = predicted_score
                        if callable(frozen_scorer):
                            frozen_score = float(_invoke_frozen(frozen_scorer, row, action=row, state=state, trace=trace))
                            if not math.isfinite(frozen_score):
                                raise ValueError("frozen value scorer returned a nonfinite score")
                        # Keep every legal action in the uniform pool.  The
                        # optional frozen-V and observation-diverse tags are
                        # additional pools consumed after that random quota.
                        result.append({"action_id": row.action_id, "stratum": "uniform", "score": predicted_score, "action": row})
                        if callable(frozen_scorer):
                            result.append({"action_id": row.action_id, "stratum": "frozen_v_high_score", "score": frozen_score, "action": row})
                        # Keep a separate observation-only semantic/spatial
                        # pool even when frozen-V scores are available.  The
                        # selector applies bounded farthest-point diversity
                        # to the row's point_ras_mm and v126 semantic slice;
                        # no target/teacher signal is introduced here.
                        result.append({"action_id": row.action_id, "stratum": "predicted_semantic_spatial", "score": predicted_score, "action": row})
            return result
    if inputs.target_provider is None:
        raise ValueError("S2 requires a deferred target_provider")
    operation_counters = _operation_counters(inputs)
    operation_counters_before = operation_counters.as_dict()
    effect_measure = inputs.effect_measure
    if effect_measure is None:
        if inputs.model is None:
            raise ValueError("S2 requires W2 effect_measure or a PFGR model for the canonical measurement adapter")

        def effect_measure(trace: object, selected: object, target_context: object, **kwargs: object) -> object:
            from .footprint import PFGRQueryLattice
            from .teacher import measure_actions

            completed_trace = trace.get("trace", trace) if isinstance(trace, Mapping) else trace
            context = trace.get("context") if isinstance(trace, Mapping) else getattr(trace, "context", None)
            if not isinstance(completed_trace, CompletedBehaviorTrace) or context is None:
                raise TypeError("canonical S2 measurement requires a completed W4 trace and ObservationContext")
            actions: list[object] = []
            for candidate in tuple(selected or ()):
                action = candidate.get("action") if isinstance(candidate, Mapping) else candidate
                if action is None:
                    raise TypeError("S2 candidate is missing its immutable ActionProposal row")
                actions.append(action)
            decoder = inputs.decoder or getattr(inputs.model, "decoder", getattr(inputs.model, "implicit_decoder", None))
            if decoder is None:
                raise ValueError("canonical S2 measurement requires PFGR decoder")
            teacher_config = _s2_teacher_config(execution.config, options)
            lattice = PFGRQueryLattice.build(
                context.geometry,
                context.feature_geometry,
                query_dtype=completed_trace.states[0].planes.xy.dtype,
                build_chunk_size=execution.config.build_chunk_size,
            )
            return measure_actions(
                completed_trace,
                actions,
                target_context,
                decoder,
                teacher_config,
                lattice=lattice,
                chunk_size=execution.config.decode_chunk_size,
                candidate_chunk_size=options.candidate_chunk_size,
                seed=options.seed,
                counters=kwargs.get("counters"),
                observation_context=context,
            )
    bindings: list[Mapping[str, Any]] = []
    selection_receipts: list[Mapping[str, Any]] = []
    # A canonical W3a writer streams detached rows shard-by-shard.  Custom
    # engineering writers retain the historical single-call adapter because
    # they may need the complete tiny fixture in memory.
    streaming_writer: object | None = None
    streamed_manifest: object | None = None
    rows: list[object] = []
    trace_count = 0
    selected_candidate_count = 0
    label_count = 0
    selected_replay_refs: list[str] = []
    for index, sample in enumerate(inputs.samples):
        trace = _invoke_frozen(
            route_builder,
            sample,
            sample=sample,
            k=4,
            seed=options.seed + index,
            forced=True,
            options=options,
            config=execution.config,
            counters=operation_counters,
        )
        trace_count += 1
        state_values: Any = getattr(trace, "states", None)
        if isinstance(trace, Mapping):
            state_values = trace.get("states")
        if state_values is None:
            state_values = (trace,)
        states = tuple(state_values)[: options.max_states_per_subject]
        context = getattr(trace, "context", None)
        if isinstance(trace, Mapping):
            context = trace.get("context", context)
        if context is not None:
            bindings.append(bind_observation_context(sample, context).as_dict())
            if inputs.bank_writer is None and streaming_writer is None:
                producer_bundle = inputs.producer or getattr(context, "producer", None)
                if producer_bundle is None:
                    raise ValueError("canonical S2 bank writer requires producer dependencies from inputs or ObservationContext")
                from .value_bank import ValueBankWriter

                streaming_writer = ValueBankWriter(
                    output_dir / "bank",
                    producer=producer_bundle,
                    split_role_hash=(inputs.role_manifest.digest if inputs.role_manifest is not None else canonical_digest("engineering", prefix="pfgr-lite-engineering-role-v1|")),
                    role_manifest=inputs.role_manifest,
                    config=execution.config.value,
                    engineering_only=options.engineering_only,
                    stage_provenance=stage_provenance,
                )
        selected: list[object] = []
        seen_subject_actions: set[str] = set()
        replay_ref_by_action: dict[str, str] = {}
        for state_index, state in enumerate(states):
            prefix_state_digests: list[str] = []
            for prefix_state in states[: state_index + 1]:
                prefix_digest = getattr(prefix_state, "state_digest", "")
                if not isinstance(prefix_digest, str) or not prefix_digest:
                    prefix_digest = canonical_digest(
                        {"subject_id": sample.subject_id, "state_index": len(prefix_state_digests)},
                        prefix="pfgr-lite-engineering-state-v1|",
                    )
                prefix_state_digests.append(prefix_digest)
            candidate_rows = _invoke_frozen(
                proposal_builder,
                trace,
                state,
                sample=sample,
                state=state,
                state_index=state_index,
                seed=options.seed + state_index,
                options=options,
                config=execution.config,
                counters=operation_counters,
            )
            if isinstance(candidate_rows, Mapping):
                candidate_rows = candidate_rows.get("candidates", ())
            pool = tuple(candidate_rows)
            for candidate in pool:
                _validate_observation_only_candidate(candidate)
            requested = min(options.candidate_count, options.candidates_per_state)
            selected_state = _select_candidates(pool, count=requested, seed=options.seed + index * 17 + state_index)
            pool_counts: dict[str, int] = {}
            for candidate in pool:
                _, stratum, _, _ = _candidate_value(candidate)
                pool_counts[stratum] = pool_counts.get(stratum, 0) + 1
            selected_counts: dict[str, int] = {}
            for candidate in selected_state:
                _, stratum, _, _ = _candidate_value(candidate)
                selected_counts[stratum] = selected_counts.get(stratum, 0) + 1
            # Canonical W3a collection writes a full selected-state snapshot
            # into the writer's private staging root; finalization publishes
            # the resulting replay/ directory atomically with the bank.  Tiny
            # custom writers retain a metadata-only fallback because they do
            # not expose the canonical bank staging contract.
            replay_ref: str
            selected_actions = [candidate.get("action") if isinstance(candidate, Mapping) else candidate for candidate in selected_state]
            canonical_snapshot = (
                context is not None
                and streaming_writer is not None
                and all(action is not None and not isinstance(action, Mapping) for action in selected_actions)
                and isinstance(sample.normalization_hash, str)
                and bool(sample.normalization_hash)
            )
            if canonical_snapshot:
                from .bank_audit import write_state_snapshot

                snapshot_root = getattr(streaming_writer, "_stage", None)
                if not isinstance(snapshot_root, Path):
                    raise RuntimeError("canonical S2 writer does not expose its staging root for replay snapshots")
                replay_ref = write_state_snapshot(
                    snapshot_root,
                    state,
                    context,
                    subject_binding=bind_observation_context(sample, context).as_dict(),
                    route_hash=canonical_digest(
                        {"context_id": getattr(context, "context_id", ""), "state_index": state_index, "prefix_state_digests": prefix_state_digests},
                        prefix="pfgr-lite-selected-route-v1|",
                    ),
                    selected_actions=tuple(selected_actions),
                    split_role_hash=(inputs.role_manifest.digest if inputs.role_manifest is not None else canonical_digest("engineering", prefix="pfgr-lite-engineering-role-v1|")),
                )
            else:
                replay_ref = _write_selected_replay(
                    output_dir,
                    sample,
                    context,
                    state,
                    state_index=state_index,
                    prefix_state_digests=prefix_state_digests,
                )
            selected_replay_refs.append(replay_ref)
            selection_receipts.append(
                {
                    "subject_id": sample.subject_id,
                    "state_index": state_index,
                    "seed": options.seed + index * 17 + state_index,
                    "requested_count": requested,
                    "pool_count": len(pool),
                    "pool_counts_by_stratum": dict(sorted(pool_counts.items())),
                    "selected_count": len(selected_state),
                    "selected_counts_by_stratum": dict(sorted(selected_counts.items())),
                    "deduplicated_subject_count": len({_candidate_value(candidate)[0] for candidate in selected_state}),
                    "procedure": "mixed_without_replacement_fixed_quotas_then_seeded_uniform_fill_v1",
                    "quotas": {"uniform": 16, "frozen_v_high_score": 8, "predicted_semantic_spatial": 8},
                    "frozen_v_artifact": bool(callable(inputs.metadata.get("frozen_value_scorer"))) if isinstance(inputs.metadata, Mapping) else False,
                    "frozen_v_fallback": not bool(callable(inputs.metadata.get("frozen_value_scorer"))) if isinstance(inputs.metadata, Mapping) else True,
                    "marginal_inclusion_probabilities": "not_declared_for_mixed_without_replacement",
                    "selected_replay_ref": replay_ref,
                }
            )
            for candidate in selected_state:
                action_id = _candidate_value(candidate)[0]
                if action_id not in seen_subject_actions:
                    selected.append(candidate)
                    seen_subject_actions.add(action_id)
                    replay_ref_by_action[action_id] = replay_ref
        selected_candidate_count += len(selected)
        if inputs.target_provider is None:  # guarded above; keeps type checkers honest
            raise ValueError("S2 requires a deferred target_provider")
        counter = inputs.metadata.get("counters") if isinstance(inputs.metadata, Mapping) else None
        # The provider is invoked only after this subject's complete
        # target-free behavior/proposal work; no cohort-wide trace retention
        # is required for a bounded bank build.
        target_join = defer_supervision(
            sample,
            inputs.target_provider,
            counters=counter,
            engineering_only=options.engineering_only,
        )
        completed_trace = trace.get("trace", trace) if isinstance(trace, Mapping) else trace
        target_context = target_join(completed_context=context, trace=completed_trace)
        measured = _invoke_frozen(
            effect_measure,
            trace,
            selected,
            target_context,
            sample=sample,
            trace=trace,
            proposals=selected,
            target_context=target_context,
            config=execution.config,
            counters=operation_counters,
        )
        if measured is None:
            continue
        measured_items = tuple(measured) if isinstance(measured, Iterable) and not isinstance(measured, (str, bytes, Mapping)) else (measured,)
        action_by_id = {_candidate_value(candidate)[0]: (candidate.get("action") if isinstance(candidate, Mapping) else candidate) for candidate in selected}
        producer_hash = getattr(getattr(context, "producer", None), "compatibility_hash", "")
        split_hash = inputs.role_manifest.digest if inputs.role_manifest is not None else canonical_digest("engineering", prefix="pfgr-lite-engineering-role-v1|")
        role_name = inputs.metadata.get("subject_role", "producer_fit") if isinstance(inputs.metadata, Mapping) else "producer_fit"
        from .types import GainLabel
        from .value_bank import ValueBankRow, ValueBankWriter

        adapted_items: list[object] = []
        for item in measured_items:
            if isinstance(item, GainLabel):
                action = action_by_id.get(item.action_id)
                if action is None:
                    raise ValueError(f"measured GainLabel {item.action_id!r} has no selected ActionProposal")
                adapted_items.append(
                    replace(ValueBankRow.from_action_label(
                        action,
                        item,
                        split_role=str(role_name),
                        subject_id=sample.subject_id,
                        geometry_id=getattr(context, "context_id", sample.geometry_hash),
                        split_role_hash=split_hash,
                        producer_compatibility_hash=producer_hash,
                        support_provenance="complete_support_v1",
                        inclusion_mechanism=("complete_support_v1" if item.role == "exact_footprint" else "fixed_q_complete_support_v1"),
                        engineering_only=options.engineering_only,
                    ), selected_replay_ref=replay_ref_by_action.get(item.action_id, ""))
                )
            else:
                if isinstance(item, Mapping):
                    payload = dict(item)
                    action_id = str(payload.get("action_id", payload.get("id", "")))
                    payload["selected_replay_ref"] = replay_ref_by_action.get(action_id, "")
                    adapted_items.append(payload)
                elif is_dataclass(item) and hasattr(item, "selected_replay_ref") and hasattr(item, "action_id"):
                    adapted_items.append(replace(item, selected_replay_ref=replay_ref_by_action.get(str(item.action_id), "")))
                else:
                    adapted_items.append(item)
        label_count += len(adapted_items)
        if inputs.bank_writer is not None:
            rows.extend(adapted_items)
            continue
        if streaming_writer is None:
            producer_bundle = inputs.producer or getattr(context, "producer", None)
            if producer_bundle is None:
                raise ValueError("canonical S2 bank writer requires producer dependencies from inputs or ObservationContext")
            streaming_writer = ValueBankWriter(
                output_dir / "bank",
                producer=producer_bundle,
                split_role_hash=split_hash,
                role_manifest=inputs.role_manifest,
                config=execution.config.value,
                engineering_only=options.engineering_only,
                stage_provenance=stage_provenance,
            )
        streaming_writer.append(adapted_items)
    if inputs.bank_writer is not None:
        producer_bundle = inputs.producer
        streamed_manifest = generate_value_bank(rows, producer_bundle, None, execution.config, output_dir / "bank", role_manifest=inputs.role_manifest, writer=inputs.bank_writer, engineering_only=options.engineering_only, stage_provenance=stage_provenance)
    elif streaming_writer is not None:
        try:
            streamed_manifest = streaming_writer.finalize()
        except BaseException:
            abort = getattr(streaming_writer, "abort", None)
            if callable(abort):
                abort()
            raise
    else:
        raise ValueError("S2 produced no measured rows; no bank was published")
    manifest = streamed_manifest
    effective_teacher = _s2_teacher_config(execution.config, options)
    metrics = {
        "trace_count": trace_count,
        "selected_candidate_count": selected_candidate_count,
        "label_count": label_count,
        "subject_context_bindings": bindings,
        "selection_receipts": selection_receipts,
        "sampling_procedure": "mixed_without_replacement_fixed_quotas_then_seeded_uniform_fill_v1",
        "measurement_mode": "exact_footprint" if options.query_mode == "exact_dense" else "iid_fixed_q",
        "measurement_q_draws": int(getattr(effective_teacher, "q_draws", 0)),
        "operation_counters": _counter_delta(operation_counters_before, operation_counters.as_dict()),
        "operation_counter_scope": "S2_target_free_trace_and_W2_measurement_v1",
        "bank_manifest": _jsonable(manifest),
        "selected_replay_refs": tuple(selected_replay_refs),
        "stage_provenance": _jsonable(stage_provenance),
    }
    return metrics, (), 0, 0, len(inputs.samples)


def generate_value_bank(
    completed_traces: Iterable[object],
    producer_bundle: object | None,
    target_provider: Callable[..., object] | None,
    config: PFGRLiteConfig | ValueModelConfig | Mapping[str, Any],
    output_dir: str | Path,
    *,
    role_manifest: TrainingRoleManifest | None = None,
    writer: Callable[..., Any] | None = None,
    effect_measure: Callable[..., Any] | None = None,
    engineering_only: bool = False,
    stage_provenance: Mapping[str, Any] | None = None,
) -> object:
    """Generate an immutable W3a bank from completed target-free traces.

    ``completed_traces`` are materialised before ``target_provider`` is ever
    called.  A supplied writer is used as an explicit adapter; otherwise the
    canonical W3a ``build_value_bank`` writer is loaded lazily.
    """

    traces = tuple(completed_traces)
    if not traces:
        raise ValueError("generate_value_bank requires at least one completed trace")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"value-bank output already exists: {destination}")
    if isinstance(config, PFGRLiteConfig):
        value_config: Any = config.value
    elif isinstance(config, ValueModelConfig):
        value_config = config
    elif isinstance(config, Mapping):
        value_config = ValueModelConfig.from_dict(config)
    else:
        raise TypeError("config must be PFGRLiteConfig, ValueModelConfig, or strict mapping")
    rows: list[object] = []
    if effect_measure is not None:
        if target_provider is None:
            raise ValueError("effect_measure requires target_provider")
        # Complete route/proposal traces are already sealed; only now join
        # targets and measure labels.
        for trace in traces:
            target_context = _invoke(target_provider, trace, trace=trace)
            measured = _invoke(effect_measure, trace, target_context, trace=trace, target_context=target_context, config=config)
            rows.extend(tuple(measured) if isinstance(measured, Iterable) and not isinstance(measured, (str, bytes, Mapping)) else (measured,))
    else:
        for trace in traces:
            values: Any = trace
            if isinstance(trace, Mapping):
                values = trace.get("rows", trace.get("labels"))
                if values is None and "action_id" in trace:
                    values = trace
                if values is None:
                    values = ()
            else:
                values = getattr(trace, "rows", getattr(trace, "labels", trace))
            if isinstance(values, Mapping) or hasattr(values, "action_id"):
                values = (values,)
            rows.extend(tuple(values))
    if not rows:
        raise ValueError("generate_value_bank produced no measured rows")
    if writer is not None:
        result = _invoke(
            writer,
            rows,
            destination,
            output_dir=destination,
            producer=producer_bundle,
            config=value_config,
            role_manifest=role_manifest,
            engineering_only=engineering_only,
            stage_provenance=stage_provenance,
        )
        return result
    from .value_bank import build_value_bank

    producer = producer_bundle
    if producer is None:
        raise ValueError("producer_bundle is required for canonical ValueBankWriter")
    split_role_hash = role_manifest.digest if role_manifest is not None else canonical_digest("engineering", prefix="pfgr-lite-engineering-role-v1|")
    return build_value_bank(
        rows,
        destination,
        producer=producer,
        split_role_hash=split_role_hash,
        config=value_config,
        role_manifest=role_manifest,
        stage_provenance=stage_provenance,
        engineering_only=bool(engineering_only or (role_manifest.engineering_only if role_manifest is not None else False)),
    )


def _fit_cached_value(
    inputs: StageInputs,
    execution: StageExecutionConfig,
    bank: object,
    output_dir: Path,
    *,
    source_scale: object | None = None,
) -> object:
    """Run the canonical W3a fit when W5 has not supplied an adapter.

    This default deliberately imports only the cached ValueNet service at the
    call site.  The bank reader owns all descriptor/scale/provenance checks;
    no observation, target, U, decoder or teacher dependency is reachable
    from this path.  W4 may still inject a richer fitter for checkpoint/RNG
    envelopes without changing the public S3/S4 stage entrypoints.
    """

    if inputs.value_fitter is not None:
        kwargs: dict[str, Any] = {
            "output_dir": output_dir,
            "config": execution.config,
            "stage_state": inputs.resume,
        }
        if source_scale is not None:
            kwargs["source_scale"] = source_scale
        return _invoke(inputs.value_fitter, bank, **kwargs)
    from .value_net import ValueFitOptions, fit_value

    fit_options = inputs.metadata.get("value_fit_options") if isinstance(inputs.metadata, Mapping) else None
    if fit_options is None:
        fit_options = ValueFitOptions(
            epochs=execution.stage_options.epochs,
            batch_size=execution.stage_options.batch_size,
            seed=execution.stage_options.seed,
            learning_rate=execution.stage_options.learning_rate,
            weight_decay=execution.stage_options.weight_decay,
            device=execution.stage_options.device,
            loss="mse",
            max_updates=execution.stage_options.max_updates,
        )
    elif isinstance(fit_options, Mapping):
        fit_options = ValueFitOptions(**dict(fit_options))
    elif not isinstance(fit_options, ValueFitOptions):
        raise TypeError("metadata['value_fit_options'] must be ValueFitOptions or a strict mapping")
    value_model = inputs.metadata.get("value_model") if isinstance(inputs.metadata, Mapping) else None
    value_optimizer = inputs.metadata.get("value_optimizer") if isinstance(inputs.metadata, Mapping) else None
    if value_optimizer is None:
        # An explicitly supplied StageInputs.optimizer may be the W3a V
        # optimizer; fit_value validates Adam and exact parameter ownership.
        value_optimizer = inputs.optimizer
    stage_state = inputs.metadata.get("value_stage_state") if isinstance(inputs.metadata, Mapping) else None
    if stage_state is None:
        stage_state = StageState(
            stage="value_fit",
            epoch=0,
            update=0,
            microstep=0,
            optimizer_groups=("value_net",),
            completion="pending",
        )
    if not isinstance(stage_state, StageState):
        raise TypeError("metadata['value_stage_state'] must be StageState")
    result = fit_value(
        bank,
        value_model=value_model,
        optimizer=value_optimizer,
        config=execution.config.value,
        stage_state=stage_state,
        epochs=fit_options.epochs,
        batch_size=fit_options.batch_size,
        seed=fit_options.seed,
        device=fit_options.device,
        learning_rate=fit_options.learning_rate,
        weight_decay=fit_options.weight_decay,
        loss=fit_options.loss,
        shuffle=fit_options.shuffle,
        max_updates=fit_options.max_updates,
        robust_ablation=fit_options.robust_ablation,
    )
    if source_scale is not None:
        # Retain the caller's original fixed-scale provenance in the stage
        # payload; fit_value itself reads the immutable bank scale and never
        # recomputes it from newly collected rows.
        return {"value_fit": result, "source_scale": source_scale}
    return result


def _fit_result_artifact(result: object) -> tuple[object, int, int, int, dict[str, Any]]:
    """Extract real W3a fit progress and a bounded fitted-state handoff."""

    fit_result = result.get("value_fit") if isinstance(result, Mapping) and "value_fit" in result else result
    fit_metrics = fit_result.get("metrics") if isinstance(fit_result, Mapping) else getattr(fit_result, "metrics", None)
    fit_metrics = dict(fit_metrics) if isinstance(fit_metrics, Mapping) else {}
    fit_stage = fit_result.get("stage_state") if isinstance(fit_result, Mapping) else getattr(fit_result, "stage_state", None)
    stage_update = fit_stage.get("update") if isinstance(fit_stage, Mapping) else getattr(fit_stage, "update", None)
    updates = int(stage_update if stage_update is not None else fit_metrics.get("train_batch_count", fit_metrics.get("optimizer_steps", 0)) or 0)
    gradient_steps = int(fit_metrics.get("v_gradient_l2_norm_count", fit_metrics.get("train_batch_count", updates)) or 0)
    subjects = int(fit_metrics.get("subject_count", fit_metrics.get("subjects", 0)) or 0)
    model = fit_result.get("model") if isinstance(fit_result, Mapping) else getattr(fit_result, "model", None)
    model_state = None
    if isinstance(model, nn.Module):
        model_state = _clone_runtime_value(model.state_dict())
    resume_state = fit_result.get("resume_state") if isinstance(fit_result, Mapping) else getattr(fit_result, "resume_state", None)
    declared_complete = fit_result.get("complete") if isinstance(fit_result, Mapping) and "complete" in fit_result else getattr(fit_result, "complete", None)
    if declared_complete is None:
        declared_complete = fit_metrics.get("fit_complete", fit_metrics.get("completed", False))
    runtime: dict[str, Any] = {
        "fit_complete": bool(declared_complete),
        "fit_stage_state": _jsonable(fit_stage),
        "model_state_dict": model_state,
        "resume_state": _clone_runtime_value(resume_state) if isinstance(resume_state, Mapping) else resume_state,
        "fit_identity": _jsonable(fit_result.get("identity") if isinstance(fit_result, Mapping) else getattr(fit_result, "identity", None)),
        "gain_scale": _jsonable(fit_result.get("gain_scale") if isinstance(fit_result, Mapping) else getattr(fit_result, "gain_scale", None)),
    }
    return fit_result, gradient_steps, updates, subjects, runtime


def _fit_subject_count(bank: object) -> int:
    """Derive the fitted subject denominator from the actual cached rows."""

    reader = bank
    if not hasattr(reader, "rows"):
        from .value_bank import ValueBankReader

        reader = ValueBankReader(bank)
    rows_attr = getattr(reader, "rows", None)
    if rows_attr is None:
        return 0
    rows = rows_attr() if callable(rows_attr) else rows_attr
    if rows is None:
        return 0
    subject_ids: set[str] = set()
    for row in rows:
        subject = getattr(row, "subject_key", getattr(row, "subject_id", None))
        if subject is None and isinstance(row, Mapping):
            subject = row.get("subject_key", row.get("subject_id"))
        if isinstance(subject, str) and subject:
            subject_ids.add(subject)
    return len(subject_ids)


def _run_s3(inputs: StageInputs, execution: StageExecutionConfig, output_dir: Path) -> tuple[dict[str, Any], tuple[str, ...], int, int, int, dict[str, Any]]:
    bank = inputs.metadata.get("bank")
    if bank is None:
        bank = inputs.metadata.get("bank_path")
    if bank is None:
        raise ValueError("S3 requires metadata['bank'] or metadata['bank_path']")
    result = _fit_cached_value(inputs, execution, bank, output_dir)
    fit_result, gradient_steps, updates, subjects, runtime = _fit_result_artifact(result)
    actual_subject_count = _fit_subject_count(bank)
    if actual_subject_count:
        subjects = actual_subject_count
    fit_metric_map = fit_result.get("metrics", {}) if isinstance(fit_result, Mapping) else getattr(fit_result, "metrics", {})
    if not isinstance(fit_metric_map, Mapping):
        fit_metric_map = {}
    metrics = {
        "fit": _jsonable(result),
        "cached_only": True,
        "fit_gradient_steps": gradient_steps,
        "fit_optimizer_steps": updates,
        "fit_subject_count": subjects,
        "fit_row_count": int(fit_metric_map.get("train_row_count", 0) or 0),
        "fit_complete": bool(runtime.get("fit_complete", False)),
        "cached_dependency_counters": dict(fit_metric_map.get("dependency_counters", {})) if isinstance(fit_metric_map.get("dependency_counters", {}), Mapping) else {},
        "scope": "cached_value_fit_only",
    }
    return metrics, ("value_net",), gradient_steps, updates, subjects, runtime


def _validate_s4_source_scale(bank: object, source_scale: object, *, engineering_only: bool) -> object:
    """Require S4's source scale to equal the immutable bank scale exactly."""

    from .value_bank import GainScale, ValueBankReader

    expected = getattr(bank, "gain_scale", None)
    if expected is None and isinstance(bank, (str, Path)):
        expected = ValueBankReader(bank).gain_scale
    if expected is None:
        if engineering_only:
            return source_scale
        raise ValueError("production S4 requires a bank reader with fixed gain-scale provenance")
    if isinstance(source_scale, Mapping):
        payload = dict(source_scale)
        declared_digest = payload.pop("digest", None)
        source_scale = GainScale(**payload)
        if declared_digest is not None and declared_digest != source_scale.digest:
            raise ValueError("S4 source_scale digest does not match its fields")
    if not isinstance(source_scale, GainScale):
        raise TypeError("S4 source_scale must be GainScale or its canonical mapping")
    if source_scale.digest != expected.digest or source_scale.as_dict() != expected.as_dict():
        raise ValueError("S4 source_scale does not match the immutable bank gain scale")
    return source_scale


def _run_s4(inputs: StageInputs, execution: StageExecutionConfig, output_dir: Path) -> tuple[dict[str, Any], tuple[str, ...], int, int, int, dict[str, Any]]:
    source_scale = inputs.metadata.get("source_scale")
    if source_scale is None:
        raise ValueError("S4 requires source_scale provenance; it cannot silently recompute gain scale")
    bank = inputs.metadata.get("bank")
    if bank is None:
        bank = inputs.metadata.get("bank_path")
    if bank is None:
        raise ValueError("S4 requires metadata['bank'] or metadata['bank_path']")
    source_scale = _validate_s4_source_scale(bank, source_scale, engineering_only=execution.stage_options.engineering_only)
    result = _fit_cached_value(inputs, execution, bank, output_dir, source_scale=source_scale)
    fit_result, gradient_steps, updates, subjects, runtime = _fit_result_artifact(result)
    actual_subject_count = _fit_subject_count(bank)
    if actual_subject_count:
        subjects = actual_subject_count
    fit_metric_map = fit_result.get("metrics", {}) if isinstance(fit_result, Mapping) else getattr(fit_result, "metrics", {})
    if not isinstance(fit_metric_map, Mapping):
        fit_metric_map = {}
    metrics = {
        "fit": _jsonable(result),
        "source_scale": _jsonable(source_scale),
        "fit_gradient_steps": gradient_steps,
        "fit_optimizer_steps": updates,
        "fit_subject_count": subjects,
        "fit_row_count": int(fit_metric_map.get("train_row_count", 0) or 0),
        "fit_complete": bool(runtime.get("fit_complete", False)),
        "cached_dependency_counters": dict(fit_metric_map.get("dependency_counters", {})) if isinstance(fit_metric_map.get("dependency_counters", {}), Mapping) else {},
        "source_scale_provenance": _jsonable(source_scale),
        "scope": "cached_value_refit_only_no_collection",
    }
    return metrics, ("value_net",), gradient_steps, updates, subjects, runtime


def _calibration_pending(result: Mapping[str, Any]) -> tuple[bool, str]:
    """Return whether a W5 calibration result is still pending.

    W5 deliberately returns ``calibration=None`` for underpowered or
    inconclusive collections.  The stage layer must preserve that absence and
    never let an old diagnostic identity calibration unlock a completed S5
    state.  Explicit status/insufficient-data metadata is checked as well so
    this remains safe during the W5 runner transition.
    """

    metrics = result.get("metrics") if isinstance(result.get("metrics"), Mapping) else {}
    status_candidates = (
        result.get("status"),
        result.get("calibration_status"),
        metrics.get("status"),
        metrics.get("calibration_status"),
    )
    statuses = {str(value).strip().upper() for value in status_candidates if value is not None}
    insufficient = bool(metrics.get("insufficient_data", result.get("insufficient_data", False)))
    pending = result.get("calibration") is None or insufficient or bool(statuses & {"INCONCLUSIVE", "PENDING", "INSUFFICIENT_DATA"})
    if pending:
        return True, next((value for value in ("INCONCLUSIVE", "PENDING") if value in statuses), "INCONCLUSIVE")
    return False, next((value for value in statuses if value not in {"READY", "COMPLETE", "COMPLETED"}), "READY")


def _run_s5(inputs: StageInputs, execution: StageExecutionConfig, output_dir: Path) -> tuple[dict[str, Any], tuple[str, ...], int, int, int, dict[str, Any]]:
    # The production collection seam is owned by the W4/W5 calibration
    # runner.  It receives the complete StageInputs bundle and returns sealed
    # ForcedCalibrationTrace/evidence artifacts; this stage must not recreate
    # a competing callback-only fitter or fabricate calibration JSON.
    runner = inputs.metadata.get("calibration_runner") if isinstance(inputs.metadata, Mapping) else None
    runner_options = inputs.metadata.get("calibration_run_options") if isinstance(inputs.metadata, Mapping) else None
    if runner is None:
        try:
            from .calibration_runner import run_calibration  # type: ignore

            runner = run_calibration
        except ImportError:
            runner = None
    if runner is not None:
        if runner_options is None:
            raise ValueError("S5 calibration_runner requires typed metadata['calibration_run_options']")
        if isinstance(runner_options, Mapping):
            raise TypeError("S5 calibration_run_options must be CalibrationRunOptions, not a mapping")
        if not execution.stage_options.engineering_only and inputs.metadata.get("subject_role") != "calibration":
            raise ValueError("production S5 StageInputs must use the calibration_fit union role")
        declared_engineering = getattr(runner_options, "engineering_only", None)
        if declared_engineering is not None and bool(declared_engineering) != execution.stage_options.engineering_only:
            raise ValueError("S5 calibration options engineering_only does not match StageOptions")
        # Direct run_stage callers may provide only PFGRLiteConfig plus
        # StageOptions; hand the concrete resolved execution envelope to the
        # runner without mutating the caller's immutable StageInputs bundle.
        runner_inputs = inputs if inputs.execution is not None else replace(inputs, execution=execution, config=execution.config, stage_options=execution.stage_options)
        result = _invoke(runner, runner_inputs, runner_options, output_dir)
        if not isinstance(result, Mapping):
            raise TypeError("calibration_runner must return a mapping")
        production = not execution.stage_options.engineering_only
        if production:
            required = {"schema_version", "calibration_evidence", "fit_winners", "allowance_winners", "completed_traces", "collection_policy", "calibration", "artifacts", "metrics"}
            missing = required - set(result)
            if missing:
                raise ValueError(f"production calibration_runner result is missing: {sorted(missing)}")
            from .calibration import ForcedCalibrationTrace

            traces = result.get("completed_traces")
            if not isinstance(traces, Sequence) or not traces or any(not isinstance(item, ForcedCalibrationTrace) for item in traces):
                raise ValueError("production S5 requires sealed ForcedCalibrationTrace values from calibration_runner")
            policy = result.get("collection_policy")
            if policy is None:
                raise ValueError("production S5 result is missing collection_policy")
        trace_count = len(result.get("completed_traces", ())) if isinstance(result.get("completed_traces", ()), Sequence) else 0
        pending, calibration_status = _calibration_pending(result)
        # Keep the runner's sealed evidence, traces, and measurements intact,
        # but remove any legacy/diagnostic identity fit when the result is
        # underpowered.  W5's current contract is calibration=None in this
        # case; this guard keeps S5 fail-closed while older runners roll over.
        result_payload = dict(result)
        if pending:
            result_payload["calibration"] = None
        metrics = dict(result_payload.get("metrics", {})) if isinstance(result_payload.get("metrics"), Mapping) else {}
        metrics.update({"target_join": "deferred_after_forced_trace", "completed_trace_count": trace_count, "calibration_runner": True})
        metrics["calibration_status"] = calibration_status
        metrics["calibration_complete"] = not pending
        runtime = {
            "calibration_result": result_payload,
            "completed_traces": result_payload.get("completed_traces"),
            "collection_policy": result_payload.get("collection_policy"),
        }
        return {"calibration": _jsonable(result_payload), **metrics}, (), 0, 0, trace_count, runtime
    if not execution.stage_options.engineering_only:
        raise ValueError("production S5 requires the concrete calibration_runner service")
    if inputs.calibration_fitter is None:
        raise ValueError("engineering S5 requires a calibration_fitter or calibration_runner adapter")
    forced_builder = inputs.behavior_builder or inputs.route_builder
    traces: list[object] = []
    bindings: list[Mapping[str, Any]] = []
    if forced_builder is None and inputs.model is not None and inputs.query is not None and inputs.writer is not None:
        forced_builder = lambda sample, **kwargs: _default_random_route(sample, model=inputs.model, options=execution.stage_options, inputs=inputs, k=4, seed=kwargs.get("seed", execution.stage_options.seed))
    if forced_builder is not None:
        for index, sample in enumerate(inputs.samples):
            trace = _invoke(forced_builder, sample, sample=sample, k=4, forced=True, seed=execution.stage_options.seed + index, options=execution.stage_options, config=execution.config)
            traces.append(trace)
            context = getattr(trace, "context", None)
            if isinstance(trace, Mapping):
                context = trace.get("context", context)
            if context is not None:
                bindings.append(bind_observation_context(sample, context).as_dict())
    else:
        traces = list(inputs.samples)
    result = _invoke(inputs.calibration_fitter, traces, output_dir=output_dir, config=execution.config, target_provider=inputs.target_provider, seed=execution.stage_options.seed, trace_subject_bindings=bindings, subject_context_bindings=bindings)
    return {"calibration": _jsonable(result), "target_join": "deferred_after_forced_trace"}, (), 0, 0, len(inputs.samples), {"calibration_result": result, "completed_traces": traces}


def _run_s6(inputs: StageInputs, execution: StageExecutionConfig, output_dir: Path) -> tuple[dict[str, Any], tuple[str, ...], int, int, int]:
    evaluator = inputs.evaluator
    if evaluator is None:
        try:
            from . import experiments  # type: ignore

            evaluator = getattr(experiments, "run_evaluation", None)
        except ImportError as exc:
            raise ValueError("S6 requires W5 experiments.run_evaluation or an explicit evaluator") from exc
    if evaluator is None:
        raise ValueError("W5 experiments service does not expose run_evaluation")
    # W5's frozen service signature is run_evaluation(inputs, ExperimentOptions,
    # output_dir).  StageOptions controls training/stage mechanics and is not
    # silently reinterpreted as an evaluation scenario; callers must provide
    # an explicit ExperimentOptions envelope (or documented scalar metadata
    # for a static engineering default).
    from .experiments import ExperimentOptions

    declared = inputs.metadata.get("experiment_options") if isinstance(inputs.metadata, Mapping) else None
    if declared is None:
        declared = {
            "scenario": inputs.metadata.get("scenario", "static") if isinstance(inputs.metadata, Mapping) else "static",
            "budget": inputs.metadata.get("budget", 0) if isinstance(inputs.metadata, Mapping) else 0,
            "max_subjects": inputs.metadata.get("max_subjects", max(len(inputs.samples), 1)) if isinstance(inputs.metadata, Mapping) else max(len(inputs.samples), 1),
            "seed": execution.stage_options.seed,
            "split_role": inputs.metadata.get("split_role", "validation") if isinstance(inputs.metadata, Mapping) else "validation",
            "decode_chunk_size": execution.stage_options.query_chunk_size,
            "engineering_only": execution.stage_options.engineering_only,
        }
    if isinstance(declared, Mapping):
        experiment_options = ExperimentOptions.from_dict(declared)
    elif isinstance(declared, ExperimentOptions):
        experiment_options = declared
    else:
        raise TypeError("metadata['experiment_options'] must be ExperimentOptions or a strict mapping")
    result = _invoke(evaluator, inputs, experiment_options, output_dir)
    return {"evaluation": _jsonable(result), "targets": "deferred_after_prediction"}, (), 0, 0, len(inputs.samples)


def _canonical_lattice_factory() -> object:
    """Return the W2 PFGRQueryLattice factory without importing it at module load."""

    from .footprint import PFGRQueryLattice

    class _CanonicalLatticeFactory:
        def build(self, **kwargs: Any) -> object:
            return PFGRQueryLattice.build(**kwargs)

    return _CanonicalLatticeFactory()


def build_stage_inputs(
    config: PFGRLiteConfig | Mapping[str, Any],
    *,
    data_root: str | Path,
    split_file: str | Path,
    role_manifest: TrainingRoleManifest | Mapping[str, Any] | None = None,
    roles_file: str | Path | None = None,
    model: object | None = None,
    model_factory: Callable[..., object] | None = None,
    query_lattice_factory: object | None = None,
    frontend_config: object | None = None,
    checkpoint_path: str | Path | None = None,
    medicalnet_checkpoint_path: str | Path | None = None,
    medicalnet_checkpoint_sha256: str | None = None,
    normalization_config: Mapping[str, Any] | None = None,
    stage_options: StageOptions | Mapping[str, Any] | None = None,
    counters: object | None = None,
    target_loader: Callable[..., object] | None = None,
    subject_role: str = "producer_fit",
    max_subjects: int | None = None,
) -> StageInputs:
    """Construct production StageInputs from resolved config and data paths.

    This is the CLI/W5 factory seam: it validates the externally reviewed
    baseline split before loading any target, creates/loads the authoritative
    training-role manifest, and resolves the model/data normalization recipe.
    W2/W4 query/writer/policy services remain explicit injections on the
    returned bundle and are never reconstructed here.
    """

    if isinstance(config, Mapping):
        resolved_config = PFGRLiteConfig.from_dict(config)
    elif isinstance(config, PFGRLiteConfig):
        resolved_config = config
    else:
        raise TypeError("config must be PFGRLiteConfig or strict mapping")
    if checkpoint_path is not None and medicalnet_checkpoint_path is not None:
        raise ValueError("checkpoint_path and medicalnet_checkpoint_path are mutually exclusive")
    from smagm.data.brats21_point_guided import load_point_guided_split

    split = load_point_guided_split(split_file)
    if checkpoint_path is not None and not Path(checkpoint_path).is_file():
        raise FileNotFoundError(checkpoint_path)
    if medicalnet_checkpoint_path is not None and not Path(medicalnet_checkpoint_path).is_file():
        raise FileNotFoundError(medicalnet_checkpoint_path)
    supplied_role_manifest = roles_file is not None or role_manifest is not None
    if roles_file is not None:
        payload = json.loads(Path(roles_file).read_text(encoding="utf-8"))
        resolved_roles = TrainingRoleManifest.from_dict(payload)
    elif role_manifest is not None:
        resolved_roles = TrainingRoleManifest.from_dict(role_manifest) if isinstance(role_manifest, Mapping) else role_manifest
    else:
        resolved_roles = build_training_role_manifest(split, engineering_only=resolved_config.engineering_only)
    if resolved_roles.baseline_split_hash != split.split_hash:
        raise ValueError("training-role manifest baseline split hash does not match reviewed split")
    options = stage_options
    if options is None:
        options = StageOptions(engineering_only=resolved_config.engineering_only)
    elif isinstance(options, Mapping):
        options = StageOptions.from_dict(options)
    if not isinstance(options, StageOptions):
        raise TypeError("stage_options must be StageOptions or strict mapping")
    from .data import DataAccessCounters

    if counters is None:
        counters = DataAccessCounters()
    elif not isinstance(counters, DataAccessCounters):
        raise TypeError("counters must be a DataAccessCounters instance")
    allowed_roles = {
        "producer_fit",
        "calibration",
        "calibration_fit",
        "calibration_allowance",
        "validation",
        "test",
    }
    if subject_role not in allowed_roles:
        raise ValueError(f"subject_role must be one of {sorted(allowed_roles)}")
    if max_subjects is not None:
        _positive_int("max_subjects", max_subjects, maximum=1_000_000)
    if frontend_config is not None and isinstance(frontend_config, Mapping):
        from .config import frontend_config_from_dict

        frontend_config = frontend_config_from_dict(frontend_config)
    if medicalnet_checkpoint_path is not None:
        from ..config import PointGuidedConfig

        existing_path = getattr(frontend_config, "medicalnet_checkpoint_path", None)
        if existing_path is not None and Path(existing_path).resolve() != Path(medicalnet_checkpoint_path).resolve():
            raise ValueError("frontend_config and medicalnet_checkpoint_path disagree")
        existing_hash = getattr(frontend_config, "medicalnet_checkpoint_sha256", None)
        if existing_hash is not None and medicalnet_checkpoint_sha256 is not None and existing_hash.lower() != medicalnet_checkpoint_sha256.lower():
            raise ValueError("frontend_config and medicalnet_checkpoint_sha256 disagree")
        resolved_hash = medicalnet_checkpoint_sha256 or existing_hash
        if resolved_hash is None:
            raise ValueError("medicalnet_checkpoint_sha256 is required for a verified MedicalNet source")
        if frontend_config is None:
            frontend_config = PointGuidedConfig(
                num_semantic_classes=3,
                num_points=resolved_config.num_points,
                point_candidate_multiplier=3,
                offset_hidden_channels=12,
                medicalnet_checkpoint_path=Path(medicalnet_checkpoint_path).resolve(),
                medicalnet_checkpoint_sha256=resolved_hash,
                require_pretrained_backbone=True,
            )
    if frontend_config is not None and not resolved_config.engineering_only:
        source_path = getattr(frontend_config, "medicalnet_checkpoint_path", None)
        source_hash = getattr(frontend_config, "medicalnet_checkpoint_sha256", None)
        if source_path is None or source_hash is None or not bool(getattr(frontend_config, "require_pretrained_backbone", False)):
            raise ValueError("production StageInputs require a verified MedicalNet checkpoint path and SHA-256")
    if frontend_config is None and not resolved_config.engineering_only and checkpoint_path is None:
        raise ValueError("production StageInputs require frontend_config with verified MedicalNet provenance or checkpoint_path")
    # Resolve the concrete loader recipe first, then bind the PFGR producer to
    # its derived identity.  The legacy PFGR config's historical default is a
    # schema label, not a loader policy; silently passing that label to the
    # BraTS normalizer would produce an invalid/ambiguous target join.
    normalization_fields = dict(normalization_config or {"normalization_policy": "masked_zscore"})
    provisional_execution = StageExecutionConfig(
        config=resolved_config,
        frontend_sidecar={"checkpoint_path": None if checkpoint_path is None else str(Path(checkpoint_path).resolve()), "medicalnet_checkpoint_path": None if medicalnet_checkpoint_path is None else str(Path(medicalnet_checkpoint_path).resolve()), "schema_version": "pfgr-lite-frontend-sidecar-v1"},
        normalization=normalization_fields,
        stage_options=options,
    )
    recipe_identity = str(provisional_execution.normalization.get("recipe_identity", ""))
    if not recipe_identity:
        raise ValueError("normalization recipe identity could not be resolved")
    # PFGRLiteModel computes ProducerCompatibility normalization as
    # canonical_digest(config.observation_normalization, prefix=...).  Store
    # the derived recipe hash in that authoritative config field so its
    # compatibility value equals StageExecutionConfig.normalization_hash.
    resolved_config = replace(resolved_config, observation_normalization=recipe_identity)
    execution = StageExecutionConfig(
        config=resolved_config,
        frontend_sidecar=provisional_execution.frontend_sidecar,
        normalization=normalization_fields,
        stage_options=options,
    )
    # Seed before constructing a factory/default model so arm comparisons and
    # resumed runs can reproduce initialization.  NumPy is optional for the
    # lightweight package; when present, seed it explicitly as well.
    random.seed(options.seed)
    torch.manual_seed(options.seed)
    try:
        import numpy as np

        np.random.seed(options.seed)
    except ImportError:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(options.seed)
    # Resolve a checkpoint's immutable role/source identities before selecting
    # and loading samples.  Otherwise a caller that omits roles would load a
    # newly generated assignment and only later discover that the checkpoint
    # carries a different reviewed manifest.
    checkpoint_bundle: object | None = None
    hydrated_producer: ProducerDependencies | None = None
    hydrated_stage_provenance: Mapping[str, Any] | None = None
    if checkpoint_path is not None:
        from .checkpoint import load_inference_bundle

        checkpoint_bundle = load_inference_bundle(checkpoint_path, expected_split_hash=split.split_hash)
        bundle_roles = getattr(checkpoint_bundle, "role_manifest", None)
        if bundle_roles is not None:
            if supplied_role_manifest and resolved_roles.digest != bundle_roles.digest:
                raise ValueError("checkpoint role manifest does not match the supplied role manifest")
            # A checkpoint's role assignment is part of its reviewed source
            # identity.  Use it as the authoritative manifest when callers did
            # not explicitly provide one, rather than rebuilding a potentially
            # different random assignment from the baseline split.
            resolved_roles = bundle_roles
        bundle_config = PFGRLiteConfig.from_dict(checkpoint_bundle.config.get("pfgr_config", {}))
        if bundle_config != resolved_config:
            raise ValueError("checkpoint PFGR config does not match resolved execution config")
        hydrated_producer = checkpoint_bundle.producer
        hydrated_stage_provenance = checkpoint_bundle.stage_provenance
    root = Path(data_root).resolve()
    role_ids = {
        "producer_fit": resolved_roles.producer_fit_subject_ids,
        "calibration": tuple(sorted(set(resolved_roles.calibration_fit_subject_ids) | set(resolved_roles.calibration_allowance_subject_ids))),
        "calibration_fit": resolved_roles.calibration_fit_subject_ids,
        "calibration_allowance": resolved_roles.calibration_allowance_subject_ids,
        "validation": resolved_roles.baseline_validation_subject_ids,
        "test": resolved_roles.baseline_test_subject_ids,
    }
    sample_ids = role_ids[subject_role]
    if max_subjects is not None:
        sample_ids = sample_ids[:max_subjects]
    samples: list[TargetFreeSample] = []
    for subject_id in sample_ids:
        candidate = root / subject_id
        source = candidate if candidate.exists() else subject_id
        loaded_sample = load_observation_sample(source, normalization_config=normalization_fields, counters=counters)
        # The detached sample keeps the recipe fields and records the exact
        # producer-bound identity expected by the actual PFGR context.
        sample_metadata = dict(loaded_sample.normalization_metadata)
        sample_metadata["producer_normalization_hash"] = execution.normalization_hash
        sample_metadata["recipe_identity"] = recipe_identity
        samples.append(replace(loaded_sample, normalization_hash=execution.normalization_hash, normalization_metadata=sample_metadata))
    if query_lattice_factory is None:
        query_lattice_factory = _canonical_lattice_factory()
    if checkpoint_bundle is not None:
        from .checkpoint import hydrate_inference_model

        model = hydrate_inference_model(
            checkpoint_bundle,
            model_factory=model_factory,
            query_lattice_factory=query_lattice_factory,
        )
        # The checkpoint bundle is the source of truth for this model's
        # MedicalNet provenance; an external path must not silently replace it.
    elif model is None and model_factory is not None:
        model = _invoke(model_factory, resolved_config, config=resolved_config, frontend_config=frontend_config, query_lattice_factory=query_lattice_factory, checkpoint_path=checkpoint_path, medicalnet_checkpoint_path=medicalnet_checkpoint_path)
    if model is None:
        from .model import PFGRLiteModel

        model = PFGRLiteModel(resolved_config, frontend_config=frontend_config, query_lattice_factory=query_lattice_factory)
    # Apply the requested execution device to the model before stage helpers
    # move observation tensors.  PyTorch modules mutate in place; custom test
    # factories may return a replacement module, so retain a non-None return.
    to_device = getattr(model, "to", None)
    if callable(to_device):
        moved = to_device(options.device)
        if moved is not None:
            model = moved
    setter = getattr(model, "set_query_lattice_factory", getattr(model, "set_query_lattice_builder", None))
    if callable(setter):
        if query_lattice_factory is None and getattr(model, "_query_lattice_factory", None) is None:
            query_lattice_factory = _canonical_lattice_factory()
        if query_lattice_factory is not None:
            setter(query_lattice_factory)
    if not resolved_config.engineering_only:
        prior = getattr(getattr(model, "frontend", None), "semantic_prior", None)
        source = getattr(prior, "backbone_provenance", None)
        integrity = bool(getattr(source, "integrity_verified", getattr(source, "checkpoint_integrity_verified", False))) if source is not None else False
        if source is None or not integrity or not bool(getattr(source, "official_pretrained_verified", False)):
            raise ValueError("production StageInputs require verified MedicalNet checkpoint provenance; synthetic/unverified backbones are denied")
    from .sparse_write import make_action_writer, make_point_query, make_support_legal_mask

    # W2's query is stateless and can be shared across subjects; its writer is
    # geometry-bound, so build one canonical PFGR lattice per completed
    # observation context.  This gives factory-created inputs a real S1/S2
    # path while retaining explicit injection for reviewed alternatives.
    default_query = make_point_query()

    def _context_lattice(state: object, context: object) -> object:
        if query_lattice_factory is None:
            raise RuntimeError("canonical query lattice factory is unavailable")
        geometry = getattr(context, "geometry", None)
        feature_geometry = getattr(context, "feature_geometry", None)
        planes = getattr(state, "planes", None)
        query_dtype = getattr(getattr(planes, "xy", None), "dtype", None)
        if geometry is None or feature_geometry is None or query_dtype is None:
            raise TypeError("canonical action writer requires a complete PFGR state/context")
        kwargs = {
            "output_geometry": geometry,
            "feature_geometry": feature_geometry,
            "query_dtype": query_dtype,
            "build_chunk_size": resolved_config.build_chunk_size,
        }
        return query_lattice_factory.build(**kwargs) if hasattr(query_lattice_factory, "build") else query_lattice_factory(**kwargs)

    def default_writer(state: object, context: object, action: object) -> object:
        lattice = _context_lattice(state, context)
        return make_action_writer(lattice)(state, context, action)

    def default_support_legal_mask(state: object, context: object, points_ras_mm: Tensor) -> Tensor:
        # This is the same geometry-bound lattice used by the action writer;
        # W2 computes eligibility from retained writer nodes only.
        lattice = _context_lattice(state, context)
        return make_support_legal_mask(lattice)(state, context, points_ras_mm)

    if target_loader is None:
        from smagm.data.brats21_point_guided import load_point_guided_subject

        target_normalization = dict(normalization_fields)
        target_allowed = {
            "brain_mask_threshold",
            "normalization_epsilon",
            "normalization_policy",
            "lower_percentile",
            "upper_percentile",
        }
        target_normalization = {key: target_normalization[key] for key in target_allowed if key in target_normalization}
        expected_subjects = frozenset(str(item) for item in sample_ids)

        def target_loader(subject_id: str):
            if not isinstance(subject_id, str) or subject_id not in expected_subjects:
                raise ValueError("target provider accepts only the selected subject_id strings")
            source = root / subject_id
            if not source.is_dir():
                raise FileNotFoundError(source)
            return load_point_guided_subject(
                source,
                require_target=True,
                load_target=True,
                # The optional S0 semantic arm joins labels only after the
                # target-free prediction.  Keep the default reconstruction
                # path target-only and make the late segmentation read
                # explicit in its loader flags/counters.
                require_segmentation=bool(options.semantic_objective),
                load_segmentation=bool(options.semantic_objective),
                **target_normalization,
            )
    return StageInputs(
        samples=tuple(samples),
        model=model,
        query=default_query,
        writer=default_writer,
        execution=execution,
        role_manifest=resolved_roles,
        target_provider=target_loader,
        metadata={
            "data_root": str(root),
            "split_file": str(Path(split_file).resolve()),
            "normalization_hash": execution.normalization_hash,
            "normalization_recipe_identity": recipe_identity,
            "subject_role": subject_role,
            "selected_subject_ids": tuple(sample_ids),
            "counters": counters,
            "initialization_id": canonical_digest({"seed": options.seed, "stage": options.stage, "device": options.device}, prefix="pfgr-lite-stage-initialization-v1|"),
            "source_id": str(Path(checkpoint_path).resolve()) if checkpoint_path is not None else (str(Path(medicalnet_checkpoint_path).resolve()) if medicalnet_checkpoint_path is not None else "engineering-source"),
            "checkpoint_id": str(Path(checkpoint_path).resolve()) if checkpoint_path is not None else "none",
            "support_legal_mask": default_support_legal_mask,
            "stage_provenance": hydrated_stage_provenance if checkpoint_path is not None else None,
        },
        producer=hydrated_producer if checkpoint_path is not None else None,
    )


def run_stage(stage: str, config: PFGRLiteConfig | Mapping[str, Any], inputs: StageInputs, output_dir: str | Path) -> StageResult:
    """Run one bounded stage and atomically publish its receipt.

    The output directory is reserved with ``exist_ok=False`` before any work;
    failures remove only that newly-created directory and never mutate source
    samples.  A successful invocation returns W1's canonical :class:`StageState`;
    richer metrics and provenance live in ``stage_receipt.json``.
    """

    stage_name = _stage_name(stage)
    if not isinstance(inputs, StageInputs):
        raise TypeError("inputs must be StageInputs")
    execution = _resolve_execution(stage_name, config, inputs)
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"stage output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    observation_reads_before = _access_count(inputs, "observation_reads", 0)
    target_reads_before = _access_count(inputs, "target_reads", 0)
    segmentation_reads_before = _access_count(inputs, "segmentation_reads", 0)
    try:
        if stage_name == "S0":
            result = _run_s0(inputs, execution)
            metrics, groups, grad_steps, updates, subjects, runtime_state = result
            provenance = None
        elif stage_name == "S1":
            metrics, groups, grad_steps, updates, subjects, provenance, runtime_state = _run_s1(inputs, execution)
        elif stage_name == "S2":
            metrics, groups, grad_steps, updates, subjects = _run_s2(inputs, execution, destination)
            provenance = None
            runtime_state = {}
        elif stage_name == "S3":
            metrics, groups, grad_steps, updates, subjects, runtime_state = _run_s3(inputs, execution, destination)
            provenance = None
        elif stage_name == "S4":
            metrics, groups, grad_steps, updates, subjects, runtime_state = _run_s4(inputs, execution, destination)
            provenance = None
        elif stage_name == "S5":
            metrics, groups, grad_steps, updates, subjects, runtime_state = _run_s5(inputs, execution, destination)
            provenance = None
        else:
            metrics, groups, grad_steps, updates, subjects = _run_s6(inputs, execution, destination)
            provenance = None
            runtime_state = {}
        completion_override = None
        substage_override = None
        if stage_name in {"S3", "S4"} and isinstance(metrics, Mapping) and "fit_complete" in metrics:
            completion_override = "complete" if bool(metrics.get("fit_complete")) else "pending"
            if completion_override == "pending":
                substage_override = "value_fit"
        elif stage_name == "S5" and isinstance(metrics, Mapping):
            status = str(metrics.get("calibration_status", metrics.get("status", ""))).strip().upper()
            if metrics.get("calibration_complete") is False or status in {"INCONCLUSIVE", "PENDING", "INSUFFICIENT_DATA"}:
                completion_override = "pending"
                substage_override = "calibration"
        stage_state = _stage_state_progress(
            stage_name,
            execution.stage_options,
            inputs,
            int(updates),
            tuple(groups),
            completion_override=completion_override,
            substage_override=substage_override,
        )
        receipt_metrics = dict(metrics)
        if stage_name in {"S0", "S1"} and isinstance(receipt_metrics.get("history"), Sequence) and not isinstance(receipt_metrics.get("history"), (str, bytes)):
            history_records = tuple(item for item in receipt_metrics["history"] if isinstance(item, Mapping))
            if history_records:
                receipt_metrics["history_artifact"] = _write_stage_history(destination, stage_name, history_records)
                receipt_metrics["history_record_count"] = len(history_records)
                receipt_metrics.setdefault("history_scope", "segment")
        observation_reads_after = _access_count(inputs, "observation_reads", 0)
        target_reads_after = _access_count(inputs, "target_reads", 0)
        segmentation_reads_after = _access_count(inputs, "segmentation_reads", 0)
        receipt_metrics.setdefault(
            "io_counter_scope",
            "stage_delta",
        )
        receipt_metrics.setdefault(
            "io_counters",
            {
                "observation_reads": max(0, observation_reads_after - observation_reads_before),
                "target_reads": max(0, target_reads_after - target_reads_before),
                "segmentation_reads": max(0, segmentation_reads_after - segmentation_reads_before),
            },
        )
        if stage_name in {"S3", "S4"} and "fit_complete" in receipt_metrics:
            # A bounded cached fit may legitimately return a resumable
            # pending artifact; never let the outer stage receipt imply that
            # all configured ValueNet epochs completed.
            receipt_metrics["completed"] = bool(receipt_metrics.get("fit_complete"))
        elif stage_name == "S5" and "calibration_complete" in receipt_metrics:
            # Underpowered W5 collections carry real traces/measurements but
            # no adaptive calibration; preserve that resumable pending state.
            receipt_metrics["completed"] = bool(receipt_metrics.get("calibration_complete"))
        else:
            receipt_metrics.setdefault("completed", stage_state.completion == "complete")
        receipt = StageReceipt(
            stage=stage_name,
            status="engineering_only" if execution.stage_options.engineering_only else "complete",
            optimizer_groups=tuple(groups),
            subjects=int(subjects),
            route_updates=int(updates),
            gradient_steps=int(grad_steps),
            target_reads=max(0, target_reads_after - target_reads_before),
            observation_reads=max(0, observation_reads_after - observation_reads_before),
            metrics=receipt_metrics,
            stage_provenance=provenance,
        )
        _write_receipt(destination, receipt, execution)
        return StageResult(stage_state=stage_state, receipt=receipt, output_dir=destination, runtime_state=MappingProxyType(dict(runtime_state)))
    except BaseException as error:
        # Keep the exclusively-reserved directory and publish a failure
        # receipt.  It is never marked complete and gives W5/resume tooling an
        # actionable boundary without touching source samples or replacing a
        # raced directory.
        segmentation_reads_failed = max(0, _access_count(inputs, "segmentation_reads", 0) - segmentation_reads_before)
        failure = StageReceipt(
            stage=stage_name,
            status="failed",
            subjects=len(inputs.samples),
            target_reads=max(0, _access_count(inputs, "target_reads", 0) - target_reads_before),
            observation_reads=max(0, _access_count(inputs, "observation_reads", 0) - observation_reads_before),
            metrics={
                "failure_type": type(error).__name__,
                "io_counter_scope": "stage_delta",
                "io_counters": {
                    "observation_reads": max(0, _access_count(inputs, "observation_reads", 0) - observation_reads_before),
                    "target_reads": max(0, _access_count(inputs, "target_reads", 0) - target_reads_before),
                    "segmentation_reads": segmentation_reads_failed,
                },
            },
            error=str(error)[:4000],
        )
        try:
            _write_receipt(destination, failure, execution)
        except Exception:
            # Preserve the original failure even if the filesystem itself is
            # unavailable; the reserved directory remains non-successful.
            pass
        raise


__all__ = [
    "ARMS",
    "EXECUTION_CONFIG_SCHEMA",
    "PRODUCER_STAGE_SCHEMA",
    "STAGE_OPTIONS_SCHEMA",
    "STAGE_RECEIPT_SCHEMA",
    "STAGE_RUNTIME_SCHEMA",
    "StageExecutionConfig",
    "StageInputs",
    "StageOptions",
    "StageReceipt",
    "StageResult",
    "build_stage_inputs",
    "generate_value_bank",
    "run_stage",
]
