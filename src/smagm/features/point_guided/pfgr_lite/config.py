"""Frozen PFGR-Lite configuration declarations.

These configurations are intentionally separate from the legacy
``PointGuidedConfig``.  They encode the accepted PFGR protocol and reject
silent widening of precision, candidate counts, decoder width, write bounds,
or policy semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import json
import math
from typing import Any, Literal, Mapping


PFGR_CONFIG_SCHEMA = "pfgr-lite-config-v1"
STATIC_CONFIG_SCHEMA = "pfgr-lite-static-synthesis-v1"
POLICY_CONFIG_SCHEMA = "pfgr-lite-policy-v1"
VALUE_CONFIG_SCHEMA = "pfgr-lite-value-model-v1"
TEACHER_CONFIG_SCHEMA = "pfgr-lite-effect-teacher-v1"


def _positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _canonical(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


@dataclass(frozen=True)
class StaticSynthesisConfig:
    """Versioned static-head architecture choice (B0/B1/B2/B-light)."""

    schema_version: str = STATIC_CONFIG_SCHEMA
    variant: Literal[
        "b0_legacy_v1",
        "b1_multiscale_v1",
        "b2_ordered_multiscale_v1",
        "b_light_ordered_v1",
    ] = "b2_ordered_multiscale_v1"
    base_channels: int = 64
    state_channels: int = 32
    source_channels: int = 3
    residual_blocks: int = 2
    light_residual_blocks: int = 1
    normalization: Literal["none"] = "none"

    def __post_init__(self) -> None:
        if self.schema_version != STATIC_CONFIG_SCHEMA:
            raise ValueError(f"schema_version must be {STATIC_CONFIG_SCHEMA!r}")
        variants = {"b0_legacy_v1", "b1_multiscale_v1", "b2_ordered_multiscale_v1", "b_light_ordered_v1"}
        variant = self.variant
        if variant not in variants:
            raise ValueError(f"unknown static synthesis variant: {variant!r}")
        if self.base_channels != 64 or self.state_channels != 32 or self.source_channels != 3:
            raise ValueError("PFGR static widths are locked to base=64, state=32, source=3")
        _positive_int("residual_blocks", self.residual_blocks)
        _positive_int("light_residual_blocks", self.light_residual_blocks)
        if self.light_residual_blocks != 1:
            raise ValueError("B-light residual depth is locked to exactly one block")
        if self.residual_blocks != 2:
            raise ValueError("B1/B2 residual depth is locked to exactly two blocks")
        if self.normalization != "none":
            raise ValueError("PFGR static heads do not add hidden normalization")

    @property
    def architecture_version(self) -> str:
        return self.variant

    def as_dict(self) -> dict[str, Any]:
        return _canonical(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "StaticSynthesisConfig":
        if not isinstance(values, Mapping):
            raise TypeError("static config must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown static config keys: {sorted(unknown)}")
        return cls(**dict(values))


@dataclass(frozen=True)
class PFGRPolicyConfig:
    """Target-free bounded route policy declaration."""

    schema_version: str = POLICY_CONFIG_SCHEMA
    budgets: tuple[int, ...] = (0, 1, 2, 4)
    revisit: Literal["allow"] = "allow"
    tie_break: Literal["lowest_point_id"] = "lowest_point_id"
    gain_units: Literal["raw_signed_loss"] = "raw_signed_loss"
    candidate_count: int = 2048
    quality_margin: float = 0.0
    compute_cost: float = 0.0
    mode: Literal[
        "adaptive",
        "forced_diagnostic",
        "random",
        "fixed_learned",
        "parallel_topk",
        "static",
        "noop",
    ] = "adaptive"

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_CONFIG_SCHEMA:
            raise ValueError(f"schema_version must be {POLICY_CONFIG_SCHEMA!r}")
        budgets = tuple(self.budgets)
        if budgets != (0, 1, 2, 4):
            raise ValueError("PFGR budgets are locked to (0, 1, 2, 4)")
        if self.revisit != "allow":
            raise ValueError("PFGR MAIN revisit policy is locked to 'allow'")
        if self.tie_break != "lowest_point_id":
            raise ValueError("PFGR tie rule is locked to lowest_point_id")
        if self.gain_units != "raw_signed_loss":
            raise ValueError("PFGR gain units are locked to signed raw loss")
        _positive_int("candidate_count", self.candidate_count)
        if self.candidate_count != 2048:
            raise ValueError("PFGR candidate_count is locked to 2048")
        if not math.isfinite(float(self.quality_margin)) or float(self.quality_margin) < 0.0:
            raise ValueError("quality_margin must be finite and nonnegative")
        if not math.isfinite(float(self.compute_cost)) or float(self.compute_cost) < 0.0:
            raise ValueError("compute_cost must be finite and nonnegative")
        if self.mode not in {
            "adaptive",
            "forced_diagnostic",
            "random",
            "fixed_learned",
            "parallel_topk",
            "static",
            "noop",
        }:
            raise ValueError(f"unknown policy mode: {self.mode!r}")

    def as_dict(self) -> dict[str, Any]:
        return _canonical(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "PFGRPolicyConfig":
        if not isinstance(values, Mapping):
            raise TypeError("policy config must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown policy config keys: {sorted(unknown)}")
        if "budgets" in values:
            values = dict(values)
            values["budgets"] = tuple(values["budgets"])
        return cls(**dict(values))


@dataclass(frozen=True)
class ValueModelConfig:
    """Signed V descriptor variants fitted from one shared bank."""

    schema_version: str = VALUE_CONFIG_SCHEMA
    input_variants: tuple[int, ...] = (126, 222, 270, 366)
    hidden_channels: tuple[int, int] = (128, 64)
    loss: Literal["mse"] = "mse"
    gain_scale_quantile: float = 0.90
    gain_scale_floor: float = 1e-8
    storage_dtype: Literal["float32"] = "float32"
    descriptor_schema: str = "pfgr-lite-descriptors-v1"

    def __post_init__(self) -> None:
        if self.schema_version != VALUE_CONFIG_SCHEMA:
            raise ValueError(f"schema_version must be {VALUE_CONFIG_SCHEMA!r}")
        if tuple(self.input_variants) != (126, 222, 270, 366):
            raise ValueError("V descriptor variants are locked to 126/222/270/366")
        if tuple(self.hidden_channels) != (128, 64):
            raise ValueError("V hidden widths are locked to 128 and 64")
        if self.loss != "mse":
            raise ValueError("MAIN V loss is signed MSE")
        if not 0.0 < float(self.gain_scale_quantile) < 1.0:
            raise ValueError("gain_scale_quantile must lie strictly between zero and one")
        if not math.isfinite(float(self.gain_scale_floor)) or float(self.gain_scale_floor) <= 0.0:
            raise ValueError("gain_scale_floor must be positive and finite")
        if self.storage_dtype != "float32":
            raise ValueError("PFGR bank storage is locked to FP32")
        if self.descriptor_schema != "pfgr-lite-descriptors-v1":
            raise ValueError("unknown descriptor schema")

    def as_dict(self) -> dict[str, Any]:
        return _canonical(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ValueModelConfig":
        if not isinstance(values, Mapping):
            raise TypeError("value config must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown value config keys: {sorted(unknown)}")
        values = dict(values)
        for key in ("input_variants", "hidden_channels"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


@dataclass(frozen=True)
class EffectTeacherConfig:
    """Target-after-trace teacher/effect measurement declaration."""

    schema_version: str = TEACHER_CONFIG_SCHEMA
    mode: Literal["exact_footprint", "iid_fixed_q"] = "iid_fixed_q"
    rho: Literal["charbonnier"] = "charbonnier"
    epsilon: float = 1e-3
    q_draws: int = 1024
    mask_definition: str = "observation_derived_binary"
    label_definition: str = "signed_conditional_mean_masked_global_charbonnier"

    def __post_init__(self) -> None:
        if self.schema_version != TEACHER_CONFIG_SCHEMA:
            raise ValueError(f"schema_version must be {TEACHER_CONFIG_SCHEMA!r}")
        if self.mode not in ("exact_footprint", "iid_fixed_q"):
            raise ValueError("teacher mode must be exact_footprint or iid_fixed_q")
        if self.rho != "charbonnier":
            raise ValueError("PFGR teacher rho is locked to Charbonnier")
        _finite_positive("epsilon", self.epsilon)
        _nonnegative_int("q_draws", self.q_draws)
        if self.mode == "iid_fixed_q" and self.q_draws < 2:
            raise ValueError("iid_fixed_q requires q_draws >= 2 for uncertainty metadata")
        if not isinstance(self.mask_definition, str) or not self.mask_definition:
            raise ValueError("mask_definition must be nonempty")
        if self.label_definition != "signed_conditional_mean_masked_global_charbonnier":
            raise ValueError("unknown PFGR label definition")

    def as_dict(self) -> dict[str, Any]:
        return _canonical(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "EffectTeacherConfig":
        if not isinstance(values, Mapping):
            raise TypeError("teacher config must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown teacher config keys: {sorted(unknown)}")
        return cls(**dict(values))


@dataclass(frozen=True)
class PFGRLiteConfig:
    """Aggregate PFGR-Lite protocol configuration (``pfgr-lite-config-v1``)."""

    schema_version: str = PFGR_CONFIG_SCHEMA
    static: StaticSynthesisConfig = field(default_factory=StaticSynthesisConfig)
    policy: PFGRPolicyConfig = field(default_factory=PFGRPolicyConfig)
    value: ValueModelConfig = field(default_factory=ValueModelConfig)
    teacher: EffectTeacherConfig = field(default_factory=EffectTeacherConfig)
    numeric_mode: Literal["fp32", "fp64_test"] = "fp32"
    candidate_count: int = 2048
    state_channels: int = 32
    correction_channels: int = 96
    write_scale: float = 0.1
    support_radius_mm: float = 4.0
    max_displacement_mm: float = 2.0
    build_chunk_size: int = 1024
    decode_chunk_size: int = 1024
    device: str | None = None
    num_points: int = 2048
    # Reduced point counts are available only for explicit CPU engineering
    # fixtures.  Production manifests retain N=2048 and cannot be silently
    # relabelled as production when this capability is enabled.
    engineering_only: bool = False
    observation_normalization: str = "pfgr-observation-normalization-v1"
    point_guided: Any | None = None

    def __post_init__(self) -> None:
        if self.schema_version != PFGR_CONFIG_SCHEMA:
            raise ValueError(f"schema_version must be {PFGR_CONFIG_SCHEMA!r}")
        if not isinstance(self.static, StaticSynthesisConfig):
            raise TypeError("static must be StaticSynthesisConfig")
        if not isinstance(self.policy, PFGRPolicyConfig):
            raise TypeError("policy must be PFGRPolicyConfig")
        if not isinstance(self.value, ValueModelConfig):
            raise TypeError("value must be ValueModelConfig")
        if not isinstance(self.teacher, EffectTeacherConfig):
            raise TypeError("teacher must be EffectTeacherConfig")
        if self.numeric_mode not in ("fp32", "fp64_test"):
            raise ValueError("numeric_mode must be fp32 or fp64_test")
        if self.numeric_mode == "fp32" and self.device is not None and not isinstance(self.device, str):
            raise TypeError("device must be a string or None")
        if self.candidate_count != 2048 or self.policy.candidate_count != 2048:
            raise ValueError("PFGR candidate_count is locked to 2048")
        if self.state_channels != 32 or self.static.state_channels != 32:
            raise ValueError("PFGR state_channels is locked to 32")
        if self.correction_channels != 96:
            raise ValueError("PFGR correction_channels is locked to 96")
        if not math.isclose(float(self.write_scale), 0.1, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("PFGR write_scale is locked to 0.1")
        if not math.isclose(float(self.support_radius_mm), 4.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("support_radius_mm is locked to 4.0 mm")
        if not math.isclose(float(self.max_displacement_mm), 2.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("max_displacement_mm is locked to 2.0 mm")
        _positive_int("build_chunk_size", self.build_chunk_size)
        _positive_int("decode_chunk_size", self.decode_chunk_size)
        _positive_int("num_points", self.num_points)
        if self.num_points > 2048:
            raise ValueError("PFGR num_points cannot exceed production N=2048")
        if self.num_points != 2048 and not self.engineering_only:
            raise ValueError("num_points below 2048 requires explicit engineering_only=True")
        if not isinstance(self.engineering_only, bool):
            raise TypeError("engineering_only must be bool")
        if not isinstance(self.observation_normalization, str) or not self.observation_normalization:
            raise ValueError("observation_normalization must be a nonempty declared policy")

    def as_dict(self) -> dict[str, Any]:
        return _canonical(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "PFGRLiteConfig":
        if not isinstance(values, Mapping):
            raise TypeError("PFGR-Lite config must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown PFGR config keys: {sorted(unknown)}")
        values = dict(values)
        nested = {
            "static": (StaticSynthesisConfig, StaticSynthesisConfig.from_dict),
            "policy": (PFGRPolicyConfig, PFGRPolicyConfig.from_dict),
            "value": (ValueModelConfig, ValueModelConfig.from_dict),
            "teacher": (EffectTeacherConfig, EffectTeacherConfig.from_dict),
        }
        for key, (kind, parser) in nested.items():
            if key in values and not isinstance(values[key], kind):
                values[key] = parser(values[key])
        return cls(**values)


__all__ = [
    "EffectTeacherConfig",
    "PFGRLiteConfig",
    "PFGRPolicyConfig",
    "StaticSynthesisConfig",
    "ValueModelConfig",
    "PFGR_CONFIG_SCHEMA",
    "STATIC_CONFIG_SCHEMA",
    "POLICY_CONFIG_SCHEMA",
    "VALUE_CONFIG_SCHEMA",
    "TEACHER_CONFIG_SCHEMA",
]
