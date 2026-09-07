"""Bounded PFGR-Lite target-free evaluation service.

This module is intentionally a service, not a second model or data adapter.
It accepts W3b's :class:`StageInputs`, calls the single W4 effective-policy
loader and route, decodes only the final state through W2's canonical lattice,
and performs the target join only after that target-free trace is sealed.
"""

from __future__ import annotations

import inspect
import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from .config import EffectTeacherConfig, PFGRLiteConfig

# ``StageInputs`` is imported lazily below.  W5 must remain import-safe when
# only contracts/configuration are loaded in a target-free process.
from .metrics import (
    METRICS_SCHEMA,
    action_metric_row,
    aggregate_action_metrics,
    aggregate_subject_metrics,
    paired_subject_metrics,
    scientific_decision,
    stopping_diagnostics,
    write_json,
)
from .provenance import canonical_digest, tensor_digest

EXPERIMENT_OPTIONS_SCHEMA = "pfgr-lite-experiment-options-v1"
SCENARIOS = ("static", "noop", "random", "fixed_learned", "adaptive", "parallel_topk")
POLICY_CAPABILITIES = {
    "static": "static",
    "noop": "static",
    "random": "forced_diagnostic",
    "fixed_learned": "forced_diagnostic",
    "adaptive": "adaptive",
    "parallel_topk": "forced_diagnostic",
}


@contextmanager
def _service_execution(*objects: object | None):
    """Run scientific services detached while restoring module train state."""

    modules: list[torch.nn.Module] = []
    seen: set[int] = set()
    for candidate in objects:
        if isinstance(candidate, torch.nn.Module):
            for module in candidate.modules():
                if id(module) not in seen:
                    modules.append(module)
                    seen.add(id(module))
    previous = {id(module): bool(module.training) for module in modules}
    for module in modules:
        module.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        # Assign the flag directly so restoring a parent does not recursively
        # overwrite a child that was intentionally left in eval/frozen mode.
        for module in modules:
            module.training = previous[id(module)]


def _positive_int(name: str, value: object, *, allow_zero: bool = False, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer")
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
    if isinstance(value, Tensor):
        return {"dtype": str(value.dtype), "shape": tuple(value.shape), "sha256": tensor_digest(value.detach(), name="artifact")}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _jsonable(value.as_dict())
    if hasattr(value, "to_metadata") and callable(value.to_metadata):
        return _jsonable(value.to_metadata())
    return str(value)


def _record_dict(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    # Privileged teacher wrappers retain the original action/state objects for
    # identity guards; serialize only their detached label and digest fields,
    # never the plane/target tensors embedded in the wrapper.
    if hasattr(value, "label") and hasattr(value, "scope"):
        label = value.label
        row = _record_dict(label)
        row.update(
            {
                "diagnostic_scope": getattr(value, "scope", None),
                "diagnostic_privileged": getattr(value, "privileged", None),
                "diagnostic_schema_version": getattr(value, "schema_version", None),
                "state_version": getattr(value, "state_version", row.get("state_version")),
                "state_digest": getattr(value, "state_digest", None),
                "proposal_digest": getattr(value, "proposal_digest", None),
                "target_context_digest": getattr(value, "target_context_digest", None),
                "action_digest": getattr(getattr(value, "action", None), "action_digest", None),
            }
        )
        return row
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return dict(value.as_dict())
    names = (
        "action_id",
        "context_id",
        "state_version",
        "raw_gain",
        "benefit",
        "harm",
        "mask_count",
        "role",
        "q_draws",
        "seed",
        "variance",
        "standard_error",
        "footprint_voxels",
        "valid_masked_contributions",
        "sampler_law",
        "label_definition",
    )
    return {name: getattr(value, name) for name in names if hasattr(value, name)}


def _invoke(callable_obj: Any, *positional: Any, **keyword: Any) -> Any:
    """Pass only keywords accepted by a W3/W4 seam.

    The helper is kept local to W5; it does not create a competing callback
    protocol.  A callable's own TypeError remains visible when its signature
    is inspectable.
    """

    if not callable(callable_obj):
        raise TypeError("service seam must be callable")
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return callable_obj(*positional, **keyword)
    parameters = signature.parameters
    positional_names = [
        name
        for name, parameter in parameters.items()
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    count = min(len(positional), len(positional_names))
    consumed = set(positional_names[:count])
    accepts_var_kw = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    if accepts_var_kw:
        resolved = {name: value for name, value in keyword.items() if name not in consumed}
    else:
        resolved = {
            name: value
            for name, value in keyword.items()
            if name in parameters and name not in consumed and parameters[name].kind != inspect.Parameter.POSITIONAL_ONLY
        }
    return callable_obj(*positional[:count], **resolved)


@dataclass(frozen=True)
class ExperimentOptions:
    """Strict, bounded options for one deployment evaluation scenario."""

    scenario: Literal["static", "noop", "random", "fixed_learned", "adaptive", "parallel_topk"] = "static"
    budget: int = 0
    max_subjects: int = 1
    seed: int = 20260907
    split_role: str = "validation"
    candidate_chunk_size: int = 1
    decode_chunk_size: int = 1024
    teacher_mode: Literal["exact_footprint", "iid_fixed_q"] = "exact_footprint"
    query_count: int = 1024
    data_range: float = 1.0
    charbonnier_epsilon: float = 1e-3
    ssim_window: int = 11
    numerical_tolerance: float = 1e-10
    practical_margin: float = 0.0
    minimum_subjects: int = 32
    local_footprint_audit: bool = False
    engineering_only: bool = False
    schema_version: str = EXPERIMENT_OPTIONS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != EXPERIMENT_OPTIONS_SCHEMA:
            raise ValueError("unknown ExperimentOptions schema")
        if self.scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}")
        if self.budget not in (0, 1, 2, 4):
            raise ValueError("budget must be one of 0, 1, 2, or 4")
        _positive_int("max_subjects", self.max_subjects, maximum=1_000_000)
        _positive_int("seed", self.seed, allow_zero=True, maximum=2**63 - 1)
        if not isinstance(self.split_role, str) or not self.split_role.strip():
            raise ValueError("split_role must be a nonempty string")
        _positive_int("candidate_chunk_size", self.candidate_chunk_size, maximum=1_000_000)
        _positive_int("decode_chunk_size", self.decode_chunk_size, maximum=1_000_000)
        if self.teacher_mode not in ("exact_footprint", "iid_fixed_q"):
            raise ValueError("teacher_mode must be exact_footprint or iid_fixed_q")
        _positive_int("query_count", self.query_count, maximum=10_000_000)
        if self.teacher_mode == "iid_fixed_q" and self.query_count < 2:
            raise ValueError("iid_fixed_q evaluation requires query_count >= 2")
        _finite_nonnegative("data_range", self.data_range)
        if float(self.data_range) <= 0.0:
            raise ValueError("data_range must be positive")
        _finite_nonnegative("charbonnier_epsilon", self.charbonnier_epsilon)
        if float(self.charbonnier_epsilon) <= 0.0:
            raise ValueError("charbonnier_epsilon must be positive")
        _positive_int("ssim_window", self.ssim_window, maximum=101)
        if self.ssim_window % 2 == 0:
            raise ValueError("ssim_window must be odd")
        _finite_nonnegative("numerical_tolerance", self.numerical_tolerance)
        _finite_nonnegative("practical_margin", self.practical_margin)
        _positive_int("minimum_subjects", self.minimum_subjects, maximum=1_000_000)
        if not isinstance(self.local_footprint_audit, bool) or not isinstance(
            self.engineering_only, bool
        ):
            raise TypeError("local_footprint_audit and engineering_only must be bool")

    def as_dict(self) -> dict[str, Any]:
        return {field.name: _jsonable(getattr(self, field.name)) for field in fields(self)}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ExperimentOptions:
        if not isinstance(values, Mapping):
            raise TypeError("ExperimentOptions must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown ExperimentOptions keys: {sorted(unknown)}")
        return cls(**dict(values))


class _LatticeFactory:
    """Identity-preserving factory used by the model's canonical decoder."""

    def __init__(self, lattice: object) -> None:
        self.lattice = lattice

    def build(self, **kwargs: Any) -> object:
        for name in ("output_geometry", "feature_geometry", "query_dtype"):
            if name in kwargs and kwargs[name] != getattr(self.lattice, name, None):
                raise ValueError(f"canonical lattice factory {name} mismatch")
        return self.lattice


def _local_footprint_audit_rows(
    actions: Sequence[object],
    states_by_version: Mapping[int, object],
    target_context: object,
    lattice: object,
    decoder: object,
    *,
    chunk_size: int,
    epsilon: float,
    numerical_tolerance: float,
) -> list[dict[str, Any]]:
    """Measure the legacy physical-sphere control beside complete support.

    The control is intentionally diagnostic-only: local rows use the exact
    4-mm physical sphere helper and a local masked denominator, while global
    rows use the canonical compact-writer footprint and the fixed complete
    subject mask denominator.  Neither row can alter MAIN labels or routing.
    """

    from ..reward_supervision import build_local_support_samples
    from ..sampling import ras_mm_to_voxel_dhw
    from .sparse_write import build_footprint, query_write_delta
    from .teacher import ValidatedTargetContext
    from .types import ActionProposal

    if not isinstance(target_context, ValidatedTargetContext):
        return [
            {
                "audit_scope": "local_physical_sphere_vs_complete_footprint",
                "available": False,
                "reason": "validated_target_context_unavailable",
            }
        ]
    target_context.validate_integrity()
    output_geometry = lattice.output_geometry
    rows: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, ActionProposal):
            rows.append(
                {
                    "audit_scope": "local_physical_sphere_vs_complete_footprint",
                    "action_id": str(getattr(action, "action_id", "unknown")),
                    "available": False,
                    "reason": "typed_action_required",
                }
            )
            continue
        state = states_by_version.get(int(action.state_version))
        if state is None:
            rows.append(
                {
                    "audit_scope": "local_physical_sphere_vs_complete_footprint",
                    "action_id": action.action_id,
                    "available": False,
                    "reason": "state_identity_unavailable",
                }
            )
            continue
        state.validate_integrity()
        footprint = build_footprint(lattice, action, chunk_size=chunk_size)
        local_samples = build_local_support_samples(
            action.point_ras_mm.reshape(1, 3), output_geometry
        )
        local_valid = local_samples.valid_mask[..., 0]
        local_dhw = ras_mm_to_voxel_dhw(
            local_samples.points_ras_mm, output_geometry
        ).round().to(dtype=torch.long)
        local_ids = local_dhw[0][local_valid[0]]
        global_ids = footprint.voxel_ids_dhw.to(device=state.planes.xy.device)
        if local_ids.numel() == 0:
            rows.append(
                {
                    "audit_scope": "local_physical_sphere_vs_complete_footprint",
                    "action_id": action.action_id,
                    "state_version": action.state_version,
                    "available": False,
                    "reason": "empty_local_physical_sphere",
                    "local_voxel_count": 0,
                    "global_support_voxel_count": footprint.union_size,
                }
            )
            continue
        local_ids = local_ids.to(device=state.planes.xy.device)
        global_ids = global_ids.to(device=state.planes.xy.device)
        before_local = lattice.query(state.planes, local_ids, chunk_size=chunk_size)
        before_global = lattice.query(state.planes, global_ids, chunk_size=chunk_size)
        delta_local = query_write_delta(
            lattice, footprint, local_ids, action.delta, chunk_size=chunk_size
        )
        delta_global = query_write_delta(
            lattice, footprint, global_ids, action.delta, chunk_size=chunk_size
        )
        before_local_prediction = _decode_prediction(decoder, before_local)
        after_local_prediction = _decode_prediction(
            decoder, before_local + delta_local
        )
        before_global_prediction = _decode_prediction(decoder, before_global)
        after_global_prediction = _decode_prediction(
            decoder, before_global + delta_global
        )
        target_local = target_context.gather_target(
            local_ids,
            device=before_local_prediction.device,
            dtype=before_local_prediction.dtype,
        )
        mask_local = target_context.gather_mask(
            local_ids, device=before_local_prediction.device
        )
        target_global = target_context.gather_target(
            global_ids,
            device=before_global_prediction.device,
            dtype=before_global_prediction.dtype,
        )
        mask_global = target_context.gather_mask(
            global_ids, device=before_global_prediction.device
        )
        local_difference = torch.sqrt(
            (before_local_prediction - target_local).square() + epsilon * epsilon
        ) - torch.sqrt(
            (after_local_prediction - target_local).square() + epsilon * epsilon
        )
        global_difference = torch.sqrt(
            (before_global_prediction - target_global).square()
            + epsilon * epsilon
        ) - torch.sqrt(
            (after_global_prediction - target_global).square()
            + epsilon * epsilon
        )
        local_mask_count = int(mask_local.sum().item())
        if local_mask_count > 0:
            local_gain = float(
                (
                    local_difference
                    * mask_local.to(dtype=local_difference.dtype)
                )
                .to(dtype=torch.float64)
                .sum()
                .item()
                / local_mask_count
            )
        else:
            local_gain = None
        global_gain = float(
            (
                global_difference
                * mask_global.to(dtype=global_difference.dtype)
            )
            .to(dtype=torch.float64)
            .sum()
            .item()
            / float(target_context.mask_count)
        )
        local_sign = (
            "positive"
            if local_gain is not None and local_gain > numerical_tolerance
            else "negative"
            if local_gain is not None and local_gain < -numerical_tolerance
            else "neutral"
        )
        global_sign = (
            "positive"
            if global_gain > numerical_tolerance
            else "negative"
            if global_gain < -numerical_tolerance
            else "neutral"
        )
        rows.append(
            {
                "audit_scope": "local_physical_sphere_vs_complete_footprint",
                "available": True,
                "action_id": action.action_id,
                "state_version": action.state_version,
                "state_digest": action.state_digest,
                "local_gain": local_gain,
                "global_complete_footprint_gain": global_gain,
                "local_denominator": local_mask_count,
                "global_denominator": target_context.mask_count,
                "local_voxel_count": int(local_ids.shape[0]),
                "global_support_voxel_count": footprint.union_size,
                "local_sign": local_sign,
                "global_sign": global_sign,
                "sign_disagreement": (
                    local_gain is not None
                    and local_sign != "neutral"
                    and global_sign != "neutral"
                    and local_sign != global_sign
                ),
                "epsilon": float(epsilon),
                "global_scope": "complete_writer_footprint_fixed_subject_mask_v1",
                "local_scope": "physical_4mm_sphere_local_mask_mean_v1",
            }
        )
    return rows


def _decode_prediction(decoder: object, query: Tensor) -> Tensor:
    module = getattr(decoder, "mlp", decoder)
    if not callable(module):
        raise TypeError("decoder must expose a callable mlp")
    output = module(query)
    if not isinstance(output, Tensor):
        raise TypeError("decoder output must be a tensor")
    return output.reshape(-1)


def _prepare_output(output_dir: str | Path, filenames: Sequence[str]) -> Path:
    destination = Path(output_dir)
    if destination.exists():
        if not destination.is_dir():
            raise ValueError("output_dir must be a directory")
        if any(destination.iterdir()):
            raise FileExistsError(f"output_dir must be empty and exclusive: {destination}")
    else:
        destination.mkdir(parents=True, exist_ok=False)
    for name in filenames:
        if (destination / name).exists():
            raise FileExistsError(f"refusing to overwrite artifact {destination / name}")
    return destination


def _resolve_config(inputs: Any) -> PFGRLiteConfig | None:
    execution = getattr(inputs, "execution", None)
    if execution is not None:
        return execution.config
    config = getattr(inputs, "config", None)
    return config if isinstance(config, PFGRLiteConfig) else None


def _value_services(inputs: Any) -> tuple[object | None, object | None, float | None, str | None, Mapping[str, object] | None, str | None]:
    """Resolve the one W3 value service and its immutable identities.

    W3b may expose a fitted value result or its decomposed model/identity
    fields through the explicit ``StageInputs.metadata`` envelope.  We only
    accept those documented fields; missing production identities are left
    missing so W4's loader can fail closed instead of minting a policy.
    """

    metadata = getattr(inputs, "metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    fit = metadata.get("value_fit_result", metadata.get("value_fit"))
    value_model = metadata.get("value_model")
    identity = metadata.get("value_fit_identity")
    scale_value = metadata.get("gain_scale")
    scale_provenance = metadata.get("gain_scale_provenance")
    if fit is not None:
        if value_model is None:
            value_model = getattr(fit, "model", getattr(fit, "value_net", None))
        if identity is None:
            identity = getattr(fit, "identity", getattr(fit, "fit_identity", None))
        if scale_value is None:
            scale_value = getattr(fit, "gain_scale", None)
    if identity is not None and isinstance(identity, Mapping):
        from .provenance import ValueFitIdentity

        allowed = {field.name for field in fields(ValueFitIdentity)}
        unknown = set(identity) - allowed
        if unknown:
            raise ValueError(f"value_fit_identity contains unknown fields: {sorted(unknown)}")
        identity = ValueFitIdentity(**dict(identity))
    if scale_value is not None and not isinstance(scale_value, (int, float)):
        scale_provenance = scale_provenance or (scale_value.as_dict() if hasattr(scale_value, "as_dict") else None)
        scale_value = getattr(scale_value, "scale", getattr(scale_value, "value", None))
    scale = None if scale_value is None else float(scale_value)
    if scale_provenance is not None and not isinstance(scale_provenance, Mapping):
        raise TypeError("gain_scale_provenance must be a mapping")
    role_manifest = getattr(inputs, "role_manifest", None)
    role_hash = metadata.get("role_manifest_hash")
    if role_hash is None and role_manifest is not None:
        role_hash = getattr(role_manifest, "digest", None)
    return value_model, identity, scale, (getattr(identity, "gain_scale_hash", None) if identity is not None else None), scale_provenance, role_hash


def _sample_id(sample: object, index: int) -> str:
    value = getattr(sample, "subject_id", None)
    if isinstance(value, str) and value:
        return value
    return f"sample-{index:04d}"


def _context_for_sample(
    inputs: Any,
    sample: object,
    *,
    pipeline_counters: dict[str, int] | None = None,
) -> object | None:
    metadata = getattr(inputs, "metadata", {}) or {}
    contexts = metadata.get("contexts") if isinstance(metadata, Mapping) else None
    if isinstance(contexts, Mapping):
        subject = getattr(sample, "subject_id", None)
        context = contexts.get(subject, contexts.get(str(subject)))
        if context is not None:
            return context
    builder = metadata.get("context_builder") if isinstance(metadata, Mapping) else None
    if callable(builder):
        return _invoke(builder, sample, sample=sample, inputs=inputs)
    model = getattr(inputs, "model", None)
    observations = getattr(sample, "observations", getattr(sample, "inputs", None))
    if model is not None and observations is not None and hasattr(model, "encode_observations"):
        if not isinstance(observations, Tensor):
            raise TypeError("target-free sample observations must be a tensor")
        batched = observations.unsqueeze(0) if observations.ndim == 4 else observations
        mask = getattr(sample, "brain_mask", getattr(sample, "mask", None))
        geometry = getattr(sample, "geometry", None)
        # Production StageInputs use the concrete PFGR model contract.  The
        # model performs its own shape/geometry/provenance validation; do not
        # signature-filter this call or silently select an alternate encoder.
        if pipeline_counters is not None:
            pipeline_counters["observation_encode_calls"] = pipeline_counters.get(
                "observation_encode_calls", 0
            ) + 1
        return model.encode_observations(batched, mask, geometry)
    return None


def _build_lattice(inputs: Any, context: object | None, model: object | None, config: PFGRLiteConfig | None) -> object | None:
    metadata = getattr(inputs, "metadata", {}) or {}
    if isinstance(metadata, Mapping) and metadata.get("lattice") is not None:
        lattice = metadata["lattice"]
    elif context is not None and hasattr(context, "geometry") and hasattr(context, "feature_geometry"):
        from .footprint import PFGRQueryLattice

        planes = getattr(context, "initial_planes", None)
        dtype = getattr(getattr(planes, "xy", None), "dtype", torch.float32)
        build_chunk = config.build_chunk_size if config is not None else 1024
        lattice = PFGRQueryLattice.build(
            context.geometry,
            context.feature_geometry,
            query_dtype=dtype,
            build_chunk_size=build_chunk,
        )
    else:
        lattice = None
    if lattice is not None and model is not None and hasattr(model, "set_query_lattice_factory"):
        model.set_query_lattice_factory(_LatticeFactory(lattice))
    return lattice


def _initialize_state(model: object | None, context: object | None) -> object | None:
    if model is None or context is None or not hasattr(model, "initialize_state"):
        return None
    return _invoke(model.initialize_state, context, context=context, role="deployment")


def _config_for_scenario(config: PFGRLiteConfig | None, options: ExperimentOptions) -> PFGRLiteConfig | None:
    if config is None:
        return config
    # ``noop`` is an experiment-side diagnostic alias for the locked static
    # policy.  It is never forwarded as a W4 policy mode (which would be an
    # invalid configuration and could accidentally create a new policy).
    resolved_mode = "static" if options.scenario == "noop" else options.scenario
    if config.policy.mode == resolved_mode:
        return config
    # This is a strict W4 policy-config declaration override; effective policy
    # construction remains solely in ``load_effective_policy``.
    return replace(config, policy=replace(config.policy, mode=resolved_mode))


def _load_policy(inputs: Any, context: object | None, options: ExperimentOptions, config: PFGRLiteConfig | None) -> object | None:
    metadata = getattr(inputs, "metadata", {}) or {}
    if isinstance(metadata, Mapping):
        provided = metadata.get("effective_policy")
        if provided is not None:
            from .policy import EffectivePolicy

            if not isinstance(provided, EffectivePolicy):
                raise TypeError("metadata['effective_policy'] must be the W4 EffectivePolicy object")
            return provided
    if context is None or not hasattr(context, "producer"):
        return None
    from .policy import load_effective_policy

    resolved = _config_for_scenario(config, options)
    if resolved is None:
        raise ValueError("StageInputs requires execution/config for W4 policy loading")
    calibration = metadata.get("calibration") if isinstance(metadata, Mapping) else None
    _value_model, identity, gain_scale, gain_scale_hash, gain_scale_provenance, role_manifest_hash = _value_services(inputs)
    # These are passed as the exact W3 identities, never reconstructed from a
    # bare checkpoint/Git SHA.  W4 performs the final completeness and
    # producer/role/calibration checks for learned and adaptive modes.
    return load_effective_policy(
        resolved,
        calibration,
        dependencies=context.producer,
        capability=POLICY_CAPABILITIES[options.scenario],
        budget=options.budget,
        candidate_chunk_size=options.candidate_chunk_size,
        random_seed=options.seed,
        value_fit_identity=identity,
        value_fit_identity_hash=None if identity is None else identity.digest,
        role_manifest_hash=role_manifest_hash,
        gain_scale=gain_scale,
        gain_scale_hash=gain_scale_hash,
        gain_scale_provenance=gain_scale_provenance,
    )


def _route_for_sample(
    inputs: Any,
    sample: object,
    context: object | None,
    options: ExperimentOptions,
    config: PFGRLiteConfig | None,
    lattice: object | None,
    policy: object | None,
) -> tuple[object, object | None, object | None]:
    model = getattr(inputs, "model", None)
    metadata = getattr(inputs, "metadata", {}) or {}
    query = getattr(inputs, "query", None)
    writer = getattr(inputs, "writer", None)
    legal_mask = metadata.get("legal_mask") if isinstance(metadata, Mapping) else None
    compound_writer = metadata.get("compound_writer") if isinstance(metadata, Mapping) else None
    value_model, _, _, _, _, _ = _value_services(inputs)
    route_builder = getattr(inputs, "route_builder", None)
    if route_builder is not None:
        if not options.engineering_only:
            raise ValueError("production evaluation requires the concrete W4 run_pfgr_inference route; route_builder is engineering-only")
        result = _invoke(
            route_builder,
            sample,
            sample=sample,
            context=context,
            observation_context=context,
            model=model,
            effective_policy=policy,
            policy=policy,
            options=options,
            config=config,
            lattice=lattice,
            query=query,
            writer=writer,
            value_model=value_model,
        )
        return result, query, writer
    if model is None or context is None or policy is None:
        raise ValueError("evaluation requires model/context/policy or an explicit route_builder")
    from .inference import run_pfgr_inference
    from .sparse_write import (
        make_action_writer,
        make_point_query,
        make_support_legal_mask,
    )

    if lattice is None:
        raise ValueError("evaluation requires the canonical PFGR query lattice")
    query = query or make_point_query()
    writer = writer or make_action_writer(lattice)
    if legal_mask is None:
        legal_mask = make_support_legal_mask(lattice)
    if options.scenario == "parallel_topk" and compound_writer is None:
        from .sparse_write import make_compound_writer

        compound_writer = make_compound_writer(lattice)
    return (
        run_pfgr_inference(
            model,
            context,
            policy,
            query=query,
            writer=writer,
            value_model=value_model,
            legal_mask=legal_mask,
            compound_writer=compound_writer,
        ),
        query,
        writer,
    )


def _route_attr(route: object, name: str, default: Any = None) -> Any:
    if isinstance(route, Mapping):
        return route.get(name, default)
    return getattr(route, name, default)


def _counter_metadata(route: object) -> Any:
    counters = _route_attr(route, "counters", {})
    if hasattr(counters, "as_dict") and callable(counters.as_dict):
        return counters.as_dict()
    return counters


def _prediction_for(
    model: object | None,
    route: object,
    context: object | None,
    *,
    final: bool,
    options: ExperimentOptions,
    pipeline_counters: dict[str, int] | None = None,
) -> Tensor:
    name = "final_prediction" if final else "initial_prediction"
    provided = _route_attr(route, name)
    if provided is None:
        provided = _route_attr(route, "prediction" if final else "initial")
    if isinstance(provided, Tensor):
        return provided
    state = _route_attr(route, "final_state" if final else "initial_state")
    if state is None and not final:
        parallel = _route_attr(route, "parallel_trace")
        state = _route_attr(parallel, "initial_state") if parallel is not None else None
    if state is None and not final:
        states = _route_attr(route, "states", ())
        if not states:
            trace = _completed_trace(route)
            states = _route_attr(trace, "states", ()) if trace is not None else ()
        if states:
            state = states[0]
    if model is None or context is None or state is None or not hasattr(model, "decode_final"):
        raise ValueError(f"route must expose {name} or model.decode_final-compatible state/context")
    # Production model decoding is a concrete canonical-lattice call.  A
    # route-provided tensor remains an explicit engineering fixture override.
    if pipeline_counters is not None:
        key = "final_decode_calls" if final else "initial_decode_calls"
        pipeline_counters[key] = pipeline_counters.get(key, 0) + 1
    return model.decode_final(state, context, chunk_size=options.decode_chunk_size)


def _completed_trace(route: object) -> object | None:
    return _route_attr(route, "completed_trace")


def _target_join(inputs: Any, sample: object, context: object | None, route: object, prediction: Tensor, options: ExperimentOptions) -> object:
    provider = getattr(inputs, "target_provider", None)
    if provider is None:
        raise ValueError("evaluation requires a deferred target_provider; target reads are post-trace only")
    from .data import defer_supervision

    input_metadata = getattr(inputs, "metadata", {})
    counters = None
    if isinstance(input_metadata, Mapping):
        counters = input_metadata.get("data_counters", input_metadata.get("counters"))
    # A reduced-N model may be an engineering fixture while its actual
    # ObservationContext still supplies the production binding required by
    # W2.  Engineering-only target validation is therefore limited to
    # callback fixtures that have no authoritative ObservationContext.
    bound_context = context is not None and type(context).__name__ == "ObservationContext"
    # The service owns one logical target-read count.  Some engineering
    # providers (notably the CLI synthetic fixture) already increment the
    # shared counter themselves; pass no counter into the deferred callback,
    # then account exactly one read only when the provider did not.  This
    # avoids counting one physical callback twice while retaining the
    # deferred target boundary for production providers.
    target_reads_before = (
        int(counters.target_reads)
        if counters is not None and isinstance(getattr(counters, "target_reads", None), int)
        else None
    )
    callback = defer_supervision(
        sample,
        provider,
        counters=None,
        engineering_only=(options.engineering_only or bool(getattr(_resolve_config(inputs), "engineering_only", False))) and not bound_context,
    )
    trace = _completed_trace(route)
    joined = _invoke(
        callback,
        completed_context=context,
        prediction=prediction,
        trace=trace,
    )
    if target_reads_before is not None and counters is not None:
        target_reads_after = getattr(counters, "target_reads", None)
        # Provider-owned accounting is retained; otherwise this service is
        # the sole owner and records the successful deferred read.
        if isinstance(target_reads_after, int) and target_reads_after == target_reads_before:
            counters.target_reads = target_reads_before + 1
    return joined


def _target_parts(target_context: object) -> tuple[Tensor, Tensor | None]:
    target = getattr(target_context, "target", target_context)
    mask = getattr(target_context, "observation_mask", getattr(target_context, "target_mask", None))
    if not isinstance(target, Tensor):
        raise TypeError("deferred target context must expose a target tensor")
    return target, mask if isinstance(mask, Tensor) else None


def _selected_actions(route: object) -> tuple[object, ...]:
    trace = _completed_trace(route)
    if trace is None:
        parallel = _route_attr(route, "parallel_trace")
        if parallel is None:
            return ()
        proposals = _route_attr(parallel, "proposals")
        selected_ids = tuple(_route_attr(parallel, "selected_action_ids", ()))
        if proposals is None:
            return ()
        selected = []
        for index in range(proposals.point_ids.shape[1]):
            action = proposals.row(0, index)
            if action.action_id in selected_ids:
                selected.append(action)
        return tuple(selected)
    actions: list[object] = []
    proposals_history = tuple(_route_attr(trace, "proposals", ()))
    decisions = tuple(_route_attr(trace, "decisions", ()))
    for proposal, decision in zip(proposals_history, decisions):
        if _route_attr(decision, "stop_code") != "continue":
            continue
        selected = _route_attr(decision, "selected_point_id", -1)
        locations = (proposal.point_ids == int(selected)).nonzero(as_tuple=False)
        if locations.shape[0] == 1:
            actions.append(proposal.row(int(locations[0, 0]), int(locations[0, 1])))
    return tuple(actions)


def _measure_actions(inputs: Any, route: object, actions: Sequence[object], target_context: object, context: object | None, lattice: object | None, options: ExperimentOptions) -> list[object]:
    if not actions:
        return []
    effect = getattr(inputs, "effect_measure", None)
    if effect is not None:
        measured = _invoke(
            effect,
            route,
            actions,
            target_context,
            trace=_completed_trace(route),
            proposals=actions,
            target_context=target_context,
            observation_context=context,
            config=_resolve_config(inputs),
        )
        if measured is None:
            return []
        return list(measured) if isinstance(measured, Iterable) and not isinstance(measured, (str, bytes, Mapping)) else [measured]
    trace = _completed_trace(route)
    parallel = _route_attr(route, "parallel_trace")
    model = getattr(inputs, "model", None)
    decoder = _route_attr(route, "decoder") or getattr(model, "decoder", None)
    if decoder is None or lattice is None:
        raise ValueError("selected action measurement requires W2 trace/decoder/lattice or inputs.effect_measure")
    config = _resolve_config(inputs)
    base_teacher = getattr(config, "teacher", None) if config is not None else None
    if base_teacher is not None and not math.isclose(
        float(base_teacher.epsilon), float(options.charbonnier_epsilon), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "ExperimentOptions.charbonnier_epsilon must match configured teacher epsilon"
        )
    requested_q = max(2, options.query_count)
    if base_teacher is not None:
        teacher_config = replace(
            base_teacher,
            mode=options.teacher_mode,
            q_draws=requested_q if options.teacher_mode == "iid_fixed_q" else base_teacher.q_draws,
        )
    else:
        teacher_config = EffectTeacherConfig(
            mode=options.teacher_mode,
            q_draws=requested_q,
            epsilon=float(options.charbonnier_epsilon),
        )
    counters = _route_attr(route, "counters", None)
    if parallel is not None:
        from .teacher import measure_parallel_actions

        return list(
            measure_parallel_actions(
                parallel,
                actions,
                target_context,
                decoder,
                teacher_config,
                lattice=lattice,
                observation_context=context,
                chunk_size=options.decode_chunk_size,
                candidate_chunk_size=options.candidate_chunk_size,
                seed=options.seed,
                counters=counters,
            )
        )
    if trace is None:
        raise ValueError("selected action measurement requires a completed target-free trace")
    from .teacher import measure_actions

    return list(
        measure_actions(
            trace,
            actions,
            target_context,
            decoder,
            teacher_config,
            lattice=lattice,
            chunk_size=options.decode_chunk_size,
            candidate_chunk_size=options.candidate_chunk_size,
            seed=options.seed,
            observation_context=context,
            counters=counters,
        )
    )


def _policy_metadata(policy: object) -> Any:
    if hasattr(policy, "as_dict") and callable(policy.as_dict):
        return policy.as_dict()
    if isinstance(policy, Mapping):
        return dict(policy)
    return {"repr": repr(policy)}


def _action_rows(
    route: object,
    labels: Sequence[object],
    *,
    subject_id: str,
    scenario: str,
    budget: int,
    numerical_tolerance: float,
    practical_margin: float,
) -> list[dict[str, Any]]:
    decisions = tuple(_route_attr(route, "decisions", ()))
    rows: list[dict[str, Any]] = []
    parallel = _route_attr(route, "parallel_trace") is not None
    for index, label in enumerate(labels):
        row = _record_dict(label)
        decision = decisions[index] if index < len(decisions) else None
        state_version = (
            row.get("state_version", 0)
            if parallel
            else _route_attr(decision, "step", row.get("state_version", 0))
        )
        row.update(
            {
                "subject_id": subject_id,
                "scenario": scenario,
                "budget": budget,
                "selected": True,
                "state_version": state_version,
                "predicted_raw_gain": _route_attr(decision, "raw_value") if decision is not None else None,
                "stop_code": _route_attr(decision, "stop_code") if decision is not None else "continue",
            }
        )
        rows.append(
            action_metric_row(
                row,
                numerical_tolerance=numerical_tolerance,
                practical_margin=practical_margin,
                scope="selected_only",
                selected=True,
            )
        )
    return rows


def _parallel_interaction_row(
    route: object,
    paired_row: Mapping[str, Any],
    individual_rows: Sequence[Mapping[str, Any]],
    *,
    subject_id: str,
    scenario: str,
    budget: int,
    numerical_tolerance: float,
    practical_margin: float,
) -> dict[str, Any] | None:
    """Record measured compound-vs-independent interaction for parallel_topk.

    The joint gain is the already-measured initial-to-final loss reduction;
    individual gains come from labels evaluated against the same frozen
    initial state.  This row intentionally never treats gains as additive
    losses: ``interaction = joint - sum(individual)`` is a diagnostic only.
    """

    if scenario != "parallel_topk" or _route_attr(route, "parallel_trace") is None:
        return None
    improvement = paired_row.get("improvement", {})
    joint = improvement.get("masked_charbonnier") if isinstance(improvement, Mapping) else None
    gains = [float(row["true_gain"]) for row in individual_rows if row.get("true_gain") is not None]
    if joint is None or not math.isfinite(float(joint)) or not gains:
        return action_metric_row(
            {
                "subject_id": subject_id,
                "scenario": scenario,
                "budget": budget,
                "action_id": "parallel-interaction",
                "role": "parallel_interaction",
                "interaction_gain": None,
                "raw_gain": None,
                "individual_gain_sum": float(sum(gains)),
                "joint_gain": None,
                "interaction_definition": "joint_gain_minus_sum_individual_initial_state_v1",
            },
            numerical_tolerance=numerical_tolerance,
            practical_margin=practical_margin,
            scope="parallel_interaction",
            selected=False,
        )
    interaction = float(joint) - float(sum(gains))
    return action_metric_row(
        {
            "subject_id": subject_id,
            "scenario": scenario,
            "budget": budget,
            "action_id": "parallel-interaction",
            "role": "parallel_interaction",
            "raw_gain": interaction,
            "benefit": max(interaction, 0.0),
            "harm": max(-interaction, 0.0),
            "interaction_gain": interaction,
            "individual_gain_sum": float(sum(gains)),
            "joint_gain": float(joint),
            "interaction_definition": "joint_gain_minus_sum_individual_initial_state_v1",
        },
        numerical_tolerance=numerical_tolerance,
        practical_margin=practical_margin,
        scope="parallel_interaction",
        selected=False,
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(dict(row)), sort_keys=True) + "\n")


def _source_receipt(
    inputs: Any,
    options: ExperimentOptions,
    *,
    contexts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    config = _resolve_config(inputs)
    producer = getattr(inputs, "producer", None)
    if producer is None and getattr(inputs, "model", None) is not None:
        producer = getattr(inputs.model, "producer_dependencies", None)
    metadata = getattr(inputs, "metadata", {})
    provided = metadata.get("source_receipt", metadata.get("provenance", {})) if isinstance(metadata, Mapping) else {}
    role_manifest = getattr(inputs, "role_manifest", None)
    # A sealed role manifest is authoritative.  Metadata may fill a missing
    # identity, but a conflicting value must fail rather than laundering a
    # stale/checkpoint receipt into the service artifact.
    actual_baseline_split_hash = (
        getattr(role_manifest, "baseline_split_hash", None)
        if role_manifest is not None
        else None
    )
    actual_training_role_manifest_hash = (
        getattr(role_manifest, "digest", None) if role_manifest is not None else None
    )
    baseline_split_hash = actual_baseline_split_hash
    if baseline_split_hash is None and isinstance(metadata, Mapping):
        baseline_split_hash = metadata.get("baseline_split_hash")
    if baseline_split_hash is None and isinstance(provided, Mapping):
        baseline_split_hash = provided.get("baseline_split_hash")
    training_role_manifest_hash = actual_training_role_manifest_hash
    if training_role_manifest_hash is None and isinstance(metadata, Mapping):
        training_role_manifest_hash = metadata.get(
            "training_role_manifest_hash", metadata.get("role_manifest_hash")
        )
    if training_role_manifest_hash is None and isinstance(provided, Mapping):
        training_role_manifest_hash = provided.get(
            "training_role_manifest_hash", provided.get("role_manifest_hash")
        )
    context_producer_hashes = {
        str(item.get("producer_compatibility_hash"))
        for item in contexts
        if item.get("producer_compatibility_hash")
    }
    context_normalization_hashes = {
        str(item.get("normalization_hash"))
        for item in contexts
        if item.get("normalization_hash")
    }
    initialization_hashes = {
        str(item.get("initialization_hash"))
        for item in contexts
        if item.get("initialization_hash")
    }
    subject_initialization_hashes = {
        str(item.get("subject_id")): str(item.get("initialization_hash"))
        for item in contexts
        if item.get("subject_id") and item.get("initialization_hash")
    }
    receipt = {
        "schema_version": EXPERIMENT_OPTIONS_SCHEMA,
        "options": options.as_dict(),
        "config": _jsonable(config),
        "config_hash": canonical_digest(_jsonable(config), prefix="pfgr-lite-experiment-config-v1|"),
        "producer_compatibility_hash": (
            getattr(producer, "compatibility_hash", getattr(producer, "digest", None))
            or (next(iter(context_producer_hashes)) if len(context_producer_hashes) == 1 else None)
        ),
        "normalization_hash": (
            next(iter(context_normalization_hashes))
            if len(context_normalization_hashes) == 1
            else None
        ),
        "initialization_hash": (
            next(iter(initialization_hashes))
            if len(initialization_hashes) == 1
            else None
        ),
        "subject_initialization_hashes": subject_initialization_hashes,
        "baseline_split_hash": baseline_split_hash,
        "training_role_manifest_hash": training_role_manifest_hash,
        "split_role": options.split_role,
        "mask_definition": "observation_derived_binary",
        "label_definition": "masked_charbonnier_global_v1",
        "loss_definition": "masked_charbonnier_global_v1",
        "data_range": options.data_range,
        "engineering_only": bool(options.engineering_only or getattr(config, "engineering_only", False)),
        "subject_contexts": [_jsonable(dict(item)) for item in contexts],
    }
    for key in ("producer_compatibility_hash", "normalization_hash"):
        context_values = {
            str(item[key])
            for item in contexts
            if item.get(key) is not None and str(item.get(key)).strip()
        }
        if len(context_values) > 1:
            raise ValueError(f"subject context {key!r} identities disagree")
        if context_values and receipt.get(key) is not None and str(receipt[key]) not in context_values:
            raise ValueError(f"source receipt {key!r} conflicts with subject context")
    if len(initialization_hashes) == 1:
        only_initialization = next(iter(initialization_hashes))
        if receipt.get("initialization_hash") is not None and str(receipt["initialization_hash"]) != only_initialization:
            raise ValueError("source receipt 'initialization_hash' conflicts with subject context")
    if isinstance(provided, Mapping):
        # Preserve only explicit W3 provenance identities; never invent a
        # source/checkpoint hash from a bare Git SHA or local path.
        for key in (
            "source_hash",
            "dirty_hash",
            "checkpoint_hash",
            "baseline_split_hash",
            "training_role_manifest_hash",
            "split_role",
            "split_role_hash",
            "normalization_hash",
            "producer_compatibility_hash",
            "initialization_hash",
            "subject_initialization_hashes",
            "mask_definition",
            "label_definition",
            "loss_definition",
            "data_range",
        ):
            if key in provided:
                supplied = _jsonable(provided[key])
                actual = receipt.get(key)
                if key == "initialization_hash" and len(initialization_hashes) > 1:
                    raise ValueError(
                        "source receipt 'initialization_hash' is ambiguous across subjects; "
                        "use subject_initialization_hashes"
                    )
                if key == "subject_initialization_hashes" and subject_initialization_hashes:
                    if supplied != subject_initialization_hashes:
                        raise ValueError(
                            "source receipt 'subject_initialization_hashes' conflicts with subject context"
                        )
                    continue
                if actual is not None and supplied != actual:
                    raise ValueError(
                        f"source receipt {key!r} conflicts with sealed service identity"
                    )
                # Fill only fields that were not available from the actual
                # context/role/options; never replace an observed identity.
                if actual is None:
                    receipt[key] = supplied
    # Metadata-level role identities follow the same conflict guard.  These
    # fields are intentionally checked separately because they are often
    # supplied outside the nested source_receipt mapping by StageInputs.
    if isinstance(metadata, Mapping):
        for key in (
            "producer_compatibility_hash",
            "normalization_hash",
            "initialization_hash",
            "subject_initialization_hashes",
            "baseline_split_hash",
            "training_role_manifest_hash",
            "role_manifest_hash",
            "split_role",
            "mask_definition",
            "label_definition",
            "loss_definition",
            "data_range",
        ):
            if key not in metadata:
                continue
            supplied = _jsonable(metadata[key])
            normalized_key = (
                "training_role_manifest_hash" if key == "role_manifest_hash" else key
            )
            actual = receipt.get(normalized_key)
            if key == "subject_initialization_hashes" and subject_initialization_hashes:
                if _jsonable(metadata[key]) != subject_initialization_hashes:
                    raise ValueError(
                        "metadata 'subject_initialization_hashes' conflicts with subject context"
                    )
                continue
            if key == "initialization_hash" and len(initialization_hashes) > 1:
                raise ValueError(
                    "metadata 'initialization_hash' is ambiguous across subjects; "
                    "use subject_initialization_hashes"
                )
            if actual is not None and supplied != actual:
                raise ValueError(
                    f"metadata {key!r} conflicts with sealed service identity"
                )
    return receipt


def run_evaluation(inputs: Any, options: ExperimentOptions, output_dir: Path) -> Mapping[str, Any]:
    """Execute one bounded scenario under detached/eval service semantics."""

    with _service_execution(getattr(inputs, "model", None)):
        return _run_evaluation_impl(inputs, options, output_dir)


def _run_evaluation_impl(inputs: Any, options: ExperimentOptions, output_dir: Path) -> Mapping[str, Any]:
    """Run one bounded scenario over target-free samples and late targets."""

    from .stages import StageInputs

    if not isinstance(inputs, StageInputs):
        raise TypeError("inputs must be StageInputs")
    if not isinstance(options, ExperimentOptions):
        raise TypeError("options must be ExperimentOptions")
    artifact_names = [
        "metrics.json",
        "paired_subjects.jsonl",
        "action_metrics.jsonl",
        "parallel_interactions.jsonl",
        "effective_policy.json",
    ]
    if options.local_footprint_audit:
        artifact_names.append("local_footprint_audit.jsonl")
    destination = _prepare_output(output_dir, tuple(artifact_names))
    started = time.perf_counter()
    paired: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    parallel_interactions: list[dict[str, Any]] = []
    policies: list[dict[str, Any]] = []
    stop_rows: list[dict[str, Any]] = []
    telescoping_rows: list[dict[str, Any]] = []
    local_audit_rows: list[dict[str, Any]] = []
    action_identity: dict[str, str] = {}
    action_identity_violations = 0
    repeated_selected_point_count = 0
    evaluated_experiment_count = 0
    context_receipts: list[dict[str, Any]] = []
    pipeline_counter_rows: list[dict[str, Any]] = []
    samples = tuple(getattr(inputs, "samples", ()))[: options.max_subjects]
    if not samples:
        raise ValueError("evaluation requires at least one target-free sample")
    config = _resolve_config(inputs)
    for index, sample in enumerate(samples):
        subject_started = time.perf_counter()
        subject_id = _sample_id(sample, index)
        pipeline_counters: dict[str, int] = {
            "observation_encode_calls": 0,
            "initial_decode_calls": 0,
            "final_decode_calls": 0,
        }
        context = _context_for_sample(
            inputs, sample, pipeline_counters=pipeline_counters
        )
        context_producer = getattr(context, "producer", None)
        context_compatibility = getattr(context_producer, "compatibility", context_producer)
        initial_planes = getattr(context, "initial_planes", None)
        initialization_hash = (
            canonical_digest(
                {
                    "context_id": getattr(context, "context_id", None),
                    "planes": _jsonable(initial_planes),
                },
                prefix="pfgr-lite-initialization-v1|",
            )
            if initial_planes is not None
            else None
        )
        context_receipts.append(
            {
                "subject_id": subject_id,
                "context_id": getattr(context, "context_id", None),
                "producer_compatibility_hash": getattr(
                    context_compatibility,
                    "digest",
                    getattr(context_producer, "compatibility_hash", None),
                ),
                "normalization_hash": getattr(
                    context_compatibility,
                    "observation_normalization_hash",
                    getattr(context_producer, "observation_normalization_hash", None),
                ),
                "initialization_hash": initialization_hash,
                "source_provenance": _jsonable(getattr(context, "source_provenance", None)),
            }
        )
        lattice = _build_lattice(inputs, context, getattr(inputs, "model", None), config)
        policy = _load_policy(inputs, context, options, config)
        if policy is not None:
            policies.append({"subject_id": subject_id, "policy": _policy_metadata(policy)})
        route, _, _ = _route_for_sample(inputs, sample, context, options, config, lattice, policy)
        initial_prediction = _prediction_for(
            getattr(inputs, "model", None),
            route,
            context,
            final=False,
            options=options,
            pipeline_counters=pipeline_counters,
        )
        final_prediction = _prediction_for(
            getattr(inputs, "model", None),
            route,
            context,
            final=True,
            options=options,
            pipeline_counters=pipeline_counters,
        )
        # This call is intentionally after target-free route and final decode.
        target_context = _target_join(inputs, sample, context, route, final_prediction, options)
        target, mask = _target_parts(target_context)
        row = paired_subject_metrics(
            initial_prediction,
            final_prediction,
            target,
            mask,
            data_range=options.data_range,
            charbonnier_epsilon=options.charbonnier_epsilon,
            ssim_window=options.ssim_window,
            subject_id=subject_id,
            context_id=getattr(context, "context_id", None),
            scenario=options.scenario,
            budget=_route_attr(route, "k", options.budget),
        )
        row["route"] = {
            "k": _route_attr(route, "k", 0),
            "stop_reason": _route_attr(route, "stop_reason", "unknown"),
            "context_id": getattr(context, "context_id", None),
            "policy_hash": _route_attr(route, "policy_hash", getattr(policy, "policy_hash", None)),
            "counters": _jsonable(_counter_metadata(route)),
            "counter_scope": "route_only; excludes outer encode/decode",
        }
        row["pipeline_counters"] = dict(pipeline_counters)
        pipeline_counter_rows.append(
            {
                "subject_id": subject_id,
                "counters": dict(pipeline_counters),
                "scope": "service_outer_calls_only; callback-built contexts/routes uninstrumented",
            }
        )
        row["z0_digest"] = tensor_digest(
            initial_prediction.detach(), name="z0_prediction"
        )
        initial_state = _route_attr(route, "initial_state")
        if initial_state is None:
            trace_for_initial = _completed_trace(route)
            states_for_initial = tuple(
                _route_attr(trace_for_initial, "states", ())
            ) if trace_for_initial is not None else ()
            if states_for_initial:
                initial_state = states_for_initial[0]
        if initial_state is None:
            parallel_for_initial = _route_attr(route, "parallel_trace")
            initial_state = _route_attr(parallel_for_initial, "initial_state")
        row["z0_state_digest"] = getattr(initial_state, "state_digest", None)
        paired.append(row)
        selected = _selected_actions(route)
        selected_point_ids = [
            int(action.point_id)
            for action in selected
            if isinstance(getattr(action, "point_id", None), int)
        ]
        subject_repeat_count = len(selected_point_ids) - len(set(selected_point_ids))
        repeated_selected_point_count += max(subject_repeat_count, 0)
        evaluated_experiment_count += 1
        row["selected_point_ids"] = selected_point_ids
        row["repeated_selected_point_count"] = max(subject_repeat_count, 0)
        labels = _measure_actions(inputs, route, selected, target_context, context, lattice, options)
        # Teacher/query/decode/cache counters are updated by the measurement
        # seam; retain a post-measurement snapshot separately from the route
        # snapshot captured before target access.
        row["route"]["counters_after_measurement"] = _jsonable(
            _counter_metadata(route)
        )
        row["route"]["counter_after_scope"] = (
            "route_and_teacher_measurement; excludes outer encode/decode"
        )
        action_rows = _action_rows(
            route,
            labels,
            subject_id=subject_id,
            scenario=options.scenario,
            budget=options.budget,
            numerical_tolerance=options.numerical_tolerance,
            practical_margin=options.practical_margin,
        )
        if options.local_footprint_audit:
            states = {}
            trace = _completed_trace(route)
            if trace is not None:
                states = {
                    int(state.state_version): state
                    for state in tuple(getattr(trace, "states", ()))
                }
            parallel = _route_attr(route, "parallel_trace")
            if parallel is not None:
                states = {int(parallel.initial_state.state_version): parallel.initial_state}
            decoder = getattr(getattr(inputs, "model", None), "decoder", None)
            if decoder is None:
                decoder = _route_attr(route, "decoder")
            if lattice is None or decoder is None:
                local_audit_rows.append(
                    {
                        "audit_scope": "local_physical_sphere_vs_complete_footprint",
                        "available": False,
                        "reason": "canonical_lattice_or_decoder_unavailable",
                        "subject_id": subject_id,
                    }
                )
            else:
                for audit_row in _local_footprint_audit_rows(
                    selected,
                    states,
                    target_context,
                    lattice,
                    decoder,
                    chunk_size=options.decode_chunk_size,
                    epsilon=float(options.charbonnier_epsilon),
                    numerical_tolerance=float(options.numerical_tolerance),
                ):
                    audit_row["subject_id"] = subject_id
                    local_audit_rows.append(audit_row)
        actions.extend(action_rows)
        interaction = _parallel_interaction_row(
            route,
            row,
            action_rows,
            subject_id=subject_id,
            scenario=options.scenario,
            budget=options.budget,
            numerical_tolerance=options.numerical_tolerance,
            practical_margin=options.practical_margin,
        )
        if interaction is not None:
            # Interaction is a joint-vs-independent diagnostic, not an
            # additional selected action.  Keep it in its own collection so
            # useful/harm/neutral action denominators count only actual writes.
            parallel_interactions.append(interaction)
        selected_gain_values = [float(action_row["true_gain"]) for action_row in action_rows if action_row.get("true_gain") is not None]
        initial_loss = row.get("before", {}).get("masked_charbonnier")
        final_loss = row.get("after", {}).get("masked_charbonnier")
        if initial_loss is not None and final_loss is not None and selected_gain_values:
            from .metrics import telescoping_residual

            residual = telescoping_residual(selected_gain_values, float(initial_loss), float(final_loss))
        else:
            residual = None
        telescoping_rows.append(
            {
                "subject_id": subject_id,
                "executed_gain_sum": float(sum(selected_gain_values)),
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "residual": residual,
                "exact_claim": bool(
                    options.teacher_mode == "exact_footprint"
                    and options.scenario != "parallel_topk"
                    and residual is not None
                ),
                "exact_claim_reason": (
                    "parallel_initial_state_actions_are_not_sequential_telescoping"
                    if options.scenario == "parallel_topk"
                    else None
                ),
            }
        )
        for action_row in action_rows:
            action_id = action_row.get("action_id")
            digest = action_row.get("action_digest")
            if action_id is not None and digest is not None:
                prior = action_identity.get(str(action_id))
                if prior is not None and prior != str(digest):
                    action_identity_violations += 1
                action_identity[str(action_id)] = str(digest)
        gain_by_action = {
            str(action_row.get("action_id")): action_row.get("true_gain")
            for action_row in action_rows
            if action_row.get("action_id") is not None
        }
        selected_actions = iter(selected)
        for decision in tuple(_route_attr(route, "decisions", ())):
            selected_action = next(selected_actions, None) if _route_attr(decision, "stop_code", "") == "continue" else None
            selected_action_id = None if selected_action is None else str(getattr(selected_action, "action_id", None))
            stop_rows.append(
                {
                    "subject_id": subject_id,
                    "stop_code": _route_attr(decision, "stop_code", ""),
                    "continued": _route_attr(decision, "stop_code", "") == "continue",
                    "selected": _route_attr(decision, "selected_point_id", -1) >= 0,
                    "selected_action_id": selected_action_id,
                    "selected_gain": None if selected_action_id is None else gain_by_action.get(selected_action_id),
                    "raw_gain": None if selected_action_id is None else gain_by_action.get(selected_action_id),
                    "best_true_gain": None,
                    "measured_denominator": 1 if selected_action_id in gain_by_action else 0,
                    "candidate_scope": "selected_only",
                }
            )
        row["pipeline_elapsed_seconds"] = float(time.perf_counter() - subject_started)
    aggregate = aggregate_subject_metrics(paired)
    action_summary = aggregate_action_metrics(actions, numerical_tolerance=options.numerical_tolerance, practical_margin=options.practical_margin)
    improvements = [
        float(item["improvement"]["masked_charbonnier"])
        for item in paired
        if item.get("improvement", {}).get("masked_charbonnier") is not None
    ]
    scientific = scientific_decision(
        improvements,
        practical_margin=options.practical_margin,
        minimum_subjects=options.minimum_subjects,
    )
    source_receipt = _source_receipt(inputs, options, contexts=context_receipts)
    payload = {
        "schema_version": METRICS_SCHEMA,
        "software_status": "SOFTWARE_PASS",
        "scientific_status": scientific["decision"],
        "scientific_decision": scientific,
        "options": options.as_dict(),
        "source_receipt": source_receipt,
        "subjects": aggregate,
        "actions": action_summary,
        "parallel_interactions": {
            "schema_version": METRICS_SCHEMA,
            "count": len(parallel_interactions),
            "rows": parallel_interactions,
            "scope": "joint_gain_minus_sum_individual_initial_state_v1",
        },
        "stopping": stopping_diagnostics(stop_rows, practical_margin=options.practical_margin, numerical_tolerance=options.numerical_tolerance),
        # Action IDs are fresh per state; only repeated selected point IDs are
        # counted here.  Keep the number of independently evaluated subjects
        # separate so a one-pass experiment is not mistaken for a repeat run.
        "repeat_count": repeated_selected_point_count,
        "experiment_repeat_count": evaluated_experiment_count,
        "repeat_count_definition": "duplicate_selected_point_ids_across_executed_actions_v1",
        "local_footprint_audit": {
            "enabled": options.local_footprint_audit,
            "rows": len(local_audit_rows),
            "scope": "local_physical_sphere_vs_complete_footprint",
        },
        "same_action_identity_violations": action_identity_violations,
        "telescoping": {
            "rows": telescoping_rows,
            "max_abs_residual": max((abs(float(item["residual"])) for item in telescoping_rows if item["residual"] is not None), default=None),
            "exact_claim_scope": "exact_footprint only; iid_fixed_q retains MC uncertainty",
        },
        "input_data_counters": _jsonable(getattr(inputs, "metadata", {}).get("counters")) if isinstance(getattr(inputs, "metadata", {}), Mapping) else None,
        "pipeline_counters": {
            "rows": pipeline_counter_rows,
            "scope": "service_outer_calls_only; route counters remain separately tagged",
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "scientific_scope": "paired synthetic/declared cohort only; no real-data or CUDA claim",
    }
    _write_jsonl(destination / "paired_subjects.jsonl", paired)
    _write_jsonl(destination / "action_metrics.jsonl", actions)
    _write_jsonl(destination / "parallel_interactions.jsonl", parallel_interactions)
    write_json(destination / "metrics.json", payload)
    write_json(destination / "effective_policy.json", {"schema_version": EXPERIMENT_OPTIONS_SCHEMA, "policies": policies})
    local_audit_path = None
    if options.local_footprint_audit:
        local_audit_path = destination / "local_footprint_audit.jsonl"
        _write_jsonl(local_audit_path, local_audit_rows)
    return {
        "software_status": "SOFTWARE_PASS",
        "scientific_status": scientific["decision"],
        "subject_count": len(paired),
        "action_count": len(actions),
        "metrics_path": destination / "metrics.json",
        "paired_subjects_path": destination / "paired_subjects.jsonl",
        "action_metrics_path": destination / "action_metrics.jsonl",
        "parallel_interactions_path": destination / "parallel_interactions.jsonl",
        "effective_policy_path": destination / "effective_policy.json",
        "local_footprint_audit_path": local_audit_path,
        "source_receipt": source_receipt,
    }


__all__ = ["EXPERIMENT_OPTIONS_SCHEMA", "ExperimentOptions", "run_evaluation"]
