"""Target-free PFGR-Lite data adapters and training-role preflight.

The legacy BraTS21 loader remains the single MRI implementation.  This module
only requests its observation-only mode, owns a detached/sanitised sample, and
offers an explicit late supervision join after a prediction/trace is sealed.
No target or segmentation object is reachable from :class:`TargetFreeSample`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor

from .provenance import canonical_digest, tensor_digest
from .types import CompletedBehaviorTrace, ObservationContext, TrainingRoleManifest


DATA_SCHEMA = "pfgr-lite-target-free-data-v1"
NORMALIZATION_SCHEMA = "pfgr-observation-normalization-v1"
SUBJECT_CONTEXT_BINDING_SCHEMA = "pfgr-lite-subject-context-binding-v1"
_NORMALIZATION_KEYS = {
    "policy",
    "brain_mask_threshold",
    "normalization_epsilon",
    "normalization_policy",
    "lower_percentile",
    "upper_percentile",
    "mask_version",
    "range",
}


@dataclass
class DataAccessCounters:
    """Read counters shared by observation and deferred-target callbacks."""

    observation_reads: int = 0
    target_reads: int = 0
    segmentation_reads: int = 0

    def __post_init__(self) -> None:
        for name in fields(self):
            value = getattr(self, name.name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name.name} must be a nonnegative integer")

    def as_dict(self) -> dict[str, int]:
        return {name.name: int(getattr(self, name.name)) for name in fields(self)}


def _finite_tensor(name: str, value: Tensor) -> Tensor:
    if not isinstance(value, Tensor) or value.numel() == 0:
        raise TypeError(f"{name} must be a non-empty tensor")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain finite values")
    return value


def _normalization_values(value: object | None) -> dict[str, Any]:
    if value is None:
        return {
            "brain_mask_threshold": 0.0,
            "normalization_epsilon": 1e-6,
            "normalization_policy": "masked_zscore",
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
        }
    if hasattr(value, "__dataclass_fields__"):
        raw = {item.name: getattr(value, item.name) for item in fields(value)}
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raw = {name: getattr(value, name) for name in _NORMALIZATION_KEYS if hasattr(value, name)}
    unknown = set(raw) - _NORMALIZATION_KEYS
    if unknown:
        raise ValueError(f"unknown normalization keys: {sorted(unknown)}")
    if "policy" in raw and "normalization_policy" not in raw:
        raw["normalization_policy"] = raw["policy"]
    defaults = _normalization_values(None)
    defaults.update(raw)
    for name in ("brain_mask_threshold", "normalization_epsilon", "lower_percentile", "upper_percentile"):
        try:
            number = float(defaults[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        defaults[name] = number
    if defaults["normalization_epsilon"] <= 0.0 or defaults["brain_mask_threshold"] < 0.0:
        raise ValueError("normalization epsilon must be positive and mask threshold nonnegative")
    if not isinstance(defaults["normalization_policy"], str) or not defaults["normalization_policy"]:
        raise ValueError("normalization_policy must be a nonempty string")
    if not 0.0 <= defaults["lower_percentile"] < defaults["upper_percentile"] <= 100.0:
        raise ValueError("normalization percentile bounds are invalid")
    for name in ("mask_version", "range"):
        if name in raw:
            if name == "mask_version" and (not isinstance(raw[name], str) or not raw[name]):
                raise ValueError("mask_version must be a nonempty string")
            defaults[name] = _canonical(raw[name])
    return defaults


def normalization_identity(metadata: Mapping[str, Any] | None = None, *, config: object | None = None) -> str:
    """Derive a global recipe identity, independent of subject statistics.

    Realized means/std/percentiles remain in ``TargetFreeSample`` metadata for
    replay, but are deliberately excluded from this producer-wide identity so
    two subjects using one fixed recipe share a compatibility hash.
    """

    values = _normalization_values(config)
    payload: dict[str, Any] = {"schema_version": NORMALIZATION_SCHEMA, "config": values}
    if metadata is not None:
        # Metadata is retained as an explicit recipe/field declaration.  The
        # legacy loader's realized modality statistics are omitted here.
        payload["recipe_fields"] = {
            key: _canonical(metadata[key])
            for key in sorted(metadata)
            if key in {"policy", "normalization_policy", "brain_mask_threshold", "normalization_epsilon", "lower_percentile", "upper_percentile", "mask_version", "range", "schema_version"}
        }
    return canonical_digest(payload, prefix="pfgr-lite-observation-normalization-v1|")


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _canonical(value.to_dict())
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("normalization metadata contains a nonfinite float")
        return repr(value)
    return str(value)


def _geometry_identity(geometry: object) -> str:
    shape = getattr(geometry, "shape_dhw", None)
    affine = getattr(geometry, "voxel_to_ras_mm", None)
    if shape is None or affine is None:
        raise TypeError("sample geometry must expose shape_dhw and voxel_to_ras_mm")
    return canonical_digest({"shape_dhw": tuple(shape), "voxel_to_ras_mm": affine}, prefix="pfgr-lite-data-geometry-v1|")


def _adapt_volume_geometry(geometry: object, shape_dhw: Sequence[int] | None = None) -> object:
    """Convert the legacy NIfTI geometry record to PFGR's typed geometry.

    ``NiftiGeometryMetadata`` intentionally mirrors the source affine but is
    not the ``VolumeGeometry`` type consumed by ``PFGRLiteModel``.  Keeping
    this conversion at the target-free data boundary preserves the complete
    4x4 affine (including shear and translation) rather than silently reducing
    it to voxel spacing.
    """

    from smagm.features.point_guided.contracts import VolumeGeometry

    if isinstance(geometry, VolumeGeometry):
        if shape_dhw is not None and tuple(geometry.shape_dhw) != tuple(shape_dhw):
            raise ValueError("sample geometry shape does not match observations")
        return geometry
    shape = getattr(geometry, "shape_dhw", None) or shape_dhw
    affine = getattr(geometry, "voxel_to_ras_mm", None)
    if affine is None:
        affine = getattr(geometry, "affine_xyz_to_ras_mm", None)
    if shape is None or affine is None:
        raise TypeError("legacy geometry must expose shape_dhw and a complete 4x4 affine")
    converted = VolumeGeometry(tuple(int(item) for item in shape), affine)
    if shape_dhw is not None and tuple(converted.shape_dhw) != tuple(shape_dhw):
        raise ValueError("sample geometry shape does not match observations")
    return converted


def _normalization_recipe_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only loader recipe fields from per-modality metadata."""

    values: dict[str, Any] = {}
    for key in ("normalization_policy", "policy", "lower_percentile", "upper_percentile", "mask_version", "range", "brain_mask_threshold", "normalization_epsilon"):
        if key in metadata:
            values[key] = metadata[key]
    # Legacy values are usually ``ModalityNormalizationMetadata`` records.
    records = [value for key, value in metadata.items() if key in {"T1", "T2", "FLAIR"}]
    if records:
        record = records[0]
        for key in ("normalization_policy", "lower_percentile", "upper_percentile"):
            if key not in values and hasattr(record, key):
                values[key] = getattr(record, key)
    if "normalization_policy" not in values and "policy" in values:
        values["normalization_policy"] = values["policy"]
    return values


@dataclass(frozen=True)
class TargetFreeSample:
    """Owned observation-only sample handed to PFGR model services."""

    subject_id: str
    observations: Tensor
    brain_mask: Tensor
    geometry: object
    normalization_metadata: Mapping[str, Any]
    normalization_hash: str
    geometry_hash: str
    source_paths: Mapping[str, Path] = field(default_factory=lambda: MappingProxyType({}))
    schema_version: str = DATA_SCHEMA
    target_free: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != DATA_SCHEMA:
            raise ValueError("unknown target-free sample schema")
        if not isinstance(self.subject_id, str) or not self.subject_id.strip():
            raise ValueError("subject_id must be a nonempty string")
        if not isinstance(self.target_free, bool) or not self.target_free:
            raise ValueError("TargetFreeSample must be explicitly target_free")
        _finite_tensor("observations", self.observations)
        if self.observations.ndim != 4 or self.observations.shape[0] != 3:
            raise ValueError("observations must have shape [3,D,H,W] ordered T1/T2/FLAIR")
        if self.observations.dtype != torch.float32:
            raise TypeError("target-free observations must be FP32")
        if not isinstance(self.brain_mask, Tensor) or self.brain_mask.dtype != torch.bool:
            raise TypeError("brain_mask must be bool")
        expected = tuple(self.observations.shape[1:])
        if self.brain_mask.shape not in ((1, *expected), expected):
            raise ValueError("brain_mask must align with observations")
        if self.brain_mask.ndim == 3:
            object.__setattr__(self, "brain_mask", self.brain_mask.unsqueeze(0).clone())
        else:
            object.__setattr__(self, "brain_mask", self.brain_mask.clone())
        if not bool(self.brain_mask.any()):
            raise ValueError("brain_mask must contain at least one voxel")
        expected_geometry_hash = _geometry_identity(self.geometry)
        if not isinstance(self.geometry_hash, str):
            raise TypeError("geometry_hash must be a string")
        if not self.geometry_hash:
            object.__setattr__(self, "geometry_hash", expected_geometry_hash)
        elif self.geometry_hash != expected_geometry_hash:
            raise ValueError("geometry_hash does not match sample geometry")
        metadata = {str(key): value for key, value in dict(self.normalization_metadata).items()}
        forbidden_metadata = {"target", "target_t1ce", "segmentation", "ground_truth", "oracle", "label"}
        if forbidden_metadata & {key.lower() for key in metadata}:
            raise ValueError("TargetFreeSample normalization metadata cannot carry target/segmentation payloads")
        object.__setattr__(self, "normalization_metadata", MappingProxyType(metadata))
        if not isinstance(self.normalization_hash, str):
            raise TypeError("normalization_hash must be a string")
        if not self.normalization_hash:
            recipe = {
                "normalization_policy": metadata.get("normalization_policy", metadata.get("policy", "masked_zscore")),
            }
            for key in ("brain_mask_threshold", "normalization_epsilon", "lower_percentile", "upper_percentile", "mask_version", "range"):
                if key in metadata:
                    recipe[key] = metadata[key]
            object.__setattr__(self, "normalization_hash", normalization_identity(config=recipe))
        paths = {str(key): Path(value).resolve() for key, value in dict(self.source_paths).items()}
        if forbidden_metadata & {key.lower() for key in paths}:
            raise ValueError("TargetFreeSample source_paths cannot carry target/segmentation paths")
        object.__setattr__(self, "source_paths", MappingProxyType(paths))
        object.__setattr__(self, "observations", self.observations.detach().clone())

    @property
    def inputs(self) -> Tensor:
        return self.observations

    @property
    def mask(self) -> Tensor:
        return self.brain_mask

    @property
    def shape_dhw(self) -> tuple[int, int, int]:
        return tuple(int(item) for item in self.observations.shape[-3:])

    @property
    def observation_record_id(self) -> str:
        return canonical_digest(
            {
                "schema_version": DATA_SCHEMA,
                "subject_id": self.subject_id,
                "source_paths": {key: str(value) for key, value in sorted(self.source_paths.items())},
                "observations": tensor_digest(self.observations, name="observations"),
                "brain_mask": tensor_digest(self.brain_mask, name="brain_mask"),
                "geometry_hash": self.geometry_hash,
                "normalization_hash": self.normalization_hash,
            },
            prefix="pfgr-lite-observation-record-v1|",
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "subject_id": self.subject_id,
            "shape_dhw": self.shape_dhw,
            "observation_modalities": ["T1", "T2", "FLAIR"],
            "normalization_hash": self.normalization_hash,
            "geometry_hash": self.geometry_hash,
            "source_modalities": sorted(self.source_paths),
            "target_free": True,
        }


PFGRObservationSample = TargetFreeSample
ObservationSample = TargetFreeSample


def _validate_loaded_target_identity(sample: TargetFreeSample, loaded: object) -> None:
    """Check a target-bearing legacy sample against its observation record.

    A target provider may return W2's validated target context or a legacy
    ``BraTS21PointGuidedSample``.  The latter contains an observation copy;
    checking that copy, its mask, affine, recipe and source paths prevents a
    same-shape/same-affine sample from another subject being joined.
    """

    def loaded_value(name: str, default: Any = None) -> Any:
        if isinstance(loaded, Mapping):
            return loaded.get(name, default)
        return getattr(loaded, name, default)

    loaded_subject = loaded_value("subject_id")
    if loaded_subject is not None and str(loaded_subject) != sample.subject_id:
        raise ValueError("deferred target subject does not match target-free sample")
    loaded_geometry = loaded_value("geometry")
    if loaded_geometry is not None:
        adapted = _adapt_volume_geometry(loaded_geometry, sample.shape_dhw)
        if _geometry_identity(adapted) != sample.geometry_hash:
            raise ValueError("deferred target geometry does not match target-free observations")
    loaded_observations = loaded_value("observations")
    if loaded_observations is not None:
        if not isinstance(loaded_observations, Tensor) or loaded_observations.shape != sample.observations.shape:
            raise ValueError("deferred target observations do not match target-free shape")
        if loaded_observations.dtype != torch.float32 or not bool(torch.isfinite(loaded_observations).all()):
            raise ValueError("deferred target observations must be finite float32")
        if not torch.equal(loaded_observations, sample.observations):
            raise ValueError("deferred target observations do not match target-free record")
    loaded_mask = loaded_value("brain_mask")
    if loaded_mask is not None:
        if not isinstance(loaded_mask, Tensor):
            raise TypeError("deferred target brain_mask must be a tensor")
        resolved_mask = loaded_mask if loaded_mask.ndim == 4 else loaded_mask.unsqueeze(0)
        if resolved_mask.shape != sample.brain_mask.shape or resolved_mask.dtype is not torch.bool or not torch.equal(resolved_mask, sample.brain_mask):
            raise ValueError("deferred target mask does not match target-free observations")
    loaded_paths = loaded_value("source_paths")
    if loaded_paths is not None and sample.source_paths:
        loaded_paths = {str(key): Path(value).resolve() for key, value in dict(loaded_paths).items()}
        for modality in ("T1", "T2", "FLAIR"):
            expected = sample.source_paths.get(modality)
            actual = loaded_paths.get(modality)
            if expected is not None and actual is not None and actual != expected:
                raise ValueError(f"deferred target source path for {modality} does not match observation record")
    loaded_metadata = loaded_value("normalization_metadata")
    if isinstance(loaded_metadata, Mapping):
        expected_recipe = _normalization_recipe_from_metadata(sample.normalization_metadata)
        actual_recipe = _normalization_recipe_from_metadata(loaded_metadata)
        for key in ("normalization_policy", "lower_percentile", "upper_percentile", "mask_version", "range"):
            if key in expected_recipe and key in actual_recipe and _canonical(expected_recipe[key]) != _canonical(actual_recipe[key]):
                raise ValueError("deferred target normalization recipe does not match observations")


def _sample_from_legacy(sample: object, *, config: Mapping[str, Any], counters: DataAccessCounters | None) -> TargetFreeSample:
    observations = getattr(sample, "observations", None)
    mask = getattr(sample, "brain_mask", None)
    geometry = getattr(sample, "geometry", None)
    subject_id = getattr(sample, "subject_id", None)
    metadata = getattr(sample, "normalization_metadata", {})
    paths = getattr(sample, "source_paths", {})
    if observations is None or mask is None or geometry is None or subject_id is None:
        raise TypeError("legacy loader did not return a complete observation sample")
    observations = _finite_tensor("observations", observations)
    mask = mask if mask.ndim == 4 else mask.squeeze(0)
    if mask.dtype is not torch.bool:
        mask = mask.to(dtype=torch.bool)
    metadata_dict = _canonical(metadata)
    if not isinstance(metadata_dict, dict):
        metadata_dict = {"legacy_metadata": metadata_dict}
    # Include concrete loader policy and measured modality metadata.  This
    # identity is intentionally not the arbitrary PFGR config string.
    norm_hash = normalization_identity(config=config)
    # ``ProducerCompatibility`` binds the loader recipe explicitly.  Keeping
    # this value in the detached sample metadata lets a late target join
    # reject a target loaded with a different normalization policy without
    # exposing any target tensor to the observation-only sample.
    metadata_dict["producer_normalization_hash"] = norm_hash
    metadata_dict["normalization_recipe"] = _canonical(config)
    adapted_geometry = _adapt_volume_geometry(geometry, tuple(observations.shape[-3:]))
    if counters is not None:
        counters.observation_reads += 1
    return TargetFreeSample(
        subject_id=str(subject_id),
        observations=observations,
        brain_mask=mask,
        geometry=adapted_geometry,
        normalization_metadata=metadata_dict,
        normalization_hash=norm_hash,
        geometry_hash=_geometry_identity(geometry),
        source_paths={key: value for key, value in dict(paths).items() if str(key) in {"T1", "T2", "FLAIR"}},
    )


def load_observation_sample(
    subject: str | Path | object,
    *,
    normalization_config: object | None = None,
    counters: DataAccessCounters | None = None,
) -> TargetFreeSample:
    """Load only T1/T2/FLAIR through the existing observation loader."""

    config = _normalization_values(normalization_config)
    loader_keys = {
        "brain_mask_threshold",
        "normalization_epsilon",
        "normalization_policy",
        "lower_percentile",
        "upper_percentile",
    }
    loader_config = {key: config[key] for key in loader_keys if key in config}
    # Import lazily so importing PFGR configuration never imports the MRI
    # stack; W3b data calls are explicit and target-free.
    from smagm.data.brats21_point_guided import load_point_guided_subject

    loaded = load_point_guided_subject(
        subject,
        require_target=False,
        load_target=False,
        require_segmentation=False,
        load_segmentation=False,
        **loader_config,
    )
    return _sample_from_legacy(loaded, config=config, counters=counters)


class _DeferredSupervision:
    def __init__(
        self,
        sample: TargetFreeSample,
        provider: Callable[..., object],
        counters: DataAccessCounters | None,
        engineering_only: bool,
        include_segmentation: bool,
    ):
        self.sample = sample
        self.provider = provider
        self.counters = counters
        self.engineering_only = engineering_only
        self.include_segmentation = include_segmentation
        self._joined = False

    def __call__(
        self,
        *,
        completed_context: object | None = None,
        prediction: object | None = None,
        trace: object | None = None,
    ) -> object:
        if completed_context is None and prediction is None and trace is None:
            raise ValueError("deferred supervision requires a completed prediction/context/trace")
        if self._joined:
            raise RuntimeError("deferred supervision callback may be consumed only once")
        self._joined = True
        try:
            if not self.engineering_only:
                # Production has one unambiguous seam: ``provider(subject_id
                # : str)``.  Introspection/name-based argument guessing is
                # retained only for explicitly marked engineering fixtures.
                if not isinstance(completed_context, ObservationContext):
                    raise TypeError("production deferred supervision requires an actual ObservationContext")
                if prediction is None and not isinstance(trace, CompletedBehaviorTrace):
                    raise TypeError("production deferred supervision requires a completed prediction or behavior trace")
                if trace is not None and not isinstance(trace, CompletedBehaviorTrace):
                    raise TypeError("production deferred supervision trace must be CompletedBehaviorTrace")
                loaded = self.provider(self.sample.subject_id)
            else:
                try:
                    signature = inspect.signature(self.provider)
                except (TypeError, ValueError):
                    loaded = self.provider(self.sample.subject_id)
                else:
                    params = signature.parameters
                    positional_names = [name for name, item in params.items() if item.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
                    positional_values = {
                        "subject_id": self.sample.subject_id,
                        "subject": self.sample,
                        "sample": self.sample,
                        "completed_context": completed_context,
                        "context": completed_context,
                        "prediction": prediction,
                        "trace": trace,
                    }
                    args: list[Any] = []
                    for index, name in enumerate(positional_names):
                        if name in positional_values:
                            args.append(positional_values[name])
                        elif "trace" in name.lower():
                            args.append(trace)
                        elif "context" in name.lower():
                            args.append(completed_context)
                        elif "prediction" in name.lower():
                            args.append(prediction)
                        elif "sample" in name.lower():
                            args.append(self.sample)
                        elif not args or index == 0:
                            args.append(self.sample.subject_id)
                        else:
                            break
                    accepts_var_kw = any(item.kind == inspect.Parameter.VAR_KEYWORD for item in params.values())
                    kwargs = {
                        key: value
                        for key, value in positional_values.items()
                        if key in params and params[key].kind != inspect.Parameter.POSITIONAL_ONLY and key not in positional_names[: len(args)]
                    }
                    if accepts_var_kw:
                        kwargs = {key: value for key, value in positional_values.items() if key not in positional_names[: len(args)]}
                    loaded = self.provider(*args, **kwargs)
            if self.counters is not None:
                self.counters.target_reads += 1
            if loaded is None:
                raise ValueError("target provider returned no target")
            # A provider may already return W2's validated target context; in
            # that case retain the exact object but still verify dimensions and
            # the observation binding below.
            from .teacher import ValidatedTargetContext, validate_target

            semantic_target: Tensor | None = None
            if self.include_segmentation and isinstance(loaded, ValidatedTargetContext):
                # W2's target context deliberately carries no segmentation
                # payload.  The authorized S0 semantic arm must therefore
                # receive a legacy target-bearing sample (or an equivalent
                # mapping) at this late boundary.
                raise ValueError(
                    "semantic supervision requires a deferred segmentation payload; "
                    "ValidatedTargetContext alone is insufficient"
                )
            if isinstance(loaded, ValidatedTargetContext):
                target_context = loaded
                target = loaded.target
                bound_norm = getattr(target_context, "normalization_hash", None)
                expected_producer_norm = self.sample.normalization_metadata.get("producer_normalization_hash")
                if expected_producer_norm is not None and bound_norm != expected_producer_norm:
                    raise ValueError("deferred target normalization identity does not match sample producer binding")
                if completed_context is not None:
                    expected_context_id = getattr(completed_context, "context_id", None)
                    if target_context.observation_context_id is not None and target_context.observation_context_id != expected_context_id:
                        raise ValueError("deferred target context is bound to a different observation context")
                    context_producer = getattr(completed_context, "producer", None)
                    context_compatibility = getattr(context_producer, "compatibility", context_producer)
                    expected_context_hash = getattr(context_compatibility, "digest", None)
                    if target_context.producer_compatibility_hash is not None and expected_context_hash is not None and target_context.producer_compatibility_hash != expected_context_hash:
                        raise ValueError("deferred target producer identity does not match completed observation context")
                    expected_context_norm = getattr(context_compatibility, "observation_normalization_hash", None)
                    if target_context.normalization_hash is not None and expected_context_norm is not None and target_context.normalization_hash != expected_context_norm:
                        raise ValueError("deferred target normalization identity does not match completed observation context")
                bound_geometry = getattr(target_context, "output_geometry", None)
                if bound_geometry is not None and _geometry_identity(bound_geometry) != self.sample.geometry_hash:
                    raise ValueError("deferred target geometry does not match target-free observations")
            else:
                _validate_loaded_target_identity(self.sample, loaded)
                target = getattr(loaded, "target", None)
                if target is None and isinstance(loaded, Mapping):
                    target = loaded.get("target", loaded.get("target_t1ce"))
                if target is None and isinstance(loaded, Tensor):
                    target = loaded
                if target is None:
                    raise TypeError("target provider must return ValidatedTargetContext, target-bearing sample, or tensor")
                # A reduced N/CPU config is still allowed to exercise the
                # authoritative ObservationContext binding.  Only a genuine
                # stand-in context takes the engineering-only validation
                # branch; never combine the production binding with that
                # marker because W2 rejects the ambiguous combination.
                bound_context = isinstance(completed_context, ObservationContext)
                observation_context_arg = completed_context if bound_context else None
                join_engineering = bool(self.engineering_only and not bound_context)
                target_context = validate_target(
                    getattr(completed_context, "context_id", self.sample.subject_id),
                    target,
                    self.sample.brain_mask,
                    provenance="pfgr-lite-deferred-target-v1",
                    output_geometry=getattr(completed_context, "geometry", None),
                    feature_geometry=getattr(completed_context, "feature_geometry", None),
                    producer_compatibility_hash=getattr(getattr(completed_context, "producer", None), "compatibility_hash", None),
                    # ProducerDependencies carries normalization on its
                    # nested ProducerCompatibility.  Read that authoritative
                    # identity first; falling back to the sample recipe keeps
                    # explicit engineering stand-ins usable without masking a
                    # real context mismatch.
                    normalization_hash=getattr(
                        getattr(getattr(completed_context, "producer", None), "compatibility", None),
                        "observation_normalization_hash",
                        getattr(getattr(completed_context, "producer", None), "observation_normalization_hash", self.sample.normalization_hash),
                    ),
                    completed_trace=trace if not join_engineering or isinstance(trace, CompletedBehaviorTrace) else None,
                    observation_context=observation_context_arg,
                    engineering_only=join_engineering,
                )
                if self.include_segmentation:
                    segmentation = loaded.get("segmentation") if isinstance(loaded, Mapping) else getattr(loaded, "segmentation", None)
                    if segmentation is None:
                        segmentation = loaded.get("segmentation_labels") if isinstance(loaded, Mapping) else getattr(loaded, "segmentation_labels", None)
                    if not isinstance(segmentation, Tensor):
                        raise ValueError(
                            "semantic supervision requires segmentation labels loaded after prediction"
                        )
                    from smagm.features.point_guided.semantic_supervision import build_coarse_semantic_target

                    semantic_target = build_coarse_semantic_target(
                        segmentation.unsqueeze(0) if segmentation.ndim == 3 else segmentation,
                        self.sample.brain_mask,
                    )
                    if tuple(semantic_target.shape[-3:]) != self.sample.shape_dhw:
                        raise ValueError("deferred segmentation geometry does not match target-free observations")
                    if self.counters is not None:
                        self.counters.segmentation_reads += 1
            if tuple(target.shape[-3:]) != self.sample.shape_dhw:
                raise ValueError("deferred target geometry does not match target-free observations")
            if completed_context is not None:
                context_geometry = getattr(completed_context, "geometry", None)
                if context_geometry is not None and _geometry_identity(context_geometry) != self.sample.geometry_hash:
                    raise ValueError("completed context geometry does not match target-free sample")
                producer = getattr(completed_context, "producer", None)
                compatibility = getattr(producer, "compatibility", producer)
                context_norm = getattr(compatibility, "observation_normalization_hash", None)
                expected_producer_norm = self.sample.normalization_metadata.get("producer_normalization_hash")
                if expected_producer_norm is not None and context_norm != expected_producer_norm:
                    raise ValueError("completed context normalization identity does not match sample producer binding")
            if semantic_target is not None:
                # Keep the validated target context as the authoritative
                # reconstruction binding while exposing coarse labels only
                # after the target-free prediction/trace has completed.
                return {
                    "target_context": target_context,
                    "semantic_target": semantic_target.detach().clone(),
                }
            return target_context
        except Exception:
            self._joined = False
            raise


def defer_supervision(
    sample: TargetFreeSample,
    provider: Callable[..., object],
    *,
    completed_context: object | None = None,
    prediction: object | None = None,
    trace: object | None = None,
    counters: DataAccessCounters | None = None,
    engineering_only: bool = False,
    include_segmentation: bool = False,
) -> object:
    """Create or consume a target join that is callable only after inference.

    With no completion marker this returns a one-shot callback.  Supplying a
    completed context/prediction/trace invokes that callback immediately; this
    dual form keeps tiny service calls concise while making the late boundary
    explicit in production code.
    """

    if not isinstance(sample, TargetFreeSample):
        raise TypeError("sample must be TargetFreeSample")
    if not callable(provider):
        raise TypeError("target provider must be callable")
    if not isinstance(include_segmentation, bool):
        raise TypeError("include_segmentation must be bool")
    callback = _DeferredSupervision(sample, provider, counters, engineering_only, include_segmentation)
    if completed_context is None and prediction is None and trace is None:
        return callback
    return callback(completed_context=completed_context, prediction=prediction, trace=trace)


@dataclass(frozen=True)
class SubjectContextBinding:
    """Strict subject/context join receipt consumed by W4 calibration."""

    subject_id: str
    observation_record_id: str
    context_id: str
    geometry_hash: str
    normalization_hash: str
    binding_digest: str = ""
    schema_version: str = SUBJECT_CONTEXT_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SUBJECT_CONTEXT_BINDING_SCHEMA:
            raise ValueError("unknown subject-context binding schema")
        for name in ("subject_id", "observation_record_id", "context_id", "geometry_hash", "normalization_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a complete string")
        payload = {
            "schema_version": self.schema_version,
            "subject_id": self.subject_id,
            "observation_record_id": self.observation_record_id,
            "context_id": self.context_id,
            "geometry_hash": self.geometry_hash,
            "normalization_hash": self.normalization_hash,
        }
        expected = canonical_digest(payload, prefix="pfgr-lite-subject-context-binding-v1|")
        if self.binding_digest and self.binding_digest != expected:
            raise ValueError("binding_digest does not match binding fields")
        object.__setattr__(self, "binding_digest", expected)

    def as_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "subject_id": self.subject_id,
            "observation_record_id": self.observation_record_id,
            "context_id": self.context_id,
            "geometry_hash": self.geometry_hash,
            "normalization_hash": self.normalization_hash,
            "binding_digest": self.binding_digest,
        }


def bind_observation_context(
    sample: TargetFreeSample,
    observation_context: object,
    *,
    observation_record_id: str | None = None,
) -> SubjectContextBinding:
    """Create the immutable subject/context receipt before any target read."""

    if not isinstance(sample, TargetFreeSample):
        raise TypeError("sample must be TargetFreeSample")
    context_id = getattr(observation_context, "context_id", None)
    geometry = getattr(observation_context, "geometry", None)
    producer = getattr(observation_context, "producer", None)
    compatibility = getattr(producer, "compatibility", producer)
    normalization_hash = getattr(compatibility, "observation_normalization_hash", None)
    if not isinstance(context_id, str) or not context_id:
        raise ValueError("observation_context must expose a complete context_id")
    if geometry is None or _geometry_identity(geometry) != sample.geometry_hash:
        raise ValueError("observation context geometry does not match target-free sample")
    if not isinstance(normalization_hash, str) or not normalization_hash:
        # Engineering stand-ins may carry the measured recipe directly; real
        # ObservationContext always exposes compatibility normalization hash.
        normalization_hash = sample.normalization_hash
    return SubjectContextBinding(
        subject_id=sample.subject_id,
        observation_record_id=observation_record_id or sample.observation_record_id,
        context_id=context_id,
        geometry_hash=sample.geometry_hash,
        normalization_hash=normalization_hash,
    )


def _baseline_parts(value: object) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], str]:
    if isinstance(value, (str, Path)):
        from smagm.data.brats21_point_guided import load_point_guided_split

        split = load_point_guided_split(value)
        return tuple(split.train_subject_ids), tuple(split.val_subject_ids), tuple(split.test_subject_ids), split.split_hash
    if hasattr(value, "train_subject_ids") and hasattr(value, "split_hash"):
        return tuple(value.train_subject_ids), tuple(value.val_subject_ids), tuple(value.test_subject_ids), str(value.split_hash)
    if isinstance(value, Mapping):
        train = tuple(str(item) for item in value.get("train_subject_ids", value.get("train", ())))
        validation = tuple(str(item) for item in value.get("baseline_validation_subject_ids", value.get("validation_subject_ids", value.get("val", ()))) )
        test = tuple(str(item) for item in value.get("test_subject_ids", value.get("test", ())))
        split_hash = value.get("split_hash")
        if not isinstance(split_hash, str) or not split_hash:
            raise ValueError("baseline split mapping must carry an externally validated split_hash")
        return train, validation, test, split_hash
    raise TypeError("baseline_split must be a split path, split object, or mapping")


def build_training_role_manifest(
    baseline_split: object,
    *,
    related_groups: Mapping[str, str] | None = None,
    assignment_seed: int = 20260907,
    engineering_only: bool = False,
) -> TrainingRoleManifest:
    """Validate baseline membership and deterministically partition train IDs."""

    train, validation, test, split_hash = _baseline_parts(baseline_split)
    all_ids = set(train) | set(validation) | set(test)
    if len(all_ids) != len(train) + len(validation) + len(test):
        raise ValueError("baseline split roles must be disjoint")
    if not isinstance(assignment_seed, int) or isinstance(assignment_seed, bool):
        raise ValueError("assignment_seed must be an integer")
    groups = {subject: subject for subject in all_ids}
    if related_groups is not None:
        unknown = set(related_groups) - all_ids
        if unknown:
            raise ValueError(f"related_groups contains unknown subjects: {sorted(unknown)}")
        for subject, group in related_groups.items():
            if not isinstance(group, str) or not group:
                raise ValueError("related group identifiers must be nonempty strings")
            groups[subject] = group
    group_members: dict[str, list[str]] = {}
    for subject, group in groups.items():
        group_members.setdefault(group, []).append(subject)
    partition = {subject: "train" for subject in train}
    partition.update({subject: "validation" for subject in validation})
    partition.update({subject: "test" for subject in test})
    for group, members in group_members.items():
        if len({partition[item] for item in members}) != 1:
            raise ValueError(f"related group {group!r} crosses baseline splits")
    train_groups = sorted(
        {groups[subject] for subject in train},
        key=lambda group: (hashlib.sha256(f"pfgr-lite-roles-v1|{assignment_seed}|{group}".encode("utf-8")).hexdigest(), group),
    )
    if engineering_only and len(train_groups) < 65:
        # A tiny engineering harness has no basis for adaptive calibration;
        # keep every available train group in producer_fit so S0/S1/S2/V can
        # still execute while the empty calibration roles make the release
        # limitation explicit.  Production manifests retain the planned
        # first-32/next-32 reservation and require at least one producer group
        # through TrainingRoleManifest validation.
        fit_groups = set(train_groups)
        calibration_fit_groups: set[str] = set()
        allowance_groups: set[str] = set()
    else:
        fit_groups = set(train_groups[64:])
        calibration_fit_groups = set(train_groups[:32])
        allowance_groups = set(train_groups[32:64])
    producer = tuple(sorted(subject for subject in train if groups[subject] in fit_groups))
    calibration_fit = tuple(sorted(subject for subject in train if groups[subject] in calibration_fit_groups))
    calibration_allowance = tuple(sorted(subject for subject in train if groups[subject] in allowance_groups))
    return TrainingRoleManifest(
        baseline_split_hash=split_hash,
        baseline_train_subject_ids=tuple(sorted(train)),
        baseline_validation_subject_ids=tuple(sorted(validation)),
        baseline_test_subject_ids=tuple(sorted(test)),
        producer_fit_subject_ids=producer,
        calibration_fit_subject_ids=calibration_fit,
        calibration_allowance_subject_ids=calibration_allowance,
        subject_group_ids=tuple(sorted((subject, groups[subject]) for subject in all_ids)),
        assignment_seed=assignment_seed,
        engineering_only=bool(engineering_only),
    )


make_training_role_manifest = build_training_role_manifest


__all__ = [
    "DATA_SCHEMA",
    "DataAccessCounters",
    "NORMALIZATION_SCHEMA",
    "ObservationSample",
    "PFGRObservationSample",
    "TargetFreeSample",
    "SUBJECT_CONTEXT_BINDING_SCHEMA",
    "SubjectContextBinding",
    "bind_observation_context",
    "build_training_role_manifest",
    "defer_supervision",
    "load_observation_sample",
    "make_training_role_manifest",
    "normalization_identity",
]
