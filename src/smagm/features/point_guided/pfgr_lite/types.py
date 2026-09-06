"""Authoritative PFGR-Lite tensor and metadata contracts.

The dataclasses here are the only shared declarations consumed by later W2-W5
workers.  They validate shape, dtype, role, finite values, protocol versions,
and owned-tensor mutation guards at construction/use boundaries.  No target,
teacher, oracle, data-loader, or value-fitting module is imported here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import Tensor

from ..contracts import FrontendOutput, VolumeGeometry
from ..state_init import DynamicTriPlanes
from ..spectral_query import FeatureGridGeometry
from .config import EffectTeacherConfig
from .provenance import (
    CalibrationIdentity,
    ProducerCompatibility,
    SourceProvenance,
    ValueFitIdentity,
    canonical_digest,
    tensor_digest,
)
from .static_geometry import MultiScaleFeatureGeometry


PFGR_TYPES_SCHEMA = "pfgr-lite-types-v1"
DESCRIPTOR_SCHEMA = "pfgr-lite-descriptors-v1"
V_DESCRIPTOR_DIMS: tuple[int, int, int, int] = (126, 222, 270, 366)
ACTION_SCHEMA = "pfgr-lite-action-v1"
TRACE_SCHEMA = "pfgr-lite-trace-v1"
COUNTERS_SCHEMA = "pfgr-lite-counters-v1"
INFERENCE_SCHEMA = "point-guided-pfgr-lite-inference-v1"
RESUME_SCHEMA = "point-guided-pfgr-lite-resume-v1"
VALUE_BANK_SCHEMA = "point-guided-pfgr-lite-value-bank-v1"
TRAINING_ROLES_SCHEMA = "pfgr-lite-training-roles-v1"


@dataclass(frozen=True)
class TrainingRoleManifest:
    """Immutable baseline/training role declaration for V/calibration joins."""

    baseline_split_hash: str
    baseline_train_subject_ids: tuple[str, ...]
    baseline_validation_subject_ids: tuple[str, ...]
    baseline_test_subject_ids: tuple[str, ...]
    producer_fit_subject_ids: tuple[str, ...]
    calibration_fit_subject_ids: tuple[str, ...]
    calibration_allowance_subject_ids: tuple[str, ...]
    subject_group_ids: tuple[tuple[str, str], ...]
    assignment_seed: int = 20260907
    engineering_only: bool = False
    schema_version: str = TRAINING_ROLES_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TRAINING_ROLES_SCHEMA:
            raise ValueError("unknown training role manifest schema")
        def _ids(name: str) -> tuple[str, ...]:
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"{name} must be a tuple of nonempty identifiers")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} identifiers must be unique")
            return values

        splits = [_ids(name) for name in ("baseline_train_subject_ids", "baseline_validation_subject_ids", "baseline_test_subject_ids")]
        if any(set(splits[i]) & set(splits[j]) for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("baseline train/validation/test subjects must be disjoint")
        roles = [_ids(name) for name in ("producer_fit_subject_ids", "calibration_fit_subject_ids", "calibration_allowance_subject_ids")]
        if any(set(roles[i]) & set(roles[j]) for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("training role subjects must be disjoint")
        if set().union(*roles) != set(splits[0]):
            raise ValueError("training roles must partition baseline training subjects exactly")
        if not isinstance(self.baseline_split_hash, str) or not self.baseline_split_hash or self.baseline_split_hash.lower() in {"unknown", "unset", "none", "null"}:
            raise ValueError("baseline_split_hash must be complete")
        if not isinstance(self.subject_group_ids, tuple) or not self.subject_group_ids:
            raise ValueError("subject_group_ids must be nonempty")
        group_by_subject: dict[str, str] = {}
        for row in self.subject_group_ids:
            if not isinstance(row, tuple) or len(row) != 2 or any(not isinstance(item, str) or not item for item in row):
                raise ValueError("subject_group_ids rows must be (subject, related-group)")
            subject, group = row
            if subject in group_by_subject:
                raise ValueError("each subject must have exactly one related-group assignment")
            group_by_subject[subject] = group
        all_subjects = set().union(*splits)
        if set(group_by_subject) != all_subjects:
            raise ValueError("subject_group_ids must cover every baseline subject exactly once")
        by_group: dict[str, set[str]] = {}
        for subject, group in group_by_subject.items():
            by_group.setdefault(group, set()).add(subject)
        partition_by_subject = {subject: "train" for subject in splits[0]}
        partition_by_subject.update({subject: "validation" for subject in splits[1]})
        partition_by_subject.update({subject: "test" for subject in splits[2]})
        role_by_subject = {subject: "producer_fit" for subject in roles[0]}
        role_by_subject.update({subject: "calibration_fit" for subject in roles[1]})
        role_by_subject.update({subject: "calibration_allowance" for subject in roles[2]})
        for group_subjects in by_group.values():
            if len({partition_by_subject[item] for item in group_subjects}) != 1:
                raise ValueError("related groups may not cross baseline splits")
            train_subjects = [item for item in group_subjects if item in role_by_subject]
            if train_subjects and len({role_by_subject[item] for item in train_subjects}) != 1:
                raise ValueError("related groups may not cross training roles")
        if not isinstance(self.assignment_seed, int) or isinstance(self.assignment_seed, bool):
            raise ValueError("assignment_seed must be an integer")
        if not isinstance(self.engineering_only, bool):
            raise TypeError("engineering_only must be bool")
        if not self.engineering_only:
            group_counts = {
                role: len({group_by_subject[item] for item in subjects})
                for role, subjects in zip(("producer_fit", "calibration_fit", "calibration_allowance"), roles)
            }
            if group_counts["producer_fit"] < 1:
                raise ValueError("production role manifest requires at least one producer-fit group")
            if group_counts["calibration_fit"] < 32 or group_counts["calibration_allowance"] < 32:
                raise ValueError("production calibration roles require at least 32 independent groups")

    @property
    def digest(self) -> str:
        return canonical_digest(self, prefix="pfgr-lite-training-roles-v1|")

    def as_dict(self) -> dict[str, Any]:
        payload = {field.name: getattr(self, field.name) for field in fields(self)}
        payload["baseline_train_subject_ids"] = list(self.baseline_train_subject_ids)
        payload["baseline_validation_subject_ids"] = list(self.baseline_validation_subject_ids)
        payload["baseline_test_subject_ids"] = list(self.baseline_test_subject_ids)
        payload["producer_fit_subject_ids"] = list(self.producer_fit_subject_ids)
        payload["calibration_fit_subject_ids"] = list(self.calibration_fit_subject_ids)
        payload["calibration_allowance_subject_ids"] = list(self.calibration_allowance_subject_ids)
        payload["subject_group_ids"] = [list(row) for row in self.subject_group_ids]
        return payload

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TrainingRoleManifest":
        if not isinstance(values, Mapping):
            raise TypeError("training role manifest must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown training role manifest keys: {sorted(unknown)}")
        parsed = dict(values)
        for name in (
            "baseline_train_subject_ids",
            "baseline_validation_subject_ids",
            "baseline_test_subject_ids",
            "producer_fit_subject_ids",
            "calibration_fit_subject_ids",
            "calibration_allowance_subject_ids",
        ):
            parsed[name] = tuple(parsed.get(name, ()))
        parsed["subject_group_ids"] = tuple(tuple(row) for row in parsed.get("subject_group_ids", ()))
        return cls(**parsed)


def _finite(name: str, value: Tensor, rank: int | None = None, final: int | None = None) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if rank is not None and value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}")
    if final is not None and (value.ndim == 0 or value.shape[-1] != final):
        raise ValueError(f"{name} must have final dimension {final}")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if value.numel() == 0 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be nonempty and finite")


def _integer(name: str, value: Tensor, rank: int | None = None, final: int | None = None) -> None:
    if not isinstance(value, Tensor) or not value.dtype in (torch.bool, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError(f"{name} must be an integer tensor")
    if rank is not None and value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}")
    if final is not None and (value.ndim == 0 or value.shape[-1] != final):
        raise ValueError(f"{name} must have final dimension {final}")


def _nonempty_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value or value.lower() in {"unknown", "unset", "none", "null"}:
        raise ValueError(f"{name} must be a complete non-sentinel string")


def _strict_binary_mask(name: str, value: Tensor) -> Tensor:
    """Return an owned bool mask after rejecting non-binary floating values."""

    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype == torch.bool:
        return value.clone()
    if not value.is_floating_point() and value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError(f"{name} must be bool or numeric binary mask")
    if value.numel() == 0 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite and nonempty")
    if not bool(((value == 0) | (value == 1)).all()):
        raise ValueError(f"{name} must contain only exact binary 0/1 values")
    return value.to(dtype=torch.bool).clone()


def _same_device_dtype(name: str, values: Sequence[Tensor]) -> None:
    if not values:
        return
    if any(value.device != values[0].device for value in values) or any(value.dtype != values[0].dtype for value in values):
        raise ValueError(f"{name} tensors must share device and dtype")


def _digest_tensors(named: Sequence[tuple[str, Tensor]]) -> str:
    return canonical_digest([(name, tensor_digest(value, name=name)) for name, value in named], prefix="pfgr-lite-owned-v1|")


def dynamic_planes_digest(planes: DynamicTriPlanes) -> str:
    if not isinstance(planes, DynamicTriPlanes):
        raise TypeError("planes must be DynamicTriPlanes")
    return _digest_tensors((("xy", planes.xy), ("xz", planes.xz), ("yz", planes.yz)))


def clone_dynamic_planes(planes: DynamicTriPlanes) -> DynamicTriPlanes:
    """Clone state tensors while deliberately preserving their autograd graph."""

    if not isinstance(planes, DynamicTriPlanes):
        raise TypeError("planes must be DynamicTriPlanes")
    # Do not detach, call .data, inference_mode, or no_grad here.  ``clone``
    # creates owned writable tensors while preserving S0/S1 gradients.
    return DynamicTriPlanes(xy=planes.xy.clone(), xz=planes.xz.clone(), yz=planes.yz.clone())


@dataclass(frozen=True)
class DescriptorBundle:
    """Canonical descriptor storage rows reused by every V architecture."""

    v126: Tensor
    v222: Tensor
    v270: Tensor
    v366: Tensor
    schema_version: str = DESCRIPTOR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != DESCRIPTOR_SCHEMA:
            raise ValueError("unknown descriptor schema")
        expected = {"v126": 126, "v222": 222, "v270": 270, "v366": 366}
        for name, channels in expected.items():
            _finite(name, getattr(self, name), rank=3, final=channels)
        shapes = {getattr(self, name).shape[:2] for name in expected}
        if len(shapes) != 1:
            raise ValueError("descriptor rows must share [B,N] ordering")
        _same_device_dtype("descriptor tensors", tuple(getattr(self, name) for name in expected))


def build_descriptor_bundle(
    z96: Tensor,
    f_spec168: Tensor,
    semantic3: Tensor,
    reliability3: Tensor,
    q_bar24: Tensor,
    delta96: Tensor | None = None,
) -> DescriptorBundle:
    """Pack the locked V126/V222/V270/V366 rows from one observation action."""

    for name, value, channels in (("z96", z96, 96), ("f_spec168", f_spec168, 168), ("semantic3", semantic3, 3), ("reliability3", reliability3, 3), ("q_bar24", q_bar24, 24)):
        _finite(name, value, rank=3, final=channels)
    if not (z96.shape[:2] == f_spec168.shape[:2] == semantic3.shape[:2] == reliability3.shape[:2] == q_bar24.shape[:2]):
        raise ValueError("descriptor inputs must share [B,N] ordering")
    v126 = torch.cat((z96, semantic3, q_bar24, reliability3), dim=-1)
    v270 = torch.cat((z96, f_spec168, semantic3, reliability3), dim=-1)
    if delta96 is None:
        delta96 = torch.zeros_like(z96)
    _finite("delta96", delta96, rank=3, final=96)
    if delta96.shape[:2] != z96.shape[:2] or delta96.device != z96.device or delta96.dtype != z96.dtype:
        raise ValueError("delta96 must align with z96")
    return DescriptorBundle(v126=v126, v222=torch.cat((v126, delta96), dim=-1), v270=v270, v366=torch.cat((v270, delta96), dim=-1))


@dataclass(frozen=True)
class ProducerDependencies:
    """Compatibility envelope plus human-readable source provenance."""

    compatibility: ProducerCompatibility
    source_provenance: SourceProvenance
    observation_normalization: str = "pfgr-observation-normalization-v1"
    geometry_query_version: str = "pfgr-lite-static-geometry-v1"
    static_architecture: str = "b2_ordered_multiscale_v1"
    semantic_architecture: str = "medicalnet-resnet10-semantic-1x1-v1"
    point_architecture: str = "deterministic-points-refiner-v1"
    updater_architecture: str = "update-net-270-128-96-v1"
    decoder_architecture: str = "implicit-decoder-96-64-32-1-v1"
    writer_architecture: str = "compact-writeback-4mm-v1"
    candidate_geometry: str = "point-candidate-geometry-v1"
    label_definition: str = "signed-conditional-mean-masked-global-charbonnier-v1"
    config_version: str = "pfgr-lite-config-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.compatibility, ProducerCompatibility):
            raise TypeError("compatibility must be ProducerCompatibility")
        if not isinstance(self.source_provenance, SourceProvenance):
            raise TypeError("source_provenance must be SourceProvenance")
        for name in ("observation_normalization", "geometry_query_version", "static_architecture", "semantic_architecture", "point_architecture", "updater_architecture", "decoder_architecture", "writer_architecture", "candidate_geometry", "label_definition", "config_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        if any(token in self.label_definition.lower() for token in ("target", "oracle", "teacher")):
            raise ValueError("producer label_definition must remain target-free")

    @property
    def digest(self) -> str:
        return self.compatibility.digest

    @property
    def compatibility_hash(self) -> str:
        return self.compatibility.digest


@dataclass(frozen=True)
class ObservationContext:
    """Sealed target-free frontend context used by PFGR routing."""

    context_id: str
    frontend: FrontendOutput
    q_bar: Tensor
    feature_geometry: FeatureGridGeometry
    initial_planes: DynamicTriPlanes
    producer: ProducerDependencies
    observation_mask: Tensor | None = None
    mask_provenance: str = "none"
    descriptor_schema: str = DESCRIPTOR_SCHEMA
    version: str = PFGR_TYPES_SCHEMA
    _tensor_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.context_id, str) or not self.context_id:
            raise ValueError("context_id must be a nonempty string")
        if not isinstance(self.frontend, FrontendOutput):
            raise TypeError("frontend must be FrontendOutput")
        _finite("q_bar", self.q_bar, rank=3, final=24)
        if self.q_bar.shape[:2] != self.frontend.refined_points_ras_mm.shape[:2]:
            raise ValueError("q_bar must align with frontend points")
        if not isinstance(self.feature_geometry, FeatureGridGeometry):
            raise TypeError("feature_geometry must be FeatureGridGeometry")
        if not isinstance(self.initial_planes, DynamicTriPlanes):
            raise TypeError("initial_planes must be DynamicTriPlanes")
        if self.initial_planes.xy.shape[0] != self.frontend.s_coarse.shape[0]:
            raise ValueError("initial_planes batch must match frontend")
        depth, height, width = self.feature_geometry.shape_dhw
        expected = ((height, width), (depth, width), (depth, height))
        actual = (tuple(self.initial_planes.xy.shape[-2:]), tuple(self.initial_planes.xz.shape[-2:]), tuple(self.initial_planes.yz.shape[-2:]))
        if actual != expected:
            raise ValueError("initial_planes must retain the final PFGR feature lattice")
        if not isinstance(self.producer, ProducerDependencies):
            raise TypeError("producer must be ProducerDependencies")
        if self.descriptor_schema != DESCRIPTOR_SCHEMA or self.version != PFGR_TYPES_SCHEMA:
            raise ValueError("unknown ObservationContext schema/version")
        if not isinstance(self.mask_provenance, str) or not self.mask_provenance:
            raise ValueError("mask_provenance must be a nonempty observation-only declaration")
        if any(token in self.mask_provenance.lower() for token in ("target", "segmentation", "oracle", "teacher")):
            raise ValueError("target-derived mask provenance is forbidden in ObservationContext")
        expected = self.frontend.s_coarse.shape[0:1] + self.frontend.geometry.shape_dhw
        mask = self.observation_mask
        if mask is None:
            # ``None`` means the explicit all-voxel observation mask, not an
            # absent denominator.  Owning this resolved bool mask keeps later
            # teacher/bank joins deterministic and prevents a caller from
            # changing semantics after context construction.
            mask = torch.ones(expected, dtype=torch.bool, device=self.frontend.s_coarse.device)
            object.__setattr__(self, "mask_provenance", "all_voxels_observation_default")
        else:
            if not isinstance(mask, Tensor) or mask.ndim not in (4, 5):
                raise ValueError("observation_mask must be [B,D,H,W] or [B,1,D,H,W]")
            if mask.ndim == 5 and mask.shape[1] != 1:
                raise ValueError("observation_mask rank-5 form must have one channel")
            if tuple(mask.shape[0:1] + mask.shape[-3:]) != tuple(expected):
                raise ValueError("observation_mask must match observation geometry")
            mask = _strict_binary_mask("observation_mask", mask)
            if mask.ndim == 5:
                mask = mask[:, 0]
        if mask.shape != expected:
            raise ValueError("resolved observation_mask must have shape [B,D,H,W]")
        if mask.numel() == 0 or not bool(mask.any(dim=(-3, -2, -1)).all()):
            raise ValueError("observation_mask must contain at least one observed voxel per subject")
        object.__setattr__(self, "observation_mask", mask)
        owned = [("q_bar", self.q_bar), ("xy", self.initial_planes.xy), ("xz", self.initial_planes.xz), ("yz", self.initial_planes.yz)]
        owned.append(("observation_mask", mask))
        object.__setattr__(self, "_tensor_digest", _digest_tensors(owned))

    def validate_integrity(self) -> None:
        owned = [("q_bar", self.q_bar), ("xy", self.initial_planes.xy), ("xz", self.initial_planes.xz), ("yz", self.initial_planes.yz)]
        owned.append(("observation_mask", self.observation_mask))
        current = _digest_tensors(owned)
        if current != self._tensor_digest:
            raise RuntimeError("ObservationContext tensor mutation detected")

    @property
    def geometry(self) -> VolumeGeometry:
        return self.frontend.geometry

    @property
    def mask(self) -> Tensor:
        """Short alias used by route/teacher adapters for the observation mask."""

        return self.observation_mask

    @property
    def resolved_mask(self) -> Tensor:
        return self.observation_mask

    @property
    def resolved_observation_mask(self) -> Tensor:
        return self.observation_mask

    @property
    def source_provenance(self) -> SourceProvenance:
        """Exact checkpoint/adaptation evidence retained for this context."""

        return self.producer.source_provenance

    @property
    def z0(self) -> DynamicTriPlanes:
        return self.initial_planes


@dataclass(frozen=True)
class PFGRState:
    """Owned mutable-step dynamic state with context/producer guard."""

    planes: DynamicTriPlanes
    context_id: str
    state_version: int = 0
    state_digest: str | None = None
    producer: ProducerCompatibility | None = None
    role: Literal["deployment", "training_behavior"] = "training_behavior"
    _tensor_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.planes, DynamicTriPlanes):
            raise TypeError("planes must be DynamicTriPlanes")
        if not isinstance(self.context_id, str) or not self.context_id:
            raise ValueError("context_id must be nonempty")
        if not isinstance(self.state_version, int) or isinstance(self.state_version, bool) or self.state_version < 0:
            raise ValueError("state_version must be a nonnegative integer")
        if self.role not in ("deployment", "training_behavior"):
            raise ValueError("state role must be deployment or training_behavior")
        if self.producer is None:
            raise TypeError("producer is required; unbound state cannot enter PFGR production APIs")
        if not isinstance(self.producer, ProducerCompatibility):
            raise TypeError("producer must be ProducerCompatibility")
        if getattr(self.planes, "_oracle_state", False):
            raise TypeError("privileged oracle state cannot be laundered into PFGRState")
        digest = dynamic_planes_digest(self.planes)
        if self.state_digest is not None and self.state_digest != digest:
            raise ValueError("state_digest does not match owned plane tensors")
        object.__setattr__(self, "state_digest", digest)
        object.__setattr__(self, "_tensor_digest", digest)

    def validate_integrity(self) -> None:
        digest = dynamic_planes_digest(self.planes)
        if digest != self._tensor_digest or digest != self.state_digest:
            raise RuntimeError("PFGRState tensor mutation detected")

    def next(self, planes: DynamicTriPlanes) -> "PFGRState":
        self.validate_integrity()
        return PFGRState(planes=planes, context_id=self.context_id, state_version=self.state_version + 1, producer=self.producer, role=self.role)

    @property
    def dynamic_planes(self) -> DynamicTriPlanes:
        return self.planes

    @property
    def version(self) -> int:
        return self.state_version


@dataclass(frozen=True)
class ActionProposal:
    """One immutable action row; execution must use its stored ``delta``."""

    context_id: str
    context_version: str
    producer_compatibility_hash: str
    state_version: int
    state_digest: str
    point_id: int
    point_ras_mm: Tensor
    o270: Tensor
    v126: Tensor
    delta: Tensor
    legal: bool
    updater_version: str
    updater_producer_hash: str
    writer_version: str
    writer_hash: str
    query_version: str
    query_hash: str
    geometry_version: str
    geometry_hash: str
    point_version: str
    point_identity_hash: str
    action_id: str
    action_digest: str | None = None
    _tensor_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _nonempty_text("context_id", self.context_id)
        if self.context_version != PFGR_TYPES_SCHEMA:
            raise ValueError("unknown context contract version")
        _nonempty_text("producer_compatibility_hash", self.producer_compatibility_hash)
        if not isinstance(self.state_version, int) or isinstance(self.state_version, bool) or self.state_version < 0:
            raise ValueError("state_version must be a nonnegative integer")
        _nonempty_text("state_digest", self.state_digest)
        if not isinstance(self.point_id, int) or isinstance(self.point_id, bool) or self.point_id < 0:
            raise ValueError("point_id must be a nonnegative integer")
        _finite("point_ras_mm", self.point_ras_mm, rank=1, final=3)
        _finite("o270", self.o270, rank=1, final=270)
        _finite("v126", self.v126, rank=1, final=126)
        _finite("delta", self.delta, rank=1, final=96)
        if self.point_ras_mm.device != self.delta.device or self.o270.device != self.delta.device or self.v126.device != self.delta.device:
            raise ValueError("proposal tensors must share device")
        if self.delta.dtype != self.o270.dtype or self.delta.dtype != self.v126.dtype or self.delta.dtype != self.point_ras_mm.dtype:
            raise ValueError("proposal tensors must share dtype")
        if not isinstance(self.legal, bool):
            raise TypeError("legal must be bool")
        for name in (
            "updater_version",
            "updater_producer_hash",
            "writer_version",
            "writer_hash",
            "query_version",
            "query_hash",
            "geometry_version",
            "geometry_hash",
            "point_version",
            "point_identity_hash",
            "action_id",
        ):
            _nonempty_text(name, getattr(self, name))
        digest = canonical_digest(
            {
                "context_id": self.context_id,
                "context_version": self.context_version,
                "producer_compatibility_hash": self.producer_compatibility_hash,
                "state_version": self.state_version,
                "state_digest": self.state_digest,
                "point_id": self.point_id,
                "point_ras_mm": tensor_digest(self.point_ras_mm, name="point_ras_mm"),
                "o270": tensor_digest(self.o270, name="o270"),
                "v126": tensor_digest(self.v126, name="v126"),
                "delta": tensor_digest(self.delta, name="delta"),
                "legal": self.legal,
                "updater_version": self.updater_version,
                "updater_producer_hash": self.updater_producer_hash,
                "writer_version": self.writer_version,
                "writer_hash": self.writer_hash,
                "query_version": self.query_version,
                "query_hash": self.query_hash,
                "geometry_version": self.geometry_version,
                "geometry_hash": self.geometry_hash,
                "point_version": self.point_version,
                "point_identity_hash": self.point_identity_hash,
                "action_id": self.action_id,
            },
            prefix="pfgr-lite-action-v1|",
        )
        if self.action_digest is not None and self.action_digest != digest:
            raise ValueError("action_digest does not match complete action identity")
        object.__setattr__(self, "action_digest", digest)
        object.__setattr__(self, "_tensor_digest", digest)

    def validate_integrity(self) -> None:
        digest = canonical_digest(
            {
                "context_id": self.context_id,
                "context_version": self.context_version,
                "producer_compatibility_hash": self.producer_compatibility_hash,
                "state_version": self.state_version,
                "state_digest": self.state_digest,
                "point_id": self.point_id,
                "point_ras_mm": tensor_digest(self.point_ras_mm, name="point_ras_mm"),
                "o270": tensor_digest(self.o270, name="o270"),
                "v126": tensor_digest(self.v126, name="v126"),
                "delta": tensor_digest(self.delta, name="delta"),
                "legal": self.legal,
                "updater_version": self.updater_version,
                "updater_producer_hash": self.updater_producer_hash,
                "writer_version": self.writer_version,
                "writer_hash": self.writer_hash,
                "query_version": self.query_version,
                "query_hash": self.query_hash,
                "geometry_version": self.geometry_version,
                "geometry_hash": self.geometry_hash,
                "point_version": self.point_version,
                "point_identity_hash": self.point_identity_hash,
                "action_id": self.action_id,
            },
            prefix="pfgr-lite-action-v1|",
        )
        if digest != self._tensor_digest or digest != self.action_digest:
            raise RuntimeError("ActionProposal mutation detected")

    @property
    def actual_delta(self) -> Tensor:
        return self.delta

    @property
    def stored_delta(self) -> Tensor:
        return self.delta

    @property
    def u_producer_hash(self) -> str:
        return self.updater_producer_hash

    @property
    def producer_hash(self) -> str:
        return self.updater_producer_hash

    @property
    def writer_identity_hash(self) -> str:
        return self.writer_hash

    @property
    def query_identity_hash(self) -> str:
        return self.query_hash

    @property
    def point_identity(self) -> str:
        return self.point_identity_hash


@dataclass(frozen=True)
class ActionProposalBatch:
    """Ordered batched U outputs and proposal identity metadata."""

    context_id: str
    state_version: int
    state_digest: str
    point_ids: Tensor
    points_ras_mm: Tensor
    o270: Tensor
    v126: Tensor
    delta: Tensor
    legal: Tensor
    context_version: str = PFGR_TYPES_SCHEMA
    producer_compatibility_hash: str = ""
    updater_version: str = "update-net-270-128-96-v1"
    updater_producer_hash: str = ""
    writer_version: str = "compact-writeback-4mm-v1"
    writer_hash: str = ""
    query_version: str = "pfgr-lite-query-lattice-v1"
    query_hash: str = ""
    geometry_version: str = "pfgr-lite-static-geometry-v1"
    geometry_hash: str = ""
    point_version: str = "point-candidate-geometry-v1"
    point_identity_hash: str = ""
    proposal_digest: str | None = None
    version: str = ACTION_SCHEMA
    _tensor_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.version != ACTION_SCHEMA:
            raise ValueError("unknown ActionProposalBatch schema")
        _nonempty_text("context_id", self.context_id)
        if self.context_version != PFGR_TYPES_SCHEMA:
            raise ValueError("unknown context contract version")
        _nonempty_text("producer_compatibility_hash", self.producer_compatibility_hash)
        if not isinstance(self.state_version, int) or isinstance(self.state_version, bool) or self.state_version < 0:
            raise ValueError("state_version must be nonnegative")
        _nonempty_text("state_digest", self.state_digest)
        for name, value, final in (("point_ids", self.point_ids, None), ("legal", self.legal, None)):
            _integer(name, value, rank=2)
        _finite("points_ras_mm", self.points_ras_mm, rank=3, final=3)
        _finite("o270", self.o270, rank=3, final=270)
        _finite("v126", self.v126, rank=3, final=126)
        _finite("delta", self.delta, rank=3, final=96)
        if self.legal.dtype != torch.bool:
            # Strictly permit only explicit bool/integer 0-1 masks; convert is
            # not performed because proposal identity must not change.
            if self.legal.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise TypeError("legal must be bool or integer mask")
            if not bool(((self.legal == 0) | (self.legal == 1)).all()):
                raise ValueError("legal integer mask must contain only 0/1")
        batch, count = self.point_ids.shape
        if batch <= 0 or count <= 0:
            raise ValueError("proposal tensors must have nonempty [B,N] dimensions")
        if self.point_ids.dtype == torch.bool:
            raise TypeError("point_ids cannot be bool")
        if bool((self.point_ids < 0).any()):
            raise ValueError("point_ids must be nonnegative")
        flat_ids = self.point_ids.reshape(-1).tolist()
        if len(set(flat_ids)) != len(flat_ids):
            raise ValueError("point_ids must be unique within a proposal batch")
        if self.points_ras_mm.shape[:2] != (batch, count) or self.o270.shape[:2] != (batch, count) or self.v126.shape[:2] != (batch, count) or self.delta.shape[:2] != (batch, count) or self.legal.shape != (batch, count):
            raise ValueError("proposal tensors must share [B,N] ordering")
        _same_device_dtype("proposal floating tensors", (self.points_ras_mm, self.o270, self.v126, self.delta))
        if self.point_ids.device != self.delta.device or self.legal.device != self.delta.device:
            raise ValueError("proposal tensors must share device")
        for name in ("updater_version", "updater_producer_hash", "writer_version", "writer_hash", "query_version", "query_hash", "geometry_version", "geometry_hash", "point_version", "point_identity_hash"):
            _nonempty_text(name, getattr(self, name))
        metadata = {
            "version": self.version,
            "context_id": self.context_id,
            "context_version": self.context_version,
            "producer_compatibility_hash": self.producer_compatibility_hash,
            "state_version": self.state_version,
            "state_digest": self.state_digest,
            "updater_version": self.updater_version,
            "updater_producer_hash": self.updater_producer_hash,
            "writer_version": self.writer_version,
            "writer_hash": self.writer_hash,
            "query_version": self.query_version,
            "query_hash": self.query_hash,
            "geometry_version": self.geometry_version,
            "geometry_hash": self.geometry_hash,
            "point_version": self.point_version,
            "point_identity_hash": self.point_identity_hash,
        }
        digest = canonical_digest(
            {
                "metadata": metadata,
                "point_ids": tensor_digest(self.point_ids, name="point_ids"),
                "points": tensor_digest(self.points_ras_mm, name="points"),
                "o270": tensor_digest(self.o270, name="o270"),
                "v126": tensor_digest(self.v126, name="v126"),
                "delta": tensor_digest(self.delta, name="delta"),
                "legal": tensor_digest(self.legal, name="legal"),
            },
            prefix="pfgr-lite-action-batch-v1|",
        )
        if self.proposal_digest is not None and self.proposal_digest != digest:
            raise ValueError("proposal_digest does not match tensors")
        object.__setattr__(self, "proposal_digest", digest)
        object.__setattr__(self, "_tensor_digest", digest)

    def validate_integrity(self) -> None:
        digest = canonical_digest(
            {
                "metadata": {
                    "version": self.version,
                    "context_id": self.context_id,
                    "context_version": self.context_version,
                    "producer_compatibility_hash": self.producer_compatibility_hash,
                    "state_version": self.state_version,
                    "state_digest": self.state_digest,
                    "updater_version": self.updater_version,
                    "updater_producer_hash": self.updater_producer_hash,
                    "writer_version": self.writer_version,
                    "writer_hash": self.writer_hash,
                    "query_version": self.query_version,
                    "query_hash": self.query_hash,
                    "geometry_version": self.geometry_version,
                    "geometry_hash": self.geometry_hash,
                    "point_version": self.point_version,
                    "point_identity_hash": self.point_identity_hash,
                },
                "point_ids": tensor_digest(self.point_ids, name="point_ids"),
                "points": tensor_digest(self.points_ras_mm, name="points"),
                "o270": tensor_digest(self.o270, name="o270"),
                "v126": tensor_digest(self.v126, name="v126"),
                "delta": tensor_digest(self.delta, name="delta"),
                "legal": tensor_digest(self.legal, name="legal"),
            },
            prefix="pfgr-lite-action-batch-v1|",
        )
        if digest != self._tensor_digest or digest != self.proposal_digest:
            raise RuntimeError("ActionProposalBatch mutation detected")

    def row(self, batch_index: int, point_index: int) -> ActionProposal:
        self.validate_integrity()
        if not (0 <= batch_index < self.point_ids.shape[0] and 0 <= point_index < self.point_ids.shape[1]):
            raise IndexError("proposal row out of bounds")
        return ActionProposal(
            context_id=self.context_id,
            context_version=self.context_version,
            producer_compatibility_hash=self.producer_compatibility_hash,
            state_version=self.state_version,
            state_digest=self.state_digest,
            point_id=int(self.point_ids[batch_index, point_index].item()),
            point_ras_mm=self.points_ras_mm[batch_index, point_index],
            o270=self.o270[batch_index, point_index],
            v126=self.v126[batch_index, point_index],
            delta=self.delta[batch_index, point_index],
            legal=bool(self.legal[batch_index, point_index].item()),
            updater_version=self.updater_version,
            updater_producer_hash=self.updater_producer_hash,
            writer_version=self.writer_version,
            writer_hash=self.writer_hash,
            query_version=self.query_version,
            query_hash=self.query_hash,
            geometry_version=self.geometry_version,
            geometry_hash=self.geometry_hash,
            point_version=self.point_version,
            point_identity_hash=canonical_digest(
                {
                    "batch_point_identity": self.point_identity_hash,
                    "point_id": int(self.point_ids[batch_index, point_index].item()),
                    "point": tensor_digest(self.points_ras_mm[batch_index, point_index], name="point_ras_mm"),
                },
                prefix="pfgr-lite-point-identity-v1|",
            ),
            action_id=canonical_digest(
                {
                    "context_id": self.context_id,
                    "context_version": self.context_version,
                    "producer_compatibility_hash": self.producer_compatibility_hash,
                    "state_version": self.state_version,
                    "state_digest": self.state_digest,
                    "point_id": int(self.point_ids[batch_index, point_index].item()),
                    "point": tensor_digest(self.points_ras_mm[batch_index, point_index], name="point_ras_mm"),
                    "o270": tensor_digest(self.o270[batch_index, point_index], name="o270"),
                    "v126": tensor_digest(self.v126[batch_index, point_index], name="v126"),
                    "delta": tensor_digest(self.delta[batch_index, point_index], name="delta"),
                    "updater": self.updater_producer_hash,
                    "updater_version": self.updater_version,
                    "writer": self.writer_hash,
                    "writer_version": self.writer_version,
                    "query": self.query_hash,
                    "query_version": self.query_version,
                    "geometry": self.geometry_hash,
                    "geometry_version": self.geometry_version,
                    "point_version": self.point_version,
                    "point_identity": self.point_identity_hash,
                },
                prefix="pfgr-lite-action-id-v1|",
            ),
        )

    @property
    def actual_delta(self) -> Tensor:
        return self.delta

    @property
    def stored_delta(self) -> Tensor:
        return self.delta

    @property
    def u_producer_hash(self) -> str:
        return self.updater_producer_hash

    @property
    def producer_hash(self) -> str:
        return self.updater_producer_hash

    @property
    def writer_identity_hash(self) -> str:
        return self.writer_hash

    @property
    def query_identity_hash(self) -> str:
        return self.query_hash

    @property
    def point_identity(self) -> str:
        return self.point_identity_hash

    @property
    def point_positions_ras_mm(self) -> Tensor:
        return self.points_ras_mm


@dataclass(frozen=True)
class Decision:
    """Target-free route decision; labels/targets are intentionally absent."""

    selected_point_id: int = -1
    # These identities bind the decision to the exact dense scoring pass.  A
    # stopping decision may leave ``action_digest`` empty because no action is
    # executed, but it must still identify the proposal batch whenever one was
    # scored.  A continuing decision must carry both fields and the action
    # digest must match the selected stored proposal row.
    proposal_digest: str = ""
    action_digest: str = ""
    active: bool = True
    raw_value: float = 0.0
    calibrated_value: float = 0.0
    conservative_value: float = 0.0
    allowance: float = 0.0
    quality_margin: float = 0.0
    compute_cost: float = 0.0
    policy_hash: str = ""
    stop_code: Literal["continue", "budget", "low_gain", "no_legal_action"] = "continue"
    step: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.selected_point_id, int) or isinstance(self.selected_point_id, bool) or self.selected_point_id < -1:
            raise ValueError("selected_point_id must be -1 or nonnegative")
        if not isinstance(self.active, bool):
            raise TypeError("active must be bool")
        if self.proposal_digest:
            _nonempty_text("proposal_digest", self.proposal_digest)
        if self.action_digest:
            _nonempty_text("action_digest", self.action_digest)
        for name in ("raw_value", "calibrated_value", "conservative_value", "allowance", "quality_margin", "compute_cost"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.allowance < 0.0 or self.quality_margin < 0.0 or self.compute_cost < 0.0:
            raise ValueError("allowance, quality_margin, and compute_cost must be nonnegative")
        _nonempty_text("policy_hash", self.policy_hash)
        if self.stop_code not in ("continue", "budget", "low_gain", "no_legal_action"):
            raise ValueError("unknown stop_code")
        if not isinstance(self.step, int) or self.step < 0:
            raise ValueError("step must be nonnegative")


@dataclass(frozen=True)
class CompletedBehaviorTrace:
    """Sealed target-free route to which supervision may be joined later."""

    context_id: str
    states: tuple[PFGRState, ...] = ()
    proposals: tuple[ActionProposalBatch, ...] = ()
    decisions: tuple[Decision, ...] = ()
    route_hash: str = ""
    version: str = TRACE_SCHEMA
    sealed: bool = True

    def __post_init__(self) -> None:
        if self.version != TRACE_SCHEMA:
            raise ValueError("unknown behavior trace schema")
        if not isinstance(self.context_id, str) or not self.context_id:
            raise ValueError("context_id must be nonempty")
        if not self.sealed:
            raise ValueError("only sealed target-free traces may be shared")
        if not self.states:
            raise ValueError("behavior trace must contain its initial PFGRState")
        if len(self.proposals) != len(self.decisions):
            raise ValueError("proposal/decision trace lengths must match")
        for index, state in enumerate(self.states):
            if not isinstance(state, PFGRState) or state.context_id != self.context_id:
                raise ValueError("trace states must be PFGRState rows for its context")
            if state.state_version != index:
                raise ValueError("trace state versions must form a contiguous chain beginning at version 0")
            state.validate_integrity()
            if index and state.producer.digest != self.states[0].producer.digest:
                raise ValueError("trace state producer identities must remain constant")
        if len(self.states) != len(self.proposals) + 1:
            raise ValueError("trace requires one initial state plus one state per proposal batch")
        for index, proposal in enumerate(self.proposals):
            if not isinstance(proposal, ActionProposalBatch) or proposal.context_id != self.context_id:
                raise ValueError("trace proposals must be ActionProposalBatch rows for its context")
            proposal.validate_integrity()
            state = self.states[index]
            if proposal.state_version != state.state_version or proposal.state_digest != state.state_digest:
                raise ValueError("proposal state identity does not match trace state chain")
            decision = self.decisions[index]
            if decision.step != index:
                raise ValueError("trace decision steps must match their state-transition index")
            if decision.stop_code == "continue":
                if not decision.active:
                    raise ValueError("continue decision must be active")
                if decision.selected_point_id < 0 or decision.selected_point_id not in proposal.point_ids.reshape(-1).tolist():
                    raise ValueError("continue decision must select a proposal point")
            else:
                if decision.active:
                    raise ValueError("stopping decision must be inactive")
                if decision.selected_point_id >= 0:
                    raise ValueError("stopping decision cannot carry a selected point")
            if decision.proposal_digest != proposal.proposal_digest:
                raise ValueError("decision proposal_digest does not match scored proposal batch")
            if decision.selected_point_id >= 0:
                locations = (proposal.point_ids == decision.selected_point_id).nonzero(as_tuple=False)
                if locations.shape[0] != 1:
                    raise ValueError("decision selected point is not uniquely present in scored proposals")
                expected_action = proposal.row(int(locations[0, 0]), int(locations[0, 1])).action_digest
                if decision.action_digest != expected_action:
                    raise ValueError("decision action_digest does not match selected stored action")
            elif decision.action_digest:
                action_digests = {
                    proposal.row(batch_index, point_index).action_digest
                    for batch_index in range(proposal.point_ids.shape[0])
                    for point_index in range(proposal.point_ids.shape[1])
                }
                if decision.action_digest not in action_digests:
                    raise ValueError("stopping decision action_digest is not from scored proposals")
        digest_payload = {
            "context_id": self.context_id,
            "states": [(state.state_version, state.state_digest, state.producer.digest) for state in self.states],
            "proposals": [proposal.proposal_digest for proposal in self.proposals],
            "decisions": [
                {
                    "selected_point_id": decision.selected_point_id,
                    "proposal_digest": decision.proposal_digest,
                    "action_digest": decision.action_digest,
                    "active": decision.active,
                    "raw_value": decision.raw_value,
                    "calibrated_value": decision.calibrated_value,
                    "conservative_value": decision.conservative_value,
                    "allowance": decision.allowance,
                    "quality_margin": decision.quality_margin,
                    "compute_cost": decision.compute_cost,
                    "policy_hash": decision.policy_hash,
                    "stop_code": decision.stop_code,
                    "step": decision.step,
                }
                for decision in self.decisions
            ],
        }
        expected_hash = canonical_digest(digest_payload, prefix="pfgr-lite-trace-v1|")
        if self.route_hash and self.route_hash != expected_hash:
            raise ValueError("route_hash does not match complete behavior trace identity")
        object.__setattr__(self, "route_hash", expected_hash)


@dataclass(frozen=True)
class ParallelBehaviorTrace:
    """Frozen initial-bank diagnostic trace for ``parallel_topk`` controls."""

    context_id: str
    initial_state: PFGRState
    proposals: ActionProposalBatch
    selected_action_ids: tuple[str, ...]
    selected_action_digests: tuple[str, ...]
    selected_delta_digests: tuple[str, ...]
    intermediate_states: tuple[PFGRState, ...]
    policy_hash: str
    trace_hash: str = ""
    version: str = "pfgr-lite-parallel-trace-v1"

    def __post_init__(self) -> None:
        if self.version != "pfgr-lite-parallel-trace-v1":
            raise ValueError("unknown parallel behavior trace schema")
        _nonempty_text("context_id", self.context_id)
        _nonempty_text("policy_hash", self.policy_hash)
        if not isinstance(self.initial_state, PFGRState) or self.initial_state.context_id != self.context_id or self.initial_state.state_version != 0:
            raise ValueError("parallel trace initial state must be context-bound version zero")
        self.initial_state.validate_integrity()
        if not isinstance(self.proposals, ActionProposalBatch) or self.proposals.context_id != self.context_id:
            raise ValueError("parallel trace proposals must share the initial context")
        self.proposals.validate_integrity()
        if self.proposals.state_version != self.initial_state.state_version or self.proposals.state_digest != self.initial_state.state_digest:
            raise ValueError("parallel proposal bank must be bound to the initial state")
        count = len(self.selected_action_ids)
        if count <= 0 or count > 4:
            raise ValueError("parallel trace must contain between one and four selected actions")
        if not (len(self.selected_action_digests) == len(self.selected_delta_digests) == len(self.intermediate_states) == count):
            raise ValueError("parallel trace selected identities and states must have equal lengths")
        if len(set(self.selected_action_ids)) != count or len(set(self.selected_action_digests)) != count:
            raise ValueError("parallel selected action IDs/digests must be unique")
        rows = {
            action.action_id: action
            for batch_index in range(self.proposals.point_ids.shape[0])
            for point_index in range(self.proposals.point_ids.shape[1])
            for action in (self.proposals.row(batch_index, point_index),)
        }
        if any(action_id not in rows for action_id in self.selected_action_ids):
            raise ValueError("parallel selected action IDs must come from the initial proposal bank")
        for index, (action_id, action_digest, delta_digest, state) in enumerate(zip(self.selected_action_ids, self.selected_action_digests, self.selected_delta_digests, self.intermediate_states), start=1):
            action = rows[action_id]
            if not action.legal:
                raise ValueError("parallel selected action must be legal in the initial proposal bank")
            if action.action_digest != action_digest:
                raise ValueError("parallel action digest is not bound to the initial proposal row")
            if tensor_digest(action.delta, name="delta") != delta_digest:
                raise ValueError("parallel selected delta provenance does not match stored action")
            if not isinstance(state, PFGRState) or state.context_id != self.context_id or state.state_version != index:
                raise ValueError("parallel intermediate states must form a contiguous context-bound chain")
            state.validate_integrity()
            if state.producer.digest != self.initial_state.producer.digest:
                raise ValueError("parallel producer identity changed during compound write")
        expected = canonical_digest(
            {
                "version": self.version,
                "context_id": self.context_id,
                "initial_state": (self.initial_state.state_version, self.initial_state.state_digest, self.initial_state.producer.digest),
                "proposal_digest": self.proposals.proposal_digest,
                "selected_action_ids": self.selected_action_ids,
                "selected_action_digests": self.selected_action_digests,
                "selected_delta_digests": self.selected_delta_digests,
                "intermediate_states": [(state.state_version, state.state_digest) for state in self.intermediate_states],
                "policy_hash": self.policy_hash,
            },
            prefix="pfgr-lite-parallel-trace-v1|",
        )
        if self.trace_hash and self.trace_hash != expected:
            raise ValueError("parallel trace hash does not match complete diagnostic identity")
        object.__setattr__(self, "trace_hash", expected)


@dataclass(frozen=True)
class SparseFootprint:
    """Structural positive writer support (target-independent)."""

    voxel_ids_dhw: Tensor
    multiplicity: Tensor | None = None
    plane_counts: tuple[int, int, int] = (0, 0, 0)
    support_pairs: tuple[Tensor, ...] = ()
    lattice_version: str = "pfgr-lite-query-lattice-v1"
    geometry_hash: str = "unknown"
    kernel_version: str = "quadratic_compact_4mm_v1"
    mode: Literal["indexed", "full_scan_fallback"] = "indexed"
    scanned_voxel_count: int = 0

    def __post_init__(self) -> None:
        _integer("voxel_ids_dhw", self.voxel_ids_dhw, rank=2, final=3)
        if self.voxel_ids_dhw.shape[0] <= 0:
            raise ValueError("SparseFootprint requires at least one positive-support voxel")
        if self.multiplicity is not None:
            _integer("multiplicity", self.multiplicity, rank=1)
            if self.multiplicity.shape[0] != self.voxel_ids_dhw.shape[0] or bool((self.multiplicity <= 0).any()):
                raise ValueError("multiplicity must be positive and align with voxel_ids_dhw")
        if len(self.plane_counts) != 3 or any(not isinstance(value, int) or value < 0 for value in self.plane_counts):
            raise ValueError("plane_counts must contain three nonnegative integers")
        if self.mode not in ("indexed", "full_scan_fallback"):
            raise ValueError("unknown footprint mode")
        if self.scanned_voxel_count < 0:
            raise ValueError("scanned_voxel_count must be nonnegative")

    @property
    def union_size(self) -> int:
        return int(self.voxel_ids_dhw.shape[0])


@dataclass(frozen=True)
class GainLabel:
    """Signed measured effect label and estimator metadata."""

    action_id: str
    context_id: str
    state_version: int
    raw_gain: float
    benefit: float
    harm: float
    mask_count: int
    role: Literal["exact_footprint", "iid_fixed_q", "screening"] = "exact_footprint"
    q_draws: int = 0
    seed: int | None = None
    variance: float | None = None
    standard_error: float | None = None
    footprint_voxels: int = 0
    valid_masked_contributions: int = 0
    sampler_law: str = "exact"
    label_definition: str = "signed-conditional-mean-masked-global-charbonnier-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.action_id, str) or not self.action_id or not isinstance(self.context_id, str) or not self.context_id:
            raise ValueError("action_id/context_id must be nonempty")
        if not isinstance(self.state_version, int) or self.state_version < 0:
            raise ValueError("state_version must be nonnegative")
        for name in ("raw_gain", "benefit", "harm"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if not math.isclose(self.raw_gain, self.benefit - self.harm, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("raw_gain must equal benefit-harm")
        if self.mask_count <= 0 or self.footprint_voxels < 0 or self.valid_masked_contributions < 0:
            raise ValueError("mask_count must be positive and counts nonnegative")
        if self.role not in ("exact_footprint", "iid_fixed_q", "screening"):
            raise ValueError("unknown GainLabel role")
        if self.role == "iid_fixed_q" and self.q_draws < 2:
            raise ValueError("iid_fixed_q labels require q_draws >= 2")
        if self.role == "exact_footprint" and self.variance not in (None, 0.0):
            raise ValueError("exact footprint labels have zero sampling variance")
        for name in ("variance", "standard_error"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(f"{name} must be nonnegative finite or None")


@dataclass(frozen=True)
class GainCalibration:
    """Positive affine score calibration plus pooled empirical allowance."""

    a: float
    b: float
    allowance: float
    quantile: float = 0.90
    quantile_method: str = "higher"
    fit_role_hash: str = ""
    allowance_role_hash: str = ""
    fit_count: int = 0
    allowance_count: int = 0
    producer_compatibility_hash: str = ""
    value_fit_identity_hash: str = ""
    gain_scale_hash: str = ""
    version: str = "pfgr-lite-calibration-v1"
    capability: Literal["diagnostic", "adaptive"] = "diagnostic"

    def __post_init__(self) -> None:
        if self.version != "pfgr-lite-calibration-v1":
            raise ValueError("unknown calibration version")
        if not math.isfinite(float(self.a)) or self.a < 1e-6:
            raise ValueError("calibration slope a must be finite and >=1e-6")
        for name in ("b", "allowance"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.allowance < 0.0:
            raise ValueError("allowance must be nonnegative")
        if not 0.0 < float(self.quantile) < 1.0 or self.quantile_method != "higher":
            raise ValueError("calibration quantile is locked to empirical higher order statistic")
        if self.fit_count < 0 or self.allowance_count < 0:
            raise ValueError("calibration counts must be nonnegative")
        if self.capability not in ("diagnostic", "adaptive"):
            raise ValueError("unknown calibration capability")
        if self.capability == "adaptive":
            for name in ("producer_compatibility_hash", "value_fit_identity_hash", "gain_scale_hash"):
                _nonempty_text(name, getattr(self, name))


@dataclass
class OperationCounters:
    """Explicit operation accounting; all counts are additive integers."""

    schema_version: str = COUNTERS_SCHEMA
    medicalnet_traversals: int = 0
    candidate_proposals: int = 0
    value_evaluations: int = 0
    candidate_evaluations: int = 0
    eligible_candidate_evaluations: int = 0
    executed_writes: int = 0
    behavior_states: int = 0
    states_labeled: int = 0
    candidate_labels: int = 0
    exact_sphere_valid_voxels: int = 0
    padded_cube_slots: int = 0
    footprint_unique_voxels: int = 0
    sampled_draws: int = 0
    unique_decoded_queries: int = 0
    before_decoder_outputs: int = 0
    after_decoder_outputs: int = 0
    decoder_calls: int = 0
    mlp_calls: int = 0
    dense_decodes: int = 0
    target_validations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    geometry_cache_builds: int = 0
    full_plane_clone_bytes: int = 0
    bytes_copied: int = 0

    def __post_init__(self) -> None:
        if self.schema_version != COUNTERS_SCHEMA:
            raise ValueError("unknown OperationCounters schema")
        for field_name in self.__dataclass_fields__:
            if field_name == "schema_version":
                continue
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a nonnegative integer")

    def add(self, **increments: int) -> None:
        for name, increment in increments.items():
            if name not in self.__dataclass_fields__ or name == "schema_version":
                raise KeyError(name)
            if not isinstance(increment, int) or isinstance(increment, bool) or increment < 0:
                raise ValueError("counter increments must be nonnegative integers")
            setattr(self, name, getattr(self, name) + increment)

    def as_dict(self) -> dict[str, int | str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class StageState:
    stage: str
    substage: str = "complete"
    epoch: int = 0
    update: int = 0
    microstep: int = 0
    optimizer_groups: tuple[str, ...] = ()
    completion: str = "complete"
    version: str = "pfgr-lite-stage-state-v1"

    def __post_init__(self) -> None:
        if self.version != "pfgr-lite-stage-state-v1":
            raise ValueError("unknown StageState version")
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("stage must be nonempty")
        for name in ("epoch", "update", "microstep"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.completion not in ("pending", "complete"):
            raise ValueError("completion must be pending or complete")


@dataclass(frozen=True)
class InferenceBundle:
    """Clean target-free inference package metadata and state dictionary."""

    state_dict: Mapping[str, Tensor]
    producer: ProducerDependencies
    config: Mapping[str, Any]
    capability: Literal["static", "forced_diagnostic", "adaptive"] = "static"
    calibration: GainCalibration | None = None
    split_hash: str | None = None
    schema_version: str = INFERENCE_SCHEMA
    frontend_config: Mapping[str, Any] | None = None
    value_fit_identity: ValueFitIdentity | None = None
    gain_scale_hash: str = ""
    effective_policy_hash: str = ""
    role_manifest: TrainingRoleManifest | None = None
    stage_provenance: Mapping[str, Any] | None = None
    calibration_evidence: Mapping[str, Any] | None = None
    effective_policy: Mapping[str, Any] | None = None
    gain_scale_provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != INFERENCE_SCHEMA:
            raise ValueError("unknown inference schema")
        if not isinstance(self.producer, ProducerDependencies):
            raise TypeError("producer must be ProducerDependencies")
        if self.capability not in ("static", "forced_diagnostic", "adaptive"):
            raise ValueError("unknown inference capability")
        if self.capability == "adaptive" and self.calibration is None:
            raise ValueError("adaptive inference requires calibration")
        for name, value in self.state_dict.items():
            if not isinstance(name, str) or not isinstance(value, Tensor):
                raise TypeError("state_dict must map names to tensors")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError("state_dict tensors must be finite")
        if any(name.lower().find("target") >= 0 or name.lower().find("oracle") >= 0 for name in self.state_dict):
            raise ValueError("inference bundle cannot contain target/oracle state")
        if self.capability == "adaptive":
            if self.calibration is None or any(
                not isinstance(getattr(self.calibration, name), str)
                or getattr(self.calibration, name).lower() in {"unknown", "unset", "none", "null"}
                for name in ("producer_compatibility_hash", "value_fit_identity_hash", "gain_scale_hash")
            ):
                raise ValueError("adaptive inference requires complete calibration identities")
            if self.frontend_config is None or self.value_fit_identity is None or not self.gain_scale_hash or not self.effective_policy_hash or self.role_manifest is None or self.calibration_evidence is None or self.effective_policy is None or self.gain_scale_provenance is None:
                raise ValueError("adaptive inference requires complete frontend/value/scale/policy/role evidence")
        if self.value_fit_identity is not None and not isinstance(self.value_fit_identity, ValueFitIdentity):
            raise TypeError("value_fit_identity must be ValueFitIdentity or None")
        if self.role_manifest is not None and not isinstance(self.role_manifest, TrainingRoleManifest):
            raise TypeError("role_manifest must be TrainingRoleManifest or None")
        for name in ("frontend_config", "stage_provenance", "calibration_evidence", "effective_policy", "gain_scale_provenance"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping or None")
        for name in ("gain_scale_hash", "effective_policy_hash"):
            value = getattr(self, name)
            if value and (not isinstance(value, str) or value.lower() in {"unknown", "unset", "none", "null"}):
                raise ValueError(f"{name} must be a complete non-sentinel string")


@dataclass(frozen=True)
class ResumeState:
    inference: InferenceBundle
    stage_state: StageState
    optimizer_state: Mapping[str, Any]
    rng_state: Mapping[str, Any]
    bank_state: Mapping[str, Any]
    protocol: str = RESUME_SCHEMA

    def __post_init__(self) -> None:
        if self.protocol != RESUME_SCHEMA:
            raise ValueError("unknown resume schema")
        if not isinstance(self.inference, InferenceBundle) or not isinstance(self.stage_state, StageState):
            raise TypeError("resume requires InferenceBundle and StageState")


@dataclass(frozen=True)
class ValueBankManifest:
    producer_compatibility_hash: str
    label_definition_hash: str
    split_role_hash: str
    gain_scale: float
    gain_scale_hash: str
    shard_hashes: tuple[str, ...] = ()
    row_count: int = 0
    subject_count: int = 0
    descriptor_schema: str = DESCRIPTOR_SCHEMA
    version: str = VALUE_BANK_SCHEMA

    def __post_init__(self) -> None:
        if self.version != VALUE_BANK_SCHEMA:
            raise ValueError("unknown value-bank schema")
        for name in ("producer_compatibility_hash", "label_definition_hash", "split_role_hash", "gain_scale_hash"):
            _nonempty_text(name, getattr(self, name))
        if not math.isfinite(float(self.gain_scale)) or self.gain_scale <= 0.0:
            raise ValueError("gain_scale must be positive and finite")
        if self.row_count < 0 or self.subject_count < 0:
            raise ValueError("bank counts must be nonnegative")
        if self.descriptor_schema != DESCRIPTOR_SCHEMA:
            raise ValueError("unknown descriptor schema")


@dataclass(frozen=True)
class PFGRRouteResult:
    final_state: PFGRState
    decisions: tuple[Decision, ...] = ()
    executed_action_ids: tuple[str, ...] = ()
    k: int = 0
    stop_reason: str = "budget"
    counters: OperationCounters = field(default_factory=OperationCounters)
    context_id: str = ""
    policy_hash: str = ""
    completed_trace: CompletedBehaviorTrace | None = None
    parallel_trace: ParallelBehaviorTrace | None = None
    terminal_proposals: ActionProposalBatch | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.final_state, PFGRState) or not isinstance(self.counters, OperationCounters):
            raise TypeError("PFGRRouteResult requires final state and counters")
        if not isinstance(self.k, int) or isinstance(self.k, bool) or self.k < 0 or self.k > 4:
            raise ValueError("PFGR route K must lie in 0..4")
        if self.k != self.final_state.state_version:
            raise ValueError("PFGR route K must match the final state version")
        if not self.context_id:
            object.__setattr__(self, "context_id", self.final_state.context_id)
        if self.context_id != self.final_state.context_id:
            raise ValueError("route context must match final state")
        if self.completed_trace is not None:
            if not isinstance(self.completed_trace, CompletedBehaviorTrace):
                raise TypeError("completed_trace must be CompletedBehaviorTrace or None")
            if self.completed_trace.context_id != self.context_id:
                raise ValueError("completed_trace context must match route context")
        if self.parallel_trace is not None:
            if not isinstance(self.parallel_trace, ParallelBehaviorTrace):
                raise TypeError("parallel_trace must be ParallelBehaviorTrace or None")
            if self.parallel_trace.context_id != self.context_id:
                raise ValueError("parallel_trace context must match route context")
            if self.completed_trace is not None:
                raise ValueError("parallel routes cannot masquerade as sequential CompletedBehaviorTrace")
        if self.terminal_proposals is not None:
            if not isinstance(self.terminal_proposals, ActionProposalBatch):
                raise TypeError("terminal_proposals must be ActionProposalBatch or None")
            if self.terminal_proposals.context_id != self.context_id:
                raise ValueError("terminal proposal context must match route context")
            self.terminal_proposals.validate_integrity()
            if self.terminal_proposals.state_version != self.final_state.state_version or self.terminal_proposals.state_digest != self.final_state.state_digest:
                raise ValueError("terminal proposal must be bound to the final route state")
            if not self.decisions or self.decisions[-1].stop_code == "continue" or self.decisions[-1].proposal_digest != self.terminal_proposals.proposal_digest:
                raise ValueError("terminal proposal must be bound to the final stopping decision")


__all__ = [
    "ACTION_SCHEMA",
    "ActionProposal",
    "ActionProposalBatch",
    "CalibrationIdentity",
    "CompletedBehaviorTrace",
    "COUNTERS_SCHEMA",
    "DESCRIPTOR_SCHEMA",
    "DescriptorBundle",
    "Decision",
    "EffectTeacherConfig",
    "GainCalibration",
    "GainLabel",
    "InferenceBundle",
    "MultiScaleFeatureGeometry",
    "ObservationContext",
    "OperationCounters",
    "ParallelBehaviorTrace",
    "PFGRRouteResult",
    "PFGRState",
    "PFGR_TYPES_SCHEMA",
    "ProducerCompatibility",
    "ProducerDependencies",
    "ResumeState",
    "SparseFootprint",
    "StageState",
    "SourceProvenance",
    "TRACE_SCHEMA",
    "TRAINING_ROLES_SCHEMA",
    "TrainingRoleManifest",
    "V_DESCRIPTOR_DIMS",
    "ValueBankManifest",
    "ValueFitIdentity",
    "clone_dynamic_planes",
    "build_descriptor_bundle",
    "dynamic_planes_digest",
]
