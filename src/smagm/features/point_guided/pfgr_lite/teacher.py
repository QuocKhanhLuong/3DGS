"""Detached PFGR-Lite target-after-trace effect teacher.

This module is the sole W2 target boundary.  A target and observation-derived
mask are validated into an immutable ``ValidatedTargetContext`` only after a
sealed target-free ``CompletedBehaviorTrace`` exists.  Label production then
uses the exact lattice/write algebra from :mod:`sparse_write`; no target,
target residual, or oracle state reaches proposals, geometry, or deployment.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from ..contracts import VolumeGeometry
from ..decoder import ImplicitTriPlaneDecoder
from ..spectral_query import FeatureGridGeometry
from ..state_init import DynamicTriPlanes
from .footprint import PLANE_NAMES, PFGRQueryLattice
from .sparse_write import (
    DEFAULT_QUERY_CHUNK_SIZE,
    POINT_QUERY_HASH,
    build_footprint,
    query_write_delta,
    reference_full_write,
)

if TYPE_CHECKING:
    from .config import EffectTeacherConfig
    from .types import (
        ActionProposal,
        CompletedBehaviorTrace,
        GainLabel,
        OperationCounters,
    )

TEACHER_VERSION = "pfgr-lite-teacher-v1"
TARGET_CONTEXT_VERSION = "pfgr-lite-validated-target-v1"
LABEL_DEFINITION = "signed-conditional-mean-masked-global-charbonnier-v1"
DEFAULT_CHARBONNIER_EPSILON = 1e-3
_TARGET_VALIDATION_STATS = {"hot_checks": 0, "full_audits": 0}


def _positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _tensor_digest(value: Tensor) -> str:
    cpu = value.detach().to(device="cpu").contiguous()
    header = f"{cpu.dtype}|{tuple(cpu.shape)!r}|".encode()
    return hashlib.sha256(header + cpu.numpy().tobytes()).hexdigest()


def _normalise_volume(
    name: str, value: Tensor, *, dtype: torch.dtype | None = None
) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating tensor")
    if value.ndim == 3:
        value = value.unsqueeze(0)
    elif value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError(f"{name} rank-4 form must have batch size one")
    elif value.ndim == 5:
        if value.shape[0] != 1 or value.shape[1] != 1:
            raise ValueError(f"{name} rank-5 form must be [1,1,D,H,W]")
        value = value[:, 0]
    else:
        raise ValueError(f"{name} must be [D,H,W], [1,D,H,W], or [1,1,D,H,W]")
    if any(int(size) <= 0 for size in value.shape[-3:]) or value.numel() == 0:
        raise ValueError(f"{name} must have positive spatial dimensions")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    if dtype is not None and value.dtype != dtype:
        value = value.to(dtype=dtype)
    return value.detach().clone()


def _normalise_mask(
    value: Tensor | None, *, shape: tuple[int, int, int], device: torch.device
) -> Tensor:
    if value is None:
        return torch.ones((1, *shape), dtype=torch.bool, device=device)
    if not isinstance(value, Tensor):
        raise TypeError("observation_mask must be a tensor or None")
    if value.ndim == 3:
        value = value.unsqueeze(0)
    elif value.ndim == 5:
        if value.shape[0] != 1 or value.shape[1] != 1:
            raise ValueError("observation_mask rank-5 form must be [1,1,D,H,W]")
        value = value[:, 0]
    elif value.ndim != 4 or value.shape[0] != 1:
        raise ValueError("observation_mask must be [D,H,W], [1,D,H,W], or [1,1,D,H,W]")
    if tuple(value.shape) != (1, *shape):
        raise ValueError("observation_mask shape must match target geometry")
    if value.dtype == torch.bool:
        return value.detach().clone()
    if not value.is_floating_point() and value.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("observation_mask must be bool or numeric binary")
    if value.numel() == 0 or not bool(torch.isfinite(value).all()):
        raise ValueError("observation_mask must be finite and nonempty")
    if not bool(((value == 0) | (value == 1)).all()):
        raise ValueError("observation_mask must contain only exact binary 0/1 values")
    return value.to(dtype=torch.bool).detach().clone()


@dataclass(frozen=True)
class ValidatedTargetContext:
    """Owned detached target/mask and immutable geometry identity.

    The optional geometry/lattice fields are supplied by the post-trace
    evaluator.  They are never accepted by target-free APIs and are not
    serialized into deployment bundles.
    """

    completed_context_id: str
    target: Tensor
    observation_mask: Tensor
    mask_count: int
    provenance: str
    output_geometry: VolumeGeometry | None = None
    feature_geometry: FeatureGridGeometry | None = None
    lattice: PFGRQueryLattice | None = None
    version: str = TARGET_CONTEXT_VERSION
    # Optional post-trace binding identities.  They are deliberately kept on
    # the target-owned context rather than smuggled through target-free W1
    # declarations.  A caller that has the producer/normalization records
    # must provide them; ``None`` remains useful for small standalone probes.
    producer_compatibility_hash: str | None = None
    normalization_hash: str | None = None
    mask_definition: str = "observation_derived_binary"
    label_definition: str = LABEL_DEFINITION
    trace_route_hash: str | None = None
    observation_context_id: str | None = None
    observation_context_producer_hash: str | None = None
    observation_mask_provenance: str | None = None
    engineering_only: bool = False
    _target_digest: str = field(init=False, repr=False, compare=False)
    _mask_digest: str = field(init=False, repr=False, compare=False)
    _metadata_digest: str = field(init=False, repr=False, compare=False)
    _target_version: int = field(init=False, repr=False, compare=False)
    _mask_version: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.completed_context_id, str)
            or not self.completed_context_id
        ):
            raise ValueError("completed_context_id must be nonempty")
        if self.version != TARGET_CONTEXT_VERSION:
            raise ValueError("unknown validated target context version")
        if (
            not isinstance(self.target, Tensor)
            or self.target.ndim != 4
            or self.target.shape[0] != 1
        ):
            raise ValueError("target must be owned [1,D,H,W]")
        if (
            not self.target.is_floating_point()
            or self.target.numel() == 0
            or not bool(torch.isfinite(self.target).all())
        ):
            raise ValueError("target must be finite nonempty floating data")
        if (
            not isinstance(self.observation_mask, Tensor)
            or self.observation_mask.dtype != torch.bool
            or self.observation_mask.shape != self.target.shape
        ):
            raise ValueError(
                "observation_mask must be bool [1,D,H,W] aligned with target"
            )
        if self.observation_mask.device != self.target.device:
            raise ValueError("target and observation_mask must share device")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("provenance must be a nonempty declaration")
        if not isinstance(self.engineering_only, bool):
            raise TypeError("engineering_only must be bool")
        if self.engineering_only and self.observation_context_id is not None:
            raise ValueError(
                "engineering_only target contexts cannot carry an observation context"
            )
        if not self.engineering_only:
            # Factory validation binds these identities from one immutable,
            # target-free ObservationContext.  Keep the dataclass constructor
            # fail-closed too so callers cannot bypass that boundary by
            # constructing a production context directly.
            if self.observation_context_id != self.completed_context_id:
                raise ValueError(
                    "production target contexts require an ObservationContext binding"
                )
            if (
                self.producer_compatibility_hash is None
                or self.observation_context_producer_hash
                != self.producer_compatibility_hash
            ):
                raise ValueError(
                    "production target contexts require matching observation producer identity"
                )
            if self.normalization_hash is None:
                raise ValueError(
                    "production target contexts require normalization identity"
                )
            if self.observation_mask_provenance is None:
                raise ValueError(
                    "production target contexts require observation mask provenance"
                )
            if self.output_geometry is None or self.feature_geometry is None:
                raise ValueError(
                    "production target contexts require output and feature geometry"
                )
        for name in (
            "producer_compatibility_hash",
            "normalization_hash",
            "observation_context_id",
            "observation_context_producer_hash",
            "trace_route_hash",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str)
                or not value.strip()
                or value.lower() in {"unknown", "unset", "none", "null"}
            ):
                raise ValueError(f"{name} must be a complete string when provided")
        if self.observation_context_id is not None and self.observation_context_producer_hash is None:
            raise ValueError(
                "observation context binding requires its producer compatibility hash"
            )
        if self.observation_mask_provenance is not None and (
            not isinstance(self.observation_mask_provenance, str)
            or not self.observation_mask_provenance.strip()
            or any(
                token in self.observation_mask_provenance.lower()
                for token in ("target", "teacher", "oracle", "segmentation")
            )
        ):
            raise ValueError("observation_mask_provenance must remain observation-only")
        if (
            not isinstance(self.mask_definition, str)
            or not self.mask_definition.strip()
        ):
            raise ValueError("mask_definition must be a nonempty declaration")
        if (
            not isinstance(self.label_definition, str)
            or not self.label_definition.strip()
        ):
            raise ValueError("label_definition must be a nonempty declaration")
        expected_count = int(self.observation_mask.sum().item())
        if self.mask_count != expected_count or self.mask_count <= 0:
            raise ValueError("target context requires mask_count=sum(mask)>0")
        if self.output_geometry is not None:
            if not isinstance(self.output_geometry, VolumeGeometry):
                raise TypeError("output_geometry must be VolumeGeometry")
            if tuple(self.output_geometry.shape_dhw) != tuple(self.target.shape[-3:]):
                raise ValueError("output_geometry shape must match target")
        if self.feature_geometry is not None:
            if not isinstance(self.feature_geometry, FeatureGridGeometry):
                raise TypeError("feature_geometry must be FeatureGridGeometry")
            if (
                self.output_geometry is not None
                and self.feature_geometry.source_geometry != self.output_geometry
            ):
                raise ValueError("feature_geometry source must match output_geometry")
        if self.lattice is not None:
            if not isinstance(self.lattice, PFGRQueryLattice):
                raise TypeError("lattice must be PFGRQueryLattice")
            if (
                self.output_geometry is not None
                and self.lattice.output_geometry != self.output_geometry
            ):
                raise ValueError(
                    "lattice output geometry does not match target context"
                )
            if (
                self.feature_geometry is not None
                and self.lattice.feature_geometry != self.feature_geometry
            ):
                raise ValueError(
                    "lattice feature geometry does not match target context"
                )
        object.__setattr__(self, "_target_digest", _tensor_digest(self.target))
        object.__setattr__(self, "_mask_digest", _tensor_digest(self.observation_mask))
        object.__setattr__(
            self,
            "_metadata_digest",
            hashlib.sha256(
                "|".join(
                    (
                        self.completed_context_id,
                        self.provenance,
                        str(self.output_geometry),
                        str(self.feature_geometry),
                        str(
                            self.lattice.geometry_hash
                            if self.lattice is not None
                            else None
                        ),
                        str(self.producer_compatibility_hash),
                        str(self.normalization_hash),
                        str(self.observation_context_id),
                        str(self.observation_context_producer_hash),
                        str(self.observation_mask_provenance),
                        str(self.engineering_only),
                        self.mask_definition,
                        self.label_definition,
                        str(self.trace_route_hash),
                    )
                ).encode()
            ).hexdigest(),
        )
        object.__setattr__(self, "_target_version", int(self.target._version))
        object.__setattr__(self, "_mask_version", int(self.observation_mask._version))
        _TARGET_VALIDATION_STATS["full_audits"] += 1

    @property
    def context_id(self) -> str:
        return self.completed_context_id

    @property
    def target_mask(self) -> Tensor:
        return self.observation_mask

    @property
    def M(self) -> int:
        return self.mask_count

    def validate_integrity(self, *, full_audit: bool = False) -> None:
        """Validate context guards, optionally rehashing target/mask bytes.

        The default is a cheap metadata/version check suitable for repeated
        state/action labels.  ``full_audit=True`` performs the explicit
        detached target/mask digest and denominator scan; callers should use
        it at immutable context boundaries or after any unsafe ``.data`` /
        storage-level mutation that bypasses PyTorch's version counter.
        """

        if not isinstance(full_audit, bool):
            raise TypeError("full_audit must be bool")
        self.validate_fast()
        # ``validate_fast`` already records the hot check and metadata guard;
        # convert that accounting to a full audit when requested.
        _TARGET_VALIDATION_STATS["hot_checks"] -= 1
        _TARGET_VALIDATION_STATS["full_audits" if full_audit else "hot_checks"] += 1
        if not full_audit:
            return
        metadata_digest = hashlib.sha256(
            "|".join(
                (
                    self.completed_context_id,
                    self.provenance,
                    str(self.output_geometry),
                    str(self.feature_geometry),
                    str(
                        self.lattice.geometry_hash if self.lattice is not None else None
                    ),
                    str(self.producer_compatibility_hash),
                    str(self.normalization_hash),
                    str(self.observation_context_id),
                    str(self.observation_context_producer_hash),
                    str(self.observation_mask_provenance),
                    str(self.engineering_only),
                    self.mask_definition,
                    self.label_definition,
                    str(self.trace_route_hash),
                )
            ).encode()
        ).hexdigest()
        if metadata_digest != self._metadata_digest:
            raise RuntimeError("ValidatedTargetContext metadata mutation detected")
        if (
            int(self.target._version) != self._target_version
            or int(self.observation_mask._version) != self._mask_version
        ):
            raise RuntimeError("ValidatedTargetContext tensor mutation detected")
        if (
            _tensor_digest(self.target) != self._target_digest
            or _tensor_digest(self.observation_mask) != self._mask_digest
        ):
            raise RuntimeError("ValidatedTargetContext tensor mutation detected")
        if int(self.observation_mask.sum().item()) != self.mask_count:
            raise RuntimeError("ValidatedTargetContext mask denominator changed")

    def validate_fast(self) -> None:
        """Check immutable metadata and tensor version counters only.

        Target/mask bytes are detached and hashed once at construction.  The
        normal teacher hot path therefore uses this cheap guard; callers that
        need an explicit full audit can call :meth:`validate_integrity` (or
        pass ``audit_target_context=True`` to :func:`measure_actions`).
        """

        metadata_digest = hashlib.sha256(
            "|".join(
                (
                    self.completed_context_id,
                    self.provenance,
                    str(self.output_geometry),
                    str(self.feature_geometry),
                    str(
                        self.lattice.geometry_hash if self.lattice is not None else None
                    ),
                    str(self.producer_compatibility_hash),
                    str(self.normalization_hash),
                    str(self.observation_context_id),
                    str(self.observation_context_producer_hash),
                    str(self.observation_mask_provenance),
                    str(self.engineering_only),
                    self.mask_definition,
                    self.label_definition,
                    str(self.trace_route_hash),
                )
            ).encode()
        ).hexdigest()
        if metadata_digest != self._metadata_digest:
            raise RuntimeError("ValidatedTargetContext metadata mutation detected")
        if (
            int(self.target._version) != self._target_version
            or int(self.observation_mask._version) != self._mask_version
        ):
            raise RuntimeError("ValidatedTargetContext tensor mutation detected")
        _TARGET_VALIDATION_STATS["hot_checks"] += 1

    def _validate_tensor_versions(self) -> None:
        """Cheap gather-time guard; full digest validation is done once/label call."""

        if (
            int(self.target._version) != self._target_version
            or int(self.observation_mask._version) != self._mask_version
        ):
            raise RuntimeError("ValidatedTargetContext tensor mutation detected")

    def gather_target(
        self,
        voxel_ids_dhw: Tensor,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        self._validate_tensor_versions()
        if (
            not isinstance(voxel_ids_dhw, Tensor)
            or voxel_ids_dhw.ndim != 2
            or voxel_ids_dhw.shape[-1] != 3
            or voxel_ids_dhw.dtype != torch.long
        ):
            raise ValueError("voxel_ids_dhw must be [Q,3] torch.long")
        ids = voxel_ids_dhw.to(device=self.target.device)
        values = self.target[0, ids[:, 0], ids[:, 1], ids[:, 2]]
        if device is not None or dtype is not None:
            values = values.to(
                device=device or values.device, dtype=dtype or values.dtype
            )
        return values

    def gather_mask(
        self, voxel_ids_dhw: Tensor, *, device: torch.device | None = None
    ) -> Tensor:
        self._validate_tensor_versions()
        ids = voxel_ids_dhw.to(device=self.observation_mask.device)
        values = self.observation_mask[0, ids[:, 0], ids[:, 1], ids[:, 2]]
        return values if device is None else values.to(device=device)


def validate_target(
    completed_context_id: str,
    target: Tensor,
    observation_mask: Tensor | None,
    provenance: str = "post_trace_target_provider_v1",
    *,
    output_geometry: VolumeGeometry | None = None,
    feature_geometry: FeatureGridGeometry | None = None,
    lattice: PFGRQueryLattice | None = None,
    producer_compatibility_hash: str | None = None,
    normalization_hash: str | None = None,
    mask_definition: str = "observation_derived_binary",
    label_definition: str = LABEL_DEFINITION,
    trace_route_hash: str | None = None,
    completed_trace: object | None = None,
    observation_context: object | None = None,
    engineering_only: bool = False,
) -> ValidatedTargetContext:
    """Validate and detach target supervision exactly once per context.

    Production callers must bind the target to an immutable target-free
    ``ObservationContext``.  Small CPU fixtures may opt into the explicitly
    marked ``engineering_only=True`` path, which cannot be consumed as a
    production bank/calibration context without an observation binding.
    """

    if completed_trace is not None:
        from .types import CompletedBehaviorTrace

        if (
            not isinstance(completed_trace, CompletedBehaviorTrace)
            or not completed_trace.sealed
        ):
            raise ValueError(
                "completed_trace must be a sealed target-free CompletedBehaviorTrace"
            )
        if completed_trace.context_id != completed_context_id:
            raise ValueError("target context ID must match completed trace context")

    if not isinstance(engineering_only, bool):
        raise TypeError("engineering_only must be bool")
    bound_observation_context_id: str | None = None
    bound_observation_producer_hash: str | None = None
    bound_mask_provenance: str | None = None
    if observation_context is None:
        if not engineering_only:
            raise ValueError(
                "production target validation requires observation_context; "
                "use engineering_only=True only for isolated fixtures"
            )
    else:
        from .types import ObservationContext

        if not isinstance(observation_context, ObservationContext):
            raise TypeError("observation_context must be ObservationContext")
        observation_context.validate_integrity()
        if observation_context.context_id != completed_context_id:
            raise ValueError("target context ID must match observation context")
        if engineering_only:
            raise ValueError(
                "engineering_only target validation cannot be combined with observation_context"
            )
        context_geometry = observation_context.geometry
        context_feature_geometry = observation_context.feature_geometry
        if output_geometry is None:
            output_geometry = context_geometry
        elif output_geometry != context_geometry:
            raise ValueError("output_geometry does not match observation context")
        if feature_geometry is None:
            feature_geometry = context_feature_geometry
        elif feature_geometry != context_feature_geometry:
            raise ValueError("feature_geometry does not match observation context")
        context_producer_hash = observation_context.producer.compatibility_hash
        if completed_trace is not None:
            trace_hashes = {
                state.producer.digest for state in completed_trace.states
            }
            if trace_hashes != {context_producer_hash}:
                raise ValueError(
                    "completed trace producer identity does not match observation context"
                )
        if (
            producer_compatibility_hash is not None
            and producer_compatibility_hash != context_producer_hash
        ):
            raise ValueError("producer identity does not match observation context")
        producer_compatibility_hash = context_producer_hash
        context_normalization_hash = (
            observation_context.producer.compatibility.observation_normalization_hash
        )
        if (
            normalization_hash is not None
            and normalization_hash != context_normalization_hash
        ):
            raise ValueError("normalization identity does not match observation context")
        normalization_hash = context_normalization_hash
        bound_observation_context_id = observation_context.context_id
        bound_observation_producer_hash = context_producer_hash
        bound_mask_provenance = observation_context.mask_provenance
        context_mask = observation_context.observation_mask
        if (
            context_mask.ndim != 4
            or context_mask.shape[0] != 1
            or observation_context.initial_planes.xy.shape[0] != 1
        ):
            raise ValueError(
                "validated target context requires a subject-batch-one ObservationContext"
            )
        if observation_mask is None:
            observation_mask = context_mask
        else:
            supplied_mask = _normalise_mask(
                observation_mask,
                shape=tuple(int(value) for value in context_mask.shape[-3:]),
                device=context_mask.device,
            )
            if not torch.equal(supplied_mask, context_mask):
                raise ValueError(
                    "observation_mask does not match observation context mask"
                )
            observation_mask = supplied_mask

    target_owned = _normalise_volume("target", target)
    mask_owned = _normalise_mask(
        observation_mask,
        shape=tuple(int(value) for value in target_owned.shape[-3:]),
        device=target_owned.device,
    )
    mask_count = int(mask_owned.sum().item())
    if mask_count <= 0:
        raise ValueError("observation mask must contain at least one valid voxel")
    if output_geometry is not None and tuple(output_geometry.shape_dhw) != tuple(
        target_owned.shape[-3:]
    ):
        raise ValueError("output_geometry shape must match target")
    return ValidatedTargetContext(
        completed_context_id=completed_context_id,
        target=target_owned,
        observation_mask=mask_owned,
        mask_count=mask_count,
        provenance=provenance,
        output_geometry=output_geometry,
        feature_geometry=feature_geometry,
        lattice=lattice,
        producer_compatibility_hash=producer_compatibility_hash,
        normalization_hash=normalization_hash,
        observation_context_id=bound_observation_context_id,
        observation_context_producer_hash=bound_observation_producer_hash,
        observation_mask_provenance=bound_mask_provenance,
        engineering_only=engineering_only,
        mask_definition=mask_definition,
        label_definition=label_definition,
        trace_route_hash=trace_route_hash,
    )


@dataclass(frozen=True)
class _Probe:
    linear: Tensor
    query: Tensor
    prediction: Tensor


# Candidate batching is an operational optimisation, not a new scientific
# approximation.  Groups are split before materialising their concatenated
# query/delta/MLP rows when this measured working-set estimate is exceeded.
# The exact stream path remains available for one candidate whose support is
# itself larger than the bound.
CANDIDATE_BATCH_MAX_BYTES = 64 * 1024 * 1024


@dataclass
class _CandidateProbe:
    """Prepared one-action rows used by the bounded candidate work queue."""

    action: Any
    state: Any
    footprint: Any
    action_index: int
    ids: Tensor
    sorted_linear: Tensor
    query: Tensor | None = None
    prediction: Tensor | None = None
    cache_key: tuple[Any, ...] | None = None
    source: _CandidateProbe | None = None
    cached: bool = False


_PROBE_CACHE: OrderedDict[tuple[Any, ...], _Probe] = OrderedDict()
_PROBE_CACHE_BYTES = 0
PROBE_CACHE_MAX_ENTRIES = 32
PROBE_CACHE_MAX_BYTES = 256 * 1024 * 1024
# Do not retain a giant all-output tensor merely because a footprint happens
# to be large.  Above this row bound the evaluator streams state/query/MLP
# chunks and reuses only its bounded small-footprint cache entries.
PROBE_CACHE_MAX_ROWS = 4096


def clear_teacher_cache() -> None:
    """Clear detached state/query prediction cache entries."""

    global _PROBE_CACHE_BYTES
    _PROBE_CACHE.clear()
    _PROBE_CACHE_BYTES = 0


def teacher_cache_stats() -> dict[str, int]:
    """Return bounded detached probe-cache retention statistics."""

    return {
        "entries": len(_PROBE_CACHE),
        "retained_bytes": int(_PROBE_CACHE_BYTES),
        "max_entries": PROBE_CACHE_MAX_ENTRIES,
        "max_bytes": PROBE_CACHE_MAX_BYTES,
    }


def clear_target_validation_stats() -> None:
    """Reset local hot-check/full-audit accounting for an isolated benchmark."""

    _TARGET_VALIDATION_STATS["hot_checks"] = 0
    _TARGET_VALIDATION_STATS["full_audits"] = 0


def target_validation_stats() -> dict[str, int]:
    """Return actual cheap-check and full-digest validation counts."""

    return dict(_TARGET_VALIDATION_STATS)


def _module_digest(module: object) -> str:
    if isinstance(module, nn.Module):
        # Use the same canonical producer hash as W1 compatibility envelopes;
        # this also lets measurement reject a decoder whose weights changed
        # after the target-free trace was sealed.
        from .provenance import module_state_digest

        return module_state_digest(module)
    return type(module).__name__


def _decoder_mlp(decoder: object, query: Tensor) -> Tensor:
    if isinstance(decoder, ImplicitTriPlaneDecoder):
        output = decoder.mlp(query)
    elif hasattr(decoder, "mlp"):
        mlp = decoder.mlp  # type: ignore[attr-defined]
        if not callable(mlp):
            raise TypeError("decoder.mlp must be callable")
        output = mlp(query)
    elif callable(decoder):
        output = decoder(query)  # type: ignore[misc]
    else:
        raise TypeError("decoder must expose the shared .mlp callable")
    if not isinstance(output, Tensor) or output.ndim not in (1, 2):
        raise ValueError("decoder must return [Q] or [Q,1]")
    if output.ndim == 2 and output.shape[-1] != 1:
        raise ValueError("decoder output must have one scalar channel")
    output = output.reshape(-1)
    if output.shape[0] != query.shape[0] or not bool(torch.isfinite(output).all()):
        raise ValueError("decoder output must be finite and aligned with query rows")
    return output


def _linear_ids(ids: Tensor, shape: Sequence[int]) -> Tensor:
    _, height, width = tuple(int(value) for value in shape)
    return ids[:, 0] * height * width + ids[:, 1] * width + ids[:, 2]


def _probe_cache_identity(
    lattice: PFGRQueryLattice,
    state: DynamicTriPlanes,
    decoder: object,
    ids: Tensor,
    *,
    state_identity: str | None = None,
    decoder_identity: str | None = None,
) -> tuple[tuple[Any, ...], Tensor, Tensor, Tensor]:
    """Build the immutable before-probe cache key and canonical row order."""

    linear = _linear_ids(ids, lattice.output_shape_dhw)
    order = torch.argsort(linear)
    sorted_linear = linear[order]
    resolved_state_identity = (
        state_identity
        or hashlib.sha256(
            "|".join(
                _tensor_digest(getattr(state, name)) for name in PLANE_NAMES
            ).encode()
        ).hexdigest()
    )
    resolved_decoder_identity = decoder_identity or _module_digest(decoder)
    key = (
        TEACHER_VERSION,
        resolved_state_identity,
        lattice.geometry_hash,
        lattice.output_geometry_hash,
        lattice.feature_geometry_hash,
        lattice.query_version,
        lattice.footprint_mode,
        str(state.xy.dtype),
        str(state.xy.device),
        resolved_decoder_identity,
        tuple(sorted_linear.detach().to(device="cpu").tolist()),
    )
    return key, order, sorted_linear, ids[order]


def _store_probe(
    key: tuple[Any, ...],
    sorted_linear: Tensor,
    query: Tensor,
    prediction: Tensor,
) -> None:
    """Insert one detached probe while enforcing the process cache bound."""

    global _PROBE_CACHE_BYTES
    probe = _Probe(
        linear=sorted_linear.detach().cpu(),
        query=query.detach().cpu(),
        prediction=prediction.detach().cpu(),
    )
    probe_bytes = sum(
        value.numel() * value.element_size()
        for value in (probe.linear, probe.query, probe.prediction)
    )
    while _PROBE_CACHE and (
        len(_PROBE_CACHE) >= PROBE_CACHE_MAX_ENTRIES
        or _PROBE_CACHE_BYTES + probe_bytes > PROBE_CACHE_MAX_BYTES
    ):
        _, evicted = _PROBE_CACHE.popitem(last=False)
        _PROBE_CACHE_BYTES -= sum(
            value.numel() * value.element_size()
            for value in (evicted.linear, evicted.query, evicted.prediction)
        )
    if probe_bytes <= PROBE_CACHE_MAX_BYTES:
        _PROBE_CACHE[key] = probe
        _PROBE_CACHE_BYTES += probe_bytes


def _probe_before(
    lattice: PFGRQueryLattice,
    state: DynamicTriPlanes,
    decoder: object,
    ids: Tensor,
    *,
    chunk_size: int,
    counters: OperationCounters | None,
    state_identity: str | None = None,
    decoder_identity: str | None = None,
) -> tuple[Tensor, Tensor, bool]:
    key, order, sorted_linear, sorted_ids = _probe_cache_identity(
        lattice,
        state,
        decoder,
        ids,
        state_identity=state_identity,
        decoder_identity=decoder_identity,
    )
    cached = _PROBE_CACHE.get(key)
    if cached is not None:
        _PROBE_CACHE.move_to_end(key)
        if counters is not None:
            counters.add(cache_hits=1)
        inverse = torch.argsort(order)
        return (
            cached.query.to(device=state.xy.device)[inverse],
            cached.prediction.to(device=state.xy.device)[inverse],
            True,
        )
    if counters is not None:
        counters.add(cache_misses=1)
    query = lattice.query(state, sorted_ids, chunk_size=chunk_size)
    prediction = _decode_mlp_chunked(
        decoder, query, chunk_size=chunk_size, counters=counters, before=True
    )
    # A large exact footprint is evaluated in a streaming caller and should
    # never create a process-global full-Q cache entry.  The row guard is
    # explicit in addition to the byte guard so a short-vector dtype cannot
    # bypass the operational bound.
    if query.shape[0] > PROBE_CACHE_MAX_ROWS:
        inverse = torch.argsort(order)
        return query[inverse], prediction[inverse], False
    _store_probe(key, sorted_linear, query, prediction)
    inverse = torch.argsort(order)
    return query[inverse], prediction[inverse], False


def _decode_mlp_chunked(
    decoder: object,
    query: Tensor,
    *,
    chunk_size: int,
    counters: OperationCounters | None,
    before: bool,
) -> Tensor:
    outputs: list[Tensor] = []
    for start in range(0, query.shape[0], chunk_size):
        stop = min(start + chunk_size, query.shape[0])
        outputs.append(_decoder_mlp(decoder, query[start:stop]))
        if counters is not None:
            counters.add(decoder_calls=1, mlp_calls=1)
    result = (
        torch.cat(outputs, dim=0)
        if outputs
        else torch.empty((0,), dtype=query.dtype, device=query.device)
    )
    if counters is not None:
        if before:
            counters.add(before_decoder_outputs=int(result.shape[0]))
        else:
            counters.add(after_decoder_outputs=int(result.shape[0]))
    return result


def _candidate_rows_bytes(row_count: int, dtype: torch.dtype) -> int:
    """Conservative peak estimate for one candidate's query/delta/MLP rows."""

    # During an after pass the work queue holds q_before, dq, q_after and the
    # scalar prediction buffers.  Include ids/weights overhead so the bound
    # remains honest for both FP32 production and FP64 tests.
    element = torch.empty((), dtype=dtype).element_size()
    return int(row_count) * (96 * element * 6 + 3 * 8)


def _record_tensor_bytes(counters: OperationCounters | None, *values: Tensor) -> None:
    """Record bytes actually materialised by a candidate batching operation."""

    if counters is not None:
        counters.add(
            bytes_copied=sum(
                int(value.numel()) * int(value.element_size())
                for value in values
                if isinstance(value, Tensor)
            )
        )


def _candidate_decode_chunk_size(
    records: Sequence[_CandidateProbe], chunk_size: int
) -> int:
    """Allow one decoder call to span bounded candidate query chunks."""

    # ``chunk_size`` bounds rows for the serial reference.  In candidate mode
    # we intentionally widen that bound by the number of actions in this
    # already byte-bounded group so the nonlinear MLP sees a genuine batched
    # work queue rather than a sequence of single-candidate calls.
    return max(chunk_size, chunk_size * max(len(records), 1))


def _prepare_candidate_probes(
    lattice: PFGRQueryLattice,
    records: Sequence[_CandidateProbe],
    decoder: object,
    *,
    chunk_size: int,
    counters: OperationCounters | None,
    decoder_identity: str,
) -> list[_CandidateProbe]:
    """Resolve cache entries and query rows, reusing equal before probes."""

    by_key: dict[tuple[Any, ...], _CandidateProbe] = {}
    for probe in records:
        key, _order, sorted_linear, sorted_ids = _probe_cache_identity(
            lattice,
            probe.state.planes,
            decoder,
            probe.ids,
            state_identity=probe.state.state_digest,
            decoder_identity=decoder_identity,
        )
        # Keep ``sorted_ids`` as the canonical order for all later query/delta
        # work so duplicate voxel IDs are deterministic.
        probe.cache_key = key
        probe.ids = sorted_ids
        probe.sorted_linear = sorted_linear
        cached = _PROBE_CACHE.get(key)
        if cached is not None:
            _PROBE_CACHE.move_to_end(key)
            probe.query = cached.query.to(device=probe.state.planes.xy.device)
            probe.prediction = cached.prediction.to(device=probe.state.planes.xy.device)
            probe.cached = True
            if counters is not None:
                counters.add(cache_hits=1)
            continue
        source = by_key.get(key)
        if source is not None:
            probe.source = source
            if counters is not None:
                # A same-group duplicate is a real before-probe reuse, even
                # though it did not require a process-global cache lookup.
                counters.add(cache_hits=1)
            continue
        by_key[key] = probe
        probe.query = lattice.query(
            probe.state.planes, probe.ids, chunk_size=chunk_size
        )
        if counters is not None:
            counters.add(cache_misses=1)
    return list(records)


def _resolve_probe_sources(records: Sequence[_CandidateProbe]) -> None:
    """Point duplicate rows at their group's unique query/prediction source."""

    for probe in records:
        if probe.source is not None:
            source = probe.source
            if source.source is not None:
                source = source.source
            probe.query = source.query
            probe.prediction = source.prediction


def _decode_candidate_before_batch(
    records: Sequence[_CandidateProbe],
    decoder: object,
    *,
    chunk_size: int,
    counters: OperationCounters | None,
) -> None:
    """Decode uncached candidate rows in one bounded nonlinear batch."""

    unique = [
        probe
        for probe in records
        if probe.source is None and not probe.cached and probe.prediction is None
    ]
    if not unique:
        _resolve_probe_sources(records)
        return
    query_parts = [probe.query for probe in unique if probe.query is not None]
    query_batch = torch.cat(query_parts)
    _record_tensor_bytes(counters, *query_parts, query_batch)
    prediction_batch = _decode_mlp_chunked(
        decoder,
        query_batch,
        chunk_size=_candidate_decode_chunk_size(records, chunk_size),
        counters=counters,
        before=True,
    )
    _record_tensor_bytes(counters, prediction_batch)
    offset = 0
    for probe in unique:
        if probe.query is None:
            raise RuntimeError("candidate probe query was not prepared")
        stop = offset + int(probe.query.shape[0])
        probe.prediction = prediction_batch[offset:stop]
        offset = stop
        if probe.query.shape[0] <= PROBE_CACHE_MAX_ROWS and probe.cache_key is not None:
            _store_probe(probe.cache_key, probe.sorted_linear, probe.query, probe.prediction)
        if counters is not None:
            counters.add(unique_decoded_queries=int(probe.query.shape[0]))
    _resolve_probe_sources(records)


def _evaluate_exact_candidate_batch(
    lattice: PFGRQueryLattice,
    records: Sequence[_CandidateProbe],
    target_context: ValidatedTargetContext,
    decoder: object,
    *,
    chunk_size: int,
    epsilon: float,
    counters: OperationCounters | None,
    decoder_identity: str,
) -> list[tuple[float, float, float, int, int]]:
    """Evaluate exact candidate rows with a bounded concatenated MLP batch."""

    _prepare_candidate_probes(
        lattice,
        records,
        decoder,
        chunk_size=chunk_size,
        counters=counters,
        decoder_identity=decoder_identity,
    )
    _decode_candidate_before_batch(
        records, decoder, chunk_size=chunk_size, counters=counters
    )
    after_queries: list[Tensor] = []
    for probe in records:
        if probe.query is None:
            raise RuntimeError("candidate probe query missing")
        delta_query = query_write_delta(
            lattice,
            probe.footprint,
            probe.ids,
            probe.action.delta,
            chunk_size=chunk_size,
        )
        _record_tensor_bytes(counters, delta_query)
        after_queries.append(probe.query + delta_query)
    after_batch = torch.cat(after_queries, dim=0)
    _record_tensor_bytes(counters, *after_queries, after_batch)
    prediction_after_batch = _decode_mlp_chunked(
        decoder,
        after_batch,
        chunk_size=_candidate_decode_chunk_size(records, chunk_size),
        counters=counters,
        before=False,
    )
    _record_tensor_bytes(counters, prediction_after_batch)
    result: list[tuple[float, float, float, int, int]] = []
    offset = 0
    for probe in records:
        if probe.prediction is None:
            raise RuntimeError("candidate before prediction missing")
        stop = offset + int(probe.prediction.shape[0])
        prediction_after = prediction_after_batch[offset:stop]
        offset = stop
        target = target_context.gather_target(
            probe.ids,
            device=prediction_after.device,
            dtype=probe.prediction.dtype,
        )
        mask = target_context.gather_mask(probe.ids, device=prediction_after.device)
        difference = torch.sqrt(
            (probe.prediction - target).square() + epsilon * epsilon
        ) - torch.sqrt((prediction_after - target).square() + epsilon * epsilon)
        weighted64 = (
            difference.to(dtype=torch.float64)
            * mask.to(dtype=torch.float64)
            / float(target_context.mask_count)
        )
        raw = float(weighted64.sum().item())
        benefit = float(weighted64.clamp_min(0.0).sum().item())
        harm = float((-weighted64).clamp_min(0.0).sum().item())
        result.append(
            (
                raw,
                benefit,
                harm,
                int(mask.sum().item()),
                int(probe.ids.shape[0]),
            )
        )
    return result


def _evaluate_fixed_q_candidate_batch(
    lattice: PFGRQueryLattice,
    records: Sequence[_CandidateProbe],
    target_context: ValidatedTargetContext,
    decoder: object,
    *,
    q_draws: int,
    seed: int | None,
    chunk_size: int,
    epsilon: float,
    counters: OperationCounters | None,
    decoder_identity: str,
) -> list[tuple[float, float, float, int, int, float, float, int]]:
    """Evaluate fixed-Q candidate rows with shared before/after batches."""

    draws_info: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
    for probe in records:
        action_seed = _seed_for_action(probe.action, seed, probe.action_index)
        draw_linear, probabilities, _ = _sample_iid_voxels(
            probe.footprint, q_draws=q_draws, seed=action_seed
        )
        _, height, width = lattice.output_shape_dhw
        ud = torch.div(draw_linear, height * width, rounding_mode="floor")
        rem = draw_linear - ud * height * width
        uh = torch.div(rem, width, rounding_mode="floor")
        uw = rem - uh * width
        draws = torch.stack((ud, uh, uw), dim=-1).to(device=probe.state.planes.xy.device)
        unique_linear, inverse = torch.unique(
            draw_linear, sorted=True, return_inverse=True
        )
        u_d = torch.div(unique_linear, height * width, rounding_mode="floor")
        u_rem = unique_linear - u_d * height * width
        u_h = torch.div(u_rem, width, rounding_mode="floor")
        u_w = u_rem - u_h * width
        unique_ids = torch.stack((u_d, u_h, u_w), dim=-1).to(
            device=probe.state.planes.xy.device
        )
        draws_info.append((draws, probabilities, inverse, unique_ids))
        probe.ids = unique_ids
        probe.sorted_linear = unique_linear
    _prepare_candidate_probes(
        lattice,
        records,
        decoder,
        chunk_size=chunk_size,
        counters=counters,
        decoder_identity=decoder_identity,
    )
    _decode_candidate_before_batch(
        records, decoder, chunk_size=chunk_size, counters=counters
    )
    after_queries: list[Tensor] = []
    for probe in records:
        if probe.query is None:
            raise RuntimeError("candidate probe query missing")
        delta_query = query_write_delta(
            lattice,
            probe.footprint,
            probe.ids,
            probe.action.delta,
            chunk_size=chunk_size,
        )
        _record_tensor_bytes(counters, delta_query)
        after_queries.append(probe.query + delta_query)
    after_batch = torch.cat(after_queries, dim=0)
    _record_tensor_bytes(counters, *after_queries, after_batch)
    prediction_after_batch = _decode_mlp_chunked(
        decoder,
        after_batch,
        chunk_size=_candidate_decode_chunk_size(records, chunk_size),
        counters=counters,
        before=False,
    )
    _record_tensor_bytes(counters, prediction_after_batch)
    result: list[tuple[float, float, float, int, int, float, float, int]] = []
    offset = 0
    for probe, (draws, probabilities, inverse, _unique_ids) in zip(
        records, draws_info
    ):
        if probe.prediction is None:
            raise RuntimeError("candidate before prediction missing")
        stop = offset + int(probe.prediction.shape[0])
        prediction_after = prediction_after_batch[offset:stop]
        offset = stop
        before_draw = probe.prediction[inverse.to(device=probe.prediction.device)]
        after_draw = prediction_after[inverse.to(device=prediction_after.device)]
        target = target_context.gather_target(
            draws, device=before_draw.device, dtype=before_draw.dtype
        )
        mask = target_context.gather_mask(draws, device=before_draw.device)
        difference = torch.sqrt(
            (before_draw - target).square() + epsilon * epsilon
        ) - torch.sqrt((after_draw - target).square() + epsilon * epsilon)
        weights = mask.to(dtype=difference.dtype) / (
            float(target_context.mask_count)
            * probabilities.to(device=difference.device, dtype=difference.dtype)
        )
        contributions64 = (difference * weights).to(dtype=torch.float64)
        benefit64 = (
            difference.to(dtype=torch.float64).clamp_min(0.0)
            * weights.to(dtype=torch.float64)
        )
        harm64 = (
            (-difference.to(dtype=torch.float64)).clamp_min(0.0)
            * weights.to(dtype=torch.float64)
        )
        raw = float(contributions64.mean().item())
        benefit = float(benefit64.mean().item())
        harm = float(harm64.mean().item())
        variance = (
            float(contributions64.var(unbiased=True).item()) if q_draws >= 2 else 0.0
        )
        result.append(
            (
                raw,
                benefit,
                harm,
                int(mask.sum().item()),
                int(probe.ids.shape[0]),
                variance,
                math.sqrt(max(variance, 0.0) / float(q_draws)),
                _seed_for_action(probe.action, seed, probe.action_index),
            )
        )
        if counters is not None:
            counters.add(sampled_draws=q_draws)
    return result


def _candidate_groups(
    records: Sequence[_CandidateProbe],
    *,
    candidate_chunk_size: int,
    dtype: torch.dtype,
) -> list[tuple[_CandidateProbe, ...]]:
    """Split candidate work into count- and byte-bounded groups."""

    groups: list[tuple[_CandidateProbe, ...]] = []
    current: list[_CandidateProbe] = []
    current_bytes = 0
    for probe in records:
        estimate = _candidate_rows_bytes(int(probe.footprint.union_size), dtype)
        if current and (
            len(current) >= candidate_chunk_size
            or current_bytes + estimate > CANDIDATE_BATCH_MAX_BYTES
        ):
            groups.append(tuple(current))
            current = []
            current_bytes = 0
        current.append(probe)
        current_bytes += estimate
    if current:
        groups.append(tuple(current))
    return groups


def _coerce_actions(proposals: object) -> list[ActionProposal]:
    from .types import ActionProposal, ActionProposalBatch

    if isinstance(proposals, ActionProposal):
        return [proposals]
    if isinstance(proposals, ActionProposalBatch):
        return [
            proposals.row(batch, point)
            for batch in range(proposals.point_ids.shape[0])
            for point in range(proposals.point_ids.shape[1])
        ]
    if isinstance(proposals, Sequence) and not isinstance(proposals, (str, bytes)):
        result: list[ActionProposal] = []
        for item in proposals:
            if isinstance(item, ActionProposal):
                result.append(item)
            elif isinstance(item, ActionProposalBatch):
                result.extend(_coerce_actions(item))
            else:
                raise TypeError("proposals must contain ActionProposal rows or batches")
        return result
    raise TypeError(
        "proposals must be ActionProposal, ActionProposalBatch, or a sequence"
    )


def _seed_for_action(action: ActionProposal, seed: int | None, index: int) -> int:
    if seed is not None:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        base = seed
    else:
        base = int(hashlib.sha256(action.action_id.encode()).hexdigest()[:16], 16)
    return int((base + index * 0x9E3779B97F4A7C15) % (2**63 - 1))


def _expected_label_identity(teacher_config: EffectTeacherConfig) -> str:
    """Recompute W1's producer label identity for binding checks."""

    from .provenance import canonical_digest

    return canonical_digest(
        {
            "definition": teacher_config.label_definition,
            "rho": teacher_config.rho,
            "epsilon": teacher_config.epsilon,
            "mask_definition": teacher_config.mask_definition,
            "global_mask_denominator": "sum(mask)>0_fixed_subject_v1",
        },
        prefix="pfgr-lite-label-definition-v1|",
    )


def _expected_geometry_identity(lattice: PFGRQueryLattice) -> str:
    """Match W4's per-context full affine geometry identity."""

    from .provenance import canonical_digest

    output = lattice.output_geometry
    feature = lattice.feature_geometry
    return canonical_digest(
        {
            "source_shape_dhw": output.shape_dhw,
            "source_voxel_to_ras_mm": output.voxel_to_ras_mm,
            "feature_shape_dhw": feature.shape_dhw,
            "feature_geometry": feature.feature_geometry.voxel_to_ras_mm,
            "feature_to_source_scale_dhw": feature.feature_to_source_scale_dhw,
            "feature_to_source_offset_dhw": feature.feature_to_source_offset_dhw,
            "operator_chain": feature.operator_chain,
            "version": "pfgr-lite-static-geometry-v1",
        },
        prefix="pfgr-lite-action-geometry-v1|",
    )


def _resolve_lattice(
    target_context: ValidatedTargetContext,
    state: DynamicTriPlanes,
    *,
    lattice: PFGRQueryLattice | None,
    chunk_size: int,
) -> PFGRQueryLattice:
    resolved = lattice or target_context.lattice
    if resolved is not None:
        if not isinstance(resolved, PFGRQueryLattice):
            raise TypeError("lattice must be a PFGRQueryLattice")
        resolved.validate_integrity()
        if tuple(resolved.output_shape_dhw) != tuple(target_context.target.shape[-3:]):
            raise ValueError("lattice output shape does not match target context")
        if (
            target_context.output_geometry is not None
            and resolved.output_geometry != target_context.output_geometry
        ):
            raise ValueError("lattice output affine does not match target context")
        if (
            target_context.feature_geometry is not None
            and resolved.feature_geometry != target_context.feature_geometry
        ):
            raise ValueError("lattice feature affine does not match target context")
        if resolved.query_dtype != state.xy.dtype:
            raise TypeError("lattice dtype must match traced state planes")
        return resolved
    if (
        target_context.output_geometry is None
        or target_context.feature_geometry is None
    ):
        raise ValueError(
            "target context must bind a PFGRQueryLattice or output/feature geometry"
        )
    return PFGRQueryLattice.build(
        target_context.output_geometry,
        target_context.feature_geometry,
        query_dtype=state.xy.dtype,
        build_chunk_size=chunk_size,
    )


def _evaluate_exact(
    lattice: PFGRQueryLattice,
    footprint: Any,
    state: DynamicTriPlanes,
    action: ActionProposal,
    target_context: ValidatedTargetContext,
    decoder: object,
    *,
    chunk_size: int,
    epsilon: float,
    counters: OperationCounters | None,
    state_identity: str | None = None,
    decoder_identity: str | None = None,
) -> tuple[float, float, float, int, int]:
    ids = footprint.voxel_ids_dhw.to(device=state.xy.device)
    if ids.shape[0] > PROBE_CACHE_MAX_ROWS:
        return _evaluate_exact_stream(
            lattice,
            footprint,
            state,
            action,
            target_context,
            decoder,
            chunk_size=chunk_size,
            epsilon=epsilon,
            counters=counters,
        )
    query_before, prediction_before, cached = _probe_before(
        lattice,
        state,
        decoder,
        ids,
        chunk_size=chunk_size,
        counters=counters,
        state_identity=state_identity,
        decoder_identity=decoder_identity,
    )
    delta_query = query_write_delta(
        lattice,
        footprint,
        ids,
        action.delta,
        chunk_size=chunk_size,
    )
    prediction_after = _decode_mlp_chunked(
        decoder,
        query_before + delta_query,
        chunk_size=chunk_size,
        counters=counters,
        before=False,
    )
    if not cached and counters is not None:
        counters.add(unique_decoded_queries=int(ids.shape[0]))
    target = target_context.gather_target(
        ids, device=state.xy.device, dtype=prediction_before.dtype
    )
    mask = target_context.gather_mask(ids, device=state.xy.device)
    before_rho = torch.sqrt((prediction_before - target).square() + epsilon * epsilon)
    after_rho = torch.sqrt((prediction_after - target).square() + epsilon * epsilon)
    difference = before_rho - after_rho
    weighted = (
        difference * mask.to(dtype=difference.dtype) / float(target_context.mask_count)
    )
    benefit = float(weighted.clamp_min(0.0).sum().item())
    harm = float((-weighted).clamp_min(0.0).sum().item())
    return benefit - harm, benefit, harm, int(mask.sum().item()), int(ids.shape[0])


def _evaluate_exact_stream(
    lattice: PFGRQueryLattice,
    footprint: Any,
    state: DynamicTriPlanes,
    action: ActionProposal,
    target_context: ValidatedTargetContext,
    decoder: object,
    *,
    chunk_size: int,
    epsilon: float,
    counters: OperationCounters | None,
) -> tuple[float, float, float, int, int]:
    """Exact union evaluation with bounded query/delta/MLP working sets."""

    ids = footprint.voxel_ids_dhw.to(device=state.xy.device)
    if counters is not None:
        # This is an intentional bounded-cache bypass, not a failed lookup.
        counters.add(cache_misses=1)
    total_raw = 0.0
    total_benefit = 0.0
    total_harm = 0.0
    valid_count = 0
    for start in range(0, int(ids.shape[0]), chunk_size):
        stop = min(start + chunk_size, int(ids.shape[0]))
        chunk_ids = ids[start:stop]
        query_before = lattice.query(state, chunk_ids, chunk_size=chunk_size)
        delta_query = query_write_delta(
            lattice,
            footprint,
            chunk_ids,
            action.delta,
            chunk_size=chunk_size,
        )
        prediction_before = _decode_mlp_chunked(
            decoder,
            query_before,
            chunk_size=chunk_size,
            counters=counters,
            before=True,
        )
        prediction_after = _decode_mlp_chunked(
            decoder,
            query_before + delta_query,
            chunk_size=chunk_size,
            counters=counters,
            before=False,
        )
        target = target_context.gather_target(
            chunk_ids, device=state.xy.device, dtype=prediction_before.dtype
        )
        mask = target_context.gather_mask(chunk_ids, device=state.xy.device)
        difference = torch.sqrt(
            (prediction_before - target).square() + epsilon * epsilon
        ) - torch.sqrt((prediction_after - target).square() + epsilon * epsilon)
        weighted = (
            difference
            * mask.to(dtype=difference.dtype)
            / float(target_context.mask_count)
        )
        # Accumulate the FP32 point evaluations in a fixed FP64 order.  This
        # keeps streamed exact labels algebraically tied to ``benefit-harm``
        # without relaxing the GainLabel invariant when chunk boundaries
        # change.  The MLP/query values themselves remain at the requested
        # production/test dtype; only this scalar reduction is widened.
        weighted64 = weighted.to(dtype=torch.float64)
        total_raw += float(weighted64.sum().item())
        total_benefit += float(weighted64.clamp_min(0.0).sum().item())
        total_harm += float((-weighted64).clamp_min(0.0).sum().item())
        valid_count += int(mask.sum().item())
    if counters is not None:
        counters.add(unique_decoded_queries=int(ids.shape[0]))
    return (
        total_raw,
        total_benefit,
        total_harm,
        valid_count,
        int(ids.shape[0]),
    )


def _sample_iid_voxels(
    footprint: Any,
    *,
    q_draws: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    plane_linear = tuple(footprint._pfgr_plane_voxel_linear)  # type: ignore[attr-defined]
    if len(plane_linear) != 3:
        raise ValueError("iid_fixed_q requires exactly three plane support rows")
    counts = torch.tensor(
        [int(value.numel()) for value in plane_linear], dtype=torch.float64
    )
    total = int(counts.sum().item())
    if total <= 0:
        raise ValueError("iid_fixed_q requires a nonempty union support")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_plane = torch.rand((q_draws,), generator=generator)
    active_planes = torch.nonzero(counts > 0, as_tuple=False).reshape(-1)
    active_counts = counts.index_select(0, active_planes)
    cumulative = torch.cumsum(active_counts / float(total), dim=0)
    selected_active = torch.bucketize(random_plane, cumulative)
    # torch.bucketize can return 3 only for an exact endpoint; torch.rand is
    # half-open, but clamp makes the contract explicit for alternate RNGs.
    selected_active = selected_active.clamp(max=active_planes.numel() - 1)
    plane_ids = active_planes[selected_active]
    draw_linear = torch.empty((q_draws,), dtype=torch.long)
    for plane in range(3):
        selected = plane_ids == plane
        amount = int(selected.sum().item())
        if amount:
            choices = torch.randint(
                int(plane_linear[plane].numel()), (amount,), generator=generator
            )
            draw_linear[selected] = plane_linear[plane][choices]
    union = footprint._pfgr_union_linear.to(dtype=torch.long)  # type: ignore[attr-defined]
    multiplicity = footprint.multiplicity.to(dtype=torch.float64)
    position = torch.searchsorted(union, draw_linear)
    safe = position.clamp(max=max(int(union.numel()) - 1, 0))
    if union.numel() == 0 or not bool((union[safe] == draw_linear).all()):
        raise RuntimeError("sampled support row is missing from footprint union")
    probabilities = multiplicity[safe] / float(total)
    return draw_linear, probabilities, plane_ids


def _evaluate_fixed_q(
    lattice: PFGRQueryLattice,
    footprint: Any,
    state: DynamicTriPlanes,
    action: ActionProposal,
    target_context: ValidatedTargetContext,
    decoder: object,
    *,
    q_draws: int,
    seed: int,
    chunk_size: int,
    epsilon: float,
    counters: OperationCounters | None,
    state_identity: str | None = None,
    decoder_identity: str | None = None,
) -> tuple[float, float, float, int, int, float, float]:
    draw_linear, probabilities, _ = _sample_iid_voxels(
        footprint, q_draws=q_draws, seed=seed
    )
    _, height, width = lattice.output_shape_dhw
    d = torch.div(draw_linear, height * width, rounding_mode="floor")
    rem = draw_linear - d * height * width
    h = torch.div(rem, width, rounding_mode="floor")
    w = rem - h * width
    draws = torch.stack((d, h, w), dim=-1).to(device=state.xy.device)
    unique_linear, inverse = torch.unique(draw_linear, sorted=True, return_inverse=True)
    ud = torch.div(unique_linear, height * width, rounding_mode="floor")
    urem = unique_linear - ud * height * width
    uh = torch.div(urem, width, rounding_mode="floor")
    uw = urem - uh * width
    unique_ids = torch.stack((ud, uh, uw), dim=-1).to(device=state.xy.device)
    if unique_ids.shape[0] > PROBE_CACHE_MAX_ROWS:
        return _evaluate_fixed_q_stream(
            lattice,
            footprint,
            state,
            action,
            target_context,
            decoder,
            draw_linear=draw_linear,
            probabilities=probabilities,
            q_draws=q_draws,
            chunk_size=chunk_size,
            epsilon=epsilon,
            counters=counters,
        )
    unique_footprint = footprint
    query_before, prediction_before, cached = _probe_before(
        lattice,
        state,
        decoder,
        unique_ids,
        chunk_size=chunk_size,
        counters=counters,
        state_identity=state_identity,
        decoder_identity=decoder_identity,
    )
    # Query/write delta is linear and can be evaluated for the unique rows;
    # inverse expansion preserves duplicate iid draws as separate samples.
    delta_query = query_write_delta(
        lattice,
        unique_footprint,
        unique_ids,
        action.delta,
        chunk_size=chunk_size,
    )
    prediction_after = _decode_mlp_chunked(
        decoder,
        query_before + delta_query,
        chunk_size=chunk_size,
        counters=counters,
        before=False,
    )
    if counters is not None:
        counters.add(sampled_draws=q_draws)
        if not cached:
            counters.add(unique_decoded_queries=int(unique_ids.shape[0]))
    before_draw = prediction_before[inverse.to(device=prediction_before.device)]
    after_draw = prediction_after[inverse.to(device=prediction_after.device)]
    target = target_context.gather_target(
        draws, device=state.xy.device, dtype=before_draw.dtype
    )
    mask = target_context.gather_mask(draws, device=state.xy.device)
    difference = torch.sqrt(
        (before_draw - target).square() + epsilon * epsilon
    ) - torch.sqrt((after_draw - target).square() + epsilon * epsilon)
    weights = mask.to(dtype=difference.dtype) / (
        float(target_context.mask_count)
        * probabilities.to(device=difference.device, dtype=difference.dtype)
    )
    contributions = difference * weights
    benefit_draws = difference.clamp_min(0.0) * weights
    harm_draws = (-difference).clamp_min(0.0) * weights
    raw = float(contributions.mean().item())
    benefit = float(benefit_draws.mean().item())
    harm = float(harm_draws.mean().item())
    variance = float(contributions.var(unbiased=True).item()) if q_draws >= 2 else 0.0
    standard_error = math.sqrt(max(variance, 0.0) / float(q_draws))
    return (
        raw,
        benefit,
        harm,
        int(mask.sum().item()),
        int(unique_ids.shape[0]),
        variance,
        standard_error,
    )


def _evaluate_fixed_q_stream(
    lattice: PFGRQueryLattice,
    footprint: Any,
    state: DynamicTriPlanes,
    action: ActionProposal,
    target_context: ValidatedTargetContext,
    decoder: object,
    *,
    draw_linear: Tensor,
    probabilities: Tensor,
    q_draws: int,
    chunk_size: int,
    epsilon: float,
    counters: OperationCounters | None,
) -> tuple[float, float, float, int, int, float, float]:
    """Fixed-Q estimator with no full-Q query/prediction materialization."""

    if counters is not None:
        counters.add(cache_misses=1, sampled_draws=q_draws)
    _, height, width = lattice.output_shape_dhw
    total_raw = 0.0
    total_benefit = 0.0
    total_harm = 0.0
    total_square = 0.0
    valid_count = 0
    unique_decoded = 0
    for start in range(0, q_draws, chunk_size):
        stop = min(start + chunk_size, q_draws)
        local_draws = draw_linear[start:stop]
        local_probabilities = probabilities[start:stop]
        unique_linear, inverse = torch.unique(
            local_draws, sorted=True, return_inverse=True
        )
        d = torch.div(unique_linear, height * width, rounding_mode="floor")
        rem = unique_linear - d * height * width
        h = torch.div(rem, width, rounding_mode="floor")
        w = rem - h * width
        unique_ids = torch.stack((d, h, w), dim=-1).to(device=state.xy.device)
        query_before = lattice.query(state, unique_ids, chunk_size=chunk_size)
        delta_query = query_write_delta(
            lattice,
            footprint,
            unique_ids,
            action.delta,
            chunk_size=chunk_size,
        )
        prediction_before = _decode_mlp_chunked(
            decoder,
            query_before,
            chunk_size=chunk_size,
            counters=counters,
            before=True,
        )
        prediction_after = _decode_mlp_chunked(
            decoder,
            query_before + delta_query,
            chunk_size=chunk_size,
            counters=counters,
            before=False,
        )
        before_draw = prediction_before[inverse.to(device=prediction_before.device)]
        after_draw = prediction_after[inverse.to(device=prediction_after.device)]
        ddraw = torch.div(local_draws, height * width, rounding_mode="floor")
        rdraw = local_draws - ddraw * height * width
        hdraw = torch.div(rdraw, width, rounding_mode="floor")
        wdraw = rdraw - hdraw * width
        draw_ids = torch.stack((ddraw, hdraw, wdraw), dim=-1).to(device=state.xy.device)
        target = target_context.gather_target(
            draw_ids, device=state.xy.device, dtype=before_draw.dtype
        )
        mask = target_context.gather_mask(draw_ids, device=state.xy.device)
        difference = torch.sqrt(
            (before_draw - target).square() + epsilon * epsilon
        ) - torch.sqrt((after_draw - target).square() + epsilon * epsilon)
        weights = mask.to(dtype=difference.dtype) / (
            float(target_context.mask_count)
            * local_probabilities.to(device=difference.device, dtype=difference.dtype)
        )
        contributions = difference * weights
        # As in exact mode, perform scalar reductions in FP64 so streamed
        # and cached paths have a stable signed decomposition.  ``contrib``
        # remains a pointwise tensor in the requested dtype for the nonlinear
        # decoder; widening here is not a change to the scientific objective.
        contributions64 = contributions.to(dtype=torch.float64)
        total_raw += float(contributions64.sum().item())
        total_benefit += float(
            (difference.to(dtype=torch.float64).clamp_min(0.0)
             * weights.to(dtype=torch.float64)).sum().item()
        )
        total_harm += float(
            ((-difference.to(dtype=torch.float64)).clamp_min(0.0)
             * weights.to(dtype=torch.float64)).sum().item()
        )
        total_square += float(contributions64.square().sum().item())
        valid_count += int(mask.sum().item())
        unique_decoded += int(unique_ids.shape[0])
    mean = total_raw / float(q_draws)
    variance = max(
        (total_square / float(q_draws)) - mean * mean,
        0.0,
    ) * (float(q_draws) / float(max(q_draws - 1, 1)))
    if counters is not None:
        counters.add(unique_decoded_queries=unique_decoded)
    return (
        mean,
        total_benefit / float(q_draws),
        total_harm / float(q_draws),
        valid_count,
        unique_decoded,
        variance,
        math.sqrt(variance / float(q_draws)),
    )


def measure_actions(
    completed_trace: CompletedBehaviorTrace,
    proposals: object,
    target_context: ValidatedTargetContext,
    decoder: object,
    teacher_config: EffectTeacherConfig,
    *,
    lattice: PFGRQueryLattice | None = None,
    chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
    candidate_chunk_size: int = 1,
    seed: int | None = None,
    counters: OperationCounters | None = None,
    observation_context: object | None = None,
    audit_target_context: bool = False,
) -> list[GainLabel]:
    """Measure fixed proposals after a sealed target-free behavior trace.

    ``exact_footprint`` evaluates every unique union voxel.  ``iid_fixed_q``
    samples the complete union with the declared plane-mixture law and keeps
    duplicate draws in the weighted estimator.  Both modes retain signed raw
    gains and benefit/harm metadata; this function is always detached/no-grad.
    ``candidate_chunk_size`` bounds the number of actions in one nonlinear
    decoder work queue (the actual group can be smaller when the measured
    query/delta working set reaches its 64 MiB operational bound).
    ``audit_target_context=True`` requests the explicit full target/mask byte
    digest; the default validates immutable metadata and tensor version guards
    without rehashing detached volumes for every state/action call.
    """

    from .config import EffectTeacherConfig
    from .types import CompletedBehaviorTrace, GainLabel

    if not isinstance(completed_trace, CompletedBehaviorTrace):
        raise TypeError("completed_trace must be a CompletedBehaviorTrace")
    if not completed_trace.sealed:
        raise ValueError("measure_actions requires a sealed target-free behavior trace")
    if not isinstance(target_context, ValidatedTargetContext):
        raise TypeError("target_context must be ValidatedTargetContext")
    if not isinstance(teacher_config, EffectTeacherConfig):
        raise TypeError("teacher_config must be EffectTeacherConfig")
    if not isinstance(audit_target_context, bool):
        raise TypeError("audit_target_context must be bool")
    if target_context.engineering_only:
        if observation_context is not None:
            raise ValueError(
                "engineering_only target contexts cannot be paired with observation_context"
            )
    elif target_context.observation_context_id is None:
        raise ValueError(
            "production target context is missing its ObservationContext binding"
        )
    elif observation_context is not None:
        from .types import ObservationContext

        if not isinstance(observation_context, ObservationContext):
            raise TypeError("observation_context must be ObservationContext")
        observation_context.validate_integrity()
        if observation_context.context_id != target_context.observation_context_id:
            raise ValueError("observation context identity does not match target context")
        if (
            observation_context.producer.compatibility_hash
            != target_context.observation_context_producer_hash
        ):
            raise ValueError("observation context producer identity does not match target context")
        if observation_context.geometry != target_context.output_geometry:
            raise ValueError("observation context geometry does not match target context")
        if observation_context.feature_geometry != target_context.feature_geometry:
            raise ValueError("observation context feature geometry does not match target context")
        if not torch.equal(observation_context.observation_mask, target_context.observation_mask):
            raise ValueError("observation context mask does not match target context")
    if target_context.mask_definition != teacher_config.mask_definition:
        raise ValueError("target context mask definition does not match teacher config")
    if target_context.label_definition not in {
        LABEL_DEFINITION,
        teacher_config.label_definition,
    }:
        raise ValueError(
            "target context label definition does not match teacher config"
        )
    _positive_int("chunk_size", chunk_size)
    _positive_int("candidate_chunk_size", candidate_chunk_size)
    target_context.validate_integrity(full_audit=audit_target_context)
    if counters is not None and audit_target_context:
        counters.add(target_validations=1)
    actions = _coerce_actions(proposals)
    if not actions:
        return []
    state_by_version = {state.state_version: state for state in completed_trace.states}
    # Complete identity validation is paid once per immutable trace/context,
    # then hot-path chunk loops use tensor version guards and stored digests.
    for state in state_by_version.values():
        state.validate_integrity()
    if (
        target_context.trace_route_hash is not None
        and completed_trace.route_hash != target_context.trace_route_hash
    ):
        raise ValueError("target context is bound to a different behavior trace")
    trace_producer_hashes = {
        state.producer.digest for state in state_by_version.values()
    }
    if len(trace_producer_hashes) != 1:
        raise ValueError("completed trace contains mixed producer identities")
    trace_producer_hash = next(iter(trace_producer_hashes))
    if (
        target_context.producer_compatibility_hash is not None
        and target_context.producer_compatibility_hash != trace_producer_hash
    ):
        raise ValueError("target context producer identity does not match trace")
    expected_label_hash = _expected_label_identity(teacher_config)
    trace_producer = state_by_version[min(state_by_version)].producer
    if (
        target_context.normalization_hash is not None
        and target_context.normalization_hash
        != trace_producer.observation_normalization_hash
    ):
        raise ValueError(
            "target context normalization identity does not match trace producer"
        )
    if trace_producer.label_definition_hash != expected_label_hash:
        raise ValueError(
            "teacher label definition is stale or incompatible with trace producer"
        )
    decoder_identity = _module_digest(decoder)
    if trace_producer.decoder_hash != decoder_identity:
        raise ValueError(
            "decoder weights are stale or incompatible with trace producer"
        )
    first_state = state_by_version[min(state_by_version)]
    with torch.no_grad():
        resolved_lattice = _resolve_lattice(
            target_context,
            first_state.planes,
            lattice=lattice,
            chunk_size=chunk_size,
        )
        records: list[_CandidateProbe] = []
        labelled_states: set[int] = set()
        for action_index, action in enumerate(actions):
            action.validate_integrity()
            if (
                action.context_id != completed_trace.context_id
                or action.context_id != target_context.completed_context_id
            ):
                raise ValueError("action/trace/target context IDs must match")
            state = state_by_version.get(action.state_version)
            if state is None or state.state_digest != action.state_digest:
                raise ValueError(
                    "action state identity is not present in completed trace"
                )
            if action.producer_compatibility_hash != state.producer.digest:
                raise ValueError(
                    "action producer identity is stale or incompatible with trace state"
                )
            if action.updater_producer_hash != state.producer.updater_hash:
                raise ValueError(
                    "action updater producer identity is stale or incompatible with trace state"
                )
            if action.query_version not in {
                resolved_lattice.query_version,
                "pfgr-lite-point-query-v1",
            }:
                raise ValueError(
                    "action query version is stale or incompatible with lattice"
                )
            if action.query_hash not in {
                resolved_lattice.geometry_hash,
                trace_producer.geometry_query_version_hash,
                POINT_QUERY_HASH,
            }:
                raise ValueError(
                    "action query hash is stale or incompatible with lattice"
                )
            if action.geometry_hash not in {
                resolved_lattice.geometry_hash,
                _expected_geometry_identity(resolved_lattice),
            }:
                raise ValueError(
                    "action geometry affine identity is stale or incompatible with lattice"
                )
            if action.writer_version != "compact-writeback-4mm-v1":
                raise ValueError(
                    "action writer version is stale or incompatible with PFGR writer"
                )
            if action.writer_hash != trace_producer.writer_hash:
                raise ValueError(
                    "action writer identity is stale or incompatible with trace producer"
                )
            if (
                action.delta.dtype != state.planes.xy.dtype
                or action.delta.device != state.planes.xy.device
            ):
                raise TypeError("action delta must match traced state dtype/device")
            if resolved_lattice.query_dtype != state.planes.xy.dtype:
                raise TypeError("lattice dtype must match traced state")
            footprint = build_footprint(resolved_lattice, action, chunk_size=chunk_size)
            if counters is not None:
                counters.add(
                    candidate_labels=1, footprint_unique_voxels=footprint.union_size
                )
                # Keep legacy local-support accounting separate from the
                # complete tri-plane footprint.  These rows are diagnostics
                # only and never define the sparse effect domain.
                from ..reward_supervision import build_local_support_samples

                local_samples = build_local_support_samples(
                    action.point_ras_mm.reshape(1, 3), resolved_lattice.output_geometry
                )
                counters.add(
                    exact_sphere_valid_voxels=int(
                        local_samples.valid_mask.sum().item()
                    ),
                    padded_cube_slots=int(local_samples.valid_mask.numel()),
                )
                if action.state_version not in labelled_states:
                    labelled_states.add(action.state_version)
                    counters.add(states_labeled=1)
            records.append(
                _CandidateProbe(
                    action=action,
                    state=state,
                    footprint=footprint,
                    action_index=action_index,
                    ids=footprint.voxel_ids_dhw,
                    sorted_linear=torch.empty((0,), dtype=torch.long),
                )
            )

        def _append_label(
            probe: _CandidateProbe,
            values: tuple[float, ...],
        ) -> GainLabel:
            action = probe.action
            if teacher_config.mode == "exact_footprint":
                raw, benefit, harm, valid_count, footprint_count = values
                return GainLabel(
                    action_id=action.action_id,
                    context_id=action.context_id,
                    state_version=action.state_version,
                    raw_gain=raw,
                    benefit=benefit,
                    harm=harm,
                    mask_count=target_context.mask_count,
                    role="exact_footprint",
                    q_draws=0,
                    variance=0.0,
                    standard_error=0.0,
                    footprint_voxels=int(footprint_count),
                    valid_masked_contributions=int(valid_count),
                    sampler_law="exact_union_v1",
                    label_definition=LABEL_DEFINITION,
                )
            raw, benefit, harm, valid_count, _unique_count, variance, standard_error, action_seed = values
            return GainLabel(
                action_id=action.action_id,
                context_id=action.context_id,
                state_version=action.state_version,
                raw_gain=benefit - harm,
                benefit=benefit,
                harm=harm,
                mask_count=target_context.mask_count,
                role="iid_fixed_q",
                q_draws=int(teacher_config.q_draws),
                seed=int(action_seed),
                variance=variance,
                standard_error=standard_error,
                footprint_voxels=probe.footprint.union_size,
                valid_masked_contributions=int(valid_count),
                sampler_law="iid_fixed_q_plane_mixture_c_over_S_v1",
                label_definition=LABEL_DEFINITION,
            )

        labels: list[GainLabel] = []
        if candidate_chunk_size == 1:
            # Reference path: one candidate at a time, preserving the exact
            # legacy counters and cache behavior used by existing tests.
            for probe in records:
                action = probe.action
                if teacher_config.mode == "exact_footprint":
                    values = _evaluate_exact(
                        resolved_lattice,
                        probe.footprint,
                        probe.state.planes,
                        action,
                        target_context,
                        decoder,
                        chunk_size=chunk_size,
                        epsilon=float(teacher_config.epsilon),
                        counters=counters,
                        state_identity=probe.state.state_digest,
                        decoder_identity=decoder_identity,
                    )
                else:
                    q_draws = int(teacher_config.q_draws)
                    action_seed = _seed_for_action(action, seed, probe.action_index)
                    raw, benefit, harm, valid_count, unique_count, variance, standard_error = _evaluate_fixed_q(
                        resolved_lattice,
                        probe.footprint,
                        probe.state.planes,
                        action,
                        target_context,
                        decoder,
                        q_draws=q_draws,
                        seed=action_seed,
                        chunk_size=chunk_size,
                        epsilon=float(teacher_config.epsilon),
                        counters=counters,
                        state_identity=probe.state.state_digest,
                        decoder_identity=decoder_identity,
                    )
                    values = (
                        raw,
                        benefit,
                        harm,
                        valid_count,
                        unique_count,
                        variance,
                        standard_error,
                        action_seed,
                    )
                labels.append(_append_label(probe, tuple(values)))
            return labels

        # Batched candidate mode combines per-action query chunks into a
        # bounded nonlinear decoder batch.  The requested candidate count is
        # a maximum; byte accounting may split it further, and a single huge
        # candidate falls back to the exact stream evaluator.
        for group in _candidate_groups(
            records,
            candidate_chunk_size=candidate_chunk_size,
            dtype=first_state.planes.xy.dtype,
        ):
            estimate = sum(
                _candidate_rows_bytes(
                    int(probe.footprint.union_size), first_state.planes.xy.dtype
                )
                for probe in group
            )
            if estimate > CANDIDATE_BATCH_MAX_BYTES and len(group) == 1:
                probe = group[0]
                action = probe.action
                if teacher_config.mode == "exact_footprint":
                    values = _evaluate_exact(
                        resolved_lattice,
                        probe.footprint,
                        probe.state.planes,
                        action,
                        target_context,
                        decoder,
                        chunk_size=chunk_size,
                        epsilon=float(teacher_config.epsilon),
                        counters=counters,
                        state_identity=probe.state.state_digest,
                        decoder_identity=decoder_identity,
                    )
                else:
                    q_draws = int(teacher_config.q_draws)
                    action_seed = _seed_for_action(action, seed, probe.action_index)
                    raw, benefit, harm, valid_count, unique_count, variance, standard_error = _evaluate_fixed_q(
                        resolved_lattice,
                        probe.footprint,
                        probe.state.planes,
                        action,
                        target_context,
                        decoder,
                        q_draws=q_draws,
                        seed=action_seed,
                        chunk_size=chunk_size,
                        epsilon=float(teacher_config.epsilon),
                        counters=counters,
                        state_identity=probe.state.state_digest,
                        decoder_identity=decoder_identity,
                    )
                    values = (
                        raw,
                        benefit,
                        harm,
                        valid_count,
                        unique_count,
                        variance,
                        standard_error,
                        action_seed,
                    )
                labels.append(_append_label(probe, tuple(values)))
                continue
            if teacher_config.mode == "exact_footprint":
                batch_values = _evaluate_exact_candidate_batch(
                    resolved_lattice,
                    group,
                    target_context,
                    decoder,
                    chunk_size=chunk_size,
                    epsilon=float(teacher_config.epsilon),
                    counters=counters,
                    decoder_identity=decoder_identity,
                )
            else:
                batch_values = _evaluate_fixed_q_candidate_batch(
                    resolved_lattice,
                    group,
                    target_context,
                    decoder,
                    q_draws=int(teacher_config.q_draws),
                    seed=seed,
                    chunk_size=chunk_size,
                    epsilon=float(teacher_config.epsilon),
                    counters=counters,
                    decoder_identity=decoder_identity,
                )
            labels.extend(
                _append_label(probe, tuple(values))
                for probe, values in zip(group, batch_values)
            )
        return labels


__all__ = [
    "CANDIDATE_BATCH_MAX_BYTES",
    "DEFAULT_CHARBONNIER_EPSILON",
    "LABEL_DEFINITION",
    "PROBE_CACHE_MAX_BYTES",
    "PROBE_CACHE_MAX_ENTRIES",
    "TARGET_CONTEXT_VERSION",
    "TEACHER_VERSION",
    "ValidatedTargetContext",
    "clear_target_validation_stats",
    "clear_teacher_cache",
    "measure_actions",
    "reference_full_write",
    "target_validation_stats",
    "teacher_cache_stats",
    "validate_target",
]
