"""Canonical PFGR-Lite source-voxel query lattice.

This module owns the small, geometry-only seam shared by the PFGR final
decoder and the later sparse writer.  A lattice row represents one output
voxel centre.  For each of the XY, XZ, and YZ planes it stores the four
unclipped bilinear neighbours, their half-voxel weights, and explicit validity
flags.  Invalid neighbours use a safe gather address and are explicitly
masked to zero at query time; no border clamping is used.

The implementation deliberately does not import PFGR action, trace, teacher,
bank, or policy declarations.  It can therefore be developed and tested
while those contracts are being revised by another worker.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import torch
from torch import Tensor
from torch.nn import functional as F

from ..contracts import VolumeGeometry
from ..spectral_query import FeatureGridGeometry
from ..state_init import DynamicTriPlanes

QUERY_LATTICE_VERSION = "pfgr-lite-query-lattice-v1"
"""Version of the source-voxel-to-triplane stencil algebra."""

PLANE_NAMES: tuple[str, str, str] = ("xy", "xz", "yz")
"""Permanent plane/provenance order used by every query result."""

DEFAULT_MEMORY_BOUND_BYTES = 512 * 1024 * 1024
"""Operational default for materialized stencils plus inverse index."""

CACHE_MAX_ENTRIES = 8
"""Maximum number of geometry/dtype/chunk lattice entries retained process-wide."""

CACHE_MAX_BYTES = 512 * 1024 * 1024
"""Maximum retained bytes across all process-level lattice cache entries."""


def _positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_query_dtype(query_dtype: torch.dtype) -> None:
    if query_dtype not in (torch.float32, torch.float64):
        raise TypeError("query_dtype must be torch.float32 or torch.float64")


def _shape(name: str, value: Sequence[int]) -> tuple[int, int, int]:
    result = tuple(value)
    if len(result) != 3 or any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0
        for item in result
    ):
        raise ValueError(
            f"{name} must contain three positive integers in [D,H,W] order"
        )
    return result  # type: ignore[return-value]


def _json_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _geometry_payload(geometry: VolumeGeometry) -> dict[str, object]:
    return {
        "shape_dhw": list(geometry.shape_dhw),
        "voxel_to_ras_mm": [list(row) for row in geometry.voxel_to_ras_mm],
    }


def _feature_geometry_payload(geometry: FeatureGridGeometry) -> dict[str, object]:
    # Use the declared fields rather than dataclass repr.  The scale/offset
    # and operator chain are part of the centre-transform identity even when
    # the composed affine happens to look similar.
    return {
        "source_geometry": _geometry_payload(geometry.source_geometry),
        "feature_geometry": _geometry_payload(geometry.feature_geometry),
        "tap": geometry.tap,
        "feature_to_source_scale_dhw": list(geometry.feature_to_source_scale_dhw),
        "feature_to_source_offset_dhw": list(geometry.feature_to_source_offset_dhw),
        "operator_chain": list(geometry.operator_chain),
    }


def _tensor_digest(tensor: Tensor) -> str:
    """Hash tensor metadata and canonical CPU bytes without object repr."""

    cpu = tensor.detach().to(device="cpu").contiguous()
    header = f"{cpu.dtype!s}|{tuple(cpu.shape)!r}|".encode()
    return hashlib.sha256(header + cpu.numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class BilinearStencil:
    """Four-neighbour stencil for one plane and every output voxel.

    ``neighbour_indices`` stores flattened plane-node indices in row-major
    order.  It intentionally retains negative/out-of-range values; ``valid``
    is the sole authority for zero padding.  The slot order is
    ``(row0,col0), (row0,col1), (row1,col0), (row1,col1)``.
    """

    neighbour_indices: Tensor  # [V,4], int64, unclipped
    weights: Tensor  # [V,4], query dtype
    valid: Tensor  # [V,4], bool
    rows: int
    columns: int

    def __post_init__(self) -> None:
        if (
            self.neighbour_indices.ndim != 2
            or self.neighbour_indices.shape[-1] != 4
            or self.neighbour_indices.dtype != torch.long
        ):
            raise ValueError(
                "neighbour_indices must have shape [V,4] and dtype torch.long"
            )
        if (
            self.weights.shape != self.neighbour_indices.shape
            or self.weights.dtype not in (torch.float32, torch.float64)
        ):
            raise ValueError(
                "weights must match neighbour_indices with fp32/fp64 dtype"
            )
        if (
            self.valid.shape != self.neighbour_indices.shape
            or self.valid.dtype != torch.bool
        ):
            raise ValueError("valid must match neighbour_indices with bool dtype")
        _positive_int("rows", self.rows)
        _positive_int("columns", self.columns)
        if not bool(torch.isfinite(self.weights).all()):
            raise ValueError("bilinear weights must be finite")

    @property
    def voxel_count(self) -> int:
        return int(self.neighbour_indices.shape[0])

    @property
    def indices(self) -> Tensor:
        """Short alias retained for low-level writer consumers."""

        return self.neighbour_indices

    @property
    def neighbor_indices(self) -> Tensor:
        """US-spelling alias for consumers using American terminology."""

        return self.neighbour_indices

    @property
    def valid_flags(self) -> Tensor:
        """Explicit spelling of the per-neighbour zero-padding flags."""

        return self.valid


@dataclass(frozen=True)
class PlaneNodeInverseIndex:
    """CSR node-to-output-voxel positive-support inverse index."""

    offsets: Tensor  # [nodes+1], int64
    voxel_linear: Tensor  # [E], int64
    weights: Tensor  # [E], query dtype
    node_count: int

    def __post_init__(self) -> None:
        if (
            self.offsets.ndim != 1
            or self.offsets.dtype != torch.long
            or self.offsets.shape[0] != self.node_count + 1
        ):
            raise ValueError(
                "inverse offsets must have shape [node_count+1] and dtype torch.long"
            )
        if self.voxel_linear.ndim != 1 or self.voxel_linear.dtype != torch.long:
            raise ValueError("inverse voxel_linear must be int64 [E]")
        if self.weights.shape != self.voxel_linear.shape or self.weights.dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise ValueError("inverse weights must align with voxel_linear")
        if not bool(torch.isfinite(self.weights).all()) or bool(
            (self.weights <= 0).any()
        ):
            raise ValueError("inverse weights must be finite and strictly positive")
        if self.offsets.numel() and int(self.offsets[0]) != 0:
            raise ValueError("inverse offsets must start at zero")
        if self.offsets.numel() and int(self.offsets[-1]) != self.voxel_linear.numel():
            raise ValueError("inverse offsets must terminate at edge count")

    @property
    def edge_count(self) -> int:
        return int(self.voxel_linear.numel())

    @property
    def node_ids(self) -> Tensor:
        """Expanded node IDs aligned with ``voxel_linear`` and ``weights``."""

        counts = self.offsets[1:] - self.offsets[:-1]
        return torch.repeat_interleave(
            torch.arange(self.node_count, dtype=torch.long, device=self.offsets.device),
            counts,
        )


def _plane_shape(feature_shape_dhw: Sequence[int], plane: str) -> tuple[int, int]:
    depth, height, width = _shape("feature_shape_dhw", feature_shape_dhw)
    if plane == "xy":
        return height, width
    if plane == "xz":
        return depth, width
    if plane == "yz":
        return depth, height
    raise ValueError("plane must be 'xy', 'xz', or 'yz'")


def _output_voxel_ids(
    start: int,
    stop: int,
    *,
    shape_dhw: tuple[int, int, int],
    device: torch.device,
) -> Tensor:
    _, height, width = shape_dhw
    flat = torch.arange(start, stop, device=device, dtype=torch.long)
    area = height * width
    d = torch.div(flat, area, rounding_mode="floor")
    rem = flat - d * area
    h = torch.div(rem, width, rounding_mode="floor")
    w = rem - h * width
    return torch.stack((d, h, w), dim=-1)


def _stencil_from_coordinates(
    row: Tensor, column: Tensor, *, rows: int, columns: int
) -> BilinearStencil:
    """Build exact unclipped align_corners=False four-neighbour stencils."""

    row0 = torch.floor(row)
    col0 = torch.floor(column)
    row1 = row0 + 1.0
    col1 = col0 + 1.0
    row_weight = row - row0
    col_weight = column - col0

    # Convert only the integer lattice locations to int64.  There is no
    # clamping here: invalid locations are represented by valid=False and are
    # routed to an explicitly zero-masked gather during query.
    r0 = row0.to(dtype=torch.long)
    r1 = row1.to(dtype=torch.long)
    c0 = col0.to(dtype=torch.long)
    c1 = col1.to(dtype=torch.long)
    indices = torch.stack(
        (
            r0 * columns + c0,
            r0 * columns + c1,
            r1 * columns + c0,
            r1 * columns + c1,
        ),
        dim=-1,
    )
    valid = torch.stack(
        (
            (r0 >= 0) & (r0 < rows) & (c0 >= 0) & (c0 < columns),
            (r0 >= 0) & (r0 < rows) & (c1 >= 0) & (c1 < columns),
            (r1 >= 0) & (r1 < rows) & (c0 >= 0) & (c0 < columns),
            (r1 >= 0) & (r1 < rows) & (c1 >= 0) & (c1 < columns),
        ),
        dim=-1,
    )
    weights = torch.stack(
        (
            (1.0 - row_weight) * (1.0 - col_weight),
            (1.0 - row_weight) * col_weight,
            row_weight * (1.0 - col_weight),
            row_weight * col_weight,
        ),
        dim=-1,
    )
    return BilinearStencil(indices, weights, valid, rows, columns)


class PFGRQueryLattice:
    """Canonical source-voxel to dynamic-triplane query lattice.

    The lattice is immutable at the metadata level and guarded against
    in-place tensor mutation.  ``build`` caches CPU stencils by full geometry,
    algorithm version, dtype, chunking, and memory-bound identity.  Non-CPU
    queries derive bounded chunks on their plane device instead of retaining
    unbounded full-lattice device mirrors.
    """

    _CACHE: ClassVar[OrderedDict[tuple[object, ...], PFGRQueryLattice]] = OrderedDict()
    _CACHE_BYTES: ClassVar[int] = 0
    _CACHE_EVICTIONS: ClassVar[int] = 0

    def __init__(
        self,
        *,
        output_geometry: VolumeGeometry,
        feature_geometry: FeatureGridGeometry,
        query_dtype: torch.dtype,
        build_chunk_size: int,
        memory_bound_bytes: int,
        plane_stencils: Mapping[str, BilinearStencil] | None,
        inverse_indices: Mapping[str, PlaneNodeInverseIndex] | None,
        memory_accounting: Mapping[str, int | float | str | bool],
        build_elapsed_seconds: float,
    ) -> None:
        object.__setattr__(self, "output_geometry", output_geometry)
        object.__setattr__(self, "feature_geometry", feature_geometry)
        object.__setattr__(self, "query_dtype", query_dtype)
        object.__setattr__(self, "build_chunk_size", build_chunk_size)
        object.__setattr__(self, "memory_bound_bytes", memory_bound_bytes)
        object.__setattr__(self, "query_version", QUERY_LATTICE_VERSION)
        object.__setattr__(self, "output_shape_dhw", tuple(output_geometry.shape_dhw))
        object.__setattr__(self, "feature_shape_dhw", tuple(feature_geometry.shape_dhw))
        output_hash = _json_digest(_geometry_payload(output_geometry))
        feature_hash = _json_digest(_feature_geometry_payload(feature_geometry))
        object.__setattr__(self, "output_geometry_hash", output_hash)
        object.__setattr__(self, "feature_geometry_hash", feature_hash)
        object.__setattr__(
            self,
            "geometry_hash",
            _json_digest({"output": output_hash, "feature": feature_hash}),
        )
        plane_map = {} if plane_stencils is None else dict(plane_stencils)
        inverse_map = {} if inverse_indices is None else dict(inverse_indices)
        object.__setattr__(self, "_plane_stencils", MappingProxyType(plane_map))
        object.__setattr__(self, "_inverse_indices", MappingProxyType(inverse_map))
        object.__setattr__(
            self,
            "_operation_counters",
            {
                "query_calls": 0,
                "query_voxel_count": 0,
                "query_stencil_voxel_count": 0,
                "query_stencil_bytes": 0,
                "stencil_transfer_calls": 0,
                "stencil_transfer_voxel_count": 0,
                "stencil_transfer_bytes": 0,
                "validation_fast_calls": 0,
                "validation_fast_seconds": 0.0,
                "validation_full_calls": 0,
                "validation_full_seconds": 0.0,
                "scan_calls": 0,
                "scanned_voxel_count": 0,
                "scanned_bytes": 0,
                "scan_peak_bytes": 0,
                "scan_seconds": 0.0,
                "cache_hit_count": 0,
                "build_call_count": 1,
            },
        )
        object.__setattr__(self, "_build_elapsed_seconds", float(build_elapsed_seconds))
        accounting = dict(memory_accounting)
        accounting.setdefault("query_version", QUERY_LATTICE_VERSION)
        accounting.setdefault("memory_bound_bytes", memory_bound_bytes)
        object.__setattr__(self, "_memory_accounting", MappingProxyType(accounting))
        object.__setattr__(self, "_tensor_versions", self._capture_tensor_versions())
        object.__setattr__(self, "_integrity_digest", self._compute_integrity_digest())
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("PFGRQueryLattice metadata is immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear process-local lattice cache (primarily for isolated tests)."""

        cls._CACHE.clear()
        cls._CACHE_BYTES = 0
        cls._CACHE_EVICTIONS = 0

    @classmethod
    def cache_stats(cls) -> Mapping[str, int]:
        """Return explicit process-level cache retention and eviction stats."""

        cls._CACHE_BYTES = sum(
            int(item.memory_accounting["cache_entry_bytes"])
            for item in cls._CACHE.values()
        )
        return MappingProxyType(
            {
                "entries": len(cls._CACHE),
                "retained_bytes": cls._CACHE_BYTES,
                "max_entries": CACHE_MAX_ENTRIES,
                "max_bytes": CACHE_MAX_BYTES,
                "evictions": cls._CACHE_EVICTIONS,
            }
        )

    def release(self) -> bool:
        """Release this lattice from the process cache without invalidating it.

        The caller may continue using an already-held object; subsequent
        ``build`` calls will construct a fresh entry.  Returning ``True``
        distinguishes an actual retained-entry release from a cache miss.
        """

        for key, candidate in tuple(self._CACHE.items()):
            if candidate is self:
                del self._CACHE[key]
                self.__class__._CACHE_BYTES = max(
                    0,
                    self.__class__._CACHE_BYTES
                    - int(self.memory_accounting["cache_entry_bytes"]),
                )
                return True
        return False

    release_cache_entry = release

    @classmethod
    def _cache_key(
        cls,
        output_geometry: VolumeGeometry,
        feature_geometry: FeatureGridGeometry,
        *,
        query_dtype: torch.dtype,
        build_chunk_size: int,
        memory_bound_bytes: int,
    ) -> tuple[object, ...]:
        output_hash = _json_digest(_geometry_payload(output_geometry))
        feature_hash = _json_digest(_feature_geometry_payload(feature_geometry))
        return (
            QUERY_LATTICE_VERSION,
            output_hash,
            feature_hash,
            str(query_dtype),
            build_chunk_size,
            memory_bound_bytes,
            "cpu-stencil-v1",
        )

    @classmethod
    def build(
        cls,
        output_geometry: VolumeGeometry,
        feature_geometry: FeatureGridGeometry,
        *,
        query_dtype: torch.dtype,
        build_chunk_size: int,
        memory_bound_bytes: int | None = DEFAULT_MEMORY_BOUND_BYTES,
    ) -> PFGRQueryLattice:
        """Build (or retrieve) a canonical lattice for one geometry pair.

        ``memory_bound_bytes`` is an operational guard for materialized CPU
        stencils, inverse-index arrays, and conservative inverse-build
        workspace.  If the bound would be exceeded, the lattice remains exact but uses
        ``footprint_mode='full_scan_fallback'`` and records its scan work.
        ``None`` means no explicit bound; the default is 512 MiB.
        """

        if not isinstance(output_geometry, VolumeGeometry):
            raise TypeError("output_geometry must be a VolumeGeometry")
        if not isinstance(feature_geometry, FeatureGridGeometry):
            raise TypeError("feature_geometry must be a FeatureGridGeometry")
        if output_geometry != feature_geometry.source_geometry:
            raise ValueError(
                "output_geometry must equal feature_geometry.source_geometry"
            )
        _validate_query_dtype(query_dtype)
        build_chunk_size = _positive_int("build_chunk_size", build_chunk_size)
        if memory_bound_bytes is None:
            memory_bound_bytes = (1 << 63) - 1
        elif (
            not isinstance(memory_bound_bytes, int)
            or isinstance(memory_bound_bytes, bool)
            or memory_bound_bytes <= 0
        ):
            raise ValueError("memory_bound_bytes must be a positive integer or None")

        key = cls._cache_key(
            output_geometry,
            feature_geometry,
            query_dtype=query_dtype,
            build_chunk_size=build_chunk_size,
            memory_bound_bytes=memory_bound_bytes,
        )
        cls._CACHE_BYTES = sum(
            int(item.memory_accounting["cache_entry_bytes"])
            for item in cls._CACHE.values()
        )
        cached = cls._CACHE.get(key)
        if cached is not None:
            try:
                cached.validate_integrity()
            except RuntimeError:
                del cls._CACHE[key]
                cls._CACHE_BYTES = max(
                    0,
                    cls._CACHE_BYTES
                    - int(cached.memory_accounting["cache_entry_bytes"]),
                )
            else:
                cls._CACHE.move_to_end(key)
                cached._operation_counters["cache_hit_count"] += 1
                cached._operation_counters["build_call_count"] += 1
                return cached

        started = time.perf_counter()
        output_shape = tuple(output_geometry.shape_dhw)
        voxel_count = math.prod(output_shape)
        dtype_bytes = torch.empty((), dtype=query_dtype).element_size()
        # Three planes × four slots.  Index/weight/valid are the actual
        # materialized arrays, not an estimated feature-volume clone.
        forward_estimate = 3 * voxel_count * 4 * (8 + dtype_bytes + 1)
        build_chunk_voxels = min(build_chunk_size, voxel_count)
        stencil_build_workspace = build_chunk_voxels * 3 * (
            8 + dtype_bytes
        ) + 3 * build_chunk_voxels * 4 * (8 + dtype_bytes + 1)
        # Forward arrays are preallocated before chunk temporaries.  Refuse
        # indexed materialization when that bounded construction peak itself
        # would exceed the declared limit.
        materialize_stencils = (
            forward_estimate + stencil_build_workspace <= memory_bound_bytes
        )

        plane_stencils: dict[str, BilinearStencil] | None = None
        if materialize_stencils:
            plane_shapes = {
                name: _plane_shape(feature_geometry.shape_dhw, name)
                for name in PLANE_NAMES
            }
            index_buffers = {
                name: torch.empty((voxel_count, 4), dtype=torch.long)
                for name in PLANE_NAMES
            }
            weight_buffers = {
                name: torch.empty((voxel_count, 4), dtype=query_dtype)
                for name in PLANE_NAMES
            }
            valid_buffers = {
                name: torch.empty((voxel_count, 4), dtype=torch.bool)
                for name in PLANE_NAMES
            }
            for start in range(0, voxel_count, build_chunk_size):
                stop = min(start + build_chunk_size, voxel_count)
                ids = _output_voxel_ids(
                    start, stop, shape_dhw=output_shape, device=torch.device("cpu")
                )
                coordinates = cls._feature_coordinates(
                    ids,
                    output_geometry=output_geometry,
                    feature_geometry=feature_geometry,
                    dtype=query_dtype,
                )
                d, h, w = coordinates.unbind(dim=-1)
                _chunk_stencils = {
                    "xy": _stencil_from_coordinates(
                        h,
                        w,
                        rows=plane_shapes["xy"][0],
                        columns=plane_shapes["xy"][1],
                    ),
                    "xz": _stencil_from_coordinates(
                        d,
                        w,
                        rows=plane_shapes["xz"][0],
                        columns=plane_shapes["xz"][1],
                    ),
                    "yz": _stencil_from_coordinates(
                        d,
                        h,
                        rows=plane_shapes["yz"][0],
                        columns=plane_shapes["yz"][1],
                    ),
                }
                for name, stencil in _chunk_stencils.items():
                    index_buffers[name][start:stop].copy_(stencil.neighbour_indices)
                    weight_buffers[name][start:stop].copy_(stencil.weights)
                    valid_buffers[name][start:stop].copy_(stencil.valid)

            plane_stencils = {
                name: BilinearStencil(
                    neighbour_indices=index_buffers[name],
                    weights=weight_buffers[name],
                    valid=valid_buffers[name],
                    rows=plane_shapes[name][0],
                    columns=plane_shapes[name][1],
                )
                for name in PLANE_NAMES
            }

        # Once forward stencils are retained, count their actual positive
        # edges before deciding whether the CSR inverse fits.  This avoids a
        # pessimistic all-four-neighbours estimate while still allocating no
        # inverse arrays before the bound check.
        inverse_estimate = 0
        inverse_workspace_estimate = 0
        if plane_stencils is not None:
            for name in PLANE_NAMES:
                stencil = plane_stencils[name]
                edge_count = int((stencil.valid & (stencil.weights > 0)).sum().item())
                retained = (stencil.rows * stencil.columns + 1) * 8 + edge_count * (
                    8 + dtype_bytes
                )
                inverse_estimate += retained
                # _build_inverse preallocates node/voxel/weight arrays, then
                # creates an argsort permutation and sorted voxel/weight
                # copies.  Reserve a conservative peak for that workspace in
                # addition to the final CSR arrays.
                edge_payload = edge_count * (8 + 8 + dtype_bytes)
                inverse_workspace_estimate += retained + edge_payload + 8 * edge_count
        materialize_inverse = (
            plane_stencils is not None
            and max(
                forward_estimate + stencil_build_workspace,
                forward_estimate + inverse_workspace_estimate,
            )
            <= memory_bound_bytes
        )

        inverse_indices: dict[str, PlaneNodeInverseIndex] | None = None
        if materialize_inverse:
            assert plane_stencils is not None
            inverse_indices = {
                name: cls._build_inverse(plane_stencils[name]) for name in PLANE_NAMES
            }

        build_elapsed = time.perf_counter() - started
        forward_bytes = cls._forward_stencil_bytes(plane_stencils)
        inverse_bytes = cls._inverse_index_bytes(inverse_indices)
        mode = "indexed" if inverse_indices is not None else "full_scan_fallback"
        # Full scan accounting names the actual source voxel rows visited for
        # all three planes.  It remains explicit even when forward stencils
        # were retained but the inverse index was refused by the bound.
        accounting: dict[str, int | float | str | bool] = {
            "forward_stencil_bytes": forward_bytes,
            "inverse_index_bytes": inverse_bytes,
            "inverse_retained_estimate": inverse_estimate,
            "inverse_workspace_bytes": inverse_workspace_estimate,
            "stencil_build_workspace_bytes": stencil_build_workspace,
            "materialized_stencil_bytes": forward_bytes,
            "materialized_index_bytes": inverse_bytes,
            "build_bytes": forward_bytes + inverse_bytes,
            "cache_entry_bytes": forward_bytes + inverse_bytes,
            "cache_max_entries": CACHE_MAX_ENTRIES,
            "cache_max_bytes": CACHE_MAX_BYTES,
            "peak_memory_bytes": max(
                forward_bytes + stencil_build_workspace
                if plane_stencils is not None
                else 0,
                forward_bytes + inverse_workspace_estimate
                if inverse_indices is not None
                else 0,
            ),
            "would_materialized_peak_bytes": forward_estimate
            + max(stencil_build_workspace, inverse_workspace_estimate),
            "peak_bound_exceeded": bool(
                forward_estimate
                + max(stencil_build_workspace, inverse_workspace_estimate)
                > memory_bound_bytes
            ),
            "build_elapsed_seconds": build_elapsed,
            "build_time_seconds": build_elapsed,
            "build_chunk_size": build_chunk_size,
            "output_voxel_count": voxel_count,
            "scanned_voxel_count": 0,
            "scanned_bytes": 0,
            "fallback_scanned_voxel_count": 0,
            "fallback_scanned_bytes": 0,
            "scan_peak_bytes": 0
            if mode == "indexed"
            else 3 * min(build_chunk_size, voxel_count) * 4 * (8 + dtype_bytes + 1),
            "footprint_mode": mode,
            "index_materialized": bool(inverse_indices is not None),
            "stencils_materialized": bool(plane_stencils is not None),
        }
        lattice = cls(
            output_geometry=output_geometry,
            feature_geometry=feature_geometry,
            query_dtype=query_dtype,
            build_chunk_size=build_chunk_size,
            memory_bound_bytes=memory_bound_bytes,
            plane_stencils=plane_stencils,
            inverse_indices=inverse_indices,
            memory_accounting=accounting,
            build_elapsed_seconds=build_elapsed,
        )
        entry_bytes = int(accounting["cache_entry_bytes"])
        if entry_bytes <= CACHE_MAX_BYTES:
            while cls._CACHE and (
                len(cls._CACHE) >= CACHE_MAX_ENTRIES
                or cls._CACHE_BYTES + entry_bytes > CACHE_MAX_BYTES
            ):
                _, evicted = cls._CACHE.popitem(last=False)
                cls._CACHE_BYTES = max(
                    0,
                    cls._CACHE_BYTES
                    - int(evicted.memory_accounting["cache_entry_bytes"]),
                )
                cls._CACHE_EVICTIONS += 1
            cls._CACHE[key] = lattice
            cls._CACHE_BYTES += entry_bytes
        return lattice

    @staticmethod
    def _feature_coordinates(
        voxel_ids_dhw: Tensor,
        *,
        output_geometry: VolumeGeometry,
        feature_geometry: FeatureGridGeometry,
        dtype: torch.dtype,
    ) -> Tensor:
        # Keep this affine path elementwise instead of using a batched
        # ``matmul``.  CPU/GPU BLAS kernels may choose a different reduction
        # tree for a short chunk versus a full-volume chunk, which can move a
        # near-boundary coordinate by one FP32 ulp and change ``floor``.  A
        # fixed scalar addition order gives every output row the same
        # canonical centre regardless of ``build_chunk_size``/``chunk_size``;
        # it remains the full source/feature affine (no axis shortcut).
        voxel = voxel_ids_dhw.to(dtype=dtype)
        d, h, w = voxel.unbind(dim=-1)

        def _affine_row(values: tuple[Tensor, Tensor, Tensor], row: int) -> Tensor:
            # VolumeGeometry stores affine columns in WHD order, while the
            # canonical lattice receives tensor DHW rows.  Evaluate each
            # component with an explicit, stable left-associated sum.
            x = values[2] * float(output_geometry.voxel_to_ras_mm[row][0])
            x = x + values[1] * float(output_geometry.voxel_to_ras_mm[row][1])
            x = x + values[0] * float(output_geometry.voxel_to_ras_mm[row][2])
            return x + float(output_geometry.voxel_to_ras_mm[row][3])

        xyz = torch.stack(
            tuple(_affine_row((d, h, w), row) for row in range(3)), dim=-1
        )
        affine = torch.as_tensor(
            feature_geometry.feature_geometry.voxel_to_ras_mm,
            dtype=dtype,
            device=xyz.device,
        )
        # ``feature_geometry`` maps feature DHW -> RAS.  Invert that full
        # affine once per chunk; the inverse is geometry-only and preserves
        # rotation, shear, anisotropic spacing and translation.
        inverse = torch.linalg.inv(affine)
        x, y, z = xyz.unbind(dim=-1)
        feature_values: list[Tensor] = []
        for row in range(3):
            value = x * float(inverse[row, 0].item())
            value = value + y * float(inverse[row, 1].item())
            value = value + z * float(inverse[row, 2].item())
            feature_values.append(value + float(inverse[row, 3].item()))
        # The affine consumes WHD and returns feature DHW.  Keep this explicit
        # permutation visible at the contract boundary.
        return torch.stack(
            (feature_values[2], feature_values[1], feature_values[0]), dim=-1
        )

    @staticmethod
    def _build_inverse(stencil: BilinearStencil) -> PlaneNodeInverseIndex:
        node_count = stencil.rows * stencil.columns
        masks = tuple(
            stencil.valid[:, slot] & (stencil.weights[:, slot] > 0) for slot in range(4)
        )
        edge_count = sum(int(mask.sum().item()) for mask in masks)
        nodes = torch.empty((edge_count,), dtype=torch.long)
        voxels = torch.empty((edge_count,), dtype=torch.long)
        weights = torch.empty((edge_count,), dtype=stencil.weights.dtype)
        voxel_ids = torch.arange(stencil.voxel_count, dtype=torch.long)
        cursor = 0
        for slot in range(4):
            positive = masks[slot]
            count = int(positive.sum().item())
            if count:
                stop = cursor + count
                nodes[cursor:stop] = stencil.neighbour_indices[positive, slot]
                voxels[cursor:stop] = voxel_ids[positive]
                weights[cursor:stop] = stencil.weights[positive, slot]
                cursor = stop
        counts = torch.bincount(nodes, minlength=node_count)
        order = torch.argsort(nodes)
        voxels = voxels[order]
        weights = weights[order]
        offsets = torch.cat(
            (torch.zeros((1,), dtype=torch.long), torch.cumsum(counts, dim=0))
        )
        return PlaneNodeInverseIndex(offsets, voxels, weights, node_count)

    @staticmethod
    def _forward_stencil_bytes(stencils: Mapping[str, BilinearStencil] | None) -> int:
        if stencils is None:
            return 0
        total = 0
        for stencil in stencils.values():
            total += sum(
                value.numel() * value.element_size()
                for value in (stencil.neighbour_indices, stencil.weights, stencil.valid)
            )
        return total

    @staticmethod
    def _inverse_index_bytes(
        indices: Mapping[str, PlaneNodeInverseIndex] | None,
    ) -> int:
        if indices is None:
            return 0
        total = 0
        for index in indices.values():
            total += sum(
                value.numel() * value.element_size()
                for value in (index.offsets, index.voxel_linear, index.weights)
            )
        return total

    @property
    def footprint_mode(self) -> str:
        return str(self._memory_accounting["footprint_mode"])

    @property
    def memory_accounting(self) -> Mapping[str, int | float | str | bool]:
        """Actual materialized-byte and fallback-scan accounting."""

        snapshot = dict(self._memory_accounting)
        snapshot.update(self._operation_counters)
        snapshot["fallback_scanned_voxel_count"] = self._operation_counters[
            "scanned_voxel_count"
        ]
        snapshot["fallback_scanned_bytes"] = self._operation_counters["scanned_bytes"]
        return MappingProxyType(snapshot)

    @property
    def build_elapsed_seconds(self) -> float:
        return self._build_elapsed_seconds

    @property
    def build_time_seconds(self) -> float:
        """Alias used by run-time accounting/reporting consumers."""

        return self._build_elapsed_seconds

    @property
    def scanned_voxel_count(self) -> int:
        return int(self._operation_counters["scanned_voxel_count"])

    @property
    def scanned_bytes(self) -> int:
        return int(self._operation_counters["scanned_bytes"])

    @property
    def scan_peak_bytes(self) -> int:
        return int(self._memory_accounting["scan_peak_bytes"])

    @property
    def dtype(self) -> torch.dtype:
        return self.query_dtype

    @property
    def version(self) -> str:
        return self.query_version

    @property
    def stencils(self) -> Mapping[str, BilinearStencil]:
        """Read-only snapshots of forward stencils.

        Returned tensors are clones so downstream inspection cannot mutate the
        cache; the owned private mapping remains guarded by version checks.
        """

        return MappingProxyType(
            {
                name: BilinearStencil(
                    stencil.neighbour_indices.clone(),
                    stencil.weights.clone(),
                    stencil.valid.clone(),
                    stencil.rows,
                    stencil.columns,
                )
                for name, stencil in self._plane_stencils.items()
            }
        )

    @property
    def plane_stencils(self) -> Mapping[str, BilinearStencil]:
        return self.stencils

    @property
    def inverse_indices(self) -> Mapping[str, PlaneNodeInverseIndex]:
        return MappingProxyType(
            {
                name: PlaneNodeInverseIndex(
                    index.offsets.clone(),
                    index.voxel_linear.clone(),
                    index.weights.clone(),
                    index.node_count,
                )
                for name, index in self._inverse_indices.items()
            }
        )

    @property
    def node_to_voxel_lookup(self) -> Mapping[str, PlaneNodeInverseIndex]:
        return self.inverse_indices

    @property
    def inverse_node_offsets(self) -> Mapping[str, Tensor]:
        return MappingProxyType(
            {
                name: index.offsets.clone()
                for name, index in self._inverse_indices.items()
            }
        )

    @property
    def inverse_voxel_ids(self) -> Mapping[str, Tensor]:
        return MappingProxyType(
            {
                name: index.voxel_linear.clone()
                for name, index in self._inverse_indices.items()
            }
        )

    @property
    def inverse_weights(self) -> Mapping[str, Tensor]:
        return MappingProxyType(
            {
                name: index.weights.clone()
                for name, index in self._inverse_indices.items()
            }
        )

    def _compute_integrity_digest(self) -> str:
        payload: dict[str, object] = {
            "query_version": QUERY_LATTICE_VERSION,
            "geometry_hash": self.geometry_hash,
            "query_dtype": str(self.query_dtype),
            "build_chunk_size": self.build_chunk_size,
            "memory_bound_bytes": self.memory_bound_bytes,
            "stencils": {},
            "inverse": {},
        }
        for name in PLANE_NAMES:
            stencil = self._plane_stencils.get(name)
            if stencil is not None:
                payload["stencils"][name] = {  # type: ignore[index]
                    "indices": _tensor_digest(stencil.neighbour_indices),
                    "weights": _tensor_digest(stencil.weights),
                    "valid": _tensor_digest(stencil.valid),
                    "rows": stencil.rows,
                    "columns": stencil.columns,
                }
            index = self._inverse_indices.get(name)
            if index is not None:
                payload["inverse"][name] = {  # type: ignore[index]
                    "offsets": _tensor_digest(index.offsets),
                    "voxel_linear": _tensor_digest(index.voxel_linear),
                    "weights": _tensor_digest(index.weights),
                    "node_count": index.node_count,
                }
        return _json_digest(payload)

    def _capture_tensor_versions(self) -> dict[str, int]:
        versions: dict[str, int] = {}
        for name, stencil in self._plane_stencils.items():
            versions[f"stencils.{name}.indices"] = int(
                stencil.neighbour_indices._version
            )
            versions[f"stencils.{name}.weights"] = int(stencil.weights._version)
            versions[f"stencils.{name}.valid"] = int(stencil.valid._version)
        for name, index in self._inverse_indices.items():
            versions[f"inverse.{name}.offsets"] = int(index.offsets._version)
            versions[f"inverse.{name}.voxel_linear"] = int(index.voxel_linear._version)
            versions[f"inverse.{name}.weights"] = int(index.weights._version)
        return versions

    def _fast_validate(self) -> None:
        """Check cheap PyTorch mutation versions on the query hot path."""

        started = time.perf_counter()
        for name, stencil in self._plane_stencils.items():
            checks = (
                (f"stencils.{name}.indices", stencil.neighbour_indices),
                (f"stencils.{name}.weights", stencil.weights),
                (f"stencils.{name}.valid", stencil.valid),
            )
            for key, tensor in checks:
                if int(tensor._version) != self._tensor_versions.get(key, -1):
                    raise RuntimeError(
                        "PFGRQueryLattice stencil/inverse mutation detected"
                    )
        for name, index in self._inverse_indices.items():
            checks = (
                (f"inverse.{name}.offsets", index.offsets),
                (f"inverse.{name}.voxel_linear", index.voxel_linear),
                (f"inverse.{name}.weights", index.weights),
            )
            for key, tensor in checks:
                if int(tensor._version) != self._tensor_versions.get(key, -1):
                    raise RuntimeError(
                        "PFGRQueryLattice stencil/inverse mutation detected"
                    )
        elapsed = time.perf_counter() - started
        self._operation_counters["validation_fast_calls"] += 1
        self._operation_counters["validation_fast_seconds"] += elapsed

    def validate_integrity(self) -> None:
        """Reject stale cache entries or caller-induced stencil mutation."""

        started = time.perf_counter()
        expected_output = _json_digest(_geometry_payload(self.output_geometry))
        expected_feature = _json_digest(
            _feature_geometry_payload(self.feature_geometry)
        )
        if (
            expected_output != self.output_geometry_hash
            or expected_feature != self.feature_geometry_hash
        ):
            raise RuntimeError("PFGRQueryLattice geometry identity changed")
        current = self._compute_integrity_digest()
        if current != self._integrity_digest:
            raise RuntimeError("PFGRQueryLattice stencil/inverse mutation detected")
        elapsed = time.perf_counter() - started
        self._operation_counters["validation_full_calls"] += 1
        self._operation_counters["validation_full_seconds"] += elapsed

    def _stencils_for_device(
        self, device: torch.device
    ) -> Mapping[str, BilinearStencil] | None:
        if not self._plane_stencils:
            return None
        # The canonical arrays are owned on CPU.  Query selects only requested
        # rows and transfers those bounded chunks to an accelerator; no full
        # device mirror is retained.
        return self._plane_stencils

    @staticmethod
    def _move_stencil(
        stencil: BilinearStencil, device: torch.device
    ) -> BilinearStencil:
        if stencil.neighbour_indices.device == device:
            return stencil
        return BilinearStencil(
            neighbour_indices=stencil.neighbour_indices.to(device=device),
            weights=stencil.weights.to(device=device),
            valid=stencil.valid.to(device=device),
            rows=stencil.rows,
            columns=stencil.columns,
        )

    def _chunk_stencils(self, voxel_ids: Tensor) -> Mapping[str, BilinearStencil]:
        coordinates = self._feature_coordinates(
            voxel_ids,
            output_geometry=self.output_geometry,
            feature_geometry=self.feature_geometry,
            dtype=self.query_dtype,
        )
        d, h, w = coordinates.unbind(dim=-1)
        return MappingProxyType(
            {
                "xy": _stencil_from_coordinates(
                    h,
                    w,
                    rows=self.feature_shape_dhw[1],
                    columns=self.feature_shape_dhw[2],
                ),
                "xz": _stencil_from_coordinates(
                    d,
                    w,
                    rows=self.feature_shape_dhw[0],
                    columns=self.feature_shape_dhw[2],
                ),
                "yz": _stencil_from_coordinates(
                    d,
                    h,
                    rows=self.feature_shape_dhw[0],
                    columns=self.feature_shape_dhw[1],
                ),
            }
        )

    @staticmethod
    def _query_plane(plane: Tensor, stencil: BilinearStencil) -> Tensor:
        channels = int(plane.shape[1])
        node_count = stencil.rows * stencil.columns
        # Invalid indices select node zero only as a safe gather address and
        # are then explicitly masked to zero.  This is zero-padding semantics,
        # not border clamping: no invalid value can contribute node zero.
        safe_indices = torch.where(
            stencil.valid,
            stencil.neighbour_indices,
            torch.zeros_like(stencil.neighbour_indices),
        )
        flat = plane[0].reshape(channels, node_count)
        gathered = flat.index_select(1, safe_indices.reshape(-1)).reshape(
            channels, -1, 4
        )
        values = gathered * stencil.weights.unsqueeze(0) * stencil.valid.unsqueeze(0)
        return values.sum(dim=-1).transpose(0, 1)

    @staticmethod
    def _query_plane_grid(
        plane: Tensor,
        coordinates: Tensor,
        *,
        row_axis: int,
        column_axis: int,
    ) -> Tensor:
        """Query one plane through PyTorch's canonical bilinear kernel.

        The stored four-neighbour stencil remains the support/linearity
        authority for sparse writes.  Dense canonical queries use the same
        ``grid_sample(..., align_corners=False, padding_mode='zeros')``
        arithmetic as the legacy geometry-aware path.  This avoids a
        reduction-tree-dependent FP32 discrepancy between a hand-written
        four-term sum and PyTorch's production sampler while retaining the
        explicit stencil for inverse lookup and fallback accounting.
        """

        rows, columns = plane.shape[-2:]
        row = coordinates[:, row_axis]
        column = coordinates[:, column_axis]
        grid = torch.stack(
            (
                (2.0 * column + 1.0) / float(columns) - 1.0,
                (2.0 * row + 1.0) / float(rows) - 1.0,
            ),
            dim=-1,
        ).reshape(1, -1, 1, 2)
        sampling_plane = (
            plane if plane.dtype == grid.dtype else plane.to(dtype=grid.dtype)
        )
        with torch.autocast(device_type=grid.device.type, enabled=False):
            sampled = F.grid_sample(
                sampling_plane,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        return sampled[0, :, :, 0].transpose(0, 1)

    def _validate_planes(self, planes: DynamicTriPlanes) -> None:
        if not isinstance(planes, DynamicTriPlanes):
            raise TypeError("planes must be a DynamicTriPlanes instance")
        if planes.xy.shape[0] != 1:
            raise ValueError(
                "PFGRQueryLattice.query requires subject batch size exactly one"
            )
        if planes.xy.dtype != self.query_dtype:
            raise TypeError(
                f"planes dtype {planes.xy.dtype} must equal lattice query_dtype {self.query_dtype}"
            )
        expected = {
            "xy": (self.feature_shape_dhw[1], self.feature_shape_dhw[2]),
            "xz": (self.feature_shape_dhw[0], self.feature_shape_dhw[2]),
            "yz": (self.feature_shape_dhw[0], self.feature_shape_dhw[1]),
        }
        for name in PLANE_NAMES:
            plane = getattr(planes, name)
            if (
                plane.device != planes.xy.device
                or tuple(plane.shape[-2:]) != expected[name]
            ):
                raise ValueError(
                    f"planes.{name} must retain the feature-grid shape and shared device"
                )

    def _validate_voxel_ids(
        self, voxel_ids_dhw: Tensor, *, device: torch.device
    ) -> None:
        if (
            not isinstance(voxel_ids_dhw, Tensor)
            or voxel_ids_dhw.ndim != 2
            or voxel_ids_dhw.shape[-1] != 3
        ):
            raise ValueError("voxel_ids_dhw must have shape [Q,3]")
        if voxel_ids_dhw.dtype != torch.long:
            raise TypeError("voxel_ids_dhw must have dtype torch.long")
        if voxel_ids_dhw.device != device:
            raise ValueError("voxel_ids_dhw and planes must share one device")
        if voxel_ids_dhw.numel() == 0:
            return
        if bool((voxel_ids_dhw < 0).any()):
            raise ValueError("voxel_ids_dhw must be non-negative output voxel centres")
        upper = torch.as_tensor(self.output_shape_dhw, dtype=torch.long, device=device)
        if bool((voxel_ids_dhw >= upper).any()):
            raise ValueError("voxel_ids_dhw must lie within output_geometry.shape_dhw")

    def query(
        self, planes: DynamicTriPlanes, voxel_ids_dhw: Tensor, *, chunk_size: int
    ) -> Tensor:
        """Query ``planes`` at integer output voxel centres as ``[Q,96]``.

        The subject dimension is deliberately not vectorized: candidate
        batching belongs to a caller above this geometry-only seam.  Duplicate
        IDs remain duplicate draws and preserve their input order.
        """

        self._fast_validate()
        self._validate_planes(planes)
        chunk_size = _positive_int("chunk_size", chunk_size)
        self._validate_voxel_ids(voxel_ids_dhw, device=planes.xy.device)
        query_count = int(voxel_ids_dhw.shape[0])
        self._operation_counters["query_calls"] += 1
        self._operation_counters["query_voxel_count"] += query_count
        if query_count == 0:
            return torch.empty((0, 96), dtype=planes.xy.dtype, device=planes.xy.device)

        materialized = self._stencils_for_device(planes.xy.device)
        outputs: list[Tensor] = []
        _, height, width = self.output_shape_dhw
        for start in range(0, query_count, chunk_size):
            stop = min(start + chunk_size, query_count)
            if materialized is None:
                chunk_ids = voxel_ids_dhw[start:stop]
                canonical_ids = chunk_ids.to(device="cpu")
                chunk_coordinates = self._feature_coordinates(
                    canonical_ids,
                    output_geometry=self.output_geometry,
                    feature_geometry=self.feature_geometry,
                    dtype=self.query_dtype,
                ).to(device=planes.xy.device)
                chunk_stencils_cpu = self._chunk_stencils(canonical_ids)
                self._operation_counters["query_stencil_voxel_count"] += int(
                    3 * (stop - start)
                )
                self._operation_counters["query_stencil_bytes"] += int(
                    canonical_ids.numel() * canonical_ids.element_size()
                    + chunk_coordinates.numel() * chunk_coordinates.element_size()
                    + sum(
                        value.numel() * value.element_size()
                        for stencil in chunk_stencils_cpu.values()
                        for value in (
                            stencil.neighbour_indices,
                            stencil.weights,
                            stencil.valid,
                        )
                    )
                )
                _chunk_stencils = {
                    name: self._move_stencil(stencil, planes.xy.device)
                    for name, stencil in chunk_stencils_cpu.items()
                }
                if planes.xy.device.type != "cpu":
                    self._operation_counters["stencil_transfer_calls"] += 1
                    self._operation_counters["stencil_transfer_voxel_count"] += int(
                        3 * (stop - start)
                    )
                    self._operation_counters["stencil_transfer_bytes"] += int(
                        sum(
                            value.numel() * value.element_size()
                            for stencil in chunk_stencils_cpu.values()
                            for value in (
                                stencil.neighbour_indices,
                                stencil.weights,
                                stencil.valid,
                            )
                        )
                    )
            else:
                chunk_ids = voxel_ids_dhw[start:stop]
                chunk_coordinates = self._feature_coordinates(
                    chunk_ids.to(device="cpu"),
                    output_geometry=self.output_geometry,
                    feature_geometry=self.feature_geometry,
                    dtype=self.query_dtype,
                ).to(device=planes.xy.device)
                linear = (
                    chunk_ids[:, 0] * (height * width)
                    + chunk_ids[:, 1] * width
                    + chunk_ids[:, 2]
                )
                canonical_linear = linear.to(device="cpu")
                chunk_stencils_cpu = {
                    name: BilinearStencil(
                        neighbour_indices=stencil.neighbour_indices.index_select(
                            0, canonical_linear
                        ),
                        weights=stencil.weights.index_select(0, canonical_linear),
                        valid=stencil.valid.index_select(0, canonical_linear),
                        rows=stencil.rows,
                        columns=stencil.columns,
                    )
                    for name, stencil in materialized.items()
                }
                self._operation_counters["query_stencil_voxel_count"] += int(
                    3 * (stop - start)
                )
                self._operation_counters["query_stencil_bytes"] += int(
                    chunk_ids.numel() * chunk_ids.element_size()
                    + chunk_coordinates.numel() * chunk_coordinates.element_size()
                    + sum(
                        value.numel() * value.element_size()
                        for stencil in chunk_stencils_cpu.values()
                        for value in (
                            stencil.neighbour_indices,
                            stencil.weights,
                            stencil.valid,
                        )
                    )
                )
                _chunk_stencils = {
                    name: self._move_stencil(stencil, planes.xy.device)
                    for name, stencil in chunk_stencils_cpu.items()
                }
                if planes.xy.device.type != "cpu":
                    self._operation_counters["stencil_transfer_calls"] += 1
                    self._operation_counters["stencil_transfer_voxel_count"] += int(
                        3 * (stop - start)
                    )
                    self._operation_counters["stencil_transfer_bytes"] += int(
                        sum(
                            value.numel() * value.element_size()
                            for stencil in chunk_stencils_cpu.values()
                            for value in (
                                stencil.neighbour_indices,
                                stencil.weights,
                                stencil.valid,
                            )
                        )
                    )
            queried = torch.cat(
                tuple(
                    self._query_plane_grid(
                        getattr(planes, name),
                        chunk_coordinates,
                        row_axis=(1, 0, 0)[index],
                        column_axis=(2, 2, 1)[index],
                    )
                    for index, name in enumerate(PLANE_NAMES)
                ),
                dim=-1,
            )
            outputs.append(queried)
        return torch.cat(outputs, dim=0)

    def _scan_positive_linear(self, plane: str | None = None) -> Tensor:
        started = time.perf_counter()
        names = PLANE_NAMES if plane is None else (self._validate_plane_name(plane),)
        voxel_count = math.prod(self.output_shape_dhw)
        pieces: list[Tensor] = []
        for start in range(0, voxel_count, self.build_chunk_size):
            stop = min(start + self.build_chunk_size, voxel_count)
            ids = _output_voxel_ids(
                start, stop, shape_dhw=self.output_shape_dhw, device=torch.device("cpu")
            )
            stencils = self._chunk_stencils(ids)
            self._operation_counters["scanned_voxel_count"] += int(
                (stop - start) * len(names)
            )
            chunk_bytes = ids.numel() * ids.element_size()
            for name in names:
                stencil = stencils[name]
                chunk_bytes += sum(
                    value.numel() * value.element_size()
                    for value in (
                        stencil.neighbour_indices,
                        stencil.weights,
                        stencil.valid,
                    )
                )
                positive = stencil.valid & (stencil.weights > 0)
                for slot in range(4):
                    if bool(positive[:, slot].any()):
                        pieces.append(
                            torch.arange(start, stop, dtype=torch.long)[
                                positive[:, slot]
                            ]
                        )
            self._operation_counters["scanned_bytes"] += int(chunk_bytes)
            self._operation_counters["scan_peak_bytes"] = max(
                self._operation_counters["scan_peak_bytes"], int(chunk_bytes)
            )
        self._operation_counters["scan_calls"] += 1
        self._operation_counters["scan_seconds"] += time.perf_counter() - started
        if not pieces:
            return torch.empty((0,), dtype=torch.long)
        return torch.unique(torch.cat(pieces), sorted=True)

    @staticmethod
    def _validate_plane_name(plane: str) -> str:
        if plane not in PLANE_NAMES:
            raise ValueError("plane must be 'xy', 'xz', or 'yz'")
        return plane

    def positive_support_linear(self, plane: str | None = None) -> Tensor:
        """Return unique positive-support output IDs, independently of IDs queried."""

        self.validate_integrity()
        if plane is not None:
            plane = self._validate_plane_name(plane)
        if plane is None:
            if self._inverse_indices:
                pieces = [
                    torch.unique(index.voxel_linear, sorted=True)
                    for index in self._inverse_indices.values()
                ]
                result = torch.unique(
                    torch.cat(pieces)
                    if pieces
                    else torch.empty((0,), dtype=torch.long),
                    sorted=True,
                )
            else:
                result = self._scan_positive_linear(None)
        elif plane in self._inverse_indices:
            result = torch.unique(
                self._inverse_indices[plane].voxel_linear, sorted=True
            )
        else:
            result = self._scan_positive_linear(plane)
        return result.clone()

    def support_voxel_ids(
        self, plane: str | None = None, *, device: torch.device | None = None
    ) -> Tensor:
        """Return positive-support output IDs as ``[S,3]`` DHW rows."""

        self.validate_integrity()
        linear = self.positive_support_linear(plane)
        if device is not None:
            linear = linear.to(device=device)
        _, height, width = self.output_shape_dhw
        d = torch.div(linear, height * width, rounding_mode="floor")
        rem = linear - d * height * width
        h = torch.div(rem, width, rounding_mode="floor")
        w = rem - h * width
        return torch.stack((d, h, w), dim=-1)

    positive_support_voxel_ids = support_voxel_ids

    @property
    def positive_support_by_plane(self) -> Mapping[str, Tensor]:
        """Copy-on-read per-plane support rows for sparse writer consumers."""

        return MappingProxyType(
            {name: self.support_voxel_ids(name) for name in PLANE_NAMES}
        )

    def positive_support_membership(
        self, voxel_ids_dhw: Tensor, *, plane: str | None = None
    ) -> Tensor:
        """Return exact positive-support membership for integer output IDs."""

        self.validate_integrity()
        if (
            not isinstance(voxel_ids_dhw, Tensor)
            or voxel_ids_dhw.ndim != 2
            or voxel_ids_dhw.shape[-1] != 3
            or voxel_ids_dhw.dtype != torch.long
        ):
            raise ValueError("voxel_ids_dhw must be [Q,3] torch.long")
        self._validate_voxel_ids(voxel_ids_dhw, device=voxel_ids_dhw.device)
        if voxel_ids_dhw.numel() == 0:
            return torch.empty((0,), dtype=torch.bool, device=voxel_ids_dhw.device)
        _, height, width = self.output_shape_dhw
        linear = (
            voxel_ids_dhw[:, 0] * (height * width)
            + voxel_ids_dhw[:, 1] * width
            + voxel_ids_dhw[:, 2]
        )
        support = self.positive_support_linear(plane).to(device=voxel_ids_dhw.device)
        return torch.isin(linear, support)

    def support_multiplicity(self, voxel_ids_dhw: Tensor) -> Tensor:
        """Count how many plane supports contain each output voxel row."""

        self.validate_integrity()
        if (
            not isinstance(voxel_ids_dhw, Tensor)
            or voxel_ids_dhw.ndim != 2
            or voxel_ids_dhw.shape[-1] != 3
            or voxel_ids_dhw.dtype != torch.long
        ):
            raise ValueError("voxel_ids_dhw must be [Q,3] torch.long")
        self._validate_voxel_ids(voxel_ids_dhw, device=voxel_ids_dhw.device)
        if voxel_ids_dhw.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=voxel_ids_dhw.device)
        return torch.stack(
            [
                self.positive_support_membership(voxel_ids_dhw, plane=name).to(
                    dtype=torch.long
                )
                for name in PLANE_NAMES
            ],
            dim=0,
        ).sum(dim=0)

    support_membership = positive_support_membership

    def node_to_voxel(self, plane: str) -> PlaneNodeInverseIndex | None:
        """Return the indexed positive node-to-voxel CSR record, if present."""

        self.validate_integrity()
        index = self._inverse_indices.get(self._validate_plane_name(plane))
        if index is None:
            return None
        return PlaneNodeInverseIndex(
            index.offsets.clone(),
            index.voxel_linear.clone(),
            index.weights.clone(),
            index.node_count,
        )

    def positive_node_edges(self, plane: str) -> tuple[Tensor, Tensor, Tensor] | None:
        """Return expanded ``(node_id, voxel_linear, weight)`` positive edges."""

        index = self.node_to_voxel(plane)
        if index is None:
            return None
        return index.node_ids, index.voxel_linear.clone(), index.weights.clone()


__all__ = [
    "CACHE_MAX_BYTES",
    "CACHE_MAX_ENTRIES",
    "DEFAULT_MEMORY_BOUND_BYTES",
    "PLANE_NAMES",
    "QUERY_LATTICE_VERSION",
    "BilinearStencil",
    "PFGRQueryLattice",
    "PlaneNodeInverseIndex",
]
