"""PFGR-Lite compact writes and sparse query-delta evaluation.

The legacy :class:`~smagm.features.point_guided.writeback.CompactTriPlaneWriteback`
is intentionally kept as the reference writer.  This module records its
positive discrete node support and projects that support through the canonical
PFGR query lattice.  The optimized path never allocates a hypothetical full
plane: bilinear query linearity is used to evaluate the write response directly
at requested output voxels.

The sparse algebra is exact at the level of the writer/query operators and is
therefore marked ``EXACT ALGEBRA``.  Batched floating point evaluation is a
``NUMERICALLY EQUIVALENT`` implementation; no Taylor, envelope, or
continuous-sphere approximation is used.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ..state_init import DynamicTriPlanes
from ..updater import PlaneCorrections
from ..writeback import (
    CompactTriPlaneWriteback,
    _local_window,
    _minimum_plane_spacing_mm,
)
from .footprint import PLANE_NAMES, PFGRQueryLattice
from .provenance import canonical_digest

if TYPE_CHECKING:
    from .types import ActionProposal, SparseFootprint

SPARSE_WRITE_VERSION = "pfgr-lite-sparse-write-v1"
WRITER_KERNEL_VERSION = "quadratic_compact_4mm_v1"
DEFAULT_QUERY_CHUNK_SIZE = 4096
POINT_QUERY_VERSION = "pfgr-lite-point-query-v1"
POINT_QUERY_HASH = hashlib.sha256(
    f"pfgr-lite-query-version|{POINT_QUERY_VERSION}".encode()
).hexdigest()


def _context_geometry_identity(context: object) -> str | None:
    """Match W4's full-affine action geometry identity without importing W4."""

    geometry = getattr(context, "geometry", None)
    feature = getattr(context, "feature_geometry", None)
    if geometry is None or feature is None:
        return None
    return canonical_digest(
        {
            "source_shape_dhw": geometry.shape_dhw,
            "source_voxel_to_ras_mm": geometry.voxel_to_ras_mm,
            "feature_shape_dhw": feature.shape_dhw,
            "feature_geometry": feature.feature_geometry.voxel_to_ras_mm,
            "feature_to_source_scale_dhw": feature.feature_to_source_scale_dhw,
            "feature_to_source_offset_dhw": feature.feature_to_source_offset_dhw,
            "operator_chain": feature.operator_chain,
            "version": "pfgr-lite-static-geometry-v1",
        },
        prefix="pfgr-lite-action-geometry-v1|",
    )


def _positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _tensor_digest(value: Tensor) -> str:
    cpu = value.detach().to(device="cpu").contiguous()
    header = f"{cpu.dtype}|{tuple(cpu.shape)!r}|".encode()
    return hashlib.sha256(header + cpu.numpy().tobytes()).hexdigest()


def _support_axes(plane: str) -> tuple[int, int, int]:
    if plane == "xy":
        return 1, 2, 0
    if plane == "xz":
        return 0, 2, 1
    if plane == "yz":
        return 0, 1, 2
    raise ValueError("plane must be 'xy', 'xz', or 'yz'")


def _plane_shape(lattice: PFGRQueryLattice, plane: str) -> tuple[int, int]:
    depth, height, width = lattice.feature_shape_dhw
    if plane == "xy":
        return height, width
    if plane == "xz":
        return depth, width
    if plane == "yz":
        return depth, height
    raise ValueError("plane must be 'xy', 'xz', or 'yz'")


def _validate_action(lattice: PFGRQueryLattice, action: ActionProposal) -> None:
    from .types import ActionProposal

    if not isinstance(action, ActionProposal):
        raise TypeError("action must be an ActionProposal")
    action.validate_integrity()
    if action.query_version not in (lattice.query_version, POINT_QUERY_VERSION):
        raise ValueError("action query version does not match PFGR lattice")
    if action.delta.dtype != lattice.query_dtype:
        raise TypeError("action delta dtype must equal lattice query_dtype")
    if action.point_ras_mm.dtype != lattice.query_dtype:
        raise TypeError("action point dtype must equal lattice query_dtype")
    if action.point_ras_mm.device != action.delta.device:
        raise ValueError("action point and delta must share device")


def _positive_writer_nodes_at_point(
    lattice: PFGRQueryLattice,
    point: Tensor,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Return the exact positive retained nodes/weights used by writeback.

    The window and physical distance code is the same helper used by
    ``CompactTriPlaneWriteback``.  We intentionally do not derive support from
    a sphere in output space: each retained plane node is measured in the full
    feature-grid affine with the writer's omitted coordinate fixed at the
    point's feature coordinate.
    """

    feature_geometry = lattice.feature_geometry
    if not isinstance(point, Tensor) or point.shape != (3,):
        raise ValueError("point must have shape [3]")
    # Geometry is target-independent and must not retain a proposal/U graph.
    with torch.no_grad():
        feature_dhw = feature_geometry.ras_mm_to_feature_dhw(point.reshape(1, 3))[0]
        result_nodes: list[Tensor] = []
        result_weights: list[Tensor] = []
        for plane in PLANE_NAMES:
            row_axis, column_axis, omitted_axis = _support_axes(plane)
            rows, columns = _plane_shape(lattice, plane)
            row_start, row_stop = _local_window(
                feature_dhw[row_axis],
                rows,
                radius_mm=4.0,
                minimum_spacing_mm=_minimum_plane_spacing_mm(
                    feature_geometry, (row_axis, column_axis)
                ),
            )
            column_start, column_stop = _local_window(
                feature_dhw[column_axis],
                columns,
                radius_mm=4.0,
                minimum_spacing_mm=_minimum_plane_spacing_mm(
                    feature_geometry, (row_axis, column_axis)
                ),
            )
            rows_index = torch.arange(
                row_start, row_stop, dtype=point.dtype, device=point.device
            )
            columns_index = torch.arange(
                column_start, column_stop, dtype=point.dtype, device=point.device
            )
            row_grid, column_grid = torch.meshgrid(
                rows_index, columns_index, indexing="ij"
            )
            coordinates = [row_grid * 0.0 + feature_dhw[index] for index in range(3)]
            coordinates[row_axis] = row_grid
            coordinates[column_axis] = column_grid
            coordinates[omitted_axis] = row_grid * 0.0 + feature_dhw[omitted_axis]
            locations = feature_geometry.feature_dhw_to_ras_mm(
                torch.stack(coordinates, dim=-1)
            )
            distance = torch.linalg.vector_norm(locations - point, dim=-1)
            weight = torch.square(torch.clamp(1.0 - distance / 4.0, min=0.0))
            positive = weight > 0.0
            node_ids = (
                row_grid.to(dtype=torch.long) * columns
                + column_grid.to(dtype=torch.long)
            )[positive]
            node_weights = weight[positive].to(dtype=lattice.query_dtype)
            result_nodes.append(node_ids.detach().to(device="cpu", dtype=torch.long))
            result_weights.append(node_weights.detach().to(device="cpu"))
    return tuple(result_nodes), tuple(result_weights)


def _positive_writer_nodes(
    lattice: PFGRQueryLattice,
    action: ActionProposal,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Return exact positive writer support for one immutable action row."""

    return _positive_writer_nodes_at_point(lattice, action.point_ras_mm)


def _accumulate_edges(
    voxel_linear: Tensor,
    coefficients: Tensor,
) -> tuple[Tensor, Tensor]:
    if voxel_linear.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=coefficients.dtype),
        )
    unique, inverse = torch.unique(voxel_linear, sorted=True, return_inverse=True)
    summed = torch.zeros((unique.numel(),), dtype=coefficients.dtype)
    summed.index_add_(0, inverse, coefficients)
    positive = summed > 0.0
    return unique[positive], summed[positive]


def _indexed_plane_response(
    lattice: PFGRQueryLattice,
    plane: str,
    node_ids: Tensor,
    node_weights: Tensor,
) -> tuple[Tensor, Tensor]:
    index = lattice._inverse_indices.get(plane)  # type: ignore[attr-defined]
    if index is None:
        raise RuntimeError(
            "indexed response requested without a materialized inverse index"
        )
    edge_voxels: list[Tensor] = []
    edge_weights: list[Tensor] = []
    offsets = index.offsets
    for node, node_weight in zip(node_ids.tolist(), node_weights.tolist()):
        start = int(offsets[node].item())
        stop = int(offsets[node + 1].item())
        if stop <= start:
            continue
        edge_voxels.append(index.voxel_linear[start:stop])
        edge_weights.append(index.weights[start:stop] * float(node_weight))
    if not edge_voxels:
        return (
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=lattice.query_dtype),
        )
    return _accumulate_edges(torch.cat(edge_voxels), torch.cat(edge_weights))


def _record_scan(
    lattice: PFGRQueryLattice,
    *,
    scanned_voxels: int,
    scanned_bytes: int,
    peak_bytes: int,
    elapsed: float,
) -> None:
    counters = lattice._operation_counters  # type: ignore[attr-defined]
    counters["scan_calls"] += 1
    counters["scanned_voxel_count"] += int(scanned_voxels)
    counters["scanned_bytes"] += int(scanned_bytes)
    counters["scan_peak_bytes"] = max(counters["scan_peak_bytes"], int(peak_bytes))
    counters["scan_seconds"] += float(elapsed)


def _scan_plane_response(
    lattice: PFGRQueryLattice,
    plane: str,
    node_ids: Tensor,
    node_weights: Tensor,
    *,
    chunk_size: int,
) -> tuple[Tensor, Tensor, int, int, int]:
    """Exact fallback response by scanning all output rows in bounded chunks."""

    if node_ids.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=lattice.query_dtype),
            0,
            0,
            0,
        )
    order = torch.argsort(node_ids)
    sorted_nodes = node_ids[order]
    sorted_weights = node_weights[order]
    volume = math.prod(lattice.output_shape_dhw)
    output_chunks: list[Tensor] = []
    coefficient_chunks: list[Tensor] = []
    scanned_bytes = 0
    peak_bytes = 0
    for start in range(0, volume, chunk_size):
        stop = min(start + chunk_size, volume)
        ids = torch.arange(start, stop, dtype=torch.long)
        # ``_chunk_stencils`` accepts integer DHW rows.  Construct those rows
        # locally so fallback behavior remains compatible with every lattice
        # snapshot without retaining a full output-volume index.
        _, height, width = lattice.output_shape_dhw
        area = height * width
        d = torch.div(ids, area, rounding_mode="floor")
        rem = ids - d * area
        h = torch.div(rem, width, rounding_mode="floor")
        w = rem - h * width
        stencils = lattice._chunk_stencils(torch.stack((d, h, w), dim=-1))  # type: ignore[attr-defined]
        stencil = stencils[plane]
        node_count = sorted_nodes.numel()
        positions = torch.searchsorted(sorted_nodes, stencil.neighbour_indices)
        safe = positions.clamp(max=node_count - 1)
        valid = (
            stencil.valid
            & (positions < node_count)
            & (sorted_nodes[safe] == stencil.neighbour_indices)
        )
        scalar = torch.zeros((stop - start, 4), dtype=lattice.query_dtype)
        if bool(valid.any()):
            scalar[valid] = sorted_weights[safe[valid]]
        scalar = scalar * stencil.weights.to(dtype=lattice.query_dtype) * stencil.valid
        coefficients = scalar.sum(dim=-1)
        positive = coefficients > 0.0
        if bool(positive.any()):
            output_chunks.append(ids[positive])
            coefficient_chunks.append(coefficients[positive])
        chunk_bytes = sum(
            item.numel() * item.element_size()
            for item in (
                ids,
                stencil.neighbour_indices,
                stencil.weights,
                stencil.valid,
                coefficients,
            )
        )
        scanned_bytes += int(chunk_bytes)
        peak_bytes = max(peak_bytes, int(chunk_bytes))
    if not output_chunks:
        output = torch.empty((0,), dtype=torch.long)
        coefficient = torch.empty((0,), dtype=lattice.query_dtype)
    else:
        output, coefficient = _accumulate_edges(
            torch.cat(output_chunks), torch.cat(coefficient_chunks)
        )
    return output, coefficient, volume, scanned_bytes, peak_bytes


def _linear_to_dhw(linear: Tensor, shape_dhw: Sequence[int]) -> Tensor:
    _, height, width = tuple(int(value) for value in shape_dhw)
    d = torch.div(linear, height * width, rounding_mode="floor")
    rem = linear - d * height * width
    h = torch.div(rem, width, rounding_mode="floor")
    w = rem - h * width
    return torch.stack((d, h, w), dim=-1)


@dataclass(frozen=True)
class _SparseRecord:
    """Private immutable response data kept alongside W1's public type."""

    node_ids: tuple[Tensor, Tensor, Tensor]
    node_weights: tuple[Tensor, Tensor, Tensor]
    plane_voxel_linear: tuple[Tensor, Tensor, Tensor]
    plane_coefficients: tuple[Tensor, Tensor, Tensor]
    digest: str


def _record_digest(
    node_ids: tuple[Tensor, Tensor, Tensor],
    node_weights: tuple[Tensor, Tensor, Tensor],
    plane_voxels: tuple[Tensor, Tensor, Tensor],
    plane_coefficients: tuple[Tensor, Tensor, Tensor],
) -> str:
    payload = []
    for values in (node_ids, node_weights, plane_voxels, plane_coefficients):
        payload.extend(_tensor_digest(value) for value in values)
    return hashlib.sha256("|".join(payload).encode()).hexdigest()


def _attach_record(
    footprint: SparseFootprint,
    record: _SparseRecord,
    lattice: PFGRQueryLattice,
    action: ActionProposal,
) -> None:
    # W1 intentionally kept SparseFootprint's public declaration compact.  The
    # private attributes below are owned clones; consumers only receive copies
    # through query helpers and cannot alter the lattice's process cache.
    object.__setattr__(footprint, "_pfgr_record", record)
    object.__setattr__(footprint, "_pfgr_lattice_version", lattice.query_version)
    object.__setattr__(footprint, "_pfgr_geometry_hash", lattice.geometry_hash)
    object.__setattr__(footprint, "_pfgr_action_digest", action.action_digest)
    object.__setattr__(footprint, "_pfgr_version", SPARSE_WRITE_VERSION)


def _footprint_public_digest(footprint: SparseFootprint) -> str:
    payload = [
        _tensor_digest(footprint.voxel_ids_dhw),
        _tensor_digest(footprint.multiplicity)
        if footprint.multiplicity is not None
        else "none",
    ]
    payload.extend(_tensor_digest(value) for value in footprint.support_pairs)
    payload.extend(
        (
            str(footprint.plane_counts),
            footprint.lattice_version,
            footprint.geometry_hash,
            footprint.kernel_version,
            footprint.mode,
            str(footprint.scanned_voxel_count),
        )
    )
    return hashlib.sha256("|".join(payload).encode()).hexdigest()


def _footprint_accounting_digest(footprint: SparseFootprint) -> str:
    """Guard private operation metadata attached to W1's frozen record."""

    union = getattr(footprint, "_pfgr_union_linear", None)
    plane_voxels = getattr(footprint, "_pfgr_plane_voxel_linear", ())
    plane_coefficients = getattr(footprint, "_pfgr_plane_coefficients", ())
    if (
        not isinstance(union, Tensor)
        or len(plane_voxels) != 3
        or len(plane_coefficients) != 3
    ):
        return "missing"
    payload = (
        str(getattr(footprint, "_pfgr_build_elapsed_seconds", None)),
        str(getattr(footprint, "_pfgr_scanned_bytes", None)),
        str(getattr(footprint, "_pfgr_scan_peak_bytes", None)),
        _tensor_digest(union),
        "|".join(_tensor_digest(value) for value in plane_voxels),
        "|".join(_tensor_digest(value) for value in plane_coefficients),
    )
    return hashlib.sha256("|".join(payload).encode()).hexdigest()


def _validate_footprint(
    lattice: PFGRQueryLattice,
    footprint: SparseFootprint,
) -> _SparseRecord:
    from .types import SparseFootprint

    if not isinstance(footprint, SparseFootprint):
        raise TypeError("footprint must be a SparseFootprint")
    lattice.validate_integrity()
    if getattr(footprint, "_pfgr_lattice_version", None) != lattice.query_version:
        raise ValueError("footprint lattice version is stale")
    if getattr(footprint, "_pfgr_geometry_hash", None) != lattice.geometry_hash:
        raise ValueError("footprint geometry identity is stale")
    record = getattr(footprint, "_pfgr_record", None)
    if not isinstance(record, _SparseRecord):
        raise TypeError("footprint lacks sparse response metadata")
    expected = _record_digest(
        record.node_ids,
        record.node_weights,
        record.plane_voxel_linear,
        record.plane_coefficients,
    )
    if expected != record.digest:
        raise RuntimeError("SparseFootprint support metadata mutation detected")
    public_digest = getattr(footprint, "_pfgr_public_digest", None)
    if public_digest != _footprint_public_digest(footprint):
        raise RuntimeError("SparseFootprint public support mutation detected")
    if getattr(
        footprint, "_pfgr_accounting_digest", None
    ) != _footprint_accounting_digest(footprint):
        raise RuntimeError("SparseFootprint accounting metadata mutation detected")
    if footprint.mode == "indexed" and lattice.footprint_mode != "indexed":
        raise ValueError("indexed footprint cannot be used with fallback lattice")
    if (
        footprint.mode == "full_scan_fallback"
        and lattice.footprint_mode != "full_scan_fallback"
    ):
        raise ValueError("fallback footprint cannot be used with indexed lattice")
    return record


def build_footprint(
    lattice: PFGRQueryLattice,
    action: ActionProposal,
    *,
    chunk_size: int,
) -> SparseFootprint:
    """Build the exact discrete writer-to-output support for one action.

    ``chunk_size`` bounds fallback full-output scans.  Indexed lattices use the
    one-per-geometry inverse node lookup built by ``PFGRQueryLattice``.  The
    resulting union and per-plane responses are target-independent.
    """

    from .types import SparseFootprint

    if not isinstance(lattice, PFGRQueryLattice):
        raise TypeError("lattice must be a PFGRQueryLattice")
    _positive_int("chunk_size", chunk_size)
    _validate_action(lattice, action)
    started = time.perf_counter()
    node_ids, node_weights = _positive_writer_nodes(lattice, action)
    plane_voxels: list[Tensor] = []
    plane_coefficients: list[Tensor] = []
    scanned_voxels = 0
    scanned_bytes = 0
    scan_peak = 0
    if lattice.footprint_mode == "indexed":
        for plane, nodes, weights in zip(PLANE_NAMES, node_ids, node_weights):
            voxels, coefficients = _indexed_plane_response(
                lattice, plane, nodes, weights
            )
            plane_voxels.append(voxels)
            plane_coefficients.append(coefficients)
    else:
        scan_started = time.perf_counter()
        for plane, nodes, weights in zip(PLANE_NAMES, node_ids, node_weights):
            voxels, coefficients, visited, bytes_used, peak = _scan_plane_response(
                lattice,
                plane,
                nodes,
                weights,
                chunk_size=chunk_size,
            )
            plane_voxels.append(voxels)
            plane_coefficients.append(coefficients)
            scanned_voxels += visited
            scanned_bytes += bytes_used
            scan_peak = max(scan_peak, peak)
        _record_scan(
            lattice,
            scanned_voxels=scanned_voxels,
            scanned_bytes=scanned_bytes,
            peak_bytes=scan_peak,
            elapsed=time.perf_counter() - scan_started,
        )

    plane_voxel_tuple = tuple(plane_voxels)  # type: ignore[assignment]
    plane_coefficient_tuple = tuple(plane_coefficients)  # type: ignore[assignment]
    union_linear = torch.unique(torch.cat(plane_voxel_tuple), sorted=True)
    if union_linear.numel() == 0:
        raise ValueError("action has empty geometric output support")
    multiplicity = torch.stack(
        [
            torch.isin(union_linear, values).to(dtype=torch.long)
            for values in plane_voxel_tuple
        ],
        dim=0,
    ).sum(dim=0)
    if bool((multiplicity <= 0).any()):
        raise RuntimeError("positive support union contains a zero multiplicity row")
    encoded_pairs = tuple(
        torch.stack(
            (
                nodes.to(dtype=torch.float64),
                weights.to(dtype=torch.float64),
            ),
            dim=-1,
        )
        for nodes, weights in zip(node_ids, node_weights)
    )
    record_digest = _record_digest(
        node_ids,
        node_weights,
        plane_voxel_tuple,
        plane_coefficient_tuple,
    )
    record = _SparseRecord(
        node_ids=tuple(value.clone() for value in node_ids),  # type: ignore[arg-type]
        node_weights=tuple(value.clone() for value in node_weights),  # type: ignore[arg-type]
        plane_voxel_linear=tuple(value.clone() for value in plane_voxel_tuple),  # type: ignore[arg-type]
        plane_coefficients=tuple(value.clone() for value in plane_coefficient_tuple),  # type: ignore[arg-type]
        digest=record_digest,
    )
    footprint = SparseFootprint(
        voxel_ids_dhw=_linear_to_dhw(union_linear, lattice.output_shape_dhw),
        multiplicity=multiplicity,
        plane_counts=tuple(int(value.numel()) for value in plane_voxel_tuple),
        support_pairs=encoded_pairs,
        lattice_version=lattice.query_version,
        geometry_hash=lattice.geometry_hash,
        kernel_version=WRITER_KERNEL_VERSION,
        mode=lattice.footprint_mode,  # type: ignore[arg-type]
        scanned_voxel_count=int(scanned_voxels),
    )
    _attach_record(footprint, record, lattice, action)
    object.__setattr__(
        footprint, "_pfgr_build_elapsed_seconds", time.perf_counter() - started
    )
    object.__setattr__(footprint, "_pfgr_scanned_bytes", int(scanned_bytes))
    object.__setattr__(footprint, "_pfgr_scan_peak_bytes", int(scan_peak))
    object.__setattr__(footprint, "_pfgr_union_linear", union_linear.clone())
    object.__setattr__(
        footprint,
        "_pfgr_plane_voxel_linear",
        tuple(value.clone() for value in plane_voxel_tuple),
    )
    object.__setattr__(
        footprint,
        "_pfgr_plane_coefficients",
        tuple(value.clone() for value in plane_coefficient_tuple),
    )
    object.__setattr__(
        footprint,
        "_pfgr_accounting_digest",
        _footprint_accounting_digest(footprint),
    )
    public_digest = _footprint_public_digest(footprint)
    object.__setattr__(footprint, "_pfgr_public_digest", public_digest)
    return footprint


def _validate_voxel_ids(
    lattice: PFGRQueryLattice,
    voxel_ids_dhw: Tensor,
    *,
    device: torch.device,
) -> Tensor:
    if (
        not isinstance(voxel_ids_dhw, Tensor)
        or voxel_ids_dhw.ndim != 2
        or voxel_ids_dhw.shape[-1] != 3
    ):
        raise ValueError("voxel_ids_dhw must have shape [Q,3]")
    if voxel_ids_dhw.dtype != torch.long:
        raise TypeError("voxel_ids_dhw must have dtype torch.long")
    if voxel_ids_dhw.device != device:
        raise ValueError("voxel_ids_dhw and delta must share one device")
    lattice._validate_voxel_ids(voxel_ids_dhw, device=device)  # type: ignore[attr-defined]
    _, height, width = lattice.output_shape_dhw
    return (
        voxel_ids_dhw[:, 0] * height * width
        + voxel_ids_dhw[:, 1] * width
        + voxel_ids_dhw[:, 2]
    )


def _lookup_coefficients(
    linear: Tensor,
    support_linear: Tensor,
    coefficients: Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    values = torch.zeros((linear.numel(),), dtype=dtype, device=device)
    if support_linear.numel() == 0 or linear.numel() == 0:
        return values
    # CPU search is deliberate: indexed support is a compact geometry record,
    # and only the requested row coefficients cross the device boundary.
    cpu_linear = linear.detach().to(device="cpu")
    positions = torch.searchsorted(support_linear, cpu_linear)
    safe = positions.clamp(max=support_linear.numel() - 1)
    present = (positions < support_linear.numel()) & (
        support_linear[safe] == cpu_linear
    )
    if bool(present.any()):
        selected = coefficients[safe[present]].to(device=device, dtype=dtype)
        values[present.to(device=device)] = selected
    return values


def query_write_delta(
    lattice: PFGRQueryLattice,
    footprint: SparseFootprint,
    voxel_ids_dhw: Tensor,
    delta: Tensor,
    *,
    chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
) -> Tensor:
    """Query the sparse write response at output voxel centres.

    The returned ``[Q,96]`` tensor remains differentiable with respect to the
    supplied stored ``delta`` while all support/geometry records are frozen.
    It is algebraically ``query(Z + write(delta)) - query(Z)`` and does not
    require ``Z`` or any full-plane clone.
    """

    if not isinstance(lattice, PFGRQueryLattice):
        raise TypeError("lattice must be a PFGRQueryLattice")
    record = _validate_footprint(lattice, footprint)
    _positive_int("chunk_size", chunk_size)
    if not isinstance(delta, Tensor) or delta.ndim != 1 or delta.shape[0] != 96:
        raise ValueError("delta must have shape [96]")
    if not delta.is_floating_point() or delta.dtype != lattice.query_dtype:
        raise TypeError("delta must be floating and match lattice query_dtype")
    if delta.numel() == 0 or not bool(torch.isfinite(delta).all()):
        raise ValueError("delta must be finite")
    linear = _validate_voxel_ids(lattice, voxel_ids_dhw, device=delta.device)
    query_count = int(linear.numel())
    if query_count == 0:
        return torch.empty((0, 96), dtype=delta.dtype, device=delta.device)
    chunks: list[Tensor] = []
    plane_support = record.plane_voxel_linear
    plane_coeff = record.plane_coefficients
    for start in range(0, query_count, chunk_size):
        stop = min(start + chunk_size, query_count)
        rows = linear[start:stop]
        scalars = [
            _lookup_coefficients(
                rows,
                support,
                coefficients,
                dtype=delta.dtype,
                device=delta.device,
            )
            for support, coefficients in zip(plane_support, plane_coeff)
        ]
        chunks.append(
            torch.cat(
                (
                    scalars[0].unsqueeze(-1) * delta[0:32].unsqueeze(0),
                    scalars[1].unsqueeze(-1) * delta[32:64].unsqueeze(0),
                    scalars[2].unsqueeze(-1) * delta[64:96].unsqueeze(0),
                ),
                dim=-1,
            )
        )
    return torch.cat(chunks, dim=0)


def footprint_plane_support(
    footprint: SparseFootprint,
    plane: str,
    *,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Return ``(voxel_linear, scalar_response)`` for one plane.

    Rows are unique and sorted.  This low-level read-only view is intended for
    later proposal/teacher integration; callers receive clones and therefore
    cannot mutate the footprint's immutable response record.
    """

    if plane not in PLANE_NAMES:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")
    record = getattr(footprint, "_pfgr_record", None)
    if not isinstance(record, _SparseRecord):
        raise TypeError("footprint lacks sparse response metadata")
    index = PLANE_NAMES.index(plane)
    voxels = record.plane_voxel_linear[index].clone()
    coefficients = record.plane_coefficients[index].clone()
    if device is not None:
        voxels = voxels.to(device=device)
        coefficients = coefficients.to(device=device)
    return voxels, coefficients


def footprint_writer_nodes(
    footprint: SparseFootprint,
    plane: str,
    *,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Return cloned positive legacy-writer ``(node_id, weight)`` pairs."""

    if plane not in PLANE_NAMES:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")
    record = getattr(footprint, "_pfgr_record", None)
    if not isinstance(record, _SparseRecord):
        raise TypeError("footprint lacks sparse response metadata")
    index = PLANE_NAMES.index(plane)
    nodes = record.node_ids[index].clone()
    weights = record.node_weights[index].clone()
    if device is not None:
        nodes = nodes.to(device=device)
        weights = weights.to(device=device)
    return nodes, weights


def reference_full_write(
    lattice: PFGRQueryLattice,
    state: DynamicTriPlanes,
    action: ActionProposal,
    *,
    chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
    counters: object | None = None,
) -> DynamicTriPlanes:
    """Apply the actual legacy writer as an independent full-write oracle.

    ``chunk_size`` is accepted for API symmetry and future accounting; the
    legacy writer itself operates on one compact plane window.  This function
    intentionally materializes cloned plane tensors and must remain separate
    from the optimized sparse teacher.
    """

    del chunk_size
    if not isinstance(lattice, PFGRQueryLattice):
        raise TypeError("lattice must be a PFGRQueryLattice")
    if not isinstance(state, DynamicTriPlanes):
        raise TypeError("state must be DynamicTriPlanes")
    _validate_action(lattice, action)
    if state.xy.shape[0] != 1:
        raise ValueError("reference_full_write supports subject batch size exactly one")
    if state.xy.dtype != lattice.query_dtype or state.xy.device != action.delta.device:
        raise TypeError("state and action must match lattice dtype/device")
    corrections = PlaneCorrections(
        xy=action.delta[0:32].reshape(1, 32),
        xz=action.delta[32:64].reshape(1, 32),
        yz=action.delta[64:96].reshape(1, 32),
    )
    clone_bytes = sum(
        getattr(state, name).numel() * getattr(state, name).element_size()
        for name in PLANE_NAMES
    )
    if counters is not None:
        counters.add(
            full_plane_clone_bytes=int(clone_bytes), bytes_copied=int(clone_bytes)
        )
    writer = CompactTriPlaneWriteback(support_radius_mm=4.0)
    return writer(
        state,
        action.point_ras_mm.reshape(1, 3),
        corrections,
        lattice.feature_geometry,
    )


def apply_stored_write(
    lattice: PFGRQueryLattice,
    state: DynamicTriPlanes,
    action: ActionProposal,
    *,
    chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
    counters: object | None = None,
) -> DynamicTriPlanes:
    """Execution seam for W4: apply exactly the already-scored delta row."""

    return reference_full_write(
        lattice, state, action, chunk_size=chunk_size, counters=counters
    )


execute_stored_action = apply_stored_write


class StoredActionWriter:
    """W4-compatible callable bound to one canonical context lattice.

    ``ActionProposal`` is validated by W4 before this callable is reached;
    this adapter still checks PFGR state/context identities and delegates to
    the independent legacy writer without recomputing U.
    """

    writer_version = "compact-writeback-4mm-v1"

    def __init__(self, lattice: PFGRQueryLattice) -> None:
        if not isinstance(lattice, PFGRQueryLattice):
            raise TypeError("lattice must be a PFGRQueryLattice")
        self.lattice = lattice

    def __call__(
        self, state: object, context: object, action: object
    ) -> DynamicTriPlanes:
        from .types import ActionProposal, ObservationContext, PFGRState

        if not isinstance(state, PFGRState) or not isinstance(
            context, ObservationContext
        ):
            raise TypeError(
                "StoredActionWriter requires PFGRState and ObservationContext"
            )
        if not isinstance(action, ActionProposal):
            raise TypeError("StoredActionWriter requires ActionProposal")
        if (
            state.context_id != context.context_id
            or action.context_id != context.context_id
        ):
            raise ValueError("stored action/context/state IDs do not match")
        if not state.producer.matches(context.producer.compatibility):
            raise ValueError("stored state producer does not match observation context")
        if action.producer_compatibility_hash != state.producer.digest:
            raise ValueError("stored action producer identity is stale")
        if action.updater_producer_hash != state.producer.updater_hash:
            raise ValueError("stored action updater producer identity is stale")
        if action.writer_version != self.writer_version:
            raise ValueError("stored action writer version does not match writer")
        if action.writer_hash != context.producer.compatibility.writer_hash:
            raise ValueError("stored action writer identity is stale")
        if action.query_version == POINT_QUERY_VERSION:
            if action.query_hash != POINT_QUERY_HASH:
                raise ValueError("stored point-query action identity is stale")
        elif action.query_version == self.lattice.query_version:
            if action.query_hash not in {
                self.lattice.geometry_hash,
                context.producer.compatibility.geometry_query_version_hash,
            }:
                raise ValueError("stored lattice-query action identity is stale")
        else:
            raise ValueError("stored action query version does not match writer")
        expected_geometry = _context_geometry_identity(context)
        if action.geometry_hash not in {self.lattice.geometry_hash, expected_geometry}:
            raise ValueError("stored action geometry identity is stale")
        state.validate_integrity()
        return reference_full_write(self.lattice, state.planes, action)


def make_action_writer(lattice: PFGRQueryLattice) -> StoredActionWriter:
    """Construct the W4 ActionWriter adapter for a context's lattice."""

    return StoredActionWriter(lattice)


class WriterSupportLegalMask:
    """Observation-only legality mask from actual compact-writer support.

    W4 calls this object as ``legal_mask(state, context, points_ras_mm)``.  It
    never inspects target data or proposal values: a point is legal exactly
    when at least one of the three planes retains a strictly positive writer
    node under the full feature-grid affine.
    """

    version = "compact-writeback-4mm-support-legality-v1"

    def __init__(self, lattice: PFGRQueryLattice) -> None:
        if not isinstance(lattice, PFGRQueryLattice):
            raise TypeError("lattice must be a PFGRQueryLattice")
        self.lattice = lattice
        self.writer_version = "compact-writeback-4mm-v1"

    def __call__(self, state: object, context: object, points_ras_mm: Tensor) -> Tensor:
        from .types import ObservationContext, PFGRState

        if not isinstance(state, PFGRState) or not isinstance(
            context, ObservationContext
        ):
            raise TypeError(
                "writer support legality requires PFGRState and ObservationContext"
            )
        context.validate_integrity()
        state.validate_integrity()
        if state.context_id != context.context_id:
            raise ValueError("state/context IDs do not match")
        if context.feature_geometry != self.lattice.feature_geometry:
            raise ValueError(
                "legality lattice does not match observation feature geometry"
            )
        if not isinstance(points_ras_mm, Tensor) or points_ras_mm.ndim != 3:
            raise ValueError("points_ras_mm must have shape [B,N,3]")
        if points_ras_mm.shape[0] != 1 or points_ras_mm.shape[-1] != 3:
            raise ValueError(
                "writer support legality requires subject batch B=1 and [B,N,3] points"
            )
        if (
            points_ras_mm.dtype != state.planes.xy.dtype
            or points_ras_mm.device != state.planes.xy.device
        ):
            raise ValueError("points_ras_mm must match state dtype/device")
        result: list[bool] = []
        with torch.no_grad():
            for point in points_ras_mm[0]:
                nodes, _ = _positive_writer_nodes_at_point(self.lattice, point)
                result.append(any(int(row.numel()) > 0 for row in nodes))
        return torch.tensor(
            result, dtype=torch.bool, device=points_ras_mm.device
        ).reshape(1, -1)


def make_support_legal_mask(lattice: PFGRQueryLattice) -> WriterSupportLegalMask:
    """Construct W4's explicit writer-retained-node legality callable."""

    return WriterSupportLegalMask(lattice)


class CompoundStoredActionWriter:
    """Execute a tuple of stored initial-state actions sequentially.

    This is solely for W4's ``parallel_topk`` diagnostic trace.  Every row is
    validated against the same initial state identity and its stored delta is
    applied exactly once; returned entries are the actual intermediate states
    after each write, never a fabricated full-plane clone shortcut.
    """

    writer_version = "compact-writeback-4mm-v1"

    def __init__(self, lattice: PFGRQueryLattice) -> None:
        if not isinstance(lattice, PFGRQueryLattice):
            raise TypeError("lattice must be a PFGRQueryLattice")
        self.lattice = lattice

    def __call__(
        self,
        state: object,
        context: object,
        actions: tuple[ActionProposal, ...],
    ) -> tuple[DynamicTriPlanes, ...]:
        from .types import ActionProposal, ObservationContext, PFGRState

        if not isinstance(state, PFGRState) or not isinstance(
            context, ObservationContext
        ):
            raise TypeError("compound writer requires PFGRState and ObservationContext")
        if not isinstance(actions, tuple) or not actions:
            raise ValueError("compound writer requires a nonempty tuple of actions")
        context.validate_integrity()
        state.validate_integrity()
        if state.context_id != context.context_id:
            raise ValueError("state/context IDs do not match")
        if context.feature_geometry != self.lattice.feature_geometry:
            raise ValueError("compound writer lattice does not match context geometry")
        current = state.planes
        outputs: list[DynamicTriPlanes] = []
        for action in actions:
            if not isinstance(action, ActionProposal):
                raise TypeError("compound actions must be ActionProposal rows")
            action.validate_integrity()
            if action.context_id != context.context_id:
                raise ValueError("compound action/context IDs do not match")
            if (
                action.state_version != state.state_version
                or action.state_digest != state.state_digest
            ):
                raise ValueError("compound actions must bind to the same initial state")
            if action.producer_compatibility_hash != state.producer.digest:
                raise ValueError("compound action producer identity is stale")
            current = reference_full_write(self.lattice, current, action)
            outputs.append(current)
        return tuple(outputs)


def make_compound_writer(lattice: PFGRQueryLattice) -> CompoundStoredActionWriter:
    """Construct W4's explicit tuple-of-stored-actions writer adapter."""

    return CompoundStoredActionWriter(lattice)


class PointQueryAdapter:
    """Tensor-returning adapter around the existing RAS point sampler."""

    query_version = POINT_QUERY_VERSION
    query_hash = POINT_QUERY_HASH

    def __init__(self) -> None:
        from ..reward import DynamicStatePointQuery

        self._query = DynamicStatePointQuery()

    def __call__(
        self, state: DynamicTriPlanes, points_ras_mm: Tensor, feature_geometry: object
    ) -> Tensor:
        samples = self._query(state, points_ras_mm, feature_geometry)
        return samples.packed


def make_point_query() -> PointQueryAdapter:
    """Return a Tensor-returning parameter-free RAS point query for W4."""

    return PointQueryAdapter()


__all__ = [
    "DEFAULT_QUERY_CHUNK_SIZE",
    "POINT_QUERY_HASH",
    "POINT_QUERY_VERSION",
    "SPARSE_WRITE_VERSION",
    "WRITER_KERNEL_VERSION",
    "CompoundStoredActionWriter",
    "PointQueryAdapter",
    "StoredActionWriter",
    "WriterSupportLegalMask",
    "apply_stored_write",
    "build_footprint",
    "execute_stored_action",
    "footprint_plane_support",
    "footprint_writer_nodes",
    "make_action_writer",
    "make_compound_writer",
    "make_point_query",
    "make_support_legal_mask",
    "query_write_delta",
    "reference_full_write",
]
