"""Geometry-only sparse BraTS21 plane and target selection.

The selector consumes only validated headers and a declared episode identity.
It never reads image payloads or segmentation.  Slice indices are an internal
source reference; all public positions and target geometry are expressed in
physical RAS millimetres.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Iterable, Mapping

import numpy as np

from ..contracts.coordinates import PhysicalPlane
from .brats21 import BRATS21_MODALITIES, BraTS21VolumeSummary, plane_from_nifti


@dataclass(frozen=True)
class BraTS21SamplingConfig:
    """Declared geometry-only sampling policy for one maintained episode."""

    orientation: str = "axial"
    context_planes_per_modality: int = 5
    quantile_min: float = 0.15
    quantile_max: float = 0.85
    modality_alignment: str = "aligned"
    train_jitter_fraction: float = 0.15
    validation_jitter_fraction: float = 0.0
    target_policy: str = "gap_midpoint"
    targets_per_episode: int = 1

    def __post_init__(self) -> None:
        if self.orientation != "axial":
            raise ValueError("the maintained BraTS21 protocol supports axial geometry only")
        if self.context_planes_per_modality not in (3, 5, 7, 9):
            raise ValueError("context_planes_per_modality must be one of sparse-3/5/7/9")
        if not (0.0 < self.quantile_min < self.quantile_max < 1.0):
            raise ValueError("quantile_min and quantile_max must satisfy 0 < min < max < 1")
        if self.modality_alignment not in ("aligned", "staggered"):
            raise ValueError("modality_alignment must be aligned or staggered")
        for name in ("train_jitter_fraction", "validation_jitter_fraction"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0 or value > 0.5:
                raise ValueError(f"{name} must be finite and in [0, 0.5]")
        if self.target_policy != "gap_midpoint":
            raise ValueError("the maintained target policy is gap_midpoint")
        if self.targets_per_episode != 1:
            raise ValueError("the maintained initial trainer supports exactly one target per episode")

    def to_dict(self) -> dict[str, object]:
        return {
            "context_planes_per_modality": self.context_planes_per_modality,
            "modality_alignment": self.modality_alignment,
            "orientation": self.orientation,
            "quantile_max": self.quantile_max,
            "quantile_min": self.quantile_min,
            "target_policy": self.target_policy,
            "targets_per_episode": self.targets_per_episode,
            "train_jitter_fraction": self.train_jitter_fraction,
            "validation_jitter_fraction": self.validation_jitter_fraction,
        }

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class BraTS21PlaneSelection:
    """One selected source plane with physical geometry and no image payload."""

    modality_id: str
    role: str
    ordinal: int
    source_slice_index: int
    physical_position_mm: float
    plane: PhysicalPlane
    source_slice_position_index: float | None = None

    def __post_init__(self) -> None:
        if self.role not in ("context", "target"):
            raise ValueError("plane role must be context or target")
        source_position = float(self.source_slice_index if self.source_slice_position_index is None else self.source_slice_position_index)
        if (
            self.source_slice_index < 0
            or not math.isfinite(self.physical_position_mm)
            or not math.isfinite(source_position)
            or source_position < 0.0
        ):
            raise ValueError("plane selection index and physical position are invalid")
        object.__setattr__(self, "source_slice_position_index", source_position)


@dataclass(frozen=True)
class BraTS21EpisodeSamplingPlan:
    """Immutable geometry plan used to create legal context/target metadata."""

    episode_id: str
    target_modality: str
    context: tuple[BraTS21PlaneSelection, ...]
    target: BraTS21PlaneSelection
    protocol_hash: str
    source_geometry_hash: str

    def __post_init__(self) -> None:
        if not self.context or not self.target_modality:
            raise ValueError("sampling plan requires non-empty context and target modality")
        if self.target.role != "target" or any(item.role != "context" for item in self.context):
            raise ValueError("sampling plan roles are inconsistent")
        context_positions = tuple(item.physical_position_mm for item in self.context)
        if any(abs(self.target.physical_position_mm - value) <= 1e-9 for value in context_positions):
            raise ValueError("target physical position overlaps a context plane")
        for digest_name in ("protocol_hash", "source_geometry_hash"):
            digest = getattr(self, digest_name)
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"{digest_name} must be a SHA-256 digest")


def _geometry_hash(summary: BraTS21VolumeSummary) -> str:
    payload = {
        "affine": summary.affine,
        "orientation": summary.orientation,
        "shape_xyz": summary.shape_xyz,
        "spacing_xyz_mm": summary.spacing_xyz_mm,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def physical_slice_positions(summary: BraTS21VolumeSummary) -> tuple[float, ...]:
    """Return axial slice-centre positions along the source physical normal."""

    affine = np.asarray(summary.affine, dtype=np.float64)
    normal = affine[:3, 2] / np.linalg.norm(affine[:3, 2])
    origin = affine[:3, 3]
    positions = []
    for index in range(summary.shape_xyz[2]):
        centre = affine @ np.asarray((0.0, 0.0, float(index), 1.0), dtype=np.float64)
        positions.append(float(np.dot(centre[:3] - origin, normal)))
    result = tuple(positions)
    if len(result) < 3 or any(not math.isfinite(value) for value in result) or any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError("validated BraTS21 geometry must have at least three finite axial positions")
    return result


def _fractional_slice_index(positions: tuple[float, ...], physical_position_mm: float) -> float:
    """Map a physical axial position to the source affine's fractional index."""

    if not math.isfinite(physical_position_mm) or not positions[0] <= physical_position_mm <= positions[-1]:
        raise ValueError("target physical position is outside the source axial extent")
    return float(np.interp(physical_position_mm, np.asarray(positions), np.arange(len(positions), dtype=np.float64)))


def _nearest_unused(positions: tuple[float, ...], desired: float, used: set[int]) -> int:
    candidates = sorted(
        (index for index in range(len(positions)) if index not in used),
        key=lambda index: (abs(positions[index] - desired), index),
    )
    if not candidates:
        raise ValueError("not enough distinct source planes for the sampling protocol")
    return candidates[0]


def _slice_spacing(positions: tuple[float, ...], index: int) -> float:
    neighbours = [abs(positions[index] - positions[other]) for other in (index - 1, index + 1) if 0 <= other < len(positions)]
    return min(neighbours) if neighbours else 1.0


def _local_inter_plane_spacing(positions: tuple[float, ...], desired: float) -> float:
    """Return the source-plane spacing local to a geometry-only target."""

    if len(positions) < 2:
        return 1.0
    insertion = int(np.searchsorted(np.asarray(positions, dtype=np.float64), desired, side="left"))
    if insertion <= 0:
        return abs(positions[1] - positions[0])
    if insertion >= len(positions):
        return abs(positions[-1] - positions[-2])
    return abs(positions[insertion] - positions[insertion - 1])


def _quantile_indices(
    positions: tuple[float, ...], *, config: BraTS21SamplingConfig, seed_payload: str, jitter_fraction: float,
) -> tuple[int, ...]:
    order = sorted(range(len(positions)), key=lambda index: (positions[index], index))
    sorted_positions = tuple(positions[index] for index in order)
    rng = random.Random(int.from_bytes(hashlib.sha256(seed_payload.encode()).digest()[:8], "big"))
    used: set[int] = set()
    selected: list[int] = []
    for ordinal in range(config.context_planes_per_modality):
        quantile = config.quantile_min + (config.quantile_max - config.quantile_min) * ordinal / (config.context_planes_per_modality - 1)
        desired = float(np.quantile(np.asarray(sorted_positions, dtype=np.float64), quantile, method="linear"))
        if jitter_fraction > 0.0:
            local_spacing = _local_inter_plane_spacing(sorted_positions, desired)
            desired += rng.uniform(-jitter_fraction, jitter_fraction) * local_spacing
        chosen_sorted = min(
            (index for index in range(len(sorted_positions)) if order[index] not in used),
            key=lambda index: (abs(sorted_positions[index] - desired), index),
        )
        chosen = order[chosen_sorted]
        used.add(chosen)
        selected.append(chosen)
    return tuple(sorted(selected, key=lambda index: (positions[index], index)))


def build_sampling_plan(
    summaries: Mapping[str, BraTS21VolumeSummary],
    *,
    episode_id: str,
    target_modality: str = "flair",
    split: str = "validation",
    seed: int = 0,
    inplane_stride_vu: tuple[int, int] = (1, 1),
    config: BraTS21SamplingConfig | None = None,
) -> BraTS21EpisodeSamplingPlan:
    """Select aligned/staggered context planes from headers only."""

    config = config or BraTS21SamplingConfig()
    missing = [modality for modality in BRATS21_MODALITIES if modality not in summaries]
    if missing:
        raise ValueError(f"sampling requires all four modalities; missing {missing}")
    if target_modality not in BRATS21_MODALITIES:
        raise ValueError(f"unknown target modality: {target_modality}")
    reference = summaries[BRATS21_MODALITIES[0]]
    reference_geometry_hash = _geometry_hash(reference)
    for modality in BRATS21_MODALITIES[1:]:
        if _geometry_hash(summaries[modality]) != reference_geometry_hash:
            raise ValueError("aligned sampling requires compatible shape, spacing, affine, and orientation")
    jitter = config.train_jitter_fraction if split == "train" else config.validation_jitter_fraction
    reference_positions = physical_slice_positions(reference)
    reference_indices = _quantile_indices(
        reference_positions, config=config, seed_payload=f"{episode_id}:{seed}:aligned", jitter_fraction=jitter,
    )
    context: list[BraTS21PlaneSelection] = []
    for modality_index, modality in enumerate(BRATS21_MODALITIES):
        summary = summaries[modality]
        positions = physical_slice_positions(summary)
        if config.modality_alignment == "aligned":
            indices = reference_indices
        else:
            offset = (modality_index - 1.5) * (0.25 * _slice_spacing(positions, reference_indices[0]))
            indices = tuple(
                _nearest_unused(positions, positions[index] + offset, set()) for index in reference_indices
            )
        used: set[int] = set()
        for ordinal, index in enumerate(indices):
            if index in used:
                raise ValueError("staggered modality offsets produced duplicate planes")
            used.add(index)
            plane_id = f"{episode_id}:{modality}:context:{ordinal}"
            context.append(
                BraTS21PlaneSelection(
                    modality, "context", ordinal, index, positions[index],
                    plane_from_nifti(summary.affine, summary.shape_xyz, index, observation_id=plane_id, inplane_stride_vu=inplane_stride_vu),
                )
            )
    sorted_context = sorted(context, key=lambda item: (item.physical_position_mm, item.modality_id, item.ordinal))
    all_context_positions = tuple(sorted({item.physical_position_mm for item in sorted_context}))
    target_context_positions = tuple(sorted({
        item.physical_position_mm for item in sorted_context if item.modality_id == target_modality
    }))
    if len(all_context_positions) < 3 or len(target_context_positions) < 2:
        raise ValueError("sampling protocol produced too few legal context positions")
    legal_target_gaps = tuple(
        (left, right)
        for left, right in zip(target_context_positions, target_context_positions[1:])
        if not any(abs((left + right) / 2.0 - context_position) <= 1e-9 for context_position in all_context_positions)
    )
    if not legal_target_gaps:
        raise ValueError("sampling protocol produced no target-modality gap free of context planes")
    target_rng = random.Random(int.from_bytes(hashlib.sha256(f"{episode_id}:{seed}:target".encode()).digest()[:8], "big"))
    left, right = legal_target_gaps[target_rng.randrange(len(legal_target_gaps))]
    target_position = (left + right) / 2.0
    target_summary = summaries[target_modality]
    target_positions = physical_slice_positions(target_summary)
    target_slice_position_index = _fractional_slice_index(target_positions, target_position)
    target_index = int(np.floor(target_slice_position_index + 0.5))
    target_id = f"{episode_id}:{target_modality}:target:0"
    target = BraTS21PlaneSelection(
        target_modality, "target", 0, target_index, target_position,
        plane_from_nifti(
            target_summary.affine,
            target_summary.shape_xyz,
            target_index,
            observation_id=target_id,
            inplane_stride_vu=inplane_stride_vu,
            slice_position_index=target_slice_position_index,
        ),
        target_slice_position_index,
    )
    if not left < target.physical_position_mm < right:
        raise ValueError("target midpoint is not strictly between legal context positions")
    protocol_payload = {
        "config": config.to_dict(), "episode_id": episode_id, "seed": seed,
        "split": split, "target_modality": target_modality,
        "context": [(item.modality_id, item.source_slice_index, item.physical_position_mm) for item in sorted_context],
        "target": (
            target.modality_id,
            target.source_slice_index,
            target.source_slice_position_index,
            target.physical_position_mm,
            target_position,
        ),
        "source_geometry_hash": reference_geometry_hash,
    }
    protocol_hash = hashlib.sha256(json.dumps(protocol_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return BraTS21EpisodeSamplingPlan(
        episode_id, target_modality, tuple(context), target, protocol_hash, reference_geometry_hash,
    )


__all__ = [
    "BraTS21EpisodeSamplingPlan",
    "BraTS21PlaneSelection",
    "BraTS21SamplingConfig",
    "build_sampling_plan",
    "physical_slice_positions",
]
