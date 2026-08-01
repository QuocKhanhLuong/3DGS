"""Typed query/output contracts for the shared anchor-local structural field."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FieldQueryBatch:
    points_ras_mm: torch.Tensor  # [Q,3]
    anchor_indices: torch.Tensor  # [Q,K], int64
    local_coordinates: torch.Tensor  # [Q,K,3]
    neighbor_valid: torch.Tensor  # [Q,K], bool

    def __post_init__(self) -> None:
        if self.points_ras_mm.ndim != 2 or self.points_ras_mm.shape[1] != 3:
            raise ValueError("points_ras_mm must have shape [Q,3]")
        query_count, neighbor_count = self.anchor_indices.shape
        if self.anchor_indices.dtype is not torch.int64 or self.local_coordinates.shape != (query_count, neighbor_count, 3):
            raise ValueError("anchor indices/local coordinates have incompatible shapes")
        if self.neighbor_valid.shape != (query_count, neighbor_count) or self.neighbor_valid.dtype is not torch.bool:
            raise ValueError("neighbor_valid must be bool with shape [Q,K]")
        if not bool(torch.isfinite(self.points_ras_mm).all() and torch.isfinite(self.local_coordinates).all()):
            raise ValueError("field query coordinates must be finite")


@dataclass(frozen=True)
class StructuralFieldOutput:
    value: torch.Tensor  # [Q,1], NaN only where unsupported
    supported: torch.Tensor  # [Q], bool
    total_weight: torch.Tensor  # [Q,1]
    disagreement: torch.Tensor  # [Q,1]
    local_values: torch.Tensor  # [Q,K,1]
    support_weights: torch.Tensor  # [Q,K,1]

    def __post_init__(self) -> None:
        query_count = self.value.shape[0]
        if self.value.shape != (query_count, 1) or self.supported.shape != (query_count,) or self.supported.dtype is not torch.bool:
            raise ValueError("field value/support shapes are invalid")
        if self.total_weight.shape != (query_count, 1) or self.disagreement.shape != (query_count, 1):
            raise ValueError("field diagnostics must have shape [Q,1]")
        if self.local_values.ndim != 3 or self.local_values.shape[:2] != self.support_weights.shape[:2] or self.local_values.shape[-1] != 1 or self.support_weights.shape[-1] != 1:
            raise ValueError("local field values and weights must have shape [Q,K,1]")
        for tensor in (self.total_weight, self.disagreement, self.local_values, self.support_weights):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("field diagnostics must be finite")
        if bool((self.support_weights < 0).any()) or bool((self.total_weight < 0).any()):
            raise ValueError("field support weights must be non-negative")
        if bool(torch.isnan(self.value[self.supported]).any()) or bool(torch.isfinite(self.value[~self.supported]).any()):
            raise ValueError("field values must be finite exactly on supported queries")
