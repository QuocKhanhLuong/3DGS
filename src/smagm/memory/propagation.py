"""Deterministic bounded static anchor--Gaussian propagation (P0/P1)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import torch

from ..anchors import AnchorBatch
from ..gaussians import AmplitudeGaugePolicy, restore_gauge_fixed_gaussian_batch
from .contracts import GaussianMemory, GaussianMemoryBank, PrimitiveKind, gaussian_memory_hash
from .observability import propagated_observability


@dataclass(frozen=True)
class PropagationConfig:
    variant: str = "p1"
    rounds: int = 1
    step_mm: float = 1.0
    children_per_parent_per_round: int = 1
    duplicate_radius_mm: float = 0.25
    uncertainty_growth_per_mm: float = 0.1
    maximum_structural_primitives: int = 512
    maximum_volumetric_primitives: int = 512

    def __post_init__(self) -> None:
        if self.variant not in ("p0", "p1"):
            raise ValueError("the reference propagation variant must be p0 or p1")
        if self.rounds < 0 or self.children_per_parent_per_round <= 0:
            raise ValueError("propagation rounds and child budget are invalid")
        for value in (self.step_mm, self.duplicate_radius_mm, self.uncertainty_growth_per_mm):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("propagation distances and uncertainty growth must be positive finite")
        if self.maximum_structural_primitives <= 0 or self.maximum_volumetric_primitives <= 0:
            raise ValueError("primitive budgets must be positive")

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class PropagationTransaction:
    round_index: int
    parent_memory_hash: str
    child_memory_hash: str
    accepted_primitive_ids: tuple[str, ...]
    rejected_out_of_bounds: int
    rejected_unsupported: int
    rejected_duplicate_or_budget: int
    config_hash: str
    transaction_hash: str

    def __post_init__(self) -> None:
        payload = self.__dict__.copy(); claimed = payload.pop("transaction_hash")
        actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if claimed != actual:
            raise ValueError("transaction_hash does not bind the exact propagation transaction")


def _transaction(**payload: object) -> PropagationTransaction:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return PropagationTransaction(**payload, transaction_hash=digest)  # type: ignore[arg-type]


def _append_gauge_preserving_children(
    bank: GaussianMemoryBank,
    *,
    centers: torch.Tensor,
    parent_indices: torch.Tensor,
    child_ids: tuple[str, ...],
):
    """Append children without treating gauge-fixed amplitudes as raw values.

    Existing bank amplitudes are already mean-centred by the canonical factory.
    Zero-valued child log amplitudes preserve that gauge exactly and, crucially,
    leave every pre-existing primitive unchanged.
    """

    existing = bank.gaussians
    if existing.gauge_policy is AmplitudeGaugePolicy.LEGACY_RAW or existing.gauge_config_hash is None:
        raise ValueError("propagation requires explicit Phase-1 amplitude-gauge provenance")
    child_log_amplitude = existing.log_support_amplitude.new_zeros((len(child_ids), 1))
    return restore_gauge_fixed_gaussian_batch(
        centers_ras_mm=centers,
        covariance_factor=torch.cat((existing.covariance_factor, existing.covariance_factor[parent_indices])),
        log_support_amplitude=torch.cat((existing.log_support_amplitude, child_log_amplitude)),
        appearance=torch.cat((existing.appearance, existing.appearance[parent_indices])),
        appearance_valid=torch.cat((existing.appearance_valid, existing.appearance_valid[parent_indices])),
        covariance_epsilon=existing.covariance_epsilon,
        primitive_kind=(bank.kind.value,) * centers.shape[0],
        primitive_id=tuple(existing.primitive_id or ()) + child_ids,
        gauge_policy=existing.gauge_policy,
        gauge_config_hash=existing.gauge_config_hash,
    )


def _propagate_bank(
    bank: GaussianMemoryBank, anchors: AnchorBatch, *, config: PropagationConfig,
    round_index: int, bounds_min: torch.Tensor, bounds_max: torch.Tensor, maximum_count: int,
) -> tuple[GaussianMemoryBank, tuple[str, ...], tuple[int, int, int]]:
    existing = bank.gaussians
    accepted_centers: list[torch.Tensor] = []
    parent_indices: list[int] = []
    child_ids: list[str] = []
    rejected_bounds = rejected_unsupported = rejected_other = 0
    available = max(0, maximum_count - existing.count)
    for parent_index in range(existing.count):
        if len(child_ids) >= available:
            rejected_other += existing.count - parent_index
            break
        anchor_id = bank.anchor_ids[parent_index]
        try:
            anchor_index = anchors.anchor_ids.index(anchor_id)
        except ValueError as error:
            raise ValueError("primitive references an unknown anchor") from error
        frame = anchors.frame_axes_ras[anchor_index]
        direction_axes = (0, 1) if bank.kind is PrimitiveKind.STRUCTURAL else (0, 1, 2)
        for child_offset in range(config.children_per_parent_per_round):
            axis_index = direction_axes[(parent_index + round_index + child_offset) % len(direction_axes)]
            sign = -1.0 if (parent_index + round_index + child_offset) % 2 else 1.0
            direction = sign * frame[:, axis_index]
            proposal = existing.centers_ras_mm[parent_index] + config.step_mm * direction
            if bool(((proposal < bounds_min) | (proposal > bounds_max)).any()):
                rejected_bounds += 1; continue
            local = frame.transpose(0, 1) @ (proposal - anchors.centers_ras_mm[anchor_index])
            if bool((local.abs() > anchors.support_scales_mm[anchor_index]).any()):
                rejected_unsupported += 1; continue
            comparison = torch.cat((existing.centers_ras_mm, torch.stack(accepted_centers) if accepted_centers else existing.centers_ras_mm.new_empty((0, 3))))
            if bool((torch.linalg.vector_norm(comparison - proposal, dim=1) <= config.duplicate_radius_mm).any()):
                rejected_other += 1; continue
            identity = hashlib.sha256(f"{existing.primitive_id[parent_index]}:{round_index}:{axis_index}:{sign}".encode()).hexdigest()
            accepted_centers.append(proposal); parent_indices.append(parent_index); child_ids.append(f"{bank.kind.value.lower()}-child-{identity[:16]}")
            if len(child_ids) >= available:
                break
    if not child_ids:
        return bank, (), (rejected_bounds, rejected_unsupported, rejected_other)
    index = torch.tensor(parent_indices, dtype=torch.int64, device=existing.centers_ras_mm.device)
    new_centers = torch.cat((existing.centers_ras_mm, torch.stack(accepted_centers)))
    gaussians = _append_gauge_preserving_children(
        bank,
        centers=new_centers,
        parent_indices=index,
        child_ids=tuple(child_ids),
    )
    parent_ids = tuple(existing.primitive_id[i] for i in parent_indices)  # type: ignore[index]
    provenance = tuple(hashlib.sha256(f"propagate:{parent}:{round_index}:{config.config_hash}".encode()).hexdigest() for parent in parent_ids)
    parent_observation = bank.observability
    child_observation = propagated_observability(
        parent_observation, index, uncertainty_growth=config.step_mm * config.uncertainty_growth_per_mm,
        update_round=round_index,
    )
    from .contracts import PrimitiveObservability
    observability = PrimitiveObservability(
        evidence_count=torch.cat((parent_observation.evidence_count, child_observation.evidence_count)),
        coverage=torch.cat((parent_observation.coverage, child_observation.coverage)),
        disagreement=torch.cat((parent_observation.disagreement, child_observation.disagreement)),
        uncertainty=torch.cat((parent_observation.uncertainty, child_observation.uncertainty)),
        propagation_depth=torch.cat((parent_observation.propagation_depth, child_observation.propagation_depth)),
        update_round=torch.cat((parent_observation.update_round, child_observation.update_round)),
    )
    result = GaussianMemoryBank(
        bank.kind, gaussians, bank.anchor_ids + tuple(bank.anchor_ids[i] for i in parent_indices),
        bank.parent_primitive_ids + parent_ids, bank.provenance_hashes + provenance, observability,
    )
    return result, tuple(child_ids), (rejected_bounds, rejected_unsupported, rejected_other)


def propagate_memory(
    memory: GaussianMemory, anchors: AnchorBatch, *, config: PropagationConfig,
    bounds_min_ras_mm: torch.Tensor, bounds_max_ras_mm: torch.Tensor,
) -> tuple[GaussianMemory, tuple[PropagationTransaction, ...]]:
    """Apply fixed P1 rounds; P0 returns the exact input memory object."""
    if bounds_min_ras_mm.shape != (3,) or bounds_max_ras_mm.shape != (3,) or not bool((bounds_min_ras_mm < bounds_max_ras_mm).all()):
        raise ValueError("patient bounds must be ordered [3] RAS-mm tensors")
    if config.variant == "p0" or config.rounds == 0:
        return memory, ()
    current = memory; transactions = []
    for round_index in range(1, config.rounds + 1):
        structural, structural_ids, rejected_s = _propagate_bank(
            current.structural, anchors, config=config, round_index=round_index,
            bounds_min=bounds_min_ras_mm, bounds_max=bounds_max_ras_mm,
            maximum_count=config.maximum_structural_primitives,
        )
        volumetric, volumetric_ids, rejected_v = _propagate_bank(
            current.volumetric, anchors, config=config, round_index=round_index,
            bounds_min=bounds_min_ras_mm, bounds_max=bounds_max_ras_mm,
            maximum_count=config.maximum_volumetric_primitives,
        )
        digest = gaussian_memory_hash(structural, volumetric, current.modality_ids)
        updated = GaussianMemory(structural, volumetric, current.modality_ids, digest)
        payload = dict(
            round_index=round_index, parent_memory_hash=current.memory_hash, child_memory_hash=updated.memory_hash,
            accepted_primitive_ids=structural_ids + volumetric_ids,
            rejected_out_of_bounds=rejected_s[0] + rejected_v[0],
            rejected_unsupported=rejected_s[1] + rejected_v[1],
            rejected_duplicate_or_budget=rejected_s[2] + rejected_v[2], config_hash=config.config_hash,
        )
        transactions.append(_transaction(**payload)); current = updated
    return current, tuple(transactions)
