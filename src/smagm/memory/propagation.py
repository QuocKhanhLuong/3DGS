"""Deterministic bounded static anchor--Gaussian propagation (P0/P1)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Literal

import torch

from ..anchors import AnchorBatch
from ..gaussians import AmplitudeGaugePolicy, restore_gauge_fixed_gaussian_batch
from .contracts import GaussianMemory, GaussianMemoryBank, PrimitiveKind, gaussian_memory_hash
from .observability import propagated_observability


@dataclass(frozen=True)
class PropagationConfig:
    """Resolved fixed P0/P1 propagation policy and final-state capacities.

    ``maximum_*_primitives`` are inclusive end-state capacities, rather than
    numbers of children to add.  P1 validates the supplied seed memory against
    them before it starts, so propagation never deletes or silently overlooks
    an over-budget seed state.  Volumetric children use only the local normal
    in alternating ``+/-`` directions; structural children are explicitly
    tangent-only or disabled.
    """

    variant: str = "p1"
    rounds: int = 1
    step_mm: float = 1.0
    children_per_parent_per_round: int = 1
    duplicate_radius_mm: float = 0.25
    uncertainty_growth_per_mm: float = 0.1
    maximum_structural_primitives: int = 512
    maximum_volumetric_primitives: int = 512
    maximum_children_per_anchor: int = 8
    maximum_patient_primitives: int | None = None
    maximum_uncertainty: float | None = None
    minimum_evidence_gain: float = 0.0
    minimum_cross_modality_agreement: float = 0.0
    structural_propagation_policy: Literal["tangent_only", "none"] = "tangent_only"
    maximum_total_anchors: int | None = None
    structural_seed_budget: int | None = None
    volumetric_seed_budget: int | None = None
    propagation_reserved_budget: int | None = None

    def __post_init__(self) -> None:
        if self.variant not in ("p0", "p1"):
            raise ValueError("the reference propagation variant must be p0 or p1")
        if self.rounds < 0 or self.children_per_parent_per_round <= 0:
            raise ValueError("propagation rounds and child budget are invalid")
        if self.structural_propagation_policy not in ("tangent_only", "none"):
            raise ValueError("structural_propagation_policy must be tangent_only or none")
        for value in (self.step_mm, self.duplicate_radius_mm, self.uncertainty_growth_per_mm):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("propagation distances and uncertainty growth must be positive finite")
        if self.maximum_structural_primitives <= 0 or self.maximum_volumetric_primitives <= 0:
            raise ValueError("primitive budgets must be positive")
        if self.maximum_children_per_anchor <= 0:
            raise ValueError("maximum_children_per_anchor must be positive")
        if self.maximum_patient_primitives is not None and self.maximum_patient_primitives <= 0:
            raise ValueError("maximum_patient_primitives must be positive when configured")
        for name in ("maximum_total_anchors", "structural_seed_budget", "volumetric_seed_budget"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.propagation_reserved_budget is not None and (
            isinstance(self.propagation_reserved_budget, bool)
            or not isinstance(self.propagation_reserved_budget, int)
            or self.propagation_reserved_budget < 0
        ):
            raise ValueError("propagation_reserved_budget must be a non-negative integer or None")
        if self.variant == "p1" and self.rounds > 0 and self.propagation_reserved_budget == 0:
            raise ValueError("P1 requires a positive propagation_reserved_budget")
        if (
            self.maximum_patient_primitives is not None
            and self.propagation_reserved_budget is not None
            and self.propagation_reserved_budget > self.maximum_patient_primitives
        ):
            raise ValueError("propagation_reserved_budget cannot exceed maximum_patient_primitives")
        if self.maximum_uncertainty is not None and (not math.isfinite(self.maximum_uncertainty) or self.maximum_uncertainty < 0):
            raise ValueError("maximum_uncertainty must be finite and non-negative when configured")
        if not math.isfinite(self.minimum_evidence_gain) or self.minimum_evidence_gain < 0:
            raise ValueError("minimum_evidence_gain must be finite and non-negative")
        if not 0.0 <= self.minimum_cross_modality_agreement <= 1.0:
            raise ValueError("minimum_cross_modality_agreement must be in [0,1]")

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class PropagationTransaction:
    round_index: int
    parent_memory_hash: str
    child_memory_hash: str
    accepted_primitive_ids: tuple[str, ...]
    proposal_count: int
    accepted_count: int
    rejected_out_of_bounds: int
    rejected_unsupported: int
    rejected_duplicate: int
    rejected_budget: int
    rejected_uncertainty: int
    rejected_invalid: int
    rejected_no_meaningful_gain: int
    config_hash: str
    transaction_hash: str

    def __post_init__(self) -> None:
        counters = (
            self.proposal_count,
            self.accepted_count,
            self.rejected_out_of_bounds,
            self.rejected_unsupported,
            self.rejected_duplicate,
            self.rejected_budget,
            self.rejected_uncertainty,
            self.rejected_invalid,
            self.rejected_no_meaningful_gain,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counters):
            raise ValueError("propagation transaction counters must be non-negative integers")
        if self.accepted_count != len(self.accepted_primitive_ids):
            raise ValueError("accepted_count must match accepted_primitive_ids")
        if self.proposal_count != self.accepted_count + sum(counters[2:]):
            raise ValueError("proposal_count must account for every accepted or rejected proposal")
        payload = self.__dict__.copy(); claimed = payload.pop("transaction_hash")
        actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if claimed != actual:
            raise ValueError("transaction_hash does not bind the exact propagation transaction")


def _transaction(**payload: object) -> PropagationTransaction:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return PropagationTransaction(**payload, transaction_hash=digest)  # type: ignore[arg-type]


@dataclass
class _PropagationCounts:
    """Internal exhaustive P1 proposal accounting for one memory bank."""

    proposal_count: int = 0
    accepted_count: int = 0
    rejected_out_of_bounds: int = 0
    rejected_unsupported: int = 0
    rejected_duplicate: int = 0
    rejected_budget: int = 0
    rejected_uncertainty: int = 0
    rejected_invalid: int = 0
    rejected_no_meaningful_gain: int = 0

    def merged(self, other: "_PropagationCounts") -> "_PropagationCounts":
        return _PropagationCounts(
            proposal_count=self.proposal_count + other.proposal_count,
            accepted_count=self.accepted_count + other.accepted_count,
            rejected_out_of_bounds=self.rejected_out_of_bounds + other.rejected_out_of_bounds,
            rejected_unsupported=self.rejected_unsupported + other.rejected_unsupported,
            rejected_duplicate=self.rejected_duplicate + other.rejected_duplicate,
            rejected_budget=self.rejected_budget + other.rejected_budget,
            rejected_uncertainty=self.rejected_uncertainty + other.rejected_uncertainty,
            rejected_invalid=self.rejected_invalid + other.rejected_invalid,
            rejected_no_meaningful_gain=self.rejected_no_meaningful_gain + other.rejected_no_meaningful_gain,
        )


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


def _cross_modality_agreement(anchors: AnchorBatch, anchor_index: int) -> float:
    """Return a bounded agreement score for valid normalized appearances.

    Appearance slots are modality-specific, so this gate is deliberately
    opt-in.  When enabled, the configured threshold means agreement of the
    legal, context-only normalized appearance values at an anchor; it is not
    a claim that raw MRI intensities are interchangeable across modalities.
    With fewer than two valid modalities, agreement is unavailable and the
    score is zero rather than treating modality presence as agreement.
    """

    valid = anchors.appearance_valid[anchor_index]
    values = anchors.appearance[anchor_index][valid]
    if values.numel() < 2:
        return 0.0
    pair_indices = torch.triu_indices(values.numel(), values.numel(), offset=1, device=values.device)
    differences = (values[pair_indices[0]] - values[pair_indices[1]]).abs()
    return float((1.0 - differences.clamp(0.0, 1.0).mean()).detach())


def _proposal_axis_and_sign(
    kind: PrimitiveKind,
    config: PropagationConfig,
    *,
    parent_index: int,
    round_index: int,
    child_offset: int,
) -> tuple[int, float] | None:
    """Return a fixed local-frame direction without consulting image content.

    Anchor-frame columns are ``(t1, t2, n)``.  Structural P1 children remain
    in the tangent plane, while volumetric children use only the local normal.
    The phase offset makes both signs reachable across parents and rounds while
    retaining a deterministic bounded schedule.
    """

    if kind is PrimitiveKind.STRUCTURAL:
        if config.structural_propagation_policy == "none":
            return None
        directions = ((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0))
    else:
        directions = ((2, 1.0), (2, -1.0))
    return directions[(parent_index + round_index + child_offset) % len(directions)]


def _propagate_bank(
    bank: GaussianMemoryBank, anchors: AnchorBatch, *, config: PropagationConfig,
    round_index: int, bounds_min: torch.Tensor, bounds_max: torch.Tensor, maximum_count: int,
    supported_anchor_mask: torch.Tensor | None,
    source_inverse_affine: torch.Tensor | None,
    source_shape_xyz: tuple[int, int, int] | None,
) -> tuple[GaussianMemoryBank, tuple[str, ...], _PropagationCounts]:
    existing = bank.gaussians
    if maximum_count < existing.count:
        raise ValueError("propagation capacity cannot be below the supplied seed-bank count")
    if bank.kind is PrimitiveKind.STRUCTURAL and config.structural_propagation_policy == "none":
        return bank, (), _PropagationCounts()

    accepted_centers: list[torch.Tensor] = []
    parent_indices: list[int] = []
    child_ids: list[str] = []
    counts = _PropagationCounts()
    # Count accepted descendants already present in the immutable bank so the
    # per-anchor limit remains binding across propagation rounds.
    children_by_anchor: dict[str, int] = {}
    for parent_id, anchor_id in zip(bank.parent_primitive_ids, bank.anchor_ids):
        if parent_id is not None:
            children_by_anchor[anchor_id] = children_by_anchor.get(anchor_id, 0) + 1
    for parent_index in range(existing.count):
        anchor_id = bank.anchor_ids[parent_index]
        try:
            anchor_index = anchors.anchor_ids.index(anchor_id)
        except ValueError as error:
            raise ValueError("primitive references an unknown anchor") from error
        frame = anchors.frame_axes_ras[anchor_index]
        anchor_children = children_by_anchor.get(anchor_id, 0)
        for child_offset in range(config.children_per_parent_per_round):
            direction_spec = _proposal_axis_and_sign(
                bank.kind,
                config,
                parent_index=parent_index,
                round_index=round_index,
                child_offset=child_offset,
            )
            if direction_spec is None:
                continue
            counts.proposal_count += 1
            axis_index, sign = direction_spec
            if supported_anchor_mask is not None and not bool(supported_anchor_mask[anchor_index]):
                counts.rejected_no_meaningful_gain += 1
                continue
            if config.minimum_evidence_gain > 0.0 and float(anchors.geometry_confidence[anchor_index, 0]) < config.minimum_evidence_gain:
                counts.rejected_no_meaningful_gain += 1
                continue
            if config.minimum_cross_modality_agreement > 0.0:
                agreement = _cross_modality_agreement(anchors, anchor_index)
                if agreement < config.minimum_cross_modality_agreement:
                    counts.rejected_no_meaningful_gain += 1
                    continue
            if config.maximum_uncertainty is not None and float(bank.observability.uncertainty[parent_index, 0]) > config.maximum_uncertainty:
                counts.rejected_uncertainty += 1
                continue
            if not bool(torch.isfinite(frame).all()):
                counts.rejected_invalid += 1
                continue
            if not bool(anchors.geometry.frame_validity[anchor_index, axis_index]):
                counts.rejected_unsupported += 1
                continue
            if anchor_children >= config.maximum_children_per_anchor:
                counts.rejected_budget += 1
                continue
            if existing.count + len(child_ids) >= maximum_count:
                counts.rejected_budget += 1
                continue
            direction = sign * frame[:, axis_index]
            proposal = existing.centers_ras_mm[parent_index] + config.step_mm * direction
            if not bool(torch.isfinite(proposal).all()):
                counts.rejected_invalid += 1
                continue
            if bool(((proposal < bounds_min) | (proposal > bounds_max)).any()):
                counts.rejected_out_of_bounds += 1
                continue
            if source_inverse_affine is not None and source_shape_xyz is not None:
                homogeneous = torch.cat((proposal, proposal.new_ones(1)))
                source_index = source_inverse_affine.to(device=proposal.device, dtype=proposal.dtype) @ homogeneous
                upper = proposal.new_tensor([float(value - 1) for value in source_shape_xyz] + [float("inf")])
                if bool((source_index[:3] < -1e-5).any()) or bool((source_index[:3] > upper[:3] + 1e-5).any()):
                    counts.rejected_out_of_bounds += 1
                    continue
            local = frame.transpose(0, 1) @ (proposal - anchors.centers_ras_mm[anchor_index])
            if bool((local.abs() > anchors.support_scales_mm[anchor_index]).any()):
                counts.rejected_unsupported += 1
                continue
            comparison = torch.cat((existing.centers_ras_mm, torch.stack(accepted_centers) if accepted_centers else existing.centers_ras_mm.new_empty((0, 3))))
            if bool((torch.linalg.vector_norm(comparison - proposal, dim=1) <= config.duplicate_radius_mm).any()):
                counts.rejected_duplicate += 1
                continue
            identity = hashlib.sha256(f"{existing.primitive_id[parent_index]}:{round_index}:{axis_index}:{sign}".encode()).hexdigest()
            accepted_centers.append(proposal); parent_indices.append(parent_index); child_ids.append(f"{bank.kind.value.lower()}-child-{identity[:16]}")
            anchor_children += 1
            children_by_anchor[anchor_id] = anchor_children
            counts.accepted_count += 1
    if not child_ids:
        return bank, (), counts
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
    covariance = result.gaussians.covariance()
    if not bool(torch.isfinite(covariance).all()) or not bool((torch.linalg.eigvalsh(covariance) > 0).all()):
        raise FloatingPointError("propagation produced a non-finite or non-positive-definite covariance")
    return result, tuple(child_ids), counts


def _preflight_seed_budgets(memory: GaussianMemory, config: PropagationConfig) -> None:
    """Fail before P1 if caller-supplied seeds already exceed final capacities.

    Seed counts are only available once a caller supplies its immutable memory.
    Treating the limits as final capacities here keeps that late binding safe:
    P1 neither removes seed primitives nor starts a partial update that cannot
    satisfy its declared per-bank or patient budget.
    """

    if memory.structural.gaussians.count > config.maximum_structural_primitives:
        raise ValueError("structural seed primitive count exceeds maximum_structural_primitives")
    if memory.volumetric.gaussians.count > config.maximum_volumetric_primitives:
        raise ValueError("volumetric seed primitive count exceeds maximum_volumetric_primitives")
    if (
        config.maximum_patient_primitives is not None
        and memory.primitive_count > config.maximum_patient_primitives
    ):
        raise ValueError("seed primitive count exceeds maximum_patient_primitives")


def validate_seed_and_reserve_budgets(
    memory: GaussianMemory,
    anchors: AnchorBatch,
    *,
    config: PropagationConfig,
) -> None:
    """Validate explicit product seed/reserve capacities before P1 begins.

    This is separate from propagation itself because anchor count exists only
    at static-state construction.  It makes an impossible configuration fail
    before target commitment and before a partial propagation transaction.
    """

    if config.variant == "p0" or config.rounds == 0:
        return
    _preflight_seed_budgets(memory, config)
    if config.maximum_total_anchors is not None and anchors.count > config.maximum_total_anchors:
        raise ValueError("anchor count exceeds maximum_total_anchors")
    if config.structural_seed_budget is not None and memory.structural.gaussians.count > config.structural_seed_budget:
        raise ValueError("structural seed primitive count exceeds structural_seed_budget")
    if config.volumetric_seed_budget is not None and memory.volumetric.gaussians.count > config.volumetric_seed_budget:
        raise ValueError("volumetric seed primitive count exceeds volumetric_seed_budget")
    if config.propagation_reserved_budget is not None:
        if config.maximum_patient_primitives is None:
            raise ValueError("P1 reserve requires maximum_patient_primitives")
        if memory.primitive_count + config.propagation_reserved_budget > config.maximum_patient_primitives:
            raise ValueError("seed primitive count plus propagation_reserved_budget exceeds maximum_patient_primitives")


def propagate_memory(
    memory: GaussianMemory, anchors: AnchorBatch, *, config: PropagationConfig,
    bounds_min_ras_mm: torch.Tensor, bounds_max_ras_mm: torch.Tensor,
    supported_anchor_mask: torch.Tensor | None = None,
    source_affine_ras_from_index: torch.Tensor | None = None,
    source_shape_xyz: tuple[int, int, int] | None = None,
) -> tuple[GaussianMemory, tuple[PropagationTransaction, ...]]:
    """Apply fixed P1 rounds from legal state and geometry only.

    The API deliberately accepts no target, audit, segmentation, or image
    payload.  P0 returns the exact input memory object and no transaction.
    """
    if bounds_min_ras_mm.shape != (3,) or bounds_max_ras_mm.shape != (3,) or not bool((bounds_min_ras_mm < bounds_max_ras_mm).all()):
        raise ValueError("patient bounds must be ordered [3] RAS-mm tensors")
    if supported_anchor_mask is not None and (
        supported_anchor_mask.shape != (anchors.count,) or supported_anchor_mask.dtype is not torch.bool
    ):
        raise ValueError("supported_anchor_mask must be bool [anchor_count]")
    source_inverse_affine: torch.Tensor | None = None
    if source_affine_ras_from_index is not None or source_shape_xyz is not None:
        if source_affine_ras_from_index is None or source_shape_xyz is None:
            raise ValueError("source affine and source shape must be supplied together for oriented containment")
        if (
            not isinstance(source_shape_xyz, tuple)
            or len(source_shape_xyz) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in source_shape_xyz)
        ):
            raise ValueError("source_shape_xyz must be a positive integer triple")
        if source_affine_ras_from_index.shape != (4, 4) or not bool(torch.isfinite(source_affine_ras_from_index).all()):
            raise ValueError("source_affine_ras_from_index must be a finite [4,4] tensor")
        if not bool(torch.allclose(
            source_affine_ras_from_index[3],
            source_affine_ras_from_index.new_tensor((0.0, 0.0, 0.0, 1.0)),
            atol=1e-6,
            rtol=0.0,
        )):
            raise ValueError("source_affine_ras_from_index must be homogeneous")
        try:
            source_inverse_affine = torch.linalg.inv(source_affine_ras_from_index)
        except RuntimeError as error:
            raise ValueError("source_affine_ras_from_index must be invertible") from error
        if not bool(torch.isfinite(source_inverse_affine).all()):
            raise ValueError("source_affine_ras_from_index inverse must be finite")
    if config.variant == "p0" or config.rounds == 0:
        return memory, ()
    _preflight_seed_budgets(memory, config)
    current = memory; transactions = []
    for round_index in range(1, config.rounds + 1):
        patient_budget = config.maximum_patient_primitives
        patient_available = None if patient_budget is None else patient_budget - current.primitive_count
        structural_budget = config.maximum_structural_primitives
        volumetric_budget = config.maximum_volumetric_primitives
        if patient_available is not None:
            structural_budget = min(structural_budget, current.structural.gaussians.count + patient_available)
        structural, structural_ids, rejected_s = _propagate_bank(
            current.structural, anchors, config=config, round_index=round_index,
            bounds_min=bounds_min_ras_mm, bounds_max=bounds_max_ras_mm,
            maximum_count=structural_budget, supported_anchor_mask=supported_anchor_mask,
            source_inverse_affine=source_inverse_affine, source_shape_xyz=source_shape_xyz,
        )
        if patient_available is not None:
            remaining = max(0, patient_available - len(structural_ids))
            volumetric_budget = min(volumetric_budget, current.volumetric.gaussians.count + remaining)
        volumetric, volumetric_ids, rejected_v = _propagate_bank(
            current.volumetric, anchors, config=config, round_index=round_index,
            bounds_min=bounds_min_ras_mm, bounds_max=bounds_max_ras_mm,
            maximum_count=volumetric_budget, supported_anchor_mask=supported_anchor_mask,
            source_inverse_affine=source_inverse_affine, source_shape_xyz=source_shape_xyz,
        )
        digest = gaussian_memory_hash(structural, volumetric, current.modality_ids)
        updated = GaussianMemory(structural, volumetric, current.modality_ids, digest)
        counts = rejected_s.merged(rejected_v)
        payload = dict(
            round_index=round_index, parent_memory_hash=current.memory_hash, child_memory_hash=updated.memory_hash,
            accepted_primitive_ids=structural_ids + volumetric_ids,
            proposal_count=counts.proposal_count,
            accepted_count=counts.accepted_count,
            rejected_out_of_bounds=counts.rejected_out_of_bounds,
            rejected_unsupported=counts.rejected_unsupported,
            rejected_duplicate=counts.rejected_duplicate,
            rejected_budget=counts.rejected_budget,
            rejected_uncertainty=counts.rejected_uncertainty,
            rejected_invalid=counts.rejected_invalid,
            rejected_no_meaningful_gain=counts.rejected_no_meaningful_gain,
            config_hash=config.config_hash,
        )
        transactions.append(_transaction(**payload)); current = updated
    return current, tuple(transactions)
