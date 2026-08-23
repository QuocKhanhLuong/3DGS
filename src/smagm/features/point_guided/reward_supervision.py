"""Gate-E E2--E4 measured, compact counterfactual RewardNet supervision.

The inference route remains entirely target-free.  This module is the first
place where a T1ce target is accepted, after a target-free state and point
evidence already exist.  Each hypothetical transition reuses the live
Gate-C query, updater, and 4-mm writeback, but evaluates only a bounded
physical neighbourhood and collateral tri-plane fibres through the existing
Gate-D point decoder.  It never decodes a full volume per candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .contracts import VolumeGeometry
from .decoder import ImplicitTriPlaneDecoder
from .losses import CHARBONNIER_EPSILON, pointwise_charbonnier_by_subject
from .reward import GateBDescriptorContext, build_reward_descriptor
from .sampling import ras_mm_to_voxel_dhw, sample_volume_ras_mm, voxel_dhw_to_ras_mm
from .spectral_query import FeatureGridGeometry
from .state_init import DynamicTriPlanes
from .trajectory import AdaptiveRewardCostTrajectory
from .trajectory_cost import LOCKED_SUPPORT_RADIUS_MM


def _require_int(name: str, value: int, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _require_nonnegative_finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class CounterfactualConfig:
    """Explicit, tunable Gate-E E2--E4 supervision controls.

    ``counterfactual_candidates`` is an upper bound.  The realized subset is
    exactly one selected point plus the configured high-reward and random
    slots when enough candidates exist.  These proportions are intentionally
    configuration, not a scientific claim.
    """

    counterfactual_candidates: int = 32
    high_candidate_count: int = 16
    random_candidate_count: int = 15
    spill_weight_beta: float = 1.0
    spill_sample_count: int = 12
    reward_ranking_weight: float = 0.0
    reward_ranking_min_target_gap: float = 0.001

    def __post_init__(self) -> None:
        maximum = _require_int("counterfactual_candidates", self.counterfactual_candidates, minimum=1)
        # Gate E's normal candidate mixture is deliberately not an implicit
        # ablation: whenever the candidate pool is large enough it contains a
        # selected, high-reward, and seeded-random example.
        high = _require_int("high_candidate_count", self.high_candidate_count, minimum=1)
        random = _require_int("random_candidate_count", self.random_candidate_count, minimum=1)
        if 1 + high + random > maximum:
            raise ValueError(
                "counterfactual_candidates must cover one selected candidate plus configured high/random slots"
            )
        object.__setattr__(self, "counterfactual_candidates", maximum)
        object.__setattr__(self, "high_candidate_count", high)
        object.__setattr__(self, "random_candidate_count", random)
        object.__setattr__(self, "spill_weight_beta", _require_nonnegative_finite("spill_weight_beta", self.spill_weight_beta))
        spill_count = _require_int("spill_sample_count", self.spill_sample_count, minimum=0)
        # Zero is the explicit no-spill ablation used by small synthetic
        # tests. A nonzero measured spill budget must cover all three
        # orthogonal fibres rather than silently privileging XY.
        if spill_count not in (0,) and spill_count < 3:
            raise ValueError("spill_sample_count must be zero or at least 3 to retain XY/XZ/YZ fibres")
        object.__setattr__(self, "spill_sample_count", spill_count)
        object.__setattr__(
            self,
            "reward_ranking_weight",
            _require_nonnegative_finite("reward_ranking_weight", self.reward_ranking_weight),
        )
        object.__setattr__(
            self,
            "reward_ranking_min_target_gap",
            _require_nonnegative_finite("reward_ranking_min_target_gap", self.reward_ranking_min_target_gap),
        )


@dataclass(frozen=True)
class CandidateSubset:
    """Bounded candidate indices with explicit selected/high/random provenance."""

    indices: Tensor  # [B,M], long
    selected_mask: Tensor  # [B,M], bool
    high_reward_mask: Tensor  # [B,M], bool
    random_mask: Tensor  # [B,M], bool

    def __post_init__(self) -> None:
        if not isinstance(self.indices, Tensor) or self.indices.ndim != 2 or self.indices.dtype != torch.long:
            raise ValueError("indices must be a rank-2 torch.long tensor [B,M]")
        if self.indices.shape[0] <= 0 or self.indices.shape[1] <= 0:
            raise ValueError("candidate subset must retain positive batch and candidate dimensions")
        masks = (self.selected_mask, self.high_reward_mask, self.random_mask)
        if any(not isinstance(mask, Tensor) or mask.dtype != torch.bool or mask.shape != self.indices.shape for mask in masks):
            raise ValueError("candidate provenance masks must be bool tensors matching indices")
        if any(mask.device != self.indices.device for mask in masks):
            raise ValueError("candidate provenance masks must share the index device")
        assigned = self.selected_mask.to(torch.int64) + self.high_reward_mask.to(torch.int64) + self.random_mask.to(torch.int64)
        if not bool((assigned == 1).all()):
            raise ValueError("every candidate slot must have exactly one selected/high/random provenance")
        if not bool((self.selected_mask.sum(dim=1) == 1).all()):
            raise ValueError("every batch row must retain exactly one selected candidate")
        if bool((self.indices < 0).any()):
            raise ValueError("candidate indices must be non-negative")
        if any(torch.unique(row).numel() != row.numel() for row in self.indices):
            raise ValueError("candidate subset indices must be unique within every batch row")

    @property
    def candidate_count(self) -> int:
        return int(self.indices.shape[1])


@dataclass(frozen=True)
class PhysicalPointSamples:
    """A compact physical query set with explicit valid-support provenance."""

    points_ras_mm: Tensor  # [B,Q,3]
    valid_mask: Tensor  # [B,Q,1], bool
    fiber_ids: Tensor | None = None  # [B,Q], -1 for padding; 0 XY, 1 XZ, 2 YZ

    def __post_init__(self) -> None:
        if not isinstance(self.points_ras_mm, Tensor) or self.points_ras_mm.ndim != 3 or self.points_ras_mm.shape[-1] != 3:
            raise ValueError("points_ras_mm must be a floating [B,Q,3] tensor")
        if not self.points_ras_mm.is_floating_point() or self.points_ras_mm.shape[0] <= 0 or self.points_ras_mm.shape[1] <= 0:
            raise ValueError("points_ras_mm must have positive floating [B,Q,3] shape")
        if not bool(torch.isfinite(self.points_ras_mm).all()):
            raise ValueError("points_ras_mm must be finite")
        if not isinstance(self.valid_mask, Tensor) or self.valid_mask.dtype != torch.bool or self.valid_mask.shape != (*self.points_ras_mm.shape[:2], 1):
            raise ValueError("valid_mask must be bool [B,Q,1] aligned with points_ras_mm")
        if self.valid_mask.device != self.points_ras_mm.device:
            raise ValueError("valid_mask must share the points device")
        if self.fiber_ids is not None:
            if not isinstance(self.fiber_ids, Tensor) or self.fiber_ids.dtype != torch.long or self.fiber_ids.shape != self.points_ras_mm.shape[:2]:
                raise ValueError("fiber_ids must be a [B,Q] torch.long tensor")
            if self.fiber_ids.device != self.points_ras_mm.device:
                raise ValueError("fiber_ids must share the points device")
            if bool(((self.fiber_ids < -1) | (self.fiber_ids > 2)).any()):
                raise ValueError("fiber_ids may contain only -1, 0 (XY), 1 (XZ), or 2 (YZ)")


@dataclass(frozen=True)
class CounterfactualRewardResult:
    """Detached measured reward target and live RewardNet regression result."""

    reward_prediction: Tensor  # [B,M], live RewardNet output
    reward_target: Tensor  # [B,M], detached [0,1]
    valid_mask: Tensor  # [B,M], bool local-supervision availability
    loss: Tensor  # scalar absolute loss + weighted pairwise ranking loss
    absolute_loss: Tensor  # scalar masked SmoothL1 mean over valid entries
    ranking_loss: Tensor  # scalar mean over informative within-subject pairs
    ranking_weighted_loss: Tensor  # reward_ranking_weight * ranking_loss
    valid_pair_count: int  # unique valid candidate pairs before gap filtering
    informative_pair_count: int  # unique valid pairs meeting the target-gap threshold
    ranking_violation_fraction: float  # fraction of informative pairs with positive hinge
    mean_target_pair_gap: float  # mean target gap over informative pairs
    candidates: CandidateSubset
    local_before: Tensor  # [B,M]
    local_after: Tensor  # [B,M]
    spill_before: Tensor  # [B,M]
    spill_after: Tensor  # [B,M]

    def __post_init__(self) -> None:
        reference = self.reward_prediction
        if not isinstance(reference, Tensor) or reference.ndim != 2 or not reference.is_floating_point():
            raise ValueError("reward_prediction must be a floating [B,M] tensor")
        if not bool(torch.isfinite(reference).all()):
            raise ValueError("reward_prediction must be finite")
        if reference.shape != self.candidates.indices.shape:
            raise ValueError("reward_prediction must align with sampled candidates")
        for name in ("reward_target", "local_before", "local_after", "spill_before", "spill_after"):
            value = getattr(self, name)
            if not isinstance(value, Tensor) or value.shape != reference.shape or value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError(f"{name} must align with reward_prediction")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
        if self.reward_target.requires_grad:
            raise ValueError("measured reward_target must be detached")
        if bool(((self.reward_target < 0.0) | (self.reward_target > 1.0)).any()):
            raise ValueError("measured reward_target must be clamped to [0,1]")
        if not isinstance(self.valid_mask, Tensor) or self.valid_mask.dtype != torch.bool or self.valid_mask.shape != reference.shape or self.valid_mask.device != reference.device:
            raise ValueError("valid_mask must be a bool [B,M] tensor aligned with reward_prediction")
        for name in ("loss", "absolute_loss", "ranking_loss", "ranking_weighted_loss"):
            value = getattr(self, name)
            if not isinstance(value, Tensor) or value.ndim != 0 or value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError(f"{name} must be a scalar tensor aligned with reward_prediction")
            if not bool(torch.isfinite(value)):
                raise ValueError(f"{name} must be finite")
        for name in ("valid_pair_count", "informative_pair_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.informative_pair_count > self.valid_pair_count:
            raise ValueError("informative_pair_count cannot exceed valid_pair_count")
        if not isinstance(self.ranking_violation_fraction, float) or not math.isfinite(self.ranking_violation_fraction) or not 0.0 <= self.ranking_violation_fraction <= 1.0:
            raise ValueError("ranking_violation_fraction must be a finite fraction")
        if not isinstance(self.mean_target_pair_gap, float) or not math.isfinite(self.mean_target_pair_gap) or self.mean_target_pair_gap < 0.0:
            raise ValueError("mean_target_pair_gap must be finite and non-negative")

    @property
    def valid_count(self) -> int:
        return int(self.valid_mask.sum().detach().cpu())


@dataclass(frozen=True)
class RewardRankingResult:
    """Within-subject pairwise ranking evidence for RewardNet supervision."""

    loss: Tensor
    valid_pair_count: int
    informative_pair_count: int
    violation_count: int
    violation_fraction: float
    mean_target_pair_gap: float

    def __post_init__(self) -> None:
        if not isinstance(self.loss, Tensor) or self.loss.ndim != 0 or not self.loss.is_floating_point() or not bool(torch.isfinite(self.loss)):
            raise ValueError("ranking loss must be one finite scalar tensor")
        for name in ("valid_pair_count", "informative_pair_count", "violation_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.informative_pair_count > self.valid_pair_count or self.violation_count > self.informative_pair_count:
            raise ValueError("ranking pair counts are inconsistent")
        if not isinstance(self.violation_fraction, float) or not math.isfinite(self.violation_fraction) or not 0.0 <= self.violation_fraction <= 1.0:
            raise ValueError("violation_fraction must be a finite fraction")
        if not isinstance(self.mean_target_pair_gap, float) or not math.isfinite(self.mean_target_pair_gap) or self.mean_target_pair_gap < 0.0:
            raise ValueError("mean_target_pair_gap must be finite and non-negative")


def pairwise_reward_ranking_loss(
    reward_prediction: Tensor,
    reward_target: Tensor,
    valid_mask: Tensor,
    *,
    min_target_gap: float,
) -> RewardRankingResult:
    """Compute a differentiable within-subject pairwise reward ranking hinge.

    Candidate rows are subjects and candidate columns are slots.  Only unique
    ``i < j`` pairs within each row participate; measured targets remain
    detached and the prediction path remains differentiable.
    """

    if not isinstance(reward_prediction, Tensor) or reward_prediction.ndim != 2 or not reward_prediction.is_floating_point():
        raise ValueError("reward_prediction must be a floating [B,M] tensor")
    if not isinstance(reward_target, Tensor) or reward_target.shape != reward_prediction.shape or reward_target.dtype != reward_prediction.dtype or reward_target.device != reward_prediction.device or not reward_target.is_floating_point():
        raise ValueError("reward_target must match reward_prediction")
    if reward_target.requires_grad:
        raise ValueError("reward_target must be detached")
    if not isinstance(valid_mask, Tensor) or valid_mask.dtype != torch.bool or valid_mask.shape != reward_prediction.shape or valid_mask.device != reward_prediction.device:
        raise ValueError("valid_mask must be a bool tensor matching reward_prediction")
    if not bool(torch.isfinite(reward_prediction).all()) or not bool(torch.isfinite(reward_target).all()):
        raise ValueError("reward prediction and target must be finite")
    gap_threshold = _require_nonnegative_finite("reward_ranking_min_target_gap", min_target_gap)

    candidate_count = reward_prediction.shape[1]
    upper_triangle = torch.triu(
        torch.ones((candidate_count, candidate_count), dtype=torch.bool, device=reward_prediction.device),
        diagonal=1,
    )
    valid_pairs = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1) & upper_triangle.unsqueeze(0)
    delta_target = reward_target.unsqueeze(2) - reward_target.unsqueeze(1)
    target_gap = delta_target.abs()
    informative_pairs = valid_pairs & (target_gap >= gap_threshold) & (target_gap > 0.0)
    valid_pair_count = int(valid_pairs.sum().detach().cpu())
    informative_pair_count = int(informative_pairs.sum().detach().cpu())

    delta_prediction = reward_prediction.unsqueeze(2) - reward_prediction.unsqueeze(1)
    direction = torch.sign(delta_target)
    pair_loss = torch.relu(target_gap - direction * delta_prediction)
    if informative_pair_count:
        selected_loss = pair_loss[informative_pairs]
        loss = selected_loss.mean()
        violation_count = int((selected_loss > 0.0).sum().detach().cpu())
        violation_fraction = float(violation_count / informative_pair_count)
        mean_target_pair_gap = float(target_gap[informative_pairs].mean().detach().cpu())
    else:
        loss = reward_prediction.sum() * 0.0
        violation_count = 0
        violation_fraction = 0.0
        mean_target_pair_gap = 0.0
    return RewardRankingResult(
        loss=loss,
        valid_pair_count=valid_pair_count,
        informative_pair_count=informative_pair_count,
        violation_count=violation_count,
        violation_fraction=violation_fraction,
        mean_target_pair_gap=mean_target_pair_gap,
    )


def _randperm(length: int, *, generator: torch.Generator | None, device: torch.device) -> Tensor:
    if length <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.randperm(length, generator=generator, device=device)


def sample_counterfactual_candidates(
    predicted_reward: Tensor,
    selected_indices: Tensor,
    config: CounterfactualConfig,
    *,
    generator: torch.Generator | None = None,
) -> CandidateSubset:
    """Choose selected + high-reward + random candidates without duplicates.

    Ranking is based only on target-free predicted reward.  Random choice is
    explicit through ``torch.Generator`` so training experiments can own their
    seed policy later without this module silently touching Python RNG state.
    """

    if not isinstance(config, CounterfactualConfig):
        raise TypeError("config must be a CounterfactualConfig")
    if not isinstance(predicted_reward, Tensor) or predicted_reward.ndim != 2 or not predicted_reward.is_floating_point():
        raise ValueError("predicted_reward must be a floating [B,N] tensor")
    if predicted_reward.shape[0] <= 0 or predicted_reward.shape[1] <= 0 or not bool(torch.isfinite(predicted_reward).all()):
        raise ValueError("predicted_reward must have positive finite [B,N] shape")
    if not isinstance(selected_indices, Tensor) or selected_indices.dtype != torch.long or selected_indices.shape != predicted_reward.shape[:1] or selected_indices.device != predicted_reward.device:
        raise ValueError("selected_indices must be device-matched torch.long [B]")
    point_count = predicted_reward.shape[1]
    if bool((selected_indices < 0).any()) or bool((selected_indices >= point_count).any()):
        raise ValueError("selected_indices must contain valid active candidate indices")

    desired = min(point_count, 1 + config.high_candidate_count + config.random_candidate_count, config.counterfactual_candidates)
    rows: list[Tensor] = []
    selected_masks: list[Tensor] = []
    high_masks: list[Tensor] = []
    random_masks: list[Tensor] = []
    all_indices = torch.arange(point_count, dtype=torch.long, device=predicted_reward.device)
    for batch_index in range(predicted_reward.shape[0]):
        selected = selected_indices[batch_index : batch_index + 1]
        selected_bool = torch.zeros(point_count, dtype=torch.bool, device=predicted_reward.device)
        selected_bool[selected] = True
        available = all_indices[~selected_bool]
        high_count = min(config.high_candidate_count, desired - 1, available.numel())
        ranked = available[torch.argsort(predicted_reward[batch_index, available], descending=True)]
        high = ranked[:high_count]
        high_bool = torch.zeros_like(selected_bool)
        high_bool[high] = True
        remaining = all_indices[~(selected_bool | high_bool)]
        random_count = min(config.random_candidate_count, desired - 1 - high_count, remaining.numel())
        random = remaining[_randperm(remaining.numel(), generator=generator, device=remaining.device)[:random_count]]
        random_bool = torch.zeros_like(selected_bool)
        random_bool[random] = True
        row = torch.cat((selected, high, random), dim=0)
        if row.numel() != desired:
            # If a tiny N truncates configured high/random counts, complete
            # only from the still-unseen target-free candidate set.
            remainder = all_indices[~(selected_bool | high_bool | random_bool)]
            needed = desired - row.numel()
            extra = remainder[_randperm(remainder.numel(), generator=generator, device=remainder.device)[:needed]]
            row = torch.cat((row, extra), dim=0)
            random_bool[extra] = True
        rows.append(row)
        selected_masks.append(selected_bool[row])
        high_masks.append(high_bool[row])
        random_masks.append(random_bool[row])
    return CandidateSubset(
        indices=torch.stack(rows),
        selected_mask=torch.stack(selected_masks),
        high_reward_mask=torch.stack(high_masks),
        random_mask=torch.stack(random_masks),
    )


def _minimum_spacing_mm(geometry: VolumeGeometry) -> float:
    matrix = torch.as_tensor([row[:3] for row in geometry.voxel_to_ras_mm[:3]], dtype=torch.float64)
    singular_values = torch.linalg.svdvals(matrix)
    minimum = float(singular_values[-1].item())
    if not math.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("output geometry affine must define a non-singular physical metric")
    return minimum


def _validate_centres(points_ras_mm: Tensor, geometry: VolumeGeometry) -> None:
    if not isinstance(points_ras_mm, Tensor) or points_ras_mm.ndim != 2 or points_ras_mm.shape[-1] != 3:
        raise ValueError("points_ras_mm must be a [B,3] tensor")
    if not points_ras_mm.is_floating_point() or points_ras_mm.shape[0] <= 0 or not bool(torch.isfinite(points_ras_mm).all()):
        raise ValueError("points_ras_mm must have positive batch size and finite floating values")
    if not isinstance(geometry, VolumeGeometry):
        raise TypeError("geometry must be a VolumeGeometry")


def build_local_support_samples(
    points_ras_mm: Tensor,
    geometry: VolumeGeometry,
    *,
    radius_mm: float = LOCKED_SUPPORT_RADIUS_MM,
) -> PhysicalPointSamples:
    """Enumerate source voxel centres within the locked physical 4-mm sphere."""

    _validate_centres(points_ras_mm, geometry)
    radius = float(radius_mm)
    if not math.isfinite(radius) or radius != LOCKED_SUPPORT_RADIUS_MM:
        raise ValueError("radius_mm must be exactly the locked 4.0-mm support")
    minimum_spacing = _minimum_spacing_mm(geometry)
    # The extra cell covers fractional point centres while the full affine
    # distance check below is the authoritative physical support predicate.
    index_radius = int(math.ceil(radius / minimum_spacing)) + 1
    offsets = torch.arange(-index_radius, index_radius + 1, dtype=points_ras_mm.dtype, device=points_ras_mm.device)
    d, h, w = torch.meshgrid(offsets, offsets, offsets, indexing="ij")
    offset_dhw = torch.stack((d, h, w), dim=-1).reshape(1, -1, 3)
    source_dhw = ras_mm_to_voxel_dhw(points_ras_mm, geometry)
    base_dhw = torch.floor(source_dhw).detach().unsqueeze(1)
    voxel_dhw = base_dhw + offset_dhw
    upper = points_ras_mm.new_tensor(tuple(length - 1 for length in geometry.shape_dhw)).view(1, 1, 3)
    in_bounds = ((voxel_dhw >= 0.0) & (voxel_dhw <= upper)).all(dim=-1)
    support_points = voxel_dhw_to_ras_mm(voxel_dhw, geometry)
    distance = torch.linalg.vector_norm(support_points - points_ras_mm.unsqueeze(1), dim=-1)
    valid = in_bounds & (distance <= radius + 1e-6)
    return PhysicalPointSamples(points_ras_mm=support_points, valid_mask=valid.unsqueeze(-1))


def build_spill_samples(
    points_ras_mm: Tensor,
    geometry: VolumeGeometry,
    *,
    sample_count: int,
    radius_mm: float = LOCKED_SUPPORT_RADIUS_MM,
    generator: torch.Generator | None = None,
) -> PhysicalPointSamples:
    """Sample nonlocal same-XY / same-XZ / same-YZ physical fibres.

    A valid available fibre receives one sample first; remaining slots are
    randomized from all unchosen collateral positions.  Coordinates inside
    the true 4-mm sphere are excluded before sampling.
    """

    _validate_centres(points_ras_mm, geometry)
    sample_count = _require_int("sample_count", sample_count, minimum=1)
    radius = float(radius_mm)
    if not math.isfinite(radius) or radius != LOCKED_SUPPORT_RADIUS_MM:
        raise ValueError("radius_mm must be exactly the locked 4.0-mm support")
    batch = points_ras_mm.shape[0]
    depth, height, width = geometry.shape_dhw
    source_dhw = ras_mm_to_voxel_dhw(points_ras_mm, geometry)
    dtype, device = points_ras_mm.dtype, points_ras_mm.device
    d_values = torch.arange(depth, dtype=dtype, device=device).view(1, -1).expand(batch, -1)
    h_values = torch.arange(height, dtype=dtype, device=device).view(1, -1).expand(batch, -1)
    w_values = torch.arange(width, dtype=dtype, device=device).view(1, -1).expand(batch, -1)
    xy = torch.stack((d_values, source_dhw[:, 1:2].expand(-1, depth), source_dhw[:, 2:3].expand(-1, depth)), dim=-1)
    xz = torch.stack((source_dhw[:, 0:1].expand(-1, height), h_values, source_dhw[:, 2:3].expand(-1, height)), dim=-1)
    yz = torch.stack((source_dhw[:, 0:1].expand(-1, width), source_dhw[:, 1:2].expand(-1, width), w_values), dim=-1)
    voxel_dhw = torch.cat((xy, xz, yz), dim=1)
    fiber_ids = torch.cat(
        (
            torch.zeros(depth, dtype=torch.long, device=device),
            torch.ones(height, dtype=torch.long, device=device),
            torch.full((width,), 2, dtype=torch.long, device=device),
        )
    ).unsqueeze(0).expand(batch, -1)
    sample_points = voxel_dhw_to_ras_mm(voxel_dhw, geometry)
    upper = points_ras_mm.new_tensor(tuple(length - 1 for length in geometry.shape_dhw)).view(1, 1, 3)
    in_bounds = ((voxel_dhw >= 0.0) & (voxel_dhw <= upper)).all(dim=-1)
    distance = torch.linalg.vector_norm(sample_points - points_ras_mm.unsqueeze(1), dim=-1)
    eligible = in_bounds & (distance > radius + 1e-6)

    chosen_points = points_ras_mm.unsqueeze(1).expand(-1, sample_count, -1).clone()
    chosen_valid = torch.zeros(batch, sample_count, dtype=torch.bool, device=device)
    chosen_fibers = torch.full((batch, sample_count), -1, dtype=torch.long, device=device)
    for batch_index in range(batch):
        selected: list[int] = []
        for fiber in range(3):
            choices = torch.nonzero(eligible[batch_index] & (fiber_ids[batch_index] == fiber), as_tuple=False).squeeze(1)
            if choices.numel() > 0 and len(selected) < sample_count:
                picked = choices[_randperm(choices.numel(), generator=generator, device=device)[0]].item()
                selected.append(int(picked))
        if len(selected) < sample_count:
            already = torch.zeros(eligible.shape[1], dtype=torch.bool, device=device)
            if selected:
                already[torch.as_tensor(selected, dtype=torch.long, device=device)] = True
            choices = torch.nonzero(eligible[batch_index] & ~already, as_tuple=False).squeeze(1)
            take = min(sample_count - len(selected), choices.numel())
            if take:
                selected.extend(choices[_randperm(choices.numel(), generator=generator, device=device)[:take]].tolist())
        if selected:
            selected_tensor = torch.as_tensor(selected, dtype=torch.long, device=device)
            count = selected_tensor.numel()
            chosen_points[batch_index, :count] = sample_points[batch_index, selected_tensor]
            chosen_valid[batch_index, :count] = True
            chosen_fibers[batch_index, :count] = fiber_ids[batch_index, selected_tensor]
    return PhysicalPointSamples(
        points_ras_mm=chosen_points,
        valid_mask=chosen_valid.unsqueeze(-1),
        fiber_ids=chosen_fibers,
    )


def _validate_target_and_mask(
    target: Tensor,
    valid_mask: Tensor | None,
    *,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
    geometry: VolumeGeometry,
) -> tuple[Tensor, Tensor]:
    expected = (batch, 1, *geometry.shape_dhw)
    if not isinstance(target, Tensor) or target.shape != expected or not target.is_floating_point():
        raise ValueError("target must be a floating [B,1,D,H,W] tensor matching the output geometry")
    if target.dtype != dtype or target.device != device:
        raise ValueError("target must match dynamic state dtype and device")
    if valid_mask is None:
        valid_mask = torch.ones_like(target, dtype=torch.bool)
    if not isinstance(valid_mask, Tensor) or valid_mask.dtype != torch.bool or valid_mask.shape != target.shape or valid_mask.device != device:
        raise ValueError("valid_mask must be a device-matched bool tensor matching target")
    if bool((valid_mask.flatten(1).sum(dim=1) <= 0).any()):
        raise ValueError("valid_mask must retain at least one target voxel per subject")
    if not bool(torch.isfinite(target[valid_mask]).all()):
        raise ValueError("target must be finite over valid_mask support")
    # T1ce is intentionally severed here.  Its values are used only as an
    # immutable evaluation signal and invalid locations cannot contaminate
    # interpolation or local metrics.
    return torch.where(valid_mask, target.detach(), torch.zeros_like(target)), valid_mask


def _sample_target_support(
    target: Tensor,
    valid_mask: Tensor,
    samples: PhysicalPointSamples,
    geometry: VolumeGeometry,
) -> tuple[Tensor, Tensor]:
    sampled_target = sample_volume_ras_mm(target, samples.points_ras_mm, geometry)
    sampled_mask = sample_volume_ras_mm(valid_mask.to(dtype=target.dtype), samples.points_ras_mm, geometry)
    valid = samples.valid_mask & (sampled_mask >= 1.0 - 1e-6)
    return sampled_target, valid


def spill_aware_reward_target(
    local_before: Tensor,
    local_after: Tensor,
    spill_before: Tensor,
    spill_after: Tensor,
    *,
    spill_weight_beta: float,
) -> Tensor:
    """Return detached ``clip((Δlocal - beta * relu(Δspill))/ (before+eps))``."""

    beta = _require_nonnegative_finite("spill_weight_beta", spill_weight_beta)
    reference = local_before
    if not isinstance(reference, Tensor) or not reference.is_floating_point():
        raise ValueError("local_before must be a floating tensor")
    for name, value in (("local_after", local_after), ("spill_before", spill_before), ("spill_after", spill_after)):
        if not isinstance(value, Tensor) or value.shape != reference.shape or value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError(f"{name} must align with local_before")
    if not all(bool(torch.isfinite(value).all()) for value in (local_before, local_after, spill_before, spill_after)):
        raise ValueError("all measured reconstruction errors must be finite")
    local_gain = local_before - local_after
    spill_degradation = torch.relu(spill_after - spill_before)
    result = torch.clamp(
        (local_gain - beta * spill_degradation) / (local_before + reference.new_tensor(CHARBONNIER_EPSILON)),
        min=0.0,
        max=1.0,
    ).detach()
    return result


def _gather(values: Tensor, indices: Tensor) -> Tensor:
    return values[torch.arange(values.shape[0], device=values.device), indices]


def _counterfactual_transition(
    trajectory: AdaptiveRewardCostTrajectory,
    state: DynamicTriPlanes,
    candidate_points_ras_mm: Tensor,
    candidate_semantic: Tensor,
    candidate_f_spec: Tensor,
    candidate_reliability: Tensor,
    feature_geometry: FeatureGridGeometry,
) -> DynamicTriPlanes:
    samples = trajectory.dynamic_state_query(state, candidate_points_ras_mm.unsqueeze(1), feature_geometry)
    updater_input = torch.cat(
        (
            samples.packed[:, 0],
            candidate_f_spec,
            candidate_semantic,
            candidate_reliability,
        ),
        dim=-1,
    )
    corrections = trajectory.update_net(updater_input, write_scale=trajectory.config.write_scale)
    return trajectory.writeback(state, candidate_points_ras_mm, corrections, feature_geometry)


def counterfactual_reward_supervision(
    trajectory: AdaptiveRewardCostTrajectory,
    decoder: ImplicitTriPlaneDecoder,
    state: DynamicTriPlanes,
    refined_points_ras_mm: Tensor,
    point_semantic: Tensor,
    f_spec: Tensor,
    reliability: Tensor,
    gate_b_descriptors: GateBDescriptorContext,
    feature_geometry: FeatureGridGeometry,
    output_geometry: VolumeGeometry,
    target: Tensor,
    *,
    selected_indices: Tensor,
    valid_mask: Tensor | None = None,
    config: CounterfactualConfig | None = None,
    generator: torch.Generator | None = None,
) -> CounterfactualRewardResult:
    """Measure bounded spill-aware gains and regress the shared RewardNet.

    The live reward prediction is deliberately evaluated from a detached
    target-free descriptor: ``L_reward`` directly trains RewardNet without
    making an accidental supervision route into dynamic state, UpdateNet, or
    decoder.  Measured target transitions run under ``no_grad`` and reuse the
    exact Gate-C update/write contract.
    """

    if not isinstance(trajectory, AdaptiveRewardCostTrajectory):
        raise TypeError("trajectory must be an AdaptiveRewardCostTrajectory")
    if not isinstance(decoder, ImplicitTriPlaneDecoder):
        raise TypeError("decoder must be an ImplicitTriPlaneDecoder")
    if not isinstance(state, DynamicTriPlanes) or not isinstance(feature_geometry, FeatureGridGeometry) or not isinstance(output_geometry, VolumeGeometry):
        raise TypeError("state and geometries must use typed Gate-C/D contracts")
    if output_geometry != feature_geometry.source_geometry:
        raise ValueError("output_geometry must equal the source geometry paired with feature_geometry")
    config = config or CounterfactualConfig()
    if not isinstance(config, CounterfactualConfig):
        raise TypeError("config must be a CounterfactualConfig")
    reference = refined_points_ras_mm
    expected_batch = state.xy.shape[0]
    if not isinstance(reference, Tensor) or reference.ndim != 3 or reference.shape[0] != expected_batch or reference.shape[-1] != 3:
        raise ValueError("refined_points_ras_mm must be [B,N,3] aligned with state")
    if not reference.is_floating_point() or reference.dtype != state.xy.dtype or reference.device != state.xy.device or reference.shape[1] <= 0 or not bool(torch.isfinite(reference).all()):
        raise ValueError("refined_points_ras_mm must be finite and match state dtype/device")
    for name, value, width in (("point_semantic", point_semantic, 3), ("f_spec", f_spec, 168), ("reliability", reliability, 3)):
        if not isinstance(value, Tensor) or value.shape != (*reference.shape[:2], width) or value.dtype != reference.dtype or value.device != reference.device or not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must align with refined points and its locked width")
    if not isinstance(gate_b_descriptors, GateBDescriptorContext):
        raise TypeError("gate_b_descriptors must be a GateBDescriptorContext")
    if gate_b_descriptors.q_xy.shape[:2] != reference.shape[:2] or gate_b_descriptors.q_xy.dtype != reference.dtype or gate_b_descriptors.q_xy.device != reference.device:
        raise ValueError("gate_b_descriptors must align with refined points")
    if not isinstance(selected_indices, Tensor) or selected_indices.dtype != torch.long or selected_indices.shape != (expected_batch,) or selected_indices.device != reference.device:
        raise ValueError("selected_indices must be device-matched torch.long [B]")
    target, valid_mask = _validate_target_and_mask(
        target,
        valid_mask,
        batch=expected_batch,
        dtype=reference.dtype,
        device=reference.device,
        geometry=output_geometry,
    )

    dynamic_samples = trajectory.dynamic_state_query(state, reference, feature_geometry)
    descriptor = build_reward_descriptor(dynamic_samples, point_semantic, gate_b_descriptors, reliability)
    all_reward_prediction = trajectory.reward_net(descriptor.detach())
    candidates = sample_counterfactual_candidates(all_reward_prediction.detach(), selected_indices, config, generator=generator)
    reward_prediction = all_reward_prediction.gather(1, candidates.indices)
    shape = reward_prediction.shape
    local_before = torch.zeros_like(reward_prediction)
    local_after = torch.zeros_like(reward_prediction)
    spill_before = torch.zeros_like(reward_prediction)
    spill_after = torch.zeros_like(reward_prediction)
    local_available = torch.zeros(shape, dtype=torch.bool, device=reference.device)

    with torch.no_grad():
        for slot in range(candidates.candidate_count):
            indices = candidates.indices[:, slot]
            points = _gather(reference, indices)
            after_state = _counterfactual_transition(
                trajectory,
                state,
                points,
                _gather(point_semantic, indices),
                _gather(f_spec, indices),
                _gather(reliability, indices),
                feature_geometry,
            )
            local_samples = build_local_support_samples(points, output_geometry)
            local_target, local_valid = _sample_target_support(target, valid_mask, local_samples, output_geometry)
            before_local = pointwise_charbonnier_by_subject(
                decoder.decode_points(state, local_samples.points_ras_mm, feature_geometry),
                local_target,
                local_valid,
                allow_empty=True,
            )
            after_local = pointwise_charbonnier_by_subject(
                decoder.decode_points(after_state, local_samples.points_ras_mm, feature_geometry),
                local_target,
                local_valid,
                allow_empty=True,
            )
            local_before[:, slot] = before_local
            local_after[:, slot] = after_local
            local_available[:, slot] = local_valid.squeeze(-1).any(dim=1)
            if config.spill_sample_count > 0:
                spill_samples = build_spill_samples(
                    points,
                    output_geometry,
                    sample_count=config.spill_sample_count,
                    generator=generator,
                )
                spill_target, spill_valid = _sample_target_support(target, valid_mask, spill_samples, output_geometry)
                spill_before[:, slot] = pointwise_charbonnier_by_subject(
                    decoder.decode_points(state, spill_samples.points_ras_mm, feature_geometry),
                    spill_target,
                    spill_valid,
                    allow_empty=True,
                )
                spill_after[:, slot] = pointwise_charbonnier_by_subject(
                    decoder.decode_points(after_state, spill_samples.points_ras_mm, feature_geometry),
                    spill_target,
                    spill_valid,
                    allow_empty=True,
                )

    reward_target = spill_aware_reward_target(
        local_before,
        local_after,
        spill_before,
        spill_after,
        spill_weight_beta=config.spill_weight_beta,
    )
    unreduced = F.smooth_l1_loss(reward_prediction, reward_target, reduction="none")
    valid_count = local_available.sum()
    absolute_loss = torch.where(
        valid_count > 0,
        torch.where(local_available, unreduced, torch.zeros_like(unreduced)).sum() / valid_count.to(dtype=unreduced.dtype),
        reward_prediction.sum() * 0.0,
    )
    ranking = pairwise_reward_ranking_loss(
        reward_prediction,
        reward_target,
        local_available,
        min_target_gap=config.reward_ranking_min_target_gap,
    )
    ranking_weighted_loss = config.reward_ranking_weight * ranking.loss
    loss = absolute_loss + ranking_weighted_loss
    return CounterfactualRewardResult(
        reward_prediction=reward_prediction,
        reward_target=reward_target,
        valid_mask=local_available,
        loss=loss,
        absolute_loss=absolute_loss,
        ranking_loss=ranking.loss,
        ranking_weighted_loss=ranking_weighted_loss,
        valid_pair_count=ranking.valid_pair_count,
        informative_pair_count=ranking.informative_pair_count,
        ranking_violation_fraction=ranking.violation_fraction,
        mean_target_pair_gap=ranking.mean_target_pair_gap,
        candidates=candidates,
        local_before=local_before,
        local_after=local_after,
        spill_before=spill_before,
        spill_after=spill_after,
    )


__all__ = [
    "CandidateSubset",
    "CounterfactualConfig",
    "CounterfactualRewardResult",
    "PhysicalPointSamples",
    "RewardRankingResult",
    "build_local_support_samples",
    "build_spill_samples",
    "counterfactual_reward_supervision",
    "pairwise_reward_ranking_loss",
    "sample_counterfactual_candidates",
    "spill_aware_reward_target",
]
