"""Training-only coarse semantic supervision for BraTS label maps.

This module is deliberately separate from the point-guided model and is not
used by any inference API.  It converts the BraTS labels ``{0, 1, 2, 4}``
into the locked three-class coarse-semantic contract and computes a
target-only cross-entropy grounding objective with diagnostic Dice metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .contracts import NUM_COARSE_SEMANTIC_CLASSES


BRATS_LABEL_VALUES: tuple[int, ...] = (0, 1, 2, 4)
"""The only accepted BraTS segmentation labels for this adapter."""

SEMANTIC_CLASS_COUNT = NUM_COARSE_SEMANTIC_CLASSES
"""The locked number of coarse semantic classes (normal, edema, core)."""


def _validate_ignore_index(ignore_index: int) -> int:
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, Integral):
        raise ValueError("ignore_index must be an integer distinct from semantic classes 0, 1, and 2")
    value = int(ignore_index)
    if value in range(SEMANTIC_CLASS_COUNT):
        raise ValueError("ignore_index must be distinct from semantic classes 0, 1, and 2")
    if value < -(2**63) or value > 2**63 - 1:
        raise ValueError("ignore_index must fit in int64")
    return value


def _validate_volume_shape(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim not in (4, 5):
        raise ValueError(f"{name} must have shape [B,D,H,W] or [B,1,D,H,W]")
    if value.ndim == 5 and value.shape[1] != 1:
        raise ValueError(f"{name} must have a singleton channel when rank 5")
    if value.shape[0] <= 0 or any(length <= 0 for length in value.shape[-3:]):
        raise ValueError(f"{name} must have positive batch and spatial dimensions")
    return value[:, 0] if value.ndim == 5 else value


def _validate_segmentation(segmentation: Tensor) -> Tensor:
    segmentation = _validate_volume_shape(segmentation, name="segmentation")
    if segmentation.dtype is torch.bool or segmentation.is_complex():
        raise ValueError("segmentation labels must be a real integer-valued tensor")
    if segmentation.is_floating_point():
        if not bool(torch.isfinite(segmentation).all()):
            raise ValueError("segmentation labels must be finite")
        if not bool(torch.equal(segmentation, segmentation.round())):
            raise ValueError("segmentation labels must be integer-valued")
    labels = segmentation.to(dtype=torch.long)
    allowed = torch.zeros_like(labels, dtype=torch.bool)
    for label in BRATS_LABEL_VALUES:
        allowed |= labels == label
    if not bool(allowed.all()):
        invalid = labels[~allowed]
        example = int(invalid.flatten()[0].detach().cpu())
        raise ValueError(
            f"segmentation labels must be drawn from {BRATS_LABEL_VALUES}; found {example}"
        )
    return labels


def _validate_brain_mask(brain_mask: Tensor) -> Tensor:
    brain_mask = _validate_volume_shape(brain_mask, name="brain_mask")
    if brain_mask.dtype is not torch.bool:
        raise ValueError("brain_mask must be a boolean tensor")
    # A boolean mask has no non-finite values; keeping this invariant explicit
    # makes the accepted data boundary clear without accepting numeric masks.
    if not bool(torch.isfinite(brain_mask).all()):
        raise ValueError("brain_mask must be finite")
    return brain_mask


def build_coarse_semantic_target(
    segmentation: Tensor,
    brain_mask: Tensor,
    *,
    ignore_index: int = -100,
) -> Tensor:
    """Map BraTS labels to the three-class coarse-semantic target.

    The returned tensor has shape ``[B,D,H,W]`` and dtype ``torch.long``.
    Within ``brain_mask``, label ``0`` maps to class ``0`` (normal brain),
    label ``2`` maps to class ``1`` (edema), and labels ``1`` and ``4`` map to
    class ``2`` (tumor-core candidate).  Every voxel outside the mask receives
    ``ignore_index`` regardless of its segmentation label.

    ``segmentation`` and ``brain_mask`` may each be supplied as either
    ``[B,D,H,W]`` or ``[B,1,D,H,W]``.  A singleton channel is removed from the
    result so it is directly consumable by ``torch.nn.functional.cross_entropy``.
    This function is a training/data-boundary utility; it is not called by
    model inference.
    """

    ignore_index = _validate_ignore_index(ignore_index)
    labels = _validate_segmentation(segmentation)
    mask = _validate_brain_mask(brain_mask)
    if labels.shape != mask.shape:
        raise ValueError("segmentation and brain_mask must have matching [B,D,H,W] shapes")
    if labels.device != mask.device:
        raise ValueError("segmentation and brain_mask must be on the same device")

    target = torch.full_like(labels, fill_value=ignore_index)
    valid = mask
    target[valid & (labels == 0)] = 0
    target[valid & (labels == 2)] = 1
    target[valid & ((labels == 1) | (labels == 4))] = 2
    return target


def _validate_target(target: Tensor, *, spatial_shape: Sequence[int], device: torch.device) -> Tensor:
    target = _validate_volume_shape(target, name="target")
    if tuple(target.shape) != tuple(spatial_shape):
        raise ValueError("target must match semantic_logits spatial shape [B,D,H,W]")
    if target.device != device:
        raise ValueError("target must be on the same device as semantic_logits")
    if target.dtype is torch.bool or target.is_complex():
        raise ValueError("target must be an integer-valued tensor")
    if target.is_floating_point():
        if not bool(torch.isfinite(target).all()):
            raise ValueError("target labels must be finite")
        if not bool(torch.equal(target, target.round())):
            raise ValueError("target labels must be integer-valued")
    return target.to(dtype=torch.long)


def _validate_class_weights(
    class_weights: Tensor | Sequence[float] | None,
    *,
    logits: Tensor,
) -> Tensor | None:
    if class_weights is None:
        return None
    if isinstance(class_weights, Tensor):
        weights = class_weights
    else:
        try:
            weights = torch.as_tensor(class_weights)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("class_weights must be a finite length-3 numeric sequence") from error
    if weights.ndim != 1 or weights.numel() != SEMANTIC_CLASS_COUNT:
        raise ValueError("class_weights must have shape [3]")
    if weights.dtype is torch.bool or weights.is_complex():
        raise ValueError("class_weights must be real numeric values")
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("class_weights must be finite")
    if bool((weights < 0).any()):
        raise ValueError("class_weights must be non-negative")
    if not bool((weights > 0).any()):
        raise ValueError("class_weights must contain at least one positive value")
    return weights.to(device=logits.device, dtype=logits.dtype)


def _semantic_dice(
    logits: Tensor,
    target: Tensor,
    *,
    ignore_index: int,
    valid_voxel_count: int,
) -> Tensor:
    predictions = logits.detach().argmax(dim=1)
    valid = target != ignore_index
    dice: list[Tensor] = []
    for class_index in range(SEMANTIC_CLASS_COUNT):
        predicted = (predictions == class_index) & valid
        expected = (target == class_index) & valid
        denominator = predicted.sum() + expected.sum()
        if int(denominator.detach().cpu()) == 0:
            # An absent class is a perfect agreement only when there is actual
            # supervision support.  An entirely ignored volume reports zero
            # for all classes so it cannot look like a successful metric.
            value = logits.new_zeros(()) if valid_voxel_count == 0 else logits.new_ones(())
        else:
            intersection = (predicted & expected).sum()
            value = 2.0 * intersection.to(dtype=logits.dtype) / denominator.to(dtype=logits.dtype)
        dice.append(value)
    return torch.stack(dice)


@dataclass(frozen=True)
class SemanticGroundingLossResult:
    """Cross-entropy grounding loss and three-class semantic diagnostics."""

    loss: Tensor
    dice: Tensor
    mean_dice: Tensor
    valid_voxel_count: int
    metrics: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        if not isinstance(self.loss, Tensor) or self.loss.ndim != 0 or not bool(torch.isfinite(self.loss)):
            raise ValueError("loss must be one finite scalar tensor")
        if (
            not isinstance(self.dice, Tensor)
            or self.dice.shape != (SEMANTIC_CLASS_COUNT,)
            or not bool(torch.isfinite(self.dice).all())
        ):
            raise ValueError("dice must be a finite tensor with one value for each of the three classes")
        if not isinstance(self.mean_dice, Tensor) or self.mean_dice.ndim != 0 or not bool(torch.isfinite(self.mean_dice)):
            raise ValueError("mean_dice must be one finite scalar tensor")
        if not isinstance(self.valid_voxel_count, int) or self.valid_voxel_count < 0:
            raise ValueError("valid_voxel_count must be a non-negative integer")
        checked = dict(self.metrics)
        for name, value in checked.items():
            if not isinstance(name, str) or not isinstance(value, Tensor) or value.ndim != 0:
                raise ValueError("metrics must map names to scalar tensors")
            if not bool(torch.isfinite(value)):
                raise ValueError("metrics must contain finite values")
        object.__setattr__(self, "metrics", MappingProxyType(checked))

    @property
    def total(self) -> Tensor:
        """Alias matching the repository's other loss result objects."""

        return self.loss

    @property
    def dice_per_class(self) -> Tensor:
        """The ordered ``[normal, edema, tumor-core]`` Dice vector."""

        return self.dice

    @property
    def semantic_dice(self) -> Tensor:
        """Readable alias for the ordered semantic Dice vector."""

        return self.dice


def compute_semantic_grounding_loss(
    semantic_logits: Tensor,
    target: Tensor,
    *,
    ignore_index: int = -100,
    class_weights: Tensor | Sequence[float] | None = None,
) -> SemanticGroundingLossResult:
    """Compute training-only coarse-semantic cross entropy and Dice metrics.

    ``semantic_logits`` must be ``[B,3,D,H,W]``.  ``target`` is the output of
    :func:`build_coarse_semantic_target` (or an equivalent integer map) with
    shape ``[B,D,H,W]``; a singleton channel form is also accepted.  The
    optional ``class_weights`` are explicit caller-provided weights in class
    order ``normal, edema, tumor-core candidate`` and are never inferred from
    the target.  If every voxel is ignored, the differentiable loss is zero
    and all Dice values are zero.

    The target is consumed only at this training boundary.  This function is
    intentionally not imported by or called from model inference APIs and it
    does not alter the existing Gate-E objective.
    """

    ignore_index = _validate_ignore_index(ignore_index)
    if (
        not isinstance(semantic_logits, Tensor)
        or semantic_logits.ndim != 5
        or semantic_logits.shape[1] != SEMANTIC_CLASS_COUNT
        or not semantic_logits.is_floating_point()
    ):
        raise ValueError("semantic_logits must be a floating tensor [B,3,D,H,W]")
    if semantic_logits.shape[0] <= 0 or any(length <= 0 for length in semantic_logits.shape[-3:]):
        raise ValueError("semantic_logits must have positive batch and spatial dimensions")
    if not bool(torch.isfinite(semantic_logits).all()):
        raise ValueError("semantic_logits must be finite")

    checked_target = _validate_target(
        target,
        spatial_shape=(semantic_logits.shape[0], *semantic_logits.shape[-3:]),
        device=semantic_logits.device,
    )
    valid = checked_target != ignore_index
    valid_labels = (checked_target >= 0) & (checked_target < SEMANTIC_CLASS_COUNT)
    if not bool((~valid | valid_labels).all()):
        raise ValueError("target labels must be 0, 1, 2, or ignore_index")
    weights = _validate_class_weights(class_weights, logits=semantic_logits)
    valid_voxel_count = int(valid.sum().detach().cpu())
    if weights is not None and valid_voxel_count > 0:
        weighted_support = weights.index_select(0, checked_target[valid]).sum()
        if not bool(weighted_support > 0):
            raise ValueError("class_weights must assign positive total weight to valid target classes")

    if valid_voxel_count == 0:
        loss = semantic_logits.sum() * 0.0
    else:
        loss = F.cross_entropy(
            semantic_logits,
            checked_target,
            weight=weights,
            ignore_index=ignore_index,
            reduction="mean",
        )
    if not bool(torch.isfinite(loss)):
        raise ValueError("semantic grounding loss is non-finite")

    dice = _semantic_dice(
        semantic_logits,
        checked_target,
        ignore_index=ignore_index,
        valid_voxel_count=valid_voxel_count,
    )
    mean_dice = dice.mean()
    metrics = {
        "dice_class_0": dice[0],
        "dice_class_1": dice[1],
        "dice_class_2": dice[2],
        "mean_dice": mean_dice,
    }
    return SemanticGroundingLossResult(
        loss=loss,
        dice=dice,
        mean_dice=mean_dice,
        valid_voxel_count=valid_voxel_count,
        metrics=metrics,
    )


__all__ = [
    "BRATS_LABEL_VALUES",
    "SEMANTIC_CLASS_COUNT",
    "SemanticGroundingLossResult",
    "build_coarse_semantic_target",
    "compute_semantic_grounding_loss",
]
