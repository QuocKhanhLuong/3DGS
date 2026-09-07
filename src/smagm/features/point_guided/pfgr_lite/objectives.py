"""Target-after-inference objectives for the PFGR-Lite training services.

The helpers in this module deliberately contain only tensor algebra.  They do
not load MRI data or call a decoder, updater, teacher, or value model.  A stage
service builds a complete target-free route first and then passes its detached
target context to these functions.  Keeping that boundary explicit makes it
possible to inject spies in the stage tests and prevents a reconstruction
gradient from accidentally reaching V or the detached teacher.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from typing import Any

import torch
from torch import Tensor

from ..losses import ReconstructionLossConfig, reconstruction_loss


OBJECTIVE_SCHEMA = "pfgr-lite-objective-v1"
DEFAULT_EPSILON = 1e-3
DEFAULT_WRITE_SCALE = 0.1
DEFAULT_DELTA_WIDTH = 96


def _finite_tensor(name: str, value: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point() or value.numel() == 0:
        raise TypeError(f"{name} must be a non-empty floating tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain finite values")
    return value


def _mapping_or_attr(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _target_and_mask(target_context: object, prediction: Tensor) -> tuple[Tensor, Tensor]:
    """Resolve an owned target/mask pair without accepting a raw target API."""

    if target_context is None:
        raise ValueError("a completed target context is required after prediction")
    # The S0 semantic arm carries a validated W2 target context plus a
    # coarse segmentation target in a late-join mapping.  Validate the
    # authoritative context before unwrapping it; the raw segmentation never
    # enters target-free model construction.
    if isinstance(target_context, Mapping) and "target_context" in target_context:
        owned = target_context.get("target_context")
        if owned is None:
            raise ValueError("target_context mapping must contain an owned target context")
        target_context = owned
    validate = getattr(target_context, "validate_integrity", None)
    if callable(validate):
        validate()
    target = _mapping_or_attr(target_context, "target")
    mask = _mapping_or_attr(target_context, "observation_mask")
    if target is None:
        raise ValueError("target_context must expose an owned target tensor")
    target = _finite_tensor("target_context.target", target)
    if target.ndim == prediction.ndim - 1:
        target = target.unsqueeze(1)
    if target.ndim != prediction.ndim:
        raise ValueError("target rank must match prediction rank")
    if target.shape != prediction.shape:
        raise ValueError(f"target shape {tuple(target.shape)} does not match prediction {tuple(prediction.shape)}")
    if mask is None:
        mask = torch.ones_like(target, dtype=torch.bool)
    if not isinstance(mask, Tensor):
        raise TypeError("target_context.observation_mask must be a tensor")
    if mask.ndim == prediction.ndim - 1:
        mask = mask.unsqueeze(1)
    if mask.shape != prediction.shape:
        raise ValueError("target mask must align with prediction")
    if mask.dtype != torch.bool:
        if not mask.is_floating_point() and mask.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("target mask must be bool or binary numeric")
        if not bool(torch.isfinite(mask).all()) or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("target mask must contain exact binary values")
        mask = mask.to(dtype=torch.bool)
    else:
        mask = mask.clone()
    if mask.device != prediction.device:
        mask = mask.to(device=prediction.device)
    if not bool(mask.any()):
        raise ValueError("target mask must contain at least one observed voxel")
    if target.device != prediction.device or target.dtype != prediction.dtype:
        target = target.to(device=prediction.device, dtype=prediction.dtype)
    return target, mask


def _epsilon(config: object) -> float:
    raw = _mapping_or_attr(config, "epsilon", DEFAULT_EPSILON)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("objective epsilon must be finite and positive") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("objective epsilon must be finite and positive")
    return value


def _loss_per_subject(prediction: Tensor, target_context: object, *, config: object, include_auxiliary: bool = True) -> Tensor:
    """Return one Charbonnier reconstruction loss for each subject in a batch."""

    prediction = _finite_tensor("prediction", prediction)
    if prediction.ndim == 4:
        prediction = prediction.unsqueeze(1)
    if prediction.ndim != 5 or prediction.shape[1] != 1:
        raise ValueError("prediction must have shape [B,1,D,H,W] or [B,D,H,W]")
    target, mask = _target_and_mask(target_context, prediction)
    if include_auxiliary and _mapping_or_attr(config, "loss", "charbonnier_ssim_gradient") != "charbonnier":
        # Reuse the repository's locked local 3-D SSIM/gradient algebra.  It
        # requires wholly valid unpadded 3x3x3 windows; whole-volume moments
        # are deliberately not a substitute for structural similarity.
        ssim_weight = _finite_nonnegative_config(config, "ssim_weight", 0.2)
        gradient_weight = _finite_nonnegative_config(config, "gradient_weight", 0.1)
        data_range = _finite_nonnegative_config(config, "ssim_data_range", 1.0)
        if data_range <= 0.0:
            raise ValueError("ssim_data_range must be positive and finite")
        components: list[Tensor] = []
        for index in range(prediction.shape[0]):
            local = reconstruction_loss(
                prediction[index : index + 1],
                target[index : index + 1],
                mask[index : index + 1],
                config=ReconstructionLossConfig(
                    lambda_ssim=ssim_weight,
                    lambda_grad=gradient_weight,
                    ssim_data_range=data_range,
                ),
            )
            components.append(local.total)
        return torch.stack(components)
    difference = torch.sqrt((prediction - target).square() + _epsilon(config) ** 2)
    weights = mask.to(dtype=difference.dtype)
    denominator = weights.sum(dim=(-3, -2, -1), keepdim=True)
    if bool((denominator <= 0).any()):
        raise ValueError("target mask must retain at least one observed voxel per subject")
    return (difference * weights).sum(dim=(-3, -2, -1), keepdim=True) / denominator


def _finite_nonnegative_config(config: object, name: str, default: float) -> float:
    raw = _mapping_or_attr(config, name, default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def static_objective(completed_static_prediction: Tensor | Mapping[str, Any], target_context: object, *, config: object) -> Tensor:
    """Compute the post-prediction static reconstruction objective.

    The returned scalar is the mean of per-subject masked Charbonnier losses.
    The function accepts only a completed prediction and a validated target
    context; it intentionally has no data-loader or model arguments.
    """

    semantic_prediction: Tensor | None = None
    semantic_probabilities: Tensor | None = None
    semantic_target: Tensor | None = None
    prediction: Tensor
    if isinstance(completed_static_prediction, Mapping):
        prediction = completed_static_prediction.get("prediction", completed_static_prediction.get("output"))
        semantic_prediction = completed_static_prediction.get("semantic_prediction")
        semantic_probabilities = completed_static_prediction.get("semantic_probabilities")
        semantic_target = completed_static_prediction.get("semantic_target")
        if prediction is None:
            raise TypeError("static prediction mapping must contain prediction")
    else:
        prediction = completed_static_prediction
    if isinstance(target_context, Mapping):
        if semantic_target is None:
            semantic_target = target_context.get("semantic_target", target_context.get("segmentation_target"))
        target_context_for_loss: object = target_context.get("target_context", target_context)
    else:
        target_context_for_loss = target_context
    losses = _loss_per_subject(prediction, target_context_for_loss, config=config, include_auxiliary=True)
    result = losses.reshape(losses.shape[0], -1).mean(dim=1)
    if bool(_mapping_or_attr(config, "semantic_objective", False)):
        if semantic_prediction is None:
            semantic_prediction = _mapping_or_attr(target_context, "semantic_prediction")
        if semantic_probabilities is None:
            semantic_probabilities = _mapping_or_attr(target_context, "semantic_probabilities")
        if semantic_target is None:
            semantic_target = _mapping_or_attr(target_context, "semantic_target", _mapping_or_attr(target_context, "segmentation"))
        if semantic_target is None:
            raise ValueError("semantic_objective=True requires explicit post-prediction semantic labels")
        if not isinstance(semantic_target, Tensor):
            raise TypeError("semantic_target must be a tensor")
        if semantic_target.ndim == 5 and semantic_target.shape[1] == 1:
            semantic_target = semantic_target[:, 0]
        if semantic_target.ndim != 4:
            raise ValueError("semantic_target must be [B,D,H,W]")
        if semantic_prediction is not None and semantic_probabilities is not None:
            raise ValueError("provide semantic logits or semantic probabilities, not both")
        if semantic_prediction is not None:
            if not isinstance(semantic_prediction, Tensor) or semantic_prediction.ndim != 5:
                raise ValueError("semantic_prediction must be [B,C,D,H,W]")
            if semantic_prediction.shape[0] != semantic_target.shape[0] or tuple(semantic_prediction.shape[-3:]) != tuple(semantic_target.shape[-3:]):
                raise ValueError("semantic target/logits geometry mismatch")
            if semantic_prediction.shape[1] != 3 or not semantic_prediction.is_floating_point() or not bool(torch.isfinite(semantic_prediction).all()):
                raise ValueError("semantic_prediction must be finite floating [B,3,D,H,W]")
            target_labels = semantic_target.to(device=semantic_prediction.device, dtype=torch.long)
            valid = target_labels != -100
            if bool((valid & ((target_labels < 0) | (target_labels >= 3))).any()):
                raise ValueError("semantic_target labels must be 0, 1, 2, or -100")
            cross_entropy = torch.nn.functional.cross_entropy(semantic_prediction, target_labels, ignore_index=-100, reduction="none")
            semantic_per_subject = []
            for index in range(cross_entropy.shape[0]):
                support = valid[index]
                semantic_per_subject.append(cross_entropy[index][support].mean() if bool(support.any()) else semantic_prediction[index].sum() * 0.0)
            semantic_loss = torch.stack(semantic_per_subject)
        elif semantic_probabilities is not None:
            # PFGRLiteModel exposes the one-traversal semantic branch as
            # probabilities.  Validate the simplex and use explicit NLL;
            # never pass probabilities into ``cross_entropy`` as logits.
            if not isinstance(semantic_probabilities, Tensor) or semantic_probabilities.ndim != 5:
                raise ValueError("semantic_probabilities must be [B,3,D,H,W]")
            if semantic_probabilities.shape[1] != 3 or not semantic_probabilities.is_floating_point() or not bool(torch.isfinite(semantic_probabilities).all()):
                raise ValueError("semantic_probabilities must be finite floating [B,3,D,H,W]")
            if semantic_probabilities.shape[0] != semantic_target.shape[0] or tuple(semantic_probabilities.shape[-3:]) != tuple(semantic_target.shape[-3:]):
                raise ValueError("semantic target/probabilities geometry mismatch")
            if bool((semantic_probabilities < 0).any()) or not bool(torch.allclose(semantic_probabilities.sum(dim=1), torch.ones_like(semantic_probabilities[:, 0]), atol=1e-5, rtol=1e-5)):
                raise ValueError("semantic_probabilities must be a finite nonnegative simplex")
            target_labels = semantic_target.to(device=semantic_probabilities.device, dtype=torch.long)
            valid = target_labels != -100
            if bool((valid & ((target_labels < 0) | (target_labels >= 3))).any()):
                raise ValueError("semantic_target labels must be 0, 1, 2, or -100")
            safe_labels = target_labels.clamp_min(0).unsqueeze(1)
            nll = -torch.log(semantic_probabilities.clamp_min(torch.finfo(semantic_probabilities.dtype).tiny)).gather(1, safe_labels).squeeze(1)
            semantic_per_subject = []
            for index in range(nll.shape[0]):
                support = valid[index]
                semantic_per_subject.append(nll[index][support].mean() if bool(support.any()) else semantic_probabilities[index].sum() * 0.0)
            semantic_loss = torch.stack(semantic_per_subject)
        else:
            raise ValueError("semantic_objective=True requires semantic logits or probabilities")
        result = result + _finite_nonnegative_config(config, "semantic_weight", 0.2) * semantic_loss
    return result.mean()


def _route_predictions(trace: object) -> tuple[Tensor, ...]:
    """Extract post-write predictions from a differentiable route object."""

    candidates: Any = None
    if isinstance(trace, Mapping):
        for key in ("predictions", "route_predictions", "completed_predictions", "outputs"):
            if key in trace:
                candidates = trace[key]
                break
        if candidates is None and "prediction" in trace:
            candidates = (trace["prediction"],)
    else:
        for name in ("predictions", "route_predictions", "completed_predictions", "outputs"):
            value = getattr(trace, name, None)
            if value is not None:
                candidates = value
                break
        if candidates is None:
            value = getattr(trace, "prediction", None)
            if value is not None:
                candidates = (value,)
    if candidates is None and isinstance(trace, Tensor):
        candidates = (trace,)
    if candidates is None:
        # A plain sequence of prediction tensors is a convenient tiny-fixture
        # representation and remains unambiguous because tensors are never
        # accepted as route metadata mappings.
        if isinstance(trace, Sequence) and not isinstance(trace, (str, bytes)):
            candidates = trace
    if candidates is None or isinstance(candidates, (str, bytes)):
        raise TypeError("completed_differentiable_trace must expose predictions")
    if isinstance(candidates, Tensor):
        candidates = (candidates,)
    try:
        values = tuple(candidates)
    except TypeError as exc:
        raise TypeError("route predictions must be an iterable of tensors") from exc
    if not values:
        raise ValueError("differentiable route must contain at least one post-write prediction")
    return tuple(_finite_tensor("route prediction", value) for value in values)


def _route_deltas(trace: object) -> tuple[Tensor, ...]:
    candidates: Any = None
    if isinstance(trace, Mapping):
        candidates = trace.get("deltas", trace.get("actual_deltas"))
    else:
        candidates = getattr(trace, "deltas", getattr(trace, "actual_deltas", None))
    if candidates is None:
        return ()
    if isinstance(candidates, Tensor):
        candidates = (candidates,)
    values = tuple(candidates)
    for value in values:
        _finite_tensor("route delta", value)
        if value.shape[-1] != DEFAULT_DELTA_WIDTH:
            raise ValueError("route deltas must have final width 96")
    return values


def _per_subject_route_loss(prediction: Tensor, target_context: object, *, config: object) -> Tensor:
    losses = _loss_per_subject(prediction, target_context, config=config, include_auxiliary=False)
    return losses.reshape(losses.shape[0], -1).mean(dim=1)


def updater_objective(completed_differentiable_trace: object, target_context: object, *, config: object) -> Tensor:
    """Compute the exact S1 random-route objective with optional delta penalty.

    For one action, ``L = R(Z1)``.  For ``K > 1``,
    ``L = 0.5 R(ZK) + 0.5/(K-1) * sum(R(Zt), t<K)``.  ``R`` is evaluated per
    subject before averaging, so a large batch cannot silently receive extra
    weight.  The optional per-subject correction regularizer is normalized by
    ``96 * 0.1**2`` and defaults to zero.
    """

    predictions = _route_predictions(completed_differentiable_trace)
    losses = tuple(_per_subject_route_loss(value, target_context, config=config) for value in predictions)
    subject_count = losses[0].shape[0]
    if any(value.shape != (subject_count,) for value in losses):
        raise ValueError("route predictions must have one loss per subject")
    count = len(losses)
    if count == 1:
        route_loss = losses[0]
    else:
        route_loss = 0.5 * losses[-1] + 0.5 / float(count - 1) * torch.stack(losses[:-1], dim=0).sum(dim=0)
    objective = route_loss.mean()

    raw_weight = _mapping_or_attr(config, "delta_weight", 0.0)
    try:
        delta_weight = float(raw_weight)
    except (TypeError, ValueError) as exc:
        raise ValueError("delta_weight must be finite and nonnegative") from exc
    if not math.isfinite(delta_weight) or delta_weight < 0.0:
        raise ValueError("delta_weight must be finite and nonnegative")
    deltas = _route_deltas(completed_differentiable_trace)
    if delta_weight and not deltas:
        raise ValueError("nonzero delta_weight requires route deltas")
    if deltas and delta_weight:
        penalty_parts: list[Tensor] = []
        for delta in deltas:
            # Flatten all route/subject dimensions while preserving one value
            # per subject whenever a subject axis is present.
            if delta.ndim == 1:
                if delta.shape[0] != DEFAULT_DELTA_WIDTH:
                    raise ValueError("route delta must have width 96")
                penalty_parts.append(delta.square().sum().expand(subject_count))
            else:
                if delta.shape[-1] != DEFAULT_DELTA_WIDTH:
                    raise ValueError("route delta must have width 96")
                if delta.ndim == 2:
                    flat = delta.square().sum(dim=-1)
                else:
                    flat = delta.square().sum(dim=-1).reshape(delta.shape[0], -1).mean(dim=1)
                penalty_parts.append(flat)
        penalty = torch.stack(penalty_parts, dim=0).mean(dim=0)
        objective = objective + delta_weight * (penalty / (DEFAULT_DELTA_WIDTH * DEFAULT_WRITE_SCALE**2)).mean()
    return objective


__all__ = ["OBJECTIVE_SCHEMA", "static_objective", "updater_objective"]
