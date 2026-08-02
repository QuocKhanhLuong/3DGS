"""Best-effort model-size and training-step FLOPs telemetry.

The FLOPs value is an execution diagnostic for one legal forward/renderer/loss
and backward step.  It is not a claim about total cohort compute and is kept
optional so profiler support cannot change scientific execution semantics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeVar

import torch
from torch import nn


T = TypeVar("T")


def parameter_counts(modules: Mapping[str, nn.Module]) -> dict[str, object]:
    """Return total and trainable parameter counts for named modules."""

    by_module: dict[str, dict[str, int]] = {}
    total = 0
    trainable = 0
    for name, module in modules.items():
        if not isinstance(module, nn.Module):
            raise TypeError(f"{name} must be a torch.nn.Module")
        module_total = sum(int(parameter.numel()) for parameter in module.parameters())
        module_trainable = sum(int(parameter.numel()) for parameter in module.parameters() if parameter.requires_grad)
        by_module[str(name)] = {
            "parameters": module_total,
            "trainable_parameters": module_trainable,
        }
        total += module_total
        trainable += module_trainable
    return {
        "parameters": total,
        "trainable_parameters": trainable,
        "by_module": by_module,
    }


def profile_training_step(
    operation: Callable[[], T],
) -> tuple[T, dict[str, object]]:
    """Execute ``operation`` once and collect optional profiler FLOPs.

    Profiler setup/teardown failures degrade to an explicit unavailable
    telemetry record. Exceptions raised by the operation itself are never
    swallowed or retried.
    """

    if not hasattr(torch, "profiler") or not hasattr(torch.profiler, "profile"):
        return operation(), {"flops": None, "method": "unavailable", "supported": False}
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_flops=True,
        )
        profiler.start()
    except Exception as error:  # profiler support is diagnostic only
        return operation(), {
            "flops": None,
            "method": "unavailable",
            "supported": False,
            "reason": f"{type(error).__name__}: {error}",
        }

    try:
        result = operation()
    except BaseException:
        try:
            profiler.stop()
        except Exception:
            pass
        raise
    try:
        profiler.stop()
    except Exception as error:  # do not make telemetry a training blocker
        return result, {
            "flops": None,
            "method": "unavailable",
            "supported": False,
            "reason": f"{type(error).__name__}: {error}",
        }

    total_flops = 0
    try:
        for event in profiler.key_averages():
            value = getattr(event, "flops", 0) or 0
            if isinstance(value, (int, float)) and float(value) >= 0:
                total_flops += int(value)
    except Exception as error:
        return result, {
            "flops": None,
            "method": "unavailable",
            "supported": False,
            "reason": f"{type(error).__name__}: {error}",
        }
    return result, {
        "flops": total_flops,
        "method": "torch.profiler.with_flops",
        "supported": total_flops > 0,
        "scope": "one legal training step: context encoding, rendering, loss, backward",
    }
