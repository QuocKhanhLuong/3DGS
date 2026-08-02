"""Explicit compute and timing diagnostics for legal training episodes.

The analytical counter intentionally reports only Conv2d/Linear forward work
under a declared ``2 FLOPs per MAC`` convention.  It is therefore suitable for
encoder comparison, but is not a total training-step compute estimate.  The
optional torch profiler separately reports only operators for which PyTorch
emits FLOP metadata; its coverage is deliberately labelled partial/unknown.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Any, TypeVar

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


def _conv2d_forward_flops(module: nn.Conv2d, output: torch.Tensor) -> int:
    if output.ndim != 4:
        raise ValueError("Conv2d output must have shape [B,C,H,W]")
    batch, channels, height, width = (int(value) for value in output.shape)
    if channels != module.out_channels:
        raise ValueError("Conv2d output channels disagree with its module")
    kernel_height, kernel_width = (int(value) for value in module.kernel_size)
    macs = batch * channels * height * width * (module.in_channels // module.groups) * kernel_height * kernel_width
    return 2 * macs


def _linear_forward_flops(module: nn.Linear, output: torch.Tensor) -> int:
    if output.ndim < 1 or int(output.shape[-1]) != module.out_features:
        raise ValueError("Linear output shape disagrees with its module")
    output_vectors = int(output.numel() // module.out_features)
    return 2 * output_vectors * module.in_features * module.out_features


def analytical_conv_linear_forward_flops(
    model: nn.Module,
    *args: Any,
    **kwargs: Any,
) -> dict[str, object]:
    """Execute ``model`` once and count only Conv2d/Linear forward MAC work.

    Bias adds, normalisation, activation, pooling, tensor movement, Gaussian
    construction, renderer work, loss, backward, and optimizer work are all
    intentionally excluded.  This prevents the result from being confused
    with total training-step FLOPs.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("analytical FLOP counter requires an nn.Module")
    named = {id(module): name for name, module in model.named_modules()}
    by_module: dict[str, dict[str, object]] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def _record(module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("analytical FLOP counter supports Tensor Conv2d/Linear outputs only")
        name = named.get(id(module), module.__class__.__name__)
        if isinstance(module, nn.Conv2d):
            count = _conv2d_forward_flops(module, output)
            kind = "Conv2d"
        elif isinstance(module, nn.Linear):
            count = _linear_forward_flops(module, output)
            kind = "Linear"
        else:  # pragma: no cover - hooks are registered only on supported modules.
            return
        prior = by_module.get(name)
        if prior is None:
            by_module[name] = {
                "operator": kind,
                "forward_flops_2flop_per_mac": count,
                "output_shape": list(output.shape),
                "calls": 1,
            }
        else:
            prior["forward_flops_2flop_per_mac"] = int(prior["forward_flops_2flop_per_mac"]) + count
            prior["calls"] = int(prior["calls"]) + 1

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(_record))
    try:
        model(*args, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
    total = sum(int(item["forward_flops_2flop_per_mac"]) for item in by_module.values())
    return {
        "metric": "forward_conv2d_linear_flops",
        "convention": "2_flops_per_mac",
        "scope": "forward Conv2d and Linear modules only",
        "forward_flops_2flop_per_mac": total,
        "by_module": by_module,
    }


def encoder_forward_flops_2flop_per_mac(
    encoder: nn.Module,
    *args: Any,
    **kwargs: Any,
) -> dict[str, object]:
    """Named wrapper used by product telemetry for the evidence encoder."""

    return analytical_conv_linear_forward_flops(encoder, *args, **kwargs)


def profile_supported_operator_flops(
    operation: Callable[[], T],
    *,
    enabled: bool,
    scope: str,
) -> tuple[T, dict[str, object]]:
    """Optionally profile one operation without implying complete coverage.

    PyTorch's ``with_flops`` metadata is incomplete for this pipeline.  The
    returned value is consequently named ``profiled_supported_operator_flops``
    and never ``training_step_flops`` or total FLOPs.
    """

    if not isinstance(enabled, bool):
        raise TypeError("profiler enabled flag must be boolean")
    if not isinstance(scope, str) or not scope:
        raise ValueError("profiler scope must be a non-empty string")
    common = {
        "profiler_scope": scope,
        "profiler_operator_coverage": "partial_unknown_torch_profiler_supported_operators_only",
    }
    if not enabled:
        return operation(), {
            **common,
            "profiled_supported_operator_flops": None,
            "profiler_enabled": False,
            "profiler_method": "not_invoked",
        }
    if not hasattr(torch, "profiler") or not hasattr(torch.profiler, "profile"):
        return operation(), {
            **common,
            "profiled_supported_operator_flops": None,
            "profiler_enabled": True,
            "profiler_method": "unavailable",
        }
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
    except Exception as error:  # Diagnostic support must not change execution semantics.
        return operation(), {
            **common,
            "profiled_supported_operator_flops": None,
            "profiler_enabled": True,
            "profiler_method": "unavailable",
            "profiler_reason": f"{type(error).__name__}: {error}",
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
        total = sum(
            int(value)
            for event in profiler.key_averages()
            for value in (getattr(event, "flops", 0) or 0,)
            if isinstance(value, (int, float)) and float(value) >= 0.0
        )
    except Exception as error:
        return result, {
            **common,
            "profiled_supported_operator_flops": None,
            "profiler_enabled": True,
            "profiler_method": "unavailable",
            "profiler_reason": f"{type(error).__name__}: {error}",
        }
    return result, {
        **common,
        "profiled_supported_operator_flops": total,
        "profiler_enabled": True,
        "profiler_method": "torch.profiler.with_flops",
    }


@dataclass
class PhaseTiming:
    """Accumulate named wall-clock durations without GPU synchronization.

    Callers that require device-complete timing should synchronize at their
    transaction boundary.  The object intentionally contains no tensors and
    is safe to serialize as part of an episode report.
    """

    milliseconds: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        if not isinstance(name, str) or not name:
            raise ValueError("phase timing name must be non-empty")
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - started) * 1000.0
            self.milliseconds[name] = self.milliseconds.get(name, 0.0) + elapsed

    def value(self, name: str) -> float:
        return float(self.milliseconds.get(name, 0.0))

    def report(self) -> dict[str, float]:
        return {name: float(value) for name, value in sorted(self.milliseconds.items())}


def peak_cuda_memory_bytes() -> dict[str, int | None]:
    """Return current peak CUDA allocator values, or explicit absence on CPU."""

    if not torch.cuda.is_available():
        return {"peak_cuda_allocated_bytes": None, "peak_cuda_reserved_bytes": None}
    return {
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def profile_training_step(operation: Callable[[], T]) -> tuple[T, dict[str, object]]:
    """Backward-compatible deprecated alias with corrected semantics.

    Existing callers receive the profiler result under its truthful name.  New
    product code must use :func:`profile_supported_operator_flops` explicitly.
    """

    return profile_supported_operator_flops(
        operation,
        enabled=True,
        scope="legacy one operation; supported PyTorch operator FLOPs only",
    )


__all__ = [
    "PhaseTiming",
    "analytical_conv_linear_forward_flops",
    "encoder_forward_flops_2flop_per_mac",
    "parameter_counts",
    "peak_cuda_memory_bytes",
    "profile_supported_operator_flops",
    "profile_training_step",
]
