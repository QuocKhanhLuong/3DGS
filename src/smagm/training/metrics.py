"""Bounded numeric diagnostics for T1-C software evidence."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError("non-finite gradient detected")
            total += float(parameter.grad.detach().norm().cpu())
    return total


def parameter_count(parameters: Iterable[torch.nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters)
