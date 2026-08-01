"""Diagnostics and optional regularizers for supported structural-field queries."""

from __future__ import annotations

import torch

from .contracts import StructuralFieldOutput


def overlap_consistency_loss(output: StructuralFieldOutput) -> torch.Tensor:
    legal = output.supported & (output.total_weight[:, 0] > 0)
    if not bool(legal.any()):
        raise ValueError("overlap consistency has no supported queries")
    return output.disagreement[legal].square().mean()


def field_gradient_diagnostics(values: torch.Tensor, points_ras_mm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gradients = torch.autograd.grad(values.sum(), points_ras_mm, create_graph=True, retain_graph=True)[0]
    if not bool(torch.isfinite(gradients).all()):
        raise FloatingPointError("field gradients are non-finite")
    norms = torch.linalg.vector_norm(gradients, dim=-1)
    return gradients, norms
