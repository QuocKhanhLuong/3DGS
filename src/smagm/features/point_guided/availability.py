"""Parameter-free candidate availability policies shared by Gate F and Gate G."""

from __future__ import annotations

import torch
from torch import Tensor


class ExactNoRevisitPolicy:
    """Make an exact selected candidate unavailable on later route steps.

    The policy is caller-owned: passing no policy leaves the completed Gate-C
    primitive revisit-capable.  Its finite lowest-utility sentinel preserves
    the route solver's finite-input contract and never alters raw rewards.
    """

    def initial_available(self, *, batch: int, point_count: int, device: torch.device) -> Tensor:
        if batch <= 0 or point_count <= 0:
            raise ValueError("candidate availability requires positive batch and point counts")
        return torch.ones(batch, point_count, dtype=torch.bool, device=device)

    def mask_utility(self, utility: Tensor, available: Tensor) -> Tensor:
        if (
            not isinstance(utility, Tensor)
            or utility.ndim != 2
            or not utility.is_floating_point()
            or not bool(torch.isfinite(utility).all())
            or not isinstance(available, Tensor)
            or available.dtype != torch.bool
            or available.shape != utility.shape
            or available.device != utility.device
        ):
            raise ValueError("utility and availability must be aligned finite [B,N] tensors")
        unavailable = torch.full_like(utility, torch.finfo(utility.dtype).min)
        return torch.where(available, utility, unavailable)

    def update_available(self, available: Tensor, selection_indices: Tensor, active: Tensor) -> Tensor:
        if (
            not isinstance(available, Tensor)
            or available.ndim != 2
            or available.dtype != torch.bool
            or not isinstance(selection_indices, Tensor)
            or selection_indices.shape != available.shape[:1]
            or selection_indices.dtype != torch.long
            or selection_indices.device != available.device
            or not isinstance(active, Tensor)
            or active.shape != selection_indices.shape
            or active.dtype != torch.bool
            or active.device != available.device
        ):
            raise ValueError("availability state and selection must align")
        next_available = available.clone()
        if bool(active.any()):
            selected = selection_indices[active]
            if bool((selected < 0).any()) or bool((selected >= available.shape[1]).any()):
                raise ValueError("active selections must be valid candidate indices")
            next_available[active.nonzero(as_tuple=False).squeeze(1), selected] = False
        return next_available


__all__ = ["ExactNoRevisitPolicy"]
