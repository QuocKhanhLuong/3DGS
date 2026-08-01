"""Small compatibility guards for an incomplete optional TorchDynamo install.

The execution environment used by this repository can contain a PyTorch build
whose eager optimizer code is present while its optional ``torch._dynamo``
package is internally inconsistent.  Recent PyTorch versions decorate eager
optimizer methods with wrappers that import Dynamo even when compilation is not
requested.  Unwrapping those methods preserves eager optimizer semantics and
keeps the failure local to the broken optional component.
"""

from __future__ import annotations

from collections.abc import Callable
import functools
from typing import Any


_PATCHED = False


def _unwrap_dynamo_wrapper(function: Callable[..., Any]) -> Callable[..., Any]:
    current = function
    while hasattr(current, "__wrapped__"):
        wrapped = getattr(current, "__wrapped__")
        if not callable(wrapped):
            break
        current = wrapped
    return current


def _eager_step_without_dynamo(function: Callable[..., Any]) -> Callable[..., Any]:
    """Retain the optimizer's no-grad guard while removing its Dynamo import."""

    import torch

    original = _unwrap_dynamo_wrapper(function)

    @functools.wraps(original)
    def eager_step(self: Any, *args: Any, **kwargs: Any) -> Any:
        previous = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(bool(self.defaults.get("differentiable", False)))
            return original(self, *args, **kwargs)
        finally:
            torch.set_grad_enabled(previous)

    return eager_step


def ensure_eager_optimizer_compatibility() -> bool:
    """Return whether eager optimizer wrappers were patched for broken Dynamo.

    The probe is deliberately tiny and does not perform an optimization step.
    A healthy installation is left untouched.  If only the optional Dynamo
    import is broken, eager ``torch.optim`` remains usable without changing
    tensor/autograd behavior.
    """

    global _PATCHED
    if _PATCHED:
        return True

    import torch
    import torch.optim as optim

    try:
        parameter = torch.nn.Parameter(torch.ones(1))
        optim.Adam((parameter,), lr=0.0)
    except ImportError:
        optimizer_base = optim.Optimizer
        for method_name in ("add_param_group", "zero_grad", "state_dict", "load_state_dict"):
            method = getattr(optimizer_base, method_name)
            setattr(optimizer_base, method_name, _unwrap_dynamo_wrapper(method))

        for candidate in vars(optim).values():
            if not isinstance(candidate, type) or not issubclass(candidate, optimizer_base):
                continue
            method = candidate.__dict__.get("step")
            if method is not None:
                setattr(candidate, "step", _eager_step_without_dynamo(method))
        _PATCHED = True
        return True
    return False


__all__ = ["ensure_eager_optimizer_compatibility"]
