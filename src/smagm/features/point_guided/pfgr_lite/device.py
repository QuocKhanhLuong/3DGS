"""Single requested/effective device resolver for PFGR-Lite.

Device selection is an operational concern, but it must be resolved at one
boundary so every PFGR stage and service observes the same result.  In
particular, a requested CUDA device is never silently replaced by CPU: an
unavailable accelerator or an out-of-range index is an actionable failure.
The small typed envelope is safe to persist in receipts and deliberately does
not participate in producer compatibility hashes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


DEVICE_SCHEMA = "pfgr-lite-device-resolution-v1"


@dataclass(frozen=True)
class DeviceResolution:
    """Requested/effective device identity and availability observations."""

    requested: str
    configured: str | None
    effective: str
    accelerator: str
    index: int | None
    cuda_available: bool
    cuda_device_count: int
    fallback: bool = False
    fallback_reason: str | None = None
    schema_version: str = DEVICE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != DEVICE_SCHEMA:
            raise ValueError("unknown PFGR device-resolution schema")
        for name in ("requested", "effective", "accelerator"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.configured is not None and (
            not isinstance(self.configured, str) or not self.configured.strip()
        ):
            raise ValueError("configured device must be a nonempty string or None")
        if not isinstance(self.cuda_available, bool) or not isinstance(self.fallback, bool):
            raise TypeError("cuda_available and fallback must be bool")
        if not isinstance(self.cuda_device_count, int) or isinstance(self.cuda_device_count, bool) or self.cuda_device_count < 0:
            raise ValueError("cuda_device_count must be a nonnegative integer")
        if self.index is not None and (not isinstance(self.index, int) or isinstance(self.index, bool) or self.index < 0):
            raise ValueError("device index must be a nonnegative integer or None")
        if not self.fallback and self.fallback_reason is not None:
            raise ValueError("fallback_reason requires fallback=True")

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.effective)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "requested_device": self.requested,
            "configured_device": self.configured,
            "effective_device": self.effective,
            "accelerator": self.accelerator,
            "index": self.index,
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
        }


def _normalise_requested(requested: str | None, configured: str | None) -> tuple[str, str | None]:
    configured_value = None if configured is None else str(configured).strip()
    if configured_value == "":
        configured_value = None
    requested_value = None if requested is None else str(requested).strip()
    # ``None`` means the caller did not override the config.  Empty strings are
    # rejected rather than becoming an accidental CPU fallback.
    if requested_value == "":
        raise ValueError("requested device cannot be empty")
    chosen = requested_value or configured_value or "cpu"
    return chosen, configured_value


def _cuda_observations() -> tuple[bool, int]:
    try:
        return bool(torch.cuda.is_available()), int(torch.cuda.device_count())
    except Exception as error:  # pragma: no cover - backend diagnostics
        raise RuntimeError(f"unable to query CUDA availability: {type(error).__name__}: {error}") from error


def resolve_device(
    requested: str | None = None,
    configured: str | None = None,
    *,
    torch_module: Any | None = None,
) -> DeviceResolution:
    """Resolve one device and fail closed for unavailable CUDA/indexes.

    ``requested`` is the explicit CLI override.  When omitted, ``configured``
    is used, followed by CPU.  The optional ``torch_module`` is intended for
    guarded tests that provide a CUDA availability stub; normal callers use
    the imported PyTorch module.
    """

    backend = torch if torch_module is None else torch_module
    chosen, configured_value = _normalise_requested(requested, configured)
    try:
        parsed = backend.device(chosen)
    except Exception as error:
        raise ValueError(f"invalid PFGR device {chosen!r}: {error}") from error
    accelerator = str(parsed.type)
    index = parsed.index
    cuda_available = bool(backend.cuda.is_available())
    cuda_count = int(backend.cuda.device_count())
    if accelerator == "cuda":
        if not cuda_available or cuda_count < 1:
            raise RuntimeError(
                f"CUDA device {chosen!r} was requested but CUDA is unavailable "
                f"(cuda_available={cuda_available}, device_count={cuda_count}); refusing CPU fallback"
            )
        resolved_index = 0 if index is None else int(index)
        if resolved_index < 0 or resolved_index >= cuda_count:
            raise RuntimeError(
                f"CUDA device index {resolved_index} is unavailable; "
                f"visible device count is {cuda_count}; refusing CPU fallback"
            )
        effective = f"cuda:{resolved_index}"
        return DeviceResolution(
            requested=chosen,
            configured=configured_value,
            effective=effective,
            accelerator=accelerator,
            index=resolved_index,
            cuda_available=cuda_available,
            cuda_device_count=cuda_count,
        )
    # CPU and non-CUDA accelerators (e.g. mps) are still validated by PyTorch
    # at module placement.  We intentionally do not convert unavailable MPS to
    # CPU; only the explicit CUDA guard above is needed for the supported
    # production path, and all other backend errors remain visible.
    # PyTorch canonicalizes ``cpu:0`` to the unindexed CPU device.  Preserve
    # the original request in the receipt, but expose the actual placement so
    # tensor/device parity checks do not observe a fictitious ``cpu:0``.
    effective = "cpu" if accelerator == "cpu" else str(parsed)
    if accelerator == "cpu":
        index = None
    return DeviceResolution(
        requested=chosen,
        configured=configured_value,
        effective=effective,
        accelerator=accelerator,
        index=index,
        cuda_available=cuda_available,
        cuda_device_count=cuda_count,
    )


__all__ = ["DEVICE_SCHEMA", "DeviceResolution", "resolve_device"]
