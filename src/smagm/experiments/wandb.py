"""Optional, fail-closed Weights & Biases support for experiment runners.

The optional W&B package is imported only when a run is started in an enabled
mode. Disabled runs do not inspect the dependency, and offline runs pass
mode="offline" without invoking login, sync, artifact, or network APIs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
import importlib
import math
import numbers
import os
from pathlib import Path
import re
from typing import Any


__all__ = [
    "FinishMetadata",
    "WandbLogger",
    "WandbMode",
    "redact_absolute_paths",
    "sanitize_config",
    "sanitize_metadata",
]


_REDACTED_PATH = "<redacted-absolute-path>"
_FILE_URI_PATH_PATTERN = re.compile(
    r"""(?i)(file://)(?:[A-Za-z]:)?/[^\s,;()\[\]{}<>"']*"""
)
_PATH_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_./:\\-])"
    r"(?:"
    r"/(?!/)[^\s,;()\[\]{}<>\"']*"
    r"|[A-Za-z]:[\\/][^\s,;()\[\]{}<>\"']*"
    r"|\\\\[^\s,;()\[\]{}<>\"']*"
    r")"
)
_WINDOWS_ABSOLUTE_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")


class WandbMode(str, Enum):
    """Supported W&B modes."""

    DISABLED = "disabled"
    OFFLINE = "offline"
    ONLINE = "online"

    @classmethod
    def coerce(cls, value: WandbMode | str) -> WandbMode:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("W&B mode must be a WandbMode or string")
        try:
            return cls(value.strip().lower())
        except ValueError as error:
            raise ValueError("W&B mode must be disabled, offline, or online") from error


@dataclass(frozen=True)
class FinishMetadata:
    """Stable identity and fallback information returned by finish()."""

    run_id: str | None
    url: str | None
    mode: str
    fallback_reason: str | None

    @property
    def fallback(self) -> str | None:
        """Compatibility alias for callers using the shorter field name."""

        return self.fallback_reason

    def to_dict(self) -> dict[str, str | None]:
        """Return JSON-friendly finish metadata."""

        return {
            "run_id": self.run_id,
            "url": self.url,
            "mode": self.mode,
            "fallback_reason": self.fallback_reason,
        }

    def as_dict(self) -> dict[str, str | None]:
        """Compatibility alias for callers that prefer as_dict()."""

        return self.to_dict()


def _is_absolute_path_string(value: str) -> bool:
    stripped = value.strip()
    return (
        stripped.startswith("/")
        or stripped.startswith("\\\\")
        or _WINDOWS_ABSOLUTE_PATTERN.match(stripped) is not None
    )


def redact_absolute_paths(value: str | os.PathLike[str]) -> str:
    """Redact absolute filesystem paths while retaining ordinary URLs."""

    if isinstance(value, os.PathLike):
        value = os.fsdecode(os.fspath(value))
    if not isinstance(value, str):
        raise TypeError("path redaction requires a string or path-like value")
    if _is_absolute_path_string(value):
        return _REDACTED_PATH

    value = _FILE_URI_PATH_PATTERN.sub(lambda match: f"{match.group(1)}{_REDACTED_PATH}", value)
    return _PATH_TOKEN_PATTERN.sub(_REDACTED_PATH, value)


def _sanitize_value(value: Any, *, context: str) -> object:
    if value is None:
        return None
    if isinstance(value, os.PathLike):
        return redact_absolute_paths(value)
    if isinstance(value, str):
        return redact_absolute_paths(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{context} must be finite")
        return float(value)
    if isinstance(value, numbers.Real):
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{context} must be finite")
        return converted
    if isinstance(value, Mapping):
        sanitized: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{context} mapping keys must be strings")
            safe_key = redact_absolute_paths(key)
            if safe_key in sanitized:
                raise ValueError(f"{context} contains duplicate keys after path redaction")
            sanitized[safe_key] = _sanitize_value(nested, context=f"{context}.{safe_key}")
        return sanitized
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item, context=f"{context}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{context} contains an unsupported value type: {type(value).__name__}")


def sanitize_config(config: Mapping[str, Any]) -> dict[str, object]:
    """Return JSON-like config values with absolute paths redacted."""

    if not isinstance(config, Mapping):
        raise TypeError("W&B config must be a mapping")
    sanitized = _sanitize_value(config, context="config")
    if not isinstance(sanitized, dict):
        raise TypeError("W&B config must sanitize to a mapping")
    return sanitized


def sanitize_metadata(metadata: Mapping[str, Any]) -> dict[str, object]:
    """Return JSON-like metadata values with absolute paths redacted."""

    if not isinstance(metadata, Mapping):
        raise TypeError("W&B metadata must be a mapping")
    sanitized = _sanitize_value(metadata, context="metadata")
    if not isinstance(sanitized, dict):
        raise TypeError("W&B metadata must sanitize to a mapping")
    return sanitized


def _sanitize_scalars(metrics: Mapping[str, Any]) -> dict[str, int | float | bool]:
    if not isinstance(metrics, Mapping):
        raise TypeError("W&B metrics must be a mapping")

    sanitized: dict[str, int | float | bool] = {}
    for key, value in metrics.items():
        if not isinstance(key, str):
            raise TypeError("W&B metric keys must be strings")
        safe_key = redact_absolute_paths(key)
        if safe_key in sanitized:
            raise ValueError("W&B metric keys collide after path redaction")
        if isinstance(value, bool):
            safe_value: int | float | bool = value
        elif isinstance(value, numbers.Integral):
            safe_value = int(value)
        elif isinstance(value, Decimal):
            if not value.is_finite():
                raise ValueError(f"metric {safe_key!r} must be finite")
            safe_value = float(value)
        elif isinstance(value, numbers.Real):
            safe_value = float(value)
            if not math.isfinite(safe_value):
                raise ValueError(f"metric {safe_key!r} must be finite")
        else:
            raise TypeError(f"metric {safe_key!r} must be a finite number")
        sanitized[safe_key] = safe_value
    return sanitized


def _safe_text(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return redact_absolute_paths(value)


def _safe_optional_text(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")
    return redact_absolute_paths(value)


def _mode_from_environment() -> WandbMode:
    raw_mode = os.environ.get("SMAGM_WANDB_MODE")
    if raw_mode is None:
        raw_mode = os.environ.get("WANDB_MODE")
    return WandbMode.coerce(raw_mode or WandbMode.DISABLED.value)


def _safe_error(error: Exception) -> str:
    return f"{type(error).__name__}: {redact_absolute_paths(str(error))}"


class WandbLogger:
    """Runner-facing optional W&B adapter.

    The constructor intentionally has four required runner inputs:
    config, run_name, run_dir, and metadata. The optional mode keyword is
    resolved from the explicit value, then SMAGM_WANDB_MODE, then
    WANDB_MODE, and finally disabled.

    Metadata is added to the W&B run config after initialization under its own
    mapping, while the runner config is passed as the initial W&B config.
    Both mappings are sanitized before they reach the client.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        run_name: str,
        run_dir: Path,
        metadata: Mapping[str, Any],
        *,
        mode: WandbMode | str | None = None,
        online_fallback: WandbMode | str = WandbMode.OFFLINE,
        wandb_module: Any | None = None,
    ) -> None:
        self._config = sanitize_config(config)
        self._metadata = sanitize_metadata(metadata)
        self._run_name = _safe_text(run_name, name="run_name")
        if not isinstance(run_dir, os.PathLike):
            raise TypeError("run_dir must be a pathlib.Path or path-like value")
        self._run_dir = Path(run_dir)

        configured_wandb = self._config.get("wandb")
        if isinstance(configured_wandb, Mapping) and configured_wandb.get("enabled") is False:
            self._requested_mode = WandbMode.DISABLED
        else:
            self._requested_mode = _mode_from_environment() if mode is None else WandbMode.coerce(mode)
        self._online_fallback = WandbMode.coerce(online_fallback)
        if self._online_fallback is WandbMode.ONLINE:
            raise ValueError("online_fallback must be offline or disabled")

        self._wandb_module = wandb_module
        self._run: Any | None = None
        self._active_mode: WandbMode | None = None
        self._fallback_reason: str | None = None
        self._started = False
        self._finished = False
        self._finish_metadata: FinishMetadata | None = None

    @property
    def mode(self) -> str:
        """Return the requested mode before start, otherwise effective mode."""

        effective = self._active_mode or self._requested_mode
        return effective.value

    @property
    def run_id(self) -> str | None:
        """Return the W&B run id when a run exists."""

        return self._run_text("id")

    @property
    def url(self) -> str | None:
        """Return the W&B run URL when a run exists."""

        return self._run_text("url")

    @property
    def fallback_reason(self) -> str | None:
        """Return a path-redacted explanation for a mode fallback."""

        return self._fallback_reason

    @property
    def requested_mode(self) -> str:
        """Return the configured mode before fallback."""

        return self._requested_mode.value

    @property
    def run(self) -> Any | None:
        """Return the underlying W&B-compatible run, if one exists."""

        return self._run

    def _init_kwargs(self, mode: WandbMode) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "config": self._config,
            "name": self._run_name,
            "dir": str(self._run_dir),
            "mode": mode.value,
        }
        configured_wandb = self._config.get("wandb")
        if isinstance(configured_wandb, Mapping):
            for key in ("project", "entity", "group", "job_type"):
                value = configured_wandb.get(key)
                if value is not None:
                    if not isinstance(value, str) or not value:
                        raise ValueError(f"wandb.{key} must be a non-empty string or null")
                    kwargs[key] = value
            tags = configured_wandb.get("tags")
            if tags is not None:
                if not isinstance(tags, (list, tuple)) or any(not isinstance(tag, str) or not tag for tag in tags):
                    raise ValueError("wandb.tags must be a sequence of non-empty strings")
                kwargs["tags"] = list(tags)
        return kwargs

    def _load_module(self) -> Any | None:
        if self._wandb_module is not None:
            return self._wandb_module
        try:
            self._wandb_module = importlib.import_module("wandb")
        except Exception as error:
            self._fallback_reason = f"{self._requested_mode.value} unavailable: {_safe_error(error)}"
            return None
        return self._wandb_module

    def _run_text(self, name: str) -> str | None:
        if self._run is None:
            return None
        value = getattr(self._run, name, None)
        return None if value is None else str(value)

    def _update_run_config(self, values: Mapping[str, Any]) -> None:
        if self._run is None or not values:
            return
        run_config = getattr(self._run, "config")
        run_config.update(values)

    def _try_initialize(self, module: Any, mode: WandbMode) -> tuple[bool, str | None]:
        try:
            self._run = module.init(**self._init_kwargs(mode))
            self._active_mode = mode
            if self._run is not None:
                self._update_run_config({"metadata": self._metadata})
        except Exception as error:
            self._run = None
            return False, _safe_error(error)
        return True, None

    def _record_runtime_failure(self, error: Exception) -> None:
        reason = f"{self._active_mode.value if self._active_mode else self._requested_mode.value} runtime failure: {_safe_error(error)}"
        self._fallback_reason = reason if self._fallback_reason is None else f"{self._fallback_reason}; {reason}"
        self._active_mode = WandbMode.DISABLED

    def start(self) -> WandbLogger:
        """Start the run once, falling back from online to offline/disabled."""

        if self._finished:
            raise RuntimeError("cannot start a finished W&B logger")
        if self._started:
            return self
        self._started = True

        if self._requested_mode is WandbMode.DISABLED:
            self._active_mode = WandbMode.DISABLED
            return self

        module = self._load_module()
        if module is None:
            self._active_mode = WandbMode.DISABLED
            return self

        initialized, reason = self._try_initialize(module, self._requested_mode)
        if initialized:
            return self

        self._fallback_reason = f"{self._requested_mode.value} initialization failed: {reason}"
        if self._requested_mode is WandbMode.ONLINE and self._online_fallback is WandbMode.OFFLINE:
            initialized, offline_reason = self._try_initialize(module, WandbMode.OFFLINE)
            if initialized:
                self._fallback_reason = f"{self._fallback_reason}; fell back to offline"
                return self
            self._fallback_reason = f"{self._fallback_reason}; offline initialization failed: {offline_reason}"

        self._active_mode = WandbMode.DISABLED
        return self

    def _ensure_started(self) -> None:
        if self._finished:
            raise RuntimeError("cannot log after finishing a W&B logger")
        if not self._started:
            self.start()

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> WandbLogger:
        """Log finite scalar metrics, starting the run lazily if necessary."""

        sanitized = _sanitize_scalars(metrics)
        if step is not None:
            if isinstance(step, bool) or not isinstance(step, numbers.Integral):
                raise TypeError("W&B step must be a non-negative integer or None")
            step = int(step)
            if step < 0:
                raise ValueError("W&B step must be a non-negative integer or None")

        self._ensure_started()
        if not sanitized or self._active_mode is WandbMode.DISABLED or self._run is None:
            return self
        try:
            if step is None:
                self._run.log(sanitized)
            else:
                self._run.log(sanitized, step=step)
        except Exception as error:
            self._record_runtime_failure(error)
        return self

    def log_scalars(self, scalars: Mapping[str, Any], *, step: int | None = None) -> WandbLogger:
        """Compatibility alias for the runner-facing log method."""

        return self.log(scalars, step=step)

    def update_summary(self, values: Mapping[str, Any]) -> WandbLogger:
        """Add path-redacted, JSON-safe provenance to the W&B summary."""

        sanitized = sanitize_metadata(values)
        self._ensure_started()
        if self._run is None or self._active_mode is WandbMode.DISABLED:
            return self
        try:
            summary = getattr(self._run, "summary", None)
            if summary is not None and hasattr(summary, "update"):
                summary.update(sanitized)
            else:
                self._update_run_config({"summary": sanitized})
        except Exception as error:
            self._record_runtime_failure(error)
        return self

    def log_images(self, images: Mapping[str, Any], *, step: int | None = None) -> WandbLogger:
        """Log only explicitly supplied derived images when the client supports images.

        Callers own the privacy boundary: the runner supplies normalized
        prediction/target/error/support/uncertainty maps, never source volumes
        or identifiers.  A client without ``Image`` support is a no-op.
        """

        if not isinstance(images, Mapping):
            raise TypeError("W&B images must be a mapping")
        if step is not None and (isinstance(step, bool) or not isinstance(step, numbers.Integral) or int(step) < 0):
            raise ValueError("W&B step must be a non-negative integer or None")
        self._ensure_started()
        if not images or self._run is None or self._active_mode is WandbMode.DISABLED:
            return self
        image_factory = getattr(self._wandb_module, "Image", None)
        if not callable(image_factory):
            return self
        payload: dict[str, Any] = {}
        for name, value in images.items():
            if not isinstance(name, str) or not name:
                raise TypeError("W&B image names must be non-empty strings")
            prepared = value.detach().cpu() if hasattr(value, "detach") else value
            if hasattr(prepared, "nan_to_num"):
                prepared = prepared.nan_to_num()
            if hasattr(prepared, "numpy"):
                prepared = prepared.numpy()
            payload[redact_absolute_paths(name)] = image_factory(prepared)
        try:
            self._run.log(payload, step=None if step is None else int(step))
        except Exception as error:
            self._record_runtime_failure(error)
        return self

    def _finish_summary(self, status: str, failure_reason: str | None) -> None:
        if self._run is None:
            return
        summary_values = {"status": status, "failure_reason": failure_reason}
        summary = getattr(self._run, "summary", None)
        if summary is not None and hasattr(summary, "update"):
            summary.update(summary_values)
            return
        self._update_run_config(summary_values)

    def finish(self, status: str = "finished", failure_reason: str | None = None) -> FinishMetadata:
        """Record final status, finish the client run, and return its metadata."""

        if self._finish_metadata is not None:
            return self._finish_metadata
        safe_status = _safe_text(status, name="status")
        safe_failure_reason = _safe_optional_text(failure_reason, name="failure_reason")
        if not self._started:
            self.start()

        if self._run is not None and self._active_mode is not WandbMode.DISABLED:
            try:
                self._finish_summary(safe_status, safe_failure_reason)
                finish = getattr(self._run, "finish", None)
                if callable(finish):
                    finish()
                elif self._wandb_module is not None:
                    module_finish = getattr(self._wandb_module, "finish", None)
                    if callable(module_finish):
                        module_finish()
            except Exception as error:
                self._record_runtime_failure(error)

        self._finished = True
        self._finish_metadata = FinishMetadata(
            run_id=self.run_id,
            url=self.url,
            mode=self.mode,
            fallback_reason=self._fallback_reason,
        )
        return self._finish_metadata

    def __enter__(self) -> WandbLogger:
        return self.start()

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        self.finish(
            status="failed" if exc_type is not None else "finished",
            failure_reason=None if exc_value is None else str(exc_value),
        )
        return False
