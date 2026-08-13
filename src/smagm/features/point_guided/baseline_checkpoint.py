"""Atomic Gate-F resume and clean Gate-G inference checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
import random
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .baseline_inference import baseline_checkpoint_metadata


TRAIN_RESUME_SCHEMA = "point-guided-training-resume-v1"


def _atomic_torch_save(payload: object, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def capture_rng_state() -> dict[str, object]:
    """Capture process RNG state for strict local resume."""

    result: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng_state(state: Mapping[str, object]) -> None:
    if not isinstance(state, Mapping):
        raise ValueError("resume RNG state must be a mapping")
    if "python" in state:
        random.setstate(state["python"])  # type: ignore[arg-type]
    if "numpy" in state:
        np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    if "torch" in state:
        torch.set_rng_state(state["torch"])  # type: ignore[arg-type]
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])  # type: ignore[arg-type]


def save_training_resume_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    global_step: int,
    best_validation_reconstruction_loss: float,
    training_config: Mapping[str, Any],
    split_hash: str,
    metadata: Mapping[str, Any],
) -> Path:
    """Write the resumable checkpoint atomically."""

    payload = {
        "schema": TRAIN_RESUME_SCHEMA,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": None if scaler is None else scaler.state_dict(),
        "best_validation_reconstruction_loss": float(best_validation_reconstruction_loss),
        "training_config": dict(training_config),
        "split_hash": str(split_hash),
        "rng_state": capture_rng_state(),
        "metadata": dict(metadata),
    }
    return _atomic_torch_save(payload, path)


def load_training_resume_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    expected_split_hash: str,
) -> dict[str, Any]:
    """Strictly restore a local training checkpoint and return its counters."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"training resume checkpoint does not exist: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != TRAIN_RESUME_SCHEMA:
        raise ValueError("resume checkpoint has an unsupported schema")
    required = {
        "schema",
        "epoch",
        "global_step",
        "model_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "best_validation_reconstruction_loss",
        "training_config",
        "split_hash",
        "rng_state",
        "metadata",
    }
    if set(payload) != required:
        raise ValueError(f"resume checkpoint keys must be exactly {sorted(required)}")
    if payload["split_hash"] != expected_split_hash:
        raise ValueError("resume checkpoint split hash does not match the current split")
    model_state = payload["model_state_dict"]
    if not isinstance(model_state, Mapping):
        raise ValueError("resume model_state_dict must be a mapping")
    model.load_state_dict(model_state, strict=True)
    optimizer_state = payload["optimizer_state_dict"]
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("resume optimizer_state_dict must be a mapping")
    optimizer.load_state_dict(optimizer_state)
    scaler_state = payload["scaler_state_dict"]
    if scaler is not None:
        if scaler_state is not None and not isinstance(scaler_state, Mapping):
            raise ValueError("resume scaler_state_dict must be a mapping or null")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
    restore_rng_state(payload["rng_state"])  # type: ignore[arg-type]
    return {
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "best_validation_reconstruction_loss": float(payload["best_validation_reconstruction_loss"]),
        "training_config": payload["training_config"],
        "metadata": payload["metadata"],
    }


def save_clean_inference_checkpoint(path: str | Path, model: nn.Module) -> Path:
    """Write the exact strict baseline inference payload, atomically."""

    return _atomic_torch_save(
        {
            "metadata": baseline_checkpoint_metadata(model),
            "state_dict": model.state_dict(),
        },
        path,
    )


__all__ = [
    "TRAIN_RESUME_SCHEMA",
    "capture_rng_state",
    "load_training_resume_checkpoint",
    "restore_rng_state",
    "save_clean_inference_checkpoint",
    "save_training_resume_checkpoint",
]
