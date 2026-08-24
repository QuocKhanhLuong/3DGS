"""Atomic Gate-F resume and clean Gate-G inference checkpoint helpers."""

from __future__ import annotations

import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from .baseline_inference import baseline_checkpoint_metadata


# The v1 payload captured one process's RNG state, had no protocol binding, and
# omitted early-stopping progress.  It is intentionally not migrated: a resume
# is a provenance boundary, so historical payloads must fail closed.
TRAIN_RESUME_SCHEMA = "point-guided-training-resume-v2"
TRAIN_RESUME_PROTOCOL_VERSION = 1
TRAIN_RESUME_RNG_SCHEMA = "point-guided-rng-v2"

# These names are part of the resume contract.  Keep the classification
# explicit rather than comparing a loosely filtered settings dictionary: a
# future setting must be deliberately classified before it can be resumed.
TRAIN_RESUME_IMMUTABLE_FIELDS = (
    "model",
    "trajectory",
    "supervision",
    "normalization",
    "split_hash",
    "overfit",
    "require_segmentation",
    "seed",
    "optimizer_name",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "gradient_accumulation",
    "gradient_clip",
    "lambda_semantic",
    "semantic_class_weights",
    "amp",
    "amp_dtype",
    "worker_seed_protocol",
)
TRAIN_RESUME_COMPATIBLE_FIELDS = (
    "epochs",
    "decoder_chunk_size",
    "num_workers",
    "log_interval",
    "prediction_interval",
    "device",
)
TRAIN_RESUME_RUN_STATE_FIELDS = (
    "patience_count",
    "first_train_reconstruction_loss",
    "overfit_prediction_epochs",
)

# Short aliases make the protocol discoverable for callers while the longer
# names remain the canonical documentation-facing constants.
RESUME_IMMUTABLE_FIELDS = TRAIN_RESUME_IMMUTABLE_FIELDS
RESUME_COMPATIBLE_FIELDS = TRAIN_RESUME_COMPATIBLE_FIELDS


def _canonical(value: object) -> object:
    """Return a deterministic, torch-free representation for protocol data."""

    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _protocol_shape(protocol: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(protocol, Mapping):
        raise ValueError("resume protocol must be a mapping")
    required = {"version", "immutable", "compatible"}
    if set(protocol) != required:
        raise ValueError(f"resume protocol keys must be exactly {sorted(required)}")
    if protocol["version"] != TRAIN_RESUME_PROTOCOL_VERSION:
        raise ValueError(
            "resume protocol version is unsupported; historical protocol versions are not migrated"
        )
    immutable = protocol["immutable"]
    compatible = protocol["compatible"]
    if not isinstance(immutable, Mapping) or not isinstance(compatible, Mapping):
        raise ValueError("resume protocol immutable/compatible sections must be mappings")
    if set(immutable) != set(TRAIN_RESUME_IMMUTABLE_FIELDS):
        raise ValueError(
            "resume protocol immutable fields must be exactly "
            f"{list(TRAIN_RESUME_IMMUTABLE_FIELDS)}"
        )
    if set(compatible) != set(TRAIN_RESUME_COMPATIBLE_FIELDS):
        raise ValueError(
            "resume protocol compatible fields must be exactly "
            f"{list(TRAIN_RESUME_COMPATIBLE_FIELDS)}"
        )
    return {
        "version": TRAIN_RESUME_PROTOCOL_VERSION,
        "immutable": _canonical(immutable),
        "compatible": _canonical(compatible),
    }


def build_training_resume_protocol(
    *,
    model_config: Mapping[str, Any],
    trajectory_config: Mapping[str, Any],
    supervision_config: Mapping[str, Any],
    training_settings: Mapping[str, Any],
    normalization_config: Mapping[str, Any] | None,
    split_hash: str,
    overfit: bool,
    require_segmentation: bool,
    device: str,
) -> dict[str, Any]:
    """Build the explicit v2 protocol binding for a training run.

    Optimization and model/data semantics are immutable.  The compatible
    fields are operational controls that can change on an intentional resume
    without changing the target-free architecture or the Gate-E objective.
    """

    settings = dict(training_settings)
    immutable = {
        "model": model_config,
        "trajectory": trajectory_config,
        "supervision": supervision_config,
        "normalization": normalization_config,
        "split_hash": str(split_hash),
        "overfit": bool(overfit),
        "require_segmentation": bool(require_segmentation),
        "seed": settings.get("seed"),
        "optimizer_name": "adam",
        "learning_rate": settings.get("learning_rate"),
        "weight_decay": settings.get("weight_decay"),
        "batch_size": settings.get("batch_size"),
        "gradient_accumulation": settings.get("gradient_accumulation"),
        "gradient_clip": settings.get("gradient_clip"),
        "lambda_semantic": settings.get("lambda_semantic"),
        "semantic_class_weights": settings.get("semantic_class_weights"),
        "amp": settings.get("amp"),
        "amp_dtype": settings.get("amp_dtype"),
        "worker_seed_protocol": "torch_initial_seed_v1",
    }
    compatible = {
        "epochs": settings.get("epochs"),
        "decoder_chunk_size": settings.get("decoder_chunk_size"),
        "num_workers": settings.get("num_workers"),
        "log_interval": settings.get("log_interval"),
        "prediction_interval": settings.get("prediction_interval"),
        "device": str(device),
    }
    return _protocol_shape(
        {
            "version": TRAIN_RESUME_PROTOCOL_VERSION,
            "immutable": immutable,
            "compatible": compatible,
        }
    )


def validate_training_resume_protocol(
    saved_protocol: Mapping[str, Any],
    current_protocol: Mapping[str, Any],
) -> None:
    """Fail closed when immutable resume protocol fields drift."""

    saved = _protocol_shape(saved_protocol)
    current = _protocol_shape(current_protocol)
    saved_immutable = saved["immutable"]
    current_immutable = current["immutable"]
    mismatches = [
        field
        for field in TRAIN_RESUME_IMMUTABLE_FIELDS
        if saved_immutable[field] != current_immutable[field]
    ]
    if mismatches:
        details = "; ".join(
            f"{field}: saved={saved_immutable[field]!r}, current={current_immutable[field]!r}"
            for field in mismatches
        )
        raise ValueError(f"resume immutable protocol mismatch ({details})")


def _current_rank_world_size() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = int(torch.distributed.get_rank())
        world_size = int(torch.distributed.get_world_size())
    else:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(f"invalid distributed rank/world size: rank={rank}, world_size={world_size}")
    return rank, world_size


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
    """Capture one process's Python/NumPy/CPU-Torch/CUDA RNG state."""

    result: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    # Keep the key present on CPU-only jobs so the v2 local-state shape is
    # explicit and cannot be confused with the legacy flat payload.
    result["cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return result


def restore_rng_state(state: Mapping[str, object]) -> None:
    if not isinstance(state, Mapping):
        raise ValueError("resume RNG state must be a mapping")
    required = {"python", "numpy", "torch", "cuda"}
    if set(state) != required:
        raise ValueError(f"resume local RNG state keys must be exactly {sorted(required)}")
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    cpu_state = state["torch"]
    if not isinstance(cpu_state, Tensor):
        raise ValueError("resume CPU Torch RNG state must be a Tensor")
    torch.set_rng_state(cpu_state)
    cuda_state = state["cuda"]
    if torch.cuda.is_available():
        if not isinstance(cuda_state, (tuple, list)):
            raise ValueError("resume CUDA RNG state must be a sequence on CUDA")
        current_cuda_count = torch.cuda.device_count()
        if len(cuda_state) != current_cuda_count:
            raise ValueError(
                "resume CUDA RNG state device count does not match the current process: "
                f"saved={len(cuda_state)}, current={current_cuda_count}"
            )
        if not all(isinstance(item, Tensor) for item in cuda_state):
            raise ValueError("resume CUDA RNG state entries must be Tensors")
        torch.cuda.set_rng_state_all(cuda_state)  # type: ignore[arg-type]
    elif cuda_state is not None:
        raise ValueError("resume CUDA RNG state is present but CUDA is unavailable")


def _gather_rank_rng_state() -> dict[str, object]:
    """Gather local RNG payloads from every rank into one indexed structure."""

    rank, world_size = _current_rank_world_size()
    local = capture_rng_state()
    if world_size == 1:
        return {
            "schema": TRAIN_RESUME_RNG_SCHEMA,
            "world_size": 1,
            "states": {0: local},
        }
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(
            "cannot gather rank-indexed resume RNG state without an initialized process group"
        )
    gathered: list[object | None] = [None] * world_size
    torch.distributed.all_gather_object(gathered, local)
    states: dict[int, object] = {}
    for index, value in enumerate(gathered):
        if not isinstance(value, Mapping):
            raise ValueError(f"resume RNG gather returned no mapping for rank {index}")
        states[index] = dict(value)
    # ``rank`` is intentionally read above even though all ranks receive the
    # same gathered object; this catches an invalid rank entry early and makes
    # the rank binding visible in traces/debuggers.
    if rank not in states:
        raise ValueError(f"resume RNG gather has no current rank entry: {rank}")
    return {
        "schema": TRAIN_RESUME_RNG_SCHEMA,
        "world_size": world_size,
        "states": states,
    }


def _validate_rank_rng_payload(payload: Mapping[str, object]) -> Mapping[str, object]:
    """Validate rank/world binding without mutating any process RNG."""

    if set(payload) != {"schema", "world_size", "states"}:
        raise ValueError("resume RNG payload has an unsupported schema")
    if payload["schema"] != TRAIN_RESUME_RNG_SCHEMA:
        raise ValueError("resume RNG payload has an unsupported schema")
    saved_world_size = payload["world_size"]
    if not isinstance(saved_world_size, int) or isinstance(saved_world_size, bool) or saved_world_size <= 0:
        raise ValueError("resume RNG payload world_size must be a positive integer")
    states = payload["states"]
    if not isinstance(states, Mapping):
        raise ValueError("resume RNG payload states must be a mapping")
    rank, current_world_size = _current_rank_world_size()
    if saved_world_size != current_world_size:
        raise ValueError(
            "resume RNG world size mismatch: "
            f"saved={saved_world_size}, current={current_world_size}"
        )
    expected_ranks = set(range(saved_world_size))
    actual_ranks: set[int] = set()
    for key in states:
        if isinstance(key, bool) or not isinstance(key, int):
            raise ValueError("resume RNG payload rank entries must be integer indices")
        actual_ranks.add(key)
    if actual_ranks != expected_ranks:
        raise ValueError(
            "resume RNG payload rank entries do not match world size: "
            f"expected={sorted(expected_ranks)}, got={sorted(actual_ranks)}"
        )
    local = states.get(rank)
    if not isinstance(local, Mapping):
        raise ValueError(f"resume RNG payload has no mapping for current rank {rank}")
    # Validate the complete local shape without consuming or changing streams.
    required_local = {"python", "numpy", "torch", "cuda"}
    if set(local) != required_local:
        raise ValueError(f"resume local RNG state keys must be exactly {sorted(required_local)}")
    if not isinstance(local["torch"], Tensor):
        raise ValueError("resume CPU Torch RNG state must be a Tensor")
    cuda_state = local["cuda"]
    if torch.cuda.is_available():
        if not isinstance(cuda_state, (tuple, list)):
            raise ValueError("resume CUDA RNG state must be a sequence on CUDA")
        if len(cuda_state) != torch.cuda.device_count() or not all(isinstance(item, Tensor) for item in cuda_state):
            raise ValueError("resume CUDA RNG state entries do not match current devices")
    elif cuda_state is not None:
        raise ValueError("resume CUDA RNG state is present but CUDA is unavailable")
    return local


def _restore_rank_rng_state(payload: Mapping[str, object]) -> None:
    """Validate the rank/world binding and restore only this rank's state."""

    local = _validate_rank_rng_payload(payload)
    restore_rng_state(local)


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
    protocol: Mapping[str, Any] | None = None,
    patience_count: int = 0,
    first_train_reconstruction_loss: float | None = None,
    overfit_prediction_epochs: tuple[int, ...] | list[int] = (),
    run_state: Mapping[str, Any] | None = None,
) -> Path:
    """Write the resumable checkpoint atomically."""

    if not isinstance(training_config, Mapping) or not isinstance(metadata, Mapping):
        raise TypeError("training_config and metadata must be mappings")
    if protocol is None:
        # Keep the low-level helper usable by focused synthetic callers that do
        # not construct a full model protocol.  The production trainer always
        # supplies ``build_training_resume_protocol`` output and validates it
        # on load.
        protocol = {
            "version": TRAIN_RESUME_PROTOCOL_VERSION,
            "immutable": {
                **{field: None for field in TRAIN_RESUME_IMMUTABLE_FIELDS},
                "split_hash": str(split_hash),
            },
            "compatible": {field: None for field in TRAIN_RESUME_COMPATIBLE_FIELDS},
        }
    protocol = _protocol_shape(protocol)
    if not isinstance(patience_count, int) or isinstance(patience_count, bool) or patience_count < 0:
        raise ValueError("patience_count must be a non-negative integer")
    if first_train_reconstruction_loss is not None:
        first_train_reconstruction_loss = float(first_train_reconstruction_loss)
        if not math.isfinite(first_train_reconstruction_loss):
            raise ValueError("first_train_reconstruction_loss must be finite or null")
    prediction_epochs = tuple(int(epoch) for epoch in overfit_prediction_epochs)
    if any(epoch < 0 for epoch in prediction_epochs):
        raise ValueError("overfit_prediction_epochs must contain non-negative integers")
    state_values: dict[str, Any] = {
        "patience_count": patience_count,
        "first_train_reconstruction_loss": first_train_reconstruction_loss,
        "overfit_prediction_epochs": prediction_epochs,
    }
    if run_state is not None:
        if not isinstance(run_state, Mapping) or set(run_state) != set(TRAIN_RESUME_RUN_STATE_FIELDS):
            raise ValueError(
                "resume run_state keys must be exactly "
                f"{list(TRAIN_RESUME_RUN_STATE_FIELDS)}"
            )
        state_values = {
            "patience_count": run_state["patience_count"],
            "first_train_reconstruction_loss": run_state["first_train_reconstruction_loss"],
            "overfit_prediction_epochs": tuple(run_state["overfit_prediction_epochs"]),
        }
        if not isinstance(state_values["patience_count"], int) or isinstance(state_values["patience_count"], bool) or state_values["patience_count"] < 0:
            raise ValueError("resume run_state patience_count must be a non-negative integer")
        if state_values["first_train_reconstruction_loss"] is not None:
            state_values["first_train_reconstruction_loss"] = float(state_values["first_train_reconstruction_loss"])
            if not math.isfinite(state_values["first_train_reconstruction_loss"]):
                raise ValueError("resume run_state first_train_reconstruction_loss must be finite or null")
        if any(not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0 for epoch in state_values["overfit_prediction_epochs"]):
            raise ValueError("resume run_state overfit_prediction_epochs must contain non-negative integers")
    rng_state = _gather_rank_rng_state()
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
        "protocol": protocol,
        "rng_state": rng_state,
        "run_state": state_values,
        "metadata": dict(metadata),
    }
    rank, _ = _current_rank_world_size()
    # Every rank participates in the RNG gather, but only the artifact owner
    # writes the shared checkpoint.  This preserves one-writer semantics while
    # retaining rank-local streams for an exact distributed resume.
    if rank == 0:
        return _atomic_torch_save(payload, path)
    return Path(path)


def load_training_resume_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    expected_split_hash: str,
    expected_protocol: Mapping[str, Any] | None = None,
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
        "protocol",
        "rng_state",
        "run_state",
        "metadata",
    }
    if set(payload) != required:
        raise ValueError(f"resume checkpoint keys must be exactly {sorted(required)}")
    if payload["split_hash"] != expected_split_hash:
        raise ValueError("resume checkpoint split hash does not match the current split")
    protocol = payload["protocol"]
    if not isinstance(protocol, Mapping):
        raise ValueError("resume checkpoint protocol must be a mapping")
    protocol = _protocol_shape(protocol)
    if protocol["immutable"]["split_hash"] != expected_split_hash:
        raise ValueError("resume protocol split hash does not match the current split")
    if expected_protocol is not None:
        validate_training_resume_protocol(protocol, expected_protocol)
    run_state = payload["run_state"]
    if not isinstance(run_state, Mapping) or set(run_state) != set(TRAIN_RESUME_RUN_STATE_FIELDS):
        raise ValueError(
            "resume checkpoint run_state keys must be exactly "
            f"{list(TRAIN_RESUME_RUN_STATE_FIELDS)}"
        )
    patience_count = run_state["patience_count"]
    if not isinstance(patience_count, int) or isinstance(patience_count, bool) or patience_count < 0:
        raise ValueError("resume checkpoint patience_count must be a non-negative integer")
    first_train_loss = run_state["first_train_reconstruction_loss"]
    if first_train_loss is not None and (
        not isinstance(first_train_loss, (int, float))
        or isinstance(first_train_loss, bool)
        or not math.isfinite(float(first_train_loss))
    ):
        raise ValueError("resume checkpoint first_train_reconstruction_loss must be finite or null")
    prediction_epochs = run_state["overfit_prediction_epochs"]
    if not isinstance(prediction_epochs, (tuple, list)) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in prediction_epochs
    ):
        raise ValueError("resume checkpoint overfit_prediction_epochs must contain non-negative integers")
    rng_state = payload["rng_state"]
    if not isinstance(rng_state, Mapping):
        raise ValueError("resume checkpoint rng_state must be a mapping")
    _validate_rank_rng_payload(rng_state)
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
    # Restore only after the protocol and live state have been accepted.  An
    # incompatible request therefore cannot partially mutate the model or
    # optimizer before failing.
    _restore_rank_rng_state(rng_state)
    return {
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "best_validation_reconstruction_loss": float(payload["best_validation_reconstruction_loss"]),
        "training_config": payload["training_config"],
        "protocol": protocol,
        "run_state": {
            "patience_count": int(patience_count),
            "first_train_reconstruction_loss": None if first_train_loss is None else float(first_train_loss),
            "overfit_prediction_epochs": tuple(prediction_epochs),
        },
        "patience_count": int(patience_count),
        "first_train_reconstruction_loss": None if first_train_loss is None else float(first_train_loss),
        "overfit_prediction_epochs": tuple(prediction_epochs),
        "metadata": payload["metadata"],
    }


def save_clean_inference_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    split_hash: str,
) -> Path:
    """Write the exact strict baseline inference payload, atomically."""

    return _atomic_torch_save(
        {
            "metadata": baseline_checkpoint_metadata(model, split_hash=split_hash),
            "state_dict": model.state_dict(),
        },
        path,
    )


def validate_clean_inference_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, object],
) -> None:
    """Preflight a clean checkpoint state before mutating the live model.

    ``Module.load_state_dict(strict=True)`` is not transactional: it can copy
    compatible entries before reporting a later missing, unexpected, shape, or
    dtype mismatch.  Validate the complete state mapping up front so that the
    strict load below can only run for a fully compatible payload.
    """

    if not isinstance(state_dict, Mapping):
        raise ValueError("baseline checkpoint state_dict must be a mapping")
    if not all(isinstance(name, str) for name in state_dict):
        raise ValueError("baseline checkpoint state_dict keys must be strings")

    live_state = model.state_dict()
    live_names = set(live_state)
    checkpoint_names = set(state_dict)
    missing = sorted(live_names - checkpoint_names)
    unexpected = sorted(checkpoint_names - live_names)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing keys: {missing}")
        if unexpected:
            details.append(f"unexpected keys: {unexpected}")
        raise ValueError("baseline checkpoint state_dict key mismatch (" + "; ".join(details) + ")")

    for name, live_tensor in live_state.items():
        checkpoint_tensor = state_dict[name]
        if not isinstance(checkpoint_tensor, Tensor):
            raise ValueError(f"baseline checkpoint tensor for {name!r} must be a Tensor")
        if checkpoint_tensor.shape != live_tensor.shape:
            raise ValueError(
                f"baseline checkpoint tensor shape mismatch for {name!r}: "
                f"expected {tuple(live_tensor.shape)}, got {tuple(checkpoint_tensor.shape)}"
            )
        if checkpoint_tensor.dtype != live_tensor.dtype:
            raise ValueError(
                f"baseline checkpoint tensor dtype mismatch for {name!r}: "
                f"expected {live_tensor.dtype}, got {checkpoint_tensor.dtype}"
            )


__all__ = [
    "RESUME_COMPATIBLE_FIELDS",
    "RESUME_IMMUTABLE_FIELDS",
    "TRAIN_RESUME_SCHEMA",
    "TRAIN_RESUME_COMPATIBLE_FIELDS",
    "TRAIN_RESUME_IMMUTABLE_FIELDS",
    "TRAIN_RESUME_PROTOCOL_VERSION",
    "TRAIN_RESUME_RNG_SCHEMA",
    "TRAIN_RESUME_RUN_STATE_FIELDS",
    "build_training_resume_protocol",
    "capture_rng_state",
    "load_training_resume_checkpoint",
    "restore_rng_state",
    "save_clean_inference_checkpoint",
    "save_training_resume_checkpoint",
    "validate_training_resume_protocol",
    "validate_clean_inference_state_dict",
]
