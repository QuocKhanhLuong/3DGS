"""Signed cached ValueNet fitting for PFGR-Lite.

Only detached rows from :mod:`value_bank` enter this module.  Fitting is a
plain bounded PyTorch regression loop over one immutable bank; no MRI loader,
target volume, updater, decoder, or teacher is imported or called.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import math
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from .config import ValueModelConfig
from .provenance import canonical_digest, module_state_digest
from .provenance import ValueFitIdentity
from .types import RESUME_SCHEMA, StageState
from .value_bank import GainScale, ValueBankReader, ValueBankRow, compute_gain_scale


# Value fitting does not mint a competing checkpoint protocol.  W4 wraps this
# stage payload inside the authoritative PFGR ResumeState envelope.
VALUE_FIT_RESUME_SCHEMA = RESUME_SCHEMA
MAX_EPOCHS = 10000
MAX_BATCH_SIZE = 1_000_000
_FORBIDDEN_IMPORT_PREFIXES = (
    "smagm.features.point_guided.pfgr_lite.teacher",
    "smagm.features.point_guided.pfgr_lite.data",
    "smagm.features.point_guided.pfgr_lite.updater",
    "smagm.features.point_guided.pfgr_lite.decoder",
    "smagm.features.point_guided.pfgr_lite.sparse_write",
    "smagm.features.point_guided.pfgr_lite.objectives",
    "smagm.features.point_guided.pfgr_lite.model",
)


@dataclass
class CachedFitCallCounters:
    """Auditable dependency-injection counters for the cached-only boundary."""

    mri_loader_calls: int = 0
    teacher_calls: int = 0
    updater_calls: int = 0
    decoder_calls: int = 0
    target_volume_reads: int = 0
    hypothetical_writes: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "mri_loader_calls": self.mri_loader_calls,
            "teacher_calls": self.teacher_calls,
            "updater_calls": self.updater_calls,
            "decoder_calls": self.decoder_calls,
            "target_volume_reads": self.target_volume_reads,
            "hypothetical_writes": self.hypothetical_writes,
        }


def _variant(value: int | str) -> int:
    if isinstance(value, str):
        text = value.lower().strip()
        if text.startswith("v"):
            text = text[1:]
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError("input_variant must be 126, 222, 270, or 366") from exc
    if value not in (126, 222, 270, 366):
        raise ValueError("input_variant must be 126, 222, 270, or 366")
    return int(value)


def _positive_int(name: str, value: int, *, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds bound {maximum}")
    return value


def _finite(name: str, value: float, *, nonnegative: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0.0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return value


def _clone_cpu(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _clone_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cpu(item) for item in value)
    return copy.deepcopy(value)


def _new_forbidden_imports(before: set[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name in set(sys.modules) - before
            if any(name == prefix or name.startswith(prefix + ".") for prefix in _FORBIDDEN_IMPORT_PREFIXES)
        )
    )


class SignedValueNet(nn.Module):
    """The locked signed MLP: ``d -> 128 -> SiLU -> 64 -> SiLU -> 1``."""

    def __init__(self, input_variant: int | str = 366, hidden_channels: Sequence[int] = (128, 64)) -> None:
        super().__init__()
        self.input_variant = _variant(input_variant)
        hidden = tuple(int(item) for item in hidden_channels)
        if hidden != (128, 64):
            raise ValueError("ValueNet hidden widths are locked to (128, 64)")
        self.hidden_channels = hidden
        self.net = nn.Sequential(
            nn.Linear(self.input_variant, hidden[0], bias=True),
            nn.SiLU(),
            nn.Linear(hidden[0], hidden[1], bias=True),
            nn.SiLU(),
            nn.Linear(hidden[1], 1, bias=True),
        )

    @property
    def architecture_hash(self) -> str:
        return canonical_digest(
            {
                "schema_version": "pfgr-lite-value-net-architecture-v1",
                "input_variant": self.input_variant,
                "hidden_channels": self.hidden_channels,
                "activation": "SiLU",
                "output": "signed_linear",
            },
            prefix="pfgr-lite-value-net-architecture-v1|",
        )

    def forward(self, descriptors: Tensor) -> Tensor:
        if not isinstance(descriptors, Tensor):
            raise TypeError("descriptors must be a torch.Tensor")
        if descriptors.ndim < 2 or descriptors.shape[-1] != self.input_variant:
            raise ValueError(f"descriptors must have final dimension {self.input_variant}")
        if not descriptors.is_floating_point() or not bool(torch.isfinite(descriptors).all()):
            raise ValueError("descriptors must be finite floating values")
        return self.net(descriptors)


ValueNet = SignedValueNet


@dataclass(frozen=True)
class ValueFitOptions:
    """Operational fitting options; W1's ``ValueModelConfig`` remains canonical."""

    epochs: int = 1
    batch_size: int = 32
    seed: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    loss: str = "mse"
    shuffle: bool = True
    max_updates: int | None = None
    robust_ablation: bool = False

    def __post_init__(self) -> None:
        _positive_int("epochs", self.epochs, maximum=MAX_EPOCHS)
        _positive_int("batch_size", self.batch_size, maximum=MAX_BATCH_SIZE)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        _finite("learning_rate", self.learning_rate)
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        _finite("weight_decay", self.weight_decay, nonnegative=True)
        if self.loss not in {"mse", "smooth_l1"}:
            raise ValueError("loss must be mse or smooth_l1")
        if self.loss == "smooth_l1" and not self.robust_ablation:
            raise ValueError("smooth_l1 is available only as an explicit robust_ablation")
        if self.max_updates is not None:
            _positive_int("max_updates", self.max_updates)

    def as_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "device": self.device,
            "loss": self.loss,
            "shuffle": self.shuffle,
            "max_updates": self.max_updates,
            "robust_ablation": self.robust_ablation,
        }


ValueFitConfig = ValueFitOptions


@dataclass
class ValueFitResult:
    model: SignedValueNet
    input_variant: int
    gain_scale: GainScale
    identity: ValueFitIdentity
    history: list[dict[str, Any]]
    metrics: dict[str, Any]
    resume_state: dict[str, Any]
    complete: bool
    latency_seconds: float
    bank_manifest_hash: str
    fit_options: ValueFitOptions
    stage_state: StageState | None = None

    @property
    def fit_identity(self) -> ValueFitIdentity:
        return self.identity

    @property
    def value_net(self) -> SignedValueNet:
        return self.model

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_variant": self.input_variant,
            "gain_scale": self.gain_scale.as_dict(),
            "identity": self.identity,
            "history": copy.deepcopy(self.history),
            "metrics": copy.deepcopy(self.metrics),
            "resume_state": _clone_cpu(self.resume_state),
            "complete": self.complete,
            "latency_seconds": self.latency_seconds,
            "bank_manifest_hash": self.bank_manifest_hash,
            "fit_options": self.fit_options.as_dict(),
            "stage_state": self.stage_state,
        }


@dataclass(frozen=True)
class ValueEvalResult:
    metrics: Mapping[str, Any]
    row_count: int
    input_variant: int
    split_roles: tuple[str, ...] | None
    latency_seconds: float

    def __getitem__(self, key: str) -> Any:
        return self.metrics[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.metrics.get(key, default)


def _reader_parts(bank: Any, *, allow_diagnostic: bool = False) -> tuple[tuple[ValueBankRow, ...], Any, GainScale, str]:
    reader = bank if isinstance(bank, ValueBankReader) else ValueBankReader(bank)
    index = getattr(reader, "index", {})
    status = index.get("status", {}).get("evidence_status") if isinstance(index, Mapping) and isinstance(index.get("status", {}), Mapping) else None
    if status == "DIAGNOSTIC_ONLY" and not allow_diagnostic:
        raise ValueError("diagnostic-only bank is not a MAIN ValueNet fit input")
    if hasattr(reader, "manifest"):
        manifest_attr = reader.manifest
        manifest = manifest_attr() if callable(manifest_attr) else manifest_attr
    else:
        manifest = getattr(reader, "_manifest", None)
    if manifest is None:
        raise TypeError("bank reader must expose manifest()")
    scale = getattr(reader, "gain_scale", None)
    if scale is None:
        scale = GainScale(scale=float(manifest.gain_scale), training_row_count=0)
    rows_attr = reader.rows
    rows = rows_attr() if callable(rows_attr) else rows_attr
    rows = tuple(rows)
    manifest_hash = str(getattr(reader, "manifest_hash", ""))
    if not manifest_hash:
        manifest_hash = canonical_digest(
            {
                "producer": manifest.producer_compatibility_hash,
                "label": manifest.label_definition_hash,
                "split": manifest.split_role_hash,
                "scale": manifest.gain_scale_hash,
                "rows": len(rows),
            },
            prefix="pfgr-lite-value-bank-manifest-v1|",
        )
    return rows, manifest, scale, manifest_hash


def _fit_hash(options: ValueFitOptions, model: SignedValueNet) -> str:
    # Epoch count, update cut-offs and device are operational continuation
    # choices; changing them must not invalidate an otherwise identical
    # interrupted run.  Batch/shuffle/optimizer/loss settings do affect the
    # numerical trajectory and therefore remain identity fields.
    identity_options = {
        "batch_size": options.batch_size,
        "seed": options.seed,
        "learning_rate": options.learning_rate,
        "weight_decay": options.weight_decay,
            "loss": options.loss,
            "shuffle": options.shuffle,
            "robust_ablation": options.robust_ablation,
            "optimizer": _expected_optimizer_provenance(options),
    }
    return canonical_digest(
        {
            "schema_version": "pfgr-lite-value-fit-config-v1",
            "model_architecture": model.architecture_hash,
            "options": identity_options,
        },
        prefix="pfgr-lite-value-fit-config-v1|",
    )


def _expected_optimizer_provenance(options: ValueFitOptions) -> dict[str, Any]:
    """Canonical optimizer declaration owned by the V-only fit stage."""

    return {
        "class": "torch.optim.Adam",
        "groups": [
            {
                "name": "value_net",
                "lr": float(options.learning_rate),
                "weight_decay": float(options.weight_decay),
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "amsgrad": False,
                "maximize": False,
            }
        ],
    }


def _actual_optimizer_provenance(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    if optimizer.__class__ is not torch.optim.Adam:
        raise ValueError("ValueNet fitting requires torch.optim.Adam")
    groups: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        groups.append(
            {
                "name": group.get("name"),
                "lr": float(group.get("lr")),
                "weight_decay": float(group.get("weight_decay")),
                "betas": [float(item) for item in group.get("betas", (0.9, 0.999))],
                "eps": float(group.get("eps", 1e-8)),
                "amsgrad": bool(group.get("amsgrad", False)),
                "maximize": bool(group.get("maximize", False)),
            }
        )
    return {"class": "torch.optim.Adam", "groups": groups}


def _metric_summary(rows: Sequence[ValueBankRow], predicted_scaled: Tensor, scale: GainScale) -> dict[str, Any]:
    if len(rows) != predicted_scaled.shape[0]:
        raise ValueError("prediction/row count mismatch")
    true = torch.tensor([float(row.raw_gain) for row in rows], dtype=torch.float64)
    pred = predicted_scaled.detach().to(dtype=torch.float64).reshape(-1) * float(scale.scale)
    if not bool(torch.isfinite(pred).all()):
        raise ValueError("predicted raw gains are nonfinite")
    raw_error = (pred - true) ** 2
    raw_mse_sum = float(raw_error.sum().item()) if rows else 0.0
    raw_mse = raw_mse_sum / len(rows) if rows else None
    scaled_true = true / float(scale.scale)
    scaled_pred = pred / float(scale.scale)
    scaled_error = (scaled_pred - scaled_true) ** 2
    scaled_mse_sum = float(scaled_error.sum().item()) if rows else 0.0
    scaled_mse = scaled_mse_sum / len(rows) if rows else None
    sign_defined = (true != 0.0) & (pred != 0.0)
    sign_correct = (torch.sign(true) == torch.sign(pred)) & sign_defined
    sign_ties = int((~sign_defined).sum().item())
    sign_defined_count = int(sign_defined.sum().item())
    sign_correct_count = int(sign_correct.sum().item())
    sign_accuracy = (sign_correct_count / sign_defined_count) if sign_defined_count else None
    groups: dict[tuple[str, str, int], list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault((row.subject_key, row.context_id, int(row.state_version)), []).append(index)
    regrets: list[float] = []
    group_ties = 0
    stop_count = 0
    for indices in groups.values():
        if not indices:
            continue
        sorted_indices = sorted(indices, key=lambda idx: (-float(pred[idx].item()), rows[idx].point_id, rows[idx].action_id))
        max_pred = float(pred[sorted_indices[0]].item())
        top_candidates = [idx for idx in indices if float(pred[idx].item()) == max_pred]
        if len(top_candidates) > 1:
            group_ties += 1
        best = max(float(true[idx].item()) for idx in indices)
        if max_pred <= 0.0:
            stop_count += 1
            selected_true = 0.0
        else:
            chosen = sorted(top_candidates, key=lambda idx: (rows[idx].point_id, rows[idx].action_id))[0]
            selected_true = float(true[chosen].item())
        regrets.append(max(0.0, best) - selected_true)
    top1 = float(sum(regrets) / len(regrets)) if regrets else None
    top1_regret_sum = float(sum(regrets)) if regrets else 0.0
    pairs = 0
    concordant = 0
    ranking_ties = 0
    for indices in groups.values():
        for left_pos, left in enumerate(indices):
            for right in indices[left_pos + 1 :]:
                true_delta = float(true[left] - true[right])
                pred_delta = float(pred[left] - pred[right])
                if true_delta == 0.0 or pred_delta == 0.0:
                    ranking_ties += 1
                    continue
                pairs += 1
                if true_delta * pred_delta > 0.0:
                    concordant += 1
    return {
        "row_count": len(rows),
        "mse_scaled": scaled_mse,
        "mse_scaled_sum": scaled_mse_sum,
        "mse_scaled_count": len(rows),
        "mse_raw": raw_mse,
        "mse_raw_sum": raw_mse_sum,
        "mse_raw_count": len(rows),
        "sign_accuracy": sign_accuracy,
        "sign_correct_count": sign_correct_count,
        "sign_defined_count": sign_defined_count,
        "sign_tie_count": sign_ties,
        "ranking_concordance": (concordant / pairs) if pairs else None,
        "ranking_concordant_count": concordant,
        "ranking_pair_count": pairs,
        "ranking_tie_count": ranking_ties,
        "top1_subset_regret_raw": top1,
        "top1_subset_regret_sum_raw": top1_regret_sum,
        "top1_subset_regret_count": len(regrets),
        "top1_subset_regret_scope": "same_subject_context_state_rows",
        "top1_subset_count": len(regrets),
        "top1_subset_tie_groups": group_ties,
        "top1_subset_stop_count": stop_count,
        "top1_subset_stop_rule": "stop_when_best_predicted_raw_gain_leq_zero",
        "top1_subset_tie_rule": "lowest_point_id_then_action_id",
        "top1_subset_undefined_reason": None if regrets else "no grouped actions",
        "raw_gain_signed": True,
    }


def _resolve_options(
    config: ValueModelConfig | Mapping[str, Any] | ValueFitOptions | None,
    *,
    epochs: int | None,
    batch_size: int | None,
    seed: int | None,
    device: str | None,
    learning_rate: float | None,
    weight_decay: float | None,
    loss: str | None,
    shuffle: bool | None,
    max_updates: int | None,
    robust_ablation: bool,
) -> tuple[ValueModelConfig, ValueFitOptions]:
    if config is None:
        model_config = ValueModelConfig()
        options = ValueFitOptions(
            epochs=epochs if epochs is not None else 1,
            batch_size=batch_size if batch_size is not None else 32,
            seed=seed if seed is not None else 0,
            device=device or "cpu",
            learning_rate=learning_rate if learning_rate is not None else 1e-3,
            weight_decay=weight_decay if weight_decay is not None else 0.0,
            loss=loss or "mse",
            shuffle=True if shuffle is None else shuffle,
            max_updates=max_updates,
            robust_ablation=robust_ablation,
        )
    elif isinstance(config, ValueFitOptions):
        model_config = ValueModelConfig()
        base = config.as_dict()
        options = ValueFitOptions(
            **{
                **base,
                "epochs": epochs if epochs is not None else base["epochs"],
                "batch_size": batch_size if batch_size is not None else base["batch_size"],
                "seed": seed if seed is not None else base["seed"],
                "device": device if device is not None else base["device"],
                "learning_rate": learning_rate if learning_rate is not None else base["learning_rate"],
                "weight_decay": weight_decay if weight_decay is not None else base["weight_decay"],
                "loss": loss if loss is not None else base["loss"],
                "shuffle": shuffle if shuffle is not None else base["shuffle"],
                "max_updates": max_updates if max_updates is not None else base["max_updates"],
                "robust_ablation": robust_ablation or bool(base["robust_ablation"]),
            }
        )
    else:
        model_config = ValueModelConfig.from_dict(config) if isinstance(config, Mapping) else config
        if not isinstance(model_config, ValueModelConfig):
            raise TypeError("config must be ValueModelConfig, ValueFitOptions, mapping, or None")
        options = ValueFitOptions(
            epochs=epochs if epochs is not None else 1,
            batch_size=batch_size if batch_size is not None else 32,
            seed=seed if seed is not None else 0,
            device=device or "cpu",
            learning_rate=learning_rate if learning_rate is not None else 1e-3,
            weight_decay=weight_decay if weight_decay is not None else 0.0,
            loss=loss or model_config.loss,
            shuffle=True if shuffle is None else shuffle,
            max_updates=max_updates,
            robust_ablation=robust_ablation,
        )
    if options.loss == "smooth_l1" and model_config.loss != "mse":
        raise ValueError("unknown main ValueModelConfig loss")
    return model_config, options


def _validate_optimizer(optimizer: torch.optim.Optimizer, model: nn.Module, options: ValueFitOptions) -> dict[str, Any]:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    owned = {id(parameter) for parameter in model_parameters}
    seen: list[int] = []
    for group in optimizer.param_groups:
        for parameter in group.get("params", ()):
            if not isinstance(parameter, nn.Parameter):
                raise TypeError("optimizer params must be torch.nn.Parameter")
            seen.append(id(parameter))
    if set(seen) != owned or len(seen) != len(owned):
        raise ValueError("optimizer ownership must be exactly ValueNet trainable parameters")
    actual = _actual_optimizer_provenance(optimizer)
    expected = _expected_optimizer_provenance(options)
    if len(actual["groups"]) != 1 or actual["groups"][0].get("name") != "value_net":
        raise ValueError("optimizer ownership requires one named value_net parameter group")
    if actual != expected:
        raise ValueError("optimizer hyperparameters/class do not match ValueFitOptions")
    return actual


def _predict_rows(
    model: SignedValueNet,
    rows: Sequence[ValueBankRow],
    variant: int,
    *,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    """Predict in bounded row batches; never materialize a full bank on device."""

    values: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            if not chunk:
                continue
            model_parameter = next(model.parameters())
            descriptors = torch.stack([getattr(row, f"v{variant}") for row in chunk]).to(device=device, dtype=model_parameter.dtype)
            values.append(model(descriptors).detach().cpu().reshape(-1))
    if not values:
        return torch.empty((0,), dtype=torch.float32)
    return torch.cat(values, dim=0)


def _resume_dict(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, ValueFitResult):
        return value.resume_state
    if isinstance(value, Mapping):
        return value
    raise TypeError("resume must be ValueFitResult or mapping")


def fit_value(
    bank_reader: ValueBankReader | str,
    value_model: SignedValueNet | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    config: ValueModelConfig | ValueFitOptions | Mapping[str, Any] | None = None,
    *,
    stage_state: StageState | None = None,
    input_variant: int | str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    device: str | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    loss: str | None = None,
    shuffle: bool | None = None,
    max_updates: int | None = None,
    robust_ablation: bool = False,
    resume: ValueFitResult | Mapping[str, Any] | None = None,
) -> ValueFitResult:
    """Fit one signed ValueNet on an immutable, already measured bank."""

    import_snapshot = set(sys.modules)
    model_config, options = _resolve_options(
        config,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        loss=loss,
        shuffle=shuffle,
        max_updates=max_updates,
        robust_ablation=robust_ablation,
    )
    if stage_state is not None:
        if not isinstance(stage_state, StageState) or stage_state.stage != "value_fit":
            raise ValueError("stage_state must be the authoritative value_fit StageState")
    rows, manifest, scale, manifest_hash = _reader_parts(bank_reader)
    training_role = scale.training_role
    train_rows = tuple(row for row in rows if row.split_role == training_role and not row.diagnostic)
    if not train_rows:
        raise ValueError(f"value fit requires nonempty training role {training_role!r}")
    variant = _variant(input_variant if input_variant is not None else getattr(value_model, "input_variant", 366))
    if value_model is None:
        # Seed construction as well as the sampler.  ``fork_rng`` prevents a
        # cached fit from perturbing the caller's global RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(options.seed)
            value_model = SignedValueNet(variant, hidden_channels=model_config.hidden_channels)
    if not isinstance(value_model, SignedValueNet):
        raise TypeError("value_model must be SignedValueNet")
    if value_model.input_variant != variant:
        raise ValueError("value_model/input_variant mismatch")
    if tuple(value_model.hidden_channels) != tuple(model_config.hidden_channels):
        raise ValueError("value_model hidden widths do not match ValueModelConfig")
    target_device = torch.device(options.device)
    value_model.to(target_device)
    model_dtype = next(value_model.parameters()).dtype
    if model_dtype not in (torch.float32, torch.float64):
        raise TypeError("ValueNet fitting supports FP32 production or FP64 test dtype")
    if optimizer is None:
        optimizer = torch.optim.Adam(
            [{"params": list(value_model.parameters()), "name": "value_net"}],
            lr=options.learning_rate,
            weight_decay=options.weight_decay,
        )
    optimizer_provenance = _validate_optimizer(optimizer, value_model, options)
    fit_hash = _fit_hash(options, value_model)
    train_y = torch.tensor([float(row.raw_gain) / float(scale.scale) for row in train_rows], dtype=model_dtype)
    if not bool(torch.isfinite(train_y).all()):
        raise ValueError("bank training data must be finite")
    for row in train_rows:
        descriptor = getattr(row, f"v{variant}")
        if not bool(torch.isfinite(descriptor).all()):
            raise ValueError("bank training data must be finite")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(options.seed)
    initial_parameter_snapshots = {
        name: parameter.detach().cpu().clone()
        for name, parameter in value_model.named_parameters()
        if parameter.requires_grad
    }
    initial_weights_hash = module_state_digest(value_model)
    history: list[dict[str, Any]] = []
    resume_payload = _resume_dict(resume)
    epoch = 0
    cursor = 0
    order: list[int] | None = None
    update_count = 0
    epoch_loss_sum = 0.0
    epoch_loss_rows = 0
    gradient_l2_norm_sum = 0.0
    gradient_l2_norm_count = 0
    gradient_l2_norm_max = 0.0
    gradient_nonzero_step_count = 0
    if resume_payload is not None:
        if resume_payload.get("schema_version") != VALUE_FIT_RESUME_SCHEMA or resume_payload.get("protocol", RESUME_SCHEMA) != RESUME_SCHEMA:
            raise ValueError("unknown value-fit resume schema")
        stage_payload = resume_payload.get("stage_payload")
        if stage_payload is not None and not isinstance(stage_payload, Mapping):
            raise ValueError("resume stage_payload must be a mapping")
        for key, expected in (("bank_manifest_hash", manifest_hash), ("gain_scale_hash", scale.digest), ("input_variant", variant), ("fit_config_hash", fit_hash)):
            if resume_payload.get(key) != expected:
                raise ValueError(f"resume {key} mismatch")
        if resume_payload.get("producer_compatibility_hash") != manifest.producer_compatibility_hash:
            raise ValueError("resume producer compatibility mismatch")
        bank_state = resume_payload.get("bank_state")
        if bank_state is not None:
            if not isinstance(bank_state, Mapping) or bank_state.get("manifest_hash") != manifest_hash or bank_state.get("gain_scale_hash") != scale.digest or bank_state.get("split_role_hash") != manifest.split_role_hash:
                raise ValueError("resume bank_state dependency mismatch")
        saved_optimizer_provenance = resume_payload.get("optimizer_provenance")
        if saved_optimizer_provenance != optimizer_provenance:
            raise ValueError("resume optimizer provenance mismatch")
        state_dict = resume_payload.get("model_state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("resume missing model_state_dict")
        value_model.load_state_dict(state_dict, strict=True)
        initial_parameter_snapshots = {
            name: parameter.detach().cpu().clone()
            for name, parameter in value_model.named_parameters()
            if parameter.requires_grad
        }
        saved_initial_hash = resume_payload.get("initial_weights_hash")
        if saved_initial_hash and not isinstance(saved_initial_hash, str):
            raise ValueError("resume initial_weights_hash must be text")
        if isinstance(saved_initial_hash, str) and saved_initial_hash:
            initial_weights_hash = saved_initial_hash
        optimizer_state = resume_payload.get("optimizer_state", resume_payload.get("optimizer_state_dict"))
        if not isinstance(optimizer_state, Mapping):
            raise ValueError("resume missing optimizer_state_dict")
        optimizer.load_state_dict(optimizer_state)
        stage_values = stage_payload if stage_payload is not None else resume_payload
        epoch = int(stage_values.get("epoch", 0))
        cursor = int(stage_values.get("cursor", 0))
        order_value = stage_values.get("order")
        order = None if order_value is None else [int(item) for item in order_value]
        update_count = int(stage_values.get("update_count", 0))
        epoch_loss_sum = float(stage_values.get("epoch_loss_sum", 0.0))
        epoch_loss_rows = int(stage_values.get("epoch_loss_rows", 0))
        gradient_l2_norm_sum = float(stage_values.get("gradient_l2_norm_sum", 0.0))
        gradient_l2_norm_count = int(stage_values.get("gradient_l2_norm_count", 0))
        gradient_l2_norm_max = float(stage_values.get("gradient_l2_norm_max", 0.0))
        gradient_nonzero_step_count = int(stage_values.get("gradient_nonzero_step_count", 0))
        if not math.isfinite(epoch_loss_sum) or epoch_loss_sum < 0.0 or epoch_loss_rows < 0 or not math.isfinite(gradient_l2_norm_sum) or gradient_l2_norm_sum < 0.0 or gradient_l2_norm_count < 0 or not math.isfinite(gradient_l2_norm_max) or gradient_l2_norm_max < 0.0 or gradient_nonzero_step_count < 0:
            raise ValueError("resume gradient/loss reductions must be finite and nonnegative")
        if gradient_l2_norm_count != update_count or gradient_nonzero_step_count > gradient_l2_norm_count:
            raise ValueError("resume gradient reduction counts do not match optimizer updates")
        if stage_payload is not None:
            for key, direct_value in (("epoch", epoch), ("cursor", cursor), ("update_count", update_count), ("epoch_loss_rows", epoch_loss_rows)):
                if key in stage_payload and int(stage_payload[key]) != direct_value:
                    raise ValueError(f"resume stage_payload {key} mismatch")
        rng_state = resume_payload.get("rng_state")
        saved_rng = resume_payload.get("generator_state")
        if saved_rng is None and isinstance(rng_state, Mapping):
            saved_rng = rng_state.get("torch_cpu")
        if not isinstance(saved_rng, Tensor):
            raise ValueError("resume missing generator_state")
        generator.set_state(saved_rng.cpu())
        prior_history = resume_payload.get("history", [])
        if not isinstance(prior_history, list):
            raise ValueError("resume history must be a list")
        history = copy.deepcopy(prior_history)
    start_time = time.perf_counter()
    dependency_counters = CachedFitCallCounters()
    updates_limit = options.max_updates
    complete = True
    while epoch < options.epochs:
        if order is None:
            if options.shuffle:
                order_tensor = torch.randperm(len(train_rows), generator=generator)
                order = [int(item) for item in order_tensor.tolist()]
            else:
                order = list(range(len(train_rows)))
            cursor = 0
        while cursor < len(order):
            if updates_limit is not None and update_count >= updates_limit:
                complete = False
                break
            batch_indices = order[cursor : cursor + options.batch_size]
            x_batch = torch.stack([getattr(train_rows[index], f"v{variant}") for index in batch_indices]).to(device=target_device, dtype=model_dtype)
            y_batch = train_y[batch_indices].to(target_device)
            optimizer.zero_grad(set_to_none=True)
            predicted = value_model(x_batch).reshape(-1)
            if options.loss == "mse":
                objective = torch.mean((predicted - y_batch) ** 2)
            else:
                objective = torch.nn.functional.smooth_l1_loss(predicted, y_batch)
            if not bool(torch.isfinite(objective).all()):
                raise ValueError("nonfinite ValueNet loss")
            objective.backward()
            gradient_squared_norm = 0.0
            has_gradient = False
            for parameter in value_model.parameters():
                gradient = parameter.grad
                if gradient is None:
                    continue
                if not bool(torch.isfinite(gradient).all()):
                    raise ValueError("nonfinite ValueNet gradient")
                has_gradient = True
                gradient_squared_norm += float(gradient.detach().to(dtype=torch.float64).pow(2).sum().detach().cpu().item())
            gradient_l2_norm = math.sqrt(gradient_squared_norm) if has_gradient else 0.0
            if not math.isfinite(gradient_l2_norm):
                raise ValueError("nonfinite ValueNet gradient norm")
            gradient_l2_norm_sum += gradient_l2_norm
            gradient_l2_norm_count += 1
            gradient_l2_norm_max = max(gradient_l2_norm_max, gradient_l2_norm)
            if gradient_l2_norm > 0.0:
                gradient_nonzero_step_count += 1
            optimizer.step()
            loss_value = float(objective.detach().cpu().item())
            epoch_loss_sum += loss_value * len(batch_indices)
            epoch_loss_rows += len(batch_indices)
            cursor += len(batch_indices)
            update_count += 1
        if not complete:
            break
        if cursor >= len(order):
            history.append({
                "epoch": epoch + 1,
                "loss_scaled": (epoch_loss_sum / max(epoch_loss_rows, 1)),
                "loss_scaled_sum": epoch_loss_sum,
                "loss_scaled_count": epoch_loss_rows,
                "batch_count": int(math.ceil(len(train_rows) / options.batch_size)),
                "row_count": epoch_loss_rows,
            })
            epoch += 1
            order = None
            cursor = 0
            epoch_loss_sum = 0.0
            epoch_loss_rows = 0
    if epoch >= options.epochs and order is None:
        complete = True
    elif updates_limit is not None and update_count >= updates_limit:
        complete = False
    model_cpu_state = _clone_cpu(value_model.state_dict())
    optimizer_cpu_state = _clone_cpu(optimizer.state_dict())
    resume_state = {
        "schema_version": VALUE_FIT_RESUME_SCHEMA,
        "protocol": RESUME_SCHEMA,
        "stage": "value_fit",
        "stage_payload": {
            "epoch": epoch,
            "cursor": cursor,
            "order": None if order is None else list(order),
            "update_count": update_count,
            "epoch_loss_sum": epoch_loss_sum,
            "epoch_loss_rows": epoch_loss_rows,
            "gradient_l2_norm_sum": gradient_l2_norm_sum,
            "gradient_l2_norm_count": gradient_l2_norm_count,
            "gradient_l2_norm_max": gradient_l2_norm_max,
            "gradient_nonzero_step_count": gradient_nonzero_step_count,
        },
        "rng_state": {"torch_cpu": generator.get_state().clone()},
        "model_state_dict": model_cpu_state,
        "optimizer_state": optimizer_cpu_state,
        "optimizer_provenance": _clone_cpu(optimizer_provenance),
        "bank_state": {
            "manifest_hash": manifest_hash,
            "gain_scale_hash": scale.digest,
            "producer_compatibility_hash": manifest.producer_compatibility_hash,
            "split_role_hash": manifest.split_role_hash,
        },
        "history": copy.deepcopy(history),
        "bank_manifest_hash": manifest_hash,
        "gain_scale_hash": scale.digest,
        "producer_compatibility_hash": manifest.producer_compatibility_hash,
        "input_variant": variant,
        "fit_config_hash": fit_hash,
        "initial_weights_hash": initial_weights_hash,
    }
    all_pred_scaled = _predict_rows(value_model, train_rows, variant, device=target_device, batch_size=options.batch_size)
    metrics = _metric_summary(train_rows, all_pred_scaled, scale)
    forbidden_imports = _new_forbidden_imports(import_snapshot)
    if forbidden_imports:
        raise RuntimeError(f"cached ValueNet fit imported forbidden dependencies: {forbidden_imports}")
    mean_target = float(train_y.mean().item())
    metrics["constant_training_mean_mse_scaled"] = float(torch.mean((train_y - mean_target) ** 2).item())
    metrics["constant_training_mean_mse_raw"] = metrics["constant_training_mean_mse_scaled"] * float(scale.scale) ** 2
    metrics["constant_training_mean_raw"] = mean_target * float(scale.scale)
    metrics.update(dependency_counters.as_dict())
    metrics["teacher_calls"] = dependency_counters.teacher_calls
    metrics["target_volume_reads"] = dependency_counters.target_volume_reads
    metrics["updater_calls"] = dependency_counters.updater_calls
    metrics["decoder_calls"] = dependency_counters.decoder_calls
    metrics["hypothetical_writes"] = dependency_counters.hypothetical_writes
    metrics["dependency_counters"] = dependency_counters.as_dict()
    metrics["dependency_counter_scope"] = "cached_value_fit_only"
    metrics["dependency_counters_scope_verified"] = True
    metrics["dependency_call_verification_scope"] = "cached_api_has_no_mri_teacher_updater_decoder_inputs"
    metrics["forbidden_imports_during_fit"] = forbidden_imports
    metrics["optimizer_parameter_count"] = sum(parameter.numel() for parameter in value_model.parameters() if parameter.requires_grad)
    metrics["optimizer_parameter_names"] = tuple(name for name, parameter in value_model.named_parameters() if parameter.requires_grad)
    metrics["initial_weights_hash"] = initial_weights_hash
    metrics["seed"] = options.seed
    changed_names = tuple(
        name
        for name, parameter in value_model.named_parameters()
        if parameter.requires_grad and not torch.equal(parameter.detach().cpu(), initial_parameter_snapshots[name])
    )
    metrics["changed_parameter_names"] = changed_names
    metrics["changed_parameter_count"] = len(changed_names)
    metrics["changed_parameter_groups"] = ("value_net",) if changed_names else ()
    metrics["train_batch_count"] = update_count
    metrics["v_gradient_l2_norm_sum"] = gradient_l2_norm_sum
    metrics["v_gradient_l2_norm_count"] = gradient_l2_norm_count
    metrics["v_gradient_l2_norm_max"] = gradient_l2_norm_max
    metrics["v_gradient_l2_norm_mean"] = (gradient_l2_norm_sum / gradient_l2_norm_count) if gradient_l2_norm_count else None
    metrics["v_gradient_nonzero_step_count"] = gradient_nonzero_step_count
    metrics["train_row_count"] = len(train_rows)
    metrics["fit_complete"] = complete
    metrics["fit_loss"] = options.loss
    metrics["scale_floor_applied"] = scale.floor_applied
    metrics["fit_latency_seconds"] = time.perf_counter() - start_time
    weights_hash = module_state_digest(value_model)
    identity = ValueFitIdentity(
        input_variant=variant,
        architecture_hash=value_model.architecture_hash,
        weights_hash=weights_hash,
        fit_config_hash=fit_hash,
        bank_manifest_hash=manifest_hash,
        gain_scale_hash=scale.digest,
    )
    stage = StageState(stage="value_fit", substage="complete" if complete else "interrupted", epoch=epoch, update=update_count, microstep=cursor, optimizer_groups=tuple(str(item.get("name", index)) for index, item in enumerate(optimizer.param_groups)), completion="complete" if complete else "pending")
    return ValueFitResult(
        model=value_model,
        input_variant=variant,
        gain_scale=scale,
        identity=identity,
        history=history,
        metrics=metrics,
        resume_state=resume_state,
        complete=complete,
        latency_seconds=float(metrics["fit_latency_seconds"]),
        bank_manifest_hash=manifest_hash,
        fit_options=options,
        stage_state=stage,
    )


def evaluate_value(
    bank_reader: ValueBankReader | str,
    value_model: SignedValueNet | ValueFitResult,
    *,
    input_variant: int | str | None = None,
    split_roles: Sequence[str] | None = None,
    device: str = "cpu",
    batch_size: int = 256,
) -> ValueEvalResult:
    """Evaluate signed regression/ranking controls on selected bank roles."""

    import_snapshot = set(sys.modules)
    rows, _manifest, scale, _manifest_hash = _reader_parts(bank_reader, allow_diagnostic=True)
    if isinstance(value_model, ValueFitResult):
        model = value_model.model
        if input_variant is None:
            input_variant = value_model.input_variant
        if value_model.gain_scale.digest != scale.digest:
            raise ValueError("value fit gain scale does not match bank")
    else:
        model = value_model
    if not isinstance(model, SignedValueNet):
        raise TypeError("value_model must be SignedValueNet or ValueFitResult")
    variant = _variant(input_variant if input_variant is not None else model.input_variant)
    if variant != model.input_variant:
        raise ValueError("input_variant does not match ValueNet")
    _positive_int("batch_size", batch_size, maximum=MAX_BATCH_SIZE)
    if split_roles is None:
        selected = tuple(row for row in rows if not row.diagnostic)
        roles = None
    else:
        roles = tuple(str(role) for role in split_roles)
        selected = tuple(row for row in rows if row.split_role in roles and not row.diagnostic)
    started = time.perf_counter()
    dependency_counters = CachedFitCallCounters()
    if selected:
        predicted = _predict_rows(model, selected, variant, device=torch.device(device), batch_size=batch_size)
        metrics = _metric_summary(selected, predicted, scale)
    else:
        metrics = {
            "row_count": 0,
            "mse_scaled": None,
            "mse_scaled_sum": 0.0,
            "mse_scaled_count": 0,
            "mse_raw": None,
            "mse_raw_sum": 0.0,
            "mse_raw_count": 0,
            "sign_accuracy": None,
            "sign_correct_count": 0,
            "sign_defined_count": 0,
            "sign_tie_count": 0,
            "ranking_concordance": None,
            "ranking_concordant_count": 0,
            "ranking_pair_count": 0,
            "ranking_tie_count": 0,
            "top1_subset_regret_raw": None,
            "top1_subset_regret_sum_raw": 0.0,
            "top1_subset_regret_count": 0,
            "top1_subset_regret_scope": "same_subject_context_state_rows",
            "top1_subset_count": 0,
            "top1_subset_tie_groups": 0,
            "top1_subset_stop_count": 0,
            "top1_subset_stop_rule": "stop_when_best_predicted_raw_gain_leq_zero",
            "top1_subset_tie_rule": "lowest_point_id_then_action_id",
            "undefined_reason": "no rows for requested split roles",
            "raw_gain_signed": True,
        }
    metrics = dict(metrics)
    forbidden_imports = _new_forbidden_imports(import_snapshot)
    if forbidden_imports:
        raise RuntimeError(f"cached ValueNet evaluation imported forbidden dependencies: {forbidden_imports}")
    metrics.update(dependency_counters.as_dict())
    metrics["teacher_calls"] = dependency_counters.teacher_calls
    metrics["target_volume_reads"] = dependency_counters.target_volume_reads
    metrics["updater_calls"] = dependency_counters.updater_calls
    metrics["decoder_calls"] = dependency_counters.decoder_calls
    metrics["hypothetical_writes"] = dependency_counters.hypothetical_writes
    metrics["dependency_counters"] = dependency_counters.as_dict()
    metrics["dependency_counter_scope"] = "cached_value_eval_only"
    metrics["dependency_counters_scope_verified"] = True
    metrics["dependency_call_verification_scope"] = "cached_api_has_no_mri_teacher_updater_decoder_inputs"
    metrics["forbidden_imports_during_eval"] = forbidden_imports
    metrics["fit_latency_seconds"] = time.perf_counter() - started
    return ValueEvalResult(metrics=metrics, row_count=len(selected), input_variant=variant, split_roles=roles, latency_seconds=float(metrics["fit_latency_seconds"]))


def resume_value_fit(
    bank_reader: ValueBankReader | str,
    value_model: SignedValueNet,
    resume: ValueFitResult | Mapping[str, Any],
    **kwargs: Any,
) -> ValueFitResult:
    """Convenience continuation wrapper preserving bank/RNG/optimizer guards."""

    return fit_value(bank_reader, value_model, resume=resume, **kwargs)


fit_value_net = fit_value


__all__ = [
    "CachedFitCallCounters",
    "MAX_BATCH_SIZE",
    "MAX_EPOCHS",
    "SignedValueNet",
    "VALUE_FIT_RESUME_SCHEMA",
    "ValueEvalResult",
    "ValueFitConfig",
    "ValueFitOptions",
    "ValueFitResult",
    "ValueNet",
    "compute_gain_scale",
    "evaluate_value",
    "fit_value",
    "fit_value_net",
    "resume_value_fit",
]
