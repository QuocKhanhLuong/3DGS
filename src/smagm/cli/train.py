"""Config-driven CPU diagnostic for the legal fixed-topology T1-C trainer.

The command is deliberately a software-contract smoke run.  It creates no
claim about sparse MRI reconstruction quality or clinical validity.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import torch

from ..baselines.fixed_gaussian import FixedGaussianHead, FixedGaussianHeadConfig
from ..baselines.fixed_support import FixedSupportConfig
from ..contracts.coordinates import PhysicalPlane
from ..contracts.episode import EpisodeLedger
from ..contracts.observation import AvailabilityObservationMeta, PatientSplitRegistry, SparseAvailabilityManifest
from ..data.episodes import EpisodeSamplingConfig, ModalityEpisodePolicy
from ..data.normalization import NormalizationConfig
from ..features.encoder import EncoderConfig, EvidenceEncoder
from ..losses.reconstruction import ReconstructionLossConfig
from ..renderer import RenderConfig, SlabProfile
from ..training.episode import LegalEpisodeConfig
from ..training.provenance import capture_run_provenance, module_state_hash
from ..training.sampling import MatchedExperimentIdentity, build_matched_variant_schedule
from ..training.schedule import StageConfig, TrainingSchedule, TrainingStage
from ..training.objective import T1CObjectiveConfig
from ..training.trainer import T1CTrainer, TrainerConfig


_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _ROOT / "configs" / "experiments" / "t1c_synthetic.json"


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plane(observation_id: str, z_mm: float, shape_hw: tuple[int, int]) -> PhysicalPlane:
    return PhysicalPlane(
        pixel_center_origin_ras_mm=(0.0, 0.0, z_mm),
        axis_u_ras=(1.0, 0.0, 0.0),
        axis_v_ras=(0.0, 1.0, 0.0),
        spacing_uv_mm=(1.0, 1.0),
        thickness_mm=1.0,
        shape_hw=shape_hw,
        signed_normal_ras=(0.0, 0.0, 1.0),
        observation_id=observation_id,
    )


def _image(shape_hw: tuple[int, int], phase: float) -> np.ndarray:
    height, width = shape_hw
    v, u = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    first = np.exp(-((u - 0.35 * width) ** 2 + (v - 0.45 * height) ** 2) / (2.0 * (0.12 * width) ** 2))
    second = 0.65 * np.exp(-((u - 0.68 * width) ** 2 + (v - 0.58 * height) ** 2) / (2.0 * (0.09 * width) ** 2))
    ridge = 0.2 * np.sin(u / 4.0 + phase) * np.exp(-((v - 0.5 * height) ** 2) / (2.0 * (0.18 * height) ** 2))
    return np.asarray(first + second + ridge, dtype=np.float32)


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def matched_protocol_hash(config: Mapping[str, Any]) -> str:
    """Hash only conditions that must be identical across E0, E1, and E2.

    The selected encoder variant and an output location are run-local details;
    including either would turn one matched protocol into three identities.
    """

    shared = json.loads(json.dumps(config, sort_keys=True))
    shared.pop("selected_variant", None)
    shared.pop("output_dir", None)
    return _canonical_hash(shared)


def load_resolved_config(
    path: str | Path,
    *,
    variant: str | None = None,
    seed: int | None = None,
    device: str | None = None,
    steps: int | None = None,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Load, validate, override, and canonically hash the executable config."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FileNotFoundError(f"T1-C config cannot be read: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"T1-C config is not valid JSON: {source}") from exc
    config = _mapping(raw, "root")
    required = {
        "schema_version", "claim_scope", "seed", "variants", "episode", "encoder", "head", "supports", "renderer",
        "reconstruction_loss", "normalization", "training", "objective", "checkpointing", "fairness",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"T1-C config is missing required sections: {missing}")
    if config["schema_version"] != 1 or config["claim_scope"] != "software-contract diagnostic only":
        raise ValueError("T1-C config must declare schema_version=1 and software-contract-only scope")
    if not isinstance(config["variants"], list) or tuple(config["variants"]) != ("e0", "e1", "e2"):
        raise ValueError("T1-C config variants must exactly be e0, e1, e2")
    for name in ("episode", "encoder", "head", "supports", "renderer", "reconstruction_loss", "normalization", "training", "objective", "checkpointing", "fairness"):
        config[name] = _mapping(config[name], name)
    if variant is not None:
        if variant not in config["variants"]:
            raise ValueError("variant must be declared by the config")
        config["selected_variant"] = variant
    else:
        config["selected_variant"] = "e2"
    if seed is not None:
        config["seed"] = seed
    if device is not None:
        config["device"] = device
    else:
        config["device"] = "cpu"
    if config["device"] != "cpu":
        raise ValueError("the bounded T1-C reference supports CPU only")
    if steps is not None:
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps override must be a positive integer")
        config["training"]["steps"] = steps
    if not isinstance(config["training"].get("steps"), int) or config["training"]["steps"] <= 0:
        raise ValueError("training.steps must be a positive integer")
    if config["episode"].get("target_count") != 1:
        raise ValueError("the T1-C reference config requires target_count == 1")
    mapping = config["episode"].get("modality_to_appearance_channel")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("episode.modality_to_appearance_channel must be a non-empty mapping")
    if any(not isinstance(name, str) or not name or not isinstance(channel, int) or channel < 0 for name, channel in mapping.items()):
        raise ValueError("modality_to_appearance_channel values must be non-negative integers")
    weights = config["objective"].get("structural_weights")
    if (
        not isinstance(weights, list)
        or not weights
        or any(not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str) or not item[0] for item in weights)
    ):
        raise ValueError("objective.structural_weights must be non-empty [name, weight] pairs")
    if output_dir is not None:
        config["output_dir"] = str(Path(output_dir))
    config["source_config"] = str(source)
    config["matched_protocol_hash"] = matched_protocol_hash(config)
    return config, _canonical_hash(config)


def _stage_config(value: Mapping[str, object], stage: TrainingStage) -> StageConfig:
    return StageConfig(
        stage=stage,
        reconstruction_weight=float(value["reconstruction_weight"]),
        structural_weight=float(value["structural_weight"]),
        auxiliary_only=bool(value["auxiliary_only"]),
    )


def _build_trainer(
    config: Mapping[str, Any],
    *,
    resolved_config_hash: str,
    manifest_hash: str,
    split_registry_hash: str,
    schedule_hash: str,
) -> tuple[T1CTrainer, str, str]:
    variant = config["selected_variant"]
    encoder_cfg = dict(config["encoder"])
    encoder = EvidenceEncoder(EncoderConfig(variant=variant, **encoder_cfg))
    head_cfg = FixedGaussianHeadConfig(**dict(config["head"]))
    torch.manual_seed(int(config["seed"]) + 10_000)
    head = FixedGaussianHead(head_cfg)
    encoder_initialization_hash = module_state_hash(encoder)
    head_initialization_hash = module_state_hash(head)
    episode_cfg = LegalEpisodeConfig(
        supports=FixedSupportConfig(**dict(config["supports"])),
        renderer=RenderConfig(
            support_epsilon=float(config["renderer"]["support_epsilon"]),
            profile=SlabProfile.box(int(config["renderer"]["box_radius"])),
            minimum_supported_psf_mass=float(config["renderer"]["minimum_supported_psf_mass"]),
        ),
        reconstruction_loss=ReconstructionLossConfig(**dict(config["reconstruction_loss"])),
        normalization=NormalizationConfig(**dict(config["normalization"])),
        modality_to_appearance_channel=dict(config["episode"]["modality_to_appearance_channel"]),
    )
    training = config["training"]
    schedule_value = _mapping(training["schedule"], "training.schedule")
    schedule = TrainingSchedule(
        structural_warmup_steps=int(schedule_value["structural_warmup_steps"]),
        joint_reconstruction_steps=int(schedule_value["joint_reconstruction_steps"]),
        structural_warmup=_stage_config(_mapping(schedule_value["structural_warmup"], "structural_warmup"), TrainingStage.STRUCTURAL_WARMUP),
        joint_reconstruction=_stage_config(_mapping(schedule_value["joint_reconstruction"], "joint_reconstruction"), TrainingStage.JOINT_RECONSTRUCTION),
        reconstruction_dominant=_stage_config(_mapping(schedule_value["reconstruction_dominant"], "reconstruction_dominant"), TrainingStage.RECONSTRUCTION_DOMINANT),
    )
    optimizer_cfg = _mapping(training["optimizer"], "training.optimizer")
    if optimizer_cfg.get("name") != "Adam":
        raise ValueError("the bounded synthetic reference supports Adam only")
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=float(optimizer_cfg["lr"]),
        betas=tuple(float(value) for value in optimizer_cfg.get("betas", (0.9, 0.999))),
        eps=float(optimizer_cfg.get("eps", 1e-8)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
    )
    trainer_cfg = TrainerConfig(
        gradient_clip_norm=training.get("gradient_clip_norm", 10.0),
        accumulation_steps=int(training.get("accumulation_steps", 1)),
        precision=str(training.get("precision", "float32")),
        checkpoint_interval=int(training.get("checkpoint_interval", 1)),
        schedule=schedule,
        objective=T1CObjectiveConfig(
            structural_weights=tuple((str(name), float(weight)) for name, weight in config["objective"]["structural_weights"])
        ),
    )
    fairness = dict(config["fairness"])
    identity = MatchedExperimentIdentity.from_resolved_conditions(
        manifest_hash=manifest_hash,
        split_registry_hash=split_registry_hash,
        assignment_schedule_hash=schedule_hash,
        modality_mapping_hash=episode_cfg.modality_mapping_hash,
        shared_conditions={
            **fairness,
            "episode_config_hash": episode_cfg.config_hash,
            "head_config": asdict(head_cfg),
            "matched_protocol_hash": config["matched_protocol_hash"],
            "renderer_version": episode_cfg.renderer.renderer_version,
        },
    )
    trainer = T1CTrainer(
        encoder=encoder,
        gaussian_head=head,
        optimizer=optimizer,
        episode_config=episode_cfg,
        trainer_config=trainer_cfg,
        matched_experiment_identity=identity.identity_hash,
        resolved_config_hash=resolved_config_hash,
        sampler_state_hash=schedule_hash,
    )
    return trainer, encoder_initialization_hash, head_initialization_hash


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def run_synthetic_training(
    *,
    config: Mapping[str, Any],
    resolved_config_hash: str,
    output_dir: str | Path | None = None,
    repository_root: str | Path | None = None,
    allow_dirty: bool = False,
) -> dict[str, object]:
    """Execute the resolved synthetic protocol and, optionally, write artifacts."""

    variant = str(config["selected_variant"])
    seed = int(config["seed"])
    torch.manual_seed(seed)
    shape_hw = tuple(int(value) for value in config.get("synthetic_shape_hw", (31, 29)))
    payloads = {"synthetic-a": _npy_bytes(_image(shape_hw, 0.0)), "synthetic-b": _npy_bytes(_image(shape_hw, 0.35))}
    entries = tuple(
        AvailabilityObservationMeta(
            observation_id=observation_id,
            patient_id="synthetic-patient",
            split="train",
            relative_path=f"{observation_id}.npy",
            modality_id="synthetic-mri",
            plane=_plane(observation_id, float(index), shape_hw),
            is_synthetic=True,
        )
        for index, observation_id in enumerate(payloads)
    )
    manifest = SparseAvailabilityManifest(
        entries,
        manifest_id="t1c-synthetic-v2",
        integrity_digests={key: hashlib.sha256(value).hexdigest() for key, value in payloads.items()},
    )
    split_registry = PatientSplitRegistry.create((manifest,))
    episode = dict(config["episode"])
    sampling = EpisodeSamplingConfig(
        context_count=int(episode["context_count"]),
        target_count=int(episode["target_count"]),
        episode_count=int(config["training"]["steps"]),
        seed=seed,
        modality_policy=ModalityEpisodePolicy(**dict(episode.get("modality_policy", {}))),
    )
    matched = build_matched_variant_schedule(manifest, patient_id="synthetic-patient", config=sampling)
    schedule = matched.for_variant(variant)
    trainer, encoder_initialization_hash, head_initialization_hash = _build_trainer(
        config,
        resolved_config_hash=resolved_config_hash,
        manifest_hash=manifest.manifest_hash,
        split_registry_hash=split_registry.registry_hash,
        schedule_hash=schedule.schedule_hash,
    )
    output = Path(output_dir) if output_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        (output / "resolved_config.json").write_text(json.dumps(_jsonable(config), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    start_timestamp = time.time()
    start = time.perf_counter()
    reports = []
    with tempfile.TemporaryDirectory(prefix="smagm-t1c-") as directory:
        root = Path(directory)
        for observation_id, payload in payloads.items():
            (root / f"{observation_id}.npy").write_bytes(payload)
        for assignment in schedule.assignments:
            ledger = EpisodeLedger(manifest, assignment, root, split_registry=split_registry)
            reports.append(trainer.train_step(ledger=ledger, assignment=assignment, target_id=assignment.target_ids[0]).report)
    runtime_seconds = time.perf_counter() - start
    end_timestamp = time.time()
    last = reports[-1]
    checkpoint_path: Path | None = None
    if output is not None and bool(config["checkpointing"].get("enabled", False)):
        checkpoint_path = trainer.save_checkpoint(output / "checkpoint.pt")
    if output is not None:
        with (output / "metrics.jsonl").open("w", encoding="utf-8") as handle:
            for report in reports:
                handle.write(json.dumps(_jsonable(dict(report.__dict__)), sort_keys=True) + "\n")
    root = Path(repository_root) if repository_root is not None else _ROOT
    artifact_hashes = {}
    if output is not None:
        for name in ("resolved_config.json", "metrics.jsonl", "checkpoint.pt"):
            path = output / name
            if path.exists():
                artifact_hashes[name] = _file_hash(path)
    encoder = trainer.encoder
    head = trainer.gaussian_head
    episode_config = trainer.episode_config
    renderer_hash = _canonical_hash({"version": episode_config.renderer.renderer_version, "config": str(episode_config.renderer)})
    gauge_hash = _canonical_hash({"use_reliability_amplitude": head.config.use_reliability_amplitude})
    provenance = capture_run_provenance(
        repository_root=root,
        config_hash=resolved_config_hash,
        manifest_hash=manifest.manifest_hash,
        split_registry_hash=split_registry.registry_hash,
        assignment_schedule_hash=schedule.schedule_hash,
        seed=seed,
        checkpoint_hash=_file_hash(checkpoint_path) if checkpoint_path is not None else module_state_hash(encoder, head),
        artifact_hashes=artifact_hashes,
        allow_dirty=allow_dirty,
        modality_mapping_hash=episode_config.modality_mapping_hash,
        preprocessing_policy_hash=episode_config.normalization.config_hash,
        encoder_variant=variant,
        encoder_config_hash=encoder.config.config_hash,
        encoder_state_hash=encoder.state_hash(),
        gaussian_head_initialization_hash=head_initialization_hash,
        renderer_config_hash=renderer_hash,
        amplitude_gauge_hash=gauge_hash,
        device=str(next(head.parameters()).device),
        parameter_count=sum(parameter.numel() for parameter in list(encoder.parameters()) + list(head.parameters())),
        run_started_at=str(start_timestamp),
        run_ended_at=str(end_timestamp),
    )
    report = {
        **_jsonable(dict(last.__dict__)),
        "adapter_operation_count": encoder.parameter_report.adapter_operation_count,
        "analytic_preprocessing_cost": (
            shape_hw[0] * shape_hw[1] * encoder.parameter_report.analytic_channel_count
            if variant in ("e0", "e2")
            else 0
        ),
        "assignment_schedule_hash": schedule.schedule_hash,
        "cache_bytes": last.cache_bytes,
        "encoder_initialization_hash": encoder_initialization_hash,
        "encoder_parameter_count": encoder.parameter_count,
        "encoder_runtime_seconds": sum(item.encoder_runtime_seconds for item in reports) / len(reports),
        "end_to_end_step_runtime_seconds": runtime_seconds / len(reports),
        "head_initialization_hash": head_initialization_hash,
        "head_parameter_count": sum(parameter.numel() for parameter in head.parameters()),
        "manifest_hash": manifest.manifest_hash,
        "matched_experiment_identity": trainer.matched_experiment_identity,
        "matched_protocol_hash": config["matched_protocol_hash"],
        "resolved_config_hash": resolved_config_hash,
        "provenance_record_hash": provenance.record_hash,
        "reproducible": not provenance.dirty,
        "runtime_seconds": runtime_seconds,
        "split_registry_hash": split_registry.registry_hash,
        "total_parameter_count": encoder.parameter_count + sum(parameter.numel() for parameter in head.parameters()),
    }
    if output is not None:
        (output / "provenance.json").write_text(json.dumps(_jsonable(asdict(provenance)), sort_keys=True, indent=2) + "\n", encoding="utf-8")
        (output / "summary.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a config-driven legal synthetic T1-C optimizer path")
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--variant", choices=("e0", "e1", "e2"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--json", action="store_true", help="emit one JSON summary for quality checks")
    args = parser.parse_args()
    resolved, resolved_hash = load_resolved_config(
        args.config,
        variant=args.variant,
        seed=args.seed,
        device=args.device,
        steps=args.steps,
        output_dir=args.output_dir,
    )
    report = run_synthetic_training(config=resolved, resolved_config_hash=resolved_hash, output_dir=args.output_dir)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        for key, value in report.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
