"""Legal synthetic full-static training smoke (E2 + anchor field + P1)."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import torch

from ..anchors import AggregationConfig, AnchorBootstrapConfig, CandidateSelectionConfig, ConsolidationConfig
from ..baselines import resolve_representation_plan
from ..baselines.fixed_gaussian import FixedGaussianHead, FixedGaussianHeadConfig
from ..baselines.fixed_support import FixedSupportConfig
from ..contracts.coordinates import PhysicalPlane
from ..contracts.episode import EpisodeAssignment, EpisodeLedger
from ..contracts.observation import AvailabilityObservationMeta, PatientSplitRegistry, SparseAvailabilityManifest
from ..features.encoder import EncoderConfig, EvidenceEncoder
from ..fields import SharedStructuralField, StructuralFieldConfig
from ..losses.reconstruction import ReconstructionLossConfig
from ..memory import PropagationConfig, SeedMemoryConfig
from ..renderer import RenderConfig
from ..state import save_patient_state
from ..training import LegalEpisodeConfig, build_static_episode_step


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _frozen_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in module.state_dict().items()
    }


def _model_state_hash(module: torch.nn.Module) -> str:
    payload = b"".join(
        name.encode() + value.detach().cpu().contiguous().numpy().tobytes()
        for name, value in module.state_dict().items()
    )
    return hashlib.sha256(payload).hexdigest()


def _atomic_torch_save(payload: object, path: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: object, path: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False, mode="w", encoding="utf-8") as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _git_metadata() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        difference = subprocess.run(
            ["git", "diff", "--binary", "HEAD"], check=True, capture_output=True
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("full-static training provenance requires a Git worktree") from error
    if len(commit) != 40 or len(set(commit)) == 1:
        raise RuntimeError("full-static training requires an exact non-placeholder commit")
    difference_digest = hashlib.sha256(difference)
    repository_root = Path.cwd().resolve()
    for relative in sorted(untracked):
        candidate = (repository_root / relative).resolve()
        if repository_root not in candidate.parents or not candidate.is_file():
            raise RuntimeError("untracked provenance inventory escaped the repository")
        difference_digest.update(relative.encode("utf-8"))
        difference_digest.update(candidate.read_bytes())
    return {
        "repository_commit": commit,
        "repository_dirty": bool(status),
        "repository_dirty_entries": status[:100],
        "repository_diff_hash": difference_digest.hexdigest(),
    }


def _payload(phase: float, shape: tuple[int, int]) -> bytes:
    v, u = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    array = np.asarray(np.sin(u / 3.0 + phase) + np.cos(v / 4.0), dtype=np.float32)
    buffer = BytesIO(); np.save(buffer, array, allow_pickle=False); return buffer.getvalue()


def _plane(observation_id: str, z: float, shape: tuple[int, int]) -> PhysicalPlane:
    return PhysicalPlane((0.0, 0.0, z), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0), 1.0, shape, (0.0, 0.0, 1.0), observation_id=observation_id)


def _load_config(path: Path) -> tuple[dict[str, object], str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "smagm-full-static-pipeline-v1" or config.get("t4_routing") is not False:
        raise ValueError("full-static config must use the locked schema and explicitly disable T4")
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return config, _digest(canonical)


def run(*, config_path: Path, steps: int, output_dir: Path, seed: int | None = None) -> dict[str, object]:
    config, config_hash = _load_config(config_path)
    git_metadata = _git_metadata()
    representation_plan = resolve_representation_plan(
        str(config["representation_variant"]),
        propagation_variant=str(config["propagation_variant"]),
    )
    if representation_plan.variant.value != "anchor_field" or representation_plan.propagation_variant != "p1":
        raise ValueError("the full static CLI is reserved for E2 + R4 + P1; use the episode dispatcher for ablations")
    if config.get("encoder_variant") != "e2" or config.get("device") != "cpu" or config.get("precision") != "float32":
        raise ValueError("the bounded full-static smoke is locked to E2, CPU, and float32")
    if steps <= 0:
        raise ValueError("full-static smoke steps must be positive")
    resolved_seed = int(seed if seed is not None else config["seed"]); torch.manual_seed(resolved_seed)
    training_raw = config["training"]
    modalities = tuple(str(value) for value in config["modalities"])
    if modalities != ("synthetic-mri",):
        raise ValueError("the synthetic full-static CLI requires exactly the synthetic-mri modality")
    encoder = EvidenceEncoder(EncoderConfig(variant=str(config["encoder_variant"])))
    head = FixedGaussianHead(FixedGaussianHeadConfig(
        input_dim=int(training_raw["gaussian_head_input_dim"]),
        appearance_channels=len(modalities),
        hidden_dim=int(training_raw["gaussian_head_hidden_dim"]),
    ))
    field_cfg_raw = config["field"]
    evidence_dim = int(field_cfg_raw.get("evidence_dim", 52))
    if evidence_dim != 52:
        raise ValueError("the full-static synthetic contract requires evidence_dim=52")
    field_config = StructuralFieldConfig(evidence_dim=evidence_dim, hidden_width=int(field_cfg_raw["hidden_width"]), hidden_layers=int(field_cfg_raw["hidden_layers"]), activation=str(field_cfg_raw["activation"]))
    field = SharedStructuralField(field_config)
    optimizer = torch.optim.Adam(
        tuple(encoder.parameters()) + tuple(field.parameters()),
        lr=float(training_raw["learning_rate"]),
    )
    anchor_raw = config["anchor"]
    bootstrap = AnchorBootstrapConfig(
        candidate=CandidateSelectionConfig(
            maximum_candidates=int(anchor_raw["maximum_candidates"]),
            minimum_score=float(anchor_raw["minimum_score"]),
            structural_weight=float(anchor_raw["structural_weight"]),
            reliability_weight=float(anchor_raw["reliability_weight"]),
        ),
        consolidation=ConsolidationConfig(
            nms_radius_mm=float(anchor_raw["nms_radius_mm"]), merge_radius_mm=float(anchor_raw["merge_radius_mm"]),
            maximum_component_diameter_mm=float(anchor_raw["maximum_component_diameter_mm"]), support_scale_mm=float(anchor_raw["support_scale_mm"]),
        ), aggregation=AggregationConfig(
            maximum_plane_distance_mm=float(anchor_raw["maximum_plane_distance_mm"]),
            distance_sigma_mm=float(anchor_raw["distance_sigma_mm"]),
            minimum_total_weight=float(anchor_raw["minimum_total_weight"]),
        ),
    )
    propagation_raw = config["propagation"]
    if str(propagation_raw["variant"]) != str(config["propagation_variant"]):
        raise ValueError("top-level and nested propagation variants must match")
    propagation = PropagationConfig(
        variant=str(propagation_raw["variant"]), rounds=int(propagation_raw["rounds"]), step_mm=float(propagation_raw["step_mm"]),
        children_per_parent_per_round=int(propagation_raw["children_per_parent_per_round"]), duplicate_radius_mm=float(propagation_raw["duplicate_radius_mm"]),
        uncertainty_growth_per_mm=float(propagation_raw["uncertainty_growth_per_mm"]),
        maximum_structural_primitives=int(propagation_raw["maximum_structural_primitives"]), maximum_volumetric_primitives=int(propagation_raw["maximum_volumetric_primitives"]),
    )
    memory_raw = config["memory"]
    seed_memory = SeedMemoryConfig(
        structural_tangent_fraction=float(memory_raw["structural_tangent_fraction"]),
        structural_normal_fraction=float(memory_raw["structural_normal_fraction"]),
        volumetric_scale_fraction=float(memory_raw["volumetric_scale_fraction"]),
        initial_uncertainty=float(memory_raw.get("initial_uncertainty", 1.0)),
        field_center_offset_fraction=float(memory_raw.get("field_center_offset_fraction", 0.05)),
    )
    renderer_raw = config["renderer"]
    if renderer_raw["profile"] != "delta":
        raise ValueError("the bounded synthetic smoke currently declares the delta through-plane profile")
    episode_config = LegalEpisodeConfig(
        supports=FixedSupportConfig(
            step_vu=tuple(int(value) for value in training_raw["fixed_support_step_vu"]),
            border_vu=tuple(int(value) for value in training_raw["fixed_support_border_vu"]),
        ),
        renderer=RenderConfig(
            support_epsilon=float(renderer_raw["support_epsilon"]),
            pixel_chunk_size=renderer_raw["pixel_chunk_size"],
            gaussian_chunk_size=renderer_raw["gaussian_chunk_size"],
            minimum_supported_psf_mass=float(renderer_raw["minimum_supported_psf_mass"]),
        ),
        reconstruction_loss=ReconstructionLossConfig(intensity=str(training_raw["reconstruction_intensity"])),
        modality_to_appearance_channel={modality: index for index, modality in enumerate(modalities)},
    )
    shape = tuple(int(value) for value in training_raw["synthetic_shape_hw"])
    if len(shape) != 2:
        raise ValueError("synthetic_shape_hw must have two dimensions")
    payloads = {"context": _payload(0.0, shape), "target": _payload(0.3, shape)}
    entries = tuple(AvailabilityObservationMeta(
        observation_id=name, patient_id="synthetic-patient", split="train", relative_path=f"{name}.npy", modality_id="synthetic-mri",
        plane=_plane(name, float(index), shape), is_synthetic=True,
    ) for index, name in enumerate(payloads))
    manifest = SparseAvailabilityManifest(entries, manifest_id="full-static-synthetic-v1", integrity_digests={name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()})
    registry = PatientSplitRegistry.create((manifest,)); reports = []; last_state = None
    state_encoder_snapshot: dict[str, torch.Tensor] | None = None
    state_field_snapshot: dict[str, torch.Tensor] | None = None
    state_encoder_hash: str | None = None
    state_field_hash: str | None = None
    start = time.perf_counter()
    last_assignment = None
    with tempfile.TemporaryDirectory(prefix="smagm-full-static-") as directory:
        root = Path(directory)
        for name, payload in payloads.items(): (root / f"{name}.npy").write_bytes(payload)
        for step_index in range(steps):
            assignment = EpisodeAssignment.create(manifest, episode_id=f"full-static-{step_index}", patient_id="synthetic-patient", context_ids=("context",), target_ids=("target",))
            ledger = EpisodeLedger(manifest, assignment, root, split_registry=registry)
            last_assignment = assignment
            optimizer.zero_grad(set_to_none=True)
            result = build_static_episode_step(
                ledger=ledger, assignment=assignment, target_id="target", encoder=encoder, gaussian_head=head,
                config=episode_config, patient_id="synthetic-patient", manifest_hash=manifest.manifest_hash,
                patient_config_hash=config_hash, field_model=field, field_config_hash=field_config.config_hash,
                field_maximum_neighbors=int(field_cfg_raw["maximum_neighbors"]),
                registration_id=str(anchor_raw["registration_id"]),
                bootstrap_config=bootstrap, seed_memory_config=seed_memory, propagation_config=propagation,
                gaussian_head_input_adapter=str(training_raw.get("gaussian_head_input_adapter", "anchor_evidence_projector")),
            )
            result.loss.total.backward()
            encoder_grad = torch.sqrt(sum((parameter.grad.square().sum() for parameter in encoder.parameters() if parameter.grad is not None), torch.tensor(0.0)))
            field_grad = torch.sqrt(sum((parameter.grad.square().sum() for parameter in field.parameters() if parameter.grad is not None), torch.tensor(0.0)))
            if not bool(torch.isfinite(encoder_grad) and torch.isfinite(field_grad)) or float(encoder_grad) <= 0 or float(field_grad) <= 0:
                raise FloatingPointError("full-static smoke requires finite non-zero encoder and field gradients")
            state_encoder_snapshot = _frozen_state_dict(encoder)
            state_field_snapshot = _frozen_state_dict(field)
            state_encoder_hash = encoder.state_hash()
            state_field_hash = _model_state_hash(field)
            if state_field_hash != result.patient_state.field_model_hash:
                raise RuntimeError("patient state does not bind the exact field snapshot used to create it")
            optimizer.step(); last_state = result.patient_state
            memory_tensors = (
                result.patient_state.memory.structural.gaussians.centers_ras_mm,
                result.patient_state.memory.structural.gaussians.covariance_factor,
                result.patient_state.memory.structural.gaussians.appearance,
                result.patient_state.memory.volumetric.gaussians.centers_ras_mm,
                result.patient_state.memory.volumetric.gaussians.covariance_factor,
                result.patient_state.memory.volumetric.gaussians.appearance,
            )
            reports.append({
                "step": step_index + 1, "loss": float(result.loss.total.detach()), "encoder_gradient_norm": float(encoder_grad),
                "field_gradient_norm": float(field_grad), "primitive_count": result.patient_state.memory.primitive_count,
                "anchor_count": result.patient_state.anchors.count,
                "cache_bytes": result.context_step.cache_bytes,
                "field_parameter_count": sum(parameter.numel() for parameter in field.parameters()),
                "patient_memory_tensor_bytes": sum(tensor.numel() * tensor.element_size() for tensor in memory_tensors),
                "state_version": result.patient_state.state_version, "receipt_hash": result.receipt_hash,
                "event_order": [event.event for event in ledger.event_records],
                "encoder_state_hash": state_encoder_hash,
                "field_model_hash": state_field_hash,
            })
    assert last_state is not None and last_assignment is not None
    assert state_encoder_snapshot is not None and state_field_snapshot is not None
    assert state_encoder_hash is not None and state_field_hash is not None
    output_dir.mkdir(parents=True, exist_ok=False)
    save_patient_state(last_state, output_dir / "patient_state.pt")
    encoder_after_training = _frozen_state_dict(encoder)
    field_after_training = _frozen_state_dict(field)
    _atomic_torch_save(
        {
            "schema": "smagm-full-static-checkpoint-v1",
            "encoder": state_encoder_snapshot,
            "field": state_field_snapshot,
            "encoder_for_patient_state_hash": state_encoder_hash,
            "field_for_patient_state_hash": state_field_hash,
            "encoder_after_training": encoder_after_training,
            "field_after_training": field_after_training,
            "encoder_after_training_hash": encoder.state_hash(),
            "field_after_training_hash": _model_state_hash(field),
            "config_hash": config_hash,
            "patient_state_path": "patient_state.pt",
            "patient_state_snapshot": "exact model parameters used before target commitment for the final recorded prediction",
            "post_snapshot_optimizer_updates": 1,
            "patient_state_version": last_state.state_version,
            "patient_state_field_model_hash": last_state.field_model_hash,
            **git_metadata,
            "steps": steps,
        },
        output_dir / "checkpoint.pt",
    )
    resolved = dict(config); resolved["steps"] = steps; resolved["seed"] = resolved_seed
    _atomic_json(resolved, output_dir / "resolved_config.json")
    evaluation_manifest = {
        "schema": "smagm-static-reconstruction-manifest-v1",
        "patient_id": last_state.patient_id,
        "manifest_hash": manifest.manifest_hash,
        "split_hash": _digest("train:synthetic-patient"),
        "assignment_hash": last_assignment.assignment_hash,
        "patient_state_version": last_state.state_version,
        "context_observation_ids": last_state.context_observation_ids,
        "target_observation_ids": last_assignment.target_ids,
        "contains_target_payloads": False,
    }
    _atomic_json(evaluation_manifest, output_dir / "eval_manifest.json")
    summary = {
        "schema": "smagm-full-static-train-smoke-v1", "claim_scope": config["claim_scope"],
        "config_hash": config_hash, "state_version": last_state.state_version,
        "memory_hash": last_state.memory.memory_hash, "steps": reports,
        "runtime_seconds": time.perf_counter() - start, "t4_routing": False,
        "representation_plan_hash": representation_plan.plan_hash,
        "active_modules": representation_plan.active_modules,
        "patient_state_encoder_hash": state_encoder_hash,
        "patient_state_field_model_hash": state_field_hash,
        "post_snapshot_optimizer_updates": 1,
        **git_metadata,
    }
    _atomic_json(summary, output_dir / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run legal synthetic full-static E2/anchor-field/P1 training smoke")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/full_static_pipeline.json"))
    parser.add_argument("--variant", choices=("full",), default="full")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(); report = run(config_path=args.config, steps=args.steps, output_dir=args.output_dir, seed=args.seed)
    print(json.dumps(report, sort_keys=True) if args.json else json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__": main()
