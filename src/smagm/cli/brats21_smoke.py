"""Run the smallest receipt-gated BraTS21 real-data static smoke.

This command consumes only a prepared sparse derivative bundle.  The source
NIfTI root is used by the preparation command, never by the trainer.  The
segmentation plane is opened only after prediction serialization and is never
passed to the episode ledger, encoder, optimizer, or patient-state builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import torch

from ..anchors import (
    AggregationConfig,
    AnchorBootstrapConfig,
    CandidateSelectionConfig,
    ConsolidationConfig,
)
from ..baselines.fixed_gaussian import FixedGaussianHead, FixedGaussianHeadConfig
from ..baselines.fixed_support import FixedSupportConfig
from ..baselines.interpolation import SparseInterpolationConfig
from ..contracts.coordinates import PhysicalPlane, TargetGrid
from ..contracts.episode import EpisodeAssignment, EpisodeLedger
from ..contracts.observation import PatientSplitRegistry
from ..data.brats21_prepare import PreparedBraTS21, load_prepared_bundle
from ..data.normalization import NormalizationConfig
from ..evaluation import open_serialized_audit_targets
from ..features.encoder import EncoderConfig, EvidenceEncoder
from ..fields import SharedStructuralField, StructuralFieldConfig
from ..losses.reconstruction import ReconstructionLossConfig
from ..memory import PropagationConfig, SeedMemoryConfig
from ..reconstruction import build_reconstruction_package, export_reconstruction_package, reconstruct_volume
from ..reconstruction.uncertainty import support_uncertainty
from ..renderer import RenderConfig, RenderResult, SlabProfile
from ..state import save_patient_state
from ..training import LegalEpisodeConfig, build_representation_episode_step


_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _ROOT / "configs" / "experiments" / "brats21_real_smoke.json"
_DEFAULT_EVAL_CONFIG = _ROOT / "configs" / "evaluation" / "brats21_smoke_eval.json"


def _digest(value: object) -> str:
    if isinstance(value, bytes):
        payload = value
    else:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_torch(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _frozen_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous().clone() for name, value in module.state_dict().items()}


def _model_state_hash(module: torch.nn.Module) -> str:
    payload = b"".join(
        name.encode("utf-8") + value.detach().cpu().contiguous().numpy().tobytes()
        for name, value in module.state_dict().items()
    )
    return hashlib.sha256(payload).hexdigest()


def _git_metadata() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=_ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"], cwd=_ROOT, check=True, capture_output=True,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=_ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("BraTS21 smoke provenance requires a Git worktree") from error
    if len(commit) != 40 or len(set(commit)) == 1:
        raise RuntimeError("BraTS21 smoke requires an exact repository commit")
    digest = hashlib.sha256(diff)
    for relative in sorted(untracked):
        candidate = (_ROOT / relative).resolve()
        if _ROOT not in candidate.parents or not candidate.is_file():
            raise RuntimeError("untracked provenance inventory escaped the repository")
        digest.update(relative.encode("utf-8")); digest.update(candidate.read_bytes())
    return {
        "repository_commit": commit,
        "repository_dirty": bool(status),
        "repository_dirty_entries": status[:100],
        "repository_diff_hash": digest.hexdigest(),
    }


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "smagm-brats21-real-smoke-v1":
        raise ValueError("BraTS21 runner requires the brats21 real-smoke config schema")
    if config.get("t4_routing") is not False:
        raise ValueError("BraTS21 smoke must explicitly disable T4 routing")
    if config.get("encoder_variant") != "e2" or config.get("representation_variant") != "anchor_field":
        raise ValueError("the maintained real-data smoke is locked to E2 + R4")
    if config.get("propagation_variant") != "p0":
        raise ValueError("the initial real-data smoke is locked to P0")
    if config.get("precision") != "float32":
        raise ValueError("the initial real-data smoke is locked to float32")
    return config, _digest(config)


def _resolve_device(config: Mapping[str, Any], allow_cpu_fallback: bool) -> tuple[torch.device, dict[str, object]]:
    requested = str(config.get("device", "cuda"))
    if requested != "cuda":
        raise ValueError("the real-data smoke config must request device=cuda")
    cuda_available = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    if cuda_available:
        return torch.device("cuda"), {
            "requested_device": requested,
            "actual_device": "cuda",
            "cuda_available": True,
            "cuda_fallback": False,
            "cuda_fallback_reason": None,
        }
    if not allow_cpu_fallback:
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false; rerun with --allow-cpu-fallback "
            "for an explicit CPU diagnostic"
        )
    return torch.device("cpu"), {
        "requested_device": requested,
        "actual_device": "cpu",
        "cuda_available": False,
        "cuda_fallback": True,
        "cuda_fallback_reason": "CUDA requested but no usable NVIDIA device was visible",
    }


def _target_grid(plane: PhysicalPlane, *, preprocessing_hash: str | None) -> TargetGrid:
    origin = plane.pixel_center_origin_ras_mm
    axis_u = plane.axis_u_ras
    axis_v = plane.axis_v_ras
    normal = plane.signed_normal_ras
    matrix = tuple(
        (
            float(axis_u[row] * plane.spacing_uv_mm[0]),
            float(axis_v[row] * plane.spacing_uv_mm[1]),
            float(normal[row] * plane.thickness_mm),
            float(origin[row]),
        )
        for row in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)
    records = () if not preprocessing_hash else (preprocessing_hash,)
    return TargetGrid(matrix, (1, plane.shape_hw[0], plane.shape_hw[1]), ("flair",), records)


def _volume_from_render(
    prediction: RenderResult,
    *,
    patient_id: str,
    grid: TargetGrid,
    state_version: str,
    render_config: RenderConfig,
    propagation_uncertainty: float,
) -> Any:
    intensity = prediction.intensity.detach().unsqueeze(0)
    support = prediction.support_mass.detach().unsqueeze(0)
    unsupported = prediction.unsupported_mask.detach().unsqueeze(0)
    uncertainty = support_uncertainty(support, unsupported, propagation_uncertainty=propagation_uncertainty)
    renderer_hash = _digest(render_config.renderer_version)
    from ..contracts.outputs import VolumeReconstruction, volume_output_hash

    artifact_hash = volume_output_hash(
        patient_id=patient_id,
        modality_id="flair",
        grid=grid,
        depth_chunk_size=1,
        renderer_config_hash=renderer_hash,
        patient_state_version=state_version,
        intensity=intensity,
        support_mass=support,
        unsupported_mask=unsupported,
        support_uncertainty=uncertainty,
    )
    return VolumeReconstruction(
        patient_id,
        "flair",
        grid,
        intensity,
        support,
        unsupported,
        uncertainty,
        1,
        renderer_hash,
        state_version,
        artifact_hash,
    )


def _new_ledger(bundle: PreparedBraTS21, registry: PatientSplitRegistry) -> EpisodeLedger:
    return EpisodeLedger(bundle.manifest, bundle.assignment, bundle.root, split_registry=registry)


def _episode_config(config: Mapping[str, Any]) -> LegalEpisodeConfig:
    training = dict(config["training"])
    renderer = dict(config["renderer"])
    normal = dict(config["normalization"])
    if renderer.get("profile") != "delta":
        raise ValueError("the initial real-data smoke requires the declared delta profile")
    return LegalEpisodeConfig(
        normalization=NormalizationConfig(
            policy=str(normal["policy"]),
            epsilon=float(normal["epsilon"]),
            minimum_context_scale=float(normal["minimum_context_scale"]),
            degenerate_scale_policy=str(normal["degenerate_scale_policy"]),
            unseen_modality_policy=str(normal["unseen_modality_policy"]),
        ),
        supports=FixedSupportConfig(
            step_vu=tuple(int(value) for value in training["fixed_support_step_vu"]),
            border_vu=tuple(int(value) for value in training["fixed_support_border_vu"]),
            max_points=int(training["fixed_support_max_points"]),
        ),
        renderer=RenderConfig(
            support_epsilon=float(renderer["support_epsilon"]),
            pixel_chunk_size=renderer["pixel_chunk_size"],
            gaussian_chunk_size=renderer["gaussian_chunk_size"],
            minimum_supported_psf_mass=float(renderer["minimum_supported_psf_mass"]),
            profile=SlabProfile.delta(),
        ),
        reconstruction_loss=ReconstructionLossConfig(intensity=str(training["reconstruction_intensity"])),
        modality_to_appearance_channel={name: index for index, name in enumerate(config["modalities"])},
    )


def _bootstrap_config(config: Mapping[str, Any]) -> AnchorBootstrapConfig:
    raw = dict(config["anchor"])
    return AnchorBootstrapConfig(
        candidate=CandidateSelectionConfig(
            maximum_candidates=int(raw["maximum_candidates"]),
            minimum_score=float(raw["minimum_score"]),
            structural_weight=float(raw["structural_weight"]),
            reliability_weight=float(raw["reliability_weight"]),
        ),
        consolidation=ConsolidationConfig(
            nms_radius_mm=float(raw["nms_radius_mm"]),
            merge_radius_mm=float(raw["merge_radius_mm"]),
            maximum_component_diameter_mm=float(raw["maximum_component_diameter_mm"]),
            support_scale_mm=float(raw["support_scale_mm"]),
        ),
        aggregation=AggregationConfig(
            maximum_plane_distance_mm=float(raw["maximum_plane_distance_mm"]),
            distance_sigma_mm=float(raw["distance_sigma_mm"]),
            minimum_total_weight=float(raw["minimum_total_weight"]),
        ),
    )


def _propagation_config(config: Mapping[str, Any]) -> PropagationConfig:
    raw = dict(config["propagation"])
    return PropagationConfig(
        variant=str(raw["variant"]),
        rounds=int(raw["rounds"]),
        step_mm=float(raw["step_mm"]),
        children_per_parent_per_round=int(raw["children_per_parent_per_round"]),
        duplicate_radius_mm=float(raw["duplicate_radius_mm"]),
        uncertainty_growth_per_mm=float(raw["uncertainty_growth_per_mm"]),
        maximum_structural_primitives=int(raw["maximum_structural_primitives"]),
        maximum_volumetric_primitives=int(raw["maximum_volumetric_primitives"]),
    )


def _seed_memory_config(config: Mapping[str, Any]) -> SeedMemoryConfig:
    raw = dict(config["memory"])
    return SeedMemoryConfig(
        structural_tangent_fraction=float(raw["structural_tangent_fraction"]),
        structural_normal_fraction=float(raw["structural_normal_fraction"]),
        volumetric_scale_fraction=float(raw["volumetric_scale_fraction"]),
        initial_uncertainty=float(raw["initial_uncertainty"]),
        field_center_offset_fraction=float(raw["field_center_offset_fraction"]),
    )


def _interpolation_config(config: Mapping[str, Any]) -> SparseInterpolationConfig:
    raw = dict(config["interpolation"])
    return SparseInterpolationConfig(
        stride_vu=tuple(int(value) for value in raw["stride_vu"]),
        tangent_scale_fraction=float(raw["tangent_scale_fraction"]),
        normal_scale_fraction=float(raw["normal_scale_fraction"]),
        maximum_points=int(raw["maximum_points"]),
    )


def _event_report(ledger: EpisodeLedger) -> dict[str, object]:
    events = [event.to_canonical_dict() for event in ledger.event_records]
    names = [str(item["event"]) for item in events]
    required = ("OPEN_CONTEXT", "COMMIT_TARGET", "REGISTER_PREDICTION", "REVEAL_TARGET")
    if any(name not in names for name in required):
        raise RuntimeError(f"episode did not record the required receipt order: {names}")
    if names.index("REGISTER_PREDICTION") > names.index("REVEAL_TARGET"):
        raise RuntimeError("target reveal occurred before prediction registration")
    commit_index = names.index("COMMIT_TARGET")
    if any(name != "OPEN_CONTEXT" for name in names[:commit_index]):
        raise RuntimeError("non-context event occurred before target commitment")
    return {
        "events": events,
        "opened_files": [row.to_canonical_dict() for row in ledger.audit_records],
        "prediction_records": [row.to_canonical_dict() for row in ledger.prediction_records],
        "audit_hash": ledger.audit_hash,
        "target_reveal_after_prediction": True,
    }


def _package_and_export(
    *,
    output_dir: Path,
    volume: Any,
    config_hash: str,
    manifest_hash: str,
    split_hash: str,
    assignment_hash: str,
    encoder_identity: str,
    field_identity: str,
    gaussian_identity: str,
    propagation_identity: str,
    environment_hash: str,
    git: Mapping[str, object],
    runtime_seconds: float,
) -> dict[str, object]:
    package = build_reconstruction_package(
        (volume,),
        repository_commit=str(git["repository_commit"]),
        config_hash=config_hash,
        manifest_hash=manifest_hash,
        split_hash=split_hash,
        assignment_hash=assignment_hash,
        encoder_identity=encoder_identity,
        field_identity=field_identity,
        gaussian_identity=gaussian_identity,
        propagation_identity=propagation_identity,
        environment_hash=environment_hash,
        runtime_seconds=runtime_seconds,
        execution_status="COMPLETE" if not bool(volume.unsupported_mask.all()) else "INSUFFICIENTLY_OBSERVED",
    )
    export_reconstruction_package(package, (volume,), output_dir, write_nifti=False)
    return {
        "package_hash": package.package_hash,
        "patient_state_version": package.patient_state_version,
        "prediction_package": str(output_dir),
        "unsupported_fraction": float(volume.unsupported_mask.float().mean()),
        "supported_fraction": float((~volume.unsupported_mask).float().mean()),
    }


def _write_targets(
    path: Path,
    *,
    patient_id: str,
    split_hash: str,
    grid: TargetGrid,
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    segmentation: torch.Tensor,
) -> dict[str, object]:
    if values.ndim != 2:
        raise ValueError("the target plane must remain [H,W] before evaluator packaging")
    target_values = values.detach().cpu().unsqueeze(0).contiguous()
    target_valid = valid_mask.detach().cpu().unsqueeze(0).contiguous()
    segmentation_payload = segmentation.detach().cpu().to(torch.uint8)
    if segmentation_payload.ndim == 2:
        segmentation_payload = segmentation_payload.unsqueeze(0)
    segmentation_payload = segmentation_payload.contiguous()
    if target_values.shape != tuple(grid.shape_dhw) or target_valid.shape != target_values.shape:
        raise ValueError("evaluator target does not match the held-out target grid")
    if segmentation_payload.shape != target_values.shape:
        raise ValueError("evaluator segmentation does not match the held-out target grid")
    _atomic_torch(
        {
            "schema": "smagm-audit-targets-v1",
            "targets": [{
                "patient_id": patient_id,
                "split_hash": split_hash,
                "modality_id": "flair",
                "grid": grid.to_canonical_dict(),
                "values": target_values,
                "valid_mask": target_valid,
            }],
            "segmentation": segmentation_payload,
            "segmentation_labels": [0, 1, 2, 4],
            "segmentation_evaluator_only": True,
        },
        path,
    )
    return {
        "path": str(path),
        "sha256": _file_hash(path),
        "segmentation_lesion_fraction": float((segmentation_payload > 0).float().mean()),
        "segmentation_access_phase": "after_prediction_receipt_and_serialization",
    }


def _evaluation_plan(template: Path, *, target_file: Path, target_hash: str) -> dict[str, object]:
    plan = json.loads(template.read_text(encoding="utf-8"))
    if plan.get("target_mode") != "external_tensor_file":
        raise ValueError("BraTS21 smoke evaluation must use external_tensor_file mode")
    plan["target_file"] = target_file.name
    plan["target_file_sha256"] = target_hash
    return plan


def _pseudonymous_patient(patient_id: str, manifest_hash: str) -> str:
    return "patient-" + hashlib.sha256(f"{patient_id}:{manifest_hash}".encode("utf-8")).hexdigest()[:16]


def _gradient_norm(module: torch.nn.Module) -> tuple[float, bool]:
    values = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    if not values:
        return 0.0, False
    norm = torch.sqrt(sum((value.detach().square().sum() for value in values), torch.tensor(0.0, device=values[0].device)))
    return float(norm.detach().cpu()), bool(torch.isfinite(norm) and norm > 0)


class _AdamFallback:
    """Small dependency-free Adam used only when the installed torch optimizer is broken.

    PyTorch's optimizer decorator imports ``torch._dynamo`` during
    construction.  The supplied environment currently has an internally
    inconsistent optional Dynamo module, while tensor autograd itself works.
    This fallback keeps the smoke executable and records the environment
    workaround in the resolved run metadata.
    """

    def __init__(self, parameters: tuple[torch.nn.Parameter, ...], learning_rate: float) -> None:
        self.parameters = parameters
        self.learning_rate = float(learning_rate)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.step_index = 0
        self.first = [torch.zeros_like(parameter) for parameter in parameters]
        self.second = [torch.zeros_like(parameter) for parameter in parameters]

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.zero_()

    def step(self) -> None:
        self.step_index += 1
        correction_one = 1.0 - self.beta1**self.step_index
        correction_two = 1.0 - self.beta2**self.step_index
        with torch.no_grad():
            for parameter, first, second in zip(self.parameters, self.first, self.second):
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach()
                first.mul_(self.beta1).add_(gradient, alpha=1.0 - self.beta1)
                second.mul_(self.beta2).addcmul_(gradient, gradient, value=1.0 - self.beta2)
                first_hat = first / correction_one
                second_hat = second / correction_two
                parameter.addcdiv_(first_hat, second_hat.sqrt().add(self.epsilon), value=-self.learning_rate)


def _make_optimizer(
    encoder: EvidenceEncoder, field: SharedStructuralField, learning_rate: float
) -> tuple[Any, str]:
    parameters = tuple(encoder.parameters()) + tuple(field.parameters())
    try:
        return torch.optim.Adam(parameters, lr=learning_rate), "torch.optim.Adam"
    except ImportError:
        return _AdamFallback(parameters, learning_rate), "adam-fallback-torch-dynamo-import-error"


def _run_r0(
    *,
    bundle: PreparedBraTS21,
    registry: PatientSplitRegistry,
    config: Mapping[str, Any],
    episode_config: LegalEpisodeConfig,
    interpolation: SparseInterpolationConfig,
    output_dir: Path,
    config_hash: str,
    split_hash: str,
    git: Mapping[str, object],
    environment_hash: str,
) -> tuple[dict[str, object], Any, torch.Tensor, torch.Tensor, str]:
    start = time.perf_counter()
    ledger = _new_ledger(bundle, registry)
    result = build_representation_episode_step(
        ledger=ledger,
        assignment=bundle.assignment,
        target_id=bundle.target_id,
        representation_variant="r0",
        propagation_variant="p0",
        config=episode_config,
        interpolation_config=interpolation,
    )
    events = _event_report(ledger)
    state_version = _digest({"r0": True, "manifest_hash": bundle.manifest.manifest_hash, "assignment_hash": bundle.assignment.assignment_hash})
    grid = _target_grid(bundle.target_plane, preprocessing_hash=result.preprocessing_record_hash)
    volume = _volume_from_render(
        result.prediction,
        patient_id=bundle.patient_id,
        grid=grid,
        state_version=state_version,
        render_config=episode_config.renderer,
        propagation_uncertainty=0.0,
    )
    package_dir = output_dir / "predictions"
    package = _package_and_export(
        output_dir=package_dir,
        volume=volume,
        config_hash=config_hash,
        manifest_hash=bundle.manifest.manifest_hash,
        split_hash=split_hash,
        assignment_hash=bundle.assignment.assignment_hash,
        encoder_identity=_digest("r0-interpolation"),
        field_identity=_digest("no-field"),
        gaussian_identity=_digest("r0-interpolation:" + result.receipt_hash),
        propagation_identity=_digest("p0"),
        environment_hash=environment_hash,
        git=git,
        runtime_seconds=time.perf_counter() - start,
    )
    _atomic_torch({
        "schema": "smagm-brats21-r0-checkpoint-v1",
        "config_hash": config_hash,
        "manifest_hash": bundle.manifest.manifest_hash,
        "assignment_hash": bundle.assignment.assignment_hash,
        "split_hash": split_hash,
        "patient_state_version": state_version,
        "representation_variant": "interpolation",
        "propagation_variant": "p0",
        "target_payload_not_in_checkpoint": True,
    }, output_dir / "checkpoint.pt")
    _atomic_json(events, output_dir / "episode_ledger.json")
    report = {
        "variant": "r0",
        "representation_variant": "interpolation",
        "propagation_variant": "p0",
        "loss": float(result.loss.total.detach().cpu()),
        "legal_pixel_count": result.loss.legal_pixel_count,
        "target_valid_pixel_count": result.loss.target_valid_pixel_count,
        "supported_fraction": result.loss.supported_fraction,
        "unsupported_fraction": float(result.prediction.unsupported_mask.float().mean().detach().cpu()),
        "primitive_count": result.primitive_count,
        "anchor_count": 0,
        "runtime_seconds": time.perf_counter() - start,
        "checkpoint": str(output_dir / "checkpoint.pt"),
        "prediction_package": str(package_dir),
        **package,
    }
    _atomic_json(report, output_dir / "summary.json")
    return report, result, result.target.detach(), result.target_valid_mask.detach(), result.preprocessing_record_hash or ""


def _run_r4(
    *,
    bundle: PreparedBraTS21,
    registry: PatientSplitRegistry,
    config: Mapping[str, Any],
    episode_config: LegalEpisodeConfig,
    encoder: EvidenceEncoder,
    gaussian_head: FixedGaussianHead,
    field: SharedStructuralField,
    optimizer: Any,
    bootstrap: AnchorBootstrapConfig,
    seed_memory: SeedMemoryConfig,
    propagation: PropagationConfig,
    output_dir: Path,
    config_hash: str,
    split_hash: str,
    git: Mapping[str, object],
    environment_hash: str,
    steps: int,
) -> tuple[dict[str, object], Any, torch.Tensor, torch.Tensor, str]:
    start = time.perf_counter()
    reports: list[dict[str, object]] = []
    last_result = None
    last_state = None
    state_encoder_snapshot: dict[str, torch.Tensor] | None = None
    state_field_snapshot: dict[str, torch.Tensor] | None = None
    state_encoder_hash: str | None = None
    state_field_hash: str | None = None
    last_ledger: EpisodeLedger | None = None
    for step_index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        ledger = _new_ledger(bundle, registry)
        result = build_representation_episode_step(
            ledger=ledger,
            assignment=bundle.assignment,
            target_id=bundle.target_id,
            representation_variant="r4",
            propagation_variant="p0",
            config=episode_config,
            encoder=encoder,
            gaussian_head=gaussian_head,
            local_field=field,
            field_maximum_neighbors=int(config["field"]["maximum_neighbors"]),
            registration_id=str(config["anchor"]["registration_id"]),
            bootstrap_config=bootstrap,
            seed_memory_config=seed_memory,
            propagation_config=propagation,
        )
        result.loss.total.backward()
        encoder_grad, encoder_ok = _gradient_norm(encoder)
        field_grad, field_ok = _gradient_norm(field)
        if not encoder_ok or not field_ok or not np.isfinite(result.loss.total.detach().cpu().item()):
            raise FloatingPointError("BraTS21 R4-P0 smoke requires finite non-zero encoder and field gradients")
        state_encoder_snapshot = _frozen_state_dict(encoder)
        state_field_snapshot = _frozen_state_dict(field)
        state_encoder_hash = encoder.state_hash()
        state_field_hash = _model_state_hash(field)
        if state_field_hash != result.patient_state.field_model_hash:
            raise RuntimeError("patient state does not bind the exact field snapshot used before optimizer update")
        events = _event_report(ledger)
        optimizer.step()
        last_result = result; last_state = result.patient_state; last_ledger = ledger
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        reports.append({
            "step": step_index + 1,
            "loss": float(result.loss.total.detach().cpu()),
            "legal_pixel_count": result.loss.legal_pixel_count,
            "target_valid_pixel_count": result.loss.target_valid_pixel_count,
            "supported_fraction": result.loss.supported_fraction,
            "unsupported_fraction": float(result.prediction.unsupported_mask.float().mean().detach().cpu()),
            "encoder_gradient_norm": encoder_grad,
            "field_gradient_norm": field_grad,
            "primitive_count": result.patient_state.memory.primitive_count,
            "anchor_count": result.patient_state.anchors.count,
            "state_version": result.patient_state.state_version,
            "receipt_hash": result.receipt_hash,
            "event_order": [item["event"] for item in events["events"]],
            "encoder_state_hash": state_encoder_hash,
            "field_model_hash": state_field_hash,
        })
    if last_result is None or last_state is None or last_ledger is None or state_encoder_snapshot is None or state_field_snapshot is None or state_encoder_hash is None or state_field_hash is None:
        raise RuntimeError("R4 smoke produced no optimizer step")
    patient_state_path = output_dir / "patient_state.pt"
    save_patient_state(last_state, patient_state_path)
    checkpoint = {
        "schema": "smagm-brats21-r4-checkpoint-v1",
        "config_hash": config_hash,
        "manifest_hash": bundle.manifest.manifest_hash,
        "split_hash": split_hash,
        "assignment_hash": bundle.assignment.assignment_hash,
        "patient_state_path": patient_state_path.name,
        "patient_state_version": last_state.state_version,
        "patient_state_field_model_hash": last_state.field_model_hash,
        "encoder": state_encoder_snapshot,
        "field": state_field_snapshot,
        "gaussian_head": _frozen_state_dict(gaussian_head),
        "encoder_for_patient_state_hash": state_encoder_hash,
        "field_for_patient_state_hash": state_field_hash,
        "gaussian_head_hash": _model_state_hash(gaussian_head),
        "patient_state_snapshot": "exact parameters used before target commitment for the final recorded episode",
        "post_snapshot_optimizer_updates": 1,
        "steps": steps,
        **git,
    }
    _atomic_torch(checkpoint, output_dir / "checkpoint.pt")
    grid = _target_grid(bundle.target_plane, preprocessing_hash=last_result.preprocessing_record_hash)
    volume = reconstruct_volume(
        last_state,
        grid,
        modality_id="flair",
        depth_chunk_size=1,
        render_config=episode_config.renderer,
    )
    package_dir = output_dir / "predictions"
    package = _package_and_export(
        output_dir=package_dir,
        volume=volume,
        config_hash=config_hash,
        manifest_hash=bundle.manifest.manifest_hash,
        split_hash=split_hash,
        assignment_hash=bundle.assignment.assignment_hash,
        encoder_identity=state_encoder_hash,
        field_identity=state_field_hash,
        gaussian_identity=last_state.memory.memory_hash,
        propagation_identity=propagation.config_hash,
        environment_hash=environment_hash,
        git=git,
        runtime_seconds=time.perf_counter() - start,
    )
    _atomic_json(_event_report(last_ledger), output_dir / "episode_ledger.json")
    report = {
        "variant": "e2_r4_p0",
        "encoder_variant": "e2",
        "representation_variant": "anchor_field",
        "propagation_variant": "p0",
        "steps": reports,
        "loss": reports[-1]["loss"],
        "supported_fraction": reports[-1]["supported_fraction"],
        "unsupported_fraction": reports[-1]["unsupported_fraction"],
        "encoder_gradient_norm": reports[-1]["encoder_gradient_norm"],
        "field_gradient_norm": reports[-1]["field_gradient_norm"],
        "primitive_count": reports[-1]["primitive_count"],
        "anchor_count": reports[-1]["anchor_count"],
        "runtime_seconds": time.perf_counter() - start,
        "checkpoint": str(output_dir / "checkpoint.pt"),
        "patient_state": str(patient_state_path),
        "prediction_package": str(package_dir),
        **package,
    }
    _atomic_json(report, output_dir / "summary.json")
    return report, last_result, last_result.target.detach(), last_result.target_valid_mask.detach(), last_result.preprocessing_record_hash or ""


def run(
    *,
    config_path: Path = _DEFAULT_CONFIG,
    prepared_dir: Path,
    output_dir: Path,
    evaluation_config_path: Path = _DEFAULT_EVAL_CONFIG,
    allow_cpu_fallback: bool = False,
    steps: int | None = None,
    wandb_mode: str | None = None,
) -> dict[str, object]:
    config, config_hash = _load_config(config_path)
    if wandb_mode is not None:
        if wandb_mode not in ("disabled", "offline", "online"):
            raise ValueError("wandb_mode must be disabled, offline, or online")
        config["wandb"] = dict(config.get("wandb", {}))
        config["wandb"]["mode"] = wandb_mode
        config_hash = _digest(config)
    requested_steps = int(steps if steps is not None else config["training"]["steps"])
    if not 2 <= requested_steps <= 5:
        raise ValueError("BraTS21 smoke steps must be between 2 and 5")
    bundle = load_prepared_bundle(prepared_dir)
    if len(bundle.assignment.context_ids) != 4 or len(bundle.assignment.target_ids) != 1:
        raise ValueError("the initial BraTS21 smoke requires exactly four context observations and one target")
    if any(bundle.manifest.metadata(item).modality_id not in tuple(config["modalities"]) for item in bundle.assignment.context_ids):
        raise ValueError("prepared context modalities disagree with the smoke config")
    if bundle.manifest.metadata(bundle.target_id).modality_id != str(config["target_modality"]):
        raise ValueError("prepared target modality disagrees with the smoke config")
    device, device_report = _resolve_device(config, allow_cpu_fallback or bool(config["training"].get("allow_cpu_fallback", False)))
    seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(f"real-data smoke output is non-empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    git = _git_metadata()
    split_registry = PatientSplitRegistry.create((bundle.manifest,))
    prepared_meta = json.loads((prepared_dir / "prepared.json").read_text(encoding="utf-8"))
    split_hash = str(prepared_meta["split_hash"])
    pseudonymous_patient = _pseudonymous_patient(bundle.patient_id, bundle.manifest.manifest_hash)
    episode_config = _episode_config(config)
    bootstrap = _bootstrap_config(config)
    seed_memory = _seed_memory_config(config)
    propagation = _propagation_config(config)
    interpolation = _interpolation_config(config)
    encoder = EvidenceEncoder(EncoderConfig(variant="e2")).to(device=device, dtype=torch.float32)
    head = FixedGaussianHead(FixedGaussianHeadConfig(
        input_dim=int(config["training"]["gaussian_head_input_dim"]),
        appearance_channels=len(tuple(config["modalities"])),
        hidden_dim=int(config["training"]["gaussian_head_hidden_dim"]),
    )).to(device=device, dtype=torch.float32)
    field_config = StructuralFieldConfig(
        evidence_dim=int(config["field"]["evidence_dim"]),
        hidden_width=int(config["field"]["hidden_width"]),
        hidden_layers=int(config["field"]["hidden_layers"]),
        activation=str(config["field"]["activation"]),
    )
    field = SharedStructuralField(field_config).to(device=device, dtype=torch.float32)
    optimizer, optimizer_name = _make_optimizer(encoder, field, float(config["training"]["learning_rate"]))
    run_start = time.perf_counter()
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not-set"),
        **device_report,
    }
    environment_hash = _digest(environment)
    resolved = json.loads(json.dumps(config))
    resolved["runtime"] = {
        "actual_device": device_report["actual_device"],
        "allow_cpu_fallback": bool(allow_cpu_fallback or config["training"].get("allow_cpu_fallback", False)),
        "prepared_dir_not_uploaded": True,
        "patient_pseudonymous_id": pseudonymous_patient,
        "split_registry_hash": split_registry.registry_hash,
        "prepared_manifest_hash": bundle.manifest.manifest_hash,
        "optimizer": optimizer_name,
    }
    _atomic_json(resolved, output_dir / "resolved_config.json")
    _atomic_json({
        "schema": "smagm-brats21-real-smoke-manifest-v1",
        "patient_pseudonymous_id": pseudonymous_patient,
        "manifest_hash": bundle.manifest.manifest_hash,
        "split_hash": split_hash,
        "assignment_hash": bundle.assignment.assignment_hash,
        "context_observation_ids": list(bundle.assignment.context_ids),
        "target_observation_ids": list(bundle.assignment.target_ids),
        "contains_target_payloads": False,
        "contains_segmentation": False,
        "target_plane": bundle.target_plane.to_canonical_dict(),
    }, output_dir / "eval_manifest.json")
    logger = None
    try:
        try:
            from ..experiments.wandb import WandbLogger

            logger = WandbLogger(
                config=resolved,
                run_name=f"brats21-smoke-{pseudonymous_patient}",
                run_dir=output_dir,
                mode=str(config.get("wandb", {}).get("mode", "disabled")),
                metadata={
                    "schema": config["schema"],
                    "repository_commit": git["repository_commit"],
                    "repository_dirty": git["repository_dirty"],
                    "seed": seed,
                    "encoder_variant": "e2",
                    "representation_variant": "anchor_field",
                    "propagation_variant": "p0",
                    "patient_pseudonymous_id": pseudonymous_patient,
                    "context_count": len(bundle.assignment.context_ids),
                    "modality_inventory": list(config["modalities"]),
                    "target_modality": "flair",
                    "target_orientation": bundle.manifest_json.get("source_geometry", {}).get("orientation"),
                    "source_kind": config["source_kind"],
                },
            )
            logger.start()
            # W&B scalar history accepts finite numeric values only.  The
            # config hash is already stored in the sanitized run config and
            # metadata; only numeric run counters belong in the history.
            logger.log({"run/seed": seed, "run/context_count": 4}, step=0)
        except ImportError:
            logger = None
        r0_dir = output_dir / "r0"
        r0_dir.mkdir()
        r0_report, r0_result, r0_target, r0_valid, r0_preprocess = _run_r0(
            bundle=bundle, registry=split_registry, config=config, episode_config=episode_config,
            interpolation=interpolation, output_dir=r0_dir, config_hash=config_hash,
            split_hash=split_hash, git=git, environment_hash=environment_hash,
        )
        if logger is not None:
            logger.log({
                "r0/loss": r0_report["loss"],
                "r0/supported_fraction": r0_report["supported_fraction"],
                "r0/unsupported_fraction": r0_report["unsupported_fraction"],
                "r0/primitive_count": r0_report["primitive_count"],
                "r0/runtime_seconds": r0_report["runtime_seconds"],
            }, step=0)
        r4_dir = output_dir / "e2_r4_p0"
        r4_dir.mkdir()
        r4_report, r4_result, r4_target, r4_valid, r4_preprocess = _run_r4(
            bundle=bundle, registry=split_registry, config=config, episode_config=episode_config,
            encoder=encoder, gaussian_head=head, field=field, optimizer=optimizer,
            bootstrap=bootstrap, seed_memory=seed_memory, propagation=propagation,
            output_dir=r4_dir, config_hash=config_hash, split_hash=split_hash, git=git,
            environment_hash=environment_hash, steps=requested_steps,
        )
        if logger is not None:
            for item in r4_report["steps"]:
                logger.log({
                    "r4_p0/loss": item["loss"],
                    "r4_p0/supported_fraction": item["supported_fraction"],
                    "r4_p0/unsupported_fraction": item["unsupported_fraction"],
                    "r4_p0/encoder_gradient_norm": item["encoder_gradient_norm"],
                    "r4_p0/field_gradient_norm": item["field_gradient_norm"],
                    "r4_p0/primitive_count": item["primitive_count"],
                    "r4_p0/anchor_count": item["anchor_count"],
                }, step=int(item["step"]))
        segmentation = torch.from_numpy(np.load(bundle.segmentation_payload_path, allow_pickle=False)).to(torch.uint8)
        target_report = _write_targets(
            r0_dir / "evaluator_targets.pt", patient_id=bundle.patient_id, split_hash=split_hash,
            grid=_target_grid(bundle.target_plane, preprocessing_hash=r0_preprocess), values=r0_target,
            valid_mask=r0_valid, segmentation=segmentation,
        )
        _write_targets(
            r4_dir / "evaluator_targets.pt", patient_id=bundle.patient_id, split_hash=split_hash,
            grid=_target_grid(bundle.target_plane, preprocessing_hash=r4_preprocess), values=r4_target,
            valid_mask=r4_valid, segmentation=segmentation,
        )
        from .evaluate import run as evaluate_run
        from .audit import run as audit_run

        evaluations: dict[str, object] = {}
        audits: dict[str, object] = {}
        for name in ("r0", "e2_r4_p0"):
            variant_dir = output_dir / name
            target_path = variant_dir / "evaluator_targets.pt"
            plan = _evaluation_plan(evaluation_config_path, target_file=target_path, target_hash=_file_hash(target_path))
            plan_path = variant_dir / "evaluation_plan.json"
            _atomic_json(plan, plan_path)
            report = evaluate_run(
                plan_path=plan_path,
                predictions_dir=variant_dir / "predictions",
                output_dir=variant_dir / "evaluation",
            )
            metrics = report.get("metrics")
            if not isinstance(metrics, list) or not metrics:
                raise RuntimeError(f"{name} evaluator produced no metrics")
            metric = metrics[0]
            if metric.get("supported_voxels", 0) <= 0:
                raise RuntimeError(f"{name} evaluator found no supported target pixels")
            finite_metric_values = [
                metric.get(field)
                for field in ("mae", "rmse", "nmse", "psnr", "ssim", "ncc", "gradient_mae", "frequency_error")
            ]
            if not all(isinstance(value, (int, float)) and np.isfinite(float(value)) for value in finite_metric_values):
                raise RuntimeError(f"{name} evaluator produced non-finite metrics")
            evaluation_path = variant_dir / "evaluation" / "evaluation.json"
            audit_report = audit_run(variant_dir / "predictions")
            audit_path = variant_dir / "audit_report.json"
            _atomic_json(audit_report, audit_path)
            evaluations[name] = {"path": str(evaluation_path), **report}
            audits[name] = {"path": str(audit_path), **audit_report}
        peak_memory = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
        summary = {
            "schema": "smagm-brats21-real-smoke-v1",
            "claim_scope": config["claim_scope"],
            "source_kind": config["source_kind"],
            "patient_pseudonymous_id": pseudonymous_patient,
            "manifest_hash": bundle.manifest.manifest_hash,
            "split_hash": split_hash,
            "assignment_hash": bundle.assignment.assignment_hash,
            "split_registry_hash": split_registry.registry_hash,
            "context_count": len(bundle.assignment.context_ids),
            "modality_inventory": list(config["modalities"]),
            "target_modality": "flair",
            "target_orientation": bundle.manifest_json.get("source_geometry", {}).get("orientation"),
            "context_target_disjoint": not (set(bundle.assignment.context_ids) & set(bundle.assignment.target_ids)),
            "target_reveal_barrier_verified": True,
            "device": device_report,
            "seed": seed,
                    "optimizer": optimizer_name,
            "r0": r0_report,
            "r4_p0": r4_report,
            "evaluations": evaluations,
            "audits": audits,
            "evaluator_target": target_report,
            "runtime_seconds": time.perf_counter() - run_start,
            "peak_cuda_memory_bytes": peak_memory,
            "repository": git,
            "environment_hash": environment_hash,
            "t4_routing": False,
            "scientific_pass_recorded": False,
            "human_gate_decision": None,
            "non_claims": config["non_claims"],
        }
        # The assignment above intentionally avoids reading source volumes; use
        # a stable wall-clock value captured from the run directory metadata.
        summary["runtime_seconds"] = max(time.perf_counter() - run_start, 0.0)
        if logger is not None:
            logger.log({
                "evaluation/r0_mae": evaluations["r0"]["metrics"][0]["mae"],
                "evaluation/r4_p0_mae": evaluations["e2_r4_p0"]["metrics"][0]["mae"],
                "evaluation/r4_p0_supported_fraction": evaluations["e2_r4_p0"]["metrics"][0]["supported_fraction"],
                "runtime/peak_cuda_memory_bytes": peak_memory or 0,
            }, step=requested_steps + 1)
            wandb_finish = logger.finish(status="finished")
            summary["wandb"] = wandb_finish.to_dict()
        else:
            summary["wandb"] = {"mode": "unavailable", "run_id": None, "url": None}
        _atomic_json(summary, output_dir / "summary.json")
        return summary
    except Exception as error:
        failure = {"schema": "smagm-brats21-real-smoke-failure-v1", "failure_reason": f"{type(error).__name__}: {error}", "repository": git}
        _atomic_json(failure, output_dir / "failure.json")
        if logger is not None:
            logger.finish(status="failed", failure_reason=failure["failure_reason"])
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a one-patient BraTS21 receipt-gated real-data smoke")
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, default=_DEFAULT_EVAL_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run(
        config_path=args.config,
        prepared_dir=args.prepared_dir,
        output_dir=args.output_dir,
        evaluation_config_path=args.evaluation_config,
        allow_cpu_fallback=args.allow_cpu_fallback,
        steps=args.steps,
        wandb_mode=args.wandb_mode,
    )
    print(json.dumps(report, sort_keys=True) if args.json else json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
