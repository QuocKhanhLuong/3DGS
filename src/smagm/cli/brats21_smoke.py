"""Run the smallest receipt-gated BraTS21 real-data static smoke.

This command consumes only a prepared sparse derivative bundle. The active
product bundle keeps target intensity and evaluator segmentation deferred; its
receipt-gated target reader opens intensity only after prediction receipt
registration, while the evaluator segmentation reader is opened after
prediction-package serialization. The source
NIfTI root is never passed to the trainer, and segmentation is never passed to
the episode ledger, encoder, optimizer, or patient-state builder. The older
synthetic smoke preparation may carry an isolated evaluator payload and is not
the product path.
"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping

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
from ..experiments.complexity import (
    PhaseTiming,
    analytical_conv_linear_forward_flops,
    parameter_counts,
    peak_cuda_memory_bytes,
    profile_supported_operator_flops,
)
from ..features.encoder import EncoderConfig, EvidenceEncoder
from ..fields import SharedStructuralField, StructuralFieldConfig
from ..losses.reconstruction import ReconstructionLossConfig
from ..memory import PropagationConfig, SeedMemoryConfig
from ..reconstruction import (
    build_reconstruction_package,
    export_reconstruction_package,
    load_reconstruction_package,
    reconstruct_volume,
)
from ..reconstruction.uncertainty import support_uncertainty
from ..renderer import RenderConfig, RenderResult, SlabProfile
from ..state import save_patient_state
from ..training import (
    AnchorEvidenceProjector,
    AnchorEvidenceProjectorConfig,
    LegalEpisodeConfig,
    build_representation_episode_step,
)


_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _ROOT / "configs" / "experiments" / "brats21_real_smoke.json"
_DEFAULT_EVAL_CONFIG = _ROOT / "configs" / "evaluation" / "brats21_smoke_eval.json"
_PROVENANCE_GENERATED_ROOTS = frozenset({
    ".agenteam", ".ateam-worktrees", ".codex", ".pytest_cache", "build", "dist", "experiments", "htmlcov", "quality/reports",
})
_PROVENANCE_BINARY_SUFFIXES = frozenset({
    ".ckpt", ".dcm", ".h5", ".jpeg", ".jpg", ".nii", ".npy", ".npz", ".png", ".pt", ".pth", ".tif", ".tiff", ".webp",
})


def _skip_untracked_provenance(relative_path: str) -> bool:
    """Keep generated payloads and binary artifacts out of Git provenance hashing."""

    normalized = Path(relative_path)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise RuntimeError("untracked provenance path is not repository-relative")
    relative = normalized.as_posix()
    if any(relative == root or relative.startswith(root + "/") for root in _PROVENANCE_GENERATED_ROOTS):
        return True
    name = normalized.name.lower()
    return name.endswith(".nii.gz") or normalized.suffix.lower() in _PROVENANCE_BINARY_SUFFIXES


def _hash_untracked_provenance(root: Path, relative_paths: list[str]) -> tuple[str, tuple[str, ...]]:
    """Hash new source/config text while binding skipped artifact paths and sizes.

    Generated report/payload trees and volumetric/binary files have their own
    artifact hashes and must not be read merely to describe the repository
    state. Their relative path and byte size remain bound in the repository
    digest so the dirty provenance still records their presence.
    """

    digest = hashlib.sha256()
    skipped: list[str] = []
    resolved_root = root.resolve(strict=True)
    for relative in sorted(relative_paths):
        candidate = (resolved_root / relative).resolve()
        if resolved_root not in candidate.parents or not candidate.is_file():
            raise RuntimeError("untracked provenance inventory escaped the repository")
        normalized = Path(relative).as_posix()
        digest.update(normalized.encode("utf-8"))
        if _skip_untracked_provenance(normalized):
            skipped.append(normalized)
            digest.update(f"\\0skipped\\0{candidate.stat().st_size}".encode("utf-8"))
            continue
        digest.update(b"\\0content\\0")
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest(), tuple(skipped)


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


def _quarantine_partial_directory(path: Path) -> Path:
    """Move an incomplete substage aside so resume can rebuild it safely."""

    if not path.is_dir():
        raise ValueError(f"partial substage is not a directory: {path}")
    for attempt in range(100):
        candidate = path.with_name(f"{path.name}.incomplete-{time.time_ns()}-{attempt}")
        if candidate.exists():
            continue
        path.replace(candidate)
        return candidate
    raise FileExistsError(f"could not quarantine partial substage: {path}")


def _frozen_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous().clone() for name, value in module.state_dict().items()}


def _model_state_hash(module: torch.nn.Module) -> str:
    payload = b"".join(
        name.encode("utf-8") + value.detach().cpu().contiguous().numpy().tobytes()
        for name, value in module.state_dict().items()
    )
    return hashlib.sha256(payload).hexdigest()


def _global_model_binding_hash(config: Mapping[str, Any]) -> str:
    """Hash model/training semantics while ignoring episode-local cursors.

    A streamed cohort reuses the encoder, Gaussian head, StructuralField, and
    optimizer across patients. The episode split and step budget may differ
    for the diagnostic validation sweep, so they are intentionally excluded
    from this global-model binding. Patient manifests and target payloads are
    never part of the global checkpoint.
    """

    value = json.loads(json.dumps(dict(config)))
    value.pop("episode_split", None)
    value.pop("manifest_hash", None)
    value.pop("assignment_hash", None)
    value.pop("split_hash", None)
    training = value.get("training")
    if isinstance(training, dict):
        training.pop("steps", None)
        # Product runtime configs keep the episode step budget under the
        # nested training section. It is an episode-local cursor and must not
        # change the shared cohort-model binding.
        episode_training = training.get("training")
        if isinstance(episode_training, dict):
            episode_training.pop("steps", None)
    value.pop("runtime", None)
    value.pop("episode_split_hash", None)
    return _digest(value)


def _load_global_model_checkpoint(
    path: Path,
    *,
    config: Mapping[str, Any],
    split_hash: str,
    encoder: EvidenceEncoder,
    gaussian_head: FixedGaussianHead,
    field: SharedStructuralField,
    optimizer: Any,
    anchor_evidence_projector: AnchorEvidenceProjector | None = None,
) -> dict[str, object]:
    """Load shared cohort parameters without admitting patient/target data."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != "smagm-brats21-global-training-checkpoint-v1":
        raise ValueError("global model checkpoint schema is invalid")
    if payload.get("target_payload_not_in_checkpoint") is not True:
        raise ValueError("global model checkpoint must exclude target payloads")
    if payload.get("model_binding_hash") != _global_model_binding_hash(config):
        raise ValueError("global model checkpoint does not match the resolved model configuration")
    checkpoint_split_hash = payload.get("cohort_split_hash", payload.get("split_hash"))
    if checkpoint_split_hash != split_hash:
        raise ValueError("global model checkpoint does not match the resolved cohort split")
    update_index = payload.get("global_update_index")
    if not isinstance(update_index, int) or update_index < 0:
        raise ValueError("global model checkpoint update index is invalid")
    try:
        encoder.load_state_dict(payload["encoder"])
        gaussian_head.load_state_dict(payload["gaussian_head"])
        field.load_state_dict(payload["field"])
        saved_projector = payload.get("anchor_evidence_projector")
        if anchor_evidence_projector is None:
            if saved_projector is not None:
                raise ValueError("checkpoint contains a projector but this execution did not construct one")
        else:
            if not isinstance(saved_projector, Mapping):
                raise ValueError("checkpoint does not contain the declared anchor evidence projector")
            anchor_evidence_projector.load_state_dict(saved_projector)
        optimizer.load_state_dict(payload["optimizer"])
    except (KeyError, TypeError, RuntimeError, ValueError) as error:
        raise ValueError("global model checkpoint state is invalid") from error
    return {
        "path": str(path),
        "global_update_index": update_index,
        "source_patient_pseudonym": payload.get("source_patient_pseudonym"),
        "model_binding_hash": payload["model_binding_hash"],
    }


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
    untracked_digest, skipped_untracked = _hash_untracked_provenance(_ROOT, untracked)
    digest = hashlib.sha256(diff + untracked_digest.encode("ascii"))
    return {
        "repository_commit": commit,
        "repository_dirty": bool(status),
        "repository_dirty_entries": status[:100],
        "repository_diff_hash": digest.hexdigest(),
        "repository_provenance_skipped_entries": list(skipped_untracked[:100]),
    }


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "smagm-brats21-real-smoke-v1":
        raise ValueError("BraTS21 runner requires the brats21 real-smoke config schema")
    if config.get("t4_routing") is not False:
        raise ValueError("BraTS21 smoke must explicitly disable T4 routing")
    if config.get("encoder_variant") != "e2" or config.get("representation_variant") != "anchor_field":
        raise ValueError("the maintained real-data smoke is locked to E2 + R4")
    if config.get("propagation_variant") not in ("p0", "p1"):
        raise ValueError("the real-data runner supports only P0 or bounded P1")
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


def _target_grid(plane: PhysicalPlane, *, preprocessing_hash: str | None, modality_id: str = "flair") -> TargetGrid:
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
    return TargetGrid(matrix, (1, plane.shape_hw[0], plane.shape_hw[1]), (modality_id,), records)


def _source_grid_from_geometry(
    source_geometry: Mapping[str, Any],
    *,
    inplane_stride_vu: tuple[int, int],
    preprocessing_hash: str | None,
    modality_id: str,
) -> TargetGrid:
    """Build a downsampled full physical grid from source headers only.

    BraTS/NIfTI payloads are indexed ``[x, y, z]`` while repository volume
    tensors are ``[d, h, w] == [z, x, y]``.  The grid therefore maps ``w``
    through the source ``y`` affine column and ``h`` through the source ``x``
    column.  No image, target, or segmentation payload is consulted.
    """

    shape_raw = source_geometry.get("shape_xyz")
    affine_raw = source_geometry.get("affine")
    if not isinstance(shape_raw, (list, tuple)) or len(shape_raw) != 3:
        raise ValueError("full-grid source geometry must declare shape_xyz")
    shape_xyz = tuple(int(value) for value in shape_raw)
    if any(value <= 0 for value in shape_xyz):
        raise ValueError("full-grid source geometry shape_xyz must be positive")
    affine = np.asarray(affine_raw, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("full-grid source geometry affine must be finite 4x4")
    stride_v, stride_u = (int(value) for value in inplane_stride_vu)
    if stride_v <= 0 or stride_u <= 0:
        raise ValueError("full-grid inplane_stride_vu must be positive")
    if any(float(np.linalg.norm(affine[:3, column])) <= 0.0 for column in range(3)):
        raise ValueError("full-grid source geometry affine axes must be non-zero")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = affine[:3, 1] * stride_u
    matrix[:3, 1] = affine[:3, 0] * stride_v
    matrix[:3, 2] = affine[:3, 2]
    matrix[:3, 3] = affine[:3, 3]
    records = () if not preprocessing_hash else (preprocessing_hash,)
    return TargetGrid(
        tuple(tuple(float(value) for value in row) for row in matrix),
        (shape_xyz[2], (shape_xyz[0] + stride_v - 1) // stride_v, (shape_xyz[1] + stride_u - 1) // stride_u),
        (modality_id,),
        records,
    )


def _source_volume_bounds_from_geometry(
    source_geometry: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return physical RAS-mm bounds for source voxel centres from headers only."""

    shape_raw = source_geometry.get("shape_xyz")
    affine_raw = source_geometry.get("affine")
    if not isinstance(shape_raw, (list, tuple)) or len(shape_raw) != 3:
        raise ValueError("source volume bounds require shape_xyz")
    shape_xyz = tuple(int(value) for value in shape_raw)
    if any(value <= 0 for value in shape_xyz):
        raise ValueError("source volume bounds shape_xyz must be positive")
    affine = np.asarray(affine_raw, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("source volume bounds affine must be finite 4x4")
    if any(float(np.linalg.norm(affine[:3, column])) <= 0.0 for column in range(3)):
        raise ValueError("source volume bounds affine axes must be non-zero")
    corners = np.asarray([
        (x, y, z, 1.0)
        for x in (0, shape_xyz[0] - 1)
        for y in (0, shape_xyz[1] - 1)
        for z in (0, shape_xyz[2] - 1)
    ], dtype=np.float64)
    ras = (affine @ corners.T).T[:, :3]
    lower = torch.as_tensor(ras.min(axis=0), dtype=torch.float64)
    upper = torch.as_tensor(ras.max(axis=0), dtype=torch.float64)
    if not bool(torch.isfinite(lower).all()) or not bool(torch.isfinite(upper).all()) or not bool((lower < upper).all()):
        raise ValueError("source volume bounds must be finite and non-degenerate")
    return lower, upper


def _volume_from_render(
    prediction: RenderResult,
    *,
    patient_id: str,
    grid: TargetGrid,
    state_version: str,
    render_config: RenderConfig,
    propagation_uncertainty: float,
    modality_id: str = "flair",
) -> Any:
    intensity = prediction.intensity.detach().unsqueeze(0)
    support = prediction.support_mass.detach().unsqueeze(0)
    unsupported = prediction.unsupported_mask.detach().unsqueeze(0)
    uncertainty = support_uncertainty(support, unsupported, propagation_uncertainty=propagation_uncertainty)
    renderer_hash = _digest(render_config.renderer_version)
    from ..contracts.outputs import VolumeReconstruction, volume_output_hash

    artifact_hash = volume_output_hash(
        patient_id=patient_id,
        modality_id=modality_id,
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
        modality_id,
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


def _new_ledger(
    bundle: PreparedBraTS21,
    registry: PatientSplitRegistry,
    *,
    deferred_target_reader: Callable[[], bytes] | None = None,
) -> EpisodeLedger:
    readers = {bundle.target_id: deferred_target_reader} if deferred_target_reader is not None else None
    return EpisodeLedger(
        bundle.manifest,
        bundle.assignment,
        bundle.root,
        split_registry=registry,
        deferred_target_readers=readers,
    )


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
            lower_percentile=float(normal.get("lower_percentile", 1.0)),
            upper_percentile=float(normal.get("upper_percentile", 99.0)),
            output_min=float(normal.get("output_min", 0.0)),
            output_max=float(normal.get("output_max", 1.0)),
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
            tile_shape_hw=tuple(int(value) for value in renderer.get("tile_shape_hw", (32, 32))),
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
        structural_propagation_policy=str(raw.get("structural_propagation_policy", "tangent_only")),
        duplicate_radius_mm=float(raw["duplicate_radius_mm"]),
        uncertainty_growth_per_mm=float(raw["uncertainty_growth_per_mm"]),
        maximum_structural_primitives=int(raw["maximum_structural_primitives"]),
        maximum_volumetric_primitives=int(raw["maximum_volumetric_primitives"]),
        maximum_children_per_anchor=int(raw.get("maximum_children_per_anchor", 8)),
        maximum_patient_primitives=None if raw.get("maximum_patient_primitives") is None else int(raw["maximum_patient_primitives"]),
        maximum_uncertainty=None if raw.get("maximum_uncertainty") is None else float(raw["maximum_uncertainty"]),
        minimum_evidence_gain=float(raw.get("minimum_evidence_gain", 0.0)),
        minimum_cross_modality_agreement=float(raw.get("minimum_cross_modality_agreement", 0.0)),
        maximum_total_anchors=None if raw.get("maximum_total_anchors") is None else int(raw["maximum_total_anchors"]),
        structural_seed_budget=None if raw.get("structural_seed_budget") is None else int(raw["structural_seed_budget"]),
        volumetric_seed_budget=None if raw.get("volumetric_seed_budget") is None else int(raw["volumetric_seed_budget"]),
        propagation_reserved_budget=None if raw.get("propagation_reserved_budget") is None else int(raw["propagation_reserved_budget"]),
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


def _build_anchor_evidence_projector(
    config: Mapping[str, Any],
    *,
    device: torch.device,
) -> AnchorEvidenceProjector | None:
    """Build the declared typed adapter, retaining prefix only as an ablation."""

    training = dict(config["training"])
    adapter = str(training.get("gaussian_head_input_adapter", "anchor_evidence_prefix"))
    if adapter == "anchor_evidence_prefix":
        return None
    if adapter != "anchor_evidence_projector":
        raise ValueError("Gaussian-head input adapter must be anchor_evidence_projector or anchor_evidence_prefix")
    raw = training.get("anchor_evidence_projector")
    if not isinstance(raw, Mapping):
        raise ValueError("anchor_evidence_projector adapter requires a declared projector config")
    projector_config = AnchorEvidenceProjectorConfig(
        evidence_dim=int(raw["evidence_dim"]),
        head_input_dim=int(raw["head_input_dim"]),
        bias=bool(raw.get("bias", True)),
    )
    if projector_config.evidence_dim != int(config["field"]["evidence_dim"]):
        raise ValueError("anchor evidence projector input must equal the declared field evidence dimension")
    if projector_config.head_input_dim != int(training["gaussian_head_input_dim"]):
        raise ValueError("anchor evidence projector output must equal gaussian_head_input_dim")
    return AnchorEvidenceProjector(projector_config).to(device=device, dtype=torch.float32)


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
    commitments = [item for item in events if item["event"] == "COMMIT_TARGET"]
    if len(commitments) != 1:
        raise RuntimeError("the maintained product episode must contain exactly one target commitment")
    return {
        "events": events,
        "target_commitment_hash": _digest(commitments[0]),
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
    write_nifti: bool = False,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        manifest_path = output_dir / "package.json"
        if not manifest_path.is_file():
            raise FileExistsError(f"prediction package output is non-empty without an immutable manifest: {output_dir}")
        existing_package, existing_volumes = load_reconstruction_package(output_dir)
        if len(existing_volumes) != 1 or existing_volumes[0].artifact_hash != volume.artifact_hash:
            raise ValueError("existing prediction package does not match the resumed reconstruction")
        return {
            "package_hash": existing_package.package_hash,
            "patient_state_version": existing_package.patient_state_version,
            "prediction_package": str(output_dir),
            "unsupported_fraction": float(volume.unsupported_mask.float().mean()),
            "supported_fraction": float((~volume.unsupported_mask).float().mean()),
            "prediction_package_size_bytes": sum(path.stat().st_size for path in output_dir.iterdir() if path.is_file()),
        }
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
    export_reconstruction_package(package, (volume,), output_dir, write_nifti=write_nifti)
    package_bytes = sum(path.stat().st_size for path in output_dir.iterdir() if path.is_file())
    return {
        "package_hash": package.package_hash,
        "patient_state_version": package.patient_state_version,
        "prediction_package": str(output_dir),
        "unsupported_fraction": float(volume.unsupported_mask.float().mean()),
        "supported_fraction": float((~volume.unsupported_mask).float().mean()),
        "prediction_package_size_bytes": package_bytes,
    }


def _context_gap_mm(target_plane: PhysicalPlane, context_planes: tuple[PhysicalPlane, ...]) -> float | None:
    if not context_planes:
        return None
    normal = torch.as_tensor(target_plane.signed_normal_ras, dtype=torch.float64)
    target_origin = torch.as_tensor(target_plane.pixel_center_origin_ras_mm, dtype=torch.float64)
    positions = sorted({
        round(float((torch.as_tensor(plane.pixel_center_origin_ras_mm, dtype=torch.float64) - target_origin) @ normal), 9)
        for plane in context_planes
    })
    target_position = 0.0
    left = [value for value in positions if value < target_position]
    right = [value for value in positions if value > target_position]
    if not left or not right:
        return None
    return float(right[0] - left[-1])


def _anchor_observability_map(state: Any, grid: TargetGrid) -> torch.Tensor | None:
    anchors = getattr(state, "anchors", None)
    if anchors is None or anchors.count == 0:
        return None
    dtype = anchors.centers_ras_mm.dtype
    device = anchors.centers_ras_mm.device
    depth, height, width = grid.shape_dhw
    d, h, w = torch.meshgrid(
        torch.arange(depth, dtype=dtype, device=device),
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    homogeneous = torch.stack((w, h, d, torch.ones_like(d)), dim=0).reshape(4, -1)
    matrix = torch.as_tensor(grid.index_to_ras_mm, dtype=dtype, device=device)
    points = (matrix @ homogeneous)[:3].transpose(0, 1)
    distances = torch.cdist(points, anchors.centers_ras_mm)
    nearest = torch.argmin(distances, dim=1)
    raw = anchors.observability[nearest]
    evidence_count = 1.0 - torch.exp(-raw[:, 0].clamp_min(0.0))
    weighted_coverage = 1.0 - torch.exp(-raw[:, 1].clamp_min(0.0))
    agreement = torch.exp(-raw[:, 2].clamp_min(0.0))
    return ((evidence_count + weighted_coverage + agreement) / 3.0).reshape(depth, height, width).clamp(0.0, 1.0)


def _write_targets(
    path: Path,
    *,
    patient_id: str,
    split_hash: str,
    grid: TargetGrid,
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    segmentation: torch.Tensor | None,
    modality_id: str,
    context_planes: tuple[PhysicalPlane, ...] = (),
    context_gap_mm: float | None = None,
    local_observability: torch.Tensor | None = None,
) -> dict[str, object]:
    if values.ndim != 2:
        raise ValueError("the target plane must remain [H,W] before evaluator packaging")
    target_values = values.detach().cpu().unsqueeze(0).contiguous()
    target_valid = valid_mask.detach().cpu().unsqueeze(0).contiguous()
    segmentation_payload = None
    if segmentation is not None:
        segmentation_payload = segmentation.detach().cpu().to(torch.uint8)
        if segmentation_payload.ndim == 2:
            segmentation_payload = segmentation_payload.unsqueeze(0)
        segmentation_payload = segmentation_payload.contiguous()
    if target_values.shape != tuple(grid.shape_dhw) or target_valid.shape != target_values.shape:
        raise ValueError("evaluator target does not match the held-out target grid")
    if segmentation_payload is not None and segmentation_payload.shape != target_values.shape:
        raise ValueError("evaluator segmentation does not match the held-out target grid")
    if local_observability is not None:
        local_observability = local_observability.detach().cpu().to(torch.float32).contiguous()
        if local_observability.shape != target_values.shape or not bool(torch.isfinite(local_observability).all()) or bool((local_observability < 0).any()) or bool((local_observability > 1).any()):
            raise ValueError("evaluator local observability must match the target grid and remain in [0,1]")
    _atomic_torch(
        {
            "schema": "smagm-audit-targets-v1",
            "targets": [{
                "patient_id": patient_id,
                "split_hash": split_hash,
                "modality_id": modality_id,
                "grid": grid.to_canonical_dict(),
                "values": target_values,
                "valid_mask": target_valid,
                "context_planes": [plane.to_canonical_dict() for plane in context_planes],
                "context_gap_mm": context_gap_mm,
                "local_observability": local_observability,
            }],
            "segmentation": segmentation_payload,
            "segmentation_labels": [0, 1, 2, 4] if segmentation_payload is not None else None,
            "segmentation_evaluator_only": True,
        },
        path,
    )
    return {
        "path": str(path),
        "sha256": _file_hash(path),
        "segmentation_lesion_fraction": None
        if segmentation_payload is None
        else float((segmentation_payload > 0).float().mean()),
        "segmentation_access_phase": "not_available"
        if segmentation_payload is None
        else "after_prediction_receipt_and_serialization",
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


def _encoder_conv_telemetry(
    encoder: EvidenceEncoder,
    bundle: PreparedBraTS21,
) -> dict[str, object]:
    """Count only E2 Conv2d/Linear forward work for legal context shapes.

    The product encoder is invoked independently for each legal context image.
    This helper mirrors those actual tensors with zero-valued legal-shape
    tensors and never opens a target payload.  It intentionally excludes the
    analytic bank, renderer, loss, backward, and optimizer work.
    """

    micro_cnn = encoder.micro_cnn
    if micro_cnn is None:
        return {
            "encoder_forward_flops_2flop_per_mac": None,
            "encoder_forward_flops_scope": "not_applicable_no_learned_e2_micro_cnn",
            "encoder_forward_flops_context_count": len(bundle.assignment.context_ids),
        }
    parameter = next(micro_cnn.parameters(), None)
    if parameter is None:
        raise RuntimeError("E2 micro-CNN has no parameters")
    total = 0
    shape_counts = _context_shape_batches(bundle)
    shapes: list[list[int]] = []
    per_shape: dict[str, int] = {}
    was_training = micro_cnn.training
    micro_cnn.eval()
    try:
        with torch.no_grad():
            for (height, width), count in sorted(shape_counts.items()):
                tensor = torch.zeros((count, 7, height, width), device=parameter.device, dtype=parameter.dtype)
                report = analytical_conv_linear_forward_flops(micro_cnn, tensor)
                count_flops = int(report["forward_flops_2flop_per_mac"])
                total += count_flops
                key = f"{count}x7x{height}x{width}"
                per_shape[key] = count_flops
                shapes.append([count, 7, height, width])
    finally:
        micro_cnn.train(was_training)
    return {
        "encoder_forward_flops_2flop_per_mac": total,
        "encoder_forward_flops_scope": "E2 micro-CNN forward Conv2d/Linear only; 2 FLOPs per MAC; legal context images only",
        "encoder_forward_flops_context_count": len(bundle.assignment.context_ids),
        "encoder_forward_flops_input_shapes": shapes,
        "encoder_forward_flops_by_shape": per_shape,
    }


def _context_shape_batches(bundle: PreparedBraTS21) -> dict[tuple[int, int], int]:
    """Return the actual legal context shapes grouped into CNN batches."""

    shape_counts: dict[tuple[int, int], int] = {}
    for observation_id in bundle.assignment.context_ids:
        height, width = bundle.manifest.metadata(observation_id).plane.shape_hw
        key = (int(height), int(width))
        shape_counts[key] = shape_counts.get(key, 0) + 1
    return shape_counts


def _resolved_encoder_conv_telemetry(
    encoder: EvidenceEncoder,
    bundle: PreparedBraTS21,
    config: Mapping[str, Any],
    *,
    cache_owner: Any | None = None,
) -> dict[str, object]:
    """Compute exact encoder work once per distinct legal context shape batch."""

    diagnostics = dict(config.get("diagnostics", {}))
    if not bool(diagnostics.get("analytical_encoder_flops", False)):
        return {
            "encoder_forward_flops_2flop_per_mac": None,
            "encoder_forward_flops_scope": "disabled_by_diagnostics.analytical_encoder_flops",
            "encoder_forward_flops_context_count": len(bundle.assignment.context_ids),
            "encoder_forward_flops_input_shapes": [],
            "encoder_forward_flops_by_shape": {},
            "encoder_forward_flops_measurement": "not_invoked",
        }
    expected_shapes = [
        [count, 7, height, width]
        for (height, width), count in sorted(_context_shape_batches(bundle).items())
    ]
    cached = None if cache_owner is None else getattr(cache_owner, "encoder_flop_telemetry", None)
    if isinstance(cached, Mapping) and cached.get("encoder_forward_flops_input_shapes") == expected_shapes:
        telemetry = dict(cached)
        telemetry["encoder_forward_flops_measurement"] = "process_cache_from_legal_context_shape_batch"
        return telemetry
    if cached is not None and not isinstance(cached, Mapping):
        raise TypeError("cohort encoder FLOP telemetry cache must be a mapping or None")
    telemetry = _encoder_conv_telemetry(encoder, bundle)
    telemetry["encoder_forward_flops_measurement"] = "computed_from_legal_context_shape_batch"
    if cache_owner is not None:
        setattr(cache_owner, "encoder_flop_telemetry", dict(telemetry))
    return telemetry


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

    def state_dict(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "first": [value.detach().cpu().clone() for value in self.first],
            "second": [value.detach().cpu().clone() for value in self.second],
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping) or int(state.get("step_index", -1)) < 0:
            raise ValueError("fallback Adam checkpoint state is invalid")
        first = state.get("first")
        second = state.get("second")
        if not isinstance(first, list) or not isinstance(second, list) or len(first) != len(self.parameters) or len(second) != len(self.parameters):
            raise ValueError("fallback Adam checkpoint moments do not match parameters")
        for parameter, saved_first, saved_second in zip(self.parameters, first, second):
            if not isinstance(saved_first, torch.Tensor) or not isinstance(saved_second, torch.Tensor):
                raise ValueError("fallback Adam checkpoint moments must be tensors")
            if saved_first.shape != parameter.shape or saved_second.shape != parameter.shape:
                raise ValueError("fallback Adam checkpoint moment shapes do not match parameters")
        self.step_index = int(state["step_index"])
        self.first = [value.to(device=parameter.device, dtype=parameter.dtype) for parameter, value in zip(self.parameters, first)]
        self.second = [value.to(device=parameter.device, dtype=parameter.dtype) for parameter, value in zip(self.parameters, second)]


def _make_optimizer(
    encoder: EvidenceEncoder,
    gaussian_head: FixedGaussianHead,
    field: SharedStructuralField,
    learning_rate: float,
    anchor_evidence_projector: AnchorEvidenceProjector | None = None,
) -> tuple[Any, str]:
    """Create the native optimizer over every declared shared learned module."""

    parameters = tuple(encoder.parameters()) + tuple(gaussian_head.parameters()) + tuple(field.parameters())
    if anchor_evidence_projector is not None:
        parameters = parameters + tuple(anchor_evidence_projector.parameters())
    try:
        return torch.optim.Adam(parameters, lr=learning_rate), "torch.optim.Adam"
    except ImportError:
        return _AdamFallback(parameters, learning_rate), "adam-fallback-torch-dynamo-import-error"


def _optimizer_learning_rate(optimizer: Any) -> float:
    """Read the configured scalar learning rate from native or fallback Adam."""

    if hasattr(optimizer, "param_groups"):
        groups = getattr(optimizer, "param_groups")
        if isinstance(groups, list) and groups and isinstance(groups[0], Mapping) and "lr" in groups[0]:
            value = float(groups[0]["lr"])
        else:
            raise ValueError("optimizer has no readable parameter-group learning rate")
    elif hasattr(optimizer, "learning_rate"):
        value = float(getattr(optimizer, "learning_rate"))
    else:
        raise ValueError("optimizer has no readable learning rate")
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("optimizer learning rate must be finite and positive")
    return value


def _save_r4_progress_checkpoint(
    *,
    path: Path,
    bundle: PreparedBraTS21,
    config_hash: str,
    split_hash: str,
    propagation: PropagationConfig,
    steps: int,
    completed_steps: int,
    reports: list[dict[str, object]],
    encoder: EvidenceEncoder,
    gaussian_head: FixedGaussianHead,
    field: SharedStructuralField,
    optimizer: Any,
    anchor_evidence_projector: AnchorEvidenceProjector | None = None,
    cohort_split_hash: str | None = None,
) -> None:
    if completed_steps < 0 or completed_steps > steps or len(reports) != completed_steps:
        raise ValueError("R4 progress checkpoint cursor and report count are inconsistent")
    _atomic_torch(
        {
            "schema": "smagm-brats21-r4-progress-v2" if anchor_evidence_projector is not None else "smagm-brats21-r4-progress-v1",
            "config_hash": config_hash,
            "manifest_hash": bundle.manifest.manifest_hash,
            "split_hash": split_hash,
            "cohort_split_hash": cohort_split_hash or split_hash,
            "assignment_hash": bundle.assignment.assignment_hash,
            "propagation_variant": propagation.variant,
            "steps": steps,
            "completed_steps": completed_steps,
            "reports": reports,
            "encoder": _frozen_state_dict(encoder),
            "gaussian_head": _frozen_state_dict(gaussian_head),
            "field": _frozen_state_dict(field),
            "anchor_evidence_projector": None if anchor_evidence_projector is None else _frozen_state_dict(anchor_evidence_projector),
            "optimizer": optimizer.state_dict(),
            "target_payload_not_in_checkpoint": True,
        },
        path,
    )


def _load_r4_progress_checkpoint(
    *,
    path: Path,
    bundle: PreparedBraTS21,
    config_hash: str,
    split_hash: str,
    propagation: PropagationConfig,
    steps: int,
    encoder: EvidenceEncoder,
    gaussian_head: FixedGaussianHead,
    field: SharedStructuralField,
    optimizer: Any,
    anchor_evidence_projector: AnchorEvidenceProjector | None = None,
    cohort_split_hash: str | None = None,
) -> tuple[int, list[dict[str, object]]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") not in ("smagm-brats21-r4-progress-v1", "smagm-brats21-r4-progress-v2"):
        raise ValueError("R4 progress checkpoint schema is invalid")
    bindings = {
        "config_hash": config_hash,
        "manifest_hash": bundle.manifest.manifest_hash,
        "split_hash": split_hash,
        "assignment_hash": bundle.assignment.assignment_hash,
        "propagation_variant": propagation.variant,
        "steps": steps,
        "cohort_split_hash": cohort_split_hash or split_hash,
    }
    if any(payload.get(key) != value for key, value in bindings.items()):
        raise ValueError("R4 progress checkpoint does not match the current product episode")
    completed_steps = payload.get("completed_steps")
    reports = payload.get("reports")
    if not isinstance(completed_steps, int) or not 0 <= completed_steps <= steps or not isinstance(reports, list) or len(reports) != completed_steps:
        raise ValueError("R4 progress checkpoint cursor is invalid")
    if payload.get("target_payload_not_in_checkpoint") is not True:
        raise ValueError("R4 progress checkpoint must not contain target payloads")
    try:
        encoder.load_state_dict(payload["encoder"])
        gaussian_head.load_state_dict(payload["gaussian_head"])
        field.load_state_dict(payload["field"])
        saved_projector = payload.get("anchor_evidence_projector")
        if anchor_evidence_projector is None:
            if saved_projector is not None:
                raise ValueError("R4 progress checkpoint contains a projector but this execution did not construct one")
        else:
            if not isinstance(saved_projector, Mapping):
                raise ValueError("R4 progress checkpoint does not contain the declared anchor evidence projector")
            anchor_evidence_projector.load_state_dict(saved_projector)
        optimizer.load_state_dict(payload["optimizer"])
    except (KeyError, TypeError, RuntimeError, ValueError) as error:
        raise ValueError("R4 progress checkpoint model or optimizer state is invalid") from error
    return completed_steps, [dict(item) for item in reports]


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
    cohort_split_hash: str | None = None,
    git: Mapping[str, object],
    environment_hash: str,
    deferred_target_reader: Callable[[], bytes] | None,
) -> tuple[dict[str, object], Any, torch.Tensor, torch.Tensor, str]:
    start = time.perf_counter()
    ledger = _new_ledger(bundle, registry, deferred_target_reader=deferred_target_reader)
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
    target_modality = bundle.manifest.metadata(bundle.target_id).modality_id
    grid = _target_grid(bundle.target_plane, preprocessing_hash=result.preprocessing_record_hash, modality_id=target_modality)
    inference_start = time.perf_counter()
    volume = _volume_from_render(
        result.prediction,
        patient_id=bundle.patient_id,
        grid=grid,
        state_version=state_version,
        render_config=episode_config.renderer,
        propagation_uncertainty=0.0,
        modality_id=target_modality,
    )
    inference_seconds = time.perf_counter() - inference_start
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
    checkpoint_sha256 = _file_hash(output_dir / "checkpoint.pt")
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
        "inference_wall_time_seconds": inference_seconds,
        "per_plane_latency_seconds": inference_seconds,
        "full_grid_latency_seconds": None,
        "full_grid_status": "not_requested_target_plane_only",
        "checkpoint": str(output_dir / "checkpoint.pt"),
        "checkpoint_sha256": checkpoint_sha256,
        "prediction_package": str(package_dir),
        "preprocessing_record_hash": result.preprocessing_record_hash,
        "target_commitment_hash": events["target_commitment_hash"],
        **package,
    }
    _atomic_json(report, output_dir / "summary.json")
    return report, result, result.target.detach(), result.target_valid_mask.detach(), result.preprocessing_record_hash or ""


def _r4_source_geometry(
    bundle: PreparedBraTS21,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, tuple[int, int, int] | None]:
    """Read only prepared header geometry required for bounded propagation."""

    patient_bounds_min_ras_mm, patient_bounds_max_ras_mm = _source_volume_bounds_from_geometry(
        bundle.manifest_json.get("source_geometry", {})
    )
    source_geometry = bundle.manifest_json.get("source_geometry", {})
    source_shape_xyz: tuple[int, int, int] | None = None
    source_affine_ras_from_index: torch.Tensor | None = None
    if isinstance(source_geometry, Mapping) and source_geometry:
        shape_raw = source_geometry.get("shape_xyz")
        affine_raw = source_geometry.get("affine")
        if not isinstance(shape_raw, (list, tuple)) or len(shape_raw) != 3:
            raise ValueError("R4 source geometry must declare shape_xyz for oriented propagation containment")
        source_shape_xyz = tuple(int(value) for value in shape_raw)
        source_affine_ras_from_index = torch.as_tensor(affine_raw, dtype=torch.float64)
    return patient_bounds_min_ras_mm, patient_bounds_max_ras_mm, source_affine_ras_from_index, source_shape_xyz


def run_cohort_training_episode(
    *,
    bundle: PreparedBraTS21,
    config: Mapping[str, Any],
    config_hash: str,
    cohort_split_hash: str,
    cohort_model: Any,
    deferred_target_reader: Callable[[], bytes] | None,
    training_updates: bool,
    profile_supported_operator_flops_enabled: bool = False,
) -> dict[str, object]:
    """Execute one legal temporary patient episode against one global model.

    This deliberately performs neither R0, package serialization, isolated
    evaluation, media export, W&B creation, nor checkpoint writing.  The
    product controller performs those only at declared validation/final
    cadence.  The receipt-gated target remains local to this function and is
    absent from its return value and the global model snapshot.
    """

    required = ("encoder", "gaussian_head", "structural_field", "optimizer", "zero_grad", "optimizer_step", "run_with_optional_profiler", "global_step")
    if any(not hasattr(cohort_model, name) for name in required):
        raise TypeError("cohort_model does not implement the required process-level ownership contract")
    encoder = cohort_model.encoder
    gaussian_head = cohort_model.gaussian_head
    field = cohort_model.structural_field
    projector = getattr(cohort_model, "evidence_projector", None)
    if not isinstance(encoder, EvidenceEncoder) or not isinstance(gaussian_head, FixedGaussianHead) or not isinstance(field, SharedStructuralField):
        raise TypeError("cohort model does not own the declared E2, Gaussian-head, and StructuralField modules")
    if projector is not None and not isinstance(projector, AnchorEvidenceProjector):
        raise TypeError("cohort model evidence_projector must be AnchorEvidenceProjector or None")

    episode_config = _episode_config(config)
    bootstrap = _bootstrap_config(config)
    seed_memory = _seed_memory_config(config)
    propagation = _propagation_config(config)
    registry = PatientSplitRegistry.create((bundle.manifest,))
    bounds_min, bounds_max, source_affine, source_shape = _r4_source_geometry(bundle)
    phase = PhaseTiming()
    collect_phase_timing = bool(
        dict(config.get("diagnostics", {})).get("phase_timing", False)
        and int(getattr(cohort_model, "global_step", 0)) == 0
    )
    start = time.perf_counter()
    if collect_phase_timing and torch.cuda.is_available():
        torch.cuda.synchronize()

    def _operation() -> Any:
        ledger = _new_ledger(bundle, registry, deferred_target_reader=deferred_target_reader)
        with phase.measure("representation_forward_wall_time_ms"):
            built = build_representation_episode_step(
                ledger=ledger,
                assignment=bundle.assignment,
                target_id=bundle.target_id,
                representation_variant="r4",
                propagation_variant=propagation.variant,
                config=episode_config,
                encoder=encoder,
                gaussian_head=gaussian_head,
                local_field=field,
                field_maximum_neighbors=int(config["field"]["maximum_neighbors"]),
                registration_id=str(config["anchor"]["registration_id"]),
                bootstrap_config=bootstrap,
                seed_memory_config=seed_memory,
                propagation_config=propagation,
                patient_bounds_min_ras_mm=bounds_min,
                patient_bounds_max_ras_mm=bounds_max,
                source_affine_ras_from_index=source_affine,
                source_shape_xyz=source_shape,
                anchor_evidence_projector=projector,
                gaussian_head_input_adapter=str(config.get("training", {}).get("gaussian_head_input_adapter", "anchor_evidence_prefix")),
                collect_phase_timing=collect_phase_timing,
            )
        if training_updates:
            if collect_phase_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            with phase.measure("backward_wall_time_ms"):
                built.loss.total.backward()
                if collect_phase_timing and torch.cuda.is_available():
                    torch.cuda.synchronize()
        return built, ledger

    if training_updates:
        cohort_model.zero_grad()
    (result, ledger), profiler = cohort_model.run_with_optional_profiler(
        _operation,
        enabled=bool(training_updates and profile_supported_operator_flops_enabled),
        scope="one legal cohort episode forward, loss, and backward; optimizer excluded",
    )
    if training_updates:
        encoder_grad, encoder_ok = _gradient_norm(encoder)
        head_grad, head_ok = _gradient_norm(gaussian_head)
        field_grad, field_ok = _gradient_norm(field)
        projector_grad, projector_ok = (0.0, True) if projector is None else _gradient_norm(projector)
        if not all((encoder_ok, head_ok, field_ok, projector_ok)) or not np.isfinite(float(result.loss.total.detach().cpu())):
            raise FloatingPointError(
                "BraTS21 cohort R4 gradient contract failed: "
                f"encoder={encoder_grad:.6e} (ok={encoder_ok}), "
                f"gaussian_head={head_grad:.6e} (ok={head_ok}), "
                f"field={field_grad:.6e} (ok={field_ok}), "
                f"evidence_projector={projector_grad:.6e} (ok={projector_ok}), "
                f"loss={float(result.loss.total.detach().cpu()):.6e}"
            )
    else:
        encoder_grad = head_grad = field_grad = projector_grad = 0.0
    field_hash = _model_state_hash(field)
    if result.patient_state is None or result.patient_state.field_model_hash != field_hash:
        raise RuntimeError("temporary patient state does not bind the exact pre-update StructuralField")
    events = _event_report(ledger)
    state_before_update = getattr(cohort_model, "state_hash", None)
    if training_updates:
        if collect_phase_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
        with phase.measure("optimizer_wall_time_ms"):
            global_step = int(cohort_model.optimizer_step())
            if collect_phase_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
    else:
        global_step = int(cohort_model.global_step)
    if collect_phase_timing and torch.cuda.is_available():
        torch.cuda.synchronize()
    full_step_host_ms = (time.perf_counter() - start) * 1000.0
    # Only the diagnostic first episode is synchronized.  A later host-side
    # enqueue duration is useful operationally but must not be presented as a
    # device-complete wall time.
    full_step_ms: float | None = full_step_host_ms if collect_phase_timing else None
    transactions = result.propagation_transactions
    static_timing = result.phase_timing_ms or {}
    telemetry = _resolved_encoder_conv_telemetry(
        encoder,
        bundle,
        config,
        cache_owner=cohort_model,
    )
    peak = peak_cuda_memory_bytes()
    report: dict[str, object] = {
        "schema": "smagm-brats21-cohort-episode-v1",
        "global_step": global_step,
        "training_update_applied": training_updates,
        "loss": float(result.loss.total.detach().cpu()),
        "legal_pixel_count": result.loss.legal_pixel_count,
        "target_valid_pixel_count": result.loss.target_valid_pixel_count,
        "supported_fraction": result.loss.supported_fraction,
        "unsupported_fraction": float(result.prediction.unsupported_mask.float().mean().detach().cpu()),
        "encoder_gradient_norm": encoder_grad,
        "gaussian_head_gradient_norm": head_grad,
        "field_gradient_norm": field_grad,
        "evidence_projector_gradient_norm": projector_grad,
        "anchor_count": result.patient_state.anchors.count,
        "structural_gaussian_count": result.patient_state.memory.structural.gaussians.count,
        "volumetric_gaussian_count": result.patient_state.memory.volumetric.gaussians.count,
        "primitive_count": result.patient_state.memory.primitive_count,
        "pixel_gaussian_candidate_pairs": result.prediction.pixel_gaussian_candidate_pairs,
        "propagation_proposal_count": sum(item.proposal_count for item in transactions),
        "propagation_child_count": sum(item.accepted_count for item in transactions),
        "propagation_rejected_budget": sum(item.rejected_budget for item in transactions),
        "propagation_rejected_duplicate": sum(item.rejected_duplicate for item in transactions),
        "propagation_rejected_out_of_bounds": sum(item.rejected_out_of_bounds for item in transactions),
        "propagation_rejected_unsupported": sum(item.rejected_unsupported for item in transactions),
        "propagation_rejected_uncertainty": sum(item.rejected_uncertainty for item in transactions),
        "propagation_rejected_no_gain": sum(item.rejected_no_meaningful_gain for item in transactions),
        "propagation_rejected_invalid": sum(item.rejected_invalid for item in transactions),
        "receipt_hash": result.receipt_hash,
        "target_commitment_hash": events["target_commitment_hash"],
        "target_reveal_after_prediction": True,
        "preprocessing_record_hash": result.preprocessing_record_hash,
        "field_model_hash_before_update": field_hash,
        "global_model_state_hash_before_update": state_before_update,
        "full_step_wall_time_ms": full_step_ms,
        "full_step_host_enqueue_time_ms": full_step_host_ms,
        "phase_timing_synchronized": collect_phase_timing,
        "encoder_wall_time_ms": static_timing.get("encoder_wall_time_ms"),
        "anchor_build_wall_time_ms": static_timing.get("anchor_build_wall_time_ms"),
        "field_query_wall_time_ms": static_timing.get("field_query_wall_time_ms"),
        "propagation_wall_time_ms": static_timing.get("propagation_wall_time_ms"),
        "renderer_wall_time_ms": static_timing.get("renderer_wall_time_ms"),
        "loss_wall_time_ms": static_timing.get("loss_wall_time_ms"),
        "backward_wall_time_ms": phase.value("backward_wall_time_ms") if training_updates else 0.0,
        "optimizer_wall_time_ms": phase.value("optimizer_wall_time_ms") if training_updates else 0.0,
        "representation_forward_wall_time_ms": phase.value("representation_forward_wall_time_ms"),
        **telemetry,
        **profiler,
        **peak,
    }
    return report


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
    anchor_evidence_projector: AnchorEvidenceProjector | None,
    bootstrap: AnchorBootstrapConfig,
    seed_memory: SeedMemoryConfig,
    propagation: PropagationConfig,
    output_dir: Path,
    config_hash: str,
    split_hash: str,
    cohort_split_hash: str | None = None,
    git: Mapping[str, object],
    environment_hash: str,
    steps: int,
    deferred_target_reader: Callable[[], bytes] | None,
    checkpoint_interval_steps: int,
    resume: bool,
    training_updates: bool = True,
    profile_supported_operator_flops_enabled: bool = False,
    telemetry_cache_owner: Any | None = None,
) -> tuple[dict[str, object], Any, torch.Tensor, torch.Tensor, str]:
    start = time.perf_counter()
    if checkpoint_interval_steps <= 0:
        raise ValueError("checkpoint_interval_steps must be positive")
    resolved_cohort_split_hash = cohort_split_hash or split_hash
    global_model_binding_hash = _global_model_binding_hash(config)
    learning_rate = _optimizer_learning_rate(optimizer)
    experiment_name = str(config.get("experiment_name", "brats21-static-diagnostic"))
    learned_modules: dict[str, torch.nn.Module] = {
        "encoder": encoder,
        "gaussian_head": gaussian_head,
        "structural_field": field,
    }
    if anchor_evidence_projector is not None:
        learned_modules["evidence_projector"] = anchor_evidence_projector
    model_complexity = parameter_counts(learned_modules)
    model_complexity.update(
        _resolved_encoder_conv_telemetry(
            encoder,
            bundle,
            config,
            cache_owner=telemetry_cache_owner,
        )
    )
    model_complexity["profiled_supported_operator_flops"] = None
    model_complexity["profiler_scope"] = "not_invoked"
    model_complexity["profiler_operator_coverage"] = "partial_unknown_torch_profiler_supported_operators_only"
    model_complexity["profiler_enabled"] = False
    print(
        f"[experiment] name={experiment_name} "
        f"parameters={model_complexity['parameters']} "
        f"trainable_parameters={model_complexity['trainable_parameters']} "
        f"encoder_forward_flops_2flop_per_mac={model_complexity['encoder_forward_flops_2flop_per_mac']}",
        flush=True,
    )
    patient_bounds_min_ras_mm, patient_bounds_max_ras_mm = _source_volume_bounds_from_geometry(
        bundle.manifest_json.get("source_geometry", {})
    )
    source_geometry = bundle.manifest_json.get("source_geometry", {})
    source_shape_xyz: tuple[int, int, int] | None = None
    source_affine_ras_from_index: torch.Tensor | None = None
    if isinstance(source_geometry, Mapping) and source_geometry:
        shape_raw = source_geometry.get("shape_xyz")
        affine_raw = source_geometry.get("affine")
        if not isinstance(shape_raw, (list, tuple)) or len(shape_raw) != 3:
            raise ValueError("R4 source geometry must declare shape_xyz for oriented propagation containment")
        source_shape_xyz = tuple(int(value) for value in shape_raw)
        source_affine_ras_from_index = torch.as_tensor(affine_raw, dtype=torch.float64)
    reports: list[dict[str, object]] = []
    state_encoder_snapshot: dict[str, torch.Tensor] | None = None
    state_field_snapshot: dict[str, torch.Tensor] | None = None
    state_encoder_hash: str | None = None
    state_field_hash: str | None = None
    progress_path = output_dir / "progress_checkpoint.pt"
    if not training_updates and progress_path.exists():
        raise FileExistsError("validation-only R4 cannot consume a training progress checkpoint")
    if progress_path.exists():
        if not resume:
            raise FileExistsError(f"R4 progress checkpoint exists; resume is required: {progress_path}")
        start_step, saved_reports = _load_r4_progress_checkpoint(
            path=progress_path,
            bundle=bundle,
            config_hash=config_hash,
            split_hash=split_hash,
            propagation=propagation,
            steps=steps,
            encoder=encoder,
            gaussian_head=gaussian_head,
            field=field,
            optimizer=optimizer,
            anchor_evidence_projector=anchor_evidence_projector,
            cohort_split_hash=resolved_cohort_split_hash,
        )
        reports = saved_reports
    elif training_updates:
        if resume and any(output_dir.iterdir()):
            raise FileNotFoundError("R4 output is non-empty but has no resumable progress checkpoint")
        start_step = 0
        _save_r4_progress_checkpoint(
            path=progress_path,
            bundle=bundle,
            config_hash=config_hash,
            split_hash=split_hash,
            propagation=propagation,
            steps=steps,
            completed_steps=0,
            reports=reports,
            encoder=encoder,
            gaussian_head=gaussian_head,
            field=field,
            optimizer=optimizer,
            anchor_evidence_projector=anchor_evidence_projector,
            cohort_split_hash=resolved_cohort_split_hash,
        )
    else:
        start_step = 0
    for step_index in range(start_step, steps if training_updates else 1):
        if training_updates:
            optimizer.zero_grad(set_to_none=True)
        ledger = _new_ledger(bundle, registry, deferred_target_reader=deferred_target_reader)

        def _execute_episode_step() -> Any:
            with torch.set_grad_enabled(training_updates):
                built = build_representation_episode_step(
                    ledger=ledger,
                    assignment=bundle.assignment,
                    target_id=bundle.target_id,
                    representation_variant="r4",
                    propagation_variant=propagation.variant,
                    config=episode_config,
                    encoder=encoder,
                    gaussian_head=gaussian_head,
                    local_field=field,
                    field_maximum_neighbors=int(config["field"]["maximum_neighbors"]),
                    registration_id=str(config["anchor"]["registration_id"]),
                    bootstrap_config=bootstrap,
                    seed_memory_config=seed_memory,
                    propagation_config=propagation,
                    patient_bounds_min_ras_mm=patient_bounds_min_ras_mm,
                    patient_bounds_max_ras_mm=patient_bounds_max_ras_mm,
                    source_affine_ras_from_index=source_affine_ras_from_index,
                    source_shape_xyz=source_shape_xyz,
                    anchor_evidence_projector=anchor_evidence_projector,
                    gaussian_head_input_adapter=str(config.get("training", {}).get("gaussian_head_input_adapter", "anchor_evidence_prefix")),
                )
            if training_updates:
                built.loss.total.backward()
            return built

        if training_updates and step_index == start_step and profile_supported_operator_flops_enabled:
            result, profile = profile_supported_operator_flops(
                _execute_episode_step,
                enabled=True,
                scope="one legal episode forward, loss, and backward; optimizer excluded",
            )
            model_complexity.update(profile)
            print(
                "[experiment] profiled_supported_operator_flops="
                f"{profile.get('profiled_supported_operator_flops')} "
                f"scope={profile.get('profiler_scope')}",
                flush=True,
            )
        else:
            result = _execute_episode_step()
        if training_updates:
            encoder_grad, encoder_ok = _gradient_norm(encoder)
            gaussian_head_grad, gaussian_head_ok = _gradient_norm(gaussian_head)
            field_grad, field_ok = _gradient_norm(field)
            if anchor_evidence_projector is None:
                projector_grad, projector_ok = 0.0, True
            else:
                projector_grad, projector_ok = _gradient_norm(anchor_evidence_projector)
            if not encoder_ok or not gaussian_head_ok or not field_ok or not projector_ok or not np.isfinite(result.loss.total.detach().cpu().item()):
                raise FloatingPointError(
                    "BraTS21 R4 gradient contract failed: "
                    f"encoder={encoder_grad:.6e} (ok={encoder_ok}), "
                    f"gaussian_head={gaussian_head_grad:.6e} (ok={gaussian_head_ok}), "
                    f"field={field_grad:.6e} (ok={field_ok}), "
                    f"evidence_projector={projector_grad:.6e} (ok={projector_ok}), "
                    f"loss={float(result.loss.total.detach().cpu()):.6e}"
                )
        else:
            encoder_grad = gaussian_head_grad = field_grad = projector_grad = 0.0
        state_encoder_snapshot = _frozen_state_dict(encoder)
        state_field_snapshot = _frozen_state_dict(field)
        state_encoder_hash = encoder.state_hash()
        state_field_hash = _model_state_hash(field)
        if state_field_hash != result.patient_state.field_model_hash:
            raise RuntimeError("patient state does not bind the exact field snapshot used before optimizer update")
        events = _event_report(ledger)
        if training_updates:
            optimizer.step()
        if training_updates and torch.cuda.is_available():
            torch.cuda.synchronize()
        reports.append({
            "step": step_index + 1 if training_updates else 0,
            "training_update_applied": training_updates,
            "learning_rate": learning_rate,
            "loss": float(result.loss.total.detach().cpu()),
            "legal_pixel_count": result.loss.legal_pixel_count,
            "target_valid_pixel_count": result.loss.target_valid_pixel_count,
            "supported_fraction": result.loss.supported_fraction,
            "unsupported_fraction": float(result.prediction.unsupported_mask.float().mean().detach().cpu()),
            "encoder_gradient_norm": encoder_grad,
            "gaussian_head_gradient_norm": gaussian_head_grad,
            "field_gradient_norm": field_grad,
            "evidence_projector_gradient_norm": projector_grad,
            "experiment_name": experiment_name,
            "parameter_count": model_complexity["parameters"],
            "trainable_parameter_count": model_complexity["trainable_parameters"],
            "encoder_forward_flops_2flop_per_mac": model_complexity["encoder_forward_flops_2flop_per_mac"],
            "profiled_supported_operator_flops": model_complexity["profiled_supported_operator_flops"],
            "profiler_scope": model_complexity["profiler_scope"],
            "profiler_operator_coverage": model_complexity["profiler_operator_coverage"],
            "primitive_count": result.patient_state.memory.primitive_count,
            "anchor_count": result.patient_state.anchors.count,
            "state_version": result.patient_state.state_version,
            "receipt_hash": result.receipt_hash,
            "target_commitment_hash": events["target_commitment_hash"],
            "event_order": [item["event"] for item in events["events"]],
            "encoder_state_hash": state_encoder_hash,
            "field_model_hash": state_field_hash,
            "propagation_proposal_count": sum(item.proposal_count for item in result.propagation_transactions),
            "propagation_child_count": sum(item.accepted_count for item in result.propagation_transactions),
            "propagation_rejected_out_of_bounds": sum(item.rejected_out_of_bounds for item in result.propagation_transactions),
            "propagation_rejected_unsupported": sum(item.rejected_unsupported for item in result.propagation_transactions),
            "propagation_rejected_duplicate": sum(item.rejected_duplicate for item in result.propagation_transactions),
            "propagation_rejected_budget": sum(item.rejected_budget for item in result.propagation_transactions),
            "propagation_rejected_uncertainty": sum(item.rejected_uncertainty for item in result.propagation_transactions),
            "propagation_rejected_invalid": sum(item.rejected_invalid for item in result.propagation_transactions),
            "propagation_rejected_no_meaningful_gain": sum(item.rejected_no_meaningful_gain for item in result.propagation_transactions),
        })
        completed_steps = step_index + 1
        if training_updates and (completed_steps == steps or completed_steps % checkpoint_interval_steps == 0):
            _save_r4_progress_checkpoint(
                path=progress_path,
                bundle=bundle,
                config_hash=config_hash,
                split_hash=split_hash,
                propagation=propagation,
                steps=steps,
                completed_steps=completed_steps,
                reports=reports,
                encoder=encoder,
                gaussian_head=gaussian_head,
                field=field,
                optimizer=optimizer,
                anchor_evidence_projector=anchor_evidence_projector,
                cohort_split_hash=resolved_cohort_split_hash,
            )

    # The patient state used for export is built from the parameters after
    # the requested optimizer updates.  This makes a progress checkpoint at
    # an optimizer boundary sufficient to resume and finalize without
    # replaying or storing target intensity.
    with torch.no_grad():
        final_ledger = _new_ledger(bundle, registry, deferred_target_reader=deferred_target_reader)
        final_result = build_representation_episode_step(
            ledger=final_ledger,
            assignment=bundle.assignment,
            target_id=bundle.target_id,
            representation_variant="r4",
            propagation_variant=propagation.variant,
            config=episode_config,
            encoder=encoder,
            gaussian_head=gaussian_head,
            local_field=field,
            field_maximum_neighbors=int(config["field"]["maximum_neighbors"]),
            registration_id=str(config["anchor"]["registration_id"]),
            bootstrap_config=bootstrap,
            seed_memory_config=seed_memory,
            propagation_config=propagation,
            patient_bounds_min_ras_mm=patient_bounds_min_ras_mm,
            patient_bounds_max_ras_mm=patient_bounds_max_ras_mm,
            source_affine_ras_from_index=source_affine_ras_from_index,
            source_shape_xyz=source_shape_xyz,
            anchor_evidence_projector=anchor_evidence_projector,
            gaussian_head_input_adapter=str(config.get("training", {}).get("gaussian_head_input_adapter", "anchor_evidence_prefix")),
        )
    last_result = final_result
    last_state = final_result.patient_state
    last_ledger = final_ledger
    state_encoder_snapshot = _frozen_state_dict(encoder)
    state_field_snapshot = _frozen_state_dict(field)
    state_encoder_hash = encoder.state_hash()
    state_field_hash = _model_state_hash(field)
    if state_field_hash != last_state.field_model_hash:
        raise RuntimeError("final patient state does not bind the exact StructuralField snapshot")
    patient_state_path = output_dir / "patient_state.pt"
    save_patient_state(last_state, patient_state_path)
    checkpoint = {
        "schema": "smagm-brats21-r4-checkpoint-v1",
        "config_hash": config_hash,
        "manifest_hash": bundle.manifest.manifest_hash,
        "split_hash": split_hash,
        "cohort_split_hash": resolved_cohort_split_hash,
        "assignment_hash": bundle.assignment.assignment_hash,
        "patient_state_path": patient_state_path.name,
        "patient_state_version": last_state.state_version,
        "patient_state_field_model_hash": last_state.field_model_hash,
        "encoder": state_encoder_snapshot,
        "field": state_field_snapshot,
        "gaussian_head": _frozen_state_dict(gaussian_head),
        "anchor_evidence_projector": None if anchor_evidence_projector is None else _frozen_state_dict(anchor_evidence_projector),
        "encoder_for_patient_state_hash": state_encoder_hash,
        "field_for_patient_state_hash": state_field_hash,
        "gaussian_head_hash": _model_state_hash(gaussian_head),
        "optimizer": optimizer.state_dict(),
        "model_binding_hash": global_model_binding_hash,
        "target_payload_not_in_checkpoint": True,
        "global_model_eligible": True,
        "training_updates_applied": training_updates,
        "validation_only": not training_updates,
        "patient_state_snapshot": "exact parameters used before target commitment for the final validation episode"
        if not training_updates
        else "exact parameters used before target commitment for the final post-update evaluation episode",
        "optimizer_updates_before_snapshot": steps if training_updates else 0,
        "post_snapshot_optimizer_updates": 0,
        "progress_checkpoint": str(progress_path),
        "steps": steps if training_updates else 0,
        **git,
    }
    _atomic_torch(checkpoint, output_dir / "checkpoint.pt")
    checkpoint_sha256 = _file_hash(output_dir / "checkpoint.pt")
    checkpoint_size_bytes = (output_dir / "checkpoint.pt").stat().st_size
    target_modality = bundle.manifest.metadata(bundle.target_id).modality_id
    grid = _target_grid(bundle.target_plane, preprocessing_hash=last_result.preprocessing_record_hash, modality_id=target_modality)
    inference_start = time.perf_counter()
    volume = reconstruct_volume(
        last_state,
        grid,
        modality_id=bundle.manifest.metadata(bundle.target_id).modality_id,
        depth_chunk_size=1,
        render_config=episode_config.renderer,
    )
    inference_seconds = time.perf_counter() - inference_start
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
    reconstruction_config = dict(config.get("reconstruction", {}))
    full_grid_latency: float | None = None
    full_grid_status = "not_requested_target_plane_only"
    full_grid_package: dict[str, object] | None = None
    full_grid_shape: tuple[int, int, int] | None = None
    if str(reconstruction_config.get("target_grid", "held_out_target_plane")) == "full_source_grid":
        raw_stride = config.get("inplane_stride_vu", config.get("interpolation", {}).get("stride_vu"))
        if not isinstance(raw_stride, (list, tuple)) or len(raw_stride) != 2:
            raise ValueError("full-grid reconstruction requires declared inplane_stride_vu")
        full_grid = _source_grid_from_geometry(
            bundle.manifest_json.get("source_geometry", {}),
            inplane_stride_vu=(int(raw_stride[0]), int(raw_stride[1])),
            preprocessing_hash=last_result.preprocessing_record_hash,
            modality_id=target_modality,
        )
        full_grid_shape = full_grid.shape_dhw
        full_start = time.perf_counter()
        full_volume = reconstruct_volume(
            last_state,
            full_grid,
            modality_id=target_modality,
            depth_chunk_size=int(reconstruction_config.get("full_grid_depth_chunk_size", reconstruction_config.get("depth_chunk_size", 1))),
            render_config=episode_config.renderer,
        )
        full_grid_latency = time.perf_counter() - full_start
        full_grid_package = _package_and_export(
            output_dir=output_dir / "full_grid_predictions",
            volume=full_volume,
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
            runtime_seconds=full_grid_latency,
            write_nifti=bool(reconstruction_config.get("write_nifti", False)),
        )
        full_grid_status = "computed_and_serialized"
    elif str(reconstruction_config.get("target_grid", "held_out_target_plane")) != "held_out_target_plane":
        raise ValueError("reconstruction target_grid must be held_out_target_plane or full_source_grid")
    final_events = _event_report(last_ledger)
    _atomic_json(final_events, output_dir / "episode_ledger.json")
    report = {
        "variant": f"e2_r4_{propagation.variant}",
        "encoder_variant": "e2",
        "representation_variant": "anchor_field",
        "propagation_variant": propagation.variant,
        "steps": reports,
        "loss": float(last_result.loss.total.detach().cpu()),
        "final_evaluation_loss": float(last_result.loss.total.detach().cpu()),
        "supported_fraction": last_result.loss.supported_fraction,
        "unsupported_fraction": float(last_result.prediction.unsupported_mask.float().mean().detach().cpu()),
        "encoder_gradient_norm": reports[-1]["encoder_gradient_norm"],
        "gaussian_head_gradient_norm": reports[-1]["gaussian_head_gradient_norm"],
        "field_gradient_norm": reports[-1]["field_gradient_norm"],
        "experiment_name": experiment_name,
        "model_complexity": model_complexity,
        "parameter_count": model_complexity["parameters"],
        "trainable_parameter_count": model_complexity["trainable_parameters"],
        "encoder_forward_flops_2flop_per_mac": model_complexity.get("encoder_forward_flops_2flop_per_mac"),
        "profiled_supported_operator_flops": model_complexity.get("profiled_supported_operator_flops"),
        "profiler_scope": model_complexity.get("profiler_scope"),
        "profiler_operator_coverage": model_complexity.get("profiler_operator_coverage"),
        "learning_rate": learning_rate,
        "primitive_count": last_state.memory.primitive_count,
        "anchor_count": last_state.anchors.count,
        "structural_gaussian_count": last_state.memory.structural.gaussians.count,
        "volumetric_gaussian_count": last_state.memory.volumetric.gaussians.count,
        "propagation_proposal_count": sum(item.proposal_count for item in last_result.propagation_transactions),
        "propagation_child_count": sum(item.accepted_count for item in last_result.propagation_transactions),
        "propagation_rejected_out_of_bounds": sum(item.rejected_out_of_bounds for item in last_result.propagation_transactions),
        "propagation_rejected_unsupported": sum(item.rejected_unsupported for item in last_result.propagation_transactions),
        "propagation_rejected_duplicate": sum(item.rejected_duplicate for item in last_result.propagation_transactions),
        "propagation_rejected_budget": sum(item.rejected_budget for item in last_result.propagation_transactions),
        "propagation_rejected_uncertainty": sum(item.rejected_uncertainty for item in last_result.propagation_transactions),
        "propagation_rejected_invalid": sum(item.rejected_invalid for item in last_result.propagation_transactions),
        "propagation_rejected_no_meaningful_gain": sum(item.rejected_no_meaningful_gain for item in last_result.propagation_transactions),
        "runtime_seconds": time.perf_counter() - start,
        "inference_wall_time_seconds": inference_seconds,
        "per_plane_latency_seconds": inference_seconds / max(volume.grid.shape_dhw[0], 1),
        "full_grid_latency_seconds": full_grid_latency,
        "full_grid_status": full_grid_status,
        "full_grid_shape_dhw": full_grid_shape,
        "full_grid_prediction_package": None if full_grid_package is None else full_grid_package["prediction_package"],
        "full_grid_package_hash": None if full_grid_package is None else full_grid_package["package_hash"],
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "checkpoint": str(output_dir / "checkpoint.pt"),
        "checkpoint_sha256": checkpoint_sha256,
        "progress_checkpoint": str(progress_path),
        "checkpoint_interval_steps": checkpoint_interval_steps,
        "resumed_from_step": start_step,
        "training_updates_applied": training_updates,
        "validation_only": not training_updates,
        "optimizer_updates_applied": steps if training_updates else 0,
        "patient_state": str(patient_state_path),
        "preprocessing_record_hash": last_result.preprocessing_record_hash,
        "target_commitment_hash": final_events["target_commitment_hash"],
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
    deferred_target_reader: Callable[[], bytes] | None = None,
    deferred_segmentation_reader: Callable[[], bytes] | None = None,
    initial_global_checkpoint: Path | None = None,
    resume: bool = False,
    validation_only: bool = False,
    shared_cohort_model: Any | None = None,
    external_logger: Any | None = None,
    profile_supported_operator_flops_enabled: bool | None = None,
) -> dict[str, object]:
    config, config_hash = _load_config(config_path)
    if wandb_mode is not None:
        if wandb_mode not in ("disabled", "offline", "online"):
            raise ValueError("wandb_mode must be disabled, offline, or online")
        config["wandb"] = dict(config.get("wandb", {}))
        config["wandb"]["mode"] = wandb_mode
        config_hash = _digest(config)
    requested_steps = int(steps if steps is not None else config["training"]["steps"])
    if requested_steps <= 0:
        raise ValueError("BraTS21 training steps must be positive")
    if config.get("execution_mode", "smoke") == "smoke" and requested_steps > 5:
        raise ValueError("the diagnostic smoke mode is limited to 2-5 steps; use the product runner for longer runs")
    if validation_only and initial_global_checkpoint is None and shared_cohort_model is None:
        raise ValueError("validation-only execution requires a final global model checkpoint")
    bundle = load_prepared_bundle(prepared_dir)
    sampling = dict(config.get("sampling", {}))
    expected_context_count = int(sampling.get("context_planes_per_modality", 1)) * len(tuple(config["modalities"]))
    if len(bundle.assignment.context_ids) != expected_context_count or len(bundle.assignment.target_ids) != 1:
        raise ValueError(f"BraTS21 smoke requires exactly {expected_context_count} context observations and one target")
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
    product_execution = config.get("execution_mode", "smoke") == "product"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not product_execution or not resume:
            raise FileExistsError(f"real-data smoke output is non-empty: {output_dir}")
    elif not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
    git = _git_metadata()
    split_registry = PatientSplitRegistry.create((bundle.manifest,))
    prepared_meta = json.loads((prepared_dir / "prepared.json").read_text(encoding="utf-8"))
    # The patient derivative has its own episode split hash. Product runs
    # additionally bind the shared model to the complete deterministic
    # cohort split hash supplied by the product resolver.
    split_hash = str(prepared_meta["split_hash"])
    cohort_split_hash = str(config.get("cohort_split_hash") or config.get("split_hash") or split_hash)
    pseudonymous_patient = _pseudonymous_patient(bundle.patient_id, bundle.manifest.manifest_hash)
    selected_planes = tuple(
        item for item in bundle.manifest_json.get("selected_planes", ())
        if isinstance(item, Mapping)
    )
    context_plane_positions_mm = sorted({
        round(float(item["physical_position_mm"]), 9)
        for item in selected_planes
        if item.get("role") == "context" and item.get("physical_position_mm") is not None
    })
    target_plane_positions = [
        round(float(item["physical_position_mm"]), 9)
        for item in selected_planes
        if item.get("role") == "target" and item.get("physical_position_mm") is not None
    ]
    target_plane_position_mm = target_plane_positions[0] if target_plane_positions else None
    sampling_protocol_hash = bundle.manifest_json.get("sampling_protocol_hash")
    episode_config = _episode_config(config)
    bootstrap = _bootstrap_config(config)
    seed_memory = _seed_memory_config(config)
    propagation = _propagation_config(config)
    interpolation = _interpolation_config(config)
    if shared_cohort_model is None:
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
        anchor_evidence_projector = _build_anchor_evidence_projector(config, device=device)
        optimizer, optimizer_name = _make_optimizer(
            encoder,
            head,
            field,
            float(config["training"]["learning_rate"]),
            anchor_evidence_projector,
        )
    else:
        if initial_global_checkpoint is not None:
            raise ValueError("a shared cohort model must be restored by its owning product controller")
        encoder = getattr(shared_cohort_model, "encoder", None)
        head = getattr(shared_cohort_model, "gaussian_head", None)
        field = getattr(shared_cohort_model, "structural_field", None)
        anchor_evidence_projector = getattr(shared_cohort_model, "evidence_projector", None)
        optimizer = getattr(shared_cohort_model, "optimizer", None)
        if not isinstance(encoder, EvidenceEncoder) or not isinstance(head, FixedGaussianHead) or not isinstance(field, SharedStructuralField):
            raise TypeError("shared cohort model does not own the expected encoder, Gaussian head, and StructuralField")
        if anchor_evidence_projector is not None and not isinstance(anchor_evidence_projector, AnchorEvidenceProjector):
            raise TypeError("shared cohort model has an invalid anchor evidence projector")
        if not callable(getattr(optimizer, "step", None)):
            raise TypeError("shared cohort model has an invalid optimizer")
        optimizer_name = "process-owned-cohort-optimizer"
    global_model_binding_hash = _global_model_binding_hash(config)
    global_model_input: dict[str, object] | None = None
    if initial_global_checkpoint is not None:
        global_model_input = _load_global_model_checkpoint(
            initial_global_checkpoint.resolve(strict=True),
            config=config,
            split_hash=cohort_split_hash,
            encoder=encoder,
            gaussian_head=head,
            field=field,
            optimizer=optimizer,
            anchor_evidence_projector=anchor_evidence_projector,
        )
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
    preview_modules: dict[str, torch.nn.Module] = {
        "encoder": encoder,
        "gaussian_head": head,
        "structural_field": field,
    }
    if anchor_evidence_projector is not None:
        preview_modules["evidence_projector"] = anchor_evidence_projector
    model_complexity_preview = parameter_counts(preview_modules)
    resolved["runtime"] = {
        "actual_device": device_report["actual_device"],
        "allow_cpu_fallback": bool(allow_cpu_fallback or config["training"].get("allow_cpu_fallback", False)),
        "prepared_dir_not_uploaded": True,
        "patient_pseudonymous_id": pseudonymous_patient,
        "split_registry_hash": split_registry.registry_hash,
        "prepared_manifest_hash": bundle.manifest.manifest_hash,
        "optimizer": optimizer_name,
        "experiment_name": str(config.get("experiment_name", "brats21-static-diagnostic")),
        "model_complexity": model_complexity_preview,
    }
    _atomic_json(resolved, output_dir / "resolved_config.json")
    _atomic_json({
        "schema": "smagm-brats21-real-smoke-manifest-v1",
        "patient_pseudonymous_id": pseudonymous_patient,
        "manifest_hash": bundle.manifest.manifest_hash,
        "split_hash": split_hash,
        "cohort_split_hash": cohort_split_hash,
        "assignment_hash": bundle.assignment.assignment_hash,
        "context_observation_ids": list(bundle.assignment.context_ids),
        "target_observation_ids": list(bundle.assignment.target_ids),
        "contains_target_payloads": False,
        "contains_segmentation": False,
        "target_plane": bundle.target_plane.to_canonical_dict(),
    }, output_dir / "eval_manifest.json")
    recovered_partial_outputs: list[str] = []
    logger = external_logger
    owns_logger = False
    external_logger_step = (
        int(getattr(shared_cohort_model, "global_step"))
        if external_logger is not None and shared_cohort_model is not None
        else None
    )
    try:
        requested_wandb_mode = str(config.get("wandb", {}).get("mode", "disabled")).strip().lower()
        if logger is None and shared_cohort_model is not None:
            # A validation/final-evaluation episode must never spin up a
            # second W&B run.  The process-level controller may inject its
            # existing logger; otherwise this helper stays unlogged.
            if requested_wandb_mode != "disabled":
                raise RuntimeError("a shared cohort model requires an externally owned W&B logger when logging is enabled")
        elif logger is None:
            try:
                from ..experiments.wandb import WandbLogger

                logger = WandbLogger(
                    config=resolved,
                    run_name=f"{str(config.get('experiment_name', 'brats21-static-diagnostic'))}-{pseudonymous_patient}",
                    run_dir=output_dir,
                    mode=requested_wandb_mode,
                    metadata={
                        "schema": config["schema"],
                        "repository_commit": git["repository_commit"],
                        "repository_dirty": git["repository_dirty"],
                        "seed": seed,
                        "encoder_variant": "e2",
                        "representation_variant": "anchor_field",
                        "propagation_variant": str(config["propagation_variant"]),
                        "patient_pseudonymous_id": pseudonymous_patient,
                        "cohort_hash": config.get("cohort_hash"),
                        "split_hash": split_hash,
                        "assignment_hash": bundle.assignment.assignment_hash,
                        "sampling_protocol_hash": sampling_protocol_hash,
                        "context_count": len(bundle.assignment.context_ids),
                        "context_plane_positions_mm": context_plane_positions_mm,
                        "target_plane_position_mm": target_plane_position_mm,
                        "modality_inventory": list(config["modalities"]),
                        "target_modality": str(config["target_modality"]),
                        "target_orientation": bundle.manifest_json.get("source_geometry", {}).get("orientation"),
                        "source_kind": config["source_kind"],
                        "experiment_name": str(config.get("experiment_name", "brats21-static-diagnostic")),
                        "parameter_count": model_complexity_preview["parameters"],
                        "trainable_parameter_count": model_complexity_preview["trainable_parameters"],
                    },
                )
                logger.start()
                owns_logger = True
                if requested_wandb_mode != "disabled" and logger.mode == "disabled":
                    raise RuntimeError(
                        f"W&B mode {requested_wandb_mode!r} was requested but no active W&B run was created: "
                        f"{logger.fallback_reason or 'the optional client is unavailable'}"
                    )
                # W&B scalar history accepts finite numeric values only. The
                # shared global logger already records process-level values.
                logger.log({
                    "run/seed": seed,
                    "run/context_count": len(bundle.assignment.context_ids),
                    "training/learning_rate": float(config["training"]["learning_rate"]),
                    "model/parameter_count": int(model_complexity_preview["parameters"]),
                    "model/trainable_parameter_count": int(model_complexity_preview["trainable_parameters"]),
                }, step=0)
            except ImportError as error:
                if requested_wandb_mode != "disabled":
                    raise RuntimeError(
                        f"W&B mode {requested_wandb_mode!r} was requested but the optional wandb package is unavailable"
                    ) from error
                logger = None
        r0_dir = output_dir / "r0"
        r0_execution_dir = r0_dir
        if r0_dir.exists() and any(r0_dir.iterdir()):
            if (r0_dir / "summary.json").exists():
                if not resume:
                    raise FileExistsError(f"real-data smoke output is non-empty and cannot be resumed safely: {r0_dir}")
                r0_execution_dir = output_dir / "r0_resume_replay"
                if r0_execution_dir.exists() and any(r0_execution_dir.iterdir()):
                    raise FileExistsError(f"R0 resume replay output is non-empty: {r0_execution_dir}")
                r0_execution_dir.mkdir()
            elif resume:
                quarantined = _quarantine_partial_directory(r0_dir)
                recovered_partial_outputs.append(str(quarantined.relative_to(output_dir)))
                r0_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise FileExistsError(f"R0 output is non-empty and cannot be resumed safely: {r0_dir}")
        else:
            r0_execution_dir.mkdir(parents=True, exist_ok=True)
        r0_report, r0_result, r0_target, r0_valid, r0_preprocess = _run_r0(
            bundle=bundle, registry=split_registry, config=config, episode_config=episode_config,
            interpolation=interpolation, output_dir=r0_execution_dir, config_hash=config_hash,
            split_hash=split_hash, git=git, environment_hash=environment_hash,
            deferred_target_reader=deferred_target_reader,
        )
        if logger is not None:
            logger.log({
                "r0/loss": r0_report["loss"],
                "r0/supported_fraction": r0_report["supported_fraction"],
                "r0/unsupported_fraction": r0_report["unsupported_fraction"],
                "r0/primitive_count": r0_report["primitive_count"],
                "r0/runtime_seconds": r0_report["runtime_seconds"],
            }, step=0 if external_logger_step is None else external_logger_step)
        r4_name = f"e2_r4_{propagation.variant}"
        r4_dir = output_dir / r4_name
        r4_progress_path = r4_dir / "progress_checkpoint.pt"
        r4_summary_path = r4_dir / "summary.json"
        if r4_dir.exists() and any(r4_dir.iterdir()) and validation_only and r4_progress_path.exists() and not r4_summary_path.exists():
            if not resume:
                raise FileExistsError(f"validation-only R4 output contains a training progress checkpoint: {r4_dir}")
            quarantined = _quarantine_partial_directory(r4_dir)
            recovered_partial_outputs.append(str(quarantined.relative_to(output_dir)))
        if r4_dir.exists() and any(r4_dir.iterdir()) and not r4_progress_path.exists() and not r4_summary_path.exists():
            if not resume:
                raise FileExistsError(f"R4 output is non-empty and cannot be resumed safely: {r4_dir}")
            quarantined = _quarantine_partial_directory(r4_dir)
            recovered_partial_outputs.append(str(quarantined.relative_to(output_dir)))
        r4_dir.mkdir(parents=True, exist_ok=True)
        r4_report, r4_result, r4_target, r4_valid, r4_preprocess = _run_r4(
            bundle=bundle, registry=split_registry, config=config, episode_config=episode_config,
            encoder=encoder, gaussian_head=head, field=field, optimizer=optimizer,
            anchor_evidence_projector=anchor_evidence_projector,
            bootstrap=bootstrap, seed_memory=seed_memory, propagation=propagation,
            output_dir=r4_dir, config_hash=config_hash, split_hash=split_hash,
            cohort_split_hash=cohort_split_hash, git=git,
            environment_hash=environment_hash, steps=requested_steps,
            deferred_target_reader=deferred_target_reader,
            checkpoint_interval_steps=int(config.get("checkpointing", {}).get("interval_steps", 1)),
            resume=resume,
            training_updates=not validation_only,
            profile_supported_operator_flops_enabled=(
                bool(dict(config.get("diagnostics", {})).get("profile_supported_operator_flops", False))
                if profile_supported_operator_flops_enabled is None
                else bool(profile_supported_operator_flops_enabled)
            ),
            telemetry_cache_owner=shared_cohort_model,
        )
        if logger is not None:
            for item in r4_report["steps"]:
                logger.log({
                    f"{r4_name}/loss": item["loss"],
                    f"{r4_name}/learning_rate": item.get("learning_rate", _optimizer_learning_rate(optimizer)),
                    f"{r4_name}/supported_fraction": item["supported_fraction"],
                    f"{r4_name}/unsupported_fraction": item["unsupported_fraction"],
                    f"{r4_name}/encoder_gradient_norm": item["encoder_gradient_norm"],
                    f"{r4_name}/gaussian_head_gradient_norm": item["gaussian_head_gradient_norm"],
                    f"{r4_name}/field_gradient_norm": item["field_gradient_norm"],
                    f"{r4_name}/evidence_projector_gradient_norm": item["evidence_projector_gradient_norm"],
                    f"{r4_name}/primitive_count": item["primitive_count"],
                    f"{r4_name}/anchor_count": item["anchor_count"],
                    f"{r4_name}/propagation_child_count": item["propagation_child_count"],
                    f"{r4_name}/propagation_proposal_count": item["propagation_proposal_count"],
                    f"{r4_name}/propagation_rejected_out_of_bounds": item["propagation_rejected_out_of_bounds"],
                    f"{r4_name}/propagation_rejected_unsupported": item["propagation_rejected_unsupported"],
                    f"{r4_name}/propagation_rejected_duplicate": item["propagation_rejected_duplicate"],
                    f"{r4_name}/propagation_rejected_budget": item["propagation_rejected_budget"],
                    f"{r4_name}/propagation_rejected_uncertainty": item["propagation_rejected_uncertainty"],
                    f"{r4_name}/propagation_rejected_invalid": item["propagation_rejected_invalid"],
                    f"{r4_name}/propagation_rejected_no_meaningful_gain": item["propagation_rejected_no_meaningful_gain"],
                }, step=int(item["step"]) if external_logger_step is None else external_logger_step)
        if bundle.segmentation_payload_deferred:
            if deferred_segmentation_reader is None:
                raise RuntimeError("prepared bundle defers segmentation but no receipt-gated evaluator reader was supplied")
            segmentation = torch.from_numpy(
                np.load(BytesIO(deferred_segmentation_reader()), allow_pickle=False)
            ).to(torch.uint8)
        elif bundle.segmentation_payload_path is None:
            segmentation = None
        else:
            segmentation = torch.from_numpy(np.load(bundle.segmentation_payload_path, allow_pickle=False)).to(torch.uint8)
        context_planes = tuple(bundle.manifest.metadata(item).plane for item in bundle.assignment.context_ids)
        context_gap = _context_gap_mm(bundle.target_plane, context_planes)
        r0_grid = _target_grid(bundle.target_plane, preprocessing_hash=r0_preprocess, modality_id=str(config["target_modality"]))
        r4_grid = _target_grid(bundle.target_plane, preprocessing_hash=r4_preprocess, modality_id=str(config["target_modality"]))
        r4_observability = _anchor_observability_map(r4_result.patient_state, r4_grid)
        target_report = _write_targets(
            r0_dir / "evaluator_targets.pt", patient_id=bundle.patient_id, split_hash=split_hash,
            grid=r0_grid, values=r0_target, valid_mask=r0_valid, segmentation=segmentation,
            modality_id=str(config["target_modality"]), context_planes=context_planes, context_gap_mm=context_gap,
        )
        _write_targets(
            r4_dir / "evaluator_targets.pt", patient_id=bundle.patient_id, split_hash=split_hash,
            grid=r4_grid, values=r4_target, valid_mask=r4_valid, segmentation=segmentation,
            modality_id=str(config["target_modality"]), context_planes=context_planes, context_gap_mm=context_gap,
            local_observability=r4_observability,
        )
        from .evaluate import run as evaluate_run
        from .audit import run as audit_run

        evaluations: dict[str, object] = {}
        audits: dict[str, object] = {}
        for name in ("r0", r4_name):
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
        peak_reserved = int(torch.cuda.max_memory_reserved()) if device.type == "cuda" else None
        if logger is not None:
            # These are derived, receipt-gated target-plane views only.  Do not
            # pass source volumes, segmentations, or filesystem identifiers to
            # the W&B client.  The support uncertainty is the same uncalibrated
            # support diagnostic used by the physical-plane reconstruction
            # path, with the final immutable state uncertainty as its offset.
            state_observability = torch.cat((
                r4_result.patient_state.memory.structural.observability.uncertainty,
                r4_result.patient_state.memory.volumetric.observability.uncertainty,
            ))
            propagation_uncertainty = float(state_observability.mean().detach().cpu())
            r4_uncertainty = support_uncertainty(
                r4_result.prediction.support_mass,
                r4_result.prediction.unsupported_mask,
                propagation_uncertainty=propagation_uncertainty,
            )
            prediction = r4_result.prediction.intensity.detach()
            revealed_target = r4_target.detach()
            absolute_error = (prediction - revealed_target).abs()
            logger.log_images(
                {
                    "prediction/r4_target": prediction,
                    "target/evaluator_revealed": revealed_target,
                    "error/absolute": absolute_error,
                    "support/mask": (~r4_result.prediction.unsupported_mask).to(torch.float32),
                    "uncertainty/support_diagnostic": r4_uncertainty,
                },
                step=requested_steps + 1 if external_logger_step is None else external_logger_step,
            )
            logger.update_summary({
                "artifacts/r0_summary": str((r0_dir / "summary.json").relative_to(output_dir)),
                "artifacts/r0_checkpoint": str((r0_dir / "checkpoint.pt").relative_to(output_dir)),
                "artifacts/r0_checkpoint_sha256": r0_report["checkpoint_sha256"],
                "artifacts/r4_summary": str((r4_dir / "summary.json").relative_to(output_dir)),
                "artifacts/r4_checkpoint": str((r4_dir / "checkpoint.pt").relative_to(output_dir)),
                "artifacts/r4_checkpoint_sha256": r4_report["checkpoint_sha256"],
                "artifacts/r4_prediction_package": str((r4_dir / "predictions").relative_to(output_dir)),
                "artifacts/r4_evaluation": str((r4_dir / "evaluation" / "evaluation.json").relative_to(output_dir)),
                "artifacts/r4_audit": str((r4_dir / "audit_report.json").relative_to(output_dir)),
                "cohort_hash": config.get("cohort_hash"),
                "split_hash": split_hash,
                "assignment_hash": bundle.assignment.assignment_hash,
                "sampling_protocol_hash": sampling_protocol_hash,
                "context_plane_positions_mm": context_plane_positions_mm,
                "target_plane_position_mm": target_plane_position_mm,
                "support_fraction": r4_report["supported_fraction"],
                "unsupported_fraction": r4_report["unsupported_fraction"],
                "anchor_count": r4_report["anchor_count"],
                "structural_gaussian_count": r4_report["structural_gaussian_count"],
                "volumetric_gaussian_count": r4_report["volumetric_gaussian_count"],
                "propagation_child_count": r4_report["propagation_child_count"],
                "runtime_seconds": r4_report["runtime_seconds"],
                "learning_rate": r4_report["learning_rate"],
                "inference_wall_time_seconds": r4_report["inference_wall_time_seconds"],
                "peak_cuda_allocated_bytes": peak_memory,
                "peak_cuda_reserved_bytes": peak_reserved,
                "support_uncertainty_semantics": "uncalibrated_support_diagnostic",
                "experiment_name": r4_report.get("experiment_name"),
                "model_complexity": r4_report.get("model_complexity"),
            })
        context_cache_size = sum(path.stat().st_size for path in (prepared_dir / "payloads" / "context").glob("*") if path.is_file())
        summary = {
            "schema": "smagm-brats21-real-smoke-v1",
            "experiment_name": str(config.get("experiment_name", "brats21-static-diagnostic")),
            "claim_scope": config["claim_scope"],
            "source_kind": config["source_kind"],
            "patient_pseudonymous_id": pseudonymous_patient,
            "manifest_hash": bundle.manifest.manifest_hash,
            "split_hash": split_hash,
            "cohort_split_hash": cohort_split_hash,
            "assignment_hash": bundle.assignment.assignment_hash,
            "split_registry_hash": split_registry.registry_hash,
            "context_count": len(bundle.assignment.context_ids),
            "modality_inventory": list(config["modalities"]),
            "target_modality": str(config["target_modality"]),
            "target_orientation": bundle.manifest_json.get("source_geometry", {}).get("orientation"),
            "context_target_disjoint": not (set(bundle.assignment.context_ids) & set(bundle.assignment.target_ids)),
            "target_reveal_barrier_verified": True,
            "device": device_report,
            "seed": seed,
            "optimizer": optimizer_name,
            "training_updates_applied": not validation_only,
            "validation_only": validation_only,
            "global_model": {
                "binding_hash": global_model_binding_hash,
                "input_available": global_model_input is not None,
                "input_update_index": None if global_model_input is None else global_model_input["global_update_index"],
            },
            "r0": r0_report,
            r4_name: r4_report,
            "evaluations": evaluations,
            "audits": audits,
            "evaluator_target": target_report,
            "runtime_seconds": time.perf_counter() - run_start,
            "model_complexity": r4_report.get("model_complexity"),
            "peak_cuda_memory_bytes": peak_memory,
            "peak_cuda_allocated_bytes": peak_memory,
            "peak_cuda_reserved_bytes": peak_reserved,
            "resource_metrics": {
                "training_wall_time_seconds": time.perf_counter() - run_start,
                "inference_wall_time_seconds": r4_report.get("inference_wall_time_seconds"),
                "per_plane_latency_seconds": r4_report.get("per_plane_latency_seconds"),
                "full_grid_latency_seconds": r4_report.get("full_grid_latency_seconds"),
                "full_grid_status": r4_report.get("full_grid_status"),
                "peak_cuda_allocated_bytes": peak_memory,
                "peak_cuda_reserved_bytes": peak_reserved,
                "anchor_count": r4_report.get("anchor_count"),
                "structural_gaussian_count": r4_report.get("structural_gaussian_count"),
                "volumetric_gaussian_count": r4_report.get("volumetric_gaussian_count"),
                "propagation_child_count": r4_report.get("propagation_child_count"),
                "cache_size_bytes": context_cache_size,
                "checkpoint_size_bytes": r4_report.get("checkpoint_size_bytes"),
                "parameter_count": r4_report.get("parameter_count"),
                "trainable_parameter_count": r4_report.get("trainable_parameter_count"),
                "encoder_forward_flops_2flop_per_mac": r4_report.get("encoder_forward_flops_2flop_per_mac"),
                "profiled_supported_operator_flops": r4_report.get("profiled_supported_operator_flops"),
                "profiler_scope": r4_report.get("profiler_scope"),
                "profiler_operator_coverage": r4_report.get("profiler_operator_coverage"),
            },
            "repository": git,
            "environment_hash": environment_hash,
            "t4_routing": False,
            "scientific_pass_recorded": False,
            "human_gate_decision": None,
            "non_claims": config["non_claims"],
            "recovered_partial_outputs": recovered_partial_outputs,
        }
        # The assignment above intentionally avoids reading source volumes; use
        # a stable wall-clock value captured from the run directory metadata.
        summary["runtime_seconds"] = max(time.perf_counter() - run_start, 0.0)
        if logger is not None:
            evaluation_scalars: dict[str, float] = {}
            evaluation_fields = (
                "mae", "rmse", "psnr", "ssim", "ncc", "gradient_mae", "gradient_rmse",
                "frequency_error", "edge_f1", "local_contrast_error", "supported_fraction", "unsupported_fraction",
            )
            for name in ("r0", r4_name):
                raw_evaluation = evaluations.get(name)
                raw_metrics = raw_evaluation.get("metrics") if isinstance(raw_evaluation, dict) else None
                metric = raw_metrics[0] if isinstance(raw_metrics, list) and raw_metrics and isinstance(raw_metrics[0], dict) else {}
                for field in evaluation_fields:
                    value = metric.get(field)
                    if isinstance(value, (int, float)) and np.isfinite(float(value)):
                        evaluation_scalars[f"evaluation/{name}_{field}"] = float(value)
            evaluation_scalars["runtime/peak_cuda_memory_bytes"] = float(peak_memory or 0)
            evaluation_scalars["model/parameter_count"] = float(model_complexity_preview["parameters"])
            evaluation_scalars["model/trainable_parameter_count"] = float(model_complexity_preview["trainable_parameters"])
            measured_flops = r4_report.get("profiled_supported_operator_flops")
            if isinstance(measured_flops, (int, float)) and np.isfinite(float(measured_flops)):
                evaluation_scalars["compute/profiled_supported_operator_flops"] = float(measured_flops)
            encoder_flops = r4_report.get("encoder_forward_flops_2flop_per_mac")
            if isinstance(encoder_flops, (int, float)) and np.isfinite(float(encoder_flops)):
                evaluation_scalars["compute/encoder_forward_flops_2flop_per_mac"] = float(encoder_flops)
            logger.log(
                evaluation_scalars,
                step=requested_steps + 1 if external_logger_step is None else external_logger_step,
            )
            if owns_logger:
                wandb_finish = logger.finish(status="finished")
                summary["wandb"] = wandb_finish.to_dict()
            else:
                summary["wandb"] = {
                    "mode": getattr(logger, "mode", "external-process-level"),
                    "run_id": getattr(logger, "run_id", None),
                    "url": getattr(logger, "url", None),
                    "ownership": "external_process_level",
                }
        else:
            summary["wandb"] = {"mode": "unavailable", "run_id": None, "url": None}
        _atomic_json(summary, output_dir / "summary.json")
        return summary
    except Exception as error:
        failure = {"schema": "smagm-brats21-real-smoke-failure-v1", "failure_reason": f"{type(error).__name__}: {error}", "repository": git}
        _atomic_json(failure, output_dir / "failure.json")
        if logger is not None and owns_logger:
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
