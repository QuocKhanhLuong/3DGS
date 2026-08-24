"""Held-out target-after-inference evaluation for point-guided checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from ..features.point_guided.artifacts import (
    atomic_output_path,
    atomic_write_json,
    reserve_artifact_directory,
)
from ..data.brats21_point_guided import (
    BraTS21PointGuidedDataset,
    discover_point_guided_subjects,
    load_point_guided_split,
)
from ..features.point_guided.baseline_inference import (
    GateGInferenceConfig,
    load_validated_baseline_checkpoint,
)
from ..features.point_guided.baseline_metrics import compute_reconstruction_metrics, semantic_dice
from ..features.point_guided.provenance import best_effort_git_head
from ..features.point_guided.semantic_supervision import build_coarse_semantic_target
from ..features.point_guided.training_objective import SupervisionConfig
from ..training.point_guided import (
    build_model_from_config,
    normalization_space_from_config,
    validate_metric_data_range,
)


def _atomic_json(path: Path, payload: object) -> None:
    atomic_write_json(path, payload)


def resolve_split_file(checkpoint: Path, split_file: Path | None = None) -> Path:
    """Resolve the exact split artifact belonging to a training checkpoint."""

    checkpoint = checkpoint.resolve()
    if split_file is None:
        candidate = checkpoint.parent.parent / "split.json"
    else:
        candidate = split_file.resolve()
    if not candidate.is_file():
        source = "inferred training run" if split_file is None else "explicit argument"
        raise FileNotFoundError(
            f"exact training split.json is required ({source}) but was not found: {candidate}"
        )
    return candidate


def _load_split(
    path: Path,
    subjects: tuple[str, ...],
) -> tuple[dict[str, tuple[str, ...]], str]:
    split_object = load_point_guided_split(path, subjects=subjects)
    groups = {
        "train": tuple(split_object.train_subject_ids),
        "val": tuple(split_object.val_subject_ids),
        "test": tuple(split_object.test_subject_ids),
    }
    return groups, split_object.split_hash


def _save_nifti(prediction: torch.Tensor, path: Path, affine: tuple[tuple[float, ...], ...]) -> None:
    try:
        import nibabel as nib  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("saving NIfTI predictions requires the optional nibabel dependency") from error
    # Model tensor order is [D,H,W]; NIfTI source order is [X,Y,Z].
    array_xyz = prediction.detach().cpu().numpy().transpose(2, 1, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".nii.gz" if path.name.endswith(".nii.gz") else (path.suffix or ".nii")
    with atomic_output_path(path, suffix=suffix) as temporary:
        nib.save(nib.Nifti1Image(array_xyz, affine), str(temporary))


def _evaluate_reserved(
    *,
    checkpoint: Path,
    config_path: Path,
    data_root: Path,
    output_dir: Path,
    split_file: Path | None,
    split_name: str,
    device_name: str,
    save_predictions: bool,
    max_subjects: int | None = None,
    medicalnet_checkpoint: Path | None = None,
    medicalnet_sha256: str | None = None,
    require_pretrained_backbone: bool | None = None,
) -> dict[str, Any]:
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError("evaluation config must contain a JSON object")
    data_config = raw_config.get("data", {})
    normalization_config = data_config.get("normalization")
    supervision = SupervisionConfig(**raw_config.get("supervision", {}))
    validate_metric_data_range(normalization_config, supervision)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    overrides = {
        "device": device_name,
        "medicalnet_checkpoint_path": None if medicalnet_checkpoint is None else str(medicalnet_checkpoint),
        "medicalnet_checkpoint_sha256": medicalnet_sha256,
        "require_pretrained_backbone": require_pretrained_backbone,
    }
    subjects = discover_point_guided_subjects(data_root)
    subject_ids = tuple(subject.subject_id for subject in subjects)
    resolved_split_file = resolve_split_file(checkpoint, split_file)
    groups, split_hash = _load_split(resolved_split_file, subject_ids)
    model, _, _ = build_model_from_config(raw_config, overrides)
    load_validated_baseline_checkpoint(model, checkpoint, expected_split_hash=split_hash)
    model.to(device).eval()
    selected_ids = list(groups[split_name])
    if max_subjects is not None:
        if max_subjects <= 0:
            raise ValueError("max_subjects must be positive when supplied")
        selected_ids = selected_ids[:max_subjects]
    dataset = BraTS21PointGuidedDataset(
        data_root,
        selected_ids,
        normalization_config=normalization_config,
        require_segmentation=False,
    )
    inference_values = dict(raw_config.get("inference", {}))
    inference_config = GateGInferenceConfig(**inference_values)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_subject: list[dict[str, Any]] = []
    stop_histogram: Counter[str] = Counter()
    with torch.no_grad():
        for sample in dataset:
            subject_id = sample.subject_id
            if sample.target_t1ce is None:
                raise ValueError(f"{subject_id}: evaluation requires a T1ce target")
            observations = sample.observations.unsqueeze(0).to(device)
            brain_mask = sample.brain_mask.unsqueeze(0).to(device)
            # This is the only model inference call and has no target argument.
            result = model.forward_baseline_inference(
                observations,
                brain_mask=brain_mask,
                spacing_mm=sample.spacing_xyz_mm,
                voxel_to_ras_mm=sample.voxel_to_ras_mm,
                inference_config=inference_config,
            )
            target = sample.target_t1ce.unsqueeze(0).to(device)
            metrics = compute_reconstruction_metrics(
                result.prediction,
                target,
                brain_mask,
                data_range=supervision.ssim_data_range,
                intensity_space=normalization_space_from_config(normalization_config),
            )
            semantic = None
            if sample.segmentation_labels is not None and result.semantic_probabilities is not None:
                semantic_target = build_coarse_semantic_target(
                    sample.segmentation_labels.unsqueeze(0).to(device),
                    brain_mask,
                    ignore_index=255,
                )
                semantic = semantic_dice(result.semantic_probabilities, semantic_target, ignore_index=255).as_dict()
            stop_histogram.update(result.stop_reasons)
            record: dict[str, Any] = {
                "subject_id": subject_id,
                "split": split_name,
                "metrics": {
                    "MAE": metrics.mae,
                    "PSNR": metrics.psnr,
                    "SSIM": metrics.ssim,
                    "intensity_space": metrics.intensity_space,
                },
                "semantic": semantic,
                "trajectory": {
                    "K_used": int(result.k_used[0].cpu()),
                    "path_length_mm": float(result.path_length_mm[0].cpu()),
                    "mean_predicted_reward": float(result.reward_mean[0].cpu()),
                    "max_predicted_reward": float(result.reward_max[0].cpu()),
                    "mean_utility": float(result.utility_mean[0].cpu()),
                    "max_utility": float(result.utility_max[0].cpu()),
                    "mean_update_magnitude": float(result.update_magnitude_mean[0].cpu()),
                    "max_update_magnitude": float(result.update_magnitude_max[0].cpu()),
                    "stop_reason": result.stop_reasons[0],
                    "candidate_evaluations": int(result.candidate_evaluations[0].cpu()),
                    "eligible_candidate_evaluations": int(result.eligible_candidate_evaluations[0].cpu()),
                },
            }
            per_subject.append(record)
            if save_predictions:
                _save_nifti(
                    result.prediction[0, 0],
                    output_dir / "predictions" / f"{subject_id}_t1ce_pred.nii.gz",
                    sample.voxel_to_ras_mm,
                )
    def _mean_metric(name: str) -> float:
        values = [float(item["metrics"][name]) for item in per_subject]
        return sum(values) / len(values) if values else float("nan")

    aggregate = {
        "split": split_name,
        "split_hash": split_hash,
        "subject_count": len(per_subject),
        "metrics": {name: _mean_metric(name) for name in ("MAE", "PSNR", "SSIM")},
        "semantic": {
            name: sum(float(item["semantic"][name]) for item in per_subject if item["semantic"] is not None)
            / max(sum(item["semantic"] is not None for item in per_subject), 1)
            for name in ("dice_normal", "dice_edema", "dice_core")
        },
        "stop_reason_histogram": dict(stop_histogram),
        "normalization_space": normalization_space_from_config(normalization_config),
        "clinical_quality_claim": False,
    }
    _atomic_json(output_dir / "per_subject_metrics.json", per_subject)
    _atomic_json(output_dir / "aggregate_metrics.json", aggregate)
    _atomic_json(output_dir / "trajectory_diagnostics.json", [item["trajectory"] for item in per_subject])
    _atomic_json(output_dir / "evaluation_metadata.json", {
        "checkpoint": str(checkpoint.resolve()),
        "git_head": best_effort_git_head(),
        "split_file": str(resolved_split_file),
        "split_hash": split_hash,
        "split": split_name,
        "training_run_dir": str(checkpoint.resolve().parent.parent),
        "normalization_space": normalization_space_from_config(normalization_config),
        "target_used_after_inference_only": True,
        "segmentation_used_after_inference_only": True,
    })
    return aggregate


def evaluate(
    *,
    checkpoint: Path,
    config_path: Path,
    data_root: Path,
    output_dir: Path,
    split_file: Path | None,
    split_name: str,
    device_name: str,
    save_predictions: bool,
    max_subjects: int | None = None,
    medicalnet_checkpoint: Path | None = None,
    medicalnet_sha256: str | None = None,
    require_pretrained_backbone: bool | None = None,
    reuse_output: bool = False,
) -> dict[str, Any]:
    """Reserve an evaluation destination before producing any artifacts.

    Existing output directories are rejected by default.  ``reuse_output`` is
    an explicit operator choice and never clears stale files; the reservation
    lock prevents another evaluator from writing the same directory at once.
    """

    reservation = reserve_artifact_directory(
        output_dir,
        reuse=reuse_output,
        purpose="evaluation output",
    )
    try:
        return _evaluate_reserved(
            checkpoint=checkpoint,
            config_path=config_path,
            data_root=data_root,
            output_dir=reservation.path,
            split_file=split_file,
            split_name=split_name,
            device_name=device_name,
            save_predictions=save_predictions,
            max_subjects=max_subjects,
            medicalnet_checkpoint=medicalnet_checkpoint,
            medicalnet_sha256=medicalnet_sha256,
            require_pretrained_backbone=require_pretrained_backbone,
        )
    finally:
        reservation.release()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a strict point-guided checkpoint on a held-out BraTS split")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--reuse-output",
        action="store_true",
        help="explicitly reuse an existing evaluation directory (never clears stale files)",
    )
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--medicalnet-checkpoint", type=Path)
    parser.add_argument("--medicalnet-sha256")
    parser.add_argument("--require-pretrained-backbone", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    result = evaluate(
        checkpoint=args.checkpoint,
        config_path=args.config,
        data_root=args.data_root,
        output_dir=args.output_dir,
        split_file=args.split_file,
        split_name=args.split,
        device_name=args.device,
        save_predictions=args.save_predictions,
        reuse_output=args.reuse_output,
        max_subjects=args.max_subjects,
        medicalnet_checkpoint=args.medicalnet_checkpoint,
        medicalnet_sha256=args.medicalnet_sha256,
        require_pretrained_backbone=args.require_pretrained_backbone,
    )
    print(json.dumps(result, sort_keys=True, indent=2, default=str))


if __name__ == "__main__":
    main()
