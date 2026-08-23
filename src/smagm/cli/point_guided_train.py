"""CLI for the dedicated full-volume point-guided BraTS21 trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..training.point_guided import preflight, run_training


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"point-guided config does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"point-guided config is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("point-guided config root must be a JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the locked point-guided T1/T2/FLAIR -> T1ce model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/point-guided"))
    parser.add_argument("--run-name")
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--max-train-subjects", type=int)
    parser.add_argument("--max-val-subjects", type=int)
    parser.add_argument("--max-test-subjects", type=int)
    parser.add_argument("--device")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--gradient-clip", type=float)
    parser.add_argument("--decoder-chunk-size", type=int)
    parser.add_argument("--counterfactual-candidates", type=int)
    parser.add_argument("--k-max", type=int)
    parser.add_argument("--lambda-semantic", type=float)
    parser.add_argument("--lambda-travel", type=float)
    parser.add_argument("--lambda-overlap", type=float)
    parser.add_argument("--lambda-step", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--medicalnet-checkpoint", type=Path)
    parser.add_argument("--medicalnet-sha256")
    parser.add_argument("--require-pretrained-backbone", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="point-guided-brats21")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--preflight", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    raw_config = _load_config(args.config)
    if args.preflight:
        data_config = raw_config.get("data", {})
        if not isinstance(data_config, dict):
            raise ValueError("point-guided data config must be an object")
        result = preflight(
            data_root=args.data_root,
            checkpoint=args.medicalnet_checkpoint,
            expected_sha256=args.medicalnet_sha256,
            require_segmentation=bool(data_config.get("require_segmentation", True)),
            split_seed=int(data_config.get("split_seed", 20260813)),
            split_fractions=tuple(data_config.get("split_fractions", (0.8, 0.1, 0.1))),
            split_file=args.split_file,
            max_train_subjects=args.max_train_subjects,
            max_val_subjects=args.max_val_subjects,
            max_test_subjects=args.max_test_subjects,
            overfit=bool(args.overfit),
        )
        result["config"] = str(args.config.resolve())
        print(json.dumps(result, sort_keys=True, indent=2))
        return
    overrides: dict[str, Any] = {
        key: value
        for key, value in {
            "device": args.device,
            "amp": args.amp,
            "amp_dtype": args.amp_dtype,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "gradient_clip": args.gradient_clip,
            "decoder_chunk_size": args.decoder_chunk_size,
            "counterfactual_candidates": args.counterfactual_candidates,
            "k_max": args.k_max,
            "lambda_semantic": args.lambda_semantic,
            "lambda_travel": args.lambda_travel,
            "lambda_overlap": args.lambda_overlap,
            "lambda_step": args.lambda_step,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "medicalnet_checkpoint_path": None if args.medicalnet_checkpoint is None else str(args.medicalnet_checkpoint),
            "medicalnet_checkpoint_sha256": args.medicalnet_sha256,
            "require_pretrained_backbone": args.require_pretrained_backbone,
        }.items()
        if value is not None
    }
    raw_config["_wandb"] = {
        "enabled": bool(args.wandb),
        "project": args.wandb_project,
        "run_name": args.wandb_run_name or args.run_name,
    }
    summary = run_training(
        raw_config=raw_config,
        data_root=args.data_root,
        output_root=args.output_root,
        run_name=args.run_name,
        split_file=args.split_file,
        resume=args.resume,
        overfit=args.overfit,
        max_train_subjects=args.max_train_subjects,
        max_val_subjects=args.max_val_subjects,
        max_test_subjects=args.max_test_subjects,
        overrides=overrides,
    )
    if summary is not None:
        print(json.dumps(summary, sort_keys=True, indent=2, default=str))


if __name__ == "__main__":
    main()
