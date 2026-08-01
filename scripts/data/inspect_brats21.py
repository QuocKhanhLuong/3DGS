#!/usr/bin/env python3
"""Inspect and fail-closed validate a BraTS21 source root."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from smagm.data.brats21 import (  # noqa: E402
    BRATS21_MODALITIES,
    BRATS21_PATIENT_PATTERN,
    BraTS21ValidationError,
    discover_patient,
    validate_patient,
)


def inspect_root(root: Path, *, limit: int = 5, require_segmentation: bool = False) -> dict[str, object]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    source_root = root.resolve(strict=True)
    directories = tuple(sorted(path for path in source_root.iterdir() if path.is_dir()))
    malformed_names = [path.name for path in directories if BRATS21_PATIENT_PATTERN.fullmatch(path.name) is None]
    invalid: list[dict[str, object]] = []
    complete: list[tuple[object, object]] = []
    suffix_counts = {suffix: 0 for suffix in (*BRATS21_MODALITIES, "seg")}
    unknown_files: dict[str, list[str]] = {}
    for directory in directories:
        if BRATS21_PATIENT_PATTERN.fullmatch(directory.name) is None:
            invalid.append({"patient_id": directory.name, "error": "malformed patient directory name"})
            continue
        try:
            patient = discover_patient(directory)
            for suffix in (*BRATS21_MODALITIES, "seg"):
                if suffix in patient.modality_paths or (suffix == "seg" and patient.segmentation_path is not None):
                    suffix_counts[suffix] += 1
            if patient.unknown_nifti_files:
                unknown_files[patient.patient_id] = list(patient.unknown_nifti_files)
            result = validate_patient(
                patient,
                require_segmentation=require_segmentation,
                include_data=False,
                include_source_hash=False,
            )
            if not result.valid:
                invalid.append(result.to_dict())
            else:
                complete.append((patient, result))
        except (BraTS21ValidationError, OSError) as error:
            invalid.append({"patient_id": directory.name, "error": str(error)})

    samples: list[dict[str, object]] = []
    for patient, _ in complete[:limit]:
        result = validate_patient(
            patient,
            require_segmentation=require_segmentation,
            include_data=True,
            include_source_hash=True,
        )
        samples.append(result.to_dict())
        if not result.valid:
            invalid.append(result.to_dict())
    report: dict[str, object] = {
        "root_is_absolute": source_root.is_absolute(),
        "patient_directory_count": len(directories),
        "valid_patient_count": len(complete),
        "incomplete_or_malformed_count": len(invalid),
        "malformed_directory_names": malformed_names,
        "modality_suffixes": suffix_counts,
        "unknown_nifti_files": unknown_files,
        "sample_limit": limit,
        "samples": samples,
        "invalid_patients": invalid,
        "validation_scope": {
            "all_patients": "header, dimensionality, affine, and cross-modality geometry",
            "sample_patients": "full finite-data ranges and segmentation labels",
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and validate a BraTS21 NIfTI source root")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--require-segmentation", action="store_true")
    parser.add_argument("--allow-invalid", action="store_true", help="print the inventory without failing on malformed patients")
    args = parser.parse_args()
    report = inspect_root(args.root, limit=args.limit, require_segmentation=args.require_segmentation)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if args.allow_invalid or report["incomplete_or_malformed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
