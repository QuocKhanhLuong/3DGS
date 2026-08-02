#!/usr/bin/env python3
"""Inspect and fail-closed validate a BraTS21 source root."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import json
import re
import sys
import tempfile
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from smagm.data.brats21 import (  # noqa: E402
    BRATS21_MODALITIES,
    BRATS21_PATIENT_PATTERN,
    BraTS21ValidationError,
    discover_patient,
    validate_patient,
)


def _pseudonym(patient_id: str) -> str:
    return hashlib.sha256(f"smagm-brats21-patient-v1:{patient_id}".encode()).hexdigest()


def _atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _markdown(report: dict[str, object]) -> str:
    return "\n".join(
        (
            "# BraTS21 dataset inventory",
            "",
            "This inventory is metadata and validation evidence. It does not copy the source NIfTI volumes.",
            "",
            f"- Source root: `{report['source_root']}`",
            f"- Discovered patient directories: {report['patient_directory_count']}",
            f"- Valid patients: {report['valid_patient_count']}",
            f"- Rejected patients: {report['incomplete_or_malformed_count']}",
            f"- Estimated source bytes: {report['estimated_disk_usage_bytes']}",
            f"- Cohort hash: `{report['cohort_hash']}`",
            "",
            "## Modalities",
            "",
            "```json",
            json.dumps(report["modality_suffixes"], sort_keys=True, indent=2),
            "```",
            "",
            "## Geometry distributions",
            "",
            f"- Shapes: `{json.dumps(report['shape_distribution'], sort_keys=True)}`",
            f"- Spacing: `{json.dumps(report['spacing_distribution'], sort_keys=True)}`",
            f"- Orientation: `{json.dumps(report['orientation_distribution'], sort_keys=True)}`",
            f"- Segmentation labels: `{json.dumps(report['segmentation_label_inventory'], sort_keys=True)}`",
            "",
            "Rejected patients and reasons are retained in the JSON report. Patient identifiers are not sent to W&B; runs use the pseudonyms in the prepared manifest.",
            "",
        )
    )


def inspect_root(root: Path, *, limit: int = 5, require_segmentation: bool = False, include_data: bool = True) -> dict[str, object]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    source_root = root.resolve(strict=True)
    directories = tuple(sorted(path for path in source_root.iterdir() if path.is_dir()))
    malformed_names = [path.name for path in directories if BRATS21_PATIENT_PATTERN.fullmatch(path.name) is None]
    invalid: list[dict[str, object]] = []
    complete: list[tuple[object, object]] = []
    suffix_counts = {suffix: 0 for suffix in (*BRATS21_MODALITIES, "seg")}
    unknown_files: dict[str, list[str]] = {}
    valid_records: list[dict[str, object]] = []
    shape_counts: Counter[str] = Counter()
    spacing_counts: Counter[str] = Counter()
    orientation_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    estimated_disk_usage = 0
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
            estimated_disk_usage += sum(path.stat().st_size for path in patient.modality_paths.values())
            if patient.segmentation_path is not None:
                estimated_disk_usage += patient.segmentation_path.stat().st_size
            result = validate_patient(
                patient,
                require_segmentation=require_segmentation,
                include_data=include_data,
                include_source_hash=True,
            )
            if not result.valid:
                invalid.append(result.to_dict())
            else:
                complete.append((patient, result))
                shape_counts[json.dumps(result.summaries[0].shape_xyz)] += 1
                spacing_counts[json.dumps(tuple(round(value, 4) for value in result.summaries[0].spacing_xyz_mm))] += 1
                orientation_counts["".join(result.summaries[0].orientation)] += 1
                for summary in result.summaries:
                    for label in summary.segmentation_labels or ():
                        label_counts[str(label)] += 1
                valid_records.append(
                    {
                        "patient_id": patient.patient_id,
                        "patient_pseudonym": _pseudonym(patient.patient_id),
                        "has_segmentation": patient.segmentation_path is not None,
                        "unknown_nifti_files": list(patient.unknown_nifti_files),
                        "summaries": [item.to_dict() for item in result.summaries],
                    }
                )
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
    cohort_payload = [
        {
            "patient_pseudonym": item["patient_pseudonym"],
            "summaries": [
                {
                    "source_hash": summary["source_hash"],
                    "suffix": summary["suffix"],
                    "shape_xyz": summary["shape_xyz"],
                    "affine": summary["affine"],
                }
                for summary in item["summaries"]
            ],
        }
        for item in valid_records
    ]
    cohort_hash = hashlib.sha256(json.dumps(cohort_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report: dict[str, object] = {
        "schema": "smagm-brats21-dataset-inventory-v2",
        "source_root": str(source_root),
        "root_is_absolute": source_root.is_absolute(),
        "patient_directory_count": len(directories),
        "valid_patient_count": len(complete),
        "incomplete_or_malformed_count": len(invalid),
        "malformed_directory_names": malformed_names,
        "modality_suffixes": suffix_counts,
        "unknown_nifti_files": unknown_files,
        "estimated_disk_usage_bytes": estimated_disk_usage,
        "cohort_hash": cohort_hash,
        "shape_distribution": dict(sorted(shape_counts.items())),
        "spacing_distribution": dict(sorted(spacing_counts.items())),
        "orientation_distribution": dict(sorted(orientation_counts.items())),
        "segmentation_label_inventory": dict(sorted(label_counts.items())),
        "valid_patients": valid_records,
        "sample_limit": limit,
        "samples": samples,
        "invalid_patients": invalid,
        "validation_scope": {
            "all_patients": "header, dimensionality, affine, cross-modality geometry, full finite-data validation, source hashes, and segmentation-label validation",
            "sample_patients": "repeated full finite-data ranges and segmentation labels for the bounded sample",
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and validate a BraTS21 NIfTI source root")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--require-segmentation", action="store_true")
    parser.add_argument("--output", type=Path, help="write JSON and a sibling Markdown inventory")
    parser.add_argument("--headers-only", action="store_true", help="skip full finite-value validation; not the product default")
    parser.add_argument("--quiet", action="store_true", help="write the requested output without dumping all patient records")
    parser.add_argument("--allow-invalid", action="store_true", help="print the inventory without failing on malformed patients")
    args = parser.parse_args()
    report = inspect_root(
        args.root,
        limit=args.limit,
        require_segmentation=args.require_segmentation,
        include_data=not args.headers_only,
    )
    encoded = json.dumps(report, sort_keys=True, indent=2)
    if args.quiet:
        print(json.dumps({key: report[key] for key in ("valid_patient_count", "incomplete_or_malformed_count", "cohort_hash")}, sort_keys=True))
    else:
        print(encoded)
    if args.output is not None:
        _atomic_json(report, args.output)
        markdown_path = args.output.with_suffix(".md")
        markdown_path.write_text(_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
