#!/usr/bin/env python3
"""Create a metadata-only BraTS21 manifest for streamed product runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from smagm.data.brats21 import (  # noqa: E402
    BRATS21_MODALITIES,
    BRATS21_PATIENT_PATTERN,
    BraTS21ValidationError,
    discover_patient,
    validate_patient,
)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _pseudonym(patient_id: str) -> str:
    return hashlib.sha256(f"smagm-brats21-patient-v1:{patient_id}".encode()).hexdigest()


def _split_for(pseudonym: str, fractions: dict[str, float]) -> str:
    value = int.from_bytes(hashlib.sha256(pseudonym.encode()).digest()[:8], "big") / float(2**64)
    cumulative = 0.0
    for name in ("train", "validation", "t1_lesion_validation", "t5_final_audit"):
        cumulative += fractions.get(name, 0.0)
        if value < cumulative:
            return name
    return "t5_final_audit"


def prepare(root: Path, output_dir: Path, config: dict[str, Any], *, headers_only: bool = False) -> dict[str, object]:
    source_root = root.resolve(strict=True)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"prepared output is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    fractions = {str(key): float(value) for key, value in dict(config.get("split_fractions", {})).items()}
    if not fractions or abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError("split_fractions must be a non-empty mapping summing to one")
    patients: list[dict[str, object]] = []
    rejected: list[dict[str, str]] = []
    directories = tuple(sorted(path for path in source_root.iterdir() if path.is_dir()))
    for directory in directories:
        if BRATS21_PATIENT_PATTERN.fullmatch(directory.name) is None:
            rejected.append({"patient_id": directory.name, "reason": "malformed patient directory name"})
            continue
        try:
            patient = discover_patient(directory)
            result = validate_patient(
                patient,
                require_segmentation=False,
                include_data=not headers_only,
                include_source_hash=True,
            )
            if not result.valid:
                rejected.append({"patient_id": patient.patient_id, "reason": str(result.error)})
                continue
            pseudonym = _pseudonym(patient.patient_id)
            split = _split_for(pseudonym, fractions)
            patients.append(
                {
                    "patient_pseudonym": pseudonym,
                    "split": split,
                    "source_relative_directory": patient.directory.relative_to(source_root).as_posix(),
                    "has_segmentation": patient.segmentation_path is not None,
                    "modalities": {
                        item.suffix: {
                            "source_relative_path": patient.modality_paths[item.suffix].relative_to(source_root).as_posix(),
                            "source_sha256": item.source_hash,
                            "shape_xyz": item.shape_xyz,
                            "spacing_xyz_mm": item.spacing_xyz_mm,
                            "affine": item.affine,
                            "orientation": item.orientation,
                            "finite_validated": not headers_only,
                        }
                        for item in result.summaries
                        if item.suffix in BRATS21_MODALITIES
                    },
                    "segmentation": None
                    if patient.segmentation_path is None
                    else {
                        "source_relative_path": patient.segmentation_path.relative_to(source_root).as_posix(),
                        "source_sha256": next(item.source_hash for item in result.summaries if item.suffix == "seg"),
                        "shape_xyz": next(item.shape_xyz for item in result.summaries if item.suffix == "seg"),
                        "spacing_xyz_mm": next(item.spacing_xyz_mm for item in result.summaries if item.suffix == "seg"),
                        "affine": next(item.affine for item in result.summaries if item.suffix == "seg"),
                        "orientation": next(item.orientation for item in result.summaries if item.suffix == "seg"),
                        "evaluator_only": True,
                    },
                }
            )
        except (BraTS21ValidationError, OSError, ValueError) as error:
            rejected.append({"patient_id": directory.name, "reason": str(error)})
    patients.sort(key=lambda item: str(item["patient_pseudonym"]))
    cohort_payload = []
    for item in sorted(patients, key=lambda value: str(value["source_relative_directory"])):
        summaries = []
        for modality, metadata in item["modalities"].items():
            summaries.append({
                "source_hash": metadata["source_sha256"], "suffix": modality,
                "shape_xyz": metadata["shape_xyz"], "affine": metadata["affine"],
            })
        segmentation = item.get("segmentation")
        if isinstance(segmentation, dict):
            summaries.append({
                "source_hash": segmentation["source_sha256"], "suffix": "seg",
                "shape_xyz": segmentation["shape_xyz"], "affine": segmentation["affine"],
            })
        cohort_payload.append({"patient_pseudonym": item["patient_pseudonym"], "summaries": summaries})
    cohort_hash = hashlib.sha256(json.dumps(cohort_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    split_counts: dict[str, int] = {}
    for item in patients:
        split = str(item["split"])
        split_counts[split] = split_counts.get(split, 0) + 1
    prepared = {
        "schema": "smagm-brats21-prepared-manifest-v1",
        "source_root": str(source_root),
        "source_root_not_copied": True,
        "config": config,
        "cohort_hash": cohort_hash,
        "patient_count": len(patients),
        "rejected_count": len(rejected),
        "split_counts": split_counts,
        "patients": patients,
        "rejected": rejected,
        "target_references_are_evaluator_only": True,
    }
    _atomic_json(output_dir / "prepared.json", prepared)
    _atomic_json(output_dir / "manifest.json", {
        "schema": "smagm-brats21-stream-manifest-v1",
        "cohort_hash": cohort_hash,
        "patients": [
            {"patient_pseudonym": item["patient_pseudonym"], "split": item["split"]}
            for item in patients
        ],
    })
    (output_dir / "README.md").write_text(
        "# Prepared BraTS21 manifest\n\n"
        "This directory contains metadata, hashes, geometry, and split assignments only. "
        "Dense NIfTI files remain under the configured source root; segmentation and target "
        "references are evaluator-only.\n",
        encoding="utf-8",
    )
    return prepared


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a streamed, metadata-only BraTS21 manifest")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--headers-only", action="store_true", help="skip full finite-value scans")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != "smagm-brats21-data-v2":
        raise ValueError("BraTS21 data config must use schema smagm-brats21-data-v2")
    root = Path(str(config["dataset_root"]))
    report = prepare(root, args.output_dir, config, headers_only=args.headers_only)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
