"""Verify immutable reconstruction package inventory and freeze metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..evaluation import open_serialized_predictions


def run(package_dir: Path) -> dict[str, object]:
    predictions = open_serialized_predictions(package_dir)
    return {"package_hash": predictions.package.package_hash, "patient_id": predictions.package.patient_id, "artifact_count": len(predictions.package.output_artifacts), "status": predictions.package.execution_status}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit immutable prediction package provenance")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(); report = run(args.package)
    print(json.dumps(report, sort_keys=True) if args.json else json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__": main()
