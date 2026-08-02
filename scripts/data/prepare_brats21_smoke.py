#!/usr/bin/env python3
"""Prepare one legal, bounded BraTS21 sparse episode without copying volumes."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from smagm.data.brats21_prepare import prepare_brats21_smoke  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a one-patient BraTS21 sparse smoke bundle")
    parser.add_argument("--root", type=Path, required=True, help="original dense BraTS21 root; never modified")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patient-id")
    parser.add_argument("--stride-v", type=int, default=4)
    parser.add_argument("--stride-u", type=int, default=4)
    args = parser.parse_args()
    report = prepare_brats21_smoke(
        source_root=args.root,
        output_dir=args.output_dir,
        patient_id=args.patient_id,
        inplane_stride_vu=(args.stride_v, args.stride_u),
    )
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
