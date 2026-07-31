"""Thin source-checkout wrapper for :mod:`smagm.cli.train`."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smagm.cli.train import main  # noqa: E402


if __name__ == "__main__":
    main()
