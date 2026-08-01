"""Source-checkout wrapper for the T3 smoke CLI."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from smagm.cli.t3 import main  # noqa: E402
if __name__ == "__main__": main()
