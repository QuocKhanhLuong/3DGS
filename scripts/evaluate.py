"""Source-checkout wrapper for isolated evaluation."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from smagm.cli.evaluate import main  # noqa: E402
if __name__ == "__main__": main()
