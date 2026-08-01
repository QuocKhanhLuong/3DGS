"""The public T0 package must import in a fresh Python subprocess."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CleanImportTests(unittest.TestCase):
    def test_public_package_imports_in_clean_subprocess(self) -> None:
        source = Path(__file__).resolve().parents[2] / "src"
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(source)
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, "-c", "import smagm; print(smagm.__name__)"],
                cwd=directory,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "smagm")

