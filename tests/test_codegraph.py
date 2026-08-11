from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "codegraph.py"


def test_codegraph_lists_and_resolves_the_frontend_scope() -> None:
    listed = subprocess.run(
        [sys.executable, str(SCRIPT), "--list"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert listed.returncode == 0
    assert "frontend" in listed.stdout
    resolved = subprocess.run(
        [sys.executable, str(SCRIPT), "--task", "frontend", "--check", "src/smagm/features/point_guided/model.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert resolved.returncode == 0, resolved.stderr
    assert "entrypoints:" in resolved.stdout


def test_codegraph_denies_a_legacy_training_path_for_frontend_work() -> None:
    forbidden = [
        "src/smagm/anchors/anchor.py",
        "src/smagm/fields/blend.py",
        "src/smagm/memory/state.py",
        "src/smagm/routing/controller.py",
        "src/smagm/training/trainer.py",
        "src/smagm/evaluation/report.py",
        "src/smagm/reconstruction/decoder.py",
        "src/smagm/cli/train.py",
        "src/smagm/data/brats21.py",
    ]
    denied = subprocess.run(
        [sys.executable, str(SCRIPT), "--task", "frontend", "--check", *forbidden],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert denied.returncode == 2
    assert "denied read paths" in denied.stderr
    assert all(path in denied.stderr for path in forbidden)
