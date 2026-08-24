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


def test_codegraph_authorizes_only_gate_c_trajectory_paths() -> None:
    allowed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--task",
            "trajectory",
            "--check",
            "src/smagm/features/point_guided/state_init.py",
            "src/smagm/features/point_guided/reward.py",
            "src/smagm/features/point_guided/trajectory.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert allowed.returncode == 0, allowed.stderr
    assert "Gate C C1-C7" in allowed.stdout

    inactive = [
        "src/smagm/features/point_guided/decoder.py",
        "src/smagm/features/point_guided/losses.py",
        "src/smagm/features/point_guided/reward_supervision.py",
        "src/smagm/features/point_guided/training.py",
        "src/smagm/features/point_guided/gate_f.py",
        "src/smagm/features/point_guided/gate_g.py",
    ]
    denied = subprocess.run(
        [sys.executable, str(SCRIPT), "--task", "trajectory", "--check", *inactive],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert denied.returncode == 2
    assert "denied read paths" in denied.stderr
    assert all(path in denied.stderr for path in inactive)


def test_codegraph_authorizes_only_gate_d_decoder_paths() -> None:
    allowed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--task",
            "decoder",
            "--check",
            "src/smagm/features/point_guided/decoder.py",
            "src/smagm/features/point_guided/model.py",
            "tests/features/point_guided/test_decoder.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert allowed.returncode == 0, allowed.stderr
    assert "Gate D D1" in allowed.stdout

    inactive = [
        "src/smagm/features/point_guided/losses.py",
        "src/smagm/features/point_guided/reward_supervision.py",
        "src/smagm/features/point_guided/training.py",
        "src/smagm/features/point_guided/gate_f.py",
        "src/smagm/features/point_guided/gate_g.py",
    ]
    denied = subprocess.run(
        [sys.executable, str(SCRIPT), "--task", "decoder", "--check", *inactive],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert denied.returncode == 2
    assert "denied read paths" in denied.stderr
    assert all(path in denied.stderr for path in inactive)


def test_codegraph_authorizes_only_gate_e_supervision_paths() -> None:
    allowed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--task",
            "supervision",
            "--check",
            "src/smagm/features/point_guided/losses.py",
            "src/smagm/features/point_guided/reward_supervision.py",
            "src/smagm/features/point_guided/training_objective.py",
            "tests/features/point_guided/test_losses.py",
            "tests/features/point_guided/test_reward_supervision.py",
            "tests/features/point_guided/test_training_objective.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert allowed.returncode == 0, allowed.stderr
    assert "Gate E E1-E9" in allowed.stdout

    inactive = [
        "src/smagm/training/trainer.py",
        "src/smagm/features/point_guided/optimizer.py",
        "src/smagm/features/point_guided/scheduler.py",
        "src/smagm/features/point_guided/training.py",
        "src/smagm/features/point_guided/gate_f.py",
        "src/smagm/features/point_guided/gate_g.py",
    ]
    denied = subprocess.run(
        [sys.executable, str(SCRIPT), "--task", "supervision", "--check", *inactive],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert denied.returncode == 2
    assert "denied read paths" in denied.stderr
    assert all(path in denied.stderr for path in inactive)


def test_codegraph_activates_only_gate_f_baseline_training_paths() -> None:
    allowed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--task",
            "baseline_training",
            "--check",
            "src/smagm/features/point_guided/baseline_training.py",
            "src/smagm/features/point_guided/baseline_metrics.py",
            "src/smagm/features/point_guided/baseline_checkpoint.py",
            "tests/features/point_guided/test_baseline_training.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert allowed.returncode == 0, allowed.stderr
    assert "Gate F F1/F2" in allowed.stdout

    inactive = [
        "src/smagm/features/point_guided/gate_g.py",
        "src/smagm/features/point_guided/baseline_inference.py",
        "src/smagm/features/point_guided/heldout_evaluation.py",
        "src/smagm/training/trainer.py",
        "src/smagm/data/brats21.py",
    ]
    denied = subprocess.run(
        [sys.executable, str(SCRIPT), "--task", "baseline_training", "--check", *inactive],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert denied.returncode == 2
    assert "denied read paths" in denied.stderr
    assert all(path in denied.stderr for path in inactive)


def test_codegraph_declared_entrypoints_and_read_paths_exist() -> None:
    validated = subprocess.run(
        [sys.executable, str(SCRIPT), "--validate-paths"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert validated.returncode == 0, validated.stderr
    assert "validated declared entrypoints and read paths" in validated.stdout


def test_codegraph_activates_only_gate_g_baseline_inference_paths() -> None:
    allowed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--task",
            "baseline_inference",
            "--check",
            "src/smagm/features/point_guided/baseline_inference.py",
            "src/smagm/features/point_guided/availability.py",
            "src/smagm/features/point_guided/model.py",
            "tests/features/point_guided/test_baseline_inference.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert allowed.returncode == 0, allowed.stderr
    assert "Gate G G1-G4" in allowed.stdout

    blocked = [
        "src/smagm/features/point_guided/baseline_training.py",
        "src/smagm/features/point_guided/heldout_evaluation.py",
        "tests/features/point_guided/test_heldout_evaluation.py",
        "src/smagm/evaluation/report.py",
    ]
    denied = subprocess.run(
        [sys.executable, str(SCRIPT), "--task", "baseline_inference", "--check", *blocked],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert denied.returncode == 2
    assert "denied read paths" in denied.stderr
    assert all(path in denied.stderr for path in blocked)


def test_codegraph_exposes_server_pipeline_without_unblocking_legacy_data() -> None:
    allowed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--task",
            "server_pipeline",
            "--check",
            "src/smagm/data/brats21_point_guided.py",
            "src/smagm/training/point_guided.py",
            "src/smagm/cli/point_guided_eval.py",
            "configs/training/point_guided_brats21_4070.json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert allowed.returncode == 0, allowed.stderr
    assert "server-ready" in allowed.stdout

    denied = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--task",
            "server_pipeline",
            "--check",
            "src/smagm/data/brats21.py",
            "src/smagm/anchors/anchor.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert denied.returncode == 2
    assert "denied read paths" in denied.stderr
