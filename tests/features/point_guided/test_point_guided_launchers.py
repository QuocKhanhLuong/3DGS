"""MAIN-003 shell contracts for the supported point-guided launchers."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LAUNCHERS = (
    (REPOSITORY_ROOT / "scripts/point_guided_preflight.sh", ()),
    (REPOSITORY_ROOT / "scripts/point_guided_train_4070.sh", ()),
    (REPOSITORY_ROOT / "scripts/point_guided_train_2xa4000.sh", ()),
    (REPOSITORY_ROOT / "scripts/point_guided_overfit_4070.sh", ()),
)


_POINT_GUIDED_PYTHON_SPY = r"""#!/usr/bin/env bash
set -euo pipefail

: "${POINT_GUIDED_LAUNCH_LOG:?POINT_GUIDED_LAUNCH_LOG is required}"
: "${POINT_GUIDED_REAL_PYTHON:?POINT_GUIDED_REAL_PYTHON is required}"
{
  printf 'interpreter=%s\n' "$POINT_GUIDED_REAL_PYTHON"
  for argument in "$@"; do
    printf 'arg=%s\n' "$argument"
  done
} >> "$POINT_GUIDED_LAUNCH_LOG"

if [[ "${1:-}" == "-" ]]; then
  "$POINT_GUIDED_REAL_PYTHON" "$@" | tee -a "$POINT_GUIDED_LAUNCH_LOG"
  exit "${PIPESTATUS[0]}"
fi

for argument in "$@"; do
  case "$argument" in
    smagm.cli.point_guided_train|smagm.cli.point_guided_eval|torch.distributed.run)
      exit 0
      ;;
  esac
done

echo "unexpected point-guided interpreter invocation: $*" >&2
exit 97
"""


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _launcher_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path, Path]:
    fake_path = tmp_path / "conflicting-bin"
    fake_path.mkdir()
    conflict_log = tmp_path / "ambient-command.log"
    _write_executable(
        fake_path / "python",
        "#!/usr/bin/env bash\n"
        "printf 'ambient python was selected\\n' >> \"${POINT_GUIDED_CONFLICT_LOG:?}\"\n"
        "exit 96\n",
    )
    _write_executable(
        fake_path / "python3",
        "#!/usr/bin/env bash\n"
        "printf 'ambient python3 was selected\\n' >> \"${POINT_GUIDED_CONFLICT_LOG:?}\"\n"
        "exit 96\n",
    )
    _write_executable(
        fake_path / "torchrun",
        "#!/usr/bin/env bash\n"
        "printf 'ambient torchrun was selected\\n' >> \"${POINT_GUIDED_CONFLICT_LOG:?}\"\n"
        "exit 96\n",
    )

    point_guided_python = tmp_path / "point-guided-python"
    _write_executable(point_guided_python, _POINT_GUIDED_PYTHON_SPY)

    data_root = tmp_path / "brats21"
    data_root.mkdir()
    medicalnet_checkpoint = tmp_path / "medicalnet.pth"
    medicalnet_checkpoint.touch()
    output_root = tmp_path / "runs"
    output_root.mkdir()

    environment = os.environ.copy()
    environment.update(
        {
            "BRATS21_ROOT": str(data_root),
            "MEDICALNET_CKPT": str(medicalnet_checkpoint),
            "MEDICALNET_SHA256": "0" * 64,
            "OUTPUT_ROOT": str(output_root),
            "POINT_GUIDED_PYTHON": str(point_guided_python),
            "POINT_GUIDED_LAUNCH_LOG": str(tmp_path / "launcher.log"),
            "POINT_GUIDED_REAL_PYTHON": sys.executable,
            "POINT_GUIDED_CONFLICT_LOG": str(conflict_log),
            # Bash 3.2 treats an empty array expansion under `set -u` as an
            # error; keep this contract test portable across server shells.
            "POINT_GUIDED_EPOCHS": "1",
            "PATH": str(fake_path) + os.pathsep + environment.get("PATH", ""),
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )
    for variable in (
        "POINT_GUIDED_DISABLE_AMP",
        "POINT_GUIDED_RUN_NAME",
        "POINT_GUIDED_WANDB",
        "POINT_GUIDED_WANDB_RUN_NAME",
        "WANDB_PROJECT",
    ):
        environment.pop(variable, None)
    return environment, conflict_log, point_guided_python, output_root


def test_supported_launchers_share_explicit_python_and_torch_environment(tmp_path: Path) -> None:
    environment, conflict_log, point_guided_python, output_root = _launcher_environment(tmp_path)
    checkpoint = output_root / "run-01" / "checkpoints" / "best_model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (checkpoint.parent.parent / "split.json").write_text("{}", encoding="utf-8")

    launchers = LAUNCHERS + (
        (REPOSITORY_ROOT / "scripts/point_guided_eval.sh", (str(checkpoint),)),
    )
    observed_torch_versions: list[str] = []
    for launcher, arguments in launchers:
        result = subprocess.run(
            ["bash", str(launcher), *arguments],
            cwd=REPOSITORY_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"{launcher.name} failed with stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert f"Python executable: {point_guided_python}" in result.stdout
        observed_torch_versions.append(
            next(
                line.split(":", 1)[1].strip()
                for line in result.stdout.splitlines()
                if line.startswith("PyTorch version:")
            )
        )

    expected_torch_version = subprocess.check_output(
        [sys.executable, "-c", "import torch; print(torch.__version__)"],
        text=True,
    ).strip()
    assert observed_torch_versions == [expected_torch_version] * len(launchers)

    invocation_log = (tmp_path / "launcher.log").read_text(encoding="utf-8")
    interpreter_lines = [
        line.removeprefix("interpreter=")
        for line in invocation_log.splitlines()
        if line.startswith("interpreter=")
    ]
    assert interpreter_lines
    # Every wrapper emits one probe and then one explicit module/DDP launch;
    # both invocations must go through the same selected environment.
    assert len(interpreter_lines) == 2 * len(launchers)
    assert interpreter_lines == [sys.executable] * len(interpreter_lines)
    assert "arg=torch.distributed.run" in invocation_log
    assert "arg=smagm.cli.point_guided_train" in invocation_log
    assert "arg=smagm.cli.point_guided_eval" in invocation_log
    assert not conflict_log.exists(), "an ambient PATH python/torchrun executable was selected"
