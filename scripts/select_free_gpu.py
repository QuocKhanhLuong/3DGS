#!/usr/bin/env python3
"""Print the visible GPU with the most free memory, failing closed."""

from __future__ import annotations

import subprocess


def main() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise SystemExit("nvidia-smi is unavailable; refusing to select a CPU fallback")
    rows = []
    for line in result.stdout.splitlines():
        index, free = (item.strip() for item in line.split(",", 1))
        rows.append((int(free), index))
    if not rows:
        raise SystemExit("no NVIDIA devices were reported")
    print(max(rows)[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
