"""Collision-safe reservations and atomic persistence for point-guided runs.

Run directories are deliberately exclusive by default.  A caller may reuse an
existing directory only by passing ``reuse=True`` (training does this only for
an explicit resume path; evaluation exposes an explicit ``--reuse-output``
flag).  A lock is held for the lifetime of the writer so an explicit reuse
cannot run concurrently with another writer.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import tempfile
from typing import Iterator
from uuid import uuid4


class ArtifactCollisionError(FileExistsError):
    """Raised when an artifact destination is already owned or in use."""


@dataclass(slots=True)
class ArtifactReservation:
    """An exclusive lock held for one run/output directory."""

    path: Path
    lock_path: Path
    token: str
    _released: bool = False

    def release(self) -> None:
        """Release this reservation without removing the run directory."""

        if self._released:
            return
        try:
            owner = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self._released = True
            return
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"cannot validate artifact reservation {self.lock_path}") from error
        if not isinstance(owner, dict) or owner.get("token") != self.token:
            raise RuntimeError(f"artifact reservation ownership changed: {self.lock_path}")
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass
        self._released = True

    def __enter__(self) -> "ArtifactReservation":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.release()


def _acquire_lock(path: Path, *, purpose: str) -> ArtifactReservation:
    lock_path = path / ".point-guided.lock"
    token = uuid4().hex
    payload = {
        "token": token,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "purpose": purpose,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise ArtifactCollisionError(
            f"artifact directory is already reserved: {path}; "
            "wait for the active writer or remove a verified stale .point-guided.lock"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        lock_path.unlink(missing_ok=True)
        raise
    return ArtifactReservation(path=path, lock_path=lock_path, token=token)


def reserve_artifact_directory(
    path: str | Path,
    *,
    reuse: bool = False,
    purpose: str = "artifact",
) -> ArtifactReservation:
    """Atomically reserve ``path`` and hold an exclusive writer lock.

    A missing destination is created with one atomic ``mkdir``.  Existing
    destinations are rejected unless ``reuse=True`` is explicit.  Reuse never
    removes or clears existing artifacts; it only permits an intentional
    replacement under the lock.
    """

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir()
    except FileExistsError as error:
        if not destination.is_dir():
            raise ArtifactCollisionError(f"artifact destination is not a directory: {destination}") from error
        if not reuse:
            raise ArtifactCollisionError(
                f"{purpose} destination already exists: {destination}; "
                "choose a new destination or explicitly request reuse"
            ) from error
    reservation = _acquire_lock(destination, purpose=purpose)
    return reservation


def reserve_run_directory(
    output_root: str | Path,
    run_name: str | None = None,
    *,
    reuse: bool = False,
) -> ArtifactReservation:
    """Reserve a training run directory, disambiguating generated names."""

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if run_name is not None:
        name = Path(run_name)
        if name.name != str(name) or name.is_absolute() or str(name) in {"", ".", ".."}:
            raise ValueError("run_name must be a single relative directory name")
        return reserve_artifact_directory(root / name, reuse=reuse, purpose="training run")

    base = datetime.now(timezone.utc).strftime("point-guided-%Y%m%dT%H%M%SZ")
    for attempt in range(1000):
        suffix = "" if attempt == 0 else f"-{uuid4().hex[:8]}"
        try:
            return reserve_artifact_directory(root / f"{base}{suffix}", purpose="training run")
        except ArtifactCollisionError:
            continue
    raise RuntimeError(f"could not reserve a unique training run below {root}")


@contextmanager
def atomic_output_path(path: str | Path, *, suffix: str = ".tmp") -> Iterator[Path]:
    """Yield a unique sibling temporary path and atomically replace ``path``."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=suffix,
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write text through a unique sibling and atomic replacement."""

    destination = Path(path)
    with atomic_output_path(destination) as temporary:
        with temporary.open("w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    return destination


def atomic_write_json(path: str | Path, payload: object) -> Path:
    """Serialize JSON and atomically replace the destination."""

    return atomic_write_text(
        path,
        json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n",
    )


def atomic_torch_save(payload: object, path: str | Path) -> Path:
    """Save a PyTorch payload through a unique sibling and atomic replacement."""

    import torch

    destination = Path(path)
    with atomic_output_path(destination) as temporary:
        torch.save(payload, temporary)
    return destination


__all__ = [
    "ArtifactCollisionError",
    "ArtifactReservation",
    "atomic_output_path",
    "atomic_torch_save",
    "atomic_write_json",
    "atomic_write_text",
    "reserve_artifact_directory",
    "reserve_run_directory",
]
