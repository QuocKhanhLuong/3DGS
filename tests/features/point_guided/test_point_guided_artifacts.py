"""Bounded concurrency checks for point-guided run and artifact persistence."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading

import pytest

from smagm.cli import point_guided_eval
from smagm.features.point_guided.artifacts import (
    ArtifactCollisionError,
    atomic_output_path,
    atomic_write_json,
    reserve_artifact_directory,
    reserve_run_directory,
)


def test_explicit_run_reservation_rejects_collision_and_allows_only_explicit_reuse(tmp_path: Path) -> None:
    run_path = tmp_path / "run-01"
    owner = reserve_artifact_directory(run_path, purpose="test run")
    try:
        with pytest.raises(ArtifactCollisionError, match="already exists|already reserved"):
            reserve_artifact_directory(run_path, purpose="test run")
    finally:
        owner.release()

    with pytest.raises(ArtifactCollisionError, match="already exists"):
        reserve_artifact_directory(run_path, purpose="test run")
    reused = reserve_artifact_directory(run_path, reuse=True, purpose="test run")
    reused.release()
    assert run_path.is_dir()
    assert not (run_path / ".point-guided.lock").exists()


def test_simultaneous_run_reservation_has_one_owner(tmp_path: Path) -> None:
    destination = tmp_path / "same-run"
    start = threading.Barrier(8)
    release = threading.Event()
    acquired = threading.Event()

    def attempt() -> str:
        start.wait(timeout=5)
        try:
            reservation = reserve_artifact_directory(destination, purpose="concurrency test")
        except ArtifactCollisionError:
            return "collision"
        acquired.set()
        assert release.wait(timeout=5)
        reservation.release()
        return "owner"

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(attempt) for _ in range(8)]
        assert acquired.wait(timeout=5)
        release.set()
        results = [future.result(timeout=5) for future in futures]

    assert results.count("owner") == 1
    assert results.count("collision") == 7
    assert not (destination / ".point-guided.lock").exists()


def test_generated_run_names_are_unique_under_bounded_concurrency(tmp_path: Path) -> None:
    start = threading.Barrier(6)

    def reserve_and_release() -> Path:
        start.wait(timeout=5)
        reservation = reserve_run_directory(tmp_path / "runs")
        path = reservation.path
        reservation.release()
        return path

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(reserve_and_release) for _ in range(6)]
        paths = [future.result(timeout=5) for future in futures]

    assert len(set(paths)) == 6
    assert all(path.is_dir() for path in paths)


def test_atomic_json_writers_do_not_share_fixed_temp_files(tmp_path: Path) -> None:
    destination = tmp_path / "aggregate_metrics.json"
    start = threading.Barrier(8)

    def write(index: int) -> None:
        start.wait(timeout=5)
        atomic_write_json(destination, {"writer": index, "values": [index] * 128})

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(write, index) for index in range(8)]
        for future in futures:
            future.result(timeout=5)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["writer"] in range(8)
    assert payload["values"] == [payload["writer"]] * 128
    assert not list(tmp_path.glob(".aggregate_metrics.json.*.tmp"))


def test_atomic_output_path_replaces_prediction_payload_without_partial_file(tmp_path: Path) -> None:
    destination = tmp_path / "subject_t1ce_pred.nii.gz"
    with atomic_output_path(destination, suffix=".nii.gz") as temporary:
        temporary.write_bytes(b"complete-prediction-payload")

    assert destination.read_bytes() == b"complete-prediction-payload"
    assert not list(tmp_path.glob(".subject_t1ce_pred.nii.gz.*.nii.gz"))


def test_evaluation_reuse_is_explicit_and_locked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    output.mkdir()
    observed: list[Path] = []

    def fake_evaluate_reserved(**kwargs: object) -> dict[str, bool]:
        path = kwargs["output_dir"]
        assert isinstance(path, Path)
        observed.append(path)
        assert (path / ".point-guided.lock").exists()
        return {"ok": True}

    monkeypatch.setattr(point_guided_eval, "_evaluate_reserved", fake_evaluate_reserved)
    with pytest.raises(ArtifactCollisionError, match="already exists"):
        point_guided_eval.evaluate(
            checkpoint=tmp_path / "checkpoint.pt",
            config_path=tmp_path / "config.json",
            data_root=tmp_path / "data",
            output_dir=output,
            split_file=None,
            split_name="test",
            device_name="cpu",
            save_predictions=False,
        )
    assert point_guided_eval.evaluate(
        checkpoint=tmp_path / "checkpoint.pt",
        config_path=tmp_path / "config.json",
        data_root=tmp_path / "data",
        output_dir=output,
        split_file=None,
        split_name="test",
        device_name="cpu",
        save_predictions=False,
        reuse_output=True,
    ) == {"ok": True}
    assert observed == [output.resolve()]
    assert not (output / ".point-guided.lock").exists()
