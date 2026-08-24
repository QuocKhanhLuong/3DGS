"""Focused tests for best-effort Git provenance in point-guided runs."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
from torch import nn

import smagm.cli.point_guided_eval as point_guided_eval
import smagm.features.point_guided.provenance as provenance


def test_best_effort_git_head_reads_the_current_repository() -> None:
    repository_root = Path(__file__).resolve().parents[3]

    head = provenance.best_effort_git_head(repository_root)

    assert head is not None
    assert len(head) == 40
    assert all(character in "0123456789abcdef" for character in head)


@pytest.mark.parametrize("failure", (FileNotFoundError("git"), subprocess.CalledProcessError(128, "git")))
def test_best_effort_git_head_returns_none_when_git_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: BaseException,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> bytes:
        raise failure

    monkeypatch.setattr(provenance.subprocess, "check_output", fail)

    assert provenance.best_effort_git_head(tmp_path) is None


def test_evaluation_metadata_persists_git_head_and_allows_unavailable_git(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run" / "checkpoints" / "best_model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    split_file = checkpoint.parent.parent / "split.json"
    split_file.touch()
    config_path = tmp_path / "evaluation.json"
    config_path.write_text(
        json.dumps({"data": {"normalization": {"normalization_policy": "masked_robust_01"}}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(point_guided_eval, "discover_point_guided_subjects", lambda _root: ())
    monkeypatch.setattr(point_guided_eval, "resolve_split_file", lambda *_args, **_kwargs: split_file)
    monkeypatch.setattr(
        point_guided_eval,
        "_load_split",
        lambda *_args, **_kwargs: ({"train": (), "val": (), "test": ()}, "a" * 64),
    )
    monkeypatch.setattr(
        point_guided_eval,
        "build_model_from_config",
        lambda *_args, **_kwargs: (nn.Module(), None, None),
    )
    monkeypatch.setattr(point_guided_eval, "load_validated_baseline_checkpoint", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(point_guided_eval, "BraTS21PointGuidedDataset", lambda *_args, **_kwargs: ())

    expected_head = "0123456789abcdef" * 2 + "0123456789abcdef"[:8]
    monkeypatch.setattr(point_guided_eval, "best_effort_git_head", lambda: expected_head)
    output_dir = tmp_path / "evaluation-with-git"
    point_guided_eval._evaluate_reserved(
        checkpoint=checkpoint,
        config_path=config_path,
        data_root=tmp_path / "data",
        output_dir=output_dir,
        split_file=split_file,
        split_name="test",
        device_name="cpu",
        save_predictions=False,
    )
    metadata = json.loads((output_dir / "evaluation_metadata.json").read_text(encoding="utf-8"))
    assert metadata["git_head"] == expected_head

    monkeypatch.setattr(point_guided_eval, "best_effort_git_head", lambda: None)
    unavailable_output_dir = tmp_path / "evaluation-without-git"
    point_guided_eval._evaluate_reserved(
        checkpoint=checkpoint,
        config_path=config_path,
        data_root=tmp_path / "data",
        output_dir=unavailable_output_dir,
        split_file=split_file,
        split_name="test",
        device_name="cpu",
        save_predictions=False,
    )
    unavailable_metadata = json.loads(
        (unavailable_output_dir / "evaluation_metadata.json").read_text(encoding="utf-8")
    )
    assert unavailable_metadata["git_head"] is None
