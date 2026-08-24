"""Focused resume protocol, RNG, logging, and worker-stream contracts."""

from __future__ import annotations

import copy
import csv
import os
from pathlib import Path
import random
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from smagm.features.point_guided.baseline_checkpoint import (
    build_training_resume_protocol,
    load_training_resume_checkpoint,
    save_training_resume_checkpoint,
)
from smagm.training.point_guided import _seed_point_guided_worker, _write_epoch_logs


def _protocol(*, batch_size: int = 1, epochs: int = 3) -> dict[str, object]:
    return build_training_resume_protocol(
        model_config={"name": "synthetic", "width": 4},
        trajectory_config={"k_max": 2, "write_scale": 0.1},
        supervision_config={"counterfactual_candidates": 2},
        training_settings={
            "seed": 17,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "batch_size": batch_size,
            "gradient_accumulation": 1,
            "gradient_clip": 1.0,
            "lambda_semantic": 0.2,
            "semantic_class_weights": None,
            "amp": False,
            "amp_dtype": "fp32",
            "epochs": epochs,
            "decoder_chunk_size": 32,
            "num_workers": 0,
            "log_interval": 1,
            "prediction_interval": 1,
        },
        normalization_config={"normalization_policy": "masked_robust_01"},
        split_hash="a" * 64,
        overfit=False,
        require_segmentation=True,
        device="cpu",
    )


def _save(path: Path, protocol: dict[str, object], *, run_state: dict[str, object] | None = None) -> None:
    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    save_training_resume_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=None,
        epoch=2,
        global_step=9,
        best_validation_reconstruction_loss=0.25,
        training_config={"settings": {"epochs": 3}},
        split_hash="a" * 64,
        protocol=protocol,
        run_state=run_state,
        metadata={"synthetic": True},
    )


def test_resume_protocol_is_versioned_and_compatible_fields_are_explicit(tmp_path: Path) -> None:
    protocol = _protocol()
    assert protocol["version"] == 1
    assert set(protocol) == {"version", "immutable", "compatible"}
    assert set(protocol["immutable"])  # type: ignore[index]
    assert set(protocol["compatible"])  # type: ignore[index]

    _save(
        tmp_path / "resume.pt",
        protocol,
        run_state={
            "patience_count": 2,
            "first_train_reconstruction_loss": 0.75,
            "overfit_prediction_epochs": (1, 2),
        },
    )
    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    compatible = copy.deepcopy(protocol)
    compatible["compatible"]["epochs"] = 8  # type: ignore[index]
    state = load_training_resume_checkpoint(
        tmp_path / "resume.pt",
        model=model,
        optimizer=optimizer,
        scaler=None,
        expected_split_hash="a" * 64,
        expected_protocol=compatible,
    )
    assert state["run_state"] == {
        "patience_count": 2,
        "first_train_reconstruction_loss": 0.75,
        "overfit_prediction_epochs": (1, 2),
    }


def test_historical_resume_schema_is_rejected_without_migration(tmp_path: Path) -> None:
    path = tmp_path / "historical.pt"
    torch.save({"schema": "point-guided-training-resume-v1"}, path)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="unsupported schema"):
        load_training_resume_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scaler=None,
            expected_split_hash="a" * 64,
        )


def test_resume_immutable_mismatch_fails_before_model_mutation(tmp_path: Path) -> None:
    protocol = _protocol()
    path = tmp_path / "resume.pt"
    _save(path, protocol)
    model = nn.Linear(4, 2)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    incompatible = copy.deepcopy(protocol)
    incompatible["immutable"]["batch_size"] = 2  # type: ignore[index]
    with pytest.raises(ValueError, match="immutable protocol mismatch"):
        load_training_resume_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scaler=None,
            expected_split_hash="a" * 64,
            expected_protocol=incompatible,
        )
    assert all(torch.equal(before[name], model.state_dict()[name]) for name in before)


def test_resume_world_size_mismatch_fails_before_model_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "resume.pt"
    _save(path, _protocol())
    model = nn.Linear(4, 2)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="world size mismatch"):
        load_training_resume_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scaler=None,
            expected_split_hash="a" * 64,
            expected_protocol=_protocol(),
        )
    assert all(torch.equal(before[name], model.state_dict()[name]) for name in before)


def test_metrics_csv_header_mismatch_is_rejected_before_jsonl_append(tmp_path: Path) -> None:
    record = {"epoch": 1, "loss": 0.5}
    _write_epoch_logs(tmp_path, record, header_written=False)
    before_jsonl = (tmp_path / "train.jsonl").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="metrics.csv header"):
        _write_epoch_logs(tmp_path, {"loss": 0.4, "epoch": 2}, header_written=True)
    assert (tmp_path / "train.jsonl").read_text(encoding="utf-8") == before_jsonl
    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        assert tuple(next(csv.reader(handle))) == ("epoch", "loss")


class _RandomStreamDataset(Dataset[tuple[int, float, float, torch.Tensor]]):
    def __len__(self) -> int:
        return 12

    def __getitem__(self, index: int) -> tuple[int, float, float, torch.Tensor]:
        return index, random.random(), float(np.random.random()), torch.rand(())


def _worker_stream() -> list[tuple[int, float, float, float]]:
    loader = DataLoader(_RandomStreamDataset(), batch_size=1, num_workers=2, worker_init_fn=_seed_point_guided_worker)
    return [
        (int(index.item()), float(py), float(numpy), float(tensor))
        for index, py, numpy, tensor in loader
    ]


def test_dataloader_worker_stream_is_deterministic() -> None:
    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    first = _worker_stream()
    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    second = _worker_stream()
    assert first == second


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed is unavailable")
def test_two_rank_torchrun_resume_rng_roundtrip(tmp_path: Path) -> None:
    script = tmp_path / "distributed_resume.py"
    checkpoint = tmp_path / "last_train.pt"
    script.write_text(
        textwrap.dedent(
            """
            import random
            import sys
            from pathlib import Path
            import numpy as np
            import torch
            from torch import nn
            from smagm.features.point_guided.baseline_checkpoint import (
                build_training_resume_protocol,
                load_training_resume_checkpoint,
                save_training_resume_checkpoint,
            )

            checkpoint = Path(sys.argv[1])
            torch.distributed.init_process_group("gloo", init_method="env://")
            rank = torch.distributed.get_rank()
            torch.manual_seed(100 + rank)
            random.seed(100 + rank)
            np.random.seed(100 + rank)
            model = nn.Linear(2, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            protocol = build_training_resume_protocol(
                model_config={"name": "distributed"}, trajectory_config={"k_max": 1},
                supervision_config={"lambda": 1.0},
                training_settings={"seed": 100, "learning_rate": 0.1, "weight_decay": 0.0,
                    "batch_size": 1, "gradient_accumulation": 1, "gradient_clip": None,
                    "lambda_semantic": 0.0, "semantic_class_weights": None, "amp": False,
                    "amp_dtype": "fp32", "epochs": 1, "decoder_chunk_size": 4,
                    "num_workers": 0, "log_interval": 1, "prediction_interval": 1},
                normalization_config=None, split_hash="b" * 64, overfit=False,
                require_segmentation=False, device="cpu")
            save_training_resume_checkpoint(
                checkpoint, model=model, optimizer=optimizer, scaler=None, epoch=1,
                global_step=2, best_validation_reconstruction_loss=0.5,
                training_config={}, split_hash="b" * 64, protocol=protocol,
                metadata={},
            )
            torch.distributed.barrier()
            if rank == 0:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                assert payload["rng_state"]["world_size"] == 2
                assert set(payload["rng_state"]["states"]) == {0, 1}
            random.seed(0); np.random.seed(0); torch.manual_seed(0)
            load_training_resume_checkpoint(
                checkpoint, model=model, optimizer=optimizer, scaler=None,
                expected_split_hash="b" * 64, expected_protocol=protocol)
            print(f"rank={rank}:" + str((random.random(), float(np.random.random()), float(torch.rand(())))))
            torch.distributed.barrier()
            """
        ),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[3] / "src")
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", str(script), str(checkpoint)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "rank=0:" in result.stdout
    assert "rank=1:" in result.stdout
