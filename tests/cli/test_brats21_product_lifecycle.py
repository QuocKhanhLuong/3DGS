"""Controller-level lifecycle regression without data, CUDA, or network access."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from smagm.cli.brats21_cohort import BraTS21CohortModel
from smagm.cli import brats21_product as product


class _Finish:
    def to_dict(self) -> dict[str, object]:
        return {"mode": "disabled", "run_id": None, "url": None, "fallback_reason": None}


class _Logger:
    instances = 0
    starts = 0

    def __init__(self, **_kwargs: Any) -> None:
        type(self).instances += 1
        self.mode = "disabled"
        self.fallback_reason = None

    def start(self) -> None:
        type(self).starts += 1

    def log(self, _metrics: dict[str, object], step: int | None = None) -> None:
        assert step is None or step >= 0

    def update_summary(self, _values: dict[str, object]) -> None:
        return None

    def finish(self, **_kwargs: Any) -> _Finish:
        return _Finish()


def _cohort() -> BraTS21CohortModel:
    encoder = nn.Linear(1, 1, bias=False)
    head = nn.Linear(1, 1, bias=False)
    field = nn.Linear(1, 1, bias=False)
    projector = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.Adam(
        (parameter for module in (encoder, head, field, projector) for parameter in module.parameters()),
        lr=1e-3,
    )
    return BraTS21CohortModel(
        encoder=encoder,
        gaussian_head=head,
        structural_field=field,
        evidence_projector=projector,
        optimizer=optimizer,
    )


def test_product_controller_initializes_one_global_owner_and_interleaves_patients(tmp_path: Path, monkeypatch) -> None:
    _Logger.instances = _Logger.starts = 0
    calls: dict[str, Any] = {"model_factory": 0, "train": [], "validation": []}
    patients = ("patient-a", "patient-b", "patient-c")
    validation_patients = ("patient-v",)
    config = {
        "product_path": str(Path("configs/experiments/brats21_product_full.json").resolve()),
        "product": {
            "stage": "full",
            "experiment_name": "lifecycle-test",
            "training_split": "train",
            "epochs": 1,
            "max_global_steps": 3,
            "wandb_mode": "disabled",
            "wandb_group": "test",
            "evaluation_config": "../evaluation/brats21_product_eval.json",
            "validation": {"enabled": True, "cadence": "final", "split": "validation", "steps": 1},
            "output_paths": {
                "state_file": "state.json",
                "completion_marker": "complete.json",
                "global_checkpoint": "global.pt",
                "patient_metrics": "metrics.csv",
                "aggregate_metrics": "aggregate.json",
            },
        },
        "training": {"seed": 41, "diagnostics": {"profile_supported_operator_flops": False}},
        "evaluation": {"aggregation": {}},
        "sampling_config_hash": "sampling",
    }
    monkeypatch.setattr(product, "_load_product_config", lambda _path: (config, "config-hash"))
    monkeypatch.setattr(product, "_cohort_and_split_hashes", lambda _config: ("cohort", "split"))
    monkeypatch.setattr(
        product,
        "_patient_ids",
        lambda _config, _stage, _patient_id, _limit, *, split_name=None: patients if split_name == "train" else validation_patients,
    )
    monkeypatch.setattr(product, "_runtime_config", lambda *_args, **_kwargs: {"wandb": {"enabled": False}})
    monkeypatch.setattr(product, "_cuda_preflight", lambda: {"cuda_available": True, "device_count": 1})
    monkeypatch.setattr(product.torch.cuda, "manual_seed_all", lambda _seed: None)
    monkeypatch.setattr(product.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(product.torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(product.torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(product, "_write_product_metric_reports", lambda **_kwargs: {"row_count": 2})

    cohort = _cohort()

    def build(*, logger: Any, global_checkpoint: Path, **_kwargs: Any) -> tuple[BraTS21CohortModel, dict[str, object], str, str]:
        calls["model_factory"] += 1
        calls["logger"] = logger
        cohort.wandb_logger = logger
        cohort.checkpoint_manager = product.GlobalCheckpointManager(global_checkpoint)
        return cohort, {"wandb": {"enabled": False}}, "binding", "adam"

    monkeypatch.setattr(product, "_build_cohort_model", build)

    def one_patient(*, patient_id: str, cohort_model: BraTS21CohortModel | None, cohort_episode_only: bool = False, validation_only: bool = False, external_logger: Any | None = None, **_kwargs: Any) -> dict[str, Any]:
        assert cohort_model is cohort
        if cohort_episode_only:
            calls["train"].append((patient_id, id(cohort_model), id(cohort_model.optimizer), cohort_model.global_step))
            cohort_model.zero_grad()
            x = torch.ones((1, 1))
            loss = cohort_model.structural_field(
                cohort_model.gaussian_head(cohort_model.evidence_projector(cohort_model.encoder(x)))
            ).square().sum()
            loss.backward()
            step = cohort_model.optimizer_step()
            report = {
                "global_step": step,
                "loss": float(loss.detach()),
                "supported_fraction": 0.5,
                "unsupported_fraction": 0.5,
                "anchor_count": 2,
                "structural_gaussian_count": 2,
                "volumetric_gaussian_count": 2,
                "propagation_proposal_count": 2,
                "propagation_child_count": 1,
                "propagation_rejected_budget": 0,
                "propagation_rejected_duplicate": 0,
                "full_step_wall_time_ms": 1.0,
                "pixel_gaussian_candidate_pairs": 4,
                "encoder_forward_flops_2flop_per_mac": 3_290_112_000,
                "profiled_supported_operator_flops": None,
                "receipt_hash": f"receipt-{step}",
            }
            return {"status": "episode_complete", "episode_report": f"episode-{step}.json", "report": report}
        assert validation_only is True
        calls["validation"].append((patient_id, id(cohort_model), id(cohort_model.optimizer), cohort_model.global_step, id(external_logger)))
        return {"status": "complete", "summary": "validation-summary.json", "report": {"e2_r4_p1": {"loss": 1.0, "supported_fraction": 0.5, "unsupported_fraction": 0.5}}}

    monkeypatch.setattr(product, "_run_one_patient", one_patient)
    monkeypatch.setattr("smagm.experiments.wandb.WandbLogger", _Logger)

    state = product.run(
        config_path=Path("unused.json"),
        output_dir=tmp_path,
        wandb_mode="disabled",
    )

    expected_order = product._epoch_patient_order(patients, seed=41, epoch_index=0)
    assert [item[0] for item in calls["train"]] == list(expected_order)
    assert all(item[1] == id(cohort) for item in calls["train"])
    assert len({item[2] for item in calls["train"]}) == 1
    assert [item[3] for item in calls["train"]] == [0, 1, 2]
    assert calls["model_factory"] == 1
    assert _Logger.instances == 1 and _Logger.starts == 1
    assert calls["validation"] == [("patient-v", id(cohort), id(cohort.optimizer), 3, id(calls["logger"]))]
    assert [entry["global_step"] for entry in state["training_history"]] == [1, 2, 3]
    assert state["global_step"] == 3
    assert (tmp_path / "global.pt").is_file()
