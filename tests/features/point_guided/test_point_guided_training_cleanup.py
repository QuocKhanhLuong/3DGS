"""Focused exception-cleanup regressions for the point-guided trainer."""

from __future__ import annotations

from types import MethodType, SimpleNamespace
import sys
import types

import pytest
import torch
from torch import nn

import smagm.training.point_guided as training
from smagm.features.point_guided.training_objective import SupervisionConfig


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.semantic_prior = SimpleNamespace(pretrained_loaded=False, checkpoint_loaded=False)


def _zero_route_stats() -> dict[str, object]:
    zero = torch.zeros(())
    return {
        name: zero
        for name in (
            "k_used",
            "path_length_mm",
            "predicted_reward",
            "utility",
            "update_magnitude",
            "fraction_K0",
            "fraction_positive_utility",
            "candidate_reward_mean",
            "candidate_reward_max",
            "r_star_mean",
            "r_star_max",
            "r_star_positive_fraction",
            "travel_cost_mean",
            "overlap_cost_mean",
            "step_cost",
            "utility_before_cost_mean",
            "utility_after_cost_mean",
            "utility_after_cost_max",
        )
    } | {"stop_reasons": {}}


def test_run_epoch_clears_partial_accumulation_gradients_and_preserves_error() -> None:
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = training.PointGuidedTrainer(
        model,
        optimizer,
        training.PointGuidedTrainingSettings(gradient_accumulation=2, amp=False),
        SupervisionConfig(),
        training.DistributedContext(rank=0, local_rank=0, world_size=1, device=torch.device("cpu")),
    )
    trainer.bind_context_module(nn.Identity())
    calls = 0
    injected_error = RuntimeError("injected accumulation failure")

    def injected_forward(self, _batch, *, training: bool, stage_timer):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise injected_error
        total = self.model.weight.square()
        objective = SimpleNamespace(
            total=total,
            reconstruction=SimpleNamespace(total=total),
            reward=total.detach() * 0.0,
            local=total.detach() * 0.0,
            monotonic=total.detach() * 0.0,
            delta=total.detach() * 0.0,
        )
        return objective, total, total.detach() * 0.0, None, None, _zero_route_stats()

    batch = SimpleNamespace(observations=torch.ones(1, 1))
    trainer._forward_objective = MethodType(injected_forward, trainer)
    with pytest.raises(RuntimeError, match="injected accumulation failure") as caught:
        trainer.run_epoch([batch, batch], training=True)  # type: ignore[arg-type]

    assert caught.value is injected_error
    assert calls == 2
    assert model.weight.grad is None
    assert trainer.global_step == 0


class _FakeWandbRun:
    def __init__(self, *, finish_error: BaseException | None = None) -> None:
        self.url = "https://wandb.example/injected"
        self.finish_error = finish_error
        self.finish_calls = 0
        self.log_calls = 0

    def log(self, *_args, **_kwargs) -> None:
        self.log_calls += 1

    def finish(self) -> None:
        self.finish_calls += 1
        if self.finish_error is not None:
            raise self.finish_error


class _FakeTrainer:
    should_fail = False
    failure_error: RuntimeError | None = None

    def __init__(self, model, optimizer, settings, supervision, context) -> None:
        self.model = model
        self.optimizer = optimizer
        self.settings = settings
        self.supervision = supervision
        self.context = context
        self.scaler = None
        self.global_step = 0

    def bind_context_modules(self, *_args, **_kwargs) -> None:
        return None

    def run_epoch(self, _loader, *, training: bool) -> dict[str, float]:
        if training and type(self).should_fail:
            assert type(self).failure_error is not None
            raise type(self).failure_error
        return {}


def _run_training_with_fake_wandb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    should_fail: bool,
    finish_error: BaseException | None = None,
) -> _FakeWandbRun:
    fake_run = _FakeWandbRun(finish_error=finish_error)
    fake_wandb = types.ModuleType("wandb")
    init_calls: list[dict[str, object]] = []

    def fake_init(**kwargs):
        init_calls.append(kwargs)
        return fake_run

    fake_wandb.init = fake_init
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    settings = training.PointGuidedTrainingSettings(
        epochs=1,
        batch_size=1,
        num_workers=0,
        amp=False,
        early_stopping_patience=1,
    )
    supervision = SupervisionConfig()
    model = _TinyModel()
    structural = SimpleNamespace(to_dict=lambda: {"eligible_subject_ids": ["S1", "S2"]})
    inventory = SimpleNamespace(to_dict=lambda: {"complete_subjects": ["S1", "S2"]})
    split = {"train": ("S1",), "val": ("S2",), "test": (), "excluded": (), "all": ("S1", "S2")}
    _FakeTrainer.should_fail = should_fail
    injected_error = RuntimeError("injected run_training failure")
    _FakeTrainer.failure_error = injected_error

    monkeypatch.setattr(
        training,
        "initialize_distributed",
        lambda _device: training.DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device("cpu"),
        ),
    )
    monkeypatch.setattr(training, "build_model_from_config", lambda *_args, **_kwargs: (model, supervision, settings))
    monkeypatch.setattr(training, "validate_metric_data_range", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        training,
        "normalization_space_from_config",
        lambda *_args, **_kwargs: "masked_robust_01_[0,1]",
    )
    monkeypatch.setattr(
        training,
        "_prepare_structurally_eligible_split",
        lambda **_kwargs: (structural, split, "0123456789abcdef" * 4, ("S1", "S2"), inventory),
    )
    monkeypatch.setattr(
        training,
        "build_baseline_optimizer",
        lambda *_args, **_kwargs: (torch.optim.SGD(model.parameters(), lr=0.1), ()),
    )
    monkeypatch.setattr(training, "BraTS21PointGuidedDataset", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(training, "_make_loader", lambda *_args, **_kwargs: (object(), None))
    monkeypatch.setattr(training, "PointGuidedTrainer", _FakeTrainer)
    monkeypatch.setattr(training, "save_clean_inference_checkpoint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(training, "save_training_resume_checkpoint", lambda *_args, **_kwargs: None)

    raw_config = {
        "training": {"device": "cpu", "seed": 1},
        "data": {"normalization": {}, "require_segmentation": False},
        "_wandb": {"enabled": True, "project": "injected"},
    }
    if should_fail:
        with pytest.raises(RuntimeError, match="injected run_training failure") as caught:
            training.run_training(
                raw_config=raw_config,
                data_root=tmp_path / "data",
                output_root=tmp_path / "runs",
                run_name="failure",
            )
        assert caught.value is injected_error
    else:
        summary = training.run_training(
            raw_config=raw_config,
            data_root=tmp_path / "data",
            output_root=tmp_path / "runs",
            run_name="success",
        )
        assert summary is not None
    assert len(init_calls) == 1
    return fake_run


def test_run_training_finishes_initialized_wandb_once_on_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    success_run = _run_training_with_fake_wandb(monkeypatch, tmp_path / "success", should_fail=False)
    assert success_run.finish_calls == 1

    failure_run = _run_training_with_fake_wandb(
        monkeypatch,
        tmp_path / "failure",
        should_fail=True,
        finish_error=RuntimeError("finish cleanup failure"),
    )
    assert failure_run.finish_calls == 1
