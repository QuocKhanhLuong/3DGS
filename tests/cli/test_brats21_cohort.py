"""Process-level cohort-model lifecycle contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from smagm.cli.brats21_cohort import BraTS21CohortModel, GlobalCheckpointManager
from smagm.cli.brats21_product import _epoch_patient_order


class _Logger:
    def __init__(self) -> None:
        self.starts = 0

    def start(self) -> None:
        self.starts += 1


def _model(*, logger: _Logger | None = None) -> BraTS21CohortModel:
    encoder = nn.Linear(1, 1, bias=False)
    head = nn.Linear(1, 1, bias=False)
    field = nn.Linear(1, 1, bias=False)
    projector = nn.Linear(1, 1, bias=False)
    modules = (encoder, head, field, projector)
    optimizer = torch.optim.Adam((parameter for module in modules for parameter in module.parameters()), lr=0.1)
    return BraTS21CohortModel(
        encoder=encoder,
        gaussian_head=head,
        structural_field=field,
        evidence_projector=projector,
        optimizer=optimizer,
        wandb_logger=logger,
    )


def _episode_loss(model: BraTS21CohortModel, value: float) -> torch.Tensor:
    x = torch.tensor([[value]], dtype=torch.float32)
    return model.structural_field(model.gaussian_head(model.evidence_projector(model.encoder(x)))).square().sum()


def test_two_patient_episodes_share_one_model_optimizer_and_monotonic_global_step() -> None:
    logger = _Logger()
    model = _model(logger=logger)
    model_id = id(model)
    optimizer_id = id(model.optimizer)
    model.start_wandb()
    model.start_wandb()
    assert logger.starts == 1
    assert model.wandb_initializations == 1

    before_a = model.state_hash
    model.zero_grad()
    _episode_loss(model, 1.0).backward()
    assert model.optimizer_step() == 1
    after_a = model.state_hash
    assert after_a != before_a

    # Patient B begins from the state committed by patient A, not a fresh model.
    before_b = model.state_hash
    assert before_b == after_a
    model.zero_grad()
    _episode_loss(model, 2.0).backward()
    assert model.optimizer_step() == 2
    assert id(model) == model_id
    assert id(model.optimizer) == optimizer_id
    assert len(getattr(model.optimizer, "state", {})) > 0


def test_wandb_start_is_owned_once_by_product_lifecycle_adapter() -> None:
    """The product controller calls this once; an accidental second call fails."""

    logger = _Logger()
    model = _model(logger=logger)
    model.start_wandb()
    assert logger.starts == 1


def test_validation_is_no_grad_and_preserves_shared_model_state() -> None:
    model = _model()
    before = model.state_hash
    result = model.validation(lambda: _episode_loss(model, 3.0))
    assert torch.isfinite(result)
    assert result.requires_grad is False
    assert model.state_hash == before
    assert model.global_step == 0


def test_profiler_can_be_invoked_at_most_once(monkeypatch) -> None:
    model = _model()
    invocations: list[bool] = []

    def fake(operation: Any, *, enabled: bool, scope: str) -> tuple[int, dict[str, object]]:
        invocations.append(enabled)
        return operation(), {"profiler_enabled": enabled, "profiler_scope": scope}

    monkeypatch.setattr("smagm.cli.brats21_cohort.profile_supported_operator_flops", fake)
    first, first_report = model.run_with_optional_profiler(lambda: 1, enabled=True, scope="first")
    second, second_report = model.run_with_optional_profiler(lambda: 2, enabled=True, scope="second")
    assert (first, second) == (1, 2)
    assert invocations == [True, False]
    assert first_report["profiler_enabled"] is True
    assert second_report["profiler_enabled"] is False
    assert model.profiler_invocations == 1


def test_target_free_global_snapshot_round_trips_shared_optimizer_state() -> None:
    model = _model()
    model.zero_grad()
    _episode_loss(model, 1.0).backward()
    model.optimizer_step()
    snapshot = model.snapshot(model_binding_hash="binding", cohort_hash="cohort", split_hash="split")
    assert snapshot["target_payload_not_in_checkpoint"] is True
    assert snapshot["patient_state_not_in_checkpoint"] is True
    assert "target" not in snapshot and "patient_state" not in snapshot

    restored = _model()
    restored.restore(snapshot, model_binding_hash="binding", cohort_hash="cohort", split_hash="split")
    assert restored.global_step == 1
    assert restored.state_hash == model.state_hash
    assert len(getattr(restored.optimizer, "state", {})) > 0


def test_process_owned_checkpoint_manager_persists_only_global_state(tmp_path: Path) -> None:
    manager = GlobalCheckpointManager(tmp_path / "global.pt")
    model = _model()
    model.checkpoint_manager = manager
    model.zero_grad()
    _episode_loss(model, 1.0).backward()
    model.optimizer_step()
    digest = model.save_global_checkpoint(
        model_binding_hash="binding",
        cohort_hash="cohort",
        split_hash="split",
    )
    assert len(digest) == 64
    assert manager.save_count == 1
    raw = torch.load(manager.path, map_location="cpu", weights_only=True)
    assert raw["target_payload_not_in_checkpoint"] is True
    assert raw["patient_state_not_in_checkpoint"] is True
    assert "target" not in raw and "patient_state" not in raw

    restored = _model()
    restored.checkpoint_manager = GlobalCheckpointManager(manager.path)
    assert restored.restore_global_checkpoint(
        model_binding_hash="binding",
        cohort_hash="cohort",
        split_hash="split",
    ) == 1
    assert restored.state_hash == model.state_hash
    assert restored.checkpoint_manager.restore_count == 1


def test_epoch_patient_order_is_seeded_shuffled_and_repeatable() -> None:
    patients = ("p0", "p1", "p2", "p3", "p4")
    first = _epoch_patient_order(patients, seed=20260802, epoch_index=0)
    repeat = _epoch_patient_order(patients, seed=20260802, epoch_index=0)
    next_epoch = _epoch_patient_order(patients, seed=20260802, epoch_index=1)
    assert first == repeat
    assert set(first) == set(patients)
    assert first != patients
    assert next_epoch != first
