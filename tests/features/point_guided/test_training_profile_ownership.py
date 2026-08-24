"""Regression tests proving offset predictor width and ownership across training profiles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from smagm.features.point_guided.baseline_training import (
    BaselineTrainingConfig,
    build_baseline_optimizer,
    resolve_parameter_ownership,
)
from smagm.training.point_guided import build_model_from_config


EXPECTED_TRAINABLE_MODULES: dict[str, int] = {
    "semantic_head": 1539,
    "point_refiner.offset_predictor": 1419,
    "base_plane_projector": 579,
    "spectral_anchor_builder.band_projector": 520,
    "trajectory.state_initializer": 2080,
    "trajectory.reward_net": 8193,
    "trajectory.update_net": 47072,
    "decoder": 8321,
}

EXPECTED_FROZEN_BACKBONE_PARAMS: int = 14_399_424
EXPECTED_TOTAL_TRAINABLE_PARAMS: int = 69_723

ACCEPTED_TRAINING_PROFILE_NAMES: tuple[str, ...] = (
    "point_guided_brats21_4070.json",
    "point_guided_brats21_2xa4000.json",
    "point_guided_brats21_overfit.json",
)


def _load_profile(name: str) -> dict:
    root = Path(__file__).resolve().parents[3]
    path = root / "configs" / "training" / name
    assert path.is_file(), f"Profile {path} does not exist"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("profile_name", ACCEPTED_TRAINING_PROFILE_NAMES)
def test_accepted_training_profile_declares_locked_offset_hidden_channels_12(profile_name: str) -> None:
    payload = _load_profile(profile_name)
    assert payload["model"]["offset_hidden_channels"] == 12


@pytest.mark.parametrize("profile_name", ACCEPTED_TRAINING_PROFILE_NAMES)
def test_accepted_training_profile_instantiates_exact_offset_predictor_1419_params(profile_name: str) -> None:
    payload = _load_profile(profile_name)
    model, _, _ = build_model_from_config(payload)

    assert model.config.offset_hidden_channels == 12
    assert model.point_refiner.offset_predictor.hidden_channels == 12
    assert model.point_refiner.offset_predictor.descriptor_channels == 114

    offset_params = sum(p.numel() for p in model.point_refiner.offset_predictor.parameters())
    assert offset_params == 1419
    assert all(p.requires_grad for p in model.point_refiner.offset_predictor.parameters())


@pytest.mark.parametrize("profile_name", ACCEPTED_TRAINING_PROFILE_NAMES)
def test_accepted_training_profile_exact_eight_trainable_module_ownership(profile_name: str) -> None:
    payload = _load_profile(profile_name)
    model, _, _ = build_model_from_config(payload)
    optimizer, ownership = build_baseline_optimizer(model, BaselineTrainingConfig())

    rows = {row.module: row for row in ownership}

    # Verify exact 8 authorized trainable modules and parameter counts
    actual_trainable_counts = {
        name: rows[name].parameter_count
        for name in EXPECTED_TRAINABLE_MODULES
    }
    assert actual_trainable_counts == EXPECTED_TRAINABLE_MODULES

    # Verify each authorized module requires grad and is in the optimizer
    for name in EXPECTED_TRAINABLE_MODULES:
        assert rows[name].requires_grad, f"Module {name} must require grad"
        assert rows[name].optimizer_member, f"Module {name} must be in optimizer"

    # Verify frozen MedicalNet backbone
    assert rows["semantic_prior.backbone"].parameter_count == EXPECTED_FROZEN_BACKBONE_PARAMS
    assert not rows["semantic_prior.backbone"].requires_grad
    assert not rows["semantic_prior.backbone"].optimizer_member
    assert all(not p.requires_grad for p in model.semantic_prior.backbone.parameters())

    # Verify SWT filters have zero parameters
    assert not tuple(model.spectral_anchor_builder.swt.parameters())

    # Verify total trainable parameter count
    total_trainable = sum(row.parameter_count for row in ownership if row.requires_grad)
    assert total_trainable == EXPECTED_TOTAL_TRAINABLE_PARAMS

    # Verify optimizer membership exactly matches trainable parameters
    optimizer_param_ids = {
        id(param) for group in optimizer.param_groups for param in group["params"]
    }
    trainable_model_param_ids = {
        id(param) for param in model.parameters() if param.requires_grad
    }
    assert optimizer_param_ids == trainable_model_param_ids
    assert len(optimizer_param_ids) == sum(1 for p in model.parameters() if p.requires_grad)

    # Re-verify ownership consistency
    assert resolve_parameter_ownership(model, optimizer) == ownership
