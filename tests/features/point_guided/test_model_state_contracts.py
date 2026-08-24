"""Focused regressions for Gate-E ownership and Gate-G mode restoration."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from types import SimpleNamespace

import smagm.features.point_guided.model as model_module
from smagm.features.point_guided.config import PointGuidedConfig
from smagm.features.point_guided.model import PointGuidedMRIModel
from smagm.features.point_guided.trajectory_cost import TrajectoryConfig
from smagm.features.point_guided.baseline_inference import GateGInferenceConfig


def _model() -> PointGuidedMRIModel:
    return PointGuidedMRIModel(
        PointGuidedConfig(
            num_semantic_classes=3,
            num_points=2,
            point_candidate_multiplier=2,
            offset_hidden_channels=12,
        ),
        trajectory_config=TrajectoryConfig(
            lambda_travel=0.01,
            lambda_overlap=0.01,
            lambda_step=0.01,
            k_max=1,
            selection_temperature=0.7,
            write_scale=0.1,
        ),
    )


def _input() -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, steps=3 * 7 * 7 * 7).reshape(1, 3, 7, 7, 7)


def test_gate_e_rejects_context_from_another_model_before_target_supervision() -> None:
    torch.manual_seed(6106)
    producer = _model().eval()
    receiver = _model().eval()
    context = producer.forward_training_context(_input(), chunk_size=19)

    with pytest.raises(ValueError, match="different PointGuidedMRIModel"):
        # The receiver must reject the context before the target reaches the
        # Gate-E objective implementation.  This deliberately uses a target
        # shape that would also fail later, making the ownership error the
        # observable first boundary.
        receiver.compute_training_objective(context, torch.tensor(0.0))


def _mixed_training_modes(model: PointGuidedMRIModel) -> dict[str, bool]:
    assert model.trajectory is not None and model.decoder is not None
    model.train()
    model.trajectory.eval()
    model.trajectory.update_net.train()
    model.decoder.eval()
    model.decoder.mlp[0].train()
    return {name: module.training for name, module in model.named_modules()}


@pytest.mark.parametrize("raises", (False, True), ids=("success", "exception"))
def test_gate_g_restores_each_module_mode_on_success_and_exception(
    monkeypatch: pytest.MonkeyPatch,
    raises: bool,
) -> None:
    model = _model()
    before = _mixed_training_modes(model)
    sentinel = object()

    # Keep this state-management test independent of frontend numerics.  The
    # production wrapper still enters eval/no-grad before either call below.
    monkeypatch.setattr(
        model,
        "_forward_frontend_with_gate_b_context",
        lambda *_args, **_kwargs: (
            SimpleNamespace(
                base_planes=None,
                refined_points_ras_mm=None,
                point_semantic=None,
                f_spec=None,
                reliability=None,
                geometry=None,
                s_coarse=None,
            ),
            None,
            None,
        ),
    )
    if raises:
        def fail(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("synthetic Gate-G failure")

        monkeypatch.setattr(model_module, "run_baseline_inference", fail)
        with pytest.raises(RuntimeError, match="synthetic Gate-G failure"):
            model.forward_baseline_inference(
                _input(),
                inference_config=GateGInferenceConfig(k_max=1, decoder_chunk_size=19),
            )
    else:
        monkeypatch.setattr(model_module, "run_baseline_inference", lambda *_args, **_kwargs: sentinel)
        assert (
            model.forward_baseline_inference(
                _input(),
                inference_config=GateGInferenceConfig(k_max=1, decoder_chunk_size=19),
            )
            is sentinel
        )

    after = {name: module.training for name, module in model.named_modules()}
    assert after == before
    assert model.training is True
    assert model.trajectory is not None and model.decoder is not None
    assert model.trajectory.training is False
    assert model.trajectory.update_net.training is True
    assert model.decoder.training is False
    assert isinstance(model.decoder.mlp[0], nn.Linear)
    assert model.decoder.mlp[0].training is True
