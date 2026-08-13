"""Synthetic checks for the server training/checkpoint boundary."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from smagm.features.point_guided.baseline_checkpoint import (
    load_training_resume_checkpoint,
    save_clean_inference_checkpoint,
    save_training_resume_checkpoint,
)
from smagm.features.point_guided.baseline_inference import load_validated_baseline_checkpoint
from smagm.features.point_guided.baseline_training import BaselineTrainingConfig, build_baseline_optimizer, resolve_parameter_ownership
from smagm.features.point_guided.config import PointGuidedConfig
from smagm.features.point_guided.model import PointGuidedMRIModel
from smagm.features.point_guided.semantic_supervision import build_coarse_semantic_target, compute_semantic_grounding_loss
from smagm.features.point_guided.training_objective import SupervisionConfig
from smagm.features.point_guided.trajectory_cost import TrajectoryConfig
from smagm.data.brats21_point_guided import PointGuidedBatch
from smagm.training.point_guided import (
    DistributedContext,
    DistributedEvalSampler,
    PointGuidedTrainer,
    PointGuidedTrainingSettings,
    _TrainingContextModule,
    build_model_from_config,
)


def _model() -> PointGuidedMRIModel:
    return PointGuidedMRIModel(
        PointGuidedConfig(
            num_semantic_classes=3,
            num_points=3,
            point_candidate_multiplier=3,
            offset_hidden_channels=12,
        ),
        trajectory_config=TrajectoryConfig(
            lambda_travel=0.05,
            lambda_overlap=0.2,
            lambda_step=0.05,
            k_max=2,
            selection_temperature=0.7,
            write_scale=0.1,
        ),
    )


def test_training_context_and_semantic_term_are_target_after_inference_only() -> None:
    torch.manual_seed(91)
    model = _model()
    observations = torch.randn(1, 3, 7, 7, 7)
    brain_mask = torch.ones(1, 1, 7, 7, 7, dtype=torch.bool)
    target = torch.sigmoid(torch.randn(1, 1, 7, 7, 7))
    segmentation = torch.zeros(1, 7, 7, 7, dtype=torch.long)
    segmentation[:, 1:3, 1:3, 1:3] = 2
    context = model.forward_training_context(observations, brain_mask=brain_mask, chunk_size=29)

    assert "target_t1ce" not in inspect.signature(model.forward_training_context).parameters
    assert "segmentation" not in inspect.signature(model.forward_training_context).parameters
    objective = model.compute_training_objective(
        context,
        target,
        config=SupervisionConfig(counterfactual_candidates=3, high_candidate_count=1, random_candidate_count=1, spill_sample_count=3),
    )
    semantic_target = build_coarse_semantic_target(segmentation, brain_mask, ignore_index=255)
    semantic = compute_semantic_grounding_loss(context.frontend.s_coarse.log(), semantic_target, ignore_index=255)
    combined = objective.total + 0.2 * semantic.loss
    assert torch.isfinite(combined)
    combined.backward()
    head_grad = tuple(parameter.grad for parameter in model.semantic_prior.semantic_head.parameters())
    assert any(gradient is not None and bool(torch.count_nonzero(gradient)) for gradient in head_grad)
    assert all(parameter.grad is None for parameter in model.semantic_prior.backbone.parameters())


def test_optimizer_ownership_and_clean_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = _model()
    optimizer, ownership = build_baseline_optimizer(model, BaselineTrainingConfig())
    assert all(row.optimizer_member for row in ownership if row.module != "semantic_prior.backbone")
    assert all(not parameter.requires_grad for parameter in model.semantic_prior.backbone.parameters())

    clean_path = save_clean_inference_checkpoint(tmp_path / "best_model.pt", model)
    restored = _model()
    load_validated_baseline_checkpoint(restored, clean_path)
    for left, right in zip(model.state_dict().values(), restored.state_dict().values()):
        torch.testing.assert_close(left, right)

    resume_path = save_training_resume_checkpoint(
        tmp_path / "last_train.pt",
        model=model,
        optimizer=optimizer,
        scaler=None,
        epoch=4,
        global_step=11,
        best_validation_reconstruction_loss=0.25,
        training_config={"lambda_semantic": 0.2},
        split_hash="a" * 64,
        metadata={"validation_is_held_out": True},
    )
    restored_optimizer, _ = build_baseline_optimizer(restored, BaselineTrainingConfig())
    state = load_training_resume_checkpoint(
        resume_path,
        model=restored,
        optimizer=restored_optimizer,
        scaler=None,
        expected_split_hash="a" * 64,
    )
    assert state["epoch"] == 4
    assert state["global_step"] == 11


def test_one_synthetic_point_guided_trainer_step_includes_semantic_term() -> None:
    torch.manual_seed(92)
    model = _model()
    optimizer, _ = build_baseline_optimizer(model, BaselineTrainingConfig())
    settings = PointGuidedTrainingSettings(
        epochs=1,
        batch_size=1,
        decoder_chunk_size=29,
        lambda_semantic=0.2,
        amp=False,
        early_stopping_patience=1,
    )
    supervision = SupervisionConfig(
        counterfactual_candidates=3,
        high_candidate_count=1,
        random_candidate_count=1,
        spill_sample_count=3,
    )
    observations = torch.randn(1, 3, 7, 7, 7)
    segmentation = torch.zeros(1, 7, 7, 7, dtype=torch.long)
    segmentation[:, 1:3, 1:3, 1:3] = 2
    batch = PointGuidedBatch(
        observations=observations,
        target_t1ce=torch.sigmoid(torch.randn(1, 1, 7, 7, 7)),
        segmentation=segmentation,
        brain_mask=torch.ones(1, 1, 7, 7, 7, dtype=torch.bool),
        spacing_xyz_mm=torch.ones(1, 3),
        voxel_to_ras_mm=torch.eye(4).unsqueeze(0),
        subject_ids=("BraTS2021_00000",),
        normalization_metadata=({},),
    )
    distributed = DistributedContext(rank=0, local_rank=0, world_size=1, device=torch.device("cpu"))
    trainer = PointGuidedTrainer(model, optimizer, settings, supervision, distributed)
    trainer.bind_context_module(_TrainingContextModule(model))
    stats = trainer.run_epoch([batch], training=True)  # type: ignore[arg-type]
    assert torch.isfinite(torch.tensor(stats["total_loss"]))
    assert stats["semantic_loss"] > 0.0
    assert stats["examples"] == 1.0


def test_server_configs_keep_locked_constants() -> None:
    root = Path(__file__).resolve().parents[3]
    for path in (
        root / "configs/training/point_guided_brats21_overfit.json",
        root / "configs/training/point_guided_brats21_4070.json",
        root / "configs/training/point_guided_brats21_2xa4000.json",
        root / "configs/evaluation/point_guided_brats21_eval.json",
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["model"]["num_semantic_classes"] == 3
        assert payload["model"]["num_points"] == 2048
        assert payload["model"]["support_radius_mm"] == 4.0
        assert payload["model"]["max_displacement_mm"] == 2.0

    model, supervision, settings = build_model_from_config(
        json.loads((root / "configs/training/point_guided_brats21_overfit.json").read_text(encoding="utf-8"))
    )
    assert model.config.num_points == 2048
    assert supervision.counterfactual_candidates == 8
    assert settings.lambda_semantic == 0.2


def test_distributed_eval_sampler_shards_without_padding_duplicates() -> None:
    dataset = list(range(5))
    rank_zero = tuple(DistributedEvalSampler(dataset, rank=0, world_size=2))
    rank_one = tuple(DistributedEvalSampler(dataset, rank=1, world_size=2))
    assert set(rank_zero).isdisjoint(rank_one)
    assert sorted(rank_zero + rank_one) == list(range(5))
