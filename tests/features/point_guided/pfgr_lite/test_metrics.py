from __future__ import annotations

import json

import pytest
import torch

from smagm.features.point_guided.pfgr_lite.metrics import (
    ComparisonOptions,
    action_metric_row,
    aggregate_subject_metrics,
    compare_paired_artifacts,
    dense_metrics,
    direct_dense_metrics,
    paired_subject_metrics,
    scientific_decision,
    telescoping_residual,
)


def test_direct_dense_reference_and_signed_improvement() -> None:
    target = torch.zeros((3, 3, 3), dtype=torch.float64)
    before = torch.ones_like(target)
    after = torch.full_like(target, 0.5)
    mask = torch.ones_like(target, dtype=torch.bool)
    direct = direct_dense_metrics(after, target, mask, data_range=1.0, ssim_window=3)
    paired = paired_subject_metrics(before, after, target, mask, data_range=1.0, ssim_window=3)
    assert direct["mae"] == pytest.approx(0.5)
    assert paired["before"]["mae"] == pytest.approx(1.0)
    assert paired["after"]["mae"] == pytest.approx(0.5)
    assert paired["improvement"]["mae"] == pytest.approx(0.5)
    assert paired["improvement"]["psnr"] is not None
    assert paired["after"]["ssim"] is not None


def test_ssim_unavailable_is_explicit_for_small_volume() -> None:
    value = torch.zeros((2, 2, 2), dtype=torch.float64)
    result = dense_metrics(value, value, ssim_window=3)
    assert result["ssim"] is None
    assert result["ssim_unavailable_reason"] == "ssim_window_larger_than_volume"


def test_ssim_default_window_boundary_and_mask_definition() -> None:
    small = dense_metrics(torch.zeros((9, 9, 9), dtype=torch.float64), torch.zeros((9, 9, 9), dtype=torch.float64))
    assert small["ssim"] is None
    assert small["ssim_unavailable_reason"] == "ssim_window_larger_than_volume"
    valid = dense_metrics(torch.zeros((11, 11, 11), dtype=torch.float64), torch.zeros((11, 11, 11), dtype=torch.float64))
    assert valid["ssim"] is not None
    assert valid["ssim_mask_definition"] == "center_observed_valid_window_v1"


def test_metric_reduction_is_batch_partition_invariant() -> None:
    target = torch.zeros((3, 3, 3), dtype=torch.float32)
    rows = [
        paired_subject_metrics(torch.ones_like(target), torch.zeros_like(target), target, subject_id="a", budget=1),
        paired_subject_metrics(torch.ones_like(target), torch.full_like(target, 0.5), target, subject_id="b", budget=2),
        paired_subject_metrics(torch.ones_like(target), torch.full_like(target, 0.25), target, subject_id="c", budget=0),
    ]
    whole = aggregate_subject_metrics(rows)
    left = aggregate_subject_metrics(rows[:1])
    right = aggregate_subject_metrics(rows[1:])
    assert whole["subject_count"] == left["subject_count"] + right["subject_count"]
    assert whole["sum"]["improvement"]["mae"] == pytest.approx(
        left["sum"]["improvement"]["mae"] + right["sum"]["improvement"]["mae"]
    )
    assert whole["k_bins"] == {"0": 1, "1": 1, "2": 1, "3": 0, "4": 0}


def test_action_signs_and_telescoping_are_unclipped() -> None:
    useful = action_metric_row({"action_id": "p", "raw_gain": 0.2}, practical_margin=0.05)
    harmful = action_metric_row({"action_id": "n", "raw_gain": -0.3}, practical_margin=0.05)
    neutral = action_metric_row({"action_id": "z", "raw_gain": 0.0}, numerical_tolerance=1e-6)
    assert useful["classification"] == "useful_positive"
    assert harmful["classification"] == "harmful_negative"
    assert neutral["classification"] == "numerically_neutral"
    assert telescoping_residual((0.2, -0.1), 1.0, 0.9) == pytest.approx(0.0)


def test_scientific_decision_underpowered_is_inconclusive() -> None:
    result = scientific_decision([0.5, 0.6], practical_margin=0.1, minimum_subjects=3)
    assert result["decision"] == "INCONCLUSIVE"
    assert result["underpowered"] is True


def test_paired_comparison_strict_joins_and_recovery_scope() -> None:
    def rows(values):
        return [
            {
                "subject_id": f"s{i}",
                "improvement": {"masked_charbonnier": float(value)},
                "z0_digest": f"z0-{i}",
            }
            for i, value in enumerate(values)
        ]

    provenance = {
        "checkpoint_hash": "c",
        "producer_compatibility_hash": "p",
        "baseline_split_hash": "split",
        "training_role_manifest_hash": "roles",
        "split_role": "validation",
        "normalization_hash": "n",
        "mask_definition": "observation_derived_binary",
        "loss_definition": "masked_charbonnier_global_v1",
    }

    result = compare_paired_artifacts(
        {"rows": rows((0.4, 0.5)), "source_receipt": provenance},
        {"rows": rows((0.1, 0.2)), "source_receipt": provenance},
        {"rows": rows((0.6, 0.7)), "source_receipt": provenance},
        options=ComparisonOptions(minimum_subjects=2, practical_margin=0.05),
    )
    assert result["scientific_status"] == "PASS"
    assert result["headroom"]["recovery_denominator"] == 2
    assert result["stop_aware"]["route_gap_values"] == pytest.approx((0.2, 0.2))
    assert result["stop_aware"]["top1_regret_values"] == []
    with pytest.raises(ValueError, match="provenance mismatch"):
        compare_paired_artifacts(
            {"rows": rows((0.4,)), "source_receipt": {**provenance, "checkpoint_hash": "c"}},
            {"rows": rows((0.1,)), "source_receipt": {**provenance, "checkpoint_hash": "other"}},
            {"rows": rows((0.6,)), "source_receipt": {**provenance, "checkpoint_hash": "c"}},
            options=ComparisonOptions(minimum_subjects=1),
        )


def test_paired_comparison_allows_unavailable_learned_artifact() -> None:
    def rows(values):
        return [
            {
                "subject_id": f"s{i}",
                "improvement": {"masked_charbonnier": float(value)},
                "z0_digest": f"z0-{i}",
            }
            for i, value in enumerate(values)
        ]

    provenance = {
        "initialization_hash": "init",
        "producer_compatibility_hash": "p",
        "baseline_split_hash": "split",
        "training_role_manifest_hash": "roles",
        "split_role": "validation",
        "normalization_hash": "n",
        "mask_definition": "observation_derived_binary",
        "loss_definition": "masked_charbonnier_global_v1",
    }

    result = compare_paired_artifacts(
        None,
        {"rows": rows((0.1, 0.2)), "source_receipt": provenance},
        {"rows": rows((0.4, 0.5)), "source_receipt": provenance},
        options=ComparisonOptions(minimum_subjects=2, practical_margin=0.05),
    )
    assert result["subject_count"] == 2
    assert result["pairwise"]["oracle_vs_random"]["decision"]["decision"] == "PASS"
    assert result["pairwise"]["learned_vs_random"]["decision"]["decision"] == "INCONCLUSIVE"
    assert result["headroom"]["recovery_unknown_reason"] == "learned_artifact_unavailable"


def test_paired_comparison_rejects_missing_required_provenance_or_z0() -> None:
    rows = [{"subject_id": "s0", "improvement": {"masked_charbonnier": 0.1}}]
    with pytest.raises(ValueError, match="requires one of producer"):
        compare_paired_artifacts(
            None,
            {"rows": rows, "source_receipt": {"split_role": "validation"}},
            {"rows": rows, "source_receipt": {"split_role": "validation"}},
        )


def test_paired_comparison_rejects_file_receipt_override(tmp_path) -> None:
    provenance = {
        "initialization_hash": "init",
        "baseline_split_hash": "split",
        "training_role_manifest_hash": "roles",
        "producer_compatibility_hash": "p",
        "split_role": "validation",
        "normalization_hash": "n",
        "mask_definition": "observation_derived_binary",
        "loss_definition": "masked_charbonnier_global_v1",
    }
    metrics_path = tmp_path / "metrics.json"
    rows_path = tmp_path / "paired_subjects.jsonl"
    metrics_path.write_text(
        json.dumps({"source_receipt": provenance}), encoding="utf-8"
    )
    rows_path.write_text(
        json.dumps(
            {
                "subject_id": "s0",
                "z0_digest": "z0",
                "source_receipt": {**provenance, "producer_compatibility_hash": "other"},
                "improvement": {"masked_charbonnier": 0.1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    good = {
        "rows": [
            {
                "subject_id": "s0",
                "z0_digest": "z0",
                "improvement": {"masked_charbonnier": 0.2},
            }
        ],
        "source_receipt": provenance,
    }
    with pytest.raises(ValueError, match="overlapping field 'producer_compatibility_hash'"):
        compare_paired_artifacts(
            None,
            {
                "metrics_path": metrics_path,
                "paired_subjects_path": rows_path,
            },
            good,
            options=ComparisonOptions(minimum_subjects=1),
        )
