from __future__ import annotations

import json

import pytest
import torch
from torch.nn import functional as F

from smagm.features.point_guided.pfgr_lite import metrics as metrics_module
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


def _dense_box_ssim_reference(prediction, target, *, mask, data_range, window, diagnostics=None):
    """Independent reference: the original five dense cubic FP64 reductions."""

    if min(prediction.shape) < window:
        return None, "ssim_window_larger_than_volume", 0
    prediction = prediction.to(torch.float64)[None, None]
    target = target.to(torch.float64)[None, None]
    mean_prediction = F.avg_pool3d(prediction, window, stride=1, padding=0)
    mean_target = F.avg_pool3d(target, window, stride=1, padding=0)
    variance_prediction = (
        F.avg_pool3d(prediction.square(), window, stride=1, padding=0)
        - mean_prediction.square()
    ).clamp_min(0)
    variance_target = (
        F.avg_pool3d(target.square(), window, stride=1, padding=0)
        - mean_target.square()
    ).clamp_min(0)
    covariance = (
        F.avg_pool3d(prediction * target, window, stride=1, padding=0)
        - mean_prediction * mean_target
    )
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    scores = (
        (2 * mean_prediction * mean_target + c1) * (2 * covariance + c2)
        / (
            (mean_prediction.square() + mean_target.square() + c1)
            * (variance_prediction + variance_target + c2)
        )
    )[0, 0]
    offset = window // 2
    centres = mask[tuple(slice(offset, size - offset) for size in mask.shape)]
    count = int(centres.sum().item())
    if count == 0:
        return None, "ssim_mask_has_no_valid_window_centres", 0
    return float(scores[centres].mean().item()), None, count


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["contiguous", "transposed", "strided"])
@pytest.mark.parametrize("window,shape", [(1, (1, 3, 5)), (3, (7, 9, 11)), (11, (13, 14, 15))])
def test_ssim_matches_independent_dense_3d_reference(dtype, layout, window, shape, monkeypatch):
    generator = torch.Generator().manual_seed(20260912)
    storage_shape = (*shape[:2], shape[2] * 2) if layout == "strided" else shape
    prediction = torch.randn(storage_shape, generator=generator, dtype=dtype) * 0.3
    target = torch.rand(storage_shape, generator=generator, dtype=dtype)
    mask = torch.rand(storage_shape, generator=generator) > 0.35
    if layout == "transposed":
        prediction, target, mask = (value.transpose(1, 2) for value in (prediction, target, mask))
    elif layout == "strided":
        prediction, target, mask = (value[..., ::2] for value in (prediction, target, mask))
    if layout != "contiguous":
        assert not prediction.is_contiguous()
    actual = dense_metrics(prediction, target, mask, ssim_window=window)
    # Swap only SSIM for the independent reference: every other output field
    # must remain unchanged, including original-dtype error/MAE/PSNR/Charbonnier.
    with monkeypatch.context() as patch:
        patch.setattr(metrics_module, "_global_ssim", _dense_box_ssim_reference)
        reference = dense_metrics(prediction, target, mask, ssim_window=window)
    assert actual.pop("ssim") == pytest.approx(reference.pop("ssim"), rel=0, abs=5e-11)
    actual.pop("ssim_max_roundoff_contrast_ratio")
    reference.pop("ssim_max_roundoff_contrast_ratio")
    assert actual == reference


@pytest.mark.parametrize(
    "offset,variation,data_range",
    [
        (0.0, 0.0, 1.0),
        (0.1, 0.0, 1.0),
        (0.7, 0.0, 1.0),
        (-0.3, 0.1, 1.0),
        (0.8, 1e-8, 1.0),
        (1e6, 1e-3, 1.0),
        (-1e6, 1e-3, 1.0),
        (1e6, 1e-3, 1e6),
        (0.0, 1e6, 1.0),
        (0.0, 1e6, 1e6),
    ],
)
def test_ssim_cancellation_and_dynamic_range_reference(offset, variation, data_range):
    generator = torch.Generator().manual_seed(42)
    signal = torch.randn((13, 14, 15), generator=generator, dtype=torch.float64)
    noise = torch.randn(signal.shape, generator=generator, dtype=torch.float64)
    prediction = offset + variation * signal
    target = offset + variation * (0.9 * signal + 0.1 * noise)
    mask = torch.ones_like(signal, dtype=torch.bool)
    actual = dense_metrics(prediction, target, mask, data_range=data_range)
    if abs(offset) == 1e6 and data_range == 1.0:
        assert actual["ssim"] is None
        assert actual["ssim_unavailable_reason"] == "ssim_numerical_cancellation"
        assert actual["ssim_valid_window_count"] == 0
        return
    expected, reason, count = _dense_box_ssim_reference(
        prediction, target, mask=mask, data_range=data_range, window=11
    )
    assert actual["ssim"] == pytest.approx(expected, rel=0, abs=5e-11)
    assert actual["ssim_unavailable_reason"] == reason
    assert actual["ssim_valid_window_count"] == count == 60


@pytest.mark.parametrize("offset,variation,sensitive", [(0.8, 1e-8, False), (1e6, 1e-3, True)])
def test_ssim_rejects_sensitive_moments_without_unstable_cubic_fallback(offset, variation, sensitive, monkeypatch):
    generator = torch.Generator().manual_seed(42)
    prediction = offset + variation * torch.randn((13, 14, 15), generator=generator, dtype=torch.float64)
    target = offset + variation * torch.randn(prediction.shape, generator=generator, dtype=torch.float64)
    calls = []
    original = metrics_module._ssim_box_mean

    def traced_box_mean(value, window):
        calls.append(window)
        assert value.dtype == torch.float64
        return original(value, window)

    monkeypatch.setattr(metrics_module, "_ssim_box_mean", traced_box_mean)
    result = dense_metrics(prediction, target)
    assert calls == [11] * 5
    assert (result["ssim"] is None) is sensitive
    assert result["ssim_reduction"] == "fp64_separable_axis_box_v1"
    assert result["ssim_numerical_policy"] == "raw_moment_cancellation_unavailable_v1"


def test_extreme_dc_ssim_is_unavailable_in_paired_metrics_without_affecting_point_losses():
    generator = torch.Generator().manual_seed(52)
    target = 1e8 + 1e-4 * torch.rand((20, 20, 20), generator=generator, dtype=torch.float64)
    prediction = target + 1e-4 * torch.rand(target.shape, generator=generator, dtype=torch.float64)
    result = paired_subject_metrics(prediction, prediction, target)
    assert result["before"]["ssim"] is None
    assert result["after"]["ssim"] is None
    assert result["improvement"]["ssim"] is None
    assert result["improvement"]["masked_charbonnier"] == 0.0
    assert result["before"]["ssim_unavailable_reason"] == "ssim_numerical_cancellation"


@pytest.mark.parametrize("offset", [1.69, 2.0, 2.5, 10.0])
def test_benign_out_of_range_constant_ssim_remains_available(offset):
    value = torch.full((13, 14, 15), offset, dtype=torch.float64)
    result = dense_metrics(value, value)
    assert result["ssim"] == pytest.approx(1.0, abs=5e-9)
    assert result["ssim_max_roundoff_contrast_ratio"] < result["ssim_cancellation_ratio_limit"]


def test_ssim_conditioning_guard_uses_only_scored_centres_without_dropping_windows():
    value = torch.ones((13, 14, 40), dtype=torch.float64)
    value[:, :, 25:] = 1e8
    mask = torch.zeros_like(value, dtype=torch.bool)
    mask[6, 7, 6] = True
    result = dense_metrics(value, value, mask)
    assert result["ssim"] == pytest.approx(1.0)
    assert result["ssim_valid_window_count"] == 1
    mask[6, 7, 32] = True
    result = dense_metrics(value, value, mask)
    assert result["ssim"] is None
    assert result["ssim_unavailable_reason"] == "ssim_numerical_cancellation"


def test_ssim_centre_mask_keeps_unobserved_neighbours_in_3d_statistics():
    target = torch.zeros((3, 3, 3), dtype=torch.float64)
    prediction = torch.ones_like(target)
    prediction[1, 1, 1] = 0
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[1, 1, 1] = True
    result = dense_metrics(prediction, target, mask, ssim_window=3)
    expected, _, count = _dense_box_ssim_reference(prediction, target, mask=mask, data_range=1, window=3)
    assert result["mae"] == 0.0
    assert result["ssim"] == pytest.approx(expected, rel=0, abs=5e-11)
    assert result["ssim"] < 0.01
    assert result["ssim_valid_window_count"] == count == 1


def test_ssim_nonempty_mask_with_no_valid_centres():
    value = torch.ones((5, 5, 5), dtype=torch.float64)
    mask = torch.zeros_like(value, dtype=torch.bool)
    mask[0, 0, 0] = True
    result = dense_metrics(value, value, mask, ssim_window=3)
    assert result["ssim"] is None
    assert result["ssim_unavailable_reason"] == "ssim_mask_has_no_valid_window_centres"
    assert result["ssim_valid_window_count"] == 0


@pytest.mark.parametrize("window", [0, -1, 2, True, 3.0])
def test_ssim_invalid_windows_still_fail_closed(window):
    value = torch.zeros((3, 3, 3), dtype=torch.float64)
    with pytest.raises(ValueError, match="positive odd integer"):
        dense_metrics(value, value, ssim_window=window)


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
