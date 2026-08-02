"""Regression coverage for explicit, non-total compute telemetry."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from smagm.experiments.complexity import analytical_conv_linear_forward_flops, profile_supported_operator_flops
from smagm.features.encoder import EncoderConfig, EvidenceEncoder


def test_e2_encoder_cnn_forward_flops_match_declared_20_context_60_square_baseline() -> None:
    """The baseline is forward Conv2d work only, under 2 FLOPs/MAC.

    This executes the exact E2 convolutional subnetwork with the batch and
    spatial shape used by the product sparse protocol.  The analytic bank is
    intentionally outside this specific Conv2d/Linear accounting scope.
    """

    encoder = EvidenceEncoder(EncoderConfig(variant="e2", output_stride=1))
    assert encoder.micro_cnn is not None
    runtime_analytic_tensor = torch.zeros((20, 7, 60, 60), dtype=torch.float32)
    report = analytical_conv_linear_forward_flops(encoder.micro_cnn, runtime_analytic_tensor)

    assert report["convention"] == "2_flops_per_mac"
    assert report["scope"] == "forward Conv2d and Linear modules only"
    assert report["forward_flops_2flop_per_mac"] == 3_290_112_000
    assert report["by_module"]["stem"]["output_shape"] == [20, 24, 60, 60]


def test_profiler_result_is_explicitly_partial_and_can_be_disabled() -> None:
    result, report = profile_supported_operator_flops(lambda: torch.tensor(3.0), enabled=False, scope="unit test operation")
    assert float(result) == 3.0
    assert report["profiled_supported_operator_flops"] is None
    assert report["profiler_operator_coverage"] == "partial_unknown_torch_profiler_supported_operators_only"
    assert "training_step_flops" not in report


def test_legal_shape_analytical_encoder_telemetry_is_cached_per_cohort_owner(monkeypatch) -> None:
    """The production controller must not add a full encoder pass per patient."""

    from smagm.cli import brats21_smoke

    metadata = SimpleNamespace(plane=SimpleNamespace(shape_hw=(60, 60)))
    bundle = SimpleNamespace(
        assignment=SimpleNamespace(context_ids=("context-a", "context-b")),
        manifest=SimpleNamespace(metadata=lambda _observation_id: metadata),
    )
    owner = SimpleNamespace(encoder_flop_telemetry=None)
    calls: list[int] = []

    def fake_counter(_encoder, _bundle) -> dict[str, object]:
        calls.append(1)
        return {
            "encoder_forward_flops_2flop_per_mac": 329_011_200,
            "encoder_forward_flops_input_shapes": [[2, 7, 60, 60]],
        }

    monkeypatch.setattr(brats21_smoke, "_encoder_conv_telemetry", fake_counter)
    config = {"diagnostics": {"analytical_encoder_flops": True}}
    first = brats21_smoke._resolved_encoder_conv_telemetry(object(), bundle, config, cache_owner=owner)
    second = brats21_smoke._resolved_encoder_conv_telemetry(object(), bundle, config, cache_owner=owner)

    assert calls == [1]
    assert first["encoder_forward_flops_measurement"] == "computed_from_legal_context_shape_batch"
    assert second["encoder_forward_flops_measurement"] == "process_cache_from_legal_context_shape_batch"
