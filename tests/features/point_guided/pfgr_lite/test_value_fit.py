from __future__ import annotations

from pathlib import Path

import pytest
import torch

import smagm.data as smagm_data
from smagm.data import brats21_point_guided as point_guided_data
from smagm.features.point_guided import decoder as decoder_module
from smagm.features.point_guided import updater as updater_module
from smagm.features.point_guided.pfgr_lite import teacher as teacher_module

from smagm.features.point_guided.pfgr_lite.value_bank import ValueBankReader, ValueBankRow, build_value_bank
from smagm.features.point_guided.pfgr_lite.value_net import (
    SignedValueNet,
    ValueFitOptions,
    evaluate_value,
    fit_value,
    resume_value_fit,
)
from smagm.features.point_guided.pfgr_lite.provenance import ProducerCompatibility, SourceProvenance
from smagm.features.point_guided.pfgr_lite.types import ProducerDependencies


def _producer() -> ProducerDependencies:
    compatibility = ProducerCompatibility(
        observation_normalization_hash="norm",
        geometry_query_version_hash="geometry",
        medicalnet_provenance_hash="medicalnet",
        frozen_bn_hash="bn",
        static_head_hash="static",
        semantic_head_hash="semantic",
        point_refiner_hash="points",
        spectral_projector_hash="spectral",
        state_initializer_hash="state",
        updater_hash="updater",
        decoder_hash="decoder",
        writer_hash="writer",
        candidate_geometry_hash="candidate",
        label_definition_hash="label",
    )
    source = SourceProvenance(
        synthetic_untrained=False,
        official_pretrained_verified=True,
        parameter_hash="parameters",
        frozen_bn_hash="bn",
        traversal_count=1,
    )
    return ProducerDependencies(compatibility=compatibility, source_provenance=source)


def _row(index: int, gain: float, *, split_role: str = "producer_fit", subject: str | None = None) -> ValueBankRow:
    return ValueBankRow(
        state96=torch.full((96,), float(index)),
        f_spec168=torch.full((168,), float(index + 1)),
        semantic3=torch.tensor([0.1, 0.2, 0.7]),
        reliability3=torch.tensor([0.2, 0.3, 0.5]),
        q_bar24=torch.linspace(0.0, 1.0, 24),
        delta96=torch.full((96,), float(index) / 100.0),
        raw_gain=gain,
        benefit=max(gain, 0.0),
        harm=max(-gain, 0.0),
        action_id=f"a-{index}",
        context_id=f"ctx-{index // 2}",
        point_id=index,
        subject_id=subject or f"subject-{index // 2}",
        split_role=split_role,
        support_provenance="complete_support_v1",
        inclusion_mechanism="complete_support_v1",
    )


def _bank(tmp_path: Path, *, count: int = 8) -> ValueBankReader:
    rows = [_row(i, float((-1) ** i * (i + 1))) for i in range(count)]
    path = tmp_path / "bank"
    build_value_bank(rows, path, producer=_producer(), split_role_hash="split", role_membership={row.subject_key: row.split_role for row in rows}, engineering_only=True)
    return ValueBankReader(path)


def test_signed_mlp_architecture_variants_and_output_shape() -> None:
    for variant in (126, 222, 270, 366):
        model = SignedValueNet(variant)
        output = model(torch.zeros(2, variant))
        assert output.shape == (2, 1)
        assert model.net[0].in_features == variant
        assert model.net[0].out_features == 128
        assert model.net[2].out_features == 64
        assert model.net[4].out_features == 1
    with pytest.raises(ValueError, match="hidden"):
        SignedValueNet(366, hidden_channels=(64, 32))


def test_fit_uses_fixed_training_scale_and_preserves_raw_sign(tmp_path: Path) -> None:
    reader = _bank(tmp_path)
    torch.manual_seed(10)
    model = SignedValueNet(366)
    result = fit_value(reader, model, epochs=2, batch_size=3, seed=9)
    assert result.complete
    assert result.gain_scale.digest == reader.gain_scale.digest
    assert result.metrics["raw_gain_signed"]
    assert result.metrics["constant_training_mean_mse_raw"] >= 0.0
    assert result.metrics["teacher_calls"] == 0
    assert result.metrics["target_volume_reads"] == 0
    assert result.metrics["updater_calls"] == 0
    assert result.metrics["decoder_calls"] == 0
    assert result.metrics["hypothetical_writes"] == 0
    assert result.metrics["train_row_count"] == 8
    assert result.metrics["train_batch_count"] == 6
    assert result.metrics["changed_parameter_count"] > 0
    assert result.metrics["v_gradient_l2_norm_count"] == result.metrics["train_batch_count"]
    assert result.metrics["v_gradient_l2_norm_max"] > 0.0
    evaluated = evaluate_value(reader, result)
    assert evaluated.metrics["mse_raw"] is not None
    assert evaluated.metrics["sign_accuracy"] is not None
    assert evaluated.metrics["top1_subset_regret_scope"] == "same_subject_context_state_rows"


def test_same_bank_controls_fit_all_variants(tmp_path: Path) -> None:
    reader = _bank(tmp_path)
    for variant in (126, 222, 270, 366):
        torch.manual_seed(2)
        model = SignedValueNet(variant)
        result = fit_value(reader, model, input_variant=variant, epochs=1, batch_size=4, seed=1)
        assert result.identity.input_variant == variant
        assert result.metrics["train_row_count"] == reader.manifest().row_count
        assert evaluate_value(reader, result, input_variant=variant).row_count == 8


def test_optimizer_ownership_rejects_foreign_parameter(tmp_path: Path) -> None:
    reader = _bank(tmp_path)
    model = SignedValueNet(366)
    foreign = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam(list(model.parameters()) + [foreign])
    with pytest.raises(ValueError, match="ownership"):
        fit_value(reader, model, optimizer=optimizer, epochs=1)


def test_resume_is_bitwise_deterministic_and_dependency_bound(tmp_path: Path) -> None:
    reader = _bank(tmp_path)
    torch.manual_seed(42)
    full_model = SignedValueNet(366)
    full = fit_value(reader, full_model, epochs=3, batch_size=2, seed=7)
    torch.manual_seed(42)
    resumed_model = SignedValueNet(366)
    partial = fit_value(reader, resumed_model, epochs=3, batch_size=2, seed=7, max_updates=3)
    assert not partial.complete
    resumed = resume_value_fit(reader, resumed_model, partial, epochs=3, batch_size=2, seed=7)
    assert resumed.complete
    assert full.identity.weights_hash == resumed.identity.weights_hash
    assert full.history == resumed.history
    assert all("loss_scaled_sum" in item and "loss_scaled_count" in item for item in resumed.history)
    assert full.metrics["v_gradient_l2_norm_sum"] == pytest.approx(resumed.metrics["v_gradient_l2_norm_sum"])
    assert full.metrics["v_gradient_l2_norm_count"] == resumed.metrics["v_gradient_l2_norm_count"]
    for name, value in full.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[name])
    changed = dict(partial.resume_state)
    changed["bank_manifest_hash"] = "different"
    with pytest.raises(ValueError, match="bank_manifest_hash"):
        fit_value(reader, resumed_model, resume=changed, epochs=3, batch_size=2, seed=7)


def test_resume_cursor_and_optimizer_state_are_exported(tmp_path: Path) -> None:
    reader = _bank(tmp_path)
    result = fit_value(reader, epochs=2, batch_size=3, seed=4, max_updates=1)
    state = result.resume_state
    assert state["schema_version"] == "point-guided-pfgr-lite-resume-v1"
    assert state["protocol"] == "point-guided-pfgr-lite-resume-v1"
    assert isinstance(state["stage_payload"]["cursor"], int)
    assert state["optimizer_state"]
    assert isinstance(state["rng_state"]["torch_cpu"], torch.Tensor)
    assert state["bank_manifest_hash"] == reader.manifest_hash


def test_loss_and_bounds_are_explicit() -> None:
    with pytest.raises(ValueError, match="epochs"):
        ValueFitOptions(epochs=0)
    with pytest.raises(ValueError, match="smooth_l1"):
        ValueFitOptions(loss="smooth_l1")
    assert ValueFitOptions(loss="smooth_l1", robust_ablation=True).loss == "smooth_l1"


def test_no_training_role_does_not_claim_fit(tmp_path: Path) -> None:
    row = _row(0, 1.0, split_role="validation")
    path = tmp_path / "validation-only"
    build_value_bank([row], path, producer=_producer(), split_role_hash="s", role_membership={row.subject_key: row.split_role}, engineering_only=True)
    reader = ValueBankReader(path)
    assert reader.verify()["status"]["evidence_status"] in {"BLOCKED_MISSING_TRAINING", "ENGINEERING_ONLY"}
    with pytest.raises(ValueError, match="training role"):
        fit_value(reader, epochs=1)


def test_diagnostic_only_bank_is_not_a_main_fit_input(tmp_path: Path) -> None:
    row = _row(0, 1.0)
    path = tmp_path / "diagnostic"
    build_value_bank(
        [row],
        path,
        producer=_producer(),
        split_role_hash="split",
        role_membership={row.subject_key: row.split_role},
        engineering_only=True,
        diagnostic=True,
    )
    reader = ValueBankReader(path)
    with pytest.raises(ValueError, match="diagnostic-only"):
        fit_value(reader, epochs=1)


def test_metric_ties_are_explicit(tmp_path: Path) -> None:
    reader = _bank(tmp_path, count=2)
    model = SignedValueNet(366)
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)
    result = evaluate_value(reader, model)
    assert result.metrics["sign_tie_count"] == 2
    assert result.metrics["top1_subset_tie_groups"] == 1
    assert result.metrics["top1_subset_regret_scope"] == "same_subject_context_state_rows"


def test_optimizer_hyperparameters_are_bound_to_fit_options(tmp_path: Path) -> None:
    reader = _bank(tmp_path)
    model = SignedValueNet(366)
    optimizer = torch.optim.Adam(
        [{"params": list(model.parameters()), "name": "value_net"}],
        lr=2e-3,
    )
    with pytest.raises(ValueError, match="hyperparameters"):
        fit_value(reader, model, optimizer=optimizer, learning_rate=1e-3)


def test_stop_aware_regret_and_partition_sums_are_reported(tmp_path: Path) -> None:
    reader = _bank(tmp_path, count=2)
    model = SignedValueNet(366)
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)
    result = evaluate_value(reader, model)
    metrics = result.metrics
    assert metrics["top1_subset_stop_count"] == 1
    assert metrics["top1_subset_stop_rule"] == "stop_when_best_predicted_raw_gain_leq_zero"
    assert metrics["mse_scaled_count"] == 2
    assert metrics["mse_raw_count"] == 2
    assert metrics["mse_scaled_sum"] >= 0.0
    assert metrics["mse_raw_sum"] >= 0.0
    assert metrics["dependency_counter_scope"] == "cached_value_eval_only"
    assert metrics["dependency_counters_scope_verified"] is True


def test_cached_fit_and_eval_do_not_call_preimported_mri_teacher_u_or_decoder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    reader = _bank(tmp_path)
    calls = {"loader": 0, "teacher": 0, "updater": 0, "decoder": 0}

    def forbidden(name: str):
        def _raise(*args: object, **kwargs: object) -> object:
            calls[name] += 1
            raise AssertionError(f"forbidden cached dependency called: {name}")

        return _raise

    monkeypatch.setattr(smagm_data, "load_point_guided_subject", forbidden("loader"))
    monkeypatch.setattr(point_guided_data, "load_point_guided_subject", forbidden("loader"))
    monkeypatch.setattr(updater_module.UpdateNet, "forward", forbidden("updater"))
    monkeypatch.setattr(decoder_module.ImplicitTriPlaneDecoder, "forward", forbidden("decoder"))
    monkeypatch.setattr(decoder_module.ImplicitTriPlaneDecoder, "decode_points", forbidden("decoder"))
    monkeypatch.setattr(teacher_module, "measure_actions", forbidden("teacher"))
    result = fit_value(reader, epochs=1, batch_size=4, seed=3)
    evaluated = evaluate_value(reader, result)
    assert calls == {"loader": 0, "teacher": 0, "updater": 0, "decoder": 0}
    assert result.metrics["dependency_counter_scope"] == "cached_value_fit_only"
    assert evaluated.metrics["dependency_counter_scope"] == "cached_value_eval_only"
    assert result.metrics["forbidden_imports_during_fit"] == ()
