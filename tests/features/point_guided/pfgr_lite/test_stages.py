from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from smagm.features.point_guided.contracts import VolumeGeometry
from smagm.features.point_guided.pfgr_lite.config import EffectTeacherConfig, PFGRLiteConfig
from smagm.features.point_guided.pfgr_lite.data import DataAccessCounters, TargetFreeSample
from smagm.features.point_guided.pfgr_lite.objectives import updater_objective
from smagm.features.point_guided.pfgr_lite.footprint import PFGRQueryLattice
from smagm.features.point_guided.pfgr_lite.model import PFGRLiteModel
from smagm.features.point_guided.pfgr_lite.provenance import module_state_digest
from smagm.features.point_guided.pfgr_lite.sparse_write import make_action_writer, make_point_query, make_support_legal_mask
from smagm.features.point_guided.pfgr_lite.stages import (
    StageExecutionConfig,
    StageInputs,
    StageOptions,
    _stage_provenance,
    _s2_teacher_config,
    generate_value_bank,
    run_stage,
)
from smagm.features.point_guided.pfgr_lite.stages import _select_candidates


def _sample() -> TargetFreeSample:
    geometry = VolumeGeometry.from_spacing((2, 2, 2))
    return TargetFreeSample("subject-01", torch.zeros(3, 2, 2, 2), torch.ones(1, 2, 2, 2, dtype=torch.bool), geometry, {}, "", "")


def _target_context() -> SimpleNamespace:
    return SimpleNamespace(target=torch.zeros(1, 1, 2, 2, 2), observation_mask=torch.ones(1, 1, 2, 2, 2, dtype=torch.bool))


def test_stage_options_and_execution_config_are_strict() -> None:
    with pytest.raises(ValueError, match="unknown StageOptions"):
        StageOptions.from_dict({"bogus": 1})
    options = StageOptions.from_dict({"stage": "S1", "max_updates": 2, "engineering_only": True})
    config = PFGRLiteConfig(num_points=4, engineering_only=True)
    execution = StageExecutionConfig(config=config, frontend_sidecar={}, normalization={}, stage_options=options)
    assert execution.stage_options.stage == "S1"
    with pytest.raises(ValueError, match="unknown frontend sidecar"):
        StageExecutionConfig(config=config, frontend_sidecar={"unexpected": True}, normalization={}, stage_options=options)
    with pytest.raises(ValueError, match="unknown StageExecutionConfig"):
        StageExecutionConfig.from_dict({"pfgr_config": config.as_dict(), "unknown": 1})


def test_s2_teacher_envelope_records_exact_q0_or_configured_iid_q() -> None:
    config = PFGRLiteConfig(
        num_points=4,
        engineering_only=True,
        teacher=EffectTeacherConfig(mode="iid_fixed_q", q_draws=7),
    )
    exact = _s2_teacher_config(config, StageOptions(stage="S2", query_mode="exact_dense", engineering_only=True))
    sampled = _s2_teacher_config(config, StageOptions(stage="S2", query_mode="iid_fixed_q", engineering_only=True))
    assert exact.mode == "exact_footprint" and exact.q_draws == 0
    assert sampled.mode == "iid_fixed_q" and sampled.q_draws == 7


def test_s2_teacher_q_override_is_explicit_sidecar_without_config_mutation() -> None:
    config = PFGRLiteConfig(
        num_points=4,
        engineering_only=True,
        teacher=EffectTeacherConfig(mode="iid_fixed_q", q_draws=7),
    )
    options = StageOptions(stage="S2", query_mode="iid_fixed_q", teacher_q_draws=5, engineering_only=True)
    sampled = _s2_teacher_config(config, options)
    assert sampled.mode == "iid_fixed_q" and sampled.q_draws == 5
    assert config.teacher.q_draws == 7
    exact = _s2_teacher_config(config, StageOptions(stage="S2", query_mode="exact_dense", teacher_q_draws=0, engineering_only=True))
    assert exact.mode == "exact_footprint" and exact.q_draws == 0
    with pytest.raises(ValueError, match="exact_dense teacher_q_draws"):
        StageOptions(stage="S2", query_mode="exact_dense", teacher_q_draws=2, engineering_only=True)
    with pytest.raises(ValueError, match="at least 2"):
        StageOptions(stage="S2", query_mode="iid_fixed_q", teacher_q_draws=1, engineering_only=True)


def test_production_stage_provenance_rejects_checkpoint_sentinel_early() -> None:
    options = StageOptions(stage="S1", arm="u_plus_spectral")
    with pytest.raises(ValueError, match="actual checkpoint_id"):
        _stage_provenance(
            StageInputs(),
            options,
            gradient_norm=1.0,
            nonzero_steps=1,
            measured_steps=1,
            optimizer_steps=1,
            changed=1,
            before="before",
            after="after",
            completed=True,
            producer_hash="producer",
        )


def test_updater_objective_uses_exact_intermediate_final_weights_and_gradients() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    predictions = [parameter * torch.ones(1, 1, 2, 2, 2), 2.0 * parameter * torch.ones(1, 1, 2, 2, 2)]
    objective = updater_objective(SimpleNamespace(predictions=predictions), _target_context(), config=SimpleNamespace(delta_weight=0.0))
    assert float(objective.detach()) == pytest.approx(1.5)
    objective.backward()
    assert parameter.grad is not None and parameter.grad.item() > 0.0


@pytest.mark.parametrize("k", (1, 2, 4))
def test_delta_regularizer_uses_sum_over_96_channels_once(k: int) -> None:
    target = _target_context()
    predictions = [torch.zeros(1, 1, 2, 2, 2) for _ in range(k)]
    deltas = [torch.full((1, 96), 0.1) for _ in range(k)]
    objective = updater_objective(SimpleNamespace(predictions=predictions, deltas=deltas), target, config=SimpleNamespace(delta_weight=1.0))
    assert float(objective) == pytest.approx(1.001, abs=1e-6)


def test_s1_runs_random_k_with_target_join_after_route(tmp_path) -> None:
    class Updater(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))

    model = SimpleNamespace(updater=Updater())
    order: list[str] = []

    def route(sample, *, k, seed, **_kwargs):
        del sample, seed
        order.append("route")
        return SimpleNamespace(predictions=[model.updater.weight * torch.ones(1, 1, 2, 2, 2) for _ in range(k)], target_context=_target_context())

    options = StageOptions(stage="S1", epochs=3, max_updates=3, engineering_only=True)
    inputs = StageInputs(samples=(_sample(),), model=model, route_builder=route, stage_options=options)
    state = run_stage("S1", PFGRLiteConfig(num_points=4, engineering_only=True), inputs, tmp_path / "s1")
    assert state.stage == "S1" and state.update == 3
    assert len(order) == 3
    receipt = (tmp_path / "s1" / "stage_receipt.json").read_text()
    assert "sampled_k_counts" in receipt and "stage_provenance" in receipt
    with pytest.raises(FileExistsError):
        run_stage("S1", PFGRLiteConfig(num_points=4, engineering_only=True), inputs, tmp_path / "s1")


def test_s1_records_paired_same_route_z0_and_final_metrics(tmp_path) -> None:
    class Updater(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))

    model = SimpleNamespace(updater=Updater())

    def route(_sample, *, k, **_kwargs):
        final = [model.updater.weight * torch.ones(1, 1, 2, 2, 2) for _ in range(k)]
        return SimpleNamespace(
            initial_prediction=torch.zeros(1, 1, 2, 2, 2),
            predictions=final,
            target_context=_target_context(),
            context=SimpleNamespace(context_id="context-01"),
        )

    result = run_stage(
        "S1",
        PFGRLiteConfig(
            num_points=4,
            engineering_only=True,
            teacher=EffectTeacherConfig(epsilon=0.007),
        ),
        StageInputs(
            samples=(_sample(),),
            model=model,
            route_builder=route,
            stage_options=StageOptions(stage="S1", epochs=1, max_updates=1, engineering_only=True),
        ),
        tmp_path / "paired",
    )
    metrics = result.receipt.metrics
    assert metrics["paired_dense_metrics_measured_count"] == 1
    assert not metrics["paired_dense_metrics_unmeasured"]
    row = metrics["paired_dense_metrics"][0]
    assert row["before"]["mask_count"] == 8
    assert row["before"]["masked_charbonnier"] == pytest.approx(0.007)
    assert row["improvement"]["masked_charbonnier"] is not None
    assert row["after"]["ssim"] is None
    assert row["after"]["ssim_unavailable_reason"] == "ssim_window_larger_than_volume"
    assert metrics["paired_dense_metrics_aggregate"]["count"]["improvement"]["ssim"] == 0


def test_stage_runtime_schema_is_strict_and_max_updates_override_is_recorded(tmp_path) -> None:
    class Updater(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))

    model = SimpleNamespace(updater=Updater())

    def route(_sample, *, k, **_kwargs):
        return SimpleNamespace(
            predictions=[model.updater.weight * torch.ones(1, 1, 2, 2, 2) for _ in range(k)],
            target_context=_target_context(),
        )

    config = PFGRLiteConfig(num_points=4, engineering_only=True)
    first = run_stage(
        "S1",
        config,
        StageInputs(
            samples=(_sample(),),
            model=model,
            route_builder=route,
            stage_options=StageOptions(stage="S1", epochs=3, max_updates=1, engineering_only=True),
        ),
        tmp_path / "runtime-first",
    )
    runtime = dict(first.runtime_state)
    assert {
        "schema_version",
        "stage_state",
        "optimizer_state",
        "rng_state",
        "cursor",
        "parameter_names",
        "execution_config_hash",
        "training_config_hash",
        "producer_compatibility_hash",
        "split_role_hash",
        "execution_config",
    } <= set(runtime)
    assert set(runtime["cursor"]) == {"epoch", "batch_index", "update", "microstep", "sample_order", "route_rng_state"}
    assert "stage" not in runtime and "local_route_rng_state" not in runtime["cursor"]
    for value in runtime["optimizer_state"].get("state", {}).values():
        for tensor in value.values():
            if isinstance(tensor, torch.Tensor):
                assert tensor.device.type == "cpu" and not tensor.requires_grad
    resumed = run_stage(
        "S1",
        config,
        StageInputs(
            samples=(_sample(),),
            model=model,
            route_builder=route,
            resume=runtime,
            stage_options=StageOptions(stage="S1", epochs=3, max_updates=2, engineering_only=True),
        ),
        tmp_path / "runtime-resumed",
    )
    assert resumed.runtime_state["continuation"]["max_updates_override"] == {"previous": 1, "requested": 2}
    missing = dict(runtime)
    missing.pop("training_config_hash")
    with pytest.raises(ValueError, match="missing required"):
        run_stage(
            "S1",
            config,
            StageInputs(
                samples=(_sample(),),
                model=model,
                route_builder=route,
                resume=missing,
                stage_options=StageOptions(stage="S1", epochs=3, max_updates=2, engineering_only=True),
            ),
            tmp_path / "runtime-missing",
        )
    unknown = dict(runtime)
    unknown["local_route_rng_state"] = None
    with pytest.raises(ValueError, match="unknown fields"):
        run_stage(
            "S1",
            config,
            StageInputs(
                samples=(_sample(),),
                model=model,
                route_builder=route,
                resume=unknown,
                stage_options=StageOptions(stage="S1", epochs=3, max_updates=2, engineering_only=True),
            ),
            tmp_path / "runtime-unknown",
        )
    replacement = TargetFreeSample(
        "subject-01",
        torch.ones(3, 2, 2, 2),
        torch.ones(1, 2, 2, 2, dtype=torch.bool),
        _sample().geometry,
        {},
        "",
        "",
    )
    with pytest.raises(ValueError, match="input observation identity"):
        run_stage(
            "S1",
            config,
            StageInputs(
                samples=(replacement,),
                model=model,
                route_builder=route,
                resume=runtime,
                stage_options=StageOptions(stage="S1", epochs=3, max_updates=2, engineering_only=True),
            ),
            tmp_path / "runtime-input-mismatch",
        )


def test_s2_materializes_target_free_candidates_before_target_provider(tmp_path) -> None:
    sample = _sample()
    order: list[str] = []

    def route(_sample, **_kwargs):
        order.append("route")
        return SimpleNamespace(states=("state",), context=None)

    def proposals(*_args, **_kwargs):
        order.append("proposals")
        return [{"action_id": f"a-{index}", "stratum": "uniform"} for index in range(4)]

    def target_provider(_trace):
        order.append("target")
        return _target_context()

    def measure(_trace, selected, _target, **_kwargs):
        order.append("measure")
        return [{"action_id": row["action_id"], "raw_gain": 0.1, "subject_key": sample.subject_id, "split_role": "producer_fit", "label_definition": "signed-conditional-mean-masked-global-charbonnier-v1", "measurement_mode": "exact_footprint", "role": "exact_footprint", "support_provenance": "complete_support_v1", "inclusion_mechanism": "complete_support_v1", "sampler_law": "complete_support_v1", "engineering_only": True, "diagnostic": True} for row in selected]

    def writer(rows, output_dir, **_kwargs):
        order.append("writer")
        output_dir.mkdir(parents=True, exist_ok=True)
        return {"rows": len(rows)}

    inputs = StageInputs(samples=(sample,), route_builder=route, proposal_builder=proposals, target_provider=target_provider, effect_measure=measure, bank_writer=writer, stage_options=StageOptions(stage="S2", candidate_count=4, engineering_only=True))
    state = run_stage("S2", PFGRLiteConfig(num_points=4, engineering_only=True), inputs, tmp_path / "s2")
    assert state.stage == "S2"
    assert order.index("target") > order.index("proposals")
    assert order.index("measure") > order.index("target")
    replay_refs = state.receipt.metrics["selected_replay_refs"]
    assert len(replay_refs) == 1
    replay = (tmp_path / "s2" / replay_refs[0]).read_text(encoding="utf-8")
    assert '"snapshot_kind":"metadata_only"' in replay
    assert '"tensor_payload":"omitted"' in replay
    assert '"raw_target_payload":"omitted"' in replay


def test_s2_freezes_and_restores_producer_training_mode(tmp_path) -> None:
    class Producer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.bn = torch.nn.BatchNorm1d(1)

    model = Producer()
    model.train()
    seen_training_modes: list[bool] = []

    def route(_sample, **_kwargs):
        seen_training_modes.append(model.training)
        return SimpleNamespace(states=("state",), context=None)

    def proposals(*_args, **_kwargs):
        return [{"action_id": f"a-{index}", "stratum": "uniform"} for index in range(4)]

    def provider(_trace):
        return _target_context()

    def measure(_trace, selected, _target, **_kwargs):
        return [{"action_id": item["action_id"], "raw_gain": 0.1} for item in selected]

    def writer(rows, output_dir, **_kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        return {"row_count": len(rows)}

    result = run_stage(
        "S2",
        PFGRLiteConfig(num_points=4, engineering_only=True),
        StageInputs(
            samples=(_sample(),),
            model=model,
            route_builder=route,
            proposal_builder=proposals,
            target_provider=provider,
            effect_measure=measure,
            bank_writer=writer,
            stage_options=StageOptions(stage="S2", candidate_count=4, engineering_only=True),
        ),
        tmp_path / "s2-freeze",
    )
    assert seen_training_modes == [False]
    assert model.training and model.bn.training
    assert model.weight.requires_grad
    assert result.receipt.metrics["producer_eval_mode"] is True
    assert result.receipt.metrics["frozen_model_unchanged"] is True


def test_s2_sampling_deduplicates_strata_and_uses_observation_diversity() -> None:
    candidates: list[dict[str, object]] = []
    for index in range(64):
        base = {
            "action_id": f"a-{index:03d}",
            "point_ras_mm": (float(index), float(index % 7), float(index % 5)),
            "semantic": (float(index % 3), float(index % 11), float(index % 13)),
            "score": float(index),
        }
        candidates.extend(
            [
                {**base, "stratum": "uniform"},
                {**base, "stratum": "frozen_v_high_score", "score": float(1000 - index)},
                {**base, "stratum": "predicted_semantic_spatial"},
            ]
        )
    selected = _select_candidates(candidates, count=32, seed=1729)
    assert len(selected) == 32
    assert len({str(item["action_id"]) for item in selected}) == 32
    strata = [str(item["stratum"]) for item in selected]
    assert strata.count("uniform") == 16
    assert strata.count("frozen_v_high_score") == 8
    assert strata.count("predicted_semantic_spatial") == 8


def test_generate_value_bank_calls_target_provider_only_after_traces(tmp_path) -> None:
    order: list[str] = []
    traces = (SimpleNamespace(rows=({"action_id": "a", "raw_gain": 0.1},)),)

    def provider(_trace):
        order.append("target")
        return _target_context()

    def measure(_trace, _target, **_kwargs):
        order.append("measure")
        return ({"action_id": "a", "raw_gain": 0.1},)

    def writer(rows, output_dir, **_kwargs):
        order.append("writer")
        output_dir.mkdir()
        return {"row_count": len(rows)}

    result = generate_value_bank(traces, object(), provider, PFGRLiteConfig(num_points=4, engineering_only=True), tmp_path / "bank", writer=writer, effect_measure=measure, engineering_only=True)
    assert result["row_count"] == 1
    assert order == ["target", "measure", "writer"]


def test_s1_live_model_updates_u_and_spectral_only(tmp_path) -> None:
    """A tiny real PFGR route keeps D/backbone frozen and carries U gradients."""

    torch.manual_seed(63190)
    config = PFGRLiteConfig(
        num_points=4,
        engineering_only=True,
        build_chunk_size=64,
        decode_chunk_size=64,
    )

    class LatticeFactory:
        def build(self, **kwargs):
            return PFGRQueryLattice.build(**kwargs)

    model = PFGRLiteModel(config, query_lattice_factory=LatticeFactory())
    geometry = VolumeGeometry.from_spacing((9, 9, 9))
    sample = TargetFreeSample(
        "synthetic-live-s1",
        torch.randn(3, 9, 9, 9),
        torch.ones(1, 9, 9, 9, dtype=torch.bool),
        geometry,
        {},
        "",
        "",
    )
    events: list[str] = []

    def writer(state, context, action):
        events.append("write")
        lattice = PFGRQueryLattice.build(
            context.geometry,
            context.feature_geometry,
            query_dtype=state.planes.xy.dtype,
            build_chunk_size=64,
        )
        return make_action_writer(lattice)(state, context, action)

    def target_provider(subject_id: str):
        events.append("target")
        assert subject_id == sample.subject_id
        return torch.zeros(1, 9, 9, 9)

    frozen_before = {
        "backbone": module_state_digest(model.frontend.semantic_prior.backbone),
        "decoder": module_state_digest(model.decoder),
        "static": module_state_digest(model.static_head),
    }
    spectral_before = module_state_digest(model.frontend.spectral_anchor_builder)
    counters = SimpleNamespace(target_reads=0, observation_reads=1)
    inputs = StageInputs(
        samples=(sample,),
        model=model,
        query=make_point_query(),
        writer=writer,
        target_provider=target_provider,
        stage_options=StageOptions(stage="S1", epochs=1, max_updates=1, engineering_only=True),
        metadata={"counters": counters},
    )
    result = run_stage("S1", config, inputs, tmp_path / "live-s1")
    metrics = result.receipt.metrics
    assert result.receipt.gradient_steps == 1
    assert metrics["spectral_gradient_l2_max"] > 0.0
    assert metrics["spectral_nonzero_steps"] == 1
    assert metrics["changed_parameter_count"] > 0
    assert metrics["changed_parameter_count_total"] >= metrics["changed_parameter_count"]
    assert metrics["gradient_evidence"]["updater"]["measured_steps"] == 1
    assert metrics["gradient_evidence"]["updater"]["l2_norm_max"] > 0.0
    assert metrics["history_record_count"] == 1
    assert (tmp_path / "live-s1" / "stage_history.jsonl").is_file()
    assert module_state_digest(model.frontend.spectral_anchor_builder) != spectral_before
    assert all(module_state_digest(getattr(model, name) if name == "decoder" else model.static_head if name == "static" else model.frontend.semantic_prior.backbone) == digest for name, digest in frozen_before.items())
    assert counters.target_reads == 1
    assert events and events[-1] == "target"


def test_s1_zero_initialized_u_uses_writer_support_and_receives_gradient(tmp_path) -> None:
    """A zero U may write only at canonical compact-writer support nodes."""

    torch.manual_seed(63190)
    config = PFGRLiteConfig(num_points=4, engineering_only=True, build_chunk_size=64, decode_chunk_size=64)

    class LatticeFactory:
        def build(self, **kwargs):
            return PFGRQueryLattice.build(**kwargs)

    model = PFGRLiteModel(config, query_lattice_factory=LatticeFactory())
    for parameter in model.updater.parameters():
        parameter.data.zero_()
    geometry = VolumeGeometry.from_spacing((9, 9, 9))
    sample = TargetFreeSample(
        "synthetic-zero-u",
        torch.randn(3, 9, 9, 9),
        torch.ones(1, 9, 9, 9, dtype=torch.bool),
        geometry,
        {},
        "",
        "",
    )

    def writer(state, context, action):
        lattice = PFGRQueryLattice.build(context.geometry, context.feature_geometry, query_dtype=state.planes.xy.dtype, build_chunk_size=64)
        return make_action_writer(lattice)(state, context, action)

    def support_mask(state, context, points):
        lattice = PFGRQueryLattice.build(context.geometry, context.feature_geometry, query_dtype=state.planes.xy.dtype, build_chunk_size=64)
        return make_support_legal_mask(lattice)(state, context, points)

    result = run_stage(
        "S1",
        config,
        StageInputs(
            samples=(sample,),
            model=model,
            query=make_point_query(),
            writer=writer,
            target_provider=lambda _subject_id: torch.zeros(1, 9, 9, 9),
            stage_options=StageOptions(stage="S1", epochs=1, max_updates=1, engineering_only=True),
            metadata={"support_legal_mask": support_mask},
        ),
        tmp_path / "zero-u",
    )
    assert result.receipt.metrics["changed_parameter_count_total"] > 0
    assert result.receipt.metrics["operation_counters"]["executed_writes"] == sum(
        int(count) * int(k) for k, count in result.receipt.metrics["sampled_k_counts"].items()
    )
    assert result.receipt.metrics["operation_counters"]["executed_writes"] > 0


def test_s1_rejects_no_writer_support_instead_of_fabricating_legal_rows(tmp_path) -> None:
    """An empty W2 support mask fails closed before random selection."""

    torch.manual_seed(63191)
    config = PFGRLiteConfig(num_points=4, engineering_only=True, build_chunk_size=64, decode_chunk_size=64)

    class LatticeFactory:
        def build(self, **kwargs):
            return PFGRQueryLattice.build(**kwargs)

    model = PFGRLiteModel(config, query_lattice_factory=LatticeFactory())
    geometry = VolumeGeometry.from_spacing((9, 9, 9))
    sample = TargetFreeSample(
        "synthetic-no-support",
        torch.randn(3, 9, 9, 9),
        torch.ones(1, 9, 9, 9, dtype=torch.bool),
        geometry,
        {},
        "",
        "",
    )

    def writer(state, context, action):
        lattice = PFGRQueryLattice.build(context.geometry, context.feature_geometry, query_dtype=state.planes.xy.dtype, build_chunk_size=64)
        return make_action_writer(lattice)(state, context, action)

    def no_support(_state, _context, points):
        return torch.zeros((1, points.shape[1]), dtype=torch.bool, device=points.device)

    with pytest.raises(RuntimeError, match="no writer-support eligible candidate"):
        run_stage(
            "S1",
            config,
            StageInputs(
                samples=(sample,),
                model=model,
                query=make_point_query(),
                writer=writer,
                target_provider=lambda _subject_id: torch.zeros(1, 9, 9, 9),
                stage_options=StageOptions(stage="S1", epochs=1, max_updates=1, engineering_only=True),
                metadata={"support_legal_mask": no_support},
            ),
            tmp_path / "no-support",
        )


def test_s0_live_model_updates_static_b_and_decoder_only(tmp_path) -> None:
    """The actual default S0 path updates B/D while frozen frontend hashes hold."""

    torch.manual_seed(7123)
    config = PFGRLiteConfig(num_points=4, engineering_only=True, build_chunk_size=64, decode_chunk_size=64)

    class LatticeFactory:
        def build(self, **kwargs):
            return PFGRQueryLattice.build(**kwargs)

    model = PFGRLiteModel(config, query_lattice_factory=LatticeFactory())
    geometry = VolumeGeometry.from_spacing((9, 9, 9))
    sample = TargetFreeSample(
        "synthetic-live-s0",
        torch.randn(3, 9, 9, 9),
        torch.ones(1, 9, 9, 9, dtype=torch.bool),
        geometry,
        {},
        "",
        "",
    )
    counters = SimpleNamespace(target_reads=0, observation_reads=1)
    frozen_before = {
        "backbone": module_state_digest(model.frontend.semantic_prior.backbone),
        "semantic": module_state_digest(model.frontend.semantic_prior.semantic_head),
        "point_refiner": module_state_digest(model.frontend.point_refiner),
        "spectral": module_state_digest(model.frontend.spectral_anchor_builder),
    }

    def target_provider(subject_id: str):
        assert subject_id == sample.subject_id
        return torch.zeros(1, 9, 9, 9)

    inputs = StageInputs(
        samples=(sample,),
        model=model,
        target_provider=target_provider,
        stage_options=StageOptions(stage="S0", epochs=1, max_updates=1, engineering_only=True),
        metadata={"counters": counters},
    )
    result = run_stage("S0", config, inputs, tmp_path / "live-s0")
    metrics = result.receipt.metrics
    assert result.receipt.gradient_steps == result.receipt.route_updates == 1
    assert result.completion == "complete"
    assert all(item["nonzero_steps"] == 1 for item in metrics["gradient_evidence"].values())
    assert all(item["changed_parameter_count"] > 0 for item in metrics["update_evidence"].values())
    assert metrics["history_record_count"] == 1
    assert (tmp_path / "live-s0" / "stage_history.jsonl").is_file()
    frozen_after = {
        "backbone": module_state_digest(model.frontend.semantic_prior.backbone),
        "semantic": module_state_digest(model.frontend.semantic_prior.semantic_head),
        "point_refiner": module_state_digest(model.frontend.point_refiner),
        "spectral": module_state_digest(model.frontend.spectral_anchor_builder),
    }
    assert frozen_after == frozen_before
    assert counters.target_reads == 1


def test_s0_semantic_arm_joins_labels_late_and_updates_only_semantic_head(tmp_path) -> None:
    """The explicit semantic arm uses one traversal and a post-prediction label join."""

    torch.manual_seed(8123)
    config = PFGRLiteConfig(num_points=4, engineering_only=True, build_chunk_size=64, decode_chunk_size=64)

    class LatticeFactory:
        def build(self, **kwargs):
            return PFGRQueryLattice.build(**kwargs)

    model = PFGRLiteModel(config, query_lattice_factory=LatticeFactory())
    geometry = VolumeGeometry.from_spacing((9, 9, 9))
    sample = TargetFreeSample(
        "synthetic-semantic-s0",
        torch.randn(3, 9, 9, 9),
        torch.ones(1, 9, 9, 9, dtype=torch.bool),
        geometry,
        {},
        "",
        "",
    )
    loaded = SimpleNamespace(
        subject_id=sample.subject_id,
        observations=sample.observations,
        brain_mask=sample.brain_mask,
        geometry=geometry,
        normalization_metadata={},
        source_paths={},
        target=torch.zeros(1, 9, 9, 9),
        segmentation=torch.zeros(9, 9, 9, dtype=torch.int64),
    )
    loaded.segmentation[0, 0, 0] = 2
    loaded.segmentation[0, 0, 1] = 4
    counters = DataAccessCounters()
    semantic_before = module_state_digest(model.frontend.semantic_prior.semantic_head)
    frozen_before = {
        "backbone": module_state_digest(model.frontend.semantic_prior.backbone),
        "point_refiner": module_state_digest(model.frontend.point_refiner),
        "spectral": module_state_digest(model.frontend.spectral_anchor_builder),
    }

    def target_provider(subject_id: str):
        assert subject_id == sample.subject_id
        return loaded

    result = run_stage(
        "S0",
        config,
        StageInputs(
            samples=(sample,),
            model=model,
            target_provider=target_provider,
            stage_options=StageOptions(
                stage="S0",
                epochs=1,
                max_updates=1,
                loss="charbonnier",
                semantic_objective=True,
                engineering_only=True,
            ),
            metadata={"counters": counters},
        ),
        tmp_path / "semantic-s0",
    )
    metrics = result.receipt.metrics
    assert metrics["gradient_evidence"]["semantic_head"]["nonzero_steps"] == 1
    assert metrics["update_evidence"]["semantic_head"]["changed_parameter_count"] > 0
    assert module_state_digest(model.frontend.semantic_prior.semantic_head) != semantic_before
    assert module_state_digest(model.frontend.semantic_prior.backbone) == frozen_before["backbone"]
    assert module_state_digest(model.frontend.point_refiner) == frozen_before["point_refiner"]
    assert module_state_digest(model.frontend.spectral_anchor_builder) == frozen_before["spectral"]
    assert counters.target_reads == 1 and counters.segmentation_reads == 1
    assert metrics["io_counters"]["segmentation_reads"] == 1


def test_s6_uses_explicit_experiment_options_envelope(tmp_path) -> None:
    seen: list[object] = []

    def evaluator(inputs, options, output_dir):
        del inputs
        seen.append(options)
        output_dir.joinpath("evaluation.json").write_text("{}", encoding="utf-8")
        return {"scenario": options.scenario, "budget": options.budget}

    options = StageOptions(stage="S6", engineering_only=True)
    inputs = StageInputs(
        samples=(_sample(),),
        evaluator=evaluator,
        stage_options=options,
        metadata={"experiment_options": {"scenario": "noop", "budget": 0, "max_subjects": 1, "engineering_only": True}},
    )
    result = run_stage("S6", PFGRLiteConfig(num_points=4, engineering_only=True), inputs, tmp_path / "s6")
    assert seen and seen[0].__class__.__name__ == "ExperimentOptions"
    assert result.receipt.metrics["evaluation"] == {"scenario": "noop", "budget": 0}


def test_s3_preserves_cached_fit_subject_denominator_and_pending_completion(tmp_path) -> None:
    class Bank:
        def rows(self):
            return [SimpleNamespace(subject_key="subject-01", diagnostic=False) for _ in range(4)]

    def fitter(_bank, **_kwargs):
        return {
            "complete": False,
            "stage_state": {"update": 1},
            "metrics": {"fit_complete": False, "train_row_count": 4, "subject_count": 0},
            "resume_state": {},
        }

    result = run_stage(
        "S3",
        PFGRLiteConfig(num_points=4, engineering_only=True),
        StageInputs(
            samples=(_sample(),),
            value_fitter=fitter,
            stage_options=StageOptions(stage="S3", engineering_only=True),
            metadata={"bank": Bank()},
        ),
        tmp_path / "s3-pending",
    )
    assert result.receipt.metrics["fit_subject_count"] == 1
    assert result.receipt.metrics["fit_complete"] is False
    assert result.receipt.metrics["completed"] is False
    assert result.stage_state.substage == "value_fit"
    assert result.stage_state.completion == "pending"


def test_s5_inconclusive_calibration_preserves_pending_state_without_identity_fit(tmp_path) -> None:
    """W5 underpowered collection remains resumable and calibration-free."""

    def runner(_inputs, _options, _output_dir):
        return {
            "schema_version": "pfgr-lite-calibration-run-v1",
            "completed_traces": (),
            "collection_policy": None,
            "calibration": None,
            "metrics": {"status": "INCONCLUSIVE", "insufficient_data": True, "trace_count": 0},
            "artifacts": {},
        }

    options = StageOptions(stage="S5", engineering_only=True)
    inputs = StageInputs(
        samples=(_sample(),),
        stage_options=options,
        metadata={
            "calibration_runner": runner,
            "calibration_run_options": SimpleNamespace(engineering_only=True),
            "subject_role": "calibration",
        },
    )
    result = run_stage("S5", PFGRLiteConfig(num_points=4, engineering_only=True), inputs, tmp_path / "s5-inconclusive")
    assert result.stage_state.substage == "calibration"
    assert result.stage_state.completion == "pending"
    assert result.receipt.metrics["calibration_status"] == "INCONCLUSIVE"
    assert result.receipt.metrics["calibration_complete"] is False
    assert result.receipt.metrics["completed"] is False
    assert result.receipt.metrics["calibration"]["calibration"] is None
