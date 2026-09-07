"""Same-work PFGR-Lite sparse/reference benchmark service.

The benchmark reuses one frozen target-free state, stored action rows, voxel
IDs, dtype, chunk size, and nonlinear decoder for both engines.  It reports
parity and measured work separately from any reduced-Q or reduced-candidate
experiment; no speedup is claimed unless the same-work measurements support it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from .experiments import (
    ExperimentOptions,
    _build_lattice,
    _completed_trace,
    _context_for_sample,
    _jsonable,
    _load_policy,
    _policy_metadata,
    _prediction_for,
    _route_for_sample,
    _sample_id,
    _service_execution,
)
from .metrics import write_json
from .provenance import canonical_digest

BENCHMARK_OPTIONS_SCHEMA = "pfgr-lite-benchmark-options-v1"


def _positive_int(name: str, value: object, *, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds bound {maximum}")
    return int(value)


@dataclass(frozen=True)
class BenchmarkOptions:
    """Strict bounded benchmark options."""

    max_subjects: int = 1
    max_states: int = 2
    candidate_count: int = 4
    teacher_mode: Literal["exact_footprint", "iid_fixed_q"] = "iid_fixed_q"
    query_count: int = 64
    repeats: int = 3
    seed: int = 20260907
    chunk_size: int = 1024
    candidate_chunk_size: int = 1
    dtype: Literal["float32", "float64"] = "float32"
    engineering_only: bool = False
    schema_version: str = BENCHMARK_OPTIONS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != BENCHMARK_OPTIONS_SCHEMA:
            raise ValueError("unknown BenchmarkOptions schema")
        _positive_int("max_subjects", self.max_subjects, maximum=1_000_000)
        _positive_int("max_states", self.max_states, maximum=4)
        _positive_int("candidate_count", self.candidate_count, maximum=2048)
        if self.teacher_mode not in ("exact_footprint", "iid_fixed_q"):
            raise ValueError("teacher_mode must be exact_footprint or iid_fixed_q")
        _positive_int("query_count", self.query_count, maximum=10_000_000)
        if self.teacher_mode == "iid_fixed_q" and self.query_count < 2:
            raise ValueError("iid_fixed_q benchmark requires query_count >= 2")
        _positive_int("repeats", self.repeats, maximum=10_000)
        if self.repeats < 3:
            raise ValueError("benchmark requires at least three bounded repeats")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        _positive_int("chunk_size", self.chunk_size, maximum=1_000_000)
        _positive_int("candidate_chunk_size", self.candidate_chunk_size, maximum=1_000_000)
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if not isinstance(self.engineering_only, bool):
            raise TypeError("engineering_only must be bool")

    def as_dict(self) -> dict[str, Any]:
        return {field.name: _jsonable(getattr(self, field.name)) for field in fields(self)}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> BenchmarkOptions:
        if not isinstance(values, Mapping):
            raise TypeError("BenchmarkOptions must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown BenchmarkOptions keys: {sorted(unknown)}")
        return cls(**dict(values))


def _prepare_output(output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    if destination.exists():
        if not destination.is_dir():
            raise ValueError("output_dir must be a directory")
        if any(destination.iterdir()):
            raise FileExistsError(f"output_dir must be empty and exclusive: {destination}")
    else:
        destination.mkdir(parents=True, exist_ok=False)
    return destination


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _all_voxels(lattice: object) -> Tensor:
    depth, height, width = tuple(int(item) for item in lattice.output_shape_dhw)
    flat = torch.arange(depth * height * width, dtype=torch.long, device=lattice.device)
    area = height * width
    d = torch.div(flat, area, rounding_mode="floor")
    hw = flat - d * area
    h = torch.div(hw, width, rounding_mode="floor")
    w = hw - h * width
    return torch.stack((d, h, w), dim=-1)


def _state_action_cases(inputs: Any, route: object, lattice: object, options: BenchmarkOptions) -> list[tuple[object, object, Tensor]]:
    metadata = getattr(inputs, "metadata", {}) or {}
    provided = metadata.get("benchmark_cases") if isinstance(metadata, Mapping) else None
    if provided is not None:
        cases: list[tuple[object, object, Tensor]] = []
        for row in tuple(provided)[: options.max_states * options.candidate_count]:
            if not isinstance(row, Mapping):
                raise TypeError("benchmark_cases rows must be mappings")
            state, action = row.get("state"), row.get("action")
            ids = row.get("voxel_ids_dhw")
            if state is None or action is None:
                raise ValueError("benchmark case requires state and action")
            if ids is None:
                from .sparse_write import build_footprint

                ids = build_footprint(lattice, action, chunk_size=options.chunk_size).voxel_ids_dhw
            if not isinstance(ids, Tensor):
                ids = torch.as_tensor(ids, dtype=torch.long, device=action.delta.device)
            cases.append((state, action, ids))
        return cases
    trace = _completed_trace(route)
    if trace is None:
        return []
    states = tuple(getattr(trace, "states", ()))
    proposals = tuple(getattr(trace, "proposals", ()))
    cases = []
    for state_index, (state, proposal) in enumerate(zip(states, proposals)):
        if state_index >= options.max_states:
            break
        # The benchmark candidate bound is applied to each stored proposal
        # bank, rather than silently benchmarking only the route winner.
        for point_index in range(min(options.candidate_count, proposal.point_ids.shape[1])):
            action = proposal.row(0, point_index)
            if not action.legal:
                continue
            from .sparse_write import build_footprint

            footprint = build_footprint(lattice, action, chunk_size=options.chunk_size)
            cases.append((state, action, footprint.voxel_ids_dhw))
    return cases


def _mlp(decoder: object, query: Tensor) -> Tensor:
    module = getattr(decoder, "mlp", decoder)
    if not callable(module):
        raise TypeError("decoder must expose a callable mlp")
    output = module(query)
    if not isinstance(output, Tensor):
        raise TypeError("decoder must return a tensor")
    return output.reshape(-1)


def _target_vector(target_context: object, ids: Tensor) -> tuple[Tensor, Tensor, int] | None:
    """Gather target/mask at stored IDs with one fixed global denominator."""

    target = getattr(target_context, "target", None)
    mask = getattr(target_context, "observation_mask", getattr(target_context, "target_mask", None))
    if not isinstance(target, Tensor):
        return None
    if target.ndim == 5:
        target = target[0, 0]
    elif target.ndim == 4:
        target = target[0]
    if target.ndim != 3:
        raise ValueError("benchmark target must be [D,H,W], [1,D,H,W], or [1,1,D,H,W]")
    if mask is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    elif isinstance(mask, Tensor):
        if mask.ndim == 5:
            valid = mask[0, 0].to(dtype=torch.bool)
        elif mask.ndim == 4:
            valid = mask[0].to(dtype=torch.bool)
        elif mask.ndim == 3:
            valid = mask.to(dtype=torch.bool)
        else:
            raise ValueError("benchmark target mask has unsupported rank")
    else:
        raise TypeError("benchmark target mask must be a tensor or None")
    denominator = int(valid.sum().item())
    if denominator <= 0:
        raise ValueError("benchmark target mask must contain at least one valid voxel")
    target = target.to(device=ids.device)
    valid = valid.to(device=ids.device)
    gathered_target = target[ids[:, 0], ids[:, 1], ids[:, 2]]
    gathered_mask = valid[ids[:, 0], ids[:, 1], ids[:, 2]]
    return gathered_target, gathered_mask, denominator


def _masked_gain(
    before: Tensor,
    after: Tensor,
    target_data: tuple[Tensor, Tensor, int] | None,
    *,
    probabilities: Tensor | None = None,
    epsilon: float = 1e-3,
) -> float | None:
    if target_data is None:
        return None
    target, valid, denominator = target_data
    before_error = torch.sqrt((before - target).square() + float(epsilon) ** 2)
    after_error = torch.sqrt((after - target).square() + float(epsilon) ** 2)
    difference = before_error - after_error
    # The denominator is the complete subject mask, not the sampled/support
    # count.  Invalid support rows therefore contribute exactly zero.  For
    # iid_fixed_q, use the W2 plane-mixture inverse probability p(v)=c/S and
    # preserve duplicate draws; this is the same weighted estimator used by
    # the detached teacher rather than a uniform/truncated proxy.
    if probabilities is None:
        weighted = difference * valid.to(dtype=before.dtype)
        return float(weighted.to(dtype=torch.float64).sum().item() / denominator)
    if probabilities.ndim != 1 or probabilities.shape[0] != difference.shape[0]:
        raise ValueError("sample probabilities must align with benchmark query rows")
    weights = valid.to(dtype=difference.dtype) / (
        float(denominator)
        * probabilities.to(device=difference.device, dtype=difference.dtype)
    )
    return float((difference * weights).to(dtype=torch.float64).mean().item())


def _one_parity_case(
    lattice: object,
    state: object,
    action: object,
    ids: Tensor,
    decoder: object,
    *,
    chunk_size: int,
    footprint: object | None = None,
    target_context: object | None = None,
    probabilities: Tensor | None = None,
    sampling_law: str = "exact_union_v1",
    sampling_seed: int | None = None,
    charbonnier_epsilon: float = 1e-3,
) -> dict[str, Any]:
    from .sparse_write import query_write_delta, reference_full_write

    ids = ids.to(device=action.delta.device, dtype=torch.long)
    device = ids.device
    if footprint is None:
        footprint = _build_footprint(lattice, action, chunk_size)
    target_data = _target_vector(target_context, ids) if target_context is not None else None
    # Shared initial query is measured once; both methods consume this exact
    # tensor, while their method-specific timings and calls remain separate.
    _sync(device)
    shared_started = time.perf_counter()
    before = lattice.query(state.planes, ids, chunk_size=chunk_size)
    before_prediction = _mlp(decoder, before)
    _sync(device)
    shared_elapsed = time.perf_counter() - shared_started
    _sync(device)
    optimized_started = time.perf_counter()
    delta_query = query_write_delta(lattice, footprint, ids, action.delta, chunk_size=chunk_size)
    sparse_query = before + delta_query
    sparse_prediction = _mlp(decoder, sparse_query)
    sparse_gain = _masked_gain(
        before_prediction,
        sparse_prediction,
        target_data,
        probabilities=probabilities,
        epsilon=charbonnier_epsilon,
    )
    _sync(device)
    optimized_elapsed = time.perf_counter() - optimized_started
    _sync(device)
    reference_started = time.perf_counter()
    full_state = reference_full_write(lattice, state.planes, action, chunk_size=chunk_size)
    full_query = lattice.query(full_state, ids, chunk_size=chunk_size)
    full_prediction = _mlp(decoder, full_query)
    full_gain = _masked_gain(
        before_prediction,
        full_prediction,
        target_data,
        probabilities=probabilities,
        epsilon=charbonnier_epsilon,
    )
    _sync(device)
    reference_elapsed = time.perf_counter() - reference_started
    prediction_error = float((sparse_prediction - full_prediction).abs().max().item()) if ids.numel() else 0.0
    query_error = float((sparse_query - full_query).abs().max().item()) if ids.numel() else 0.0
    gain_error = None if sparse_gain is None or full_gain is None else abs(sparse_gain - full_gain)
    atol, rtol = ((1e-10, 1e-9) if action.delta.dtype == torch.float64 else (1e-6, 1e-5))
    parity_failure: str | None = None
    try:
        torch.testing.assert_close(sparse_query, full_query, atol=atol, rtol=rtol)
        torch.testing.assert_close(sparse_prediction, full_prediction, atol=atol, rtol=rtol)
        if sparse_gain is not None and full_gain is not None:
            torch.testing.assert_close(
                torch.tensor(sparse_gain, dtype=action.delta.dtype),
                torch.tensor(full_gain, dtype=action.delta.dtype),
                atol=atol,
                rtol=rtol,
            )
    except AssertionError as exc:
        parity_failure = str(exc).splitlines()[0]
    return {
        "voxel_count": int(ids.shape[0]),
        "query_error_max": query_error,
        "prediction_error_max": prediction_error,
        "gain_error_max": gain_error,
        "charbonnier_epsilon": float(charbonnier_epsilon),
        "sampling_law": sampling_law,
        "sampling_seed": sampling_seed,
        "query_draws": int(ids.shape[0]),
        "candidate_batch_size": 1,
        "candidate_batch_scope": "single_action_serial",
        "cache_scope": "lattice_query_cache_only",
        "optimized_gain": sparse_gain,
        "reference_gain": full_gain,
        "shared_before_elapsed_seconds": float(shared_elapsed),
        "optimized_elapsed_seconds": float(optimized_elapsed),
        "reference_elapsed_seconds": float(reference_elapsed),
        "parity_failure": parity_failure,
        "dtype": str(sparse_query.dtype),
        "query_calls": {"shared_before": 1, "optimized_delta": 0, "reference_after": 1},
        "decoder_calls": {"shared_before": 1, "optimized": 1, "reference": 1},
        "decoded_outputs": {"optimized": int(sparse_prediction.numel()), "reference": int(full_prediction.numel())},
        "stored_action_reused": True,
        "reference_rebased_action": False,
        "full_clone_bytes": int(sum(getattr(state.planes, name).numel() * getattr(state.planes, name).element_size() for name in ("xy", "xz", "yz"))),
        "optimized_clone_bytes": 0,
    }


def _build_footprint(lattice: object, action: object, chunk_size: int) -> object:
    from .sparse_write import build_footprint

    return build_footprint(lattice, action, chunk_size=chunk_size)


def run_teacher_benchmark(inputs: Any, options: BenchmarkOptions, output_dir: Path) -> Mapping[str, Any]:
    """Run benchmark paths under detached/eval service semantics."""

    model = getattr(inputs, "model", None)
    decoder = getattr(model, "decoder", None)
    with _service_execution(model, decoder):
        return _run_teacher_benchmark_impl(inputs, options, output_dir)


def _run_teacher_benchmark_impl(inputs: Any, options: BenchmarkOptions, output_dir: Path) -> Mapping[str, Any]:
    """Run same-work sparse/reference parity and measured bounded timings."""

    from .stages import StageInputs

    if not isinstance(inputs, StageInputs):
        raise TypeError("inputs must be StageInputs")
    if not isinstance(options, BenchmarkOptions):
        raise TypeError("options must be BenchmarkOptions")
    destination = _prepare_output(output_dir)
    exp_options = ExperimentOptions(
        scenario="random",
        budget=4,
        max_subjects=options.max_subjects,
        seed=options.seed,
        candidate_chunk_size=options.candidate_chunk_size,
        decode_chunk_size=options.chunk_size,
        teacher_mode=options.teacher_mode,
        query_count=options.query_count,
        engineering_only=options.engineering_only,
    )
    rows: list[dict[str, Any]] = []
    max_query_error = 0.0
    max_prediction_error = 0.0
    max_gain_error = 0.0
    gain_unavailable_rows = 0
    parity_failures: list[dict[str, Any]] = []
    cases_count = 0
    context_receipts: list[dict[str, Any]] = []
    policy_receipts: list[dict[str, Any]] = []
    for index, sample in enumerate(tuple(getattr(inputs, "samples", ()))[: options.max_subjects]):
        sample_started = time.perf_counter()
        sample_row_start = len(rows)
        pipeline_counters: dict[str, int] = {
            "observation_encode_calls": 0,
            "initial_decode_calls": 0,
            "final_decode_calls": 0,
        }
        context = _context_for_sample(
            inputs, sample, pipeline_counters=pipeline_counters
        )
        context_producer = getattr(context, "producer", None)
        context_compatibility = getattr(context_producer, "compatibility", context_producer)
        context_producer_hash = getattr(
            context_compatibility,
            "digest",
            getattr(context_producer, "compatibility_hash", None),
        )
        context_normalization_hash = getattr(
            context_compatibility,
            "observation_normalization_hash",
            getattr(context_producer, "observation_normalization_hash", None),
        )
        initial_planes = getattr(context, "initial_planes", None)
        initialization_hash = (
            canonical_digest(
                {
                    "context_id": getattr(context, "context_id", None),
                    "planes": _jsonable(initial_planes),
                },
                prefix="pfgr-lite-initialization-v1|",
            )
            if initial_planes is not None
            else None
        )
        role_manifest = getattr(inputs, "role_manifest", None)
        metadata = getattr(inputs, "metadata", {})
        provided = (
            metadata.get("source_receipt", metadata.get("provenance", {}))
            if isinstance(metadata, Mapping)
            else {}
        )
        actual_role_baseline = getattr(role_manifest, "baseline_split_hash", None)
        supplied_role_baseline = (
            metadata.get("baseline_split_hash")
            if isinstance(metadata, Mapping)
            else None
        ) or (
            provided.get("baseline_split_hash")
            if isinstance(provided, Mapping)
            else None
        )
        if (
            actual_role_baseline is not None
            and supplied_role_baseline is not None
            and str(actual_role_baseline) != str(supplied_role_baseline)
        ):
            raise ValueError("benchmark baseline split identity conflicts with role manifest")
        actual_role_manifest = getattr(role_manifest, "digest", None)
        supplied_role_manifest = (
            metadata.get(
                "training_role_manifest_hash", metadata.get("role_manifest_hash")
            )
            if isinstance(metadata, Mapping)
            else None
        ) or (
            provided.get(
                "training_role_manifest_hash", provided.get("role_manifest_hash")
            )
            if isinstance(provided, Mapping)
            else None
        )
        if (
            actual_role_manifest is not None
            and supplied_role_manifest is not None
            and str(actual_role_manifest) != str(supplied_role_manifest)
        ):
            raise ValueError("benchmark role manifest identity conflicts with role manifest")
        metadata_actuals = {
            "producer_compatibility_hash": context_producer_hash,
            "normalization_hash": context_normalization_hash,
            "initialization_hash": initialization_hash,
            "baseline_split_hash": actual_role_baseline,
            "training_role_manifest_hash": actual_role_manifest,
        }
        for key, actual in metadata_actuals.items():
            supplied = metadata.get(key) if isinstance(metadata, Mapping) else None
            if actual is not None and supplied is not None and str(actual) != str(supplied):
                raise ValueError(
                    f"benchmark metadata {key!r} conflicts with sealed service identity"
                )
        context_receipts.append(
            {
                "subject_id": _sample_id(sample, index),
                "context_id": getattr(context, "context_id", None),
                "producer_compatibility_hash": context_producer_hash,
                "normalization_hash": context_normalization_hash,
                "initialization_hash": initialization_hash,
                "baseline_split_hash": actual_role_baseline or supplied_role_baseline,
                "training_role_manifest_hash": actual_role_manifest or supplied_role_manifest,
                "pipeline_counters": pipeline_counters,
                "pipeline_counter_scope": (
                    "service_outer_calls_only; route counters remain separately tagged"
                ),
                "source_provenance": _jsonable(getattr(context, "source_provenance", None)),
            }
        )
        config = getattr(getattr(inputs, "execution", None), "config", getattr(inputs, "config", None))
        teacher_config = getattr(config, "teacher", None)
        charbonnier_epsilon = float(
            getattr(teacher_config, "epsilon", 1e-3)
        )
        if not torch.isfinite(torch.tensor(charbonnier_epsilon)) or charbonnier_epsilon <= 0.0:
            raise ValueError("benchmark teacher epsilon must be finite and positive")
        lattice = _build_lattice(inputs, context, getattr(inputs, "model", None), config)
        policy = _load_policy(inputs, context, exp_options, config)
        policy_receipt = {
            "subject_id": _sample_id(sample, index),
            "effective_policy_hash": getattr(policy, "policy_hash", None),
            "policy": _jsonable(_policy_metadata(policy)) if policy is not None else None,
            "status": "RECORDED" if policy is not None else "UNRECORDED_REQUIRES_EFFECTIVE_POLICY_SEAM",
        }
        policy_receipts.append(policy_receipt)
        route, _, _ = _route_for_sample(inputs, sample, context, exp_options, config, lattice, policy)
        if lattice is None:
            raise ValueError("benchmark requires canonical PFGR query lattice")
        decoder = getattr(getattr(inputs, "model", None), "decoder", None) or getattr(route, "decoder", None)
        if decoder is None:
            raise ValueError("benchmark requires the shared nonlinear decoder")
        cases = _state_action_cases(inputs, route, lattice, options)
        target_context = None
        if getattr(inputs, "target_provider", None) is not None:
            from .experiments import _target_join

            final_prediction = _prediction_for(
                getattr(inputs, "model", None),
                route,
                context,
                final=True,
                options=exp_options,
                pipeline_counters=pipeline_counters,
            )
            target_context = _target_join(inputs, sample, context, route, final_prediction, exp_options)
        if target_context is None:
            raise ValueError(
                "configured label benchmark requires a deferred target_provider and measured target context"
            )
        for case_index, (state, action, ids) in enumerate(cases):
            expected_dtype = torch.float64 if options.dtype == "float64" else torch.float32
            if action.delta.dtype != expected_dtype:
                raise TypeError(
                    f"benchmark dtype={options.dtype} requires stored action dtype {expected_dtype}; "
                    f"got {action.delta.dtype}"
                )
            if lattice.query_dtype != expected_dtype:
                raise TypeError(
                    f"benchmark dtype={options.dtype} requires canonical lattice dtype {expected_dtype}; "
                    f"got {lattice.query_dtype}"
                )
            build_started = time.perf_counter()
            footprint = _build_footprint(lattice, action, options.chunk_size)
            footprint_build_elapsed = time.perf_counter() - build_started
            benchmark_ids = ids
            benchmark_probabilities: Tensor | None = None
            benchmark_sampling_law = "exact_union_v1"
            benchmark_sampling_seed: int | None = None
            if options.teacher_mode == "iid_fixed_q":
                from .teacher import _sample_iid_voxels

                benchmark_sampling_seed = int(
                    options.seed + index * 1_000_003 + case_index
                )
                draw_linear, benchmark_probabilities, _ = _sample_iid_voxels(
                    footprint,
                    q_draws=options.query_count,
                    seed=benchmark_sampling_seed,
                )
                _, height, width = lattice.output_shape_dhw
                depth = torch.div(
                    draw_linear, height * width, rounding_mode="floor"
                )
                remainder = draw_linear - depth * height * width
                row = torch.div(remainder, width, rounding_mode="floor")
                column = remainder - row * width
                benchmark_ids = torch.stack((depth, row, column), dim=-1).to(
                    device=action.delta.device
                )
                benchmark_sampling_law = "iid_fixed_q_plane_mixture_c_over_S_v1"
            cases_count += 1
            for repeat in range(options.repeats):
                device = action.delta.device
                cache_reset = False
                if repeat == 0 and hasattr(lattice, "release"):
                    # Explicitly release the process cache entry before the
                    # first timed pass.  The immutable lattice object remains
                    # the same input for every repeat, preserving same-work
                    # semantics while making the cold/warm distinction
                    # auditable.
                    lattice.release()
                    cache_reset = True
                lattice_accounting_before = dict(
                    getattr(lattice, "memory_accounting", {})
                )
                _sync(device)
                started = time.perf_counter()
                parity = _one_parity_case(
                    lattice,
                    state,
                    action,
                    benchmark_ids,
                    decoder,
                    chunk_size=options.chunk_size,
                    footprint=footprint,
                    target_context=target_context,
                    probabilities=benchmark_probabilities,
                    sampling_law=benchmark_sampling_law,
                    sampling_seed=benchmark_sampling_seed,
                    charbonnier_epsilon=charbonnier_epsilon,
                )
                _sync(device)
                elapsed = time.perf_counter() - started
                lattice_accounting_after = dict(
                    getattr(lattice, "memory_accounting", {})
                )
                lattice_counter_delta: dict[str, int | float] = {}
                for key, value in lattice_accounting_after.items():
                    previous = lattice_accounting_before.get(key)
                    if isinstance(value, (int, float)) and isinstance(previous, (int, float)):
                        lattice_counter_delta[key] = value - previous
                allocated: int | None = None
                reserved: int | None = None
                if device.type == "cuda":
                    allocated = int(torch.cuda.memory_allocated(device))
                    reserved = int(torch.cuda.memory_reserved(device))
                parity.update(
                    {
                        "subject_id": _sample_id(sample, index),
                        "case_index": case_index,
                        "repeat": repeat,
                        "cache_state": (
                            "cold_lattice_query_cache"
                            if repeat == 0
                            else "warm_lattice_query_cache"
                        ),
                        "cache_reset": cache_reset,
                        "cache_reset_scope": "lattice_query_cache_only",
                        "footprint_build_elapsed_seconds": float(footprint_build_elapsed),
                        "elapsed_seconds": float(elapsed),
                        "allocated_memory_bytes": allocated,
                        "reserved_memory_bytes": reserved,
                        "device": str(device),
                        "state_version": int(getattr(state, "state_version", 0)),
                        "action_id": str(getattr(action, "action_id", f"action-{case_index}")),
                        "effective_policy_hash": policy_receipt["effective_policy_hash"],
                        "lattice_counter_delta": lattice_counter_delta,
                        "sampling_probability_digest": (
                            canonical_digest(
                                benchmark_probabilities.detach()
                                .to(device="cpu")
                                .contiguous()
                                .numpy()
                                .tobytes()
                                .hex(),
                                prefix="pfgr-lite-benchmark-pv-v1|",
                            )
                            if benchmark_probabilities is not None
                            else None
                        ),
                    }
                )
                rows.append(parity)
                max_query_error = max(max_query_error, parity["query_error_max"])
                max_prediction_error = max(max_prediction_error, parity["prediction_error_max"])
                if parity["gain_error_max"] is None:
                    gain_unavailable_rows += 1
                else:
                    max_gain_error = max(max_gain_error, parity["gain_error_max"])
                if parity["parity_failure"] is not None:
                    parity_failures.append(parity)
        sample_elapsed = float(time.perf_counter() - sample_started)
        for row in rows[sample_row_start:]:
            row["full_pipeline_elapsed_seconds"] = sample_elapsed
            row["pipeline_counters"] = dict(pipeline_counters)
            row["pipeline_counter_scope"] = (
                "service_outer_calls_only; route counters remain separately tagged"
            )
    if not rows:
        raise ValueError("benchmark requires at least one stored action case")
    subject_initialization_hashes = {
        str(item["subject_id"]): str(item["initialization_hash"])
        for item in context_receipts
        if item.get("initialization_hash")
    }
    # Initialization is a per-subject identity (different live initial
    # lattices are expected across subjects), while producer and observation
    # normalization identities are global benchmark prerequisites.  Preserve a
    # single global value only when all observed contexts agree; never copy the
    # first subject's value into a multi-subject receipt.
    producer_hashes = {
        str(item["producer_compatibility_hash"])
        for item in context_receipts
        if item.get("producer_compatibility_hash")
    }
    normalization_hashes = {
        str(item["normalization_hash"])
        for item in context_receipts
        if item.get("normalization_hash")
    }
    baseline_hashes = {
        str(item["baseline_split_hash"])
        for item in context_receipts
        if item.get("baseline_split_hash")
    }
    role_manifest_hashes = {
        str(item["training_role_manifest_hash"])
        for item in context_receipts
        if item.get("training_role_manifest_hash")
    }
    dtype = rows[0]["dtype"]
    query_atol, query_rtol = ((1e-10, 1e-9) if dtype == "torch.float64" else (1e-6, 1e-5))
    parity_payload = {
        "schema_version": BENCHMARK_OPTIONS_SCHEMA,
        "status": "PASS" if not parity_failures else "FAIL",
        "same_work": True,
        "query_error_max": max_query_error,
        "prediction_error_max": max_prediction_error,
        "gain_error_max": max_gain_error,
        "gain_unavailable_rows": gain_unavailable_rows,
        "query_tolerance_atol": query_atol,
        "query_tolerance_rtol": query_rtol,
        "prediction_tolerance_atol": query_atol,
        "prediction_tolerance_rtol": query_rtol,
        "gain_tolerance_atol": query_atol,
        "gain_tolerance_rtol": query_rtol,
        "gain_parity_required": True,
        "failure_count": len(parity_failures),
        "cases": cases_count,
        "repeats": options.repeats,
        "scientific_status": "NOT_EVALUATED",
    }
    benchmark_payload = {
        "schema_version": BENCHMARK_OPTIONS_SCHEMA,
        "software_status": "SOFTWARE_PASS" if not parity_failures else "SOFTWARE_FAIL",
        "scientific_status": "NOT_EVALUATED",
        "options": options.as_dict(),
        "same_work": True,
        "cold_warm_labels": [
            "cold_lattice_query_cache",
            "warm_lattice_query_cache",
        ],
        "rows": len(rows),
        "cases": cases_count,
        "parity": parity_payload,
        "timing": {
            "optimized_mean_seconds": float(sum(row["optimized_elapsed_seconds"] for row in rows) / len(rows)),
            "reference_mean_seconds": float(sum(row["reference_elapsed_seconds"] for row in rows) / len(rows)),
            "shared_before_mean_seconds": float(sum(row["shared_before_elapsed_seconds"] for row in rows) / len(rows)),
            "optimized_to_reference_ratio": float(
                (sum(row["optimized_elapsed_seconds"] for row in rows) / sum(row["reference_elapsed_seconds"] for row in rows))
                if sum(row["reference_elapsed_seconds"] for row in rows) > 0.0
                else float("nan")
            ),
            "matched_work_only": True,
            "speedup_claim": "none_without_stable_independent_repeats",
        },
        "scientific_scope": "CPU/CUDA software parity only; no speedup or real-data claim",
        "effective_policy": {
            "rows": policy_receipts,
            "status": "RECORDED" if all(item["status"] == "RECORDED" for item in policy_receipts) else "UNRECORDED_REQUIRES_EFFECTIVE_POLICY_SEAM",
        },
        "source_receipt": {
            "options_hash": canonical_digest(options.as_dict(), prefix="pfgr-lite-benchmark-source-v1|"),
            "sample_ids": [_sample_id(sample, index) for index, sample in enumerate(tuple(getattr(inputs, "samples", ()))[: options.max_subjects])],
            "producer_compatibility_hash": next(iter(producer_hashes)) if len(producer_hashes) == 1 else None,
            "normalization_hash": next(iter(normalization_hashes)) if len(normalization_hashes) == 1 else None,
            "initialization_hash": next(iter(subject_initialization_hashes.values())) if len(subject_initialization_hashes) == 1 else None,
            "subject_initialization_hashes": subject_initialization_hashes,
            "baseline_split_hash": next(iter(baseline_hashes)) if len(baseline_hashes) == 1 else None,
            "training_role_manifest_hash": next(iter(role_manifest_hashes)) if len(role_manifest_hashes) == 1 else None,
            "engineering_only": bool(options.engineering_only),
            "contexts": context_receipts,
            "actual_devices": sorted({row.get("device", "cpu") for row in rows}),
        },
    }
    provided = getattr(inputs, "metadata", {})
    explicit_source = (
        provided.get("source_receipt", provided.get("provenance", {}))
        if isinstance(provided, Mapping)
        else {}
    )
    if isinstance(explicit_source, Mapping):
        for key in (
            "source_hash",
            "dirty_hash",
            "checkpoint_hash",
            "baseline_split_hash",
            "training_role_manifest_hash",
            "initialization_hash",
            "subject_initialization_hashes",
            "split_role",
            "split_role_hash",
            "normalization_hash",
            "producer_compatibility_hash",
            "mask_definition",
            "label_definition",
            "loss_definition",
            "data_range",
            "engineering_only",
        ):
            if key in explicit_source:
                supplied = _jsonable(explicit_source[key])
                actual = benchmark_payload["source_receipt"].get(key)
                if key == "initialization_hash" and len(subject_initialization_hashes) > 1:
                    raise ValueError(
                        "benchmark scalar initialization_hash is ambiguous across subjects; "
                        "use subject_initialization_hashes"
                    )
                if key == "subject_initialization_hashes" and subject_initialization_hashes:
                    if supplied != subject_initialization_hashes:
                        raise ValueError(
                            "benchmark subject_initialization_hashes conflict with contexts"
                        )
                    continue
                if actual is not None and supplied != actual:
                    raise ValueError(
                        f"benchmark source receipt {key!r} conflicts with sealed service identity"
                    )
                if actual is None:
                    benchmark_payload["source_receipt"][key] = supplied
    with (destination / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
    write_json(destination / "benchmark.json", benchmark_payload)
    write_json(destination / "parity.json", parity_payload)
    return {
        "software_status": benchmark_payload["software_status"],
        "scientific_status": "NOT_EVALUATED",
        "row_count": len(rows),
        "case_count": cases_count,
        "benchmark_path": destination / "benchmark.json",
        "rows_path": destination / "rows.jsonl",
        "parity_path": destination / "parity.json",
        "source_receipt": benchmark_payload["source_receipt"],
        "effective_policy": benchmark_payload["effective_policy"],
    }


__all__ = ["BENCHMARK_OPTIONS_SCHEMA", "BenchmarkOptions", "run_teacher_benchmark"]
