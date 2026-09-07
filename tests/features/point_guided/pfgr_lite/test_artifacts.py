"""CPU-only contract tests for the bounded PFGR-Lite evidence package."""

from __future__ import annotations

import hashlib
import inspect
import json
import zipfile
from pathlib import Path

import pytest

from smagm.data.brats21_point_guided import deterministic_subject_split
from smagm.features.point_guided.pfgr_lite.data import build_training_role_manifest
from smagm.features.point_guided.pfgr_lite.artifacts import (
    DestinationExistsError,
    EvidencePathError,
    EvidenceValidationError,
    UnsafeEvidenceError,
    package_evidence,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _basic_run(root: Path, name: str = "run") -> Path:
    run = root / name
    run.mkdir(parents=True)
    _write_json(
        run / "receipt.json", {"schema": "synthetic-v1", "status": "SOFTWARE_PASS"}
    )
    _write_json(run / "resolved_config.json", {"device": "cpu", "synthetic": True})
    _write_json(run / "effective_policy.json", {"budget": 0})
    _write_json(run / "source.json", {"commit": "synthetic"})
    _write_json(run / "environment.json", {"python": "test"})
    _write_json(run / "weights.json", {"provenance": "synthetic-untrained"})
    _write_json(run / "metrics_summary.json", {"mae": 0.25, "count": 1})
    return run


def test_valid_package_has_hashes_and_exclusions(tmp_path: Path) -> None:
    run = _basic_run(tmp_path)
    (run / "unknown.json").write_text('{"not": "allow-listed"}\n', encoding="utf-8")
    (run / "test_output.txt").write_text("pytest: 1 passed\n", encoding="utf-8")
    output = tmp_path / "evidence"

    manifest = package_evidence([run], output)

    assert manifest["schema"] == "pfgr-lite-evidence-v1"
    assert manifest["status"] == "SOFTWARE_PASS"
    assert manifest["scientific_status"] == "NOT_EVALUATED"
    assert manifest["evidence_status"] == "READY"
    assert manifest["limits"]["max_file_size_bytes"] > 0
    included = manifest["included"]
    assert included
    with zipfile.ZipFile(output / "evidence.zip") as archive:
        assert archive.namelist() == sorted(archive.namelist())
        assert all(name.startswith("runs/run/") for name in archive.namelist())
        for row in included:
            payload = archive.read(row["archive_path"])
            assert len(payload) == row["size_bytes"]
            assert hashlib.sha256(payload).hexdigest() == row["sha256"]
    assert any(item["path"] == "unknown.json" for item in manifest["exclusions"])


def test_json_secret_patterns_are_scanned_in_nested_values(tmp_path: Path) -> None:
    run = _basic_run(tmp_path)
    _write_json(
        run / "metrics.json",
        {
            "message": "api_key=synthetic_not_a_real_key",
            "nested": {"argv": ["python", "--safe"]},
        },
    )
    with pytest.raises(UnsafeEvidenceError, match="credential-like string"):
        package_evidence([run], tmp_path / "secret-json")

    _write_json(
        run / "metrics.json",
        {"nested": {"argv": ["python", "--token=synthetic_not_a_real_key"]}},
    )
    with pytest.raises(UnsafeEvidenceError, match="credential-like string"):
        package_evidence([run], tmp_path / "secret-json-nested")


def test_structural_metadata_allows_geometry_and_scalars_but_rejects_arrays(
    tmp_path: Path,
) -> None:
    run = _basic_run(tmp_path)
    _write_json(
        run / "resolved_config.json",
        {
            "target_modality": "T1ce",
            "tensor_dtype": "float32",
            "geometry": {
                "affine": [
                    [1.0, 0.0, 0.0, 10.0],
                    [0.0, 1.0, 0.0, 20.0],
                    [0.0, 0.0, 1.0, 30.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            },
        },
    )
    _write_json(
        run / "action_metrics.json",
        {
            "target_modality": "T1ce",
            "raw_gain": 0.5,
            "uncertainty": 0.1,
            "delta": [0.1, -0.2, 0.3],
            "k_bins": [1, 0, 0, 0, 0],
        },
    )
    manifest = package_evidence([run], tmp_path / "safe-geometry")
    assert manifest["status"] == "SOFTWARE_PASS"

    _write_json(run / "metrics.json", {"pixels": [[[0, 1], [2, 3]]]})
    with pytest.raises(
        UnsafeEvidenceError, match="unrecognised array|raw target/image"
    ):
        package_evidence([run], tmp_path / "pixel-array")

    _write_json(run / "metrics.json", {"alternate_key": [0, 1, 2, 3]})
    with pytest.raises(UnsafeEvidenceError, match="unrecognised array"):
        package_evidence([run], tmp_path / "alternate-array")

    _write_json(run / "metrics.json", {"descriptor": list(range(25))})
    with pytest.raises(UnsafeEvidenceError, match="oversized numeric vector"):
        package_evidence([run], tmp_path / "oversized-vector")


def test_declared_cli_metadata_handoff_fixtures_are_packaged_without_payloads(
    tmp_path: Path,
) -> None:
    """Cover the concrete CLI handoffs while retaining the payload boundary."""

    run = _basic_run(tmp_path, "cli-run")
    _write_json(
        run / "service_receipt.json",
        {
            "service_result": {
                "software_status": "SOFTWARE_PASS",
                "source_receipt": {"target_volume_reads": 0},
            },
            "service_artifacts": {"metrics_path": "metrics.json"},
            "data_counters": {"target_reads": 0},
            "operation_calls": {"teacher_calls": 0},
        },
    )
    _write_json(
        run / "bank_verify.json",
        {
            "schema_version": "pfgr-lite-bank-verify-v1",
            "status": "SOFTWARE_PASS",
            "row_count": 1,
            "replay": {"rows_checked": 1, "target_reads": 0},
        },
    )
    _write_json(
        run / "value_fit.json",
        {
            "fit_complete": True,
            "input_variant": 366,
            "row_count": 1,
            "mse_raw": 0.1,
            "target_volume_reads": 0,
            "teacher_calls": 0,
        },
    )
    _write_json(
        run / "value_evaluate.json",
        {
            "row_count": 1,
            "same_bank_scope": True,
            "paired_ranking_row_count": 1,
            "target_volume_reads": 0,
            "teacher_calls": 0,
        },
    )
    _write_json(
        run / "value_evaluate_pairs.json",
        {
            "schema_version": "pfgr-lite-value-evaluation-pairs-v1",
            "same_bank": True,
            "rows": [{"row_id": 0, "predicted_raw": 0.1, "measured_raw_gain": 0.2}],
        },
    )
    _write_json(
        run / "calibration_evidence.json",
        {
            "version": "pfgr-lite-calibration-evidence-v1",
            "producer_fit_subjects": ["producer-01"],
            "fit_subjects": ["fit-01"],
            "allowance_subjects": ["allowance-01"],
            "completed_trace_hashes": ["trace-01"],
            "completed_trace_receipts": [
                {
                    "trace_hash": "trace-01",
                    "context_id": "context-01",
                    "state_versions": [0, 1],
                    "proposal_digests": ["proposal-01"],
                    "action_digests": ["action-01"],
                }
            ],
            "winner_bindings": [
                ["fit-01", "action-01", "proposal-01", "action-01", 0]
            ],
            "winner_confirmations": [
                [
                    "fit-01",
                    "action-01",
                    "proposal-01",
                    "action-01",
                    "exact",
                    0,
                    None,
                    None,
                    "confirmation-01",
                ]
            ],
            "role_manifest": {
                "baseline_split_hash": "split-01",
                "subject_group_ids": [["producer-01", "group-01"]],
            },
        },
    )
    _write_json(
        run / "trace_receipts.json",
        {
            "rows": [
                {
                    "trace_hash": "trace-01",
                    "state_versions": [0, 1],
                    "proposal_digests": ["proposal-01"],
                    "action_digests": ["action-01"],
                }
            ]
        },
    )
    _write_json(
        run / "collection_policy.json",
        {
            "schema_version": "pfgr-lite-effective-policy-v1",
            "mode": "forced_diagnostic",
            "budget": 4,
            "target_volume_reads": 0,
        },
    )
    _write_json(run / "fit_winners.json", {"rows": [{"subject_id": "fit-01", "raw_gain": 0.1}]})
    _write_json(
        run / "allowance_winners.json",
        {"rows": [{"subject_id": "allowance-01", "raw_gain": 0.1}]},
    )
    _write_json(
        run / "review_context.json",
        {
            "schema_version": "pfgr-lite-review-context-v1",
            "scope": "R9-final-evaluation",
            "status": "REVIEW_REQUIRED",
            "context": {"selected_subject_ids": ["test-01"]},
        },
    )
    _write_json(
        run / "stage_state.json",
        {
            "stage": "S0",
            "substage": "complete",
            "epoch": 1,
            "update": 1,
            "microstep": 0,
            "optimizer_groups": ["static_head"],
            "completion": "complete",
            "version": "pfgr-lite-stage-state-v1",
        },
    )
    _write_json(
        run / "resume_summary.json",
        {
            "schema_version": "pfgr-lite-resume-summary-v1",
            "status": "SOFTWARE_PASS",
            "stage": "S0",
            "optimizer_groups": ["static_head"],
            "restored_rng_streams": ["torch_cpu"],
        },
    )

    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "model.pt").write_bytes(b"checkpoint")
    (run / "patient.nii.gz").write_bytes(b"patient-volume")
    (run / "raw_bank.json").write_text('{"rows": [{"target_values": [1]}]}\n', encoding="utf-8")

    manifest = package_evidence([run], tmp_path / "cli-package")
    included_names = {Path(item["source_path"]).name for item in manifest["included"]}
    expected = {
        "service_receipt.json",
        "bank_verify.json",
        "value_fit.json",
        "value_evaluate.json",
        "value_evaluate_pairs.json",
        "calibration_evidence.json",
        "trace_receipts.json",
        "collection_policy.json",
        "fit_winners.json",
        "allowance_winners.json",
        "review_context.json",
        "stage_state.json",
        "resume_summary.json",
    }
    assert expected <= included_names
    excluded_paths = {str(item["path"]) for item in manifest["exclusions"]}
    assert "patient.nii.gz" in excluded_paths
    assert "raw_bank.json" in excluded_paths
    assert "checkpoints/model.pt" in excluded_paths
    assert all(not path.endswith("stage_runtime.json") for path in included_names)


def test_scalar_counter_exceptions_preserve_raw_payload_guard(tmp_path: Path) -> None:
    run = _basic_run(tmp_path)
    _write_json(
        run / "metrics.json",
        {
            "target_volume_reads": 0,
            "prediction_count": 1,
            "nested": {"observation_reads": 2},
        },
    )
    manifest = package_evidence([run], tmp_path / "scalar-counters")
    assert any(item["source_path"] == "metrics.json" for item in manifest["included"])

    # The original matcher must continue to reject prefixed/nested raw names;
    # only the exact finite scalar counters above are exempt.
    _write_json(run / "metrics.json", {"nested_raw_target_array": [1]})
    with pytest.raises(UnsafeEvidenceError, match="raw target/image"):
        package_evidence([run], tmp_path / "prefixed-raw")

    # An array under an exempt counter key is still unsafe and cannot smuggle
    # a target/volume payload through the scalar exception.
    _write_json(run / "metrics.json", {"target_volume_reads": [0]})
    with pytest.raises(UnsafeEvidenceError, match="invalid scalar counter"):
        package_evidence([run], tmp_path / "counter-array")


def test_full_split_role_id_lists_use_narrow_large_bounds(tmp_path: Path) -> None:
    run = _basic_run(tmp_path)
    subject_ids = [f"BraTS2021_{index:05d}" for index in range(1251)]
    split = deterministic_subject_split(subject_ids, seed=12)
    roles = build_training_role_manifest(split, engineering_only=False)
    _write_json(
        run / "split.json",
        split.to_dict(),
    )
    _write_json(run / "roles.json", roles.as_dict())
    manifest = package_evidence([run], tmp_path / "large-split-role")
    included_names = {Path(item["source_path"]).name for item in manifest["included"]}
    assert {"split.json", "roles.json"} <= included_names


def test_oracle_r4_and_benchmark_rows_use_exact_bounded_schemas(tmp_path: Path) -> None:
    run = _basic_run(tmp_path)
    (run / "privileged_oracle.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "pfgr-lite-oracle-result-v1",
                "subject_id": "s0",
                "rows": [
                    {
                        "subject_id": "s0",
                        "action_id": "a0",
                        "raw_gain": 0.1,
                        "true_gain": 0.1,
                        "selected": True,
                    }
                ],
                "confirmation": [],
                "selected_action_ids": ["a0"],
                "oracle_final_prediction_decoded": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "rows.jsonl").write_text(
        json.dumps(
            {
                "voxel_count": 4,
                "query_error_max": 0.0,
                "prediction_error_max": 0.0,
                "gain_error_max": None,
                "sampling_law": "exact_union_v1",
                "sampling_seed": None,
                "query_draws": 4,
                "candidate_batch_size": 1,
                "candidate_batch_scope": "single_action_serial",
                "cache_scope": "lattice_query_cache_only",
                "optimized_gain": None,
                "reference_gain": None,
                "shared_before_elapsed_seconds": 0.0,
                "optimized_elapsed_seconds": 0.0,
                "reference_elapsed_seconds": 0.0,
                "parity_failure": None,
                "dtype": "torch.float32",
                "query_calls": {"shared_before": 1},
                "decoder_calls": {"shared_before": 1},
                "decoded_outputs": {"optimized": 1},
                "stored_action_reused": True,
                "reference_rebased_action": False,
                "full_clone_bytes": 0,
                "optimized_clone_bytes": 0,
                "subject_id": "s0",
                "case_index": 0,
                "repeat": 0,
                "cache_state": "cold_lattice_query_cache",
                "cache_reset": True,
                "cache_reset_scope": "lattice_query_cache_only",
                "footprint_build_elapsed_seconds": 0.0,
                "elapsed_seconds": 0.0,
                "allocated_memory_bytes": None,
                "reserved_memory_bytes": None,
                "device": "cpu",
                "state_version": 0,
                "action_id": "a0",
                "effective_policy_hash": None,
                "lattice_counter_delta": {},
                "sampling_probability_digest": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        run / "r4-paired.json",
        {
            "schema_version": "pfgr-lite-metrics-v1",
            "subjects": ["s0"],
            "pairwise": {
                "oracle_vs_z0": {"values": [0.1], "decision": {"decision": "INCONCLUSIVE"}},
            },
            "headroom": {"oracle_vs_z0": [0.1], "recovery_values": []},
            "stop_aware": {"route_gap_values": [], "top1_regret_values": []},
        },
    )
    manifest = package_evidence([run], tmp_path / "diagnostic-schemas")
    included_names = {Path(item["source_path"]).name for item in manifest["included"]}
    assert {"privileged_oracle.jsonl", "rows.jsonl", "r4-paired.json"} <= included_names

    _write_json(run / "r4-paired.json", {"pairwise": {"oracle_vs_z0": {"values": [[0.1]]}}})
    with pytest.raises(UnsafeEvidenceError, match="unrecognised array|numeric vector"):
        package_evidence([run], tmp_path / "r4-raw-array")

    _write_json(
        run / "r4-paired.json",
        {
            "subjects": ["s0"],
            "pairwise": {"oracle_vs_z0": {"values": [0.1]}},
            "headroom": {"oracle_vs_z0": [0.1], "recovery_values": []},
            "stop_aware": {"route_gap_values": [], "top1_regret_values": []},
        },
    )
    (run / "rows.jsonl").write_text(
        json.dumps({"subject_id": "s0", "descriptor": [0.0]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(UnsafeEvidenceError, match="unsupported keys"):
        package_evidence([run], tmp_path / "benchmark-descriptor")


def test_comparison_subject_secrets_and_decoder_counter_payloads_fail_closed(
    tmp_path: Path,
) -> None:
    run = _basic_run(tmp_path)
    _write_json(run / "r4-paired.json", {"subjects": ["api_key=synthetic_not_a_real_key"]})
    with pytest.raises(UnsafeEvidenceError, match="credential-like string"):
        package_evidence([run], tmp_path / "comparison-secret")

    _write_json(run / "r4-paired.json", {"subjects": ["s0"]})
    _write_json(
        run / "benchmark.json",
        {"decoder_calls": {"descriptor": [1, 2, 3]}},
    )
    with pytest.raises(UnsafeEvidenceError, match="invalid scalar counter|unsafe decoder counter"):
        package_evidence([run], tmp_path / "decoder-array")


def test_paired_csv_is_allowlisted_but_raw_columns_are_rejected(tmp_path: Path) -> None:
    run = _basic_run(tmp_path)
    (run / "paired_subjects.csv").write_text(
        "subject_id,mae\ns1,0.2\n", encoding="utf-8"
    )
    manifest = package_evidence([run], tmp_path / "csv-package")
    assert any(
        item["source_path"] == "paired_subjects.csv" for item in manifest["included"]
    )

    (run / "paired_action_rows.csv").write_text(
        "subject_id,target\ns1,0.2\n", encoding="utf-8"
    )
    with pytest.raises(UnsafeEvidenceError, match="raw target"):
        package_evidence([run], tmp_path / "csv-unsafe")


def test_archive_bytes_are_deterministic(tmp_path: Path) -> None:
    run = _basic_run(tmp_path)
    first = package_evidence([run], tmp_path / "first")
    second = package_evidence([run], tmp_path / "second")
    assert (tmp_path / "first" / "evidence.zip").read_bytes() == (
        tmp_path / "second" / "evidence.zip"
    ).read_bytes()
    assert first["included"] == second["included"]


def test_same_filenames_from_multiple_runs_get_collision_safe_paths(
    tmp_path: Path,
) -> None:
    run_a = _basic_run(tmp_path / "a", "same")
    run_b = _basic_run(tmp_path / "b", "same")
    output = tmp_path / "package"
    manifest = package_evidence([run_b, run_a], output)
    archive_paths = [row["archive_path"] for row in manifest["included"]]
    assert len(archive_paths) == len(set(archive_paths))
    assert len({path.split("/")[1] for path in archive_paths}) == 2


def test_required_evidence_is_denominated_per_run_without_union_borrowing(
    tmp_path: Path,
) -> None:
    complete = _basic_run(tmp_path / "complete", "complete")
    partial = tmp_path / "partial" / "partial"
    partial.mkdir(parents=True)
    _write_json(partial / "receipt.json", {"schema": "synthetic-v1"})
    manifest = package_evidence([partial, complete], tmp_path / "per-run")
    assert manifest["evidence_status"] == "MISSING_REQUIRED"
    required = manifest["required_evidence"]
    assert required["denominator_runs"] == 2
    assert required["complete_runs"] == 1
    per_run = {entry["run"]: entry for entry in required["per_run"]}
    assert per_run["partial"]["missing"]
    assert per_run["complete"]["missing"] == []


def test_existing_destination_is_refused_even_when_empty(tmp_path: Path) -> None:
    run = _basic_run(tmp_path)
    destination = tmp_path / "already-there"
    destination.mkdir()
    with pytest.raises(DestinationExistsError):
        package_evidence([run], destination)
    assert not (destination / "manifest.json").exists()


def test_destination_inside_input_run_is_rejected(tmp_path: Path) -> None:
    run = _basic_run(tmp_path)
    with pytest.raises(EvidencePathError, match="inside an input run"):
        package_evidence([run], run / "package")


def test_unknown_large_file_is_excluded_and_allowlisted_large_file_fails(
    tmp_path: Path,
) -> None:
    run = _basic_run(tmp_path)
    (run / "arbitrary.unknown").write_bytes(b"x" * 256)
    manifest = package_evidence([run], tmp_path / "unknown", max_file_size=128)
    assert any(
        item["path"] == "arbitrary.unknown"
        and item["reason"] == "unknown_or_not_whitelisted_oversized"
        for item in manifest["exclusions"]
    )

    (run / "metrics.json").write_text(
        '{"value": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}\n',
        encoding="utf-8",
    )
    with pytest.raises(EvidenceValidationError, match="max_file_size"):
        package_evidence([run], tmp_path / "oversized", max_file_size=128)
    assert not (tmp_path / "oversized").exists()


def test_public_api_has_only_canonical_optional_limits_and_manifest_fields(
    tmp_path: Path,
) -> None:
    parameters = inspect.signature(package_evidence).parameters
    assert set(parameters) == {
        "run_dirs",
        "destination",
        "max_file_size",
        "max_archive_size",
    }
    run = _basic_run(tmp_path)
    manifest = package_evidence([run], tmp_path / "canonical")
    assert set(manifest) == {
        "schema",
        "status",
        "evidence_status",
        "scientific_status",
        "scientific_claim",
        "archive",
        "limits",
        "runs",
        "included",
        "exclusions",
        "required_evidence",
        "counts",
        "exclusion_scope",
    }
    assert set(manifest["archive"]) == {"path", "sha256", "size_bytes", "format"}


def test_escape_symlink_is_rejected(tmp_path: Path) -> None:
    run = _basic_run(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    try:
        (run / "receipt-link.json").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(EvidencePathError):
        package_evidence([run], tmp_path / "package")


def test_forbidden_payloads_and_secret_files_are_excluded(tmp_path: Path) -> None:
    run = _basic_run(tmp_path)
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "resume.pt").write_bytes(b"checkpoint")
    (run / "predictions").mkdir()
    (run / "predictions" / "prediction.npy").write_bytes(b"prediction")
    (run / "patient.nii.gz").write_bytes(b"nifti")
    (run / ".env").write_text("API_KEY=secret\n", encoding="utf-8")
    (run / "raw_bank.json").write_text('{"target_values": [1, 2]}\n', encoding="utf-8")
    manifest = package_evidence([run], tmp_path / "package")
    paths = {item["path"] for item in manifest["exclusions"]}
    assert "patient.nii.gz" in paths
    assert ".env" in paths
    assert any(path.startswith("checkpoints") for path in paths)
    assert any(path.startswith("predictions") for path in paths)
    assert "raw_bank.json" in paths


@pytest.mark.parametrize(
    "payload",
    ['{"value": NaN}\n', '{"value": Infinity}\n', '{"value": -Infinity}\n', "{bad}\n"],
)
def test_malformed_or_nonfinite_allowlisted_json_fails_closed(
    tmp_path: Path, payload: str
) -> None:
    run = _basic_run(tmp_path)
    (run / "metrics.json").write_text(payload, encoding="utf-8")
    with pytest.raises(EvidenceValidationError):
        package_evidence([run], tmp_path / "package")
    assert not (tmp_path / "package").exists()


def test_known_secret_in_allowlisted_json_is_rejected(tmp_path: Path) -> None:
    run = _basic_run(tmp_path)
    _write_json(run / "wandb.json", {"run_id": "abc", "api_key": "do-not-package"})
    with pytest.raises(UnsafeEvidenceError, match="credential"):
        package_evidence([run], tmp_path / "package")


def test_only_first_traceback_is_included_and_empty_runs_are_explicit(
    tmp_path: Path,
) -> None:
    run = _basic_run(tmp_path)
    (run / "traceback_001.txt").write_text(
        "Traceback (most recent call last):\n", encoding="utf-8"
    )
    (run / "traceback_002.txt").write_text("second\n", encoding="utf-8")
    manifest = package_evidence([run], tmp_path / "package")
    included_paths = {item["source_path"] for item in manifest["included"]}
    assert "traceback_001.txt" in included_paths
    assert "traceback_002.txt" not in included_paths
    assert any(
        item["reason"] == "traceback_not_first" for item in manifest["exclusions"]
    )


def test_empty_run_reports_empty_without_scientific_claim(tmp_path: Path) -> None:
    run = tmp_path / "empty"
    run.mkdir()
    manifest = package_evidence([run], tmp_path / "package")
    assert manifest["status"] == "SOFTWARE_PASS"
    assert manifest["evidence_status"] == "EMPTY"
    assert manifest["scientific_status"] == "NOT_EVALUATED"
    assert manifest["required_evidence"]["per_run"][0]["missing"]
