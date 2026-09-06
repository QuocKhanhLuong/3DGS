"""CPU-only contract tests for the bounded PFGR-Lite evidence package."""

from __future__ import annotations

import hashlib
import inspect
import json
import zipfile
from pathlib import Path

import pytest

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
