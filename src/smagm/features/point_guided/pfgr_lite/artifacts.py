"""Bounded, deterministic packaging of PFGR-Lite run evidence.

This module deliberately has no imports from the PFGR model, data, teacher,
bank, or calibration implementations.  It is a small standard-library only
boundary used by the future ``pfgr_lite package`` command.  Packaging is
allow-list based: a file is never included merely because it happens to be a
JSON document.  Source files are copied byte-for-byte into a deterministic,
stored (uncompressed) ZIP archive and a side-car manifest records the source
relative path, archive path, SHA-256 digest and size.

The helper is evidence packaging, not an experiment evaluator.  A successful
package therefore reports ``status=SOFTWARE_PASS`` and always reports
``scientific_status=NOT_EVALUATED``.  Empty or incomplete runs are successful
software operations with an explicit ``evidence_status`` and missing-evidence
list; they are never represented as a scientific pass.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = "pfgr-lite-evidence-v1"
"""Stable manifest schema identifier."""

DEFAULT_MAX_FILE_SIZE = 8 * 1024 * 1024
DEFAULT_MAX_ARCHIVE_SIZE = 64 * 1024 * 1024


class EvidencePackagingError(ValueError):
    """Base error for invalid evidence inputs or unsafe payloads."""


class EvidencePathError(EvidencePackagingError):
    """Raised when a run path would escape its declared root."""


class EvidenceValidationError(EvidencePackagingError):
    """Raised when an allow-listed payload is malformed or unsafe."""


class UnsafeEvidenceError(EvidenceValidationError):
    """Raised when an allow-listed file contains known unsafe content."""


class DestinationExistsError(FileExistsError):
    """Raised when the requested package destination is not new."""


@dataclass(frozen=True, slots=True)
class _Candidate:
    """A validated allow-listed source file."""

    run_index: int
    run_identity: str
    source_root: Path
    source_path: Path
    relative_path: str
    archive_path: str
    category: str
    size_bytes: int
    source_stat: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class _Run:
    index: int
    identity: str
    root: Path
    display_name: str


# Names in this table are intentionally explicit.  Do not turn this into a
# broad ``*.json`` rule: arbitrary JSON can contain targets, images or secrets.
_EXACT_JSON_CATEGORIES: dict[str, str] = {
    "resolved_config.json": "resolved_config",
    "effective_policy.json": "effective_policy",
    "source.json": "source_provenance",
    "source_provenance.json": "source_provenance",
    "environment.json": "environment",
    "weights.json": "weights_provenance",
    "weights_provenance.json": "weights_provenance",
    "split.json": "split_manifest",
    "split_manifest.json": "split_manifest",
    "roles.json": "role_manifest",
    "role_manifest.json": "role_manifest",
    "receipt.json": "receipt",
    # Exact scalar/metadata handoffs emitted by the PFGR CLI services.
    # Keep these names explicit; stage_runtime and all tensor/checkpoint
    # payloads remain intentionally outside the package boundary.
    "service_receipt.json": "receipt",
    "bank_verify.json": "bank_index_metadata",
    "value_fit.json": "metrics_summary",
    "value_evaluate.json": "metrics_summary",
    "value_evaluate_pairs.json": "value_evaluation_pairs",
    "calibration_evidence.json": "calibration_manifest_metadata",
    "trace_receipts.json": "paired_action_rows",
    "collection_policy.json": "effective_policy",
    "fit_winners.json": "paired_action_rows",
    "allowance_winners.json": "paired_action_rows",
    "review_context.json": "provenance",
    "stage_state.json": "receipt",
    "resume_summary.json": "receipt",
    "r4-paired.json": "paired_comparison",
    "changes.json": "config_changes",
    "provenance.json": "provenance",
    "metrics.json": "metrics_summary",
    "metrics_summary.json": "metrics_summary",
    "metrics_history.json": "metrics_history",
    "paired_subjects.json": "paired_subject_rows",
    "action_metrics.json": "paired_action_rows",
    "paired_action_rows.json": "paired_action_rows",
    "benchmark.json": "teacher_benchmark",
    "teacher_benchmark.json": "teacher_benchmark",
    "parity.json": "teacher_parity",
    "teacher_parity.json": "teacher_parity",
    "index.json": "bank_index_metadata",
    "bank_index.json": "bank_index_metadata",
    "manifest.json": "calibration_manifest_metadata",
    "calibration.json": "calibration_metadata",
    "calibration_manifest.json": "calibration_manifest_metadata",
    "wandb.json": "wandb_run_metadata",
    "wandb_run.json": "wandb_run_metadata",
    "wandb_run_id_url.json": "wandb_run_metadata",
    "command.json": "command",
    "argv.json": "command",
    "exit.json": "exit",
    "exit_code.json": "exit",
    "command_exit.json": "command_exit",
}

_EXACT_JSONL_CATEGORIES: dict[str, str] = {
    "metrics.jsonl": "metrics_history",
    "metrics_history.jsonl": "metrics_history",
    "paired_subjects.jsonl": "paired_subject_rows",
    "paired_subject_rows.jsonl": "paired_subject_rows",
    "paired_actions.jsonl": "paired_action_rows",
    "paired_action_rows.jsonl": "paired_action_rows",
    "action_metrics.jsonl": "paired_action_rows",
    "benchmark.jsonl": "teacher_benchmark",
    "teacher_benchmark.jsonl": "teacher_benchmark",
    "parity.jsonl": "teacher_parity",
    "teacher_parity.jsonl": "teacher_parity",
    "privileged_oracle.jsonl": "oracle_diagnostic",
    "rows.jsonl": "benchmark_rows",
}

_EXACT_CSV_CATEGORIES: dict[str, str] = {
    "metrics.csv": "metrics_summary",
    "metrics_history.csv": "metrics_history",
    "paired_subjects.csv": "paired_subject_rows",
    "paired_subject_rows.csv": "paired_subject_rows",
    "paired_actions.csv": "paired_action_rows",
    "paired_action_rows.csv": "paired_action_rows",
    "action_metrics.csv": "paired_action_rows",
    "benchmark.csv": "teacher_benchmark",
    "teacher_benchmark.csv": "teacher_benchmark",
    "parity.csv": "teacher_parity",
    "teacher_parity.csv": "teacher_parity",
}

_EXACT_TEXT_CATEGORIES: dict[str, str] = {
    "test_output.txt": "test_output",
    "pytest.txt": "test_output",
    "tests.txt": "test_output",
    "tests_output.txt": "test_output",
    "pytest_output.txt": "test_output",
    "test.log": "test_output",
    "tests.log": "test_output",
    "stdout.txt": "test_output",
    "stderr.txt": "test_output",
    "command.txt": "command",
    "argv.txt": "command",
    "exit.txt": "exit",
    "exit_code.txt": "exit",
    "command_exit.txt": "command_exit",
    "wandb_run_id.txt": "wandb_run_metadata",
    "wandb_url.txt": "wandb_run_metadata",
    "wandb_run_url.txt": "wandb_run_metadata",
    "wandb_run_id_url.txt": "wandb_run_metadata",
    "traceback.txt": "traceback",
    "traceback.log": "traceback",
    "first_traceback.txt": "traceback",
    "first_traceback.log": "traceback",
}

_TRACEBACK_RE = re.compile(r"^traceback(?:[_-].+)?\.(?:txt|log|out)$", re.IGNORECASE)
_TEST_OUTPUT_RE = re.compile(r"^(?:pytest|test)[_-].+\.(?:txt|log|out)$", re.IGNORECASE)
_COMMAND_RE = re.compile(
    r"^(?:exact[_-])?command(?:[_-]?(?:argv|exit|code))?\.(?:txt|log|json)$",
    re.IGNORECASE,
)

_FORBIDDEN_DIR_NAMES = {
    ".cache",
    "cache",
    "caches",
    "checkpoint",
    "checkpoints",
    "models",
    "predictions",
    "prediction",
    "images",
    "image",
    "volumes",
    "volume",
    "targets",
    "target",
    "raw_targets",
    "raw-targets",
    "banks",
    "bank_shards",
    "shards",
    "nifti",
    "dicom",
    "wandb-cache",
}

_FORBIDDEN_SUFFIXES = (
    ".nii",
    ".nii.gz",
    ".dcm",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".pkl",
    ".pickle",
    ".h5",
    ".hdf5",
    ".onnx",
    ".bin",
    ".pem",
    ".key",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".mha",
    ".nrrd",
)

_FORBIDDEN_BASENAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "id_rsa",
    "id_ed25519",
    "credentials",
    "credentials.json",
    "secrets.json",
    "secret.json",
    "api_key",
    "api_key.txt",
    "token.txt",
    "model.pt",
    "model.pth",
    "resume.pt",
    "resume.pth",
    "checkpoint.pt",
    "checkpoint.pth",
    "secret.txt",
    "secrets.txt",
    "keys.txt",
    "predictions.json",
    "prediction.json",
    "images.json",
    "image.json",
    "volumes.json",
    "volume.json",
    "target.json",
    "raw_bank.json",
    "raw_targets.json",
}

# Key names are checked structurally in parsed JSON.  The matcher is
# deliberately conservative and does not claim perfect secret detection in
# arbitrary text; the allow-list remains the primary boundary.
_SENSITIVE_KEY_RE = re.compile(
    r"(?:^|_)(?:secret|password|passwd|token|api[_-]?key|access[_-]?key|access[_-]?token|"
    r"auth(?:orization)?|credential|private[_-]?key|client[_-]?secret|wandb[_-]?api[_-]?key|"
    r"aws[_-]?(?:secret|access)[_-]?key|hf[_-]?token)(?:$|_)",
    re.IGNORECASE,
)
# Raw payload keys are rejected structurally.  The matcher intentionally
# catches prefixed/nested variants (for example ``metrics.raw_target_array``)
# rather than relying on a basename-only filename check.  Explicit scalar
# counter exceptions are validated below before this matcher runs.
_RAW_PAYLOAD_KEY_RE = re.compile(
    r"(?:^|_)(?:raw[_-]?(?:target|image|volume|array|data)|target[_-]?(?:values?|array|volume|image|data)|"
    r"ground[_-]?truth|prediction(?:s)?|logits?|(?:nifti|dicom)|ndarray|"
    r"tensor(?:$|[_-](?:array|data|values|payload))|voxel[_-]?(?:array|data)|"
    r"image[_-]?(?:array|data)|volume[_-]?(?:array|data)|pixels?|pixel[_-]?(?:array|data)|"
    r"(?:encoded|base64|blob|payload)(?:$|[_-](?:array|data|bytes|values)))(?:$|_)",
    re.IGNORECASE,
)

# A few CLI receipts use names that contain the raw-payload tokens above but
# carry only a scalar counter (or ``null`` when unavailable).  Keep this list
# exact and validate values as finite, non-negative counts before bypassing
# the raw-payload matcher.  Arrays/objects under these keys remain rejected.
_SAFE_SCALAR_COUNTER_KEYS = {
    "target_reads",
    "target_volume_reads",
    "segmentation_reads",
    "observation_reads",
    "prediction_count",
    "teacher_calls",
    "decoder_calls",
    "medicalnet_traversals",
    "mri_loader_calls",
    "updater_calls",
    "target_validations",
    "value_evaluations",
    "candidate_evaluations",
    # Exact scalar benchmark/oracle metadata whose names contain the raw
    # ``prediction`` token but whose values are bounded counters/errors, not
    # tensors or image payloads.
    "prediction_error_max",
    "prediction_tolerance_atol",
    "prediction_tolerance_rtol",
}
_SAFE_BOOLEAN_METADATA_KEYS = {"oracle_final_prediction_decoded"}
# Benchmark parity emits ``decoder_calls`` as a bounded mapping of phase to
# scalar counts, while service receipts may emit the same name as one scalar.
# Admit this exact mapping shape through normal structural recursion; all
# other raw-token exceptions remain scalar-only.
_SAFE_COUNTER_MAPPING_KEYS = {"decoder_calls"}
_KNOWN_SECRET_TEXT_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|client[_-]?secret|password|passwd|token)\s*[:=]\s*[^\s]+"
    ),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)

_REQUIRED_EVIDENCE = (
    "receipt.json",
    "resolved_config.json",
    "effective_policy.json",
    "source.json",
    "environment.json",
    "weights.json",
)

# JSON metadata is intentionally structural rather than extension-only.  A
# finite vector is permitted only where the artifact contract declares its
# shape; arbitrary nested numeric arrays (for example ``pixels`` or an
# alternate-key volume payload) are rejected.  Bounds are deliberately small
# enough for metadata and cannot be used to smuggle a tensor into a manifest.
_NUMERIC_VECTOR_LIMITS = {
    "descriptor": 24,
    "descriptor_q": 24,
    "q": 24,
    "f_spec": 168,
    "f_spec_xy": 56,
    "f_spec_xz": 56,
    "f_spec_yz": 56,
    "plane_feature": 56,
    "plane_features": 56,
    "delta": 3,
    "deltas": 3,
    "action_delta": 3,
    "predicted_delta": 3,
    "selected_delta": 3,
    "displacement": 3,
    "offset": 3,
    "point": 3,
    "center": 3,
    "centre": 3,
    "voxel_id": 3,
    "voxel_ids": 3,
    "normal": 3,
    "reliability": 3,
    "k_bins": 5,
    "k_counts": 5,
    "stop_counts": 5,
    "shape": 5,
    "grid_shape": 5,
    "volume_shape": 5,
    "feature_shape": 5,
    "seeds": 32,
    # Small, declared configuration/receipt vectors emitted by the PFGR CLI.
    # These are metadata scalars, not descriptor/volume payloads.
    "budgets": 8,
    "hidden_channels": 8,
    "input_variants": 8,
    "range": 4,
    "directional_offsets_mm": 3,
    "state_versions": 5,
    "measurement_q_draws": 8,
    "selected_point_ids": 64,
    "split_fractions": 3,
    "group_counts": 8,
}
_AFFINE_KEYS = {"affine", "affine4x4", "affine_mm", "voxel_to_ras"}
_MAPPING_LIST_KEYS = {
    "rows",
    "history",
    "records",
    "subjects",
    "actions",
    "paired_subjects",
    "paired_actions",
    "included",
    "exclusions",
    "runs",
    "stages",
    "metrics",
    "events",
    "labels",
    "candidates",
    "candidate_rows",
    "subject_rows",
    "action_rows",
    "samples",
    "queries",
    "states",
    "steps",
    "inputs",
    "per_run",
    "dependencies",
    "per_subject",
    "candidate_evaluations",
    "eligible_candidate_evaluations",
    # Concrete CLI service/verification/calibration metadata rows.
    "selection_receipts",
    "subject_context_bindings",
    "subject_contexts",
    "completed_trace_receipts",
    "shards",
    "policies",
    "confirmation",
    "contexts",
}
_STRING_LIST_KEYS = {
    "argv",
    "args",
    "command_args",
    "modalities",
    "roles",
    "devices",
    "paths",
    "source_paths",
    "files",
    "names",
    "expected",
    "present",
    "missing",
    "subject_ids",
    "all_subject_ids",
    "excluded_subject_ids",
    "val_subject_ids",
    "selected_subject_ids",
    "subject_names",
    "train_subjects",
    "validation_subjects",
    "test_subjects",
    "deployment_subjects",
    "train",
    "val",
    "validation",
    "test",
    "deployment",
    # Concrete CLI service/verification/calibration metadata vectors.
    "shard_hashes",
    "selected_replay_refs",
    "source_scope_roots",
    "authorized_modules",
    "subject_ids",
    "baseline_test_subject_ids",
    "baseline_train_subject_ids",
    "baseline_validation_subject_ids",
    "calibration_allowance_subject_ids",
    "calibration_fit_subject_ids",
    "producer_fit_subject_ids",
    "producer_fit_subjects",
    "fit_subjects",
    "allowance_subjects",
    "train_subject_ids",
    "validation_subject_ids",
    "test_subject_ids",
    "completed_trace_hashes",
    "selected_action_ids",
    "measurement_modes",
    "proposal_digests",
    "action_digests",
    "optimizer_groups",
    "restored_rng_streams",
    "expected_per_run",
    "parameter_names",
    "changed_parameter_groups",
    "changed_parameter_names",
    "optimizer_parameter_names",
    "input_modalities",
    "forbidden_imports_during_fit",
    "forbidden_imports_during_eval",
    "reasons",
    "missing_inputs",
    "cold_warm_labels",
    "actual_devices",
    "sample_ids",
}

# A few evidence schemas intentionally use bounded tuples represented as JSON
# arrays.  They are not arbitrary nested arrays: each row has an explicit
# element type/length contract and is validated before packaging.
_STRING_PAIR_LIST_LIMITS = {
    "component_versions": 64,
    # Full split/role manifests can contain the untouched BraTS population;
    # retain a narrow per-key bound rather than applying this to arbitrary
    # arrays.
    "subject_group_ids": 16384,
    "details": 64,
}
_STRING_LIST_LIMITS = {
    "baseline_train_subject_ids": 16384,
    "baseline_validation_subject_ids": 16384,
    "baseline_test_subject_ids": 16384,
    "all_subject_ids": 16384,
    "excluded_subject_ids": 16384,
    "val_subject_ids": 16384,
    "calibration_allowance_subject_ids": 16384,
    "calibration_fit_subject_ids": 16384,
    "producer_fit_subject_ids": 16384,
    "producer_fit_subjects": 16384,
    "fit_subjects": 16384,
    "allowance_subjects": 16384,
    "train_subject_ids": 16384,
    "validation_subject_ids": 16384,
    "test_subject_ids": 16384,
    "train": 16384,
    "val": 16384,
    "validation": 16384,
    "test": 16384,
}
_TYPED_ROW_SCHEMAS: dict[str, tuple[str, ...]] = {
    "trace_subject_bindings": ("str", "str", "str"),
    "winner_bindings": ("str", "str", "str", "str", "int"),
    "winner_confirmations": (
        "str",
        "str",
        "str",
        "str",
        "str",
        "int",
        "int_or_none",
        "float_or_none",
        "str",
    ),
}
_TYPED_ROW_LIMITS = {
    "trace_subject_bindings": 16384,
    "winner_bindings": 16384,
    "winner_confirmations": 16384,
}

# R4 comparisons publish bounded scalar series under these exact keys.  The
# generic metadata validator intentionally rejects an unknown list; these
# vectors are admitted only for the explicitly named comparison artifact.
_COMPARISON_NUMERIC_VECTOR_LIMITS = {
    "values": 16384,
    "oracle_vs_z0": 16384,
    "oracle_vs_random": 16384,
    "learned_vs_random": 16384,
    "random_vs_z0": 16384,
    "learned_vs_z0": 16384,
    "recovery_values": 16384,
    "route_gap_values": 16384,
    "top1_regret_values": 16384,
}

# A benchmark row is a scalar parity record, not a feature/action payload.
# Keep this schema in sync with benchmark._one_parity_case/run_teacher_benchmark
# and reject descriptors/proposals/planes rather than relying on a generic
# numeric-vector allow-list.
_BENCHMARK_ROW_KEYS = {
    "voxel_count",
    "query_error_max",
    "prediction_error_max",
    "gain_error_max",
    "charbonnier_epsilon",
    "sampling_law",
    "sampling_seed",
    "query_draws",
    "candidate_batch_size",
    "candidate_batch_scope",
    "cache_scope",
    "optimized_gain",
    "reference_gain",
    "shared_before_elapsed_seconds",
    "optimized_elapsed_seconds",
    "reference_elapsed_seconds",
    "parity_failure",
    "dtype",
    "query_calls",
    "decoder_calls",
    "decoded_outputs",
    "stored_action_reused",
    "reference_rebased_action",
    "full_clone_bytes",
    "optimized_clone_bytes",
    "subject_id",
    "case_index",
    "repeat",
    "cache_state",
    "cache_reset",
    "cache_reset_scope",
    "footprint_build_elapsed_seconds",
    "elapsed_seconds",
    "allocated_memory_bytes",
    "reserved_memory_bytes",
    "device",
    "state_version",
    "action_id",
    "effective_policy_hash",
    "lattice_counter_delta",
    "sampling_probability_digest",
    "full_pipeline_elapsed_seconds",
    "pipeline_counters",
    "pipeline_counter_scope",
}

# Value-evaluation paired rows retain only signed gains/predictions/ranks,
# identifiers, and producer/bank provenance hashes.  In particular, a
# descriptor/proposal/plane payload under rows is never evidence metadata.
_VALUE_PAIR_ROW_KEYS = {
    "action_id",
    "bank_manifest_hash",
    "context_id",
    "group_key",
    "input_variant",
    "measured_rank",
    "measured_raw_gain",
    "offset",
    "point_id",
    "predicted_rank",
    "predicted_raw",
    "predicted_scaled",
    "proposal_hash",
    "row_hash",
    "row_id",
    "row_key",
    "shard",
    "state_digest",
    "state_version",
    "subject_id",
}
_RAW_ARRAY_KEYS = {
    "array",
    "arrays",
    "data",
    "payload",
    "values",
    "bytes",
    "encoded",
    "base64",
    "blob",
    "pixels",
    "pixel",
    "volume",
    "image",
    "tensor",
    "ndarray",
}


def _normalise_limit(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer number of bytes")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer number of bytes")
    return value


def _normalise_run_dirs(
    run_dirs: Iterable[str | os.PathLike[str]] | str | os.PathLike[str],
) -> list[Path]:
    if isinstance(run_dirs, (str, os.PathLike)):
        raw_dirs: Sequence[str | os.PathLike[str]] = [run_dirs]
    else:
        raw_dirs = list(run_dirs)
    resolved: list[Path] = []
    for raw in raw_dirs:
        path = Path(raw).expanduser()
        if ".." in path.parts:
            raise EvidencePathError(
                f"path traversal components are not allowed in run artifact path: {path}"
            )
        if not path.exists():
            raise FileNotFoundError(f"run artifact directory does not exist: {path}")
        if path.is_symlink():
            target = path.resolve(strict=False)
            raise EvidencePathError(
                f"run artifact directory may not be a symlink: {path} -> {target}"
            )
        if not path.is_dir():
            raise NotADirectoryError(f"run artifact path is not a directory: {path}")
        resolved_path = path.resolve(strict=True)
        resolved.append(resolved_path)
    # Sorting avoids dependence on caller iteration order and therefore makes
    # archive bytes stable when the same set of run directories is supplied.
    return sorted(resolved, key=lambda item: item.as_posix())


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return slug or "run"


def _build_runs(run_dirs: Sequence[Path]) -> list[_Run]:
    runs: list[_Run] = []
    used: set[str] = set()
    for index, root in enumerate(run_dirs, start=1):
        display = root.name or "run"
        base = _slug(display)
        identity = base
        if identity in used:
            # A deterministic digest of the resolved source root keeps runs
            # with identical basenames distinct without leaking all path
            # components into the archive.
            digest = hashlib.sha256(root.as_posix().encode("utf-8")).hexdigest()[:10]
            identity = f"{base}-{digest}"
            suffix = 2
            while identity in used:
                identity = f"{base}-{digest}-{suffix}"
                suffix += 1
        used.add(identity)
        runs.append(
            _Run(index=index, identity=identity, root=root, display_name=display)
        )
    return runs


def _is_forbidden_name(path: Path) -> bool:
    name = path.name.lower()
    if name in _FORBIDDEN_BASENAMES:
        return True
    return any(name.endswith(suffix) for suffix in _FORBIDDEN_SUFFIXES)


def _forbidden_reason(path: Path, relative: PurePosixPath) -> str | None:
    components = {part.lower() for part in relative.parts[:-1]}
    if components & _FORBIDDEN_DIR_NAMES:
        return "forbidden_payload_directory"
    if _is_forbidden_name(path):
        return "forbidden_payload_type"
    if path.name.startswith(".") and path.name.lower() not in _EXACT_TEXT_CATEGORIES:
        return "hidden_or_secret_file"
    return None


def _classify(path: Path, relative: PurePosixPath) -> str | None:
    """Return an exact allow-list category, or ``None`` for unknown files."""

    # Never allow a file below a known payload/cache directory, even if its
    # basename resembles an otherwise safe metadata name.
    if {part.lower() for part in relative.parts[:-1]} & _FORBIDDEN_DIR_NAMES:
        return None
    name = path.name.lower()
    if name in _EXACT_JSON_CATEGORIES:
        return _EXACT_JSON_CATEGORIES[name]
    if name in _EXACT_JSONL_CATEGORIES:
        return _EXACT_JSONL_CATEGORIES[name]
    if name in _EXACT_CSV_CATEGORIES:
        return _EXACT_CSV_CATEGORIES[name]
    if name in _EXACT_TEXT_CATEGORIES:
        return _EXACT_TEXT_CATEGORIES[name]
    if _TRACEBACK_RE.fullmatch(name):
        return "traceback"
    if _TEST_OUTPUT_RE.fullmatch(name):
        return "test_output"
    if _COMMAND_RE.fullmatch(name):
        return "command"
    return None


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON scalar {value}")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _normalise_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")


def _validate_typed_metadata_rows(
    value: Sequence[object],
    *,
    source: Path,
    key_path: str,
    schema: Sequence[str],
    limit: int = 512,
) -> None:
    """Validate one of the explicitly declared tuple-row metadata lists."""

    if len(value) > limit:
        raise UnsafeEvidenceError(
            f"allow-listed JSON evidence {source} has too many metadata rows at '{key_path}'; "
            f"expected at most {limit}"
        )
    for row_index, row in enumerate(value):
        if not isinstance(row, (list, tuple)) or len(row) != len(schema):
            raise UnsafeEvidenceError(
                f"allow-listed JSON evidence {source} has invalid metadata row at '{key_path}[{row_index}]'; "
                f"expected a bounded sequence of length {len(schema)}"
            )
        for column_index, (item, expected) in enumerate(zip(row, schema)):
            valid = (
                expected == "str" and isinstance(item, str)
            ) or (
                expected == "int"
                and isinstance(item, int)
                and not isinstance(item, bool)
            ) or (
                expected == "int_or_none"
                and (
                    item is None
                    or (isinstance(item, int) and not isinstance(item, bool))
                )
            ) or (
                expected == "float_or_none"
                and (
                    item is None
                    or (
                        isinstance(item, (int, float))
                        and not isinstance(item, bool)
                        and math.isfinite(float(item))
                    )
                )
            )
            if not valid:
                raise UnsafeEvidenceError(
                    f"allow-listed JSON evidence {source} has invalid metadata value at "
                    f"'{key_path}[{row_index}][{column_index}]'; expected {expected}"
                )
            if isinstance(item, str):
                for pattern in _KNOWN_SECRET_TEXT_PATTERNS:
                    if pattern.search(item):
                        raise UnsafeEvidenceError(
                            f"allow-listed JSON evidence {source} contains a credential-like string at "
                            f"'{key_path}[{row_index}][{column_index}]'; remove the secret and retry "
                            "(arbitrary text cannot be perfectly scanned)"
                        )


def _validate_string_pairs(
    value: Sequence[object],
    *,
    source: Path,
    key_path: str,
    limit: int,
) -> None:
    if len(value) > limit or any(
        not isinstance(row, (list, tuple))
        or len(row) != 2
        or any(not isinstance(item, str) for item in row)
        for row in value
    ):
        raise UnsafeEvidenceError(
            f"allow-listed JSON evidence {source} has invalid metadata pairs at '{key_path}'; "
            f"expected at most {limit} string pairs"
        )
    for row_index, row in enumerate(value):
        for column_index, item in enumerate(row):
            for pattern in _KNOWN_SECRET_TEXT_PATTERNS:
                if pattern.search(item):
                    raise UnsafeEvidenceError(
                        f"allow-listed JSON evidence {source} contains a credential-like string at "
                        f"'{key_path}[{row_index}][{column_index}]'; remove the secret and retry "
                        "(arbitrary text cannot be perfectly scanned)"
                    )


def _validate_safe_scalar_counter(
    value: object,
    *,
    source: Path,
    key_path: str,
) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise UnsafeEvidenceError(
            f"allow-listed JSON evidence {source} has invalid scalar counter at '{key_path}'; "
            "expected a finite non-negative number or null"
        )


def _validate_safe_boolean_metadata(
    value: object,
    *,
    source: Path,
    key_path: str,
) -> None:
    if not isinstance(value, bool):
        raise UnsafeEvidenceError(
            f"allow-listed JSON evidence {source} has invalid boolean metadata at '{key_path}'; "
            "expected a JSON boolean"
        )


def _validate_safe_counter_mapping(
    value: Mapping[object, object],
    *,
    source: Path,
    key_path: str,
    category: str | None,
) -> None:
    """Validate the one bounded mapping form used by benchmark counters."""

    if len(value) > 16:
        raise UnsafeEvidenceError(
            f"allow-listed JSON evidence {source} has too many decoder counter entries at '{key_path}'; "
            "expected at most 16 finite scalar counters"
        )
    for key, child in value.items():
        normalised = _normalise_key(key)
        child_path = f"{key_path}.{key}"
        if _SENSITIVE_KEY_RE.search(normalised) or _RAW_PAYLOAD_KEY_RE.search(normalised):
            raise UnsafeEvidenceError(
                f"allow-listed evidence {source} contains unsafe decoder counter key '{child_path}'"
            )
        _validate_safe_scalar_counter(child, source=source, key_path=child_path)
        _validate_json_value(
            child,
            source=source,
            key_path=child_path,
            parent_key=normalised,
            category=category,
        )


def _validate_benchmark_row(
    value: Mapping[object, object],
    *,
    source: Path,
    key_path: str,
) -> None:
    """Validate one exact ``rows.jsonl`` benchmark parity row.

    Benchmark rows are emitted by the teacher parity service and contain only
    scalar timings/errors, bounded counter mappings, and stable IDs.  A strict
    top-level key set prevents a future descriptor/proposal/plane field from
    becoming packageable merely because it happens to be JSON-compatible.
    """

    unknown = sorted(
        str(key)
        for key in value
        if _normalise_key(key) not in _BENCHMARK_ROW_KEYS
    )
    if unknown:
        raise UnsafeEvidenceError(
            f"allow-listed benchmark rows {source} contain unsupported keys at '{key_path}': {unknown}; "
            "only bounded scalar parity metadata is packageable"
        )


def _validate_value_pair_row(
    value: Mapping[object, object],
    *,
    source: Path,
    key_path: str,
) -> None:
    """Validate one same-bank ValueNet evaluation row.

    The row is deliberately limited to signed measured/predicted gains,
    ranks, IDs, and immutable provenance hashes; action descriptors,
    proposals, planes, and tensors are not evidence metadata.
    """

    unknown = sorted(
        str(key)
        for key in value
        if _normalise_key(key) not in _VALUE_PAIR_ROW_KEYS
    )
    if unknown:
        raise UnsafeEvidenceError(
            f"allow-listed value-evaluation rows {source} contain unsupported keys at '{key_path}': {unknown}; "
            "descriptors/proposals/planes are never packaged"
        )


def _validate_json_value(
    value: object,
    *,
    source: Path,
    key_path: str = "",
    parent_key: str | None = None,
    category: str | None = None,
) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise EvidenceValidationError(
            f"non-finite JSON value in allow-listed evidence {source}: {key_path}"
        )
    if isinstance(value, str):
        for pattern in _KNOWN_SECRET_TEXT_PATTERNS:
            if pattern.search(value):
                raise UnsafeEvidenceError(
                    f"allow-listed JSON evidence {source} contains a credential-like string at '{key_path}'; "
                    "remove the secret and retry (arbitrary text cannot be perfectly scanned)"
                )
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalised = _normalise_key(key)
            child_path = f"{key_path}.{key}" if key_path else str(key)
            if _SENSITIVE_KEY_RE.search(normalised):
                raise UnsafeEvidenceError(
                    f"allow-listed evidence {source} contains credential-like key '{child_path}'; "
                    "remove the secret and retry"
                )
            if normalised in _SAFE_SCALAR_COUNTER_KEYS:
                if normalised in _SAFE_COUNTER_MAPPING_KEYS and isinstance(child, Mapping):
                    _validate_safe_counter_mapping(
                        child,
                        source=source,
                        key_path=child_path,
                        category=category,
                    )
                    continue
                _validate_safe_scalar_counter(
                    child,
                    source=source,
                    key_path=child_path,
                )
                # This exact scalar counter is intentionally exempt from the
                # raw-payload token matcher; nested arrays/objects fail closed
                # in the helper above, and arbitrary names remain rejected.
                _validate_json_value(
                    child,
                    source=source,
                    key_path=child_path,
                    parent_key=normalised,
                    category=category,
                )
                continue
            if normalised in _SAFE_BOOLEAN_METADATA_KEYS:
                _validate_safe_boolean_metadata(
                    child,
                    source=source,
                    key_path=child_path,
                )
                continue
            if _RAW_PAYLOAD_KEY_RE.search(normalised):
                raise UnsafeEvidenceError(
                    f"allow-listed evidence {source} contains raw target/image payload key '{child_path}'; "
                    "raw targets, volumes and predictions are never packaged"
                )
            if normalised in {
                "target",
                "t1ce",
                "ground_truth",
                "image",
                "volume",
            } and not isinstance(child, str):
                raise UnsafeEvidenceError(
                    f"allow-listed evidence {source} contains raw target/image payload key '{child_path}'; "
                    "raw targets, volumes and predictions are never packaged"
                )
            if normalised in {
                "checkpoint",
                "checkpoints",
                "resume",
                "resume_checkpoint",
                "state_dict",
                "model_state",
            } and isinstance(child, (Mapping, list, tuple)):
                raise UnsafeEvidenceError(
                    f"allow-listed evidence {source} contains checkpoint/model payload key '{child_path}'; "
                    "checkpoint bytes and tensor state are never packaged"
                )
            _validate_json_value(
                child,
                source=source,
                key_path=child_path,
                parent_key=normalised,
                category=category,
            )
    elif isinstance(value, (list, tuple)):
        list_key = parent_key or ""
        if category == "value_evaluation_pairs" and list_key == "rows":
            if len(value) > 16384 or not all(isinstance(item, Mapping) for item in value):
                raise UnsafeEvidenceError(
                    f"allow-listed value-evaluation rows {source} has invalid rows at '{key_path}'; "
                    "expected at most 16384 object rows"
                )
            for index, child in enumerate(value):
                assert isinstance(child, Mapping)
                _validate_value_pair_row(
                    child,
                    source=source,
                    key_path=f"{key_path}[{index}]",
                )
                _validate_json_value(
                    child,
                    source=source,
                    key_path=f"{key_path}[{index}]",
                    category=category,
                )
            return
        if category == "paired_comparison":
            if list_key in _COMPARISON_NUMERIC_VECTOR_LIMITS:
                limit = _COMPARISON_NUMERIC_VECTOR_LIMITS[list_key]
                if len(value) > limit or not all(
                    isinstance(item, (int, float)) and not isinstance(item, bool)
                    for item in value
                ):
                    raise UnsafeEvidenceError(
                        f"allow-listed comparison evidence {source} has invalid or oversized numeric vector at '{key_path}'; "
                        f"expected at most {limit} finite scalar values"
                    )
                for index, item in enumerate(value):
                    if isinstance(item, float) and not math.isfinite(item):
                        raise EvidenceValidationError(
                            f"non-finite comparison vector value in {source}: {key_path}[{index}]"
                        )
                return
            if list_key == "subjects":
                if len(value) > 16384 or not all(isinstance(item, str) for item in value):
                    raise UnsafeEvidenceError(
                        f"allow-listed comparison evidence {source} has invalid subject IDs at '{key_path}'; "
                        "expected at most 16384 strings"
                    )
                for index, child in enumerate(value):
                    _validate_json_value(
                        child,
                        source=source,
                        key_path=f"{key_path}[{index}]",
                        parent_key=list_key,
                        category=category,
                    )
                return
        if list_key in _STRING_PAIR_LIST_LIMITS:
            _validate_string_pairs(
                value,
                source=source,
                key_path=key_path,
                limit=_STRING_PAIR_LIST_LIMITS[list_key],
            )
            return
        if list_key in _TYPED_ROW_SCHEMAS:
            _validate_typed_metadata_rows(
                value,
                source=source,
                key_path=key_path,
                schema=_TYPED_ROW_SCHEMAS[list_key],
                limit=_TYPED_ROW_LIMITS.get(list_key, 512),
            )
            return
        if list_key in _AFFINE_KEYS:
            rows = list(value)
            flat = rows and all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in rows
            )
            matrix = len(rows) == 4 and all(
                isinstance(row, (list, tuple))
                and len(row) == 4
                and all(
                    isinstance(item, (int, float)) and not isinstance(item, bool)
                    for item in row
                )
                for row in rows
            )
            if not ((flat and len(rows) == 16) or matrix):
                raise UnsafeEvidenceError(
                    f"allow-listed JSON evidence {source} has invalid affine shape at '{key_path}'; "
                    "expected a finite 4x4 matrix or flat length-16 vector"
                )
            for row_index, row in enumerate(rows):
                if isinstance(row, (list, tuple)):
                    for col_index, item in enumerate(row):
                        if isinstance(item, float) and not math.isfinite(item):
                            raise EvidenceValidationError(
                                f"non-finite affine value in {source}: {key_path}[{row_index}][{col_index}]"
                            )
            return
        if list_key in _NUMERIC_VECTOR_LIMITS:
            limit = _NUMERIC_VECTOR_LIMITS[list_key]
            if len(value) > limit or not all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in value
            ):
                raise UnsafeEvidenceError(
                    f"allow-listed JSON evidence {source} has invalid or oversized numeric vector at '{key_path}'; "
                    f"expected at most {limit} finite scalar values"
                )
            for index, item in enumerate(value):
                if isinstance(item, float) and not math.isfinite(item):
                    raise EvidenceValidationError(
                        f"non-finite vector value in {source}: {key_path}[{index}]"
                    )
            return
        if list_key in _MAPPING_LIST_KEYS:
            if not all(isinstance(item, Mapping) for item in value):
                raise UnsafeEvidenceError(
                    f"allow-listed JSON evidence {source} has non-object row in '{key_path}'"
                )
            for index, child in enumerate(value):
                _validate_json_value(
                    child,
                    source=source,
                    key_path=f"{key_path}[{index}]",
                    category=category,
                )
            return
        if list_key in _STRING_LIST_KEYS:
            limit = _STRING_LIST_LIMITS.get(list_key, 256)
            if len(value) > limit or not all(isinstance(item, str) for item in value):
                raise UnsafeEvidenceError(
                    f"allow-listed JSON evidence {source} has invalid string list at '{key_path}'; "
                    f"expected at most {limit} strings"
                )
            for index, child in enumerate(value):
                _validate_json_value(
                    child,
                    source=source,
                    key_path=f"{key_path}[{index}]",
                    parent_key=list_key,
                    category=category,
                )
            return
        # Top-level arrays are accepted only for row documents, and must be a
        # finite sequence of objects.  Any nested list with an unknown key is
        # ambiguous and therefore fail-closed rather than risk a raw payload.
        if parent_key is None and all(isinstance(item, Mapping) for item in value):
            for index, child in enumerate(value):
                _validate_json_value(
                    child,
                    source=source,
                    key_path=f"[{index}]",
                    category=category,
                )
            return
        raise UnsafeEvidenceError(
            f"allow-listed JSON evidence {source} contains an unrecognised array at '{key_path}'; "
            "only declared metadata vectors, affine4x4, bounded K bins and object rows are permitted"
        )
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        # json.loads already rejects NaN/Infinity through parse_constant; this
        # catches non-finite values when a custom decoder supplies them.
        if isinstance(value, float) and not math.isfinite(value):
            raise EvidenceValidationError(
                f"non-finite JSON value in {source}: {key_path}"
            )
    elif value is not None and not isinstance(value, (str, bool)):
        raise EvidenceValidationError(
            f"unsupported JSON value type {type(value).__name__} in allow-listed evidence {source}: {key_path}"
        )


def _load_json(path: Path, *, category: str) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceValidationError(
            f"allow-listed JSON evidence is not UTF-8: {path}"
        ) from error
    try:
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_strict_object,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise EvidenceValidationError(
            f"malformed or non-finite JSON in allow-listed evidence {path}: {error}"
        ) from error
    if category in {
        "resolved_config",
        "effective_policy",
        "source_provenance",
        "environment",
        "weights_provenance",
        "split_manifest",
        "role_manifest",
        "receipt",
        "config_changes",
        "provenance",
        "wandb_run_metadata",
        "command",
        "exit",
        "command_exit",
        "bank_index_metadata",
        "calibration_manifest_metadata",
        "calibration_metadata",
        "paired_comparison",
    } and not isinstance(value, Mapping):
        raise EvidenceValidationError(
            f"allow-listed metadata evidence must be a JSON object: {path}"
        )
    if category in {
        "metrics_summary",
        "metrics_history",
        "paired_subject_rows",
        "paired_action_rows",
        "value_evaluation_pairs",
        "teacher_benchmark",
        "teacher_parity",
    }:
        if not isinstance(value, (Mapping, list)):
            raise EvidenceValidationError(
                f"allow-listed metrics/row evidence must be a JSON object or array: {path}"
            )
        if isinstance(value, list) and any(
            not isinstance(row, Mapping) for row in value
        ):
            raise EvidenceValidationError(
                f"allow-listed metrics/row arrays must contain JSON objects: {path}"
            )
    _validate_json_value(value, source=path, category=category)
    return value


def _validate_jsonl(path: Path, *, category: str) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceValidationError(
            f"allow-listed JSONL evidence is not UTF-8: {path}"
        ) from error
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                parse_constant=_reject_constant,
                object_pairs_hook=_strict_object,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise EvidenceValidationError(
                f"malformed or non-finite JSONL in allow-listed evidence {path} line {line_number}: {error}"
            ) from error
        if not isinstance(value, Mapping):
            raise EvidenceValidationError(
                f"allow-listed JSONL rows must be objects: {path} line {line_number}"
            )
        key_path = f"line[{line_number}]"
        if category == "benchmark_rows":
            _validate_benchmark_row(value, source=path, key_path=key_path)
        _validate_json_value(
            value,
            source=path,
            key_path=key_path,
            category=category,
        )


def _validate_text(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceValidationError(
            f"allow-listed text evidence is not UTF-8: {path}"
        ) from error
    for pattern in _KNOWN_SECRET_TEXT_PATTERNS:
        if pattern.search(text):
            raise UnsafeEvidenceError(
                f"allow-listed text evidence {path} matches a known credential pattern; "
                "remove the secret and retry (arbitrary text cannot be perfectly scanned)"
            )


def _validate_csv(path: Path) -> None:
    """Validate allow-listed row CSVs without treating arbitrary CSV as safe."""

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceValidationError(
            f"allow-listed CSV evidence is not UTF-8: {path}"
        ) from error
    for pattern in _KNOWN_SECRET_TEXT_PATTERNS:
        if pattern.search(text):
            raise UnsafeEvidenceError(
                f"allow-listed CSV evidence {path} matches a known credential pattern; "
                "remove the secret and retry (arbitrary text cannot be perfectly scanned)"
            )
    try:
        reader = csv.DictReader(io.StringIO(text), strict=True)
        headers = [str(header) for header in (reader.fieldnames or [])]
        for header in headers:
            normalised = _normalise_key(header)
            if _SENSITIVE_KEY_RE.search(normalised) or _RAW_PAYLOAD_KEY_RE.search(
                normalised
            ):
                raise UnsafeEvidenceError(
                    f"allow-listed CSV evidence {path} contains unsafe column '{header}'; "
                    "remove credential/raw target or image columns"
                )
            if normalised in {"target", "t1ce", "ground_truth", "image", "volume"}:
                raise UnsafeEvidenceError(
                    f"allow-listed CSV evidence {path} contains raw target/image column '{header}'"
                )
        # Consume rows so malformed quoting is rejected by the stdlib parser.
        for _ in reader:
            pass
    except csv.Error as error:
        raise EvidenceValidationError(
            f"malformed CSV in allow-listed evidence {path}: {error}"
        ) from error


def _validate_payload(path: Path, category: str) -> None:
    if path.suffix.lower() == ".jsonl":
        _validate_jsonl(path, category=category)
    elif path.suffix.lower() == ".json":
        _load_json(path, category=category)
    elif path.suffix.lower() == ".csv":
        _validate_csv(path)
    else:
        _validate_text(path)


def _relative_safe(root: Path, path: Path) -> PurePosixPath:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise EvidencePathError(
            f"source path escapes run directory: {path} not below {root}"
        ) from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise EvidencePathError(f"unsafe relative source path: {relative}")
    if any("\x00" in part or "\\" in part for part in relative.parts):
        raise EvidencePathError(f"unsafe path separator in source path: {relative}")
    return PurePosixPath(*relative.parts)


def _stat_tuple(path: Path) -> tuple[int, int, int]:
    info = path.stat()
    return (int(info.st_size), int(info.st_mtime_ns), int(info.st_ino))


def _walk_run(
    run: _Run, *, max_file_size: int
) -> tuple[list[_Candidate], list[dict[str, object]]]:
    candidates: list[_Candidate] = []
    exclusions: list[dict[str, object]] = []
    traceback_seen = False

    # os.walk does not follow directory symlinks by default, but explicitly
    # reject both escaping and in-root symlinks so archive identity is clear.
    for current, dirs, files in os.walk(run.root, topdown=True, followlinks=False):
        current_path = Path(current)
        retained_dirs: list[str] = []
        for dirname in sorted(dirs):
            directory = current_path / dirname
            rel = _relative_safe(run.root, directory)
            if directory.is_symlink():
                target = directory.resolve(strict=False)
                try:
                    target.relative_to(run.root)
                except ValueError as error:
                    raise EvidencePathError(
                        f"symlink directory escapes run root: {directory} -> {target}"
                    ) from error
                raise EvidencePathError(
                    f"symlink directory is not allowed in run artifacts: {directory} -> {target}"
                )
            if dirname.lower() in _FORBIDDEN_DIR_NAMES:
                # Keep the directory and each member in the exclusion
                # manifest.  We do not read payload bytes, but traversing the
                # names gives the caller an exact exclusion inventory.
                exclusions.append(
                    {
                        "run": run.identity,
                        "path": rel.as_posix(),
                        "reason": "forbidden_payload_directory",
                    }
                )
            retained_dirs.append(dirname)
        dirs[:] = retained_dirs

        for filename in sorted(files):
            path = current_path / filename
            relative = _relative_safe(run.root, path)
            relative_text = relative.as_posix()
            if path.is_symlink():
                target = path.resolve(strict=False)
                try:
                    target.relative_to(run.root)
                except ValueError as error:
                    raise EvidencePathError(
                        f"symlink file escapes run root: {path} -> {target}"
                    ) from error
                raise EvidencePathError(
                    f"symlink file is not allowed in run artifacts: {path} -> {target}"
                )
            if not path.is_file():
                exclusions.append(
                    {
                        "run": run.identity,
                        "path": relative_text,
                        "reason": "not_regular_file",
                    }
                )
                continue

            category = _classify(path, relative)
            forbidden_reason = _forbidden_reason(path, relative)
            if category is None or forbidden_reason is not None:
                reason = forbidden_reason or "unknown_or_not_whitelisted"
                try:
                    unknown_size = int(path.stat().st_size)
                except OSError:
                    unknown_size = 0
                if unknown_size > max_file_size:
                    reason = f"{reason}_oversized"
                exclusions.append(
                    {
                        "run": run.identity,
                        "path": relative_text,
                        "reason": reason,
                        "size_bytes": unknown_size,
                    }
                )
                continue
            if category == "traceback":
                if traceback_seen:
                    exclusions.append(
                        {
                            "run": run.identity,
                            "path": relative_text,
                            "reason": "traceback_not_first",
                        }
                    )
                    continue
                traceback_seen = True
            size = int(path.stat().st_size)
            if size > max_file_size:
                raise EvidenceValidationError(
                    f"allow-listed evidence exceeds max_file_size ({size} > {max_file_size} bytes): {path}; "
                    "increase the explicit bound or remove the payload"
                )
            _validate_payload(path, category)
            source_stat = _stat_tuple(path)
            archive_path = f"runs/{run.identity}/{relative_text}"
            candidates.append(
                _Candidate(
                    run_index=run.index,
                    run_identity=run.identity,
                    source_root=run.root,
                    source_path=path,
                    relative_path=relative_text,
                    archive_path=archive_path,
                    category=category,
                    size_bytes=size,
                    source_stat=source_stat,
                )
            )
    return candidates, exclusions


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.create_version = 20
    info.extract_version = 20
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.flag_bits = 0x800  # UTF-8 member names
    info.extra = b""
    info.comment = b""
    return info


def _copy_to_archive(
    candidates: Sequence[_Candidate], archive_path: Path, *, max_archive_size: int
) -> list[dict[str, object]]:
    included: list[dict[str, object]] = []
    with zipfile.ZipFile(
        archive_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for candidate in sorted(candidates, key=lambda item: item.archive_path):
            # Re-stat immediately before reading and after copying to detect a
            # source mutation rather than silently packaging a mixed run.
            if _stat_tuple(candidate.source_path) != candidate.source_stat:
                raise EvidenceValidationError(
                    f"source changed while packaging: {candidate.source_path}; retry with immutable artifacts"
                )
            digest = hashlib.sha256()
            try:
                with (
                    candidate.source_path.open("rb") as source,
                    archive.open(_zip_info(candidate.archive_path), mode="w") as target,
                ):
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                        target.write(chunk)
            except OSError as error:
                raise EvidencePackagingError(
                    f"could not read allow-listed evidence {candidate.source_path}: {error}"
                ) from error
            if _stat_tuple(candidate.source_path) != candidate.source_stat:
                raise EvidenceValidationError(
                    f"source changed while packaging: {candidate.source_path}; retry with immutable artifacts"
                )
            included.append(
                {
                    "run": candidate.run_identity,
                    "source_path": candidate.relative_path,
                    "archive_path": candidate.archive_path,
                    "category": candidate.category,
                    "sha256": digest.hexdigest(),
                    "size_bytes": candidate.size_bytes,
                }
            )
            # Bound the resulting archive, including ZIP overhead.  This check
            # happens after each member so an oversized package is aborted as
            # soon as it is known, never silently truncated.
            if archive_path.stat().st_size > max_archive_size:
                raise EvidenceValidationError(
                    f"evidence archive exceeds max_archive_size ({archive_path.stat().st_size} > "
                    f"{max_archive_size} bytes); increase the explicit bound or narrow the allow-list"
                )
    if archive_path.stat().st_size > max_archive_size:
        raise EvidenceValidationError(
            f"evidence archive exceeds max_archive_size ({archive_path.stat().st_size} > {max_archive_size} bytes)"
        )
    return included


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_manifest(destination: Path, manifest: Mapping[str, object]) -> None:
    _write_json_atomic(destination / "manifest.json", manifest)


def package_evidence(
    run_dirs: Iterable[str | os.PathLike[str]] | str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    max_archive_size: int = DEFAULT_MAX_ARCHIVE_SIZE,
) -> dict[str, object]:
    """Package reviewed metadata/metrics/test evidence into a new directory.

    Parameters
    ----------
    run_dirs:
        One or more existing run-artifact directories.  Directory symlinks and
        symlink members are rejected to keep relative identity unambiguous.
    destination:
        A unique, not-yet-existing directory.  A race that creates it first is
        reported as :class:`DestinationExistsError`; existing content is never
        overwritten.
    max_file_size / max_archive_size:
        Positive byte bounds.  Allow-listed files over the first bound and a
        resulting archive over the second bound fail closed; unknown files are
        excluded with a reason and are never truncated.

    Returns
    -------
    dict
        The persisted manifest payload.  ``archive.path`` is destination-
        relative so repeated packaging of unchanged inputs is reproducible.
        ``scientific_status`` is always ``NOT_EVALUATED``.
    """

    file_limit = _normalise_limit(max_file_size, "max_file_size")
    archive_limit = _normalise_limit(max_archive_size, "max_archive_size")
    source_dirs = _normalise_run_dirs(run_dirs)
    runs = _build_runs(source_dirs)

    # Validate and fully inventory inputs before claiming the destination.  A
    # malformed allow-listed payload therefore cannot leave a partial output.
    candidates: list[_Candidate] = []
    exclusions: list[dict[str, object]] = []
    for run in runs:
        run_candidates, run_exclusions = _walk_run(run, max_file_size=file_limit)
        candidates.extend(run_candidates)
        exclusions.extend(run_exclusions)

    destination_path = Path(destination).expanduser()
    if ".." in destination_path.parts:
        raise EvidencePathError(
            f"path traversal components are not allowed in evidence destination: {destination_path}"
        )
    destination_resolved = destination_path.resolve(strict=False)
    for source_root in source_dirs:
        try:
            destination_resolved.relative_to(source_root)
        except ValueError:
            continue
        raise EvidencePathError(
            f"evidence destination must not be inside an input run directory: {destination_path}"
        )
    if destination_path.exists() or destination_path.is_symlink():
        raise DestinationExistsError(
            f"evidence destination already exists: {destination_path}; choose a new directory"
        )
    destination_parent = destination_path.parent
    destination_parent.mkdir(parents=True, exist_ok=True)
    created_destination = False
    try:
        # mkdir(exist_ok=False) is the race boundary.  No operation below can
        # replace an unrelated destination created by another process.
        destination_path.mkdir(mode=0o755, exist_ok=False)
        created_destination = True
        archive_path = destination_path / "evidence.zip"
        temporary_archive = destination_path / ".evidence.zip.tmp"
        included = _copy_to_archive(
            candidates, temporary_archive, max_archive_size=archive_limit
        )
        os.replace(temporary_archive, archive_path)
        archive_digest = _sha256_file(archive_path)
        archive_size = int(archive_path.stat().st_size)

        included_by_run: dict[str, set[str]] = {run.identity: set() for run in runs}
        for item in included:
            run_name = str(item["run"])
            included_by_run.setdefault(run_name, set()).add(
                Path(str(item["source_path"])).name.lower()
            )
        per_run_required: list[dict[str, object]] = []
        for run in runs:
            present_names = included_by_run.get(run.identity, set())
            missing_for_run = [
                name for name in _REQUIRED_EVIDENCE if name not in present_names
            ]
            per_run_required.append(
                {
                    "run": run.identity,
                    "expected": list(_REQUIRED_EVIDENCE),
                    "present": [
                        name for name in _REQUIRED_EVIDENCE if name in present_names
                    ],
                    "missing": missing_for_run,
                }
            )
        complete_runs = sum(not bool(item["missing"]) for item in per_run_required)
        all_runs_complete = bool(runs) and complete_runs == len(runs)
        if not included:
            evidence_status = "EMPTY"
        elif not all_runs_complete:
            evidence_status = "MISSING_REQUIRED"
        else:
            evidence_status = "READY"
        sorted_exclusions = sorted(
            exclusions, key=lambda item: (item["run"], item["path"], item["reason"])
        )
        missing_by_run = [
            {"run": str(item["run"]), "files": list(item["missing"])}
            for item in per_run_required
            if item["missing"]
        ]
        manifest: dict[str, object] = {
            "schema": SCHEMA_VERSION,
            "status": "SOFTWARE_PASS",
            "evidence_status": evidence_status,
            "scientific_status": "NOT_EVALUATED",
            "scientific_claim": "NOT_EVALUATED",
            "archive": {
                "path": "evidence.zip",
                "sha256": archive_digest,
                "size_bytes": archive_size,
                "format": "ZIP_STORED",
            },
            "limits": {
                "max_file_size_bytes": file_limit,
                "max_archive_size_bytes": archive_limit,
            },
            "runs": [
                {
                    "run": run.identity,
                    "name": run.display_name,
                    "relative_identity": run.display_name,
                    "index": run.index,
                }
                for run in runs
            ],
            "included": included,
            "exclusions": sorted_exclusions,
            "required_evidence": {
                "expected_per_run": list(_REQUIRED_EVIDENCE),
                "denominator": len(runs),
                "denominator_runs": len(runs),
                "complete_runs": complete_runs,
                "per_run": per_run_required,
                "missing": missing_by_run,
            },
            "counts": {
                "runs": len(runs),
                "included_files": len(included),
                "excluded_files": len(exclusions),
            },
            "exclusion_scope": (
                "Allow-list and known unsafe-content checks are applied. Arbitrary text cannot be scanned for every "
                "possible secret; credential-like filenames, keys and common token/private-key patterns are rejected. "
                "Raw target values, patient volumes, predictions, checkpoints and bank shards are never included."
            ),
        }
        _write_manifest(destination_path, manifest)
        # Ensure the published manifest is durable before returning.  The
        # archive and manifest are each atomically renamed into the reserved
        # new directory; a failure removes the whole newly-created directory.
        directory_fd = os.open(destination_path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return manifest
    except BaseException:
        if created_destination:
            shutil.rmtree(destination_path, ignore_errors=True)
        raise


__all__ = [
    "DEFAULT_MAX_ARCHIVE_SIZE",
    "DEFAULT_MAX_FILE_SIZE",
    "SCHEMA_VERSION",
    "DestinationExistsError",
    "EvidencePackagingError",
    "EvidencePathError",
    "EvidenceValidationError",
    "UnsafeEvidenceError",
    "package_evidence",
]
