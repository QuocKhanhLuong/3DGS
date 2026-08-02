"""Offline BraTS21 sparse-plane derivative preparation.

The legacy diagnostic smoke writer may materialize an isolated evaluator
target, but the product writer stores only selected context planes plus
receipt-gated target and segmentation references. Dense NIfTI sources are
never copied or passed to the training process, and evaluator payloads are not
materialized before the product prediction/receipt barrier.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from ..contracts.coordinates import PhysicalPlane, SourceAffineTransform, SourceConvention
from ..contracts.episode import EpisodeAssignment
from ..contracts.observation import AvailabilityObservationMeta, PatientSplitRegistry, SparseAvailabilityManifest
from .brats21 import (
    BRATS21_MODALITIES,
    BRATS21_SEGMENTATION,
    BraTS21Patient,
    BraTS21ValidationError,
    canonical_hash,
    deterministic_plane_schedule,
    discover_patient,
    extract_axial_plane,
    npy_bytes,
    plane_from_nifti,
    validate_patient,
)
from .brats21_sampling import BraTS21SamplingConfig, build_sampling_plan


PREPARED_SCHEMA = "smagm-brats21-prepared-smoke-v1"
MANIFEST_SCHEMA = "smagm-brats21-sparse-manifest-v1"
EVALUATOR_SCHEMA = "smagm-brats21-evaluator-manifest-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _plane_from_dict(raw: Mapping[str, Any]) -> PhysicalPlane:
    source = raw.get("source_transform")
    transform = None
    if source is not None:
        transform = SourceAffineTransform(
            source["plane_index_to_source_mm"],
            SourceConvention(source["convention"]),
        )
    return PhysicalPlane(
        raw["pixel_center_origin_ras_mm"],
        raw["axis_u_ras"],
        raw["axis_v_ras"],
        raw["spacing_uv_mm"],
        raw["thickness_mm"],
        raw["shape_hw"],
        raw["signed_normal_ras"],
        source_transform=transform,
        observation_id=raw.get("observation_id"),
    )


def _entry_json(entry: AvailabilityObservationMeta, digest: str) -> dict[str, Any]:
    raw = entry.to_canonical_dict()
    raw["content_sha256"] = digest
    return raw


@dataclass(frozen=True)
class PreparedBraTS21:
    """Loaded manifest-bound derivative bundle used by the smoke runner."""

    root: Path
    manifest: SparseAvailabilityManifest
    assignment: EpisodeAssignment
    manifest_json: Mapping[str, Any]
    evaluator_json: Mapping[str, Any]

    @property
    def patient_id(self) -> str:
        return self.assignment.patient_id

    @property
    def target_id(self) -> str:
        if len(self.assignment.target_ids) != 1:
            raise ValueError("prepared smoke bundle must contain exactly one target")
        return self.assignment.target_ids[0]

    @property
    def target_plane(self) -> PhysicalPlane:
        return _plane_from_dict(self.evaluator_json["target_plane"])

    @property
    def target_payload_path(self) -> Path:
        relative = Path(str(self.evaluator_json["target_payload_relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("prepared evaluator target path must remain relative")
        candidate = (self.root / relative).resolve()
        if not bool(self.evaluator_json.get("target_payload_deferred", False)):
            candidate = candidate.resolve(strict=True)
        return candidate

    @property
    def target_payload_deferred(self) -> bool:
        return bool(self.evaluator_json.get("target_payload_deferred", False))

    @property
    def segmentation_payload_path(self) -> Path | None:
        raw_relative = self.evaluator_json.get("segmentation_payload_relative_path")
        if raw_relative in (None, ""):
            return None
        relative = Path(str(raw_relative))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("prepared evaluator segmentation path must remain relative")
        candidate = (self.root / relative).resolve()
        if not bool(self.evaluator_json.get("segmentation_payload_deferred", False)):
            candidate = candidate.resolve(strict=True)
        return candidate

    @property
    def segmentation_payload_deferred(self) -> bool:
        return bool(self.evaluator_json.get("segmentation_payload_deferred", False))


def _choose_patient(source_root: Path, requested: str | None) -> BraTS21Patient:
    if requested is not None:
        patient = discover_patient(source_root / requested)
        result = validate_patient(patient, require_segmentation=True, include_data=False, include_source_hash=False)
        if not result.valid:
            raise BraTS21ValidationError(f"{requested}: {result.error}")
        return patient
    failures: list[str] = []
    for directory in sorted(path for path in source_root.iterdir() if path.is_dir()):
        try:
            patient = discover_patient(directory)
            result = validate_patient(patient, require_segmentation=True, include_data=False, include_source_hash=False)
            if result.valid:
                return patient
            failures.append(f"{directory.name}: {result.error}")
        except (BraTS21ValidationError, OSError) as error:
            failures.append(f"{directory.name}: {error}")
    raise BraTS21ValidationError("no complete BraTS21 patient is available; first failures: " + "; ".join(failures[:5]))


def prepare_brats21_smoke(
    *,
    source_root: str | Path,
    output_dir: str | Path,
    patient_id: str | None = None,
    inplane_stride_vu: tuple[int, int] = (4, 4),
    schedule_fractions: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Create one deterministic, manifest-bound context/target episode.

    The target payload is present only under ``payloads/target`` and is never
    included in the public context inventory.  The segmentation payload is
    written only under ``evaluator/`` and is not an observation manifest entry.
    """

    source = Path(source_root).resolve(strict=True)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"prepared smoke output is non-empty: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    patient = _choose_patient(source, patient_id)
    validation = validate_patient(
        patient,
        require_segmentation=True,
        include_data=True,
        include_source_hash=True,
    )
    if not validation.valid:
        raise BraTS21ValidationError(f"{patient.patient_id}: {validation.error}")
    summaries = {item.suffix: item for item in validation.summaries}
    schedule = deterministic_plane_schedule(summaries["flair"].shape_xyz, fractions=schedule_fractions)
    destination.joinpath("payloads", "context").mkdir(parents=True)
    destination.joinpath("payloads", "target").mkdir(parents=True)
    destination.joinpath("evaluator").mkdir(parents=True)

    payload_digests: dict[str, str] = {}
    entries: list[AvailabilityObservationMeta] = []
    context_ids: list[str] = []
    target_ids: list[str] = []
    selected_records: list[dict[str, Any]] = []
    for modality, role, slice_index in schedule:
        observation_id = f"{patient.patient_id}:{modality}:{role}"
        plane = plane_from_nifti(
            summaries[modality].affine,
            summaries[modality].shape_xyz,
            slice_index,
            observation_id=observation_id,
            inplane_stride_vu=inplane_stride_vu,
        )
        array = extract_axial_plane(patient.modality_paths[modality], slice_index, inplane_stride_vu=inplane_stride_vu)
        payload = npy_bytes(array)
        relative = Path("payloads") / ("context" if role == "context" else "target") / f"{modality}_{role}.npy"
        (destination / relative).write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        payload_digests[observation_id] = digest
        entry = AvailabilityObservationMeta(
            observation_id=observation_id,
            patient_id=patient.patient_id,
            split="train",
            relative_path=relative.as_posix(),
            modality_id=modality,
            plane=plane,
            is_synthetic=False,
            registration_record_id="brats21-source-affine-v1",
        )
        entries.append(entry)
        (context_ids if role == "context" else target_ids).append(observation_id)
        selected_records.append({
            "modality_id": modality,
            "observation_id": observation_id,
            "payload_relative_path": relative.as_posix(),
            "plane": plane.to_canonical_dict(),
            "role": role,
            "slice_index": slice_index,
            "payload_sha256": digest,
        })

    target_record = next(item for item in selected_records if item["role"] == "target")
    target_slice = int(target_record["slice_index"])
    assert patient.segmentation_path is not None
    segmentation = extract_axial_plane(patient.segmentation_path, target_slice, inplane_stride_vu=inplane_stride_vu)
    if not np.all(np.isin(segmentation.astype(np.int64), tuple(sorted({0, 1, 2, 4})))):
        raise BraTS21ValidationError("evaluator segmentation plane contains labels outside {0,1,2,4}")
    segmentation_relative = Path("evaluator") / "target_segmentation.npy"
    (destination / segmentation_relative).write_bytes(npy_bytes(segmentation.astype(np.uint8, copy=False)))

    manifest = SparseAvailabilityManifest(
        tuple(entries),
        manifest_id=f"brats21-smoke:{patient.patient_id}",
        integrity_digests=payload_digests,
    )
    assignment = EpisodeAssignment.create(
        manifest,
        episode_id=f"brats21-smoke:{patient.patient_id}:episode-000000",
        patient_id=patient.patient_id,
        context_ids=context_ids,
        target_ids=target_ids,
    )
    split_payload = {
        "patient_splits": {patient.patient_id: "train"},
        "schema": "brats21-split-v1",
        "source_kind": "PRIVILEGED_OR_SIMULATED_REAL_DATA_SMOKE",
    }
    split_hash = canonical_hash(split_payload)
    evaluator_payload = {
        "schema": EVALUATOR_SCHEMA,
        "patient_id": patient.patient_id,
        "source_kind": "PRIVILEGED_OR_SIMULATED_REAL_DATA_SMOKE",
        "target_observation_id": target_record["observation_id"],
        "target_payload_relative_path": target_record["payload_relative_path"],
        "target_payload_sha256": target_record["payload_sha256"],
        "target_plane": target_record["plane"],
        "segmentation_payload_relative_path": segmentation_relative.as_posix(),
        "segmentation_payload_sha256": _sha256_file(destination / segmentation_relative),
        "segmentation_labels": [0, 1, 2, 4],
        "contains_training_segmentation": False,
    }
    manifest_payload = {
        "schema": MANIFEST_SCHEMA,
        "manifest_id": manifest.manifest_id,
        "source_kind": "PRIVILEGED_OR_SIMULATED_REAL_DATA_SMOKE",
        "patient_id": patient.patient_id,
        "split": "train",
        "entries": [_entry_json(entry, payload_digests[entry.observation_id]) for entry in entries],
        "manifest_hash": manifest.manifest_hash,
        "context_observation_ids": tuple(sorted(context_ids)),
        "target_observation_ids": tuple(sorted(target_ids)),
        "contains_target_payloads": False,
        "source_nifti_hashes": {suffix: summaries[suffix].source_hash for suffix in (*BRATS21_MODALITIES, BRATS21_SEGMENTATION)},
        "source_geometry": {
            "shape_xyz": summaries["flair"].shape_xyz,
            "spacing_xyz_mm": summaries["flair"].spacing_xyz_mm,
            "affine": summaries["flair"].affine,
            "orientation": summaries["flair"].orientation,
        },
        "selected_planes": selected_records,
    }
    assignment_payload = assignment.to_canonical_dict() | {
        "schema": "brats21-assignment-v1",
        "assignment_hash": assignment.assignment_hash,
        "content_binding_hash": manifest._content_binding_hash,
    }
    hashes = {
        "schema": "brats21-smoke-hashes-v1",
        "source_nifti": manifest_payload["source_nifti_hashes"],
        "extracted_payload": payload_digests,
        "segmentation_payload": evaluator_payload["segmentation_payload_sha256"],
        "sparse_manifest": manifest.manifest_hash,
        "evaluator_manifest": canonical_hash(evaluator_payload),
        "split": split_hash,
        "assignment": assignment.assignment_hash,
    }
    _write_json(destination / "manifest.json", manifest_payload)
    _write_json(destination / "evaluator_manifest.json", evaluator_payload)
    _write_json(destination / "assignment.json", assignment_payload)
    _write_json(destination / "split.json", split_payload | {"split_hash": split_hash})
    prepared = {
        "schema": PREPARED_SCHEMA,
        "patient_id": patient.patient_id,
        "source_kind": "PRIVILEGED_OR_SIMULATED_REAL_DATA_SMOKE",
        "manifest_hash": manifest.manifest_hash,
        "split_hash": split_hash,
        "assignment_hash": assignment.assignment_hash,
        "evaluator_manifest_hash": hashes["evaluator_manifest"],
        "context_count": len(context_ids),
        "target_count": len(target_ids),
        "context_target_disjoint": not (set(context_ids) & set(target_ids)),
        "hashes": hashes,
        "source_root_not_copied": True,
    }
    _write_json(destination / "prepared.json", prepared)
    _write_json(destination / "hashes.json", hashes)
    return prepared


def load_prepared_bundle(root: str | Path) -> PreparedBraTS21:
    """Load only the small prepared metadata bundle and bind all hashes."""

    destination = Path(root).resolve(strict=True)
    manifest_json = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    evaluator_json = json.loads((destination / "evaluator_manifest.json").read_text(encoding="utf-8"))
    assignment_json = json.loads((destination / "assignment.json").read_text(encoding="utf-8"))
    split_json = json.loads((destination / "split.json").read_text(encoding="utf-8"))
    prepared_json = json.loads((destination / "prepared.json").read_text(encoding="utf-8"))
    hashes_json = json.loads((destination / "hashes.json").read_text(encoding="utf-8"))
    if not all(isinstance(value, dict) for value in (manifest_json, evaluator_json, assignment_json, split_json, prepared_json, hashes_json)):
        raise ValueError("prepared BraTS21 metadata files must contain JSON objects")
    if manifest_json.get("schema") != MANIFEST_SCHEMA or evaluator_json.get("schema") != EVALUATOR_SCHEMA:
        raise ValueError("prepared BraTS21 metadata schema is invalid")
    if prepared_json.get("schema") not in (PREPARED_SCHEMA, "smagm-brats21-prepared-product-v1"):
        raise ValueError("prepared BraTS21 bundle schema is invalid")
    if prepared_json.get("hashes") != hashes_json:
        raise ValueError("prepared metadata hash inventory does not match hashes.json")
    if prepared_json.get("manifest_hash") != hashes_json.get("sparse_manifest"):
        raise ValueError("prepared manifest hash binding is inconsistent")
    if prepared_json.get("evaluator_manifest_hash") != hashes_json.get("evaluator_manifest"):
        raise ValueError("prepared evaluator manifest hash binding is inconsistent")
    if prepared_json.get("split_hash") != hashes_json.get("split"):
        raise ValueError("prepared split hash binding is inconsistent")
    if prepared_json.get("assignment_hash") != hashes_json.get("assignment"):
        raise ValueError("prepared assignment hash binding is inconsistent")
    split_unsigned = dict(split_json)
    split_claimed = split_unsigned.pop("split_hash", None)
    split_actual = canonical_hash(split_unsigned)
    if split_claimed != split_actual or hashes_json.get("split") != split_actual:
        raise ValueError("prepared split hash mismatch")
    if hashes_json.get("evaluator_manifest") != canonical_hash(evaluator_json):
        raise ValueError("prepared evaluator manifest digest mismatch")
    entries = []
    digests = {}
    for raw in manifest_json.get("entries", []):
        entries.append(
            AvailabilityObservationMeta(
                raw["observation_id"], raw["patient_id"], raw["split"], raw["relative_path"],
                raw["modality_id"], _plane_from_dict(raw["plane"]), raw["is_synthetic"],
                raw.get("acquisition_cost_key"), raw.get("registration_record_id"),
            )
        )
        digests[raw["observation_id"]] = raw["content_sha256"]
    manifest = SparseAvailabilityManifest(tuple(entries), manifest_id=manifest_json.get("manifest_id", f"brats21-smoke:{manifest_json['patient_id']}"), integrity_digests=digests)
    if manifest.manifest_hash != manifest_json.get("manifest_hash"):
        raise ValueError("prepared sparse manifest hash mismatch")
    if hashes_json.get("sparse_manifest") != manifest.manifest_hash:
        raise ValueError("prepared sparse manifest digest mismatch")
    if prepared_json.get("patient_id") != manifest_json.get("patient_id"):
        raise ValueError("prepared patient identity is inconsistent across metadata")
    patient_splits = split_json.get("patient_splits")
    if not isinstance(patient_splits, dict) or patient_splits.get(manifest_json.get("patient_id")) != manifest_json.get("split"):
        raise ValueError("prepared patient split is inconsistent across metadata")
    if hashes_json.get("source_nifti") != manifest_json.get("source_nifti_hashes"):
        raise ValueError("prepared source-file hash inventory is inconsistent")
    assignment = EpisodeAssignment.create(
        manifest,
        episode_id=assignment_json["episode_id"],
        patient_id=assignment_json["patient_id"],
        context_ids=assignment_json["context_ids"],
        target_ids=assignment_json["target_ids"],
    )
    if assignment.assignment_hash != assignment_json.get("assignment_hash"):
        raise ValueError("prepared assignment hash mismatch")
    if hashes_json.get("assignment") != assignment.assignment_hash:
        raise ValueError("prepared assignment digest mismatch")
    if sorted(manifest_json.get("context_observation_ids", ())) != sorted(assignment.context_ids):
        raise ValueError("prepared manifest context IDs do not match the assignment")
    if sorted(manifest_json.get("target_observation_ids", ())) != sorted(assignment.target_ids):
        raise ValueError("prepared manifest target IDs do not match the assignment")
    if manifest_json.get("sampling_protocol_hash") is not None:
        if hashes_json.get("sampling_protocol") != manifest_json.get("sampling_protocol_hash"):
            raise ValueError("prepared sampling protocol hash is inconsistent")
        if prepared_json.get("sampling_protocol_hash") != manifest_json.get("sampling_protocol_hash"):
            raise ValueError("prepared sampling protocol binding is inconsistent")
    selected_planes = manifest_json.get("selected_planes")
    if not isinstance(selected_planes, list):
        raise ValueError("prepared manifest must declare selected plane records")
    target_records = [
        item for item in selected_planes
        if isinstance(item, dict) and item.get("role") == "target"
        and item.get("observation_id") == evaluator_json.get("target_observation_id")
    ]
    if len(target_records) != 1:
        raise ValueError("prepared manifest must contain exactly one selected target plane")
    target_record = target_records[0]
    target_plane = evaluator_json.get("target_plane")
    if target_record.get("plane") != target_plane:
        raise ValueError("prepared evaluator target plane does not match the selected manifest plane")
    target_reference = evaluator_json.get("target_reference")
    if isinstance(target_reference, dict) and target_reference.get("plane") != target_plane:
        raise ValueError("prepared deferred target reference plane does not match the evaluator plane")
    segmentation_reference = evaluator_json.get("segmentation_reference")
    if isinstance(segmentation_reference, dict) and segmentation_reference.get("target_plane") != target_plane:
        raise ValueError("prepared deferred segmentation reference plane does not match the evaluator plane")
    if evaluator_json.get("target_observation_id") not in assignment.target_ids:
        raise ValueError("prepared evaluator target is not the declared target")
    if set(assignment.context_ids) & set(assignment.target_ids):
        raise ValueError("prepared context and target IDs overlap")
    if bool(evaluator_json.get("target_payload_deferred", False)):
        reference = evaluator_json.get("target_reference")
        if not isinstance(reference, dict) or not isinstance(reference.get("reference_sha256"), str):
            raise ValueError("deferred target reference is missing or malformed")
        try:
            source_position = float(
                reference.get("source_slice_position_index", reference["source_slice_index"])
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("deferred target source slice position is missing or malformed") from error
        if not math.isfinite(source_position) or source_position < 0.0:
            raise ValueError("deferred target source slice position must be finite and non-negative")
        source_shape = reference.get("source_shape_xyz")
        if source_shape is not None:
            if (
                not isinstance(source_shape, (list, tuple))
                or len(source_shape) != 3
                or any(int(value) <= 0 for value in source_shape)
                or source_position > float(int(source_shape[2]) - 1)
            ):
                raise ValueError("deferred target source slice position is outside the source volume")
        claimed = str(reference["reference_sha256"])
        unsigned = dict(reference)
        unsigned.pop("reference_sha256", None)
        actual = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if claimed != actual:
            raise ValueError("deferred target reference hash mismatch")
        target_id = str(evaluator_json["target_observation_id"])
        if manifest._expected_sha256(target_id) != claimed:
            raise ValueError("deferred target reference is not bound to the target manifest entry")
        if evaluator_json.get("target_payload_sha256") is not None:
            raise ValueError("deferred target manifest must not contain a materialized target payload hash")
    if bool(evaluator_json.get("segmentation_payload_deferred", False)):
        segmentation_reference = evaluator_json.get("segmentation_reference")
        if not isinstance(segmentation_reference, dict) or not isinstance(segmentation_reference.get("reference_sha256"), str):
            raise ValueError("deferred segmentation reference is missing or malformed")
        try:
            segmentation_position = float(segmentation_reference["source_slice_position_index"])
            segmentation_shape = segmentation_reference["source_shape_xyz"]
            if (
                not isinstance(segmentation_shape, (list, tuple))
                or len(segmentation_shape) != 3
                or any(int(value) <= 0 for value in segmentation_shape)
                or not math.isfinite(segmentation_position)
                or segmentation_position < 0.0
                or segmentation_position > float(int(segmentation_shape[2]) - 1)
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("deferred segmentation source position is malformed or outside the source volume") from error
        claimed = str(segmentation_reference["reference_sha256"])
        unsigned = dict(segmentation_reference)
        unsigned.pop("reference_sha256", None)
        actual = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if claimed != actual:
            raise ValueError("deferred segmentation reference hash mismatch")
        relative = Path(str(evaluator_json.get("segmentation_payload_relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts or (destination / relative).exists():
            raise ValueError("deferred segmentation payload must remain absent and relative")
        if evaluator_json.get("segmentation_payload_sha256") is not None:
            raise ValueError("deferred segmentation manifest must not contain a materialized payload hash")
    return PreparedBraTS21(destination, manifest, assignment, manifest_json, evaluator_json)


def prepare_brats21_product_patient(
    *,
    source_root: str | Path,
    output_dir: str | Path,
    patient_id: str,
    split: str = "validation",
    target_modality: str = "flair",
    seed: int = 0,
    inplane_stride_vu: tuple[int, int] = (4, 4),
    sampling_config: BraTS21SamplingConfig | None = None,
    source_hashes: Mapping[str, str],
    require_segmentation: bool = True,
) -> dict[str, Any]:
    """Materialize only one patient's legal context planes for product runs.

    Dense source files remain in ``source_root``. Context payloads are small,
    per-patient derivatives; target intensity and optional segmentation remain
    evaluator-only deferred references and are not materialized in preparation.
    """

    source = Path(source_root).resolve(strict=True)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"prepared product output is non-empty: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    patient = discover_patient(source / patient_id)
    resolved_source_hashes = {str(key): str(value) for key, value in dict(source_hashes).items()}
    expected_source_suffixes = set(BRATS21_MODALITIES)
    if patient.segmentation_path is not None:
        expected_source_suffixes.add(BRATS21_SEGMENTATION)
    missing_source_hashes = sorted(expected_source_suffixes - set(resolved_source_hashes))
    if missing_source_hashes:
        raise BraTS21ValidationError(
            "product preparation requires precomputed inventory hashes for "
            + ", ".join(missing_source_hashes)
        )
    if any(
        len(resolved_source_hashes[suffix]) != 64
        or any(char not in "0123456789abcdef" for char in resolved_source_hashes[suffix])
        for suffix in expected_source_suffixes
    ):
        raise BraTS21ValidationError("product preparation received malformed source hashes")
    validation = validate_patient(
        patient,
        require_segmentation=require_segmentation,
        # Header validation is performed here; only the selected context and
        # evaluator plane are materialized below. The target is bound as a
        # source/geometry reference and deferred until receipt-gated reveal.
        # Full-volume finite-value inventory belongs to the explicit inventory
        # command.
        include_data=False,
        include_source_hash=False,
    )
    if not validation.valid:
        raise BraTS21ValidationError(f"{patient.patient_id}: {validation.error}")
    summaries = {item.suffix: item for item in validation.summaries}
    # The completed inventory is the source-file binding for this derivative.
    # Do not re-hash source NIfTI files here: that would open hidden target or
    # evaluator bytes before the receipt barrier. Context extraction below
    # reads only declared context planes; deferred target/segmentation readers
    # recheck their bound source hashes after their legal access barrier.
    resolved_sampling = sampling_config or BraTS21SamplingConfig()
    episode_id = f"brats21-product:{patient.patient_id}:{split}:{seed:08d}"
    plan = build_sampling_plan(
        summaries,
        episode_id=episode_id,
        target_modality=target_modality,
        split=split,
        seed=seed,
        inplane_stride_vu=inplane_stride_vu,
        config=resolved_sampling,
    )
    destination.joinpath("payloads", "context").mkdir(parents=True)
    destination.joinpath("evaluator").mkdir(parents=True)

    payload_digests: dict[str, str] = {}
    entries: list[AvailabilityObservationMeta] = []
    context_ids: list[str] = []
    target_ids: list[str] = []
    selected_records: list[dict[str, Any]] = []
    target_reference: dict[str, Any] | None = None
    selected = list(plan.context) + [plan.target]
    for selection in selected:
        observation_id = str(selection.plane.observation_id)
        role = selection.role
        if role == "target":
            # Keep target intensity out of the preparation process.  The
            # manifest binds a source/geometry reference, and the product
            # runner supplies a callback that materializes the plane only
            # from EpisodeLedger.reveal_target after receipt registration.
            relative = Path("deferred") / f"{selection.modality_id}_{selection.ordinal:02d}_{role}.npy"
            target_reference = {
                "schema": "smagm-deferred-target-reference-v1",
                "modality_id": selection.modality_id,
                "source_slice_index": selection.source_slice_index,
                "source_slice_position_index": selection.source_slice_position_index,
                "source_shape_xyz": list(summaries[selection.modality_id].shape_xyz),
                "source_nifti_hash": resolved_source_hashes[selection.modality_id],
                "inplane_stride_vu": list(inplane_stride_vu),
                "plane": selection.plane.to_canonical_dict(),
            }
            target_reference["reference_sha256"] = hashlib.sha256(
                json.dumps(target_reference, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            digest = str(target_reference["reference_sha256"])
        else:
            relative = Path("payloads") / "context" / f"{selection.modality_id}_{selection.ordinal:02d}_{role}.npy"
            array = extract_axial_plane(
                patient.modality_paths[selection.modality_id],
                selection.source_slice_index,
                inplane_stride_vu=inplane_stride_vu,
            )
            payload = npy_bytes(array)
            (destination / relative).write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
        payload_digests[observation_id] = digest
        entry = AvailabilityObservationMeta(
            observation_id=observation_id,
            patient_id=patient.patient_id,
            split=split,
            relative_path=relative.as_posix(),
            modality_id=selection.modality_id,
            plane=selection.plane,
            is_synthetic=False,
            registration_record_id="brats21-source-affine-v1",
        )
        entries.append(entry)
        (context_ids if role == "context" else target_ids).append(observation_id)
        selected_records.append(
            {
                "modality_id": selection.modality_id,
                "observation_id": observation_id,
                "payload_relative_path": relative.as_posix(),
                "plane": selection.plane.to_canonical_dict(),
                "physical_position_mm": selection.physical_position_mm,
                "role": role,
                "ordinal": selection.ordinal,
                "slice_index": selection.source_slice_index,
                "slice_position_index": selection.source_slice_position_index,
                "payload_sha256": digest,
                "payload_deferred": role == "target",
            }
        )

    target_record = next(item for item in selected_records if item["role"] == "target")
    if target_reference is None:
        raise RuntimeError("product preparation failed to bind its target reference")
    evaluator_payload: dict[str, Any] = {
        "schema": EVALUATOR_SCHEMA,
        "patient_id": patient.patient_id,
        "source_kind": "SIMULATED_SPARSE_ACQUISITION",
        "target_observation_id": target_record["observation_id"],
        "target_payload_relative_path": target_record["payload_relative_path"],
        "target_payload_deferred": True,
        "target_reference": target_reference,
        "target_payload_sha256": None,
        "target_plane": target_record["plane"],
        "segmentation_payload_relative_path": None,
        "segmentation_payload_deferred": False,
        "segmentation_reference": None,
        "segmentation_payload_sha256": None,
        "segmentation_labels": None,
        "contains_training_segmentation": False,
    }
    if patient.segmentation_path is not None:
        segmentation_relative = Path("deferred") / "target_segmentation.npy"
        segmentation_reference: dict[str, Any] = {
            "schema": "smagm-deferred-segmentation-reference-v1",
            "source_shape_xyz": list(summaries[BRATS21_SEGMENTATION].shape_xyz),
            "source_slice_position_index": plan.target.source_slice_position_index,
            "source_segmentation_hash": resolved_source_hashes[BRATS21_SEGMENTATION],
            "inplane_stride_vu": list(inplane_stride_vu),
            "target_plane": plan.target.plane.to_canonical_dict(),
        }
        segmentation_reference["reference_sha256"] = hashlib.sha256(
            json.dumps(segmentation_reference, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        evaluator_payload["segmentation_payload_relative_path"] = segmentation_relative.as_posix()
        evaluator_payload["segmentation_payload_deferred"] = True
        evaluator_payload["segmentation_reference"] = segmentation_reference
        evaluator_payload["segmentation_labels"] = [0, 1, 2, 4]

    manifest = SparseAvailabilityManifest(
        tuple(entries),
        manifest_id=f"brats21-product:{patient.patient_id}:{plan.protocol_hash[:16]}",
        integrity_digests=payload_digests,
    )
    assignment = EpisodeAssignment.create(
        manifest,
        episode_id=episode_id,
        patient_id=patient.patient_id,
        context_ids=context_ids,
        target_ids=target_ids,
    )
    split_payload = {
        "schema": "brats21-split-v2",
        "patient_splits": {patient.patient_id: split},
        "source_kind": "SIMULATED_SPARSE_ACQUISITION",
    }
    split_hash = canonical_hash(split_payload)
    manifest_payload = {
        "schema": MANIFEST_SCHEMA,
        "manifest_id": manifest.manifest_id,
        "source_kind": "SIMULATED_SPARSE_ACQUISITION",
        "patient_id": patient.patient_id,
        "patient_pseudonym": hashlib.sha256(f"smagm-brats21-patient-v1:{patient.patient_id}".encode()).hexdigest(),
        "split": split,
        "entries": [_entry_json(entry, payload_digests[entry.observation_id]) for entry in entries],
        "manifest_hash": manifest.manifest_hash,
        "context_observation_ids": tuple(sorted(context_ids)),
        "target_observation_ids": tuple(sorted(target_ids)),
        "contains_target_payloads": False,
        "source_nifti_hashes": {
            suffix: resolved_source_hashes[suffix]
            for suffix in (*BRATS21_MODALITIES, BRATS21_SEGMENTATION)
            if suffix in resolved_source_hashes
        },
        "source_geometry": {
            "shape_xyz": summaries["flair"].shape_xyz,
            "spacing_xyz_mm": summaries["flair"].spacing_xyz_mm,
            "affine": summaries["flair"].affine,
            "orientation": summaries["flair"].orientation,
        },
        "sampling": resolved_sampling.to_dict(),
        "sampling_protocol_hash": plan.protocol_hash,
        "selected_planes": selected_records,
    }
    assignment_payload = assignment.to_canonical_dict() | {
        "schema": "brats21-assignment-v2",
        "assignment_hash": assignment.assignment_hash,
        "content_binding_hash": manifest._content_binding_hash,
    }
    hashes = {
        "schema": "smagm-brats21-product-hashes-v1",
        "source_nifti": manifest_payload["source_nifti_hashes"],
        "extracted_context_payload": {
            observation_id: digest
            for observation_id, digest in payload_digests.items()
            if observation_id in context_ids
        },
        "deferred_target_reference": target_reference["reference_sha256"],
        "segmentation_payload": evaluator_payload["segmentation_payload_sha256"],
        "segmentation_reference": None
        if evaluator_payload["segmentation_reference"] is None
        else evaluator_payload["segmentation_reference"]["reference_sha256"],
        "sparse_manifest": manifest.manifest_hash,
        "evaluator_manifest": canonical_hash(evaluator_payload),
        "split": split_hash,
        "assignment": assignment.assignment_hash,
        "sampling_protocol": plan.protocol_hash,
    }
    _write_json(destination / "manifest.json", manifest_payload)
    _write_json(destination / "evaluator_manifest.json", evaluator_payload)
    _write_json(destination / "assignment.json", assignment_payload)
    _write_json(destination / "split.json", split_payload | {"split_hash": split_hash})
    prepared = {
        "schema": "smagm-brats21-prepared-product-v1",
        "patient_id": patient.patient_id,
        "patient_pseudonym": manifest_payload["patient_pseudonym"],
        "source_kind": "SIMULATED_SPARSE_ACQUISITION",
        "manifest_hash": manifest.manifest_hash,
        "split_hash": split_hash,
        "assignment_hash": assignment.assignment_hash,
        "evaluator_manifest_hash": hashes["evaluator_manifest"],
        "sampling_protocol_hash": plan.protocol_hash,
        "context_count": len(context_ids),
        "target_count": len(target_ids),
        "target_payload_deferred": True,
        "context_target_disjoint": not (set(context_ids) & set(target_ids)),
        "hashes": hashes,
        "source_root_not_copied": True,
    }
    _write_json(destination / "prepared.json", prepared)
    _write_json(destination / "hashes.json", hashes)
    return prepared


__all__ = [
    "EVALUATOR_SCHEMA",
    "MANIFEST_SCHEMA",
    "PREPARED_SCHEMA",
    "PreparedBraTS21",
    "load_prepared_bundle",
    "prepare_brats21_smoke",
    "prepare_brats21_product_patient",
]
