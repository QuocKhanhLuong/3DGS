"""Offline BraTS21 sparse-plane derivative preparation.

Only selected two-dimensional payloads are written.  Dense NIfTI sources are
never copied or passed to the training process, and segmentation is written
under a separate evaluator-only path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
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
        return (self.root / relative).resolve(strict=True)

    @property
    def segmentation_payload_path(self) -> Path:
        relative = Path(str(self.evaluator_json["segmentation_payload_relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("prepared evaluator segmentation path must remain relative")
        return (self.root / relative).resolve(strict=True)


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
    if manifest_json.get("schema") != MANIFEST_SCHEMA or evaluator_json.get("schema") != EVALUATOR_SCHEMA:
        raise ValueError("prepared BraTS21 metadata schema is invalid")
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
    assignment = EpisodeAssignment.create(
        manifest,
        episode_id=assignment_json["episode_id"],
        patient_id=assignment_json["patient_id"],
        context_ids=assignment_json["context_ids"],
        target_ids=assignment_json["target_ids"],
    )
    if assignment.assignment_hash != assignment_json.get("assignment_hash"):
        raise ValueError("prepared assignment hash mismatch")
    if evaluator_json.get("target_observation_id") not in assignment.target_ids:
        raise ValueError("prepared evaluator target is not the declared target")
    if set(assignment.context_ids) & set(assignment.target_ids):
        raise ValueError("prepared context and target IDs overlap")
    return PreparedBraTS21(destination, manifest, assignment, manifest_json, evaluator_json)


__all__ = [
    "EVALUATOR_SCHEMA",
    "MANIFEST_SCHEMA",
    "PREPARED_SCHEMA",
    "PreparedBraTS21",
    "load_prepared_bundle",
    "prepare_brats21_smoke",
]
