"""Immutable reconstruction-package construction."""

from __future__ import annotations

import time

from ..contracts.outputs import ReconstructionPackage, VolumeReconstruction, reconstruction_package_payload_hash


DEFAULT_NON_CLAIMS = (
    "Synthetic or software checks do not establish clinical validity.",
    "Support uncertainty is an uncalibrated diagnostic.",
    "Static reconstruction does not establish recovery of unseen pathology.",
)


def build_reconstruction_package(
    volumes: tuple[VolumeReconstruction, ...], *, repository_commit: str, config_hash: str,
    manifest_hash: str, split_hash: str, assignment_hash: str, encoder_identity: str,
    field_identity: str, gaussian_identity: str, propagation_identity: str,
    environment_hash: str, runtime_seconds: float | None = None,
    execution_status: str | None = None,
) -> ReconstructionPackage:
    if not volumes or len({volume.patient_id for volume in volumes}) != 1 or len({volume.patient_state_version for volume in volumes}) != 1:
        raise ValueError("package volumes must bind one patient state")
    modality_mapping = tuple((volume.modality_id, index) for index, volume in enumerate(volumes))
    artifacts = tuple((f"volume:{volume.modality_id}", volume.artifact_hash) for volume in volumes)
    status = execution_status or ("INSUFFICIENTLY_OBSERVED" if any(bool(v.unsupported_mask.all()) for v in volumes) else "COMPLETE")
    payload = dict(
        patient_id=volumes[0].patient_id, repository_commit=repository_commit,
        config_hash=config_hash, manifest_hash=manifest_hash, split_hash=split_hash,
        assignment_hash=assignment_hash, patient_state_version=volumes[0].patient_state_version,
        encoder_identity=encoder_identity, field_identity=field_identity,
        gaussian_identity=gaussian_identity, propagation_identity=propagation_identity,
        modality_mapping=modality_mapping, output_artifacts=artifacts, execution_status=status,
        runtime_seconds=float(runtime_seconds if runtime_seconds is not None else 0.0),
        environment_hash=environment_hash, non_claims=DEFAULT_NON_CLAIMS,
    )
    return ReconstructionPackage(**payload, package_hash=reconstruction_package_payload_hash(payload))
