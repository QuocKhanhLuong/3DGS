"""Reconstruct and serialize an immutable prediction package from patient state."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import torch

from ..contracts.coordinates import TargetGrid
from ..features.encoder import EncoderConfig, EvidenceEncoder
from ..reconstruction import build_reconstruction_package, export_reconstruction_package, reconstruct_volume
from ..renderer import RenderConfig
from ..state import PatientState, load_patient_state


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _field_state_hash(payload: object) -> str:
    if not isinstance(payload, dict) or not payload:
        raise ValueError("checkpoint field snapshot must be a non-empty tensor dictionary")
    content = bytearray()
    for name, value in payload.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("checkpoint field snapshot contains an invalid entry")
        content.extend(name.encode())
        content.extend(value.detach().cpu().contiguous().numpy().tobytes())
    return hashlib.sha256(bytes(content)).hexdigest()


def _load_config(path: Path) -> tuple[dict[str, object], str]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "smagm-full-static-pipeline-v1" or config.get("t4_routing") is not False:
        raise ValueError("reconstruction config must use the locked full-static schema with T4 disabled")
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return config, _digest(canonical)


def _resolve_state_path(checkpoint_path: Path) -> tuple[Path, str, dict[str, object] | None]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must use a safe dictionary schema")
    schema = payload.get("schema")
    if schema == "smagm-static-patient-state-v1":
        return checkpoint_path, _file_hash(checkpoint_path), None
    if schema != "smagm-full-static-checkpoint-v1":
        raise ValueError("checkpoint schema is not a full-static checkpoint or patient state")
    relative = payload.get("patient_state_path")
    if not isinstance(relative, str) or Path(relative).is_absolute() or Path(relative).name != relative:
        raise ValueError("checkpoint patient_state_path must be one sibling filename")
    state_path = checkpoint_path.parent / relative
    if not state_path.is_file():
        raise FileNotFoundError("checkpoint-bound patient state is missing")
    for name in ("repository_commit", "repository_diff_hash"):
        value = payload.get(name)
        expected_length = 40 if name == "repository_commit" else 64
        if not isinstance(value, str) or len(value) != expected_length or len(set(value)) == 1:
            raise ValueError(f"checkpoint {name} is invalid or placeholder provenance")
    if not isinstance(payload.get("repository_dirty"), bool):
        raise ValueError("checkpoint must record repository dirty state")
    return state_path, _file_hash(checkpoint_path), payload


def _verify_checkpoint_state_binding(
    checkpoint: dict[str, object],
    state: PatientState,
    config: dict[str, object],
) -> str:
    if checkpoint.get("patient_state_version") != state.state_version:
        raise ValueError("checkpoint patient-state version does not match the serialized patient state")
    field_hash = checkpoint.get("field_for_patient_state_hash")
    if field_hash != state.field_model_hash or checkpoint.get("patient_state_field_model_hash") != state.field_model_hash:
        raise ValueError("checkpoint field identity does not match the patient state")
    if _field_state_hash(checkpoint.get("field")) != state.field_model_hash:
        raise ValueError("checkpoint field tensor snapshot does not match its declared patient-state hash")
    encoder_hash = checkpoint.get("encoder_for_patient_state_hash")
    if not isinstance(encoder_hash, str) or len(encoder_hash) != 64:
        raise ValueError("checkpoint encoder identity is missing or invalid")
    encoder_payload = checkpoint.get("encoder")
    if not isinstance(encoder_payload, dict):
        raise ValueError("checkpoint encoder snapshot must be a tensor dictionary")
    encoder = EvidenceEncoder(EncoderConfig(variant=str(config["encoder_variant"])))
    encoder.load_state_dict(encoder_payload, strict=True)
    if encoder.state_hash() != encoder_hash:
        raise ValueError("checkpoint encoder tensor snapshot does not match its declared identity")
    updates = checkpoint.get("post_snapshot_optimizer_updates")
    if not isinstance(updates, int) or isinstance(updates, bool) or updates < 0:
        raise ValueError("checkpoint must declare optimizer updates after the patient-state snapshot")
    return encoder_hash


def _load_manifest(path: Path, *, state_version: str, manifest_hash: str, patient_id: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "smagm-static-reconstruction-manifest-v1":
        raise ValueError("reconstruction manifest schema is invalid")
    expected = {
        "patient_state_version": state_version,
        "manifest_hash": manifest_hash,
        "patient_id": patient_id,
        "contains_target_payloads": False,
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise ValueError("reconstruction manifest does not bind the frozen context-only patient state")
    for name in ("split_hash", "assignment_hash"):
        value = payload.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"reconstruction manifest {name} must be a SHA-256 digest")
    return payload


def run(
    *, checkpoint_path: Path, config_path: Path, manifest_path: Path,
    output_dir: Path, modality_id: str | None = None,
) -> dict[str, object]:
    config, config_hash = _load_config(config_path)
    state_path, checkpoint_hash, checkpoint_metadata = _resolve_state_path(checkpoint_path)
    state = load_patient_state(state_path)
    if state.config_hash != config_hash:
        raise ValueError("reconstruction config hash disagrees with frozen patient state")
    encoder_identity = checkpoint_hash
    if checkpoint_metadata is not None:
        encoder_identity = _verify_checkpoint_state_binding(checkpoint_metadata, state, config)
    manifest = _load_manifest(
        manifest_path,
        state_version=state.state_version,
        manifest_hash=state.manifest_hash,
        patient_id=state.patient_id,
    )
    reconstruction = config["reconstruction"]
    if not isinstance(reconstruction, dict):
        raise ValueError("reconstruction config must be an object")
    selected_modality = modality_id or state.memory.modality_ids[0]
    grid = TargetGrid(
        reconstruction["index_to_ras_mm"],
        reconstruction["grid_shape_dhw"],
        (selected_modality,),
        (),
    )
    depth_chunk_size = int(reconstruction["depth_chunk_size"])
    renderer_raw = config["renderer"]
    if not isinstance(renderer_raw, dict) or renderer_raw.get("profile") != "delta":
        raise ValueError("reconstruction requires the declared delta through-plane profile")
    render_config = RenderConfig(
        support_epsilon=float(renderer_raw["support_epsilon"]),
        pixel_chunk_size=renderer_raw["pixel_chunk_size"],
        gaussian_chunk_size=renderer_raw["gaussian_chunk_size"],
        minimum_supported_psf_mass=float(renderer_raw["minimum_supported_psf_mass"]),
    )
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("reconstruction export requires an exact repository commit") from error
    if checkpoint_metadata is not None:
        if checkpoint_metadata.get("config_hash") != config_hash:
            raise ValueError("checkpoint config hash disagrees with reconstruction config")
        if checkpoint_metadata.get("repository_commit") != commit:
            raise ValueError("checkpoint and reconstruction code commits disagree")
    start = time.perf_counter()
    volume = reconstruct_volume(
        state,
        grid,
        modality_id=selected_modality,
        depth_chunk_size=depth_chunk_size,
        render_config=render_config,
    )
    environment_payload = {"platform": platform.platform(), "python": sys.version, "torch": torch.__version__}
    if checkpoint_metadata is not None:
        environment_payload.update({
            "training_repository_dirty": checkpoint_metadata["repository_dirty"],
            "training_repository_diff_hash": checkpoint_metadata["repository_diff_hash"],
        })
    package = build_reconstruction_package(
        (volume,), repository_commit=commit, config_hash=state.config_hash,
        manifest_hash=state.manifest_hash, split_hash=str(manifest["split_hash"]), assignment_hash=str(manifest["assignment_hash"]),
        encoder_identity=encoder_identity, field_identity=state.field_model_hash,
        gaussian_identity=state.memory.memory_hash, propagation_identity=_digest(f"round:{state.update_round}:{config['propagation_variant']}"),
        environment_hash=_digest(json.dumps(environment_payload, sort_keys=True)),
        runtime_seconds=time.perf_counter() - start,
    )
    export_reconstruction_package(package, (volume,), output_dir)
    summary = {
        "package_hash": package.package_hash,
        "state_version": state.state_version,
        "volume_artifact_hash": volume.artifact_hash,
        "runtime_seconds": time.perf_counter() - start,
        "repository_commit": commit,
        "encoder_identity": encoder_identity,
        "field_identity": state.field_model_hash,
        "training_repository_dirty": None if checkpoint_metadata is None else checkpoint_metadata["repository_dirty"],
        "training_repository_diff_hash": None if checkpoint_metadata is None else checkpoint_metadata["repository_diff_hash"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct a physical plane/grid from frozen patient state")
    parser.add_argument("--checkpoint", "--state", dest="checkpoint_path", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/full_static_pipeline.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modality")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config,
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        modality_id=args.modality,
    )
    if args.json: print(json.dumps(result, sort_keys=True))
    else:
        for key, value in result.items(): print(f"{key}: {value}")


if __name__ == "__main__": main()
