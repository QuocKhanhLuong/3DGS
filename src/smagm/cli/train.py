"""CPU synthetic diagnostic for the legal T1-C episodic trainer.

The command proves ordering, provenance, and autograd software contracts only.
It is not reconstruction-quality or medical-validity evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from ..baselines.fixed_gaussian import FixedGaussianHead, FixedGaussianHeadConfig
from ..baselines.fixed_support import FixedSupportConfig
from ..contracts.coordinates import PhysicalPlane
from ..contracts.episode import EpisodeLedger
from ..contracts.observation import AvailabilityObservationMeta, PatientSplitRegistry, SparseAvailabilityManifest
from ..data.episodes import EpisodeSamplingConfig
from ..features.encoder import EncoderConfig, EvidenceEncoder
from ..losses.reconstruction import ReconstructionLossConfig
from ..renderer import RenderConfig, SlabProfile
from ..training.episode import LegalEpisodeConfig
from ..training.provenance import capture_run_provenance, module_state_hash
from ..training.sampling import build_matched_variant_schedule
from ..training.trainer import T1CTrainer


def _plane(observation_id: str, z_mm: float, shape_hw: tuple[int, int]) -> PhysicalPlane:
    return PhysicalPlane(
        pixel_center_origin_ras_mm=(0.0, 0.0, z_mm),
        axis_u_ras=(1.0, 0.0, 0.0),
        axis_v_ras=(0.0, 1.0, 0.0),
        spacing_uv_mm=(1.0, 1.0),
        thickness_mm=1.0,
        shape_hw=shape_hw,
        signed_normal_ras=(0.0, 0.0, 1.0),
        observation_id=observation_id,
    )


def _image(shape_hw: tuple[int, int], phase: float) -> np.ndarray:
    height, width = shape_hw
    v, u = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    first = np.exp(-((u - 0.35 * width) ** 2 + (v - 0.45 * height) ** 2) / (2.0 * (0.12 * width) ** 2))
    second = 0.65 * np.exp(-((u - 0.68 * width) ** 2 + (v - 0.58 * height) ** 2) / (2.0 * (0.09 * width) ** 2))
    ridge = 0.2 * np.sin(u / 4.0 + phase) * np.exp(-((v - 0.5 * height) ** 2) / (2.0 * (0.18 * height) ** 2))
    return np.asarray(first + second + ridge, dtype=np.float32)


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def run_synthetic_training(
    *,
    variant: str,
    steps: int = 1,
    seed: int = 29,
    allow_dirty: bool = False,
    repository_root: str | Path | None = None,
) -> dict[str, object]:
    if variant not in ("e0", "e1", "e2"):
        raise ValueError("variant must be e0, e1, or e2")
    if steps <= 0:
        raise ValueError("steps must be positive")
    torch.manual_seed(seed)
    shape_hw = (31, 29)
    payloads = {
        "synthetic-a": _npy_bytes(_image(shape_hw, 0.0)),
        "synthetic-b": _npy_bytes(_image(shape_hw, 0.35)),
    }
    entries = tuple(
        AvailabilityObservationMeta(
            observation_id=observation_id,
            patient_id="synthetic-patient",
            split="train",
            relative_path=f"{observation_id}.npy",
            modality_id="synthetic-mri",
            plane=_plane(observation_id, float(index), shape_hw),
            is_synthetic=True,
        )
        for index, observation_id in enumerate(payloads)
    )
    manifest = SparseAvailabilityManifest(
        entries,
        manifest_id="t1c-synthetic-v1",
        integrity_digests={key: hashlib.sha256(value).hexdigest() for key, value in payloads.items()},
    )
    split_registry = PatientSplitRegistry.create((manifest,))
    sampling = EpisodeSamplingConfig(context_count=1, target_count=1, episode_count=steps, seed=seed)
    matched = build_matched_variant_schedule(manifest, patient_id="synthetic-patient", config=sampling)
    schedule = matched.for_variant(variant)

    encoder = EvidenceEncoder(EncoderConfig(variant=variant, output_stride=1))
    head_config = FixedGaussianHeadConfig(
        input_dim=25,
        appearance_channels=1,
        hidden_dim=32,
        max_center_offset_mm=0.25,
        min_scale_mm=1.5,
        max_scale_mm=5.0,
        max_off_diagonal_mm=0.5,
    )
    # Isolate common-head initialization from variant-specific encoder RNG use.
    torch.manual_seed(seed + 10_000)
    head = FixedGaussianHead(head_config)
    encoder_initialization_hash = module_state_hash(encoder)
    head_initialization_hash = module_state_hash(head)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=2e-3)
    episode_config = LegalEpisodeConfig(
        supports=FixedSupportConfig(step_vu=(4, 4), border_vu=(1, 1)),
        renderer=RenderConfig(
            support_epsilon=1e-10,
            profile=SlabProfile.box(3),
            minimum_supported_psf_mass=1.0,
        ),
        reconstruction_loss=ReconstructionLossConfig(intensity="mse"),
    )
    trainer = T1CTrainer(
        encoder=encoder,
        gaussian_head=head,
        optimizer=optimizer,
        episode_config=episode_config,
    )
    start = time.perf_counter()
    last = None
    with tempfile.TemporaryDirectory(prefix="smagm-t1c-") as directory:
        root = Path(directory)
        for observation_id, payload in payloads.items():
            (root / f"{observation_id}.npy").write_bytes(payload)
        for assignment in schedule.assignments:
            ledger = EpisodeLedger(manifest, assignment, root, split_registry=split_registry)
            last = trainer.train_step(
                ledger=ledger,
                assignment=assignment,
                target_id=assignment.target_ids[0],
            )
    assert last is not None
    runtime_seconds = time.perf_counter() - start
    root = Path(repository_root) if repository_root is not None else Path(__file__).resolve().parents[3]
    run_config = {
        "episode_config_hash": episode_config.config_hash,
        "head": asdict(head_config),
        "sampling": asdict(sampling),
        "seed": seed,
        "variant": variant,
    }
    config_hash = hashlib.sha256(json.dumps(run_config, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    provenance = capture_run_provenance(
        repository_root=root,
        config_hash=config_hash,
        manifest_hash=manifest.manifest_hash,
        split_registry_hash=split_registry.registry_hash,
        assignment_schedule_hash=schedule.schedule_hash,
        seed=seed,
        checkpoint_hash=module_state_hash(encoder, head),
        allow_dirty=allow_dirty,
    )
    report = asdict(last.report)
    encoder_report = encoder.parameter_report
    report.update(
        {
            "adapter_operation_count": encoder_report.adapter_operation_count,
            "assignment_schedule_hash": schedule.schedule_hash,
            "checkpoint_hash": provenance.checkpoint_hash,
            "commit": provenance.commit,
            "config_hash": config_hash,
            "dirty": provenance.dirty,
            "environment_hash": provenance.environment_hash,
            "encoder_parameter_count": encoder_report.parameter_count,
            "manifest_hash": manifest.manifest_hash,
            "encoder_initialization_hash": encoder_initialization_hash,
            "head_initialization_hash": head_initialization_hash,
            "head_parameter_count": sum(parameter.numel() for parameter in head.parameters()),
            "loss_components": {
                name: float(value.detach().cpu()) for name, value in last.step.loss.components.items()
            },
            "opened_context_ids": last.step.context_ids,
            "provenance_record_hash": provenance.record_hash,
            "runtime_seconds": runtime_seconds,
            "split_registry_hash": split_registry.registry_hash,
            "total_parameter_count": encoder_report.parameter_count + sum(parameter.numel() for parameter in head.parameters()),
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one legal synthetic T1-C optimizer path")
    parser.add_argument("--variant", choices=("e0", "e1", "e2"), default="e2")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow development-only provenance on a dirty tree; gate evidence still requires clean",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_synthetic_training(
        variant=args.variant,
        steps=args.steps,
        seed=args.seed,
        allow_dirty=args.allow_dirty,
    )
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        for key, value in report.items():
            if isinstance(value, float):
                print(f"{key}: {value:.8f}")
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()
