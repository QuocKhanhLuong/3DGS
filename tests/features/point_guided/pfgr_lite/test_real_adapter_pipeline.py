"""Bounded generated-NIfTI acceptance coverage for the PFGR CLI adapter.

The fixture deliberately uses a local random one-channel MedicalNet state
dict and an engineering-only role manifest.  It exercises the real NIfTI
loader and CLI stage boundaries without making a pretrained, patient, GPU, or
scientific-result claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pytest
import torch

nib = pytest.importorskip("nibabel")

from smagm.cli.pfgr_lite import main  # noqa: E402
from smagm.data.brats21_point_guided import BraTS21PointGuidedSplit, build_subject_split  # noqa: E402
from smagm.features.point_guided.medicalnet_resnet10 import MedicalNetResNet10  # noqa: E402
from smagm.features.point_guided.pfgr_lite.data import build_training_role_manifest, load_observation_sample  # noqa: E402
from smagm.features.point_guided.pfgr_lite.types import TrainingRoleManifest  # noqa: E402


_CONFIG = Path("configs/pfgr_lite/synthetic.json")
_SEED = 12347
_SUBJECT_IDS = tuple(f"BraTS2021_{index:05d}" for index in range(1, 5))
_AFFINE = np.asarray(
    (
        (1.1, 0.1, 0.0, 12.0),
        (0.0, 1.4, 0.1, -4.0),
        (0.0, 0.0, 1.7, 8.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)


@dataclass(frozen=True)
class _GeneratedCase:
    root: Path
    data_root: Path
    split_file: Path
    roles_file: Path
    checkpoint: Path
    checkpoint_sha256: str
    split: BraTS21PointGuidedSplit
    roles: TrainingRoleManifest


def _make_generated_case() -> _GeneratedCase:
    cache = Path(".pytest_cache")
    cache.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="w5-real-adapter-", dir=str(cache)))
    data_root = root / "data"
    data_root.mkdir()
    try:
        rng = np.random.default_rng(_SEED)
        for subject_id in _SUBJECT_IDS:
            subject = data_root / subject_id
            subject.mkdir()
            for modality in ("t1", "t2", "flair", "t1ce"):
                volume = rng.normal(3.0, 1.0, (9, 9, 9)).astype(np.float32)
                nib.save(nib.Nifti1Image(volume, _AFFINE), str(subject / f"{subject_id}_{modality}.nii.gz"))

        split = build_subject_split(_SUBJECT_IDS, seed=_SEED, split_fractions=(0.5, 0.25, 0.25))
        split_file = root / "split.json"
        split_file.write_text(json.dumps(split.to_dict(), sort_keys=True), encoding="utf-8")
        roles = build_training_role_manifest(split, engineering_only=True)
        roles_file = root / "roles.json"
        roles_file.write_text(json.dumps(roles.as_dict(), sort_keys=True), encoding="utf-8")

        # This is intentionally a local random engineering checkpoint, not an
        # official/pretrained source.  The CLI must still verify its digest and
        # adapt the one-channel conv1 into the three observation channels.
        torch.manual_seed(_SEED)
        checkpoint = root / "engineering-random-1channel.pt"
        torch.save(MedicalNetResNet10(in_channels=1).state_dict(), checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        return _GeneratedCase(root, data_root, split_file, roles_file, checkpoint, digest, split, roles)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


@pytest.fixture(scope="module")
def generated_case() -> Any:
    case = _make_generated_case()
    try:
        yield case
    finally:
        shutil.rmtree(case.root, ignore_errors=True)


def _invoke(case: _GeneratedCase, command: str, run_name: str, extra: Sequence[str]) -> Path:
    output_root = case.root / "runs"
    argv = [
        command,
        "--config",
        str(_CONFIG),
        "--data-root",
        str(case.data_root),
        "--split-file",
        str(case.split_file),
        "--roles-file",
        str(case.roles_file),
        "--output-root",
        str(output_root),
        "--run-name",
        run_name,
        "--max-subjects",
        "2",
        "--device",
        "cpu",
        "--no-amp",
        "--seed",
        str(_SEED),
        *extra,
    ]
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        code = main(argv)
    run_dir = output_root / run_name
    if code != 0:
        receipt = run_dir / "receipt.json"
        detail = receipt.read_text(encoding="utf-8") if receipt.is_file() else "<no receipt>"
        raise AssertionError(f"{command} failed with exit {code}:\n{output.getvalue()}\n{detail}")
    assert run_dir.is_dir()
    receipt = json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "SOFTWARE_PASS"
    assert receipt["scientific_status"] == "NOT_EVALUATED"
    assert receipt["scientific_claim"] == "NOT_EVALUATED"
    assert receipt["capability"] == "production_pending"
    assert "--synthetic" not in receipt["argv"]
    return run_dir


def _assert_nested_equal(left: Any, right: Any, *, path: str = "root") -> None:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor), path
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert torch.equal(left, right), path
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        assert isinstance(left, Mapping) and isinstance(right, Mapping), path
        assert set(left) == set(right), path
        for key in left:
            _assert_nested_equal(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        assert isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)), path
        assert len(left) == len(right), path
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_nested_equal(left_item, right_item, path=f"{path}[{index}]")
        return
    assert left == right, path


def test_generated_nifti_cli_static_updater_bank_chain(generated_case: _GeneratedCase) -> None:
    """Run the actual non-synthetic CLI chain on generated affine-aware NIfTI."""

    # Confirm the real adapter sees ordered T1/T2/FLAIR and preserves shear,
    # anisotropic spacing, and translation before the CLI consumes the files.
    first_subject = generated_case.roles.producer_fit_subject_ids[0]
    sample = load_observation_sample(generated_case.data_root / first_subject)
    assert sample.observations.shape == (3, 9, 9, 9)
    assert sample.geometry.voxel_to_ras_mm[0][1] == pytest.approx(0.1)
    assert sample.geometry.voxel_to_ras_mm[1][2] == pytest.approx(0.1)
    assert sample.geometry.voxel_to_ras_mm[0][0] == pytest.approx(1.1)
    assert sample.geometry.voxel_to_ras_mm[2][2] == pytest.approx(1.7)
    assert sample.geometry.voxel_to_ras_mm[0][3] == pytest.approx(12.0)
    assert sample.target_free

    s0 = _invoke(
        generated_case,
        "static-train",
        "chain-s0",
        [
            "--max-steps",
            "2",
            "--medicalnet-checkpoint",
            str(generated_case.checkpoint),
            "--medicalnet-sha256",
            generated_case.checkpoint_sha256,
        ],
    )
    assert (s0 / "inference.pt").is_file()
    assert (s0 / "resume.pt").is_file()

    s1 = _invoke(
        generated_case,
        "updater-train",
        "chain-s1",
        ["--checkpoint", str(s0 / "inference.pt"), "--max-steps", "2", "--spectral-arm", "u_plus_spectral"],
    )
    assert (s1 / "inference.pt").is_file()
    assert (s1 / "resume.pt").is_file()

    s2 = _invoke(
        generated_case,
        "bank-build",
        "chain-s2",
        [
            "--checkpoint",
            str(s1 / "inference.pt"),
            "--candidate-count",
            "2",
            "--query-count",
            "4",
            "--max-states",
            "1",
        ],
    )
    bank_index = s2 / "S2" / "bank" / "index.json"
    assert bank_index.is_file()
    bank = json.loads(bank_index.read_text(encoding="utf-8"))
    assert bank["status"]["evidence_status"] == "ENGINEERING_ONLY"
    assert bank["manifest"]["row_count"] > 0

    from smagm.features.point_guided.pfgr_lite.checkpoint import load_inference_bundle

    bundle = load_inference_bundle(s0 / "inference.pt", expected_split_hash=generated_case.split.split_hash)
    source = bundle.producer.source_provenance
    assert source.checkpoint_path == str(generated_case.checkpoint.resolve())
    assert source.checkpoint_sha256 == generated_case.checkpoint_sha256
    assert source.checkpoint_integrity_verified is True
    assert source.source_input_channels == 1
    assert source.adapted_input_channels == 3
    assert source.input_conv_adapted is True
    assert source.official_pretrained_verified is False
    assert source.synthetic_untrained is True
    assert bundle.role_manifest is not None
    assert bundle.role_manifest.digest == generated_case.roles.digest
    assert bundle.role_manifest.baseline_split_hash == generated_case.split.split_hash


def test_generated_nifti_s0_full_and_resume_are_bitwise_identical(generated_case: _GeneratedCase) -> None:
    """A one-step stop and explicit resume must match an uninterrupted S0."""

    full = _invoke(
        generated_case,
        "static-train",
        "resume-full",
        [
            "--max-steps",
            "2",
            "--medicalnet-checkpoint",
            str(generated_case.checkpoint),
            "--medicalnet-sha256",
            generated_case.checkpoint_sha256,
        ],
    )
    partial = _invoke(
        generated_case,
        "static-train",
        "resume-partial",
        [
            "--max-steps",
            "1",
            "--medicalnet-checkpoint",
            str(generated_case.checkpoint),
            "--medicalnet-sha256",
            generated_case.checkpoint_sha256,
        ],
    )
    resumed = _invoke(
        generated_case,
        "resume",
        "resume-complete",
        ["--resume-checkpoint", str(partial / "resume.pt"), "--max-steps", "2"],
    )

    from smagm.features.point_guided.pfgr_lite.checkpoint import load_inference_bundle, load_resume

    full_bundle = load_inference_bundle(full / "inference.pt")
    resumed_bundle = load_inference_bundle(resumed / "inference.pt")
    assert set(full_bundle.state_dict) == set(resumed_bundle.state_dict)
    for name in full_bundle.state_dict:
        assert torch.equal(full_bundle.state_dict[name], resumed_bundle.state_dict[name]), name

    full_resume = load_resume(full / "resume.pt")
    resumed_resume = load_resume(resumed / "resume.pt")
    assert full_resume.stage_state == resumed_resume.stage_state
    assert full_resume.stage_state.update == 2
    _assert_nested_equal(full_resume.optimizer_state, resumed_resume.optimizer_state, path="optimizer_state")
    assert full_resume.inference.producer.compatibility_hash == resumed_resume.inference.producer.compatibility_hash


def test_generated_nifti_review_dry_manifest_is_loader_free(generated_case: _GeneratedCase, monkeypatch: pytest.MonkeyPatch) -> None:
    """R9 metadata context uses split/roles/checkpoint only, never MRI/rollouts."""

    checkpoint_run = _invoke(
        generated_case,
        "static-train",
        "review-source",
        [
            "--max-steps",
            "1",
            "--medicalnet-checkpoint",
            str(generated_case.checkpoint),
            "--medicalnet-sha256",
            generated_case.checkpoint_sha256,
        ],
    )
    loader_calls: list[tuple[Any, ...]] = []

    def fail_if_loaded(*args: Any, **kwargs: Any) -> Any:
        loader_calls.append((args, tuple(sorted(kwargs.items()))))
        raise AssertionError("review dry-manifest must not load MRI observations or targets")

    monkeypatch.setattr("smagm.data.brats21_point_guided.load_point_guided_subject", fail_if_loaded)
    review = _invoke(
        generated_case,
        "evaluate",
        "review-r9",
        [
            "--checkpoint",
            str(checkpoint_run / "inference.pt"),
            "--scenario",
            "random",
            "--budget",
            "1",
            "--split-role",
            "test",
            "--dry-manifest",
        ],
    )
    assert loader_calls == []
    payload = json.loads((review / "dry_manifest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "DRY_MANIFEST"
    context_path = review / "review_context.json"
    assert context_path.is_file()
    context = json.loads(context_path.read_text(encoding="utf-8"))
    assert context["status"] == "REVIEW_REQUIRED"
    assert context["scope"] == "R9-final-evaluation"
    assert context["scientific_status"] == "NOT_EVALUATED"
    assert context["context"]["selected_subject_ids"] == list(generated_case.roles.baseline_test_subject_ids[:2])
    assert context["context"]["split_role"] == "test"
    assert context["expected_artifacts"]["checkpoint_sha256"] == hashlib.sha256((checkpoint_run / "inference.pt").read_bytes()).hexdigest()
