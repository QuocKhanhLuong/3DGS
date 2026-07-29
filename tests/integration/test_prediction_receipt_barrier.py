"""Render-before-reveal integration tests for the T0.5 receipt contract."""

from __future__ import annotations

import hashlib
import inspect

import pytest
import torch

from smagm.contracts.coordinates import PhysicalPlane
from smagm.contracts.episode import (
    EpisodeAssignment,
    EpisodeController,
    EpisodeLedger,
    FrozenPatientState,
    PredictionRegistrar,
    prediction_digest_from_render_result,
)
from smagm.contracts.observation import AvailabilityObservationMeta, SparseAvailabilityManifest
from smagm.gaussians import GaussianBatch
from smagm.renderer import RenderConfig, RenderResult, render_plane


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _plane(observation_id: str) -> PhysicalPlane:
    return PhysicalPlane(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        (1.0, 1.0), 1.0, (2, 2), (0.0, 0.0, 1.0), observation_id=observation_id,
    )


def _episode(tmp_path, *, episode_id: str = "episode-a") -> tuple[EpisodeLedger, EpisodeAssignment]:
    payloads = {"context": b"context tensor bytes", "target": b"TARGET-SENTINEL-MUST-NOT-LEAK"}
    for observation_id, payload in payloads.items():
        (tmp_path / f"{observation_id}.bin").write_bytes(payload)
    entries = tuple(
        AvailabilityObservationMeta(
            observation_id=observation_id,
            patient_id="patient-a",
            split="train",
            relative_path=f"{observation_id}.bin",
            modality_id="T2",
            plane=_plane(observation_id),
            is_synthetic=True,
        )
        for observation_id in payloads
    )
    manifest = SparseAvailabilityManifest(entries, integrity_digests={key: hashlib.sha256(value).hexdigest() for key, value in payloads.items()})
    assignment = EpisodeAssignment.create(manifest, episode_id=episode_id, patient_id="patient-a", context_ids=("context",), target_ids=("target",))
    return EpisodeLedger(manifest, assignment, tmp_path), assignment


def _batch(*, requires_grad: bool = False) -> GaussianBatch:
    dtype = torch.float64
    centers = torch.tensor([[0.5, 0.5, 0.0], [1.0, 1.0, 0.0]], dtype=dtype, requires_grad=requires_grad)
    amplitude = torch.zeros((2, 1), dtype=dtype, requires_grad=requires_grad)
    appearance = torch.tensor([[1.0], [2.0]], dtype=dtype, requires_grad=requires_grad)
    return GaussianBatch(
        centers_ras_mm=centers,
        covariance_factor=torch.eye(3, dtype=dtype).expand(2, -1, -1).clone(),
        log_support_amplitude=amplitude,
        appearance=appearance,
        appearance_valid=torch.ones((2, 1), dtype=torch.bool),
    )


def test_commit_alone_never_reveals_and_successful_flow_is_context_commit_receipt_reveal(tmp_path) -> None:
    ledger, _ = _episode(tmp_path)
    sentinel = b"TARGET-SENTINEL-MUST-NOT-LEAK"
    assert ledger.open_context("context") == b"context tensor bytes"
    target_metadata = ledger.expose_target_metadata("target")
    assert sentinel not in repr(target_metadata).encode()
    assert sentinel not in repr(ledger.event_records).encode()
    assert sentinel not in repr(ledger.audit_records).encode()

    commit = ledger.commit_target("target", _hash("state-v1"))
    with pytest.raises(PermissionError):
        ledger.reveal_target("target", commit)  # type: ignore[arg-type]
    assert [event.event for event in ledger.event_records] == ["OPEN_CONTEXT", "COMMIT_TARGET"]
    assert [row.observation_id for row in ledger.audit_records] == ["context"]

    result, receipt = EpisodeController().render_and_register(
        ledger=ledger,
        commit_capability=commit,
        frozen_state=FrozenPatientState(_hash("state-v1")),
        gaussians=_batch(),
        render_config=RenderConfig(),
    )
    assert isinstance(result, RenderResult)
    assert ledger.reveal_target("target", receipt) == sentinel
    assert [event.event for event in ledger.event_records] == ["OPEN_CONTEXT", "COMMIT_TARGET", "REGISTER_PREDICTION", "REVEAL_TARGET"]
    assert [row.observation_id for row in ledger.audit_records] == ["context", "target"]
    with pytest.raises(PermissionError):
        ledger.reveal_target("target", receipt)


def test_receipts_bind_ledger_episode_assignment_target_state_plane_and_are_single_use(tmp_path) -> None:
    ledger, assignment = _episode(tmp_path, episode_id="episode-a")
    other_ledger, other_assignment = _episode(tmp_path, episode_id="episode-b")
    state = _hash("frozen-state")
    commit = ledger.commit_target("target", state)
    wrong_commit = other_ledger.commit_target("target", state)
    registrar = PredictionRegistrar()
    result = render_plane(_batch(), ledger.expose_target_metadata("target").plane)

    with pytest.raises(PermissionError):
        registrar.register_prediction_receipt(
            ledger=other_ledger,
            commit_capability=commit,
            frozen_state=FrozenPatientState(state),
            render_result=result,
            render_config=RenderConfig(),
        )
    with pytest.raises(PermissionError):
        registrar.register_prediction_receipt(
            ledger=ledger,
            commit_capability=commit,
            frozen_state=FrozenPatientState(_hash("wrong-state")),
            render_result=result,
            render_config=RenderConfig(),
        )
    with pytest.raises(TypeError):
        registrar.register_prediction_receipt(
            ledger=ledger,
            commit_capability=commit,
            frozen_state=FrozenPatientState(state),
            render_result=object(),  # type: ignore[arg-type]
            render_config=RenderConfig(),
        )
    receipt = registrar.register_prediction_receipt(
        ledger=ledger,
        commit_capability=commit,
        frozen_state=FrozenPatientState(state),
        render_result=result,
        render_config=RenderConfig(),
    )
    with pytest.raises(PermissionError):
        registrar.register_prediction_receipt(
            ledger=ledger,
            commit_capability=commit,
            frozen_state=FrozenPatientState(state),
            render_result=result,
            render_config=RenderConfig(),
        )
    with pytest.raises(PermissionError):
        other_ledger.reveal_target("target", receipt)
    with pytest.raises(PermissionError):
        ledger.reveal_target("context", receipt)
    assert assignment.assignment_hash != other_assignment.assignment_hash
    assert wrong_commit is not commit
    registrar_parameters = inspect.signature(PredictionRegistrar.register_prediction_receipt).parameters
    assert not {"prediction_digest", "plane_hash", "renderer_version"} & set(registrar_parameters)
    ledger_parameters = inspect.signature(EpisodeLedger.register_prediction_receipt).parameters
    assert not {"prediction_digest", "plane_hash", "renderer_version"} & set(ledger_parameters)


def test_ledger_rejects_caller_built_registration_and_a_receipt_cannot_cross_ledgers(tmp_path) -> None:
    ledger, assignment = _episode(tmp_path, episode_id="same-episode")
    other_root = tmp_path / "same-assignment-other-ledger"
    other_root.mkdir()
    other_ledger, other_assignment = _episode(other_root, episode_id="same-episode")
    state = _hash("state")
    commit = ledger.commit_target("target", state)
    with pytest.raises(PermissionError):
        ledger.register_prediction_receipt(commit, registration=object())  # type: ignore[arg-type]
    assert [event.event for event in ledger.event_records] == ["COMMIT_TARGET"]
    assert ledger.audit_records == ()

    receipt = PredictionRegistrar().register_prediction_receipt(
        ledger=ledger,
        commit_capability=commit,
        frozen_state=FrozenPatientState(state),
        render_result=render_plane(_batch(), ledger.expose_target_metadata("target").plane),
        render_config=RenderConfig(),
    )
    assert assignment.assignment_hash == other_assignment.assignment_hash
    with pytest.raises(PermissionError):
        other_ledger.reveal_target("target", receipt)
    assert other_ledger.audit_records == ()
    assert ledger.reveal_target("target", receipt) == b"TARGET-SENTINEL-MUST-NOT-LEAK"
    with pytest.raises(PermissionError):
        ledger.reveal_target("target", receipt)


def test_symlink_escape_and_content_mutation_fail_without_successful_open_audit(tmp_path) -> None:
    ledger, _ = _episode(tmp_path)
    context_path = tmp_path / "context.bin"
    context_path.write_bytes(b"tampered context")
    with pytest.raises(OSError, match="content digest"):
        ledger.open_context("context")
    assert ledger.audit_records == ()
    assert ledger.event_records == ()

    # Restore the content, then ensure the manifest-root provider rejects an
    # otherwise valid file reached by a symlink escaping that root.
    context_path.write_bytes(b"context tensor bytes")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.bin"
    outside.write_bytes(b"context tensor bytes")
    try:
        context_path.unlink()
        context_path.symlink_to(outside)
        with pytest.raises(PermissionError, match="escapes provider root"):
            ledger.open_context("context")
        assert ledger.audit_records == ()
        assert ledger.event_records == ()
    finally:
        if context_path.is_symlink():
            context_path.unlink()
        outside.unlink(missing_ok=True)


def test_renderer_is_pure_and_digest_uses_detached_copy_without_cutting_live_autograd(tmp_path) -> None:
    ledger, _ = _episode(tmp_path)
    batch = _batch(requires_grad=True)
    before_events = ledger.event_records
    before_audit = ledger.audit_records
    result = render_plane(batch, ledger.expose_target_metadata("target").plane)
    assert ledger.event_records == before_events
    assert ledger.audit_records == before_audit
    assert result.intensity.requires_grad

    digest = prediction_digest_from_render_result(
        result,
        plane_hash=_hash(ledger.expose_target_metadata("target").plane.canonical_json()),
        renderer_version=RenderConfig().renderer_version,
    )
    assert len(digest) == 64
    torch.nan_to_num(result.intensity, nan=0.0).sum().backward()
    assert batch.appearance.grad is not None
    assert torch.isfinite(batch.appearance.grad).all()


def test_prediction_digest_is_canonical_for_layout_and_unsupported_nans() -> None:
    intensity_storage = torch.tensor([[1.0, 99.0, float("nan"), 99.0], [2.0, 99.0, 3.0, 99.0]], dtype=torch.float64)
    support_storage = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]], dtype=torch.float64)
    psf_storage = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]], dtype=torch.float64)
    unsupported_storage = torch.tensor([[False, True, True, True], [False, True, False, True]], dtype=torch.bool)
    non_contiguous = RenderResult(intensity_storage[:, ::2], support_storage[:, ::2], psf_storage[:, ::2], unsupported_storage[:, ::2])
    contiguous = RenderResult(*(tensor.contiguous() for tensor in (non_contiguous.intensity, non_contiguous.support_mass, non_contiguous.supported_psf_mass, non_contiguous.unsupported_mask)))
    plane_hash = _hash("plane")
    version = RenderConfig().renderer_version
    assert prediction_digest_from_render_result(non_contiguous, plane_hash=plane_hash, renderer_version=version) == prediction_digest_from_render_result(contiguous, plane_hash=plane_hash, renderer_version=version)
    assert prediction_digest_from_render_result(contiguous, plane_hash=plane_hash, renderer_version=version) != prediction_digest_from_render_result(contiguous, plane_hash=plane_hash, renderer_version=RenderConfig(pixel_chunk_size=1).renderer_version)


def test_renderer_version_is_controlled_and_profile_aware() -> None:
    base = RenderConfig()
    assert base.renderer_version.startswith("through-plane-profile-aware-gaussian-reference-renderer/v1:")
    assert base.renderer_version == RenderConfig().renderer_version
    assert base.renderer_version != RenderConfig(support_epsilon=1e-7).renderer_version
