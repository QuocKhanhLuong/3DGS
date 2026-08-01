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
from smagm.contracts.observation import AvailabilityObservationMeta, PatientSplitRegistry, SparseAvailabilityManifest
from smagm.gaussians import GaussianBatch, RawGaussianParameters, gaussian_batch_from_raw
from smagm.renderer import RenderConfig, RenderResult, render_plane


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _plane(observation_id: str) -> PhysicalPlane:
    return PhysicalPlane(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        (1.0, 1.0), 1.0, (2, 2), (0.0, 0.0, 1.0), observation_id=observation_id,
    )


def _episode(tmp_path, *, episode_id: str = "episode-a", extra_context: bool = False) -> tuple[EpisodeLedger, EpisodeAssignment]:
    payloads = {"context": b"context tensor bytes", "target": b"TARGET-SENTINEL-MUST-NOT-LEAK"}
    if extra_context:
        payloads["context-extra"] = b"later context bytes"
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
    context_ids = ("context", "context-extra") if extra_context else ("context",)
    assignment = EpisodeAssignment.create(manifest, episode_id=episode_id, patient_id="patient-a", context_ids=context_ids, target_ids=("target",))
    return EpisodeLedger(manifest, assignment, tmp_path, split_registry=PatientSplitRegistry.create((manifest,))), assignment


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


def _runtime_batch(*, requires_grad: bool = False) -> GaussianBatch:
    legacy = _batch(requires_grad=requires_grad)
    return gaussian_batch_from_raw(
        RawGaussianParameters(
            centers_ras_mm=legacy.centers_ras_mm,
            covariance_factor=legacy.covariance_factor,
            raw_log_support_amplitude=legacy.log_support_amplitude,
            appearance=legacy.appearance,
            appearance_valid=legacy.appearance_valid,
        )
    )


def _frozen(ledger: EpisodeLedger, *, upstream_state_hash: str = "upstream", requires_grad: bool = False) -> FrozenPatientState:
    if not ledger.audit_records:
        ledger.open_context("context")
    return FrozenPatientState.create(
        ledger=ledger,
        gaussians=_runtime_batch(requires_grad=requires_grad),
        upstream_state_hash=_hash(upstream_state_hash),
    )


def test_commit_alone_never_reveals_and_successful_flow_is_context_commit_receipt_reveal(tmp_path) -> None:
    ledger, _ = _episode(tmp_path)
    sentinel = b"TARGET-SENTINEL-MUST-NOT-LEAK"
    assert ledger.open_context("context") == b"context tensor bytes"
    target_metadata = ledger.expose_target_metadata("target")
    assert sentinel not in repr(target_metadata).encode()
    assert sentinel not in repr(ledger.event_records).encode()
    assert sentinel not in repr(ledger.audit_records).encode()

    frozen = _frozen(ledger, upstream_state_hash="state-v1")
    commit = ledger.commit_target("target", frozen.state_version)
    with pytest.raises(PermissionError):
        ledger.reveal_target("target", commit)  # type: ignore[arg-type]
    assert [event.event for event in ledger.event_records] == ["OPEN_CONTEXT", "COMMIT_TARGET"]
    assert [row.observation_id for row in ledger.audit_records] == ["context"]

    result, receipt = EpisodeController().render_and_register(
        ledger=ledger,
        commit_capability=commit,
        frozen_state=frozen,
        render_config=RenderConfig(),
    )
    assert isinstance(result, RenderResult)
    records_before_reveal = ledger.prediction_records
    assert len(records_before_reveal) == 1
    record = records_before_reveal[0]
    assert record.target_id == "target"
    assert record.state_version == frozen.state_version
    assert len(record.gaussian_state_digest) == 64
    assert record.renderer_output_schema_version == "render-result-v1"
    assert record.receipt_sequence == 2
    audit_before_reveal = ledger.audit_hash
    assert ledger.reveal_target("target", receipt) == sentinel
    assert [event.event for event in ledger.event_records] == ["OPEN_CONTEXT", "COMMIT_TARGET", "REGISTER_PREDICTION", "REVEAL_TARGET"]
    assert [row.observation_id for row in ledger.audit_records] == ["context", "target"]
    assert ledger.prediction_records == records_before_reveal
    assert ledger.audit_hash != audit_before_reveal
    with pytest.raises(PermissionError):
        ledger.reveal_target("target", receipt)


def test_receipts_bind_ledger_episode_assignment_target_state_plane_and_are_single_use(tmp_path) -> None:
    ledger, assignment = _episode(tmp_path, episode_id="episode-a")
    other_ledger, other_assignment = _episode(tmp_path, episode_id="episode-b")
    frozen = _frozen(ledger, upstream_state_hash="frozen-state")
    other_frozen = _frozen(other_ledger, upstream_state_hash="frozen-state")
    wrong_frozen = _frozen(ledger, upstream_state_hash="wrong-state")
    commit = ledger.commit_target("target", frozen.state_version)
    wrong_commit = other_ledger.commit_target("target", other_frozen.state_version)
    registrar = PredictionRegistrar()
    result = render_plane(_batch(), ledger.expose_target_metadata("target").plane)

    with pytest.raises(PermissionError):
        registrar.register_prediction_receipt(
            ledger=other_ledger,
            commit_capability=commit,
            frozen_state=other_frozen,
            render_evidence=result,  # type: ignore[arg-type]
        )
    with pytest.raises(PermissionError):
        registrar.register_prediction_receipt(
            ledger=ledger,
            commit_capability=commit,
                frozen_state=wrong_frozen,
                render_evidence=result,  # type: ignore[arg-type]
        )
    with pytest.raises(PermissionError):
        registrar.register_prediction_receipt(
            ledger=ledger,
            commit_capability=commit,
                frozen_state=frozen,
                render_evidence=object(),  # type: ignore[arg-type]
        )
    _, receipt = EpisodeController().render_and_register(
        ledger=ledger,
        commit_capability=commit,
        frozen_state=frozen,
        render_config=RenderConfig(),
    )
    with pytest.raises(PermissionError):
        registrar.register_prediction_receipt(
            ledger=ledger,
            commit_capability=commit,
            frozen_state=frozen,
            render_evidence=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(PermissionError):
        other_ledger.reveal_target("target", receipt)
    with pytest.raises(PermissionError):
        ledger.reveal_target("context", receipt)
    assert assignment.assignment_hash != other_assignment.assignment_hash
    assert wrong_commit is not commit
    registrar_parameters = inspect.signature(PredictionRegistrar.register_prediction_receipt).parameters
    assert not {"prediction_digest", "plane_hash", "renderer_version", "render_result", "render_config"} & set(registrar_parameters)
    ledger_parameters = inspect.signature(EpisodeLedger.register_prediction_receipt).parameters
    assert not {"prediction_digest", "plane_hash", "renderer_version"} & set(ledger_parameters)


def test_ledger_rejects_caller_built_registration_and_a_receipt_cannot_cross_ledgers(tmp_path) -> None:
    ledger, assignment = _episode(tmp_path, episode_id="same-episode")
    other_root = tmp_path / "same-assignment-other-ledger"
    other_root.mkdir()
    other_ledger, other_assignment = _episode(other_root, episode_id="same-episode")
    frozen = _frozen(ledger, upstream_state_hash="state")
    commit = ledger.commit_target("target", frozen.state_version)
    with pytest.raises(PermissionError):
        ledger.register_prediction_receipt(commit, registration=object())  # type: ignore[arg-type]
    assert [event.event for event in ledger.event_records] == ["OPEN_CONTEXT", "COMMIT_TARGET"]
    assert [row.observation_id for row in ledger.audit_records] == ["context"]

    _, receipt = EpisodeController().render_and_register(
        ledger=ledger,
        commit_capability=commit,
        frozen_state=frozen,
    )
    assert assignment.assignment_hash == other_assignment.assignment_hash
    with pytest.raises(PermissionError):
        other_ledger.reveal_target("target", receipt)
    assert other_ledger.audit_records == ()
    assert ledger.reveal_target("target", receipt) == b"TARGET-SENTINEL-MUST-NOT-LEAK"
    with pytest.raises(PermissionError):
        ledger.reveal_target("target", receipt)


def test_hand_built_render_result_is_not_registrar_evidence(tmp_path) -> None:
    ledger, _ = _episode(tmp_path)
    frozen = _frozen(ledger, upstream_state_hash="state")
    commit = ledger.commit_target("target", frozen.state_version)
    hand_built = RenderResult(
        intensity=torch.ones((2, 2), dtype=torch.float64),
        support_mass=torch.ones((2, 2), dtype=torch.float64),
        supported_psf_mass=torch.ones((2, 2), dtype=torch.float64),
        unsupported_mask=torch.zeros((2, 2), dtype=torch.bool),
    )
    with pytest.raises(PermissionError, match="actual|[Rr]ender"):
        PredictionRegistrar().register_prediction_receipt(
            ledger=ledger,
            commit_capability=commit,
            frozen_state=frozen,
            render_evidence=hand_built,  # type: ignore[arg-type]
    )
    assert [event.event for event in ledger.event_records] == ["OPEN_CONTEXT", "COMMIT_TARGET"]
    assert [row.observation_id for row in ledger.audit_records] == ["context"]


def test_controller_owns_live_state_and_target_plane_and_rejects_state_mutation(tmp_path) -> None:
    controller_parameters = inspect.signature(EpisodeController.render_and_register).parameters
    assert not {"gaussians", "plane", "target_plane", "render_result", "renderer_version"} & set(controller_parameters)
    assert "render_config" in controller_parameters
    assert set(inspect.signature(FrozenPatientState.create).parameters) == {"ledger", "gaussians", "upstream_state_hash"}
    with pytest.raises(TypeError):
        FrozenPatientState(ledger=object(), gaussians=_runtime_batch(), upstream_state_hash=_hash("forged"))  # type: ignore[call-arg]

    extra_root = tmp_path / "extra-context"
    extra_root.mkdir()
    extra_ledger, _ = _episode(extra_root)
    assert extra_ledger.open_context("context") == b"context tensor bytes"
    extra_state = _frozen(extra_ledger, upstream_state_hash="freeze-context-audit")
    assert extra_ledger.open_context("context") == b"context tensor bytes"
    extra_commit = extra_ledger.commit_target("target", extra_state.state_version)
    with pytest.raises(RuntimeError, match="context|audit"):
        EpisodeController().render_and_register(
            ledger=extra_ledger,
            commit_capability=extra_commit,
            frozen_state=extra_state,
        )
    assert [event.event for event in extra_ledger.event_records] == ["OPEN_CONTEXT", "OPEN_CONTEXT", "COMMIT_TARGET"]
    assert [row.observation_id for row in extra_ledger.audit_records] == ["context", "context"]

    wrong_ledger_root = tmp_path / "wrong-ledger"
    wrong_ledger_root.mkdir()
    bound_ledger, _ = _episode(wrong_ledger_root)
    bound_state = _frozen(bound_ledger, upstream_state_hash="bound-ledger")
    bound_commit = bound_ledger.commit_target("target", bound_state.state_version)
    foreign_root = tmp_path / "foreign-ledger"
    foreign_root.mkdir()
    foreign_ledger, _ = _episode(foreign_root)
    with pytest.raises(PermissionError, match="ledger|commit|bound"):
        EpisodeController().render_and_register(
            ledger=foreign_ledger,
            commit_capability=bound_commit,
            frozen_state=bound_state,
        )
    assert [event.event for event in bound_ledger.event_records] == ["OPEN_CONTEXT", "COMMIT_TARGET"]
    assert foreign_ledger.event_records == ()

    ledger, _ = _episode(tmp_path)
    frozen = _frozen(ledger, upstream_state_hash="live-version", requires_grad=True)
    commit = ledger.commit_target("target", frozen.state_version)
    result, receipt = EpisodeController().render_and_register(
        ledger=ledger,
        commit_capability=commit,
        frozen_state=frozen,
        render_config=RenderConfig(pixel_chunk_size=1),
    )
    assert result.intensity.requires_grad
    torch.nan_to_num(result.intensity, nan=0.0).sum().backward()
    assert frozen.gaussians.appearance.grad is not None
    assert ledger.reveal_target("target", receipt) == b"TARGET-SENTINEL-MUST-NOT-LEAK"

    mutated_root = tmp_path / "mutated"
    mutated_root.mkdir()
    mutated_ledger, _ = _episode(mutated_root)
    mutated_batch = _runtime_batch()
    assert mutated_ledger.open_context("context") == b"context tensor bytes"
    mutated_state = FrozenPatientState.create(ledger=mutated_ledger, gaussians=mutated_batch, upstream_state_hash=_hash("mutated-version"))
    mutated_commit = mutated_ledger.commit_target("target", mutated_state.state_version)
    with torch.no_grad():
        mutated_batch.appearance.add_(1.0)
    with pytest.raises(RuntimeError, match="changed after"):
        EpisodeController().render_and_register(
            ledger=mutated_ledger,
            commit_capability=mutated_commit,
            frozen_state=mutated_state,
        )
    assert [event.event for event in mutated_ledger.event_records] == ["OPEN_CONTEXT", "COMMIT_TARGET"]
    assert [row.observation_id for row in mutated_ledger.audit_records] == ["context"]

    wrong_state_root = tmp_path / "wrong-state"
    wrong_state_root.mkdir()
    wrong_state_ledger, _ = _episode(wrong_state_root)
    committed_state = _frozen(wrong_state_ledger, upstream_state_hash="committed-version")
    alternative_state = _frozen(wrong_state_ledger, upstream_state_hash="other-version")
    wrong_commit = wrong_state_ledger.commit_target("target", committed_state.state_version)
    with pytest.raises(PermissionError, match="state version"):
        EpisodeController().render_and_register(
            ledger=wrong_state_ledger,
            commit_capability=wrong_commit,
            frozen_state=alternative_state,
        )
    assert [event.event for event in wrong_state_ledger.event_records] == ["OPEN_CONTEXT", "COMMIT_TARGET"]

    bad_config_root = tmp_path / "bad-config"
    bad_config_root.mkdir()
    bad_config_ledger, _ = _episode(bad_config_root)
    bad_config_state = _frozen(bad_config_ledger, upstream_state_hash="bad-config-version")
    bad_config_commit = bad_config_ledger.commit_target("target", bad_config_state.state_version)
    with pytest.raises(TypeError, match="RenderConfig"):
        EpisodeController().render_and_register(
            ledger=bad_config_ledger,
            commit_capability=bad_config_commit,
            frozen_state=bad_config_state,
            render_config=object(),  # type: ignore[arg-type]
        )
    assert [event.event for event in bad_config_ledger.event_records] == ["OPEN_CONTEXT", "COMMIT_TARGET"]
    assert [row.observation_id for row in bad_config_ledger.audit_records] == ["context"]

    legacy = _batch()
    with pytest.raises(ValueError, match="LEGACY_RAW"):
        FrozenPatientState.create(ledger=ledger, gaussians=legacy, upstream_state_hash=_hash("legacy"))
    # Backward-compatible direct T0 renderer usage remains valid outside the
    # Phase-1 training/controller path.
    assert render_plane(legacy, ledger.expose_target_metadata("target").plane).intensity.shape == (2, 2)


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
        try:
            context_path.symlink_to(outside)
        except OSError as error:
            if getattr(error, "winerror", None) == 1314:
                pytest.skip("Windows symlink privilege is unavailable")
            raise
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
