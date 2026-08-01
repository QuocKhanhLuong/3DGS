from __future__ import annotations

import subprocess
import hashlib

import pytest
import torch

from smagm.training import provenance


def _completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _commit(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8")).hexdigest()


def test_provenance_binds_commit_dirty_state_environment_and_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    responses = iter((_completed(_commit("commit") + "\n"), _completed(" M src/file.py\n")))
    monkeypatch.setattr(provenance.subprocess, "run", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(provenance.platform, "platform", lambda: "test-platform")
    record = provenance.capture_run_provenance(
        repository_root=tmp_path,
        config_hash=_digest("config"),
        manifest_hash=_digest("manifest"),
        split_registry_hash=_digest("split"),
        assignment_schedule_hash=_digest("schedule"),
        seed=17,
        checkpoint_hash=_digest("checkpoint"),
        artifact_hashes={"artifact.json": _digest("artifact")},
        modality_mapping_hash=_digest("modality"),
        preprocessing_policy_hash=_digest("preprocess-policy"),
        preprocessing_record_hash=_digest("preprocess-record"),
        opened_file_ledger_hash=_digest("opened-ledger"),
        dependency_manifest_hash=_digest("dependency"),
        artifact_manifest_hash=_digest("artifact-manifest"),
        encoder_variant="e2",
        encoder_config_hash=_digest("encoder-config"),
        encoder_state_hash=_digest("encoder-state"),
        gaussian_head_initialization_hash=_digest("head-init"),
        renderer_config_hash=_digest("renderer"),
        amplitude_gauge_hash=_digest("gauge"),
        frozen_patient_state_schema_version="smagm-frozen-patient-state-v1",
        device="cpu",
        parameter_count=1,
        allow_dirty=True,
    )
    assert record.commit == _commit("commit")
    assert record.dirty is True
    assert len(record.environment_hash) == 64
    assert len(record.record_hash) == 64


def test_dirty_provenance_is_rejected_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    responses = iter((_completed(_commit("commit") + "\n"), _completed("?? artifact\n")))
    monkeypatch.setattr(provenance.subprocess, "run", lambda *args, **kwargs: next(responses))
    with pytest.raises(RuntimeError, match="clean repository"):
        provenance.capture_run_provenance(
            repository_root=tmp_path,
            config_hash=_digest("config"),
            manifest_hash=_digest("manifest"),
            split_registry_hash=_digest("split"),
            assignment_schedule_hash=_digest("schedule"),
            seed=17,
            checkpoint_hash=_digest("checkpoint"),
            artifact_hashes={"artifact.json": _digest("artifact")},
            modality_mapping_hash=_digest("modality"),
            preprocessing_policy_hash=_digest("preprocess-policy"),
            preprocessing_record_hash=_digest("preprocess-record"),
            opened_file_ledger_hash=_digest("opened-ledger"),
            dependency_manifest_hash=_digest("dependency"),
            artifact_manifest_hash=_digest("artifact-manifest"),
            encoder_variant="e2",
            encoder_config_hash=_digest("encoder-config"),
            encoder_state_hash=_digest("encoder-state"),
            gaussian_head_initialization_hash=_digest("head-init"),
            renderer_config_hash=_digest("renderer"),
            amplitude_gauge_hash=_digest("gauge"),
            frozen_patient_state_schema_version="smagm-frozen-patient-state-v1",
            device="cpu",
            parameter_count=1,
        )


def test_module_state_hash_changes_with_independent_weights() -> None:
    torch.manual_seed(1)
    first = torch.nn.Linear(3, 2)
    torch.manual_seed(2)
    second = torch.nn.Linear(3, 2)
    assert provenance.module_state_hash(first) != provenance.module_state_hash(second)
