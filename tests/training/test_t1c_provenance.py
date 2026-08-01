from __future__ import annotations

import subprocess

import pytest
import torch

from smagm.training import provenance


def _completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


def test_provenance_binds_commit_dirty_state_environment_and_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    responses = iter((_completed("abc123\n"), _completed(" M src/file.py\n")))
    monkeypatch.setattr(provenance.subprocess, "run", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(provenance.platform, "platform", lambda: "test-platform")
    record = provenance.capture_run_provenance(
        repository_root=tmp_path,
        config_hash="a" * 64,
        manifest_hash="b" * 64,
        split_registry_hash="c" * 64,
        assignment_schedule_hash="d" * 64,
        seed=17,
        checkpoint_hash="e" * 64,
        artifact_hashes={"artifact.json": "f" * 64},
        modality_mapping_hash="1" * 64,
        preprocessing_policy_hash="2" * 64,
        preprocessing_record_hash="3" * 64,
        opened_file_ledger_hash="4" * 64,
        dependency_manifest_hash="5" * 64,
        encoder_variant="e2",
        encoder_config_hash="6" * 64,
        encoder_state_hash="7" * 64,
        gaussian_head_initialization_hash="8" * 64,
        renderer_config_hash="9" * 64,
        amplitude_gauge_hash="a" * 64,
        frozen_patient_state_schema_version="smagm-frozen-patient-state-v1",
        device="cpu",
        parameter_count=1,
        allow_dirty=True,
    )
    assert record.commit == "abc123"
    assert record.dirty is True
    assert len(record.environment_hash) == 64
    assert len(record.record_hash) == 64


def test_dirty_provenance_is_rejected_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    responses = iter((_completed("abc123\n"), _completed("?? artifact\n")))
    monkeypatch.setattr(provenance.subprocess, "run", lambda *args, **kwargs: next(responses))
    with pytest.raises(RuntimeError, match="clean repository"):
        provenance.capture_run_provenance(
            repository_root=tmp_path,
            config_hash="a" * 64,
            manifest_hash="b" * 64,
            split_registry_hash="c" * 64,
            assignment_schedule_hash="d" * 64,
            seed=17,
            checkpoint_hash="e" * 64,
            artifact_hashes={"artifact.json": "f" * 64},
            modality_mapping_hash="1" * 64,
            preprocessing_policy_hash="2" * 64,
            preprocessing_record_hash="3" * 64,
            opened_file_ledger_hash="4" * 64,
            dependency_manifest_hash="5" * 64,
            encoder_variant="e2",
            encoder_config_hash="6" * 64,
            encoder_state_hash="7" * 64,
            gaussian_head_initialization_hash="8" * 64,
            renderer_config_hash="9" * 64,
            amplitude_gauge_hash="a" * 64,
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
