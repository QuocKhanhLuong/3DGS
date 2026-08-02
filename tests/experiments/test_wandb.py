from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from smagm.experiments import FinishMetadata, WandbLogger, sanitize_config
import smagm.experiments.wandb as wandb_support


class FakeRun:
    def __init__(self, initial_config: object) -> None:
        self.id = "offline-id"
        self.url = "https://wandb.example/r/offline-id"
        self.config: dict[str, object] = dict(initial_config) if isinstance(initial_config, dict) else {}
        self.summary: dict[str, object] = {}
        self.logged: list[tuple[dict[str, object], int | None]] = []
        self.finish_calls = 0

    def log(self, payload: dict[str, object], *, step: int | None = None) -> None:
        self.logged.append((payload, step))

    def finish(self) -> None:
        self.finish_calls += 1


class FakeWandb:
    def __init__(self, *, fail_online: bool = False, fail_offline: bool = False) -> None:
        self.fail_online = fail_online
        self.fail_offline = fail_offline
        self.init_calls: list[dict[str, object]] = []
        self.runs: list[FakeRun] = []
        self.network_calls: list[str] = []

    def init(self, **kwargs: object) -> FakeRun:
        self.init_calls.append(kwargs)
        mode = kwargs["mode"]
        if mode == "online" and self.fail_online:
            raise RuntimeError("online unavailable at /private/checkpoint")
        if mode == "offline" and self.fail_offline:
            raise RuntimeError("offline unavailable")
        run = FakeRun(kwargs.get("config"))
        self.runs.append(run)
        return run

    def login(self) -> None:
        self.network_calls.append("login")

    def sync(self) -> None:
        self.network_calls.append("sync")


class FakeImage:
    def __init__(self, value: object) -> None:
        self.value = value


def _fake_module(fake: FakeWandb) -> SimpleNamespace:
    return SimpleNamespace(init=fake.init, login=fake.login, sync=fake.sync, Image=FakeImage)


def test_disabled_does_not_import_or_call_wandb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail_import(name: str) -> object:
        raise AssertionError(f"disabled mode imported {name}")

    monkeypatch.setattr(wandb_support.importlib, "import_module", fail_import)
    logger = WandbLogger(
        {"learning_rate": 0.001},
        "disabled-run",
        tmp_path,
        {"output": Path("/home/researcher/private/run")},
        mode="disabled",
    )

    logger.start()
    logger.log({"loss": 0.5}, step=1)
    metadata = logger.finish(status="finished")

    assert logger.mode == "disabled"
    assert logger.run_id is None
    assert logger.url is None
    assert logger.fallback_reason is None
    assert metadata == FinishMetadata(None, None, "disabled", None)


def test_offline_fake_logs_safe_values_without_network_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", _fake_module(fake))
    logger = WandbLogger(
        {
            "learning_rate": 0.001,
            "absolute_config_path": Path("/home/researcher/private/run"),
        },
        "offline-run",
        tmp_path,
        {
            "checkpoint": "/tmp/private/checkpoint.pt",
            "relative_path": "runs/synthetic",
            "service_url": "https://wandb.example/project",
        },
        mode="offline",
    )

    logger.start()
    logger.log({"loss": 0.25, "step_count": 2}, step=3)
    metadata = logger.finish(status="complete", failure_reason="/tmp/none")

    assert fake.init_calls == [
        {
            "config": {
                "learning_rate": 0.001,
                "absolute_config_path": "<redacted-absolute-path>",
            },
            "name": "offline-run",
            "dir": str(tmp_path),
            "mode": "offline",
        }
    ]
    assert fake.runs[0].config["metadata"] == {
        "checkpoint": "<redacted-absolute-path>",
        "relative_path": "runs/synthetic",
        "service_url": "https://wandb.example/project",
    }
    assert fake.runs[0].logged == [({"loss": 0.25, "step_count": 2}, 3)]
    assert fake.runs[0].summary == {
        "status": "complete",
        "failure_reason": "<redacted-absolute-path>",
    }
    assert fake.runs[0].finish_calls == 1
    assert fake.network_calls == []
    assert logger.mode == "offline"
    assert logger.run_id == "offline-id"
    assert logger.url == "https://wandb.example/r/offline-id"
    assert logger.fallback_reason is None
    assert metadata.to_dict() == {
        "run_id": "offline-id",
        "url": "https://wandb.example/r/offline-id",
        "mode": "offline",
        "fallback_reason": None,
    }


def test_online_init_failure_falls_back_to_offline_without_network_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeWandb(fail_online=True)
    monkeypatch.setitem(sys.modules, "wandb", _fake_module(fake))
    logger = WandbLogger(
        {},
        "fallback-run",
        tmp_path,
        {},
        mode="online",
    )

    logger.start()
    metadata = logger.finish()

    assert [call["mode"] for call in fake.init_calls] == ["online", "offline"]
    assert logger.mode == "offline"
    assert logger.fallback_reason is not None
    assert "online initialization failed" in logger.fallback_reason
    assert "fell back to offline" in logger.fallback_reason
    assert "/private/checkpoint" not in logger.fallback_reason
    assert metadata.fallback_reason == logger.fallback_reason
    assert fake.network_calls == []


def test_online_init_failure_can_fall_back_to_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeWandb(fail_online=True, fail_offline=True)
    monkeypatch.setitem(sys.modules, "wandb", _fake_module(fake))
    logger = WandbLogger(
        {},
        "disabled-fallback-run",
        tmp_path,
        {},
        mode="online",
        online_fallback="disabled",
    )

    logger.log({"loss": 1.0})
    metadata = logger.finish(status="failed", failure_reason="network failure")

    assert [call["mode"] for call in fake.init_calls] == ["online"]
    assert logger.mode == "disabled"
    assert logger.fallback_reason is not None
    assert "online initialization failed" in logger.fallback_reason
    assert metadata.mode == "disabled"
    assert fake.network_calls == []


def test_configured_project_and_group_are_forwarded_to_wandb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", _fake_module(fake))
    logger = WandbLogger(
        {
            "wandb": {
                "enabled": True,
                "project": "smagm-brats21",
                "entity": None,
                "group": "brats21-real-smoke",
                "job_type": "real-data-smoke",
                "tags": ["brats21", "s11"],
            }
        },
        "configured-run",
        tmp_path,
        {},
        mode="offline",
    )
    logger.start()
    logger.finish()
    assert fake.init_calls[0]["project"] == "smagm-brats21"
    assert fake.init_calls[0]["group"] == "brats21-real-smoke"
    assert fake.init_calls[0]["job_type"] == "real-data-smoke"
    assert fake.init_calls[0]["tags"] == ["brats21", "s11"]


def test_config_and_scalar_validation_is_strict(tmp_path: Path) -> None:
    assert sanitize_config(
        {
            "nested": [Path("relative/file"), "/var/tmp/absolute"],
            "url": "https://example.org/a/b",
        }
    ) == {
        "nested": ["relative/file", "<redacted-absolute-path>"],
        "url": "https://example.org/a/b",
    }

    logger = WandbLogger({}, "validation-run", tmp_path, {}, mode="disabled")
    with pytest.raises(ValueError, match="finite"):
        logger.log({"loss": float("nan")})
    with pytest.raises(TypeError, match="finite number"):
        logger.log({"loss": "0.5"})


def test_derived_images_and_summary_are_safe_and_network_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", _fake_module(fake))
    logger = WandbLogger(
        {},
        "derived-image-run",
        tmp_path,
        {},
        mode="offline",
    )

    logger.start()
    logger.log_images(
        {
            "prediction/target": torch.tensor([[float("nan"), 1.0]]),
            "support/mask": torch.tensor([[0.0, 1.0]]),
        },
        step=4,
    )
    logger.update_summary({
        "artifacts/checkpoint": "/home/researcher/private/checkpoint.pt",
        "artifacts/evaluation": "r4/evaluation/evaluation.json",
    })
    logger.finish()

    assert len(fake.runs[0].logged) == 1
    image_payload, step = fake.runs[0].logged[0]
    assert step == 4
    assert isinstance(image_payload["prediction/target"], FakeImage)
    assert image_payload["prediction/target"].value.tolist() == [[0.0, 1.0]]
    assert fake.runs[0].summary["artifacts/checkpoint"] == "<redacted-absolute-path>"
    assert fake.runs[0].summary["artifacts/evaluation"] == "r4/evaluation/evaluation.json"
    assert fake.network_calls == []
