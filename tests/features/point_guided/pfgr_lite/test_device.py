from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from smagm.cli.pfgr_lite import main
from smagm.features.point_guided.pfgr_lite.device import resolve_device
from smagm.features.point_guided.pfgr_lite.teacher import validate_target


class _CudaStub:
    def __init__(self, available: bool, count: int) -> None:
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count


class _TorchStub:
    device = staticmethod(torch.device)

    def __init__(self, available: bool, count: int) -> None:
        self.cuda = _CudaStub(available, count)


def test_device_resolver_preserves_requested_and_canonical_effective_cpu() -> None:
    resolved = resolve_device("cpu:0", "cuda:7", torch_module=_TorchStub(False, 0))
    assert resolved.requested == "cpu:0"
    assert resolved.configured == "cuda:7"
    assert resolved.effective == "cpu"
    assert resolved.index is None
    assert resolved.fallback is False


def test_device_resolver_explicit_cuda_wins_over_cpu_config_when_available() -> None:
    resolved = resolve_device("cuda:0", "cpu", torch_module=_TorchStub(True, 1))
    assert resolved.requested == "cuda:0"
    assert resolved.configured == "cpu"
    assert resolved.effective == "cuda:0"
    assert resolved.fallback is False


@pytest.mark.parametrize(
    ("requested", "count", "message"),
    [
        ("cuda", 0, "unavailable"),
        ("cuda:2", 1, "index"),
    ],
)
def test_device_resolver_fails_closed_for_unavailable_cuda(
    requested: str, count: int, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        resolve_device(requested, torch_module=_TorchStub(count > 0, count))


@pytest.mark.parametrize(
    ("command", "required_args"),
    [
        ("benchmark", []),
        ("evaluate", ["--checkpoint", "fixture.pt", "--scenario", "static", "--budget", "0"]),
        ("oracle-evaluate", ["--checkpoint", "fixture.pt", "--oracle-mode", "sampled_one", "--budget", "0"]),
    ],
)
def test_cli_config_cpu_override_wins_over_stale_cuda_sidecar(
    command: str, required_args: list[str], tmp_path: Path
) -> None:
    from smagm.cli.pfgr_lite import _config_for_command, _parser

    source = json.loads(Path("configs/pfgr_lite/main.json").read_text(encoding="utf-8"))
    source["pfgr_config"]["device"] = "cuda:0"
    source["stage_options"]["device"] = "cuda:0"
    config_path = tmp_path / "cuda-sidecar.json"
    config_path.write_text(json.dumps(source), encoding="utf-8")
    args = _parser().parse_args(
        [command, *required_args, "--config", str(config_path), "--device", "cpu"]
    )
    config, details = _config_for_command(args)
    assert config.device == "cuda:0"  # strict PFGR config remains immutable
    assert args.device == "cpu"  # CLI override is the operational authority
    assert details["execution"]["stage_options"]["device"] == "cpu"


@pytest.mark.parametrize(
    ("command", "required_args"),
    [
        ("benchmark", []),
        ("evaluate", ["--checkpoint", "fixture.pt", "--scenario", "static", "--budget", "0"]),
        ("oracle-evaluate", ["--checkpoint", "fixture.pt", "--oracle-mode", "sampled_one", "--budget", "0"]),
    ],
)
def test_cli_config_device_is_used_when_cuda_override_is_explicit_and_available(
    command: str,
    required_args: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The opposite override direction keeps CLI CUDA authoritative."""

    import smagm.features.point_guided.pfgr_lite.device as device_module
    from smagm.cli.pfgr_lite import _config_for_command, _parser

    source = json.loads(Path("configs/pfgr_lite/main.json").read_text(encoding="utf-8"))
    source["pfgr_config"]["device"] = "cpu"
    source["stage_options"]["device"] = "cpu"
    config_path = tmp_path / "cpu-sidecar.json"
    config_path.write_text(json.dumps(source), encoding="utf-8")
    monkeypatch.setattr(
        device_module,
        "resolve_device",
        lambda requested=None, configured=None, **_: device_module.DeviceResolution(
            requested=str(requested or configured or "cpu"),
            configured=configured,
            # The real resolver is idempotent for both the ``cuda`` alias and
            # its canonical ``cuda:0`` effective spelling.  Keep the fixture
            # aligned so it does not force production code to accommodate a
            # non-idempotent mock.
            effective="cuda:0" if str(requested).startswith("cuda") else "cpu",
            accelerator="cuda" if str(requested).startswith("cuda") else "cpu",
            index=0 if str(requested).startswith("cuda") else None,
            cuda_available=True,
            cuda_device_count=1,
        ),
    )
    args = _parser().parse_args(
        [command, *required_args, "--config", str(config_path), "--device", "cuda"]
    )
    config, details = _config_for_command(args)
    assert config.device == "cpu"  # persisted PFGR config remains unchanged
    assert args.device == "cuda:0"
    assert details["execution"]["stage_options"]["device"] == "cuda:0"


def test_service_dry_manifest_records_configured_device_when_cli_omits_override(tmp_path: Path) -> None:
    output = tmp_path / "runs"
    assert main(
        [
            "benchmark",
            "--synthetic",
            "--config",
            "configs/pfgr_lite/synthetic.json",
            "--output-root",
            str(output),
            "--run-name",
            "device-omitted",
            "--dry-manifest",
        ]
    ) == 0
    receipt = json.loads((output / "device-omitted" / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["requested_device"] == "cpu"
    assert receipt["effective_device"] == "cpu"
    assert receipt["environment"]["device_resolution"]["requested_device"] == "cpu"


def test_service_dry_manifest_records_omitted_cli_cuda_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An omitted flag reports the configured CUDA request, not a CPU fallback."""

    import smagm.features.point_guided.pfgr_lite.device as device_module

    source = json.loads(Path("configs/pfgr_lite/synthetic.json").read_text(encoding="utf-8"))
    source["pfgr_config"]["device"] = "cuda:0"
    source["stage_options"]["device"] = "cuda:0"
    config_path = tmp_path / "cuda-config.json"
    config_path.write_text(json.dumps(source), encoding="utf-8")

    def fake_resolve(requested=None, configured=None, **_: object):
        value = str(requested or configured or "cpu")
        if value.startswith("cuda"):
            return device_module.DeviceResolution(
                requested=value,
                configured=configured,
                effective="cuda:0",
                accelerator="cuda",
                index=0,
                cuda_available=True,
                cuda_device_count=1,
            )
        return device_module.DeviceResolution(
            requested=value,
            configured=configured,
            effective="cpu",
            accelerator="cpu",
            index=None,
            cuda_available=False,
            cuda_device_count=0,
        )

    monkeypatch.setattr(device_module, "resolve_device", fake_resolve)
    output = tmp_path / "runs"
    assert main(
        [
            "benchmark",
            "--synthetic",
            "--config",
            str(config_path),
            "--output-root",
            str(output),
            "--run-name",
            "device-configured-cuda",
            "--dry-manifest",
        ]
    ) == 0
    receipt = json.loads(
        (output / "device-configured-cuda" / "receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["requested_device"] == "cuda:0"
    assert receipt["effective_device"] == "cuda:0"
    assert receipt["environment"]["device_resolution"]["requested_device"] == "cuda:0"


@pytest.mark.parametrize(
    "requested_device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable on this host")),
    ],
)
def test_target_join_moves_cpu_target_and_mask_to_context_device(requested_device: str) -> None:
    from smagm.features.point_guided import PointGuidedConfig
    from smagm.features.point_guided.pfgr_lite.config import PFGRLiteConfig
    from smagm.features.point_guided.pfgr_lite.model import PFGRLiteModel

    device = torch.device(requested_device)
    model = PFGRLiteModel(
        PFGRLiteConfig(num_points=2, engineering_only=True),
        frontend_config=PointGuidedConfig(
            num_points=2,
            num_semantic_classes=3,
            point_candidate_multiplier=2,
            offset_hidden_channels=12,
        ),
    ).to(device).eval()
    observations = torch.zeros((1, 3, 9, 9, 9), dtype=torch.float32, device=device)
    observation_context = model.encode_observations(observations, None, (1.0, 1.0, 1.0))
    # This is the actual bool-mask placement regression: callers may provide
    # CPU target/mask tensors while the bound target-free context is CUDA.
    target_cpu = torch.zeros((1, 9, 9, 9), dtype=torch.float64)
    mask_cpu = observation_context.observation_mask.cpu().to(dtype=torch.float32)
    initial_planes = {
        name: getattr(observation_context.initial_planes, name).detach().clone()
        for name in ("xy", "xz", "yz")
    }
    moved = validate_target(
        observation_context.context_id,
        target_cpu,
        mask_cpu,
        observation_context=observation_context,
    )
    # ``torch.device("cuda")`` is an alias while tensors expose the concrete
    # ``cuda:0`` placement; compare against the context's actual feature grid.
    expected_device = observation_context.initial_planes.xy.device
    expected_dtype = observation_context.initial_planes.xy.dtype
    assert moved.target.dtype == expected_dtype
    assert moved.target.device == expected_device
    assert moved.observation_mask.dtype == torch.bool
    assert moved.observation_mask.device == expected_device
    assert torch.equal(moved.observation_mask, observation_context.observation_mask)

    # Native context masks are already bool and on the feature-grid device;
    # validation must leave that placement intact while still owning a clone.
    native_target = torch.full_like(target_cpu, 0.5)
    native = validate_target(
        observation_context.context_id,
        native_target,
        observation_context.observation_mask,
        observation_context=observation_context,
    )
    assert native.target.dtype == expected_dtype
    assert native.target.device == expected_device
    assert native.observation_mask.dtype == torch.bool
    assert native.observation_mask.device == expected_device

    # A CPU query-ID tensor is accepted by the owned gather helpers and both
    # gathers produce tensors on the sealed context device.
    query_ids = torch.tensor([[0, 0, 0], [8, 8, 8]], dtype=torch.long)
    gathered_target = native.gather_target(
        query_ids, device=expected_device, dtype=expected_dtype
    )
    gathered_mask = native.gather_mask(query_ids, device=expected_device)
    assert query_ids.device.type == "cpu"
    assert gathered_target.device == expected_device
    assert gathered_target.dtype == expected_dtype
    assert gathered_mask.device == expected_device
    assert gathered_mask.dtype == torch.bool

    # Substituting the target changes only owned target data; the observation
    # identity, mask, and static initial planes remain exactly unchanged.
    assert not torch.equal(moved.target, native.target)
    assert native.completed_context_id == moved.completed_context_id == observation_context.context_id
    assert torch.equal(native.observation_mask, moved.observation_mask)
    for name, before in initial_planes.items():
        assert torch.equal(getattr(observation_context.initial_planes, name), before)
