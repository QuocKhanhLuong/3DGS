from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from smagm.cli.pfgr_lite import CLIError, COMMANDS, _config_for_command, _parser, _resolve_cached_value_config, _validate_headroom_checkpoint_lineage, _validate_resolved_pfgr_config, main
from smagm.features.point_guided.pfgr_lite.config import PFGRLiteConfig


def test_cli_registers_frozen_command_family_and_strict_unknown_flags() -> None:
    parser = _parser()
    for command in COMMANDS:
        with pytest.raises(SystemExit) as outcome:
            parser.parse_args([command, "--help"])
        assert outcome.value.code == 0
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "--scenario", "adaptive", "--budget", "4", "--unknown", "1"])


def test_synthetic_dry_manifest_is_honest_and_non_mutating(tmp_path: Path) -> None:
    output = tmp_path / "runs"
    assert main(
        [
            "preflight",
            "--synthetic",
            "--config",
            "configs/pfgr_lite/synthetic.json",
            "--output-root",
            str(output),
            "--run-name",
            "preflight",
            "--dry-manifest",
        ]
    ) == 0
    run = output / "preflight"
    manifest = json.loads((run / "dry_manifest.json").read_text(encoding="utf-8"))
    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "DRY_MANIFEST"
    assert manifest["scientific_status"] == "NOT_EVALUATED"
    assert receipt["capability"] == "engineering_only"
    assert not (run / "weights.pt").exists()
    assert main(
        [
            "preflight",
            "--synthetic",
            "--config",
            "configs/pfgr_lite/synthetic.json",
            "--output-root",
            str(output),
            "--run-name",
            "preflight",
            "--dry-manifest",
        ]
    ) == 1


def test_static_base_flag_is_bound_into_strict_execution_config() -> None:
    parser = _parser()
    args = parser.parse_args(["static-train", "--config", "configs/pfgr_lite/main.json", "--base", "b1"])
    config, details = _config_for_command(args, stage="S0")
    assert config.static.variant == "b1_multiscale_v1"
    assert details["execution"]["pfgr_config"]["static"]["variant"] == "b1_multiscale_v1"


@pytest.mark.parametrize("explicit_seed, expected_seed", [(17, 17), (None, 41)])
def test_training_seed_reaches_production_factory_and_receipt_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_seed: int | None, expected_seed: int,
) -> None:
    from smagm.cli import pfgr_lite as cli
    from smagm.features.point_guided.pfgr_lite import stages

    document = json.loads(Path("configs/pfgr_lite/synthetic.json").read_text())
    document["stage_options"]["seed"] = 41
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(document))
    roles_path = tmp_path / "roles.json"
    roles_path.write_text("{}")
    argv = ["static-train", "--config", str(config_path), "--roles-file", str(roles_path)]
    if explicit_seed is not None:
        argv += ["--seed", str(explicit_seed)]
    args = _parser().parse_args(argv)
    config, details = _config_for_command(args, stage="S0")
    assert args.seed == expected_seed
    assert details["execution"]["stage_options"]["seed"] == expected_seed
    assert args._seed_resolution == {
        "requested": explicit_seed, "effective": expected_seed,
        "source": "cli" if explicit_seed is not None else "execution_config",
    }
    captured = []

    class FactoryReached(Exception):
        pass

    def capture_factory(_config, **kwargs):
        captured.append(kwargs["stage_options"].seed)
        raise FactoryReached

    monkeypatch.setattr(cli, "_input_missing", lambda _args: [])
    monkeypatch.setattr(stages, "build_stage_inputs", capture_factory)
    with pytest.raises(FactoryReached):
        cli._production_inputs(args, config, stage="S0")
    assert captured == [expected_seed]


@pytest.mark.parametrize("invalid_seed", [-1, 2**32])
def test_cli_rejects_seed_outside_numpy_rng_domain(invalid_seed: int) -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["static-train", "--seed", str(invalid_seed)])


def test_cached_value_config_binds_only_unresolved_normalization() -> None:
    requested = PFGRLiteConfig(engineering_only=True)
    cached = replace(requested, observation_normalization="measured-recipe-hash")
    bundle = SimpleNamespace(config={"pfgr_config": cached.as_dict()})
    bound = _resolve_cached_value_config(requested, bundle)
    assert bound.observation_normalization == "measured-recipe-hash"

    explicit = replace(requested, observation_normalization="caller-supplied-policy")
    with pytest.raises(CLIError, match="normalization identity"):
        _resolve_cached_value_config(explicit, bundle)


def test_factory_normalization_resolution_rejects_explicit_wrong_policy() -> None:
    requested = PFGRLiteConfig(engineering_only=True)
    resolved = replace(requested, observation_normalization="measured-recipe-hash")
    _validate_resolved_pfgr_config(requested, resolved)
    with pytest.raises(CLIError, match="unresolved policy label"):
        _validate_resolved_pfgr_config(
            replace(requested, observation_normalization="caller-supplied-policy"),
            resolved,
        )


def test_engineering_hydration_preserves_historical_checkpoint_pfgr_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Explicit diagnostics hydrate the checkpoint recipe without widening MAIN."""

    from smagm.features.point_guided import PointGuidedConfig
    from smagm.features.point_guided.pfgr_lite import checkpoint as checkpoint_module
    from smagm.features.point_guided.pfgr_lite.config import frontend_config_to_dict

    # The retired width-128 value lives in the serialized legacy frontend
    # sidecar; PFGRLiteConfig itself remains the production protocol envelope.
    historical_config = PFGRLiteConfig(num_points=2048, engineering_only=False)
    historical_frontend = PointGuidedConfig(
        num_points=2048,
        num_semantic_classes=3,
        point_candidate_multiplier=4,
        offset_hidden_channels=128,
    )
    checkpoint = tmp_path / "historical-width128.pt"
    checkpoint.write_bytes(b"typed-fixture")
    monkeypatch.setattr(
        checkpoint_module,
        "load_inference_bundle",
        lambda _path: SimpleNamespace(
            config={
                "pfgr_config": historical_config.as_dict(),
                "frontend_config": frontend_config_to_dict(historical_frontend),
            }
        ),
    )
    args = _parser().parse_args(
        [
            "evaluate",
            "--engineering-only",
            "--checkpoint",
            str(checkpoint),
            "--scenario",
            "static",
            "--budget",
            "0",
            "--device",
            "cpu",
        ]
    )
    config, details = _config_for_command(args)
    assert config.num_points == 2048
    assert config.engineering_only is False
    assert args._engineering_hydration is True
    assert details["execution"]["stage_options"]["engineering_only"] is True
    assert details["execution"]["frontend_sidecar"]["config"]["offset_hidden_channels"] == 128
    from smagm.features.point_guided.pfgr_lite.model import PFGRLiteModel

    hydrated = PFGRLiteModel(historical_config, frontend_config=historical_frontend, engineering_only=True)
    assert hydrated.frontend_config.offset_hidden_channels == 128
    with pytest.raises(ValueError, match="offset_hidden_channels is locked to 12"):
        PFGRLiteModel(historical_config, frontend_config=historical_frontend)


def test_real_dry_manifest_reports_missing_inputs_without_claiming_success(tmp_path: Path) -> None:
    output = tmp_path / "runs"
    assert main(
        [
            "evaluate",
            "--config",
            "configs/pfgr_lite/main.json",
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--scenario",
            "adaptive",
            "--budget",
            "4",
            "--output-root",
            str(output),
            "--run-name",
            "missing",
            "--dry-manifest",
        ]
    ) == 0
    payload = json.loads((output / "missing" / "dry_manifest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKED"
    assert payload["scientific_status"] == "NOT_EVALUATED"
    assert payload["missing_inputs"]


@pytest.mark.parametrize(
    ("command_args", "run_name"),
    [
        (["benchmark", "--synthetic", "--device", "cpu"], "device-benchmark"),
        (
            [
                "evaluate",
                "--synthetic",
                "--device",
                "cpu",
                "--checkpoint",
                "missing.pt",
                "--scenario",
                "static",
                "--budget",
                "0",
            ],
            "device-evaluate",
        ),
        (
            [
                "oracle-evaluate",
                "--synthetic",
                "--device",
                "cpu",
                "--checkpoint",
                "missing.pt",
                "--oracle-mode",
                "sampled_one",
                "--budget",
                "1",
            ],
            "device-oracle",
        ),
    ],
)
def test_service_dry_manifests_retain_requested_and_effective_device(
    tmp_path: Path, command_args: list[str], run_name: str
) -> None:
    output = tmp_path / "runs"
    assert main(
        command_args
        + [
            "--config",
            "configs/pfgr_lite/synthetic.json",
            "--output-root",
            str(output),
            "--run-name",
            run_name,
            "--dry-manifest",
        ]
    ) == 0
    receipt = json.loads((output / run_name / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["requested_device"] == "cpu"
    assert receipt["effective_device"] == "cpu"
    assert receipt["environment"]["device_resolution"]["effective_device"] == "cpu"


def test_every_subcommand_help_is_available_in_a_fresh_parser() -> None:
    parser = _parser()
    # argparse's help path exits cleanly; this loop also guards accidental
    # imports of teacher/oracle dependencies during top-level parser creation.
    for command in COMMANDS:
        with pytest.raises(SystemExit) as outcome:
            parser.parse_args([command, "--help"])
        assert outcome.value.code == 0


def test_existing_run_directory_is_never_mutated_on_reservation_failure(tmp_path: Path) -> None:
    output = tmp_path / "runs"
    existing = output / "owned-by-teammate"
    existing.mkdir(parents=True)
    sentinel = existing / "sentinel.txt"
    sentinel.write_bytes(b"leave this byte-for-byte unchanged\n")
    assert main([
        "preflight",
        "--synthetic",
        "--config",
        "configs/pfgr_lite/synthetic.json",
        "--output-root",
        str(output),
        "--run-name",
        existing.name,
    ]) == 1
    assert sentinel.read_bytes() == b"leave this byte-for-byte unchanged\n"
    assert not (existing / "receipt.json").exists()


def test_wandb_receipt_logs_finite_metrics_and_counts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Summary(dict):
        pass

    class FakeRun:
        id = "offline-test-id"
        url = None

        def __init__(self) -> None:
            self.logged: list[dict[str, object]] = []
            self.summary = Summary()

        def log(self, payload: dict[str, object]) -> None:
            self.logged.append(dict(payload))

        def finish(self) -> None:
            return None

    run = FakeRun()
    fake = SimpleNamespace(init=lambda **_: run)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    output = tmp_path / "runs"
    assert main(
        [
            "preflight",
            "--synthetic",
            "--config",
            "configs/pfgr_lite/synthetic.json",
            "--output-root",
            str(output),
            "--run-name",
            "wandb",
            "--wandb",
        ]
    ) == 0
    assert run.logged
    assert "counts.target_reads" in run.logged[0]
    assert "metrics" not in run.logged[0]  # flattened numeric fields only
    assert run.summary


@pytest.mark.parametrize("exact_pool_audit", [False, True])
def test_typed_next1_headroom_evaluate_runs_four_subjects_and_retains_dense_receipt(tmp_path: Path, exact_pool_audit: bool) -> None:
    """Exercise the real PFGR model/query/writer R4B seam on CPU.

    This intentionally uses the explicit engineering fixture; the result is
    software evidence only and must remain INCONCLUSIVE for MAIN admission.
    """

    output = tmp_path / "runs"
    assert main(
        [
            "headroom-evaluate",
            "--synthetic",
            "--config",
            "configs/pfgr_lite/synthetic.json",
            "--output-root",
            str(output),
            "--run-name",
            "typed-next1",
        ] + (["--exact-pool-audit"] if exact_pool_audit else [])
    ) == 0
    run = output / "typed-next1"
    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    result = json.loads((run / "next1" / "headroom_metrics.json").read_text(encoding="utf-8"))
    evidence = json.loads((run / "next1" / "next1_evidence.json").read_text(encoding="utf-8"))
    assert receipt["scientific_status"] == "INCONCLUSIVE"
    assert result["privileged"] is True
    assert result["target_dependent"] is True
    assert len(evidence["subject_ids"]) == 4
    assert evidence["candidate_count"] == 32
    assert evidence["query_count"] == 1024
    assert evidence["privileged"] is True
    assert evidence["target_dependent"] is True
    assert evidence["exact_pool_audit"]["enabled"] is exact_pool_audit
    assert receipt["counts"]["target_reads"] is None or isinstance(receipt["counts"]["target_reads"], int)
    assert receipt["counts"]["target_access_phase"] == "post-proposal-seal"
    assert receipt["artifacts"]["evidence"].endswith("next1_evidence.json")
    assert all(
        len(subject["screening"]["rows"]) == 32
        and subject["oracle"]["metric"] is not None
        and subject["oracle"]["same_winner"] is True
        and subject["no_op"]["gain"] == 0.0
        and subject["no_op"]["metric"]["improvement"]["masked_charbonnier"] == 0.0
        and subject["confirmation"]["query_count"] == subject["confirmation"]["rows"][0]["footprint_voxels"]
        and subject["confirmation"]["confirmation_action_id"] == subject["confirmation"]["winner_action_id"]
        and subject["confirmation"]["rows"][0]["candidate_index"] == next(
            index
            for index, candidate in enumerate(subject["candidate_pool"])
            if candidate["action_id"] == subject["oracle"]["winner_action_id"]
        )
        and isinstance(subject["candidate_pool"][0]["point_ras_mm"], list)
        and all(item["prediction_available"] is True for item in subject["random_controls"])
        for subject in evidence["subjects"]
    )
    # Exercise the complete generated receipt, not a permissive hand-written
    # subset: the original packager silently omitted every NEXT-1 artifact.
    from smagm.features.point_guided.pfgr_lite.artifacts import package_evidence

    manifest = package_evidence([run], tmp_path / "packaged")
    included = {row["archive_path"].split("/typed-next1/", 1)[-1] for row in manifest["included"]}
    assert {"next1/headroom_metrics.json", "next1/headroom_decision.json", "next1/next1_evidence.json"} <= included


def test_environment_hardware_uses_effective_cuda_device_and_reports_missing_properties(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import torch

    from smagm.cli.pfgr_lite import _environment_receipt

    resolution = SimpleNamespace(effective="cuda:1", as_dict=lambda: {"effective": "cuda:1"})
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    seen = []
    def properties(index):
        seen.append(index)
        return SimpleNamespace(name="synthetic CUDA properties", total_memory=123456, major=9, minor=1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    result = _environment_receipt("cuda:1", configured="cpu", resolution=resolution)
    assert seen == [1]
    assert result["cuda_hardware"] == {
        "status": "observed", "effective_device": "cuda:1", "index": 1,
        "name": "synthetic CUDA properties", "total_memory_bytes": 123456,
        "compute_capability_major": 9, "compute_capability_minor": 1,
        "identity_source": "resolved_device_properties; not_tensor_placement_observation",
        "cuda_visible_devices": None,
        "uuid": None,
    }
    def unavailable(index):
        raise RuntimeError("synthetic unavailable properties")
    monkeypatch.setattr(torch.cuda, "get_device_properties", unavailable)
    result = _environment_receipt("cuda:1", resolution=resolution)
    assert result["device_effective"] == "cuda:1"
    assert result["cuda_hardware"]["status"] == "unavailable"
    assert "name" not in result["cuda_hardware"]
    result = _environment_receipt("cpu")
    assert result["cuda_hardware"]["effective_device"] == "cpu"
    assert result["cuda_hardware"]["status"] == "not_applicable"


def test_runtime_model_counts_shared_parameters_without_claiming_flops_or_cpu_cuda_peaks():
    import torch

    from smagm.cli.pfgr_lite import _runtime_model_receipt

    layer = torch.nn.Linear(2, 3)
    layer.bias.requires_grad_(False)
    model = torch.nn.ModuleDict({"first": layer, "shared": layer})
    result = _runtime_model_receipt(model)
    assert result["parameter_count"] == 9
    assert result["requires_grad_parameter_count"] == 6
    assert result["parameter_bytes"] == 36
    assert result["actual_devices"] == ["cpu"]
    assert result["cuda_memory"] == {}
    assert result["flops"] is None
    assert _runtime_model_receipt(None)["parameter_count"] is None


def test_headroom_checkpoint_lineage_compares_frozen_component_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Matching metadata cannot hide a changed frozen base component."""

    from smagm.features.point_guided.pfgr_lite import checkpoint as checkpoint_module

    base_path = tmp_path / "base.pt"
    updater_path = tmp_path / "updater.pt"
    base_path.write_bytes(b"base-bytes")
    updater_path.write_bytes(b"updater-bytes")
    config = PFGRLiteConfig()
    frontend = {"schema_version": "frontend", "config": {"num_points": 2}}
    base_state = {"frontend.semantic_head.weight": torch.ones(2, 2), "updater.network.weight": torch.ones(2, 2)}
    updater_state = {"frontend.semantic_head.weight": torch.ones(2, 2), "updater.network.weight": torch.ones(2, 2)}
    role = SimpleNamespace(digest="roles-v1")

    def bundle(state: dict[str, torch.Tensor]) -> SimpleNamespace:
        return SimpleNamespace(
            config={"pfgr_config": config.as_dict()},
            frontend_config=frontend,
            split_hash="split-v1",
            role_manifest=role,
            stage_provenance={"completed": True, "producer_compatibility_hash": "producer-v1", "checkpoint_id": str(base_path)},
            producer=SimpleNamespace(compatibility_hash="producer-v1"),
            state_dict=state,
        )

    monkeypatch.setattr(
        checkpoint_module,
        "load_inference_bundle",
        lambda path: bundle(base_state if Path(path) == base_path else updater_state),
    )
    args = SimpleNamespace(synthetic=False, engineering_only=False, checkpoint=updater_path, base_checkpoint=base_path)
    _validate_headroom_checkpoint_lineage(args, config)

    updater_state["frontend.semantic_head.weight"][0, 0] = 9.0
    with pytest.raises(CLIError, match="frozen component bytes differ"):
        _validate_headroom_checkpoint_lineage(args, config)
