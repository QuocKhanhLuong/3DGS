from __future__ import annotations

from dataclasses import asdict
import json
import random
from types import SimpleNamespace

import pytest
import torch
import numpy as np

from smagm.features.point_guided.pfgr_lite.checkpoint import (
    CHECKPOINT_CONFIG_SCHEMA,
    hydrate_inference_model,
    load_inference_bundle,
    load_legacy_inference_bundle,
    load_resume,
    load_value_artifact,
    restore_rng_state,
    save_inference_bundle,
    save_resume,
    save_value_artifact,
)
from smagm.features.point_guided.pfgr_lite.config import PFGRLiteConfig, frontend_config_to_dict
from smagm.features.point_guided import PointGuidedConfig
from smagm.features.point_guided.pfgr_lite.model import PFGRLiteModel
from smagm.features.point_guided.pfgr_lite.provenance import ProducerCompatibility, SourceProvenance, ValueFitIdentity, module_state_digest
from smagm.features.point_guided.pfgr_lite.types import InferenceBundle, ProducerDependencies, StageState
from smagm.features.point_guided.pfgr_lite.value_bank import GainScale
from smagm.features.point_guided.pfgr_lite.value_net import SignedValueNet, ValueFitOptions, _fit_hash
from smagm.features.point_guided.pfgr_lite.footprint import PFGRQueryLattice


def _producer() -> ProducerDependencies:
    compatibility = ProducerCompatibility(
        observation_normalization_hash="norm",
        geometry_query_version_hash="query",
        medicalnet_provenance_hash="medicalnet",
        frozen_bn_hash="bn",
        static_head_hash="static",
        semantic_head_hash="semantic",
        point_refiner_hash="points",
        spectral_projector_hash="spectral",
        state_initializer_hash="state",
        updater_hash="updater",
        decoder_hash="decoder",
        writer_hash="writer",
        candidate_geometry_hash="candidate",
        label_definition_hash="label",
    )
    source = SourceProvenance(
        source_sha="source-sha",
        config_sha="config-sha",
        checkpoint_sha256="a" * 64,
        checkpoint_integrity_verified=True,
        source_state_dict_key_count=3,
        loaded_backbone_key_count=3,
        adaptation_digest="adaptation",
        parameter_hash="parameter",
        frozen_bn_hash="bn",
        official_pretrained_verified=False,
        synthetic_untrained=True,
        traversal_count=1,
    )
    return ProducerDependencies(compatibility=compatibility, source_provenance=source)


def _bundle(capability: str = "static") -> InferenceBundle:
    pfgr = PFGRLiteConfig().as_dict()
    frontend = frontend_config_to_dict(PointGuidedConfig(num_semantic_classes=3))
    config = {
        "schema_version": CHECKPOINT_CONFIG_SCHEMA,
        "pfgr_config": pfgr,
        "frontend_config": frontend,
        "stage": "inference",
        "split_roles": {"producer_fit": "producer-role", "calibration_fit": "fit-role", "calibration_allowance": "allowance-role"},
        "value_fit_identity_hash": None,
        "gain_scale_hash": None,
        "effective_policy_hash": None,
    }
    return InferenceBundle(
        state_dict={"decoder.mlp.0.weight": torch.ones(2, 2)},
        producer=_producer(),
        config=config,
        capability=capability,
        split_hash="split",
    )


def test_inference_bundle_roundtrip_and_capability_split_guards(tmp_path) -> None:
    path = tmp_path / "inference.pt"
    bundle = _bundle()
    save_inference_bundle(path, bundle)
    loaded = load_inference_bundle(path, expected_split_hash="split", required_capability="static")
    assert loaded.config == bundle.config
    assert torch.equal(loaded.state_dict["decoder.mlp.0.weight"], bundle.state_dict["decoder.mlp.0.weight"])
    with pytest.raises(FileExistsError):
        save_inference_bundle(path, bundle)
    with pytest.raises(ValueError, match="split hash"):
        load_inference_bundle(path, expected_split_hash="other")
    with pytest.raises(ValueError, match="insufficient"):
        load_inference_bundle(path, required_capability="adaptive")


def test_inference_artifact_tampering_and_unknown_payload_are_rejected(tmp_path) -> None:
    path = tmp_path / "inference.pt"
    save_inference_bundle(path, _bundle())
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["state_dict"]["decoder.mlp.0.weight"][0, 0] = 2.0
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    with pytest.raises(ValueError, match="digest"):
        load_inference_bundle(tampered)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["unknown"] = True
    unknown = tmp_path / "unknown.pt"
    torch.save(payload, unknown)
    with pytest.raises(ValueError, match="unknown"):
        load_inference_bundle(unknown)


def test_resume_roundtrip_preserves_stage_optimizer_rng_and_bank_state(tmp_path) -> None:
    path = tmp_path / "resume.pt"
    stage = StageState(stage="S1", epoch=2, update=7, microstep=0, optimizer_groups=("updater",))
    save_resume(
        path,
        _bundle(),
        stage,
        {"updater": {"step": 7, "exp_avg": torch.ones(2)}},
        {"torch_cpu": torch.arange(3, dtype=torch.long), "seed": 42},
        {"shard_cursor": 3, "index_hash": "index"},
    )
    loaded = load_resume(path)
    assert loaded.stage_state.stage == "S1"
    assert loaded.stage_state.update == 7
    assert torch.equal(loaded.optimizer_state["updater"]["exp_avg"], torch.ones(2))
    assert loaded.rng_state["seed"] == 42
    assert loaded.bank_state["shard_cursor"] == 3
    with pytest.raises(FileExistsError):
        save_resume(path, _bundle(), stage, {}, {}, {})


def test_resume_rejects_target_keys_and_legacy_requires_explicit_adapter(tmp_path) -> None:
    path = tmp_path / "resume.pt"
    with pytest.raises(ValueError, match="target"):
        save_resume(path, _bundle(), StageState(stage="S0"), {"target": torch.ones(1)}, {}, {})
    with pytest.raises(RuntimeError, match="explicit adapter"):
        load_legacy_inference_bundle(tmp_path / "missing.pt")


def test_value_artifact_roundtrip_is_v_only_and_strict(tmp_path) -> None:
    model = SignedValueNet(366)
    scale = GainScale(scale=1.0)
    identity = ValueFitIdentity(
        input_variant=366,
        architecture_hash=model.architecture_hash,
        weights_hash=module_state_digest(model),
        fit_config_hash="fit-config",
        bank_manifest_hash="bank-manifest",
        gain_scale_hash=scale.digest,
    )
    value_fit = SimpleNamespace(model=model, identity=identity, gain_scale=scale, complete=True)
    path = tmp_path / "value.pt"
    save_value_artifact(path, value_fit, producer=_producer(), config={"fit": "synthetic"}, completion={"complete": True, "stage": "value_fit"})
    loaded = load_value_artifact(path, expected_producer=_producer(), expected_input_variant=366)
    assert loaded.value_fit_identity == identity
    assert set(loaded.state_dict) == set(model.state_dict())
    assert loaded.producer.source_provenance.digest == _producer().source_provenance.digest
    with pytest.raises(FileExistsError):
        save_value_artifact(path, value_fit, producer=_producer(), config={"fit": "synthetic"})

    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["state_dict"]["net.0.weight"][0, 0] += 1.0
    tampered = tmp_path / "value-tampered.pt"
    torch.save(payload, tampered)
    with pytest.raises(ValueError, match="digest"):
        load_value_artifact(tampered)


def test_value_fit_result_roundtrip_binds_actual_fit_options_and_predictions(tmp_path) -> None:
    model = SignedValueNet(366)
    options = ValueFitOptions(batch_size=8, seed=17, learning_rate=2e-3)
    scale = GainScale(scale=1.0)
    identity = ValueFitIdentity(
        input_variant=366,
        architecture_hash=model.architecture_hash,
        weights_hash=module_state_digest(model),
        fit_config_hash=_fit_hash(options, model),
        bank_manifest_hash="bank-real-fit",
        gain_scale_hash=scale.digest,
    )
    # This mirrors W3a's concrete ValueFitResult fields while retaining a
    # synthetic source fixture; save_value_artifact derives the versioned fit
    # config/completion envelope from the actual options and model identity.
    value_fit = SimpleNamespace(
        model=model,
        input_variant=366,
        gain_scale=scale,
        identity=identity,
        complete=True,
        fit_options=options,
    )
    path = tmp_path / "value-fit-result.pt"
    save_value_artifact(path, value_fit, producer=_producer(), config={"source": "w3a-fit"})
    loaded = load_value_artifact(path)
    assert loaded.config["schema_version"] == "pfgr-lite-value-fit-config-v1"
    assert loaded.completion["schema_version"] == "pfgr-lite-value-completion-v1"
    restored = SignedValueNet(366)
    restored.load_state_dict(dict(loaded.state_dict), strict=True)
    descriptors = torch.randn(3, 366)
    with torch.no_grad():
        assert torch.equal(restored(descriptors), model(descriptors))

    # Metadata is part of the signed artifact envelope, not an advisory side
    # channel.  A replacement descriptor variant must fail before exposing V.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = json.loads(payload["metadata_json"])
    metadata["input_variant"] = 126
    tampered = tmp_path / "value-fit-result-metadata-tampered.pt"
    payload["metadata_json"] = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    torch.save(payload, tampered)
    with pytest.raises(ValueError, match="input_variant"):
        load_value_artifact(tampered)


def test_resume_roundtrip_restores_real_adam_and_all_rng_streams(tmp_path) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.01)
    parameter.grad = torch.tensor([0.5])
    optimizer.step()
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    path = tmp_path / "resume-real.pt"
    save_resume(path, _bundle(), StageState(stage="value_fit", update=1), optimizer.state_dict(), rng_state, {"cursor": 1})
    loaded = load_resume(path)
    assert loaded.optimizer_state["state"]
    assert isinstance(loaded.rng_state["numpy"], tuple)
    # Restoration is explicit and should reproduce the next draw from each
    # stream, not merely preserve a user-provided seed field.
    expected_python = random.Random()
    expected_python.setstate(rng_state["python"])
    expected_py_draw = expected_python.random()
    expected_numpy = np.random.RandomState()
    expected_numpy.set_state(rng_state["numpy"])
    expected_np_draw = expected_numpy.rand()
    expected_torch = torch.Generator()
    expected_torch.set_state(rng_state["torch_cpu"])
    expected_torch_draw = torch.rand((), generator=expected_torch)
    restore_rng_state(loaded.rng_state)
    assert random.random() == expected_py_draw
    assert np.random.rand() == expected_np_draw
    assert torch.rand(()) == expected_torch_draw


def test_hydrate_inference_model_uses_frontend_sidecar_and_strict_state(tmp_path) -> None:
    pfgr = PFGRLiteConfig(num_points=4, engineering_only=True)
    frontend = PointGuidedConfig(
        num_semantic_classes=3,
        num_points=4,
        point_candidate_multiplier=3,
        offset_hidden_channels=12,
        detach_backbone_features=False,
    )
    model = PFGRLiteModel(pfgr, frontend_config=frontend).eval()
    context = model.encode_observations(torch.randn(1, 3, 9, 9, 9), None, (1.0, 1.0, 1.0))
    config = {
        "schema_version": CHECKPOINT_CONFIG_SCHEMA,
        "pfgr_config": pfgr.as_dict(),
        "frontend_config": frontend_config_to_dict(frontend),
        "stage": "inference",
        "split_roles": {"producer_fit": "producer-role", "calibration_fit": "fit-role", "calibration_allowance": "allowance-role"},
        "value_fit_identity_hash": None,
        "gain_scale_hash": None,
        "effective_policy_hash": None,
    }
    bundle = InferenceBundle(
        state_dict=model.state_dict(),
        producer=context.producer,
        config=config,
        capability="static",
        split_hash="split",
        frontend_config=config["frontend_config"],
    )
    path = tmp_path / "model.pt"
    save_inference_bundle(path, bundle)
    loaded = load_inference_bundle(path)
    hydrated = hydrate_inference_model(loaded)
    assert set(hydrated.state_dict()) == set(model.state_dict())


def test_hydrated_model_roundtrips_target_free_decode_with_canonical_lattice(tmp_path) -> None:
    pfgr = PFGRLiteConfig(num_points=4, engineering_only=True, decode_chunk_size=40)
    frontend = PointGuidedConfig(
        num_semantic_classes=3,
        num_points=4,
        point_candidate_multiplier=3,
        offset_hidden_channels=12,
        detach_backbone_features=False,
    )
    model = PFGRLiteModel(pfgr, frontend_config=frontend, query_lattice_factory=PFGRQueryLattice).eval()
    torch.manual_seed(81)
    observations = torch.randn(1, 3, 9, 9, 9)
    context = model.encode_observations(observations, None, (1.0, 1.0, 1.0))
    state = model.initialize_state(context, role="deployment")
    config = {
        "schema_version": CHECKPOINT_CONFIG_SCHEMA,
        "pfgr_config": pfgr.as_dict(),
        "frontend_config": frontend_config_to_dict(frontend),
        "stage": "inference",
        "split_roles": {
            "producer_fit": "producer-role",
            "calibration_fit": "fit-role",
            "calibration_allowance": "allowance-role",
        },
        "value_fit_identity_hash": None,
        "gain_scale_hash": None,
        "effective_policy_hash": None,
    }
    bundle = InferenceBundle(
        state_dict=model.state_dict(),
        producer=context.producer,
        config=config,
        capability="static",
        split_hash="split",
        frontend_config=config["frontend_config"],
    )
    path = tmp_path / "hydrated-model.pt"
    save_inference_bundle(path, bundle)
    loaded = load_inference_bundle(path)
    hydrated = hydrate_inference_model(loaded, query_lattice_factory=PFGRQueryLattice).eval()
    hydrated_context = hydrated.encode_observations(observations, None, (1.0, 1.0, 1.0))
    hydrated_state = hydrated.initialize_state(hydrated_context, role="deployment")
    with torch.no_grad():
        expected = model.decode_final(state, context, chunk_size=17)
        actual = hydrated.decode_final(hydrated_state, hydrated_context, chunk_size=17)
    assert torch.equal(expected, actual)


def test_present_checkpoint_provenance_missing_source_channels_fails_closed() -> None:
    """A present checkpoint receipt may not fall back to the live 3-channel stem."""

    from smagm.features.point_guided.medicalnet_resnet10 import MedicalNetResNet10
    from smagm.features.point_guided.pfgr_lite.provenance import source_provenance_from_semantic_prior

    prior = SimpleNamespace(
        backbone=MedicalNetResNet10(in_channels=3),
        # Deliberately omit ``source_input_channels`` from a present receipt.
        # The provenance helper must reject this instead of reporting live 3.
        backbone_provenance=SimpleNamespace(
            adapted_input_channels=3,
            input_conv_adapted=False,
            official_pretrained_verified=False,
            integrity_verified=False,
            sha256="a" * 64,
        ),
    )
    with pytest.raises(ValueError, match="source_input_channels"):
        source_provenance_from_semantic_prior(prior)
