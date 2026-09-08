from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import torch

from smagm.features.point_guided.pfgr_lite import (
    PFGRLiteConfig,
    PFGRLiteModel,
    ProducerCompatibility,
    ValueFitIdentity,
    batchnorm_state_digest,
    canonical_digest,
    module_state_digest,
    tensor_digest,
)
from smagm.features.point_guided import PointGuidedConfig


def test_tensor_and_module_hashes_include_dtype_shape_and_bytes() -> None:
    value = torch.tensor([1.0, 2.0], dtype=torch.float32)
    assert tensor_digest(value) != tensor_digest(value.to(torch.float64))
    module = torch.nn.BatchNorm3d(2)
    before = batchnorm_state_digest(module)
    module.running_mean[0] = 1.0
    assert batchnorm_state_digest(module) != before
    assert module_state_digest(module) != canonical_digest("module")


def test_producer_compatibility_excludes_value_fit_identity() -> None:
    complete = dict(
        observation_normalization_hash="norm",
        geometry_query_version_hash="geometry",
        medicalnet_provenance_hash="medicalnet",
        frozen_bn_hash="bn",
        static_head_hash="static-a",
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
    producer = ProducerCompatibility(**complete)
    changed_v = ValueFitIdentity(input_variant=126, architecture_hash="v1", weights_hash="w1", fit_config_hash="f1", bank_manifest_hash="b1", gain_scale_hash="s1")
    changed_v2 = ValueFitIdentity(input_variant=126, architecture_hash="v2", weights_hash="w2", fit_config_hash="f2", bank_manifest_hash="b1", gain_scale_hash="s1")
    assert producer.digest == ProducerCompatibility(**complete).digest
    assert producer.digest == ProducerCompatibility(**complete).digest
    assert changed_v.digest != changed_v2.digest
    assert producer.digest != changed_v.digest


def test_context_provenance_reports_one_traversal_and_synthetic_weights() -> None:
    frontend = PointGuidedConfig(num_semantic_classes=3, num_points=2, point_candidate_multiplier=2, offset_hidden_channels=12)
    model = PFGRLiteModel(PFGRLiteConfig(num_points=2, engineering_only=True), frontend_config=frontend).eval()
    context = model.encode_observations(torch.randn(1, 3, 7, 7, 7), None, (1.0, 1.0, 1.0))
    assert context.producer.source_provenance.synthetic_untrained
    assert context.producer.source_provenance.traversal_count == 1
    assert context.producer.compatibility.digest == context.producer.digest


def test_local_one_channel_checkpoint_reports_adaptation_and_conditional_mean_stem(tmp_path) -> None:
    from smagm.features.point_guided.medicalnet_resnet10 import MedicalNetResNet10

    source_backbone = MedicalNetResNet10(in_channels=1)
    checkpoint = tmp_path / "synthetic-medicalnet.pt"
    torch.save(source_backbone.state_dict(), checkpoint)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    frontend = PointGuidedConfig(
        num_semantic_classes=3,
        num_points=2,
        point_candidate_multiplier=2,
        offset_hidden_channels=12,
        medicalnet_checkpoint_path=checkpoint,
        medicalnet_checkpoint_sha256=checkpoint_sha,
    )
    model = PFGRLiteModel(PFGRLiteConfig(num_points=2, engineering_only=True), frontend_config=frontend).eval()
    source = model.frontend.semantic_prior.backbone_provenance
    assert source is not None
    context = model.encode_observations(torch.randn(1, 3, 7, 7, 7), None, (1.0, 1.0, 1.0))
    provenance = context.source_provenance
    assert provenance.source_input_channels == 1
    assert provenance.adapted_input_channels == 3
    assert provenance.input_conv_adapted is True
    assert provenance.checkpoint_sha256 == checkpoint_sha
    assert provenance.checkpoint_integrity_verified is True
    assert provenance.official_pretrained_verified is False
    assert provenance.synthetic_untrained is True

    base = torch.randn(1, 1, 9, 9, 9)
    perturbation = torch.randn_like(base) * 0.1
    ordered = torch.cat((base, base, base), dim=1)
    mean_preserving = torch.cat((base + perturbation, base - perturbation, base), dim=1)
    with torch.no_grad():
        ordered_features = model.frontend.semantic_prior.backbone.forward_intermediate_features(ordered)
        mean_preserving_features = model.frontend.semantic_prior.backbone.forward_intermediate_features(mean_preserving)
    for left, right in zip(
        (ordered_features.shallow, ordered_features.layer1, ordered_features.deep),
        (mean_preserving_features.shallow, mean_preserving_features.layer1, mean_preserving_features.deep),
    ):
        assert torch.allclose(left, right, atol=1e-5, rtol=1e-5)


def test_independent_three_channel_stem_remains_order_sensitive() -> None:
    from smagm.features.point_guided.medicalnet_resnet10 import MedicalNetResNet10

    backbone = MedicalNetResNet10(in_channels=3).eval()
    base = torch.randn(1, 1, 7, 7, 7)
    first = torch.cat((base, torch.zeros_like(base), torch.zeros_like(base)), dim=1)
    second = torch.cat((torch.zeros_like(base), base, torch.zeros_like(base)), dim=1)
    with torch.no_grad():
        first_features = backbone.forward_intermediate_features(first).shallow
        second_features = backbone.forward_intermediate_features(second).shallow
    assert not torch.equal(first_features, second_features)


def test_producer_identity_excludes_source_sha_and_value_fit_settings() -> None:
    complete = dict(
        observation_normalization_hash="norm",
        geometry_query_version_hash="geometry",
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
    left = ProducerCompatibility(**complete, source_version="pfgr-lite-v1", component_versions=(("git", "a"), ("v", "one")))
    right = ProducerCompatibility(**complete, source_version="pfgr-lite-v1", component_versions=(("git", "b"), ("v", "two")))
    assert left.digest == right.digest

    # Scoped bank compatibility intentionally uses the component fields, not
    # SourceProvenance.digest (which includes source SHA/traversal metadata).
    from smagm.features.point_guided.pfgr_lite.provenance import SourceProvenance

    source_a = SourceProvenance(source_sha="a", config_sha="cfg", parameter_hash="weights", frozen_bn_hash="bn")
    source_b = SourceProvenance(source_sha="b", config_sha="cfg2", parameter_hash="weights", frozen_bn_hash="bn")
    scoped_a = canonical_digest({"parameter_hash": source_a.parameter_hash, "frozen_bn_hash": source_a.frozen_bn_hash}, prefix="pfgr-lite-medicalnet-producer-v1|")
    scoped_b = canonical_digest({"parameter_hash": source_b.parameter_hash, "frozen_bn_hash": source_b.frozen_bn_hash}, prefix="pfgr-lite-medicalnet-producer-v1|")
    assert scoped_a == scoped_b


def test_executable_source_manifest_covers_imports_and_excludes_docs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Manifest identity is deterministic, byte-addressed, and scoped."""

    from smagm.cli import pfgr_lite as cli

    (tmp_path / "src" / "legacy").mkdir(parents=True)
    imported = tmp_path / "src" / "legacy" / "imported.py"
    imported.write_text("VALUE = 7\n", encoding="utf-8")
    (tmp_path / "src" / "legacy" / "notes.md").write_text("not executable scope\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "design.md").write_text("excluded\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_SOURCE_SCOPE_ROOTS", ("src/",))
    monkeypatch.setattr(cli, "_SOURCE_SCOPE_SUFFIXES", {".py"})

    paths = cli._scoped_source_paths(tmp_path)
    assert paths == ["src/legacy/imported.py"]
    first = cli._source_manifest(tmp_path, paths)
    second = cli._source_manifest(tmp_path, paths)
    assert first == second
    assert first[0] == {
        "path": "src/legacy/imported.py",
        "size": imported.stat().st_size,
        "sha256": hashlib.sha256(imported.read_bytes()).hexdigest(),
    }
    # The manifest carries metadata only, never source bytes or unrelated docs.
    assert b"VALUE = 7" not in json.dumps(first).encode("utf-8")
    imported.write_text("VALUE = 8\n", encoding="utf-8")
    mutated = cli._source_manifest(tmp_path, paths)
    assert mutated != first
    assert mutated[0]["sha256"] != first[0]["sha256"]


def test_source_receipt_clean_and_mutation_boundaries_use_real_git_scope(tmp_path: Path) -> None:
    """Exercise tracked/imported/untracked/ignored/docs changes in a clean repo."""

    from smagm.cli import pfgr_lite as cli

    src_root = tmp_path / "src"
    src_root.mkdir()
    legacy = src_root / "legacy.py"
    legacy.write_text("VALUE = 1\n", encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    doc_path = docs / "design.md"
    doc_path.write_text("design-v1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "pfgr-tests@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "PFGR Tests"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "src/legacy.py", "docs/design.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)

    clean = cli._source_receipt(tmp_path)
    repeated = cli._source_receipt(tmp_path)
    assert clean["provenance_status"] == "CLEAN"
    assert clean["whole_tree_dirty"] is False
    assert clean["source_scope_manifest_complete"] is True
    assert clean["executable_manifest_sha256"] == repeated["executable_manifest_sha256"]
    clean_manifest = clean["executable_manifest_sha256"]

    # Imported legacy source bytes are executable identity and must alter
    # the manifest/digest even though no new Git path is introduced.
    legacy.write_text("VALUE = 2\n", encoding="utf-8")
    changed_source = cli._source_receipt(tmp_path)
    assert changed_source["provenance_status"] == "ENGINEERING_NONFINAL"
    assert changed_source["executable_manifest_sha256"] != clean_manifest

    subprocess.run(["git", "checkout", "--", "src/legacy.py"], cwd=tmp_path, check=True)
    untracked = src_root / "imported_untracked.py"
    untracked.write_text("VALUE = 3\n", encoding="utf-8")
    untracked_receipt = cli._source_receipt(tmp_path)
    assert untracked_receipt["source_scope_untracked_paths"] == ["src/imported_untracked.py"]
    assert untracked_receipt["provenance_status"] == "ENGINEERING_NONFINAL"
    assert untracked_receipt["executable_manifest_sha256"] != clean_manifest
    untracked.unlink()

    # Commit the ignore rule and keep the only remaining source mutation
    # ignored.  This proves an ignored executable remains in the manifest even
    # when Git reports a clean whole tree.
    (tmp_path / ".gitignore").write_text("src/ignored.py\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "ignore rule"], cwd=tmp_path, check=True)
    ignored = src_root / "ignored.py"
    ignored.write_text("VALUE = 4\n", encoding="utf-8")
    ignored_receipt = cli._source_receipt(tmp_path)
    assert ignored_receipt["whole_tree_dirty"] is False
    assert ignored_receipt["source_scope_ignored_paths"] == ["src/ignored.py"]
    assert ignored_receipt["provenance_status"] == "ENGINEERING_NONFINAL"
    assert ignored_receipt["executable_manifest_sha256"] != clean_manifest

    # Documentation is intentionally outside the executable scope.  Its
    # mutation may dirty the whole tree but cannot change scoped bytes.
    before_doc_scope = cli._source_receipt(tmp_path)
    doc_path.write_text("design-v2\n", encoding="utf-8")
    after_doc_scope = cli._source_receipt(tmp_path)
    assert after_doc_scope["source_scope_sha256"] == before_doc_scope["source_scope_sha256"]
    assert after_doc_scope["executable_manifest_sha256"] == before_doc_scope["executable_manifest_sha256"]


def test_source_receipt_exposes_whole_tree_dirty_and_nonfinal_status() -> None:
    from smagm.cli.pfgr_lite import _source_receipt

    receipt = _source_receipt()
    assert "whole_tree_dirty" in receipt
    assert "executable_manifest" in receipt
    assert "source_scope_manifest_complete" in receipt
    # A clean CI checkout is production-verifiable; local uncommitted or
    # untracked source remains explicitly engineering/non-final.
    assert receipt["provenance_status"] in {"CLEAN", "ENGINEERING_NONFINAL"}
    if receipt["whole_tree_dirty"]:
        assert receipt["provenance_status"] == "ENGINEERING_NONFINAL"
