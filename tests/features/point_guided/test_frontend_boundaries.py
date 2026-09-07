"""Regression tests for the locked frontend's access and state boundaries."""

from __future__ import annotations

import ast
import builtins
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch

from smagm.features.point_guided import (
    FrontendOutput,
    PointGuidedConfig,
    PointGuidedMRIModel,
)
from smagm.features.point_guided.contracts import PointSpectralEvidence
from smagm.features.point_guided.interfaces import (
    ReconstructionLossConfig,
    StoppingPolicyBase,
    TrajectoryHistory,
)
from smagm.features.point_guided.spectral_anchor import SpectralAnchor
from smagm.features.point_guided.triplane_projection import BaseTriPlanes


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "src" / "smagm" / "features" / "point_guided"
FORBIDDEN_IMPORT_PREFIXES = (
    "smagm.anchors",
    "smagm.fields",
    "smagm.memory",
    "smagm.routing",
    "smagm.training",
    "smagm.evaluation",
    "smagm.reconstruction",
    "smagm.cli",
    "smagm.data",
)
AUTHORIZED_FRONTEND_INTERNAL_MODULES = frozenset(
    {
        "smagm.features.point_guided.swt_haar",
        "smagm.features.point_guided.spectral_anchor",
        "smagm.features.point_guided.spectral_query",
        "smagm.features.point_guided.cross_plane_consistency",
        "smagm.features.point_guided.state_init",
        "smagm.features.point_guided.reward",
        "smagm.features.point_guided.trajectory_cost",
        "smagm.features.point_guided.trajectory_solver",
        "smagm.features.point_guided.updater",
        "smagm.features.point_guided.writeback",
        "smagm.features.point_guided.trajectory",
        "smagm.features.point_guided.decoder",
        "smagm.features.point_guided.losses",
        "smagm.features.point_guided.reward_supervision",
        "smagm.features.point_guided.training_objective",
        "smagm.features.point_guided.availability",
        "smagm.features.point_guided.baseline_inference",
        "smagm.features.point_guided.baseline_training",
    }
)
FORBIDDEN_EXTERNAL_WAVELET_IMPORT_PREFIXES = (
    "pywt",
    "pywavelets",
    "pytorch_wavelets",
    "kymatio",
)
FORBIDDEN_INACTIVE_GATE_IMPORT_PREFIXES = (
    "smagm.features.point_guided.dynamic_triplane",
    "smagm.features.point_guided.selector",
    "smagm.features.point_guided.top_k",
    "smagm.features.point_guided.point_revisit",
    "smagm.features.point_guided.scatter",
    "smagm.features.point_guided.overlap",
    "smagm.features.point_guided.history",
    "smagm.features.point_guided.stopping",
    "smagm.features.point_guided.training",
    "smagm.features.point_guided.reconstruction",
    "smagm.features.point_guided.synthesis",
)
POINT_GUIDED_PACKAGE = "smagm.features.point_guided"
STAGED_LOADER_FILES = frozenset({PACKAGE / "pfgr_lite" / "data.py", PACKAGE / "pfgr_lite" / "stages.py"})
STAGED_LOADER_NAMES = frozenset({"load_point_guided_subject", "load_point_guided_split"})


def _starts_with_module(name: str, prefixes: tuple[str, ...] | frozenset[str]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def _is_authorized_frontend_module(name: str) -> bool:
    return _starts_with_module(name, AUTHORIZED_FRONTEND_INTERNAL_MODULES)


def _is_forbidden_import_module(name: str) -> bool:
    """Keep the import boundary narrow without banning the word ``wavelet``."""

    normalized = name.lower()
    return (
        not _is_authorized_frontend_module(name)
        and (
            normalized == "importlib"
            or _starts_with_module(name, FORBIDDEN_IMPORT_PREFIXES)
            or _starts_with_module(normalized, FORBIDDEN_EXTERNAL_WAVELET_IMPORT_PREFIXES)
            or _starts_with_module(name, FORBIDDEN_INACTIVE_GATE_IMPORT_PREFIXES)
            or _starts_with_module(name, ("torch.fft",))
        )
    )


def _is_authorized_staged_loader_import(path: Path, node: ast.ImportFrom) -> bool:
    """Allow only the two reviewed observation/split loader symbols in W3b."""

    return (
        path in STAGED_LOADER_FILES
        and node.level == 0
        and node.module == "smagm.data.brats21_point_guided"
        and bool(node.names)
        and all(alias.name in STAGED_LOADER_NAMES for alias in node.names)
    )


def _import_from_candidates(node: ast.ImportFrom) -> tuple[str, ...]:
    """Expand only local imports needed to recognize authorized/blocked modules."""

    candidates: list[str] = []
    if node.module is not None:
        candidates.append(node.module)
        if node.level == 1:
            candidates.append(f"{POINT_GUIDED_PACKAGE}.{node.module}")
        elif node.module == POINT_GUIDED_PACKAGE:
            candidates.extend(f"{node.module}.{alias.name}" for alias in node.names)
    elif node.level == 1:
        candidates.extend(f"{POINT_GUIDED_PACKAGE}.{alias.name}" for alias in node.names)
    return tuple(candidates)


def _small_model() -> PointGuidedMRIModel:
    return PointGuidedMRIModel(
        PointGuidedConfig(
            num_semantic_classes=3,
            num_points=3,
            point_candidate_multiplier=3,
            offset_hidden_channels=12,
        )
    )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    (
        ("support_radius_mm", 3.99, "exactly 4.0 mm"),
        ("support_radius_mm", 4.01, "exactly 4.0 mm"),
        ("max_displacement_mm", 2.01, "must not exceed"),
    ),
)
def test_locked_physical_constants_cannot_be_relaxed(
    keyword: str,
    value: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PointGuidedConfig(num_semantic_classes=3, **{keyword: value})


def test_frontend_rejects_a_fourth_target_channel() -> None:
    model = _small_model().eval()
    with pytest.raises(ValueError, match="exactly T1/T2/FLAIR channels"):
        model.forward_frontend(torch.randn(1, 4, 7, 7, 7))


def test_frontend_forward_performs_no_checkpoint_or_filesystem_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _small_model().eval()

    def fail_io(*args: object, **kwargs: object) -> object:
        raise AssertionError("frontend forward must not perform filesystem or checkpoint I/O")

    monkeypatch.setattr(builtins, "open", fail_io)
    monkeypatch.setattr(Path, "open", fail_io)
    monkeypatch.setattr(torch, "load", fail_io)
    with torch.no_grad():
        output = model.forward_frontend(torch.randn(1, 3, 7, 7, 7))
    assert output.s_coarse.shape == (1, 3, 7, 7, 7)
    assert isinstance(output.base_planes, BaseTriPlanes)
    assert output.f_spec.shape == (1, 3, 168)


def test_frontend_does_not_persist_patient_state_or_mutate_inputs() -> None:
    torch.manual_seed(19)
    model = _small_model().train()
    first_patient = torch.randn(1, 3, 7, 7, 7)
    second_patient = torch.randn(1, 3, 7, 7, 7)
    first_before = first_patient.clone()
    second_before = second_patient.clone()
    state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    with torch.no_grad():
        first_output = model.forward_frontend(first_patient)
        model.forward_frontend(second_patient)
        first_replayed = model.forward_frontend(first_patient)

    assert torch.equal(first_patient, first_before)
    assert torch.equal(second_patient, second_before)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state_before.items())
    torch.testing.assert_close(first_output.s_coarse, first_replayed.s_coarse)
    torch.testing.assert_close(first_output.refined_points, first_replayed.refined_points)
    torch.testing.assert_close(first_output.sparse_pou.normalized_weight, first_replayed.sparse_pou.normalized_weight)
    torch.testing.assert_close(first_output.base_planes.xy, first_replayed.base_planes.xy)
    torch.testing.assert_close(first_output.base_planes.xz, first_replayed.base_planes.xz)
    torch.testing.assert_close(first_output.base_planes.yz, first_replayed.base_planes.yz)
    torch.testing.assert_close(first_output.spectral_anchor.xy, first_replayed.spectral_anchor.xy)
    torch.testing.assert_close(first_output.spectral_anchor.xz, first_replayed.spectral_anchor.xz)
    torch.testing.assert_close(first_output.spectral_anchor.yz, first_replayed.spectral_anchor.yz)
    torch.testing.assert_close(first_output.f_spec, first_replayed.f_spec)
    torch.testing.assert_close(first_output.reliability, first_replayed.reliability)


def test_frontend_preserves_a_float64_contract() -> None:
    model = _small_model().double().eval()
    with torch.no_grad():
        output = model.forward_frontend(torch.randn(1, 3, 7, 7, 7, dtype=torch.float64))
    assert output.s_coarse.dtype == torch.float64
    assert output.refined_points.dtype == torch.float64
    assert output.sparse_pou.raw_affinity.dtype == torch.float64
    assert output.base_planes.xy.dtype == torch.float64
    assert output.base_planes.xz.dtype == torch.float64
    assert output.base_planes.yz.dtype == torch.float64
    assert output.spectral_anchor.xy.dtype == torch.float64
    assert output.spectral_anchor.xz.dtype == torch.float64
    assert output.spectral_anchor.yz.dtype == torch.float64
    assert output.f_spec.dtype == torch.float64
    assert output.reliability.dtype == torch.float64


def test_frontend_output_rejects_an_anchor_with_a_grid_mismatched_to_base_planes() -> None:
    model = _small_model().eval()
    with torch.no_grad():
        output = model.forward_frontend(torch.randn(1, 3, 7, 7, 7))
    mismatched = SpectralAnchor(
        xy=output.spectral_anchor.xy[..., :-1, :-1],
        xz=output.spectral_anchor.xz[..., :-1],
        yz=output.spectral_anchor.yz[..., :-1],
    )
    with pytest.raises(ValueError, match="retain its base-plane grid"):
        replace(output, spectral_anchor=mismatched)


def test_frontend_output_allows_a_lower_precision_static_anchor_with_float32_evidence() -> None:
    model = _small_model().eval()
    with torch.no_grad():
        output = model.forward_frontend(torch.randn(1, 3, 7, 7, 7))

    mixed_precision_anchor = SpectralAnchor(
        xy=output.spectral_anchor.xy.to(dtype=torch.float16),
        xz=output.spectral_anchor.xz.to(dtype=torch.float16),
        yz=output.spectral_anchor.yz.to(dtype=torch.float16),
    )
    reconstructed = replace(output, spectral_anchor=mixed_precision_anchor)

    assert reconstructed.s_coarse.dtype == torch.float32
    assert reconstructed.base_planes.xy.dtype == torch.float32
    assert reconstructed.spectral_anchor.xy.dtype == torch.float16
    assert reconstructed.f_spec.dtype == torch.float32
    assert reconstructed.reliability.dtype == torch.float32


def test_spectral_anchor_remains_strict_about_its_internal_dtype_contract() -> None:
    model = _small_model().eval()
    with torch.no_grad():
        output = model.forward_frontend(torch.randn(1, 3, 7, 7, 7))

    with pytest.raises(ValueError, match="share one dtype"):
        SpectralAnchor(
            xy=output.spectral_anchor.xy,
            xz=output.spectral_anchor.xz.double(),
            yz=output.spectral_anchor.yz,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device diversity")
def test_spectral_anchor_remains_strict_about_its_internal_device_contract() -> None:
    model = _small_model().eval()
    with torch.no_grad():
        output = model.forward_frontend(torch.randn(1, 3, 7, 7, 7))

    with pytest.raises(ValueError, match="share one device"):
        SpectralAnchor(
            xy=output.spectral_anchor.xy.cuda(),
            xz=output.spectral_anchor.xz.cuda(),
            yz=output.spectral_anchor.yz,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA AMP")
def test_frontend_output_accepts_the_real_cuda_amp_anchor_and_evidence_dtypes() -> None:
    model = _small_model().cuda().train()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model.forward_frontend(torch.randn(1, 3, 7, 7, 7, device="cuda", dtype=torch.float32))

    assert output.s_coarse.dtype == torch.float32
    assert output.base_planes.xy.dtype == torch.float32
    assert output.spectral_anchor.xy.dtype == torch.float16
    assert output.f_spec.dtype == torch.float32
    assert output.reliability.dtype == torch.float32


def test_frontend_output_fails_closed_for_invalid_point_spectral_evidence() -> None:
    model = _small_model().eval()
    with torch.no_grad():
        output = model.forward_frontend(torch.randn(1, 3, 7, 7, 7))

    with pytest.raises(TypeError, match="PointSpectralEvidence"):
        replace(output, spectral_evidence=None)  # type: ignore[arg-type]

    truncated = PointSpectralEvidence(
        f_spec=output.f_spec[:, :1],
        reliability=output.reliability[:, :1],
    )
    with pytest.raises(ValueError, match="align with the refined point"):
        replace(output, spectral_evidence=truncated)

    wrong_dtype = PointSpectralEvidence(
        f_spec=output.f_spec.double(),
        reliability=output.reliability.double(),
    )
    with pytest.raises(ValueError, match="dtype must match s_coarse"):
        replace(output, spectral_evidence=wrong_dtype)


def test_gate_d_and_later_type_only_interfaces_cannot_construct_runtime_state() -> None:
    for interface in (TrajectoryHistory, StoppingPolicyBase, ReconstructionLossConfig):
        with pytest.raises(TypeError):
            interface()


def test_phase7_public_output_remains_typed_point_evidence_without_gate_c_state() -> None:
    names = tuple(field.name for field in fields(FrontendOutput))
    assert names == (
        "s_coarse",
        "initial_points_ras_mm",
        "refined_points_ras_mm",
        "displacement_ras_mm",
        "point_semantic",
        "sparse_pou",
        "geometry",
        "base_planes",
        "spectral_anchor",
        "spectral_evidence",
    )
    assert not {
        "query_coordinates",
        "feature_grid_geometry",
        "trajectory",
        "dynamic_triplane",
        "selector",
        "reconstruction",
    }.intersection(names)
    assert isinstance(getattr(FrontendOutput, "f_spec"), property)
    assert isinstance(getattr(FrontendOutput, "reliability"), property)


def test_gate_g_software_policy_is_active_but_heldout_modules_remain_absent() -> None:
    assert (PACKAGE / "baseline_inference.py").exists()
    for filename in (
        "gate_g.py",
        "heldout_evaluation.py",
    ):
        assert not (PACKAGE / filename).exists()


def test_frontend_static_import_boundary_allows_only_completed_gates_and_active_gate_g_modules() -> None:
    violations: list[str] = []
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_forbidden_import_module(alias.name):
                        violations.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                candidates = _import_from_candidates(node)
                staged_loader_exception = _is_authorized_staged_loader_import(path, node)
                if (not staged_loader_exception and any(_is_forbidden_import_module(candidate) for candidate in candidates)) or (
                    node.module == "torch" and any(alias.name == "fft" for alias in node.names)
                ):
                    rendered = node.module if node.module is not None else "."
                    violations.append(f"{path.name}: from {rendered}")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {"__import__", "eval", "exec"}:
                    violations.append(f"{path.name}: dynamic {node.func.id}")
                elif isinstance(node.func, ast.Attribute) and node.func.attr in {"import_module", "reload"}:
                    violations.append(f"{path.name}: dynamic {node.func.attr}")
            elif (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "torch"
                and node.attr == "fft"
            ):
                violations.append(f"{path.name}: torch.fft")
    assert not violations, "\n".join(violations)


def test_staged_loader_exception_is_limited_to_exact_w3b_files_and_symbols() -> None:
    allowed = ast.parse("from smagm.data.brats21_point_guided import load_point_guided_subject").body[0]
    assert isinstance(allowed, ast.ImportFrom)
    assert _is_authorized_staged_loader_import(PACKAGE / "pfgr_lite" / "data.py", allowed)
    assert not _is_authorized_staged_loader_import(PACKAGE / "model.py", allowed)
    disallowed_symbol = ast.parse("from smagm.data.brats21_point_guided import load_target").body[0]
    assert isinstance(disallowed_symbol, ast.ImportFrom)
    assert not _is_authorized_staged_loader_import(PACKAGE / "pfgr_lite" / "data.py", disallowed_symbol)
