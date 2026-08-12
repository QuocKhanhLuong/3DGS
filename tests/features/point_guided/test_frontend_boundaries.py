"""Regression tests for the locked frontend's access and state boundaries."""

from __future__ import annotations

import ast
import builtins
from pathlib import Path

import pytest
import torch

from smagm.features.point_guided import PointGuidedConfig, PointGuidedMRIModel
from smagm.features.point_guided.interfaces import (
    ReconstructionLossConfig,
    StoppingPolicyBase,
    TrajectoryHistory,
)
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
AUTHORIZED_PHASE67_INTERNAL_MODULES = frozenset(
    {
        "smagm.features.point_guided.swt_haar",
        "smagm.features.point_guided.spectral_anchor",
        "smagm.features.point_guided.spectral_query",
        "smagm.features.point_guided.cross_plane_consistency",
    }
)
FORBIDDEN_EXTERNAL_WAVELET_IMPORT_PREFIXES = (
    "pywt",
    "pywavelets",
    "pytorch_wavelets",
    "kymatio",
)
FORBIDDEN_GATE_C_IMPORT_PREFIXES = (
    "smagm.features.point_guided.dynamic_triplane",
    "smagm.features.point_guided.trajectory",
    "smagm.features.point_guided.selector",
    "smagm.features.point_guided.top_k",
    "smagm.features.point_guided.point_revisit",
    "smagm.features.point_guided.updater",
    "smagm.features.point_guided.scatter",
    "smagm.features.point_guided.overlap",
    "smagm.features.point_guided.history",
    "smagm.features.point_guided.stopping",
    "smagm.features.point_guided.decoder",
    "smagm.features.point_guided.losses",
    "smagm.features.point_guided.reconstruction_loss",
    "smagm.features.point_guided.spectral_loss",
    "smagm.features.point_guided.pathology_loss",
    "smagm.features.point_guided.training",
    "smagm.features.point_guided.reconstruction",
    "smagm.features.point_guided.synthesis",
)
POINT_GUIDED_PACKAGE = "smagm.features.point_guided"


def _starts_with_module(name: str, prefixes: tuple[str, ...] | frozenset[str]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def _is_authorized_phase67_module(name: str) -> bool:
    return _starts_with_module(name, AUTHORIZED_PHASE67_INTERNAL_MODULES)


def _is_forbidden_import_module(name: str) -> bool:
    """Keep the import boundary narrow without banning the word ``wavelet``."""

    normalized = name.lower()
    return (
        not _is_authorized_phase67_module(name)
        and (
            normalized == "importlib"
            or _starts_with_module(name, FORBIDDEN_IMPORT_PREFIXES)
            or _starts_with_module(normalized, FORBIDDEN_EXTERNAL_WAVELET_IMPORT_PREFIXES)
            or _starts_with_module(name, FORBIDDEN_GATE_C_IMPORT_PREFIXES)
            or _starts_with_module(name, ("torch.fft",))
        )
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


def test_future_only_types_cannot_construct_empty_runtime_state() -> None:
    for interface in (TrajectoryHistory, StoppingPolicyBase, ReconstructionLossConfig):
        with pytest.raises(TypeError):
            interface()


def test_frontend_static_import_boundary_allows_authorized_phase67_modules_only() -> None:
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
                if any(_is_forbidden_import_module(candidate) for candidate in candidates) or (
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
