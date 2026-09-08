from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from smagm.features.point_guided.pfgr_lite.calibration_runner import CalibrationRunOptions, run_calibration
from smagm.features.point_guided.pfgr_lite.config import PFGRLiteConfig
from smagm.features.point_guided import PointGuidedConfig
from smagm.features.point_guided.contracts import VolumeGeometry
from smagm.features.point_guided.pfgr_lite.calibration_runner import _context_for_sample
from smagm.features.point_guided.pfgr_lite.data import TargetFreeSample
from smagm.features.point_guided.pfgr_lite.model import PFGRLiteModel


def test_calibration_options_default_to_exact_reference() -> None:
    options = CalibrationRunOptions()
    assert options.confirmation_mode == "exact"
    assert options.confirmation_q_draws == 0
    with pytest.raises(ValueError, match="confirmation_q_draws=0"):
        CalibrationRunOptions(confirmation_q_draws=2)
    with pytest.raises(ValueError, match="at least two"):
        CalibrationRunOptions(confirmation_mode="iid_fixed_q", confirmation_q_draws=1)


def test_synthetic_runner_is_complete_but_diagnostic_only(tmp_path: Path) -> None:
    inputs = SimpleNamespace(
        config=PFGRLiteConfig(engineering_only=True),
        execution=None,
        # Private numeric fixture: public CLI synthetic runs use the actual
        # model/teacher collection seam and remain insufficient-data only.
        metadata={"private_numeric_fixture": True},
    )
    result = run_calibration(
        inputs,
        CalibrationRunOptions(engineering_only=True, confirmation_mode="iid_fixed_q", confirmation_q_draws=4),
        tmp_path / "cal",
    )
    assert result["calibration"].capability == "diagnostic"
    assert result["metrics"]["fit_records"] >= 64
    assert result["metrics"]["allowance_records"] >= 64
    assert (tmp_path / "cal" / "calibration_evidence.json").is_file()
    assert result["calibration_evidence"].synthetic is True


def test_runner_rejects_unknown_options_and_missing_production_callbacks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown CalibrationRunOptions"):
        CalibrationRunOptions.from_dict({"unknown": 1})
    inputs = SimpleNamespace(
        config=PFGRLiteConfig(engineering_only=True),
        execution=None,
        role_manifest=None,
        samples=(),
        metadata={},
    )
    with pytest.raises(ValueError, match="TrainingRoleManifest"):
        run_calibration(inputs, CalibrationRunOptions(engineering_only=False), tmp_path / "missing")


def test_calibration_context_uses_one_resolved_model_device() -> None:
    config = PFGRLiteConfig(num_points=2, engineering_only=True, device="cpu")
    model = PFGRLiteModel(
        config,
        frontend_config=PointGuidedConfig(
            num_points=2,
            num_semantic_classes=3,
            point_candidate_multiplier=2,
            offset_hidden_channels=12,
        ),
    )
    sample = TargetFreeSample(
        "calibration-device",
        torch.zeros((3, 9, 9, 9), dtype=torch.float32),
        torch.ones((1, 9, 9, 9), dtype=torch.bool),
        VolumeGeometry.from_spacing((9, 9, 9)),
        {},
        "",
        "",
    )
    inputs = SimpleNamespace(config=config, execution=None, model=model)
    context = _context_for_sample(inputs, sample)
    assert context.initial_planes.xy.device == torch.device("cpu")
    assert next(model.parameters()).device == torch.device("cpu")
