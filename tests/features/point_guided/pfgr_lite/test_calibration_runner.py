from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from smagm.features.point_guided.pfgr_lite.calibration_runner import CalibrationRunOptions, run_calibration
from smagm.features.point_guided.pfgr_lite.config import PFGRLiteConfig


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
