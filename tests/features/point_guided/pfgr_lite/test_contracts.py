from __future__ import annotations

import pytest
import torch

from smagm.features.point_guided import PointGuidedConfig
from smagm.features.point_guided.pfgr_lite import (
    ActionProposalBatch,
    EffectTeacherConfig,
    GainCalibration,
    PFGRLiteConfig,
    PFGRPolicyConfig,
    StaticSynthesisConfig,
    ValueFitIdentity,
)


def test_locked_pfgr_configuration_defaults_and_schema() -> None:
    config = PFGRLiteConfig()
    assert config.schema_version == "pfgr-lite-config-v1"
    assert config.candidate_count == 2048
    assert config.state_channels == 32
    assert config.correction_channels == 96
    assert config.write_scale == pytest.approx(0.1)
    assert config.policy.budgets == (0, 1, 2, 4)
    assert config.static.variant == "b2_ordered_multiscale_v1"
    assert config.value.input_variants == (126, 222, 270, 366)


@pytest.mark.parametrize(
    ("factory", "message"),
    (
        (lambda: PFGRPolicyConfig(candidate_count=4), "candidate_count"),
        (lambda: PFGRLiteConfig(write_scale=0.2), "write_scale"),
        (lambda: EffectTeacherConfig(mode="iid_fixed_q", q_draws=1), "q_draws"),
        (lambda: StaticSynthesisConfig(variant="unknown"), "unknown"),
    ),
)
def test_pfgr_contract_rejects_unlocked_values(factory, message: str) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        factory()


def test_action_proposal_batch_detects_tensor_mutation() -> None:
    dtype = torch.float64
    batch = ActionProposalBatch(
        context_id="ctx",
        state_version=0,
        state_digest="state",
        point_ids=torch.tensor([[0]], dtype=torch.long),
        points_ras_mm=torch.zeros(1, 1, 3, dtype=dtype),
        o270=torch.zeros(1, 1, 270, dtype=dtype),
        v126=torch.zeros(1, 1, 126, dtype=dtype),
        delta=torch.zeros(1, 1, 96, dtype=dtype),
        legal=torch.ones(1, 1, dtype=torch.bool),
    )
    batch.delta[0, 0, 0] = 1.0
    with pytest.raises(RuntimeError, match="mutation"):
        batch.validate_integrity()


def test_calibration_and_value_identity_are_distinct() -> None:
    value = ValueFitIdentity(input_variant=366, architecture_hash="a", weights_hash="w", fit_config_hash="f", bank_manifest_hash="b", gain_scale_hash="s")
    calibration = GainCalibration(a=1.0, b=0.0, allowance=0.0, producer_compatibility_hash="p", value_fit_identity_hash=value.digest, gain_scale_hash="s")
    assert value.digest != calibration.version
    assert calibration.allowance >= 0.0

