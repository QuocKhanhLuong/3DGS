import torch

from smagm.baselines.fixed_gaussian import FixedGaussianHead, FixedGaussianHeadConfig, construct_fixed_gaussians
from smagm.baselines.fixed_support import FixedSupportBatch
from smagm.cli.t1a import run_demo
from smagm.gaussians import AmplitudeGaugePolicy


def _supports(dtype: torch.dtype = torch.float64) -> FixedSupportBatch:
    centers = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
        dtype=dtype,
    )
    features = torch.randn((4, 7), dtype=dtype, requires_grad=True)
    return FixedSupportBatch(
        centers_ras_mm=centers,
        feature_vectors=features,
        feature_indices_vu=torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long),
        reliability=torch.ones((4, 1), dtype=dtype),
        observation_ids=("obs",) * 4,
        batch_index=0,
    )


def test_constructor_produces_positive_triangular_factor_and_gauge() -> None:
    supports = _supports()
    config = FixedGaussianHeadConfig(input_dim=7, appearance_channels=2)
    head = FixedGaussianHead(config).to(dtype=torch.float64)
    raw = head(supports.feature_vectors)
    gaussians = construct_fixed_gaussians(supports, raw, config=config)
    gaussians.validate()
    diagonal = torch.diagonal(gaussians.covariance_factor, dim1=-2, dim2=-1)
    assert torch.all(diagonal > 0)
    assert torch.all(gaussians.covariance_factor.triu(diagonal=1) == 0)
    assert gaussians.gauge_policy is AmplitudeGaugePolicy.MEAN_CENTERED_LOG_AMPLITUDE_PER_PATIENT_STATE
    assert torch.allclose(gaussians.log_support_amplitude.mean(), torch.zeros((), dtype=torch.float64), atol=1e-12)
    assert gaussians.appearance.shape == (4, 2)


def test_gradient_reaches_features_and_head() -> None:
    supports = _supports()
    config = FixedGaussianHeadConfig(input_dim=7, appearance_channels=1)
    head = FixedGaussianHead(config).to(dtype=torch.float64)
    raw = head(supports.feature_vectors)
    gaussians = construct_fixed_gaussians(supports, raw, config=config)
    objective = (
        gaussians.centers_ras_mm.square().mean()
        + gaussians.covariance_factor.square().mean()
        + gaussians.appearance.square().mean()
        + gaussians.log_support_amplitude.square().mean()
    )
    objective.backward()
    assert supports.feature_vectors.grad is not None
    assert torch.isfinite(supports.feature_vectors.grad).all()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in head.parameters())


def test_executable_demo_runs_end_to_end() -> None:
    metrics = run_demo(steps=2, dtype=torch.float64)
    assert metrics["support_count"] > 0
    assert metrics["supported_fraction"] > 0.9
    assert metrics["gradient_norm"] > 0
    assert metrics["initial_loss"] >= 0
    assert metrics["final_loss"] >= 0
