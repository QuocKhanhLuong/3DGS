import pytest
import torch

from smagm.baselines.fixed_gaussian import (
    FixedGaussianHead,
    FixedGaussianHeadConfig,
    RawFixedGaussianOutput,
    construct_fixed_gaussians,
)
from smagm.baselines.fixed_support import FixedSupportBatch, FixedSupportConfig, sample_fixed_supports
from smagm.cli.t1a import run_demo
from smagm.contracts.coordinates import PhysicalPlane
from smagm.features.analytic import analytic_feature_bank
from smagm.features.contracts import EncoderFeatureMaps, FeatureGridToPlaneTransform
from smagm.gaussians import AmplitudeGaugePolicy
from smagm.renderer import RenderConfig, SlabProfile, render_plane


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
        source_plane_hashes=("a" * 64,) * 4,
        batch_index=0,
        support_basis_ras=torch.eye(3, dtype=dtype).expand(4, -1, -1),
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


def test_local_offsets_are_mapped_through_each_support_plane_basis() -> None:
    dtype = torch.float64
    supports = FixedSupportBatch(
        centers_ras_mm=torch.tensor([[10.0, 20.0, 30.0]], dtype=dtype),
        feature_vectors=torch.zeros((1, 1), dtype=dtype),
        feature_indices_vu=torch.tensor([[2, 3]], dtype=torch.long),
        reliability=torch.ones((1, 1), dtype=dtype),
        observation_ids=("rotated-obs",),
        source_plane_hashes=("b" * 64,),
        batch_index=0,
        support_basis_ras=torch.tensor([[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], dtype=dtype),
    )
    raw = RawFixedGaussianOutput(
        center_offset_raw=torch.atanh(torch.tensor([[0.5, 0.5, 0.5]], dtype=dtype)),
        covariance_raw=torch.zeros((1, 6), dtype=dtype),
        log_amplitude_raw=torch.zeros((1, 1), dtype=dtype),
        appearance_raw=torch.zeros((1, 1), dtype=dtype),
    )
    gaussians = construct_fixed_gaussians(
        supports,
        raw,
        config=FixedGaussianHeadConfig(input_dim=1, max_center_offset_mm=(2.0, 4.0, 6.0)),
    )
    assert torch.allclose(gaussians.centers_ras_mm, torch.tensor([[8.0, 21.0, 33.0]], dtype=dtype), atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_raw_constructor_path_has_finite_non_null_gradients_in_both_dtypes(dtype: torch.dtype) -> None:
    supports = _supports(dtype)
    raw = RawFixedGaussianOutput(
        center_offset_raw=torch.full((4, 3), 0.2, dtype=dtype, requires_grad=True),
        covariance_raw=torch.full((4, 6), 0.3, dtype=dtype, requires_grad=True),
        log_amplitude_raw=torch.tensor([[-0.5], [0.0], [0.5], [1.0]], dtype=dtype, requires_grad=True),
        appearance_raw=torch.full((4, 1), 0.4, dtype=dtype, requires_grad=True),
    )
    gaussians = construct_fixed_gaussians(supports, raw, config=FixedGaussianHeadConfig(input_dim=7))
    loss = (
        gaussians.centers_ras_mm.square().mean()
        + gaussians.covariance_factor.square().mean()
        + gaussians.log_support_amplitude.square().mean()
        + gaussians.appearance.square().mean()
    )
    loss.backward()
    for tensor in (raw.center_offset_raw, raw.covariance_raw, raw.log_amplitude_raw, raw.appearance_raw):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) > 0


def test_t1a_constructor_applies_amplitude_gauge_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import smagm.gaussians as gaussians_module

    calls = 0
    original = gaussians_module.fix_log_amplitude_gauge

    def instrumented(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(gaussians_module, "fix_log_amplitude_gauge", instrumented)
    supports = _supports()
    raw = RawFixedGaussianOutput(
        center_offset_raw=torch.zeros((4, 3), dtype=torch.float64),
        covariance_raw=torch.zeros((4, 6), dtype=torch.float64),
        log_amplitude_raw=torch.arange(4, dtype=torch.float64).reshape(4, 1),
        appearance_raw=torch.zeros((4, 1), dtype=torch.float64),
    )
    batch = construct_fixed_gaussians(supports, raw, config=FixedGaussianHeadConfig(input_dim=7))
    assert calls == 1
    assert torch.allclose(batch.log_support_amplitude.mean(), torch.zeros((), dtype=torch.float64), atol=1e-12)


def test_renderer_reconstruction_loss_backpropagates_through_t1a_evidence_and_head() -> None:
    """Exercise the live analytic-evidence → support → head → renderer path."""
    torch.manual_seed(19)
    dtype = torch.float64
    plane = PhysicalPlane(
        pixel_center_origin_ras_mm=(0.0, 0.0, 0.0),
        axis_u_ras=(1.0, 0.0, 0.0),
        axis_v_ras=(0.0, 1.0, 0.0),
        spacing_uv_mm=(1.0, 1.0),
        thickness_mm=1.0,
        shape_hw=(8, 8),
        signed_normal_ras=(0.0, 0.0, 1.0),
        observation_id="gradient-context",
    )
    v, u = torch.meshgrid(torch.arange(8, dtype=dtype), torch.arange(8, dtype=dtype), indexing="ij")
    image = (0.2 * u + 0.3 * v + 0.1 * torch.sin(u)).reshape(1, 1, 8, 8).requires_grad_()
    analytic = analytic_feature_bank(image, spacing_uv_mm=plane.spacing_uv_mm)
    analytic.tensor.retain_grad()
    features = EncoderFeatureMaps(
        structural=analytic.tensor[:, :5],
        appearance=analytic.tensor[:, :1],
        reliability=analytic.tensor[:, 7:8],
        grid_to_planes=(FeatureGridToPlaneTransform((8, 8), (8, 8), input_plane=plane),),
        modality_ids=("mri",),
        valid_feature_mask=analytic.valid_mask,
    )
    supports = sample_fixed_supports(features, plane, config=FixedSupportConfig(step_vu=(2, 2), border_vu=(1, 1)))
    supports.feature_vectors.retain_grad()
    head = FixedGaussianHead(FixedGaussianHeadConfig(input_dim=supports.feature_vectors.shape[1], min_scale_mm=1.5)).to(dtype=dtype)
    rendered = render_plane(
        construct_fixed_gaussians(supports, head(supports.feature_vectors), config=head.config),
        plane,
        config=RenderConfig(support_epsilon=1e-12, profile=SlabProfile.box(3)),
    )
    valid = ~rendered.unsupported_mask
    assert bool(valid.any())
    target = (0.1 * u + 0.05 * v).detach()
    loss = (rendered.intensity[valid] - target[valid]).square().mean()
    loss.backward()

    for tensor in (image, analytic.tensor, supports.feature_vectors):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) > 0
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() and torch.count_nonzero(parameter.grad) > 0
        for parameter in head.parameters()
    )
