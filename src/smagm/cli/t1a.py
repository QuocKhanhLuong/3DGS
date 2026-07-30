"""Run the executable T1-A synthetic vertical slice.

This is a reference-contract demo, not an accuracy baseline or paper result.  It
proves that analytic evidence can be sampled at deterministic physical support
points, mapped through a safe Gaussian head, rendered on a target plane, and
optimized end to end.
"""

from __future__ import annotations

import argparse

import torch

from ..baselines.fixed_gaussian import FixedGaussianHead, FixedGaussianHeadConfig, construct_fixed_gaussians
from ..baselines.fixed_support import FixedSupportConfig, sample_fixed_supports
from ..contracts.coordinates import PhysicalPlane
from ..features.analytic import analytic_feature_bank
from ..features.contracts import EncoderFeatureMaps, FeatureGridToPlaneTransform
from ..renderer import RenderConfig, SlabProfile, render_plane


def _plane(*, z_mm: float, observation_id: str, shape_hw: tuple[int, int]) -> PhysicalPlane:
    return PhysicalPlane(
        pixel_center_origin_ras_mm=(0.0, 0.0, z_mm),
        axis_u_ras=(1.0, 0.0, 0.0),
        axis_v_ras=(0.0, 1.0, 0.0),
        spacing_uv_mm=(1.0, 1.0),
        thickness_mm=1.0,
        shape_hw=shape_hw,
        signed_normal_ras=(0.0, 0.0, 1.0),
        observation_id=observation_id,
    )


def _synthetic_image(shape_hw: tuple[int, int], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    height, width = shape_hw
    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    blob_a = torch.exp(-((u - 0.35 * width) ** 2 + (v - 0.45 * height) ** 2) / (2.0 * (0.12 * width) ** 2))
    blob_b = 0.65 * torch.exp(-((u - 0.68 * width) ** 2 + (v - 0.58 * height) ** 2) / (2.0 * (0.09 * width) ** 2))
    ridge = 0.2 * torch.sin(u / 4.0) * torch.exp(-((v - 0.5 * height) ** 2) / (2.0 * (0.18 * height) ** 2))
    return (blob_a + blob_b + ridge).unsqueeze(0).unsqueeze(0)


def run_demo(*, steps: int = 4, device: str = "cpu", dtype: torch.dtype = torch.float32) -> dict[str, float]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    torch.manual_seed(7)
    torch_device = torch.device(device)
    shape_hw = (32, 32)
    context_plane = _plane(z_mm=0.0, observation_id="synthetic-context", shape_hw=shape_hw)
    target_plane = _plane(z_mm=1.0, observation_id="synthetic-target", shape_hw=shape_hw)
    image = _synthetic_image(shape_hw, device=torch_device, dtype=dtype)
    analytic = analytic_feature_bank(image)
    features = EncoderFeatureMaps(
        structural=analytic.tensor[:, 0:5],
        appearance=analytic.tensor[:, 0:1],
        reliability=analytic.tensor[:, 7:8],
        grid_to_plane=FeatureGridToPlaneTransform(
            input_shape_hw=shape_hw,
            feature_shape_hw=shape_hw,
            stride_vu=(1, 1),
            offset_vu_input_pixels=(0.0, 0.0),
        ),
        modality_ids=("synthetic-mri",),
    )
    supports = sample_fixed_supports(
        features,
        context_plane,
        config=FixedSupportConfig(step_vu=(4, 4), border_vu=(1, 1)),
    )
    head_config = FixedGaussianHeadConfig(
        input_dim=supports.feature_vectors.shape[1],
        appearance_channels=1,
        hidden_dim=32,
        max_center_offset_mm=0.25,
        min_scale_mm=1.5,
        max_scale_mm=5.0,
        max_off_diagonal_mm=0.5,
    )
    head = FixedGaussianHead(head_config).to(device=torch_device, dtype=dtype)
    optimizer = torch.optim.Adam(head.parameters(), lr=3e-3)
    target = torch.tanh(analytic.tensor[0, 0]).detach()
    renderer_config = RenderConfig(
        support_epsilon=1e-10,
        profile=SlabProfile.box(3),
        minimum_supported_psf_mass=1.0,
    )
    initial_loss = None
    last_loss = None
    supported_fraction = 0.0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        raw = head(supports.feature_vectors)
        gaussians = construct_fixed_gaussians(supports, raw, config=head_config)
        rendered = render_plane(gaussians, target_plane, config=renderer_config)
        valid = ~rendered.unsupported_mask
        if not bool(valid.any()):
            raise RuntimeError("synthetic T1-A demo produced no supported target pixels")
        loss = torch.mean((rendered.intensity[valid] - target[valid]) ** 2)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        if initial_loss is None:
            initial_loss = value
        last_loss = value
        supported_fraction = float(valid.to(dtype=torch.float32).mean().detach().cpu())
    gradient_norm = 0.0
    for parameter in head.parameters():
        if parameter.grad is not None:
            gradient_norm += float(parameter.grad.detach().norm().cpu())
    assert initial_loss is not None and last_loss is not None
    return {
        "initial_loss": initial_loss,
        "final_loss": last_loss,
        "gradient_norm": gradient_norm,
        "support_count": float(supports.count),
        "supported_fraction": supported_fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the executable T1-A synthetic reference")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--float64", action="store_true")
    args = parser.parse_args()
    metrics = run_demo(
        steps=args.steps,
        device=args.device,
        dtype=torch.float64 if args.float64 else torch.float32,
    )
    for key, value in metrics.items():
        print(f"{key}: {value:.8f}")


if __name__ == "__main__":
    main()
