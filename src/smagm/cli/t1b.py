"""Run the CPU-only synthetic T1-B encoder and fixed-support demo.

The printed values are software-contract diagnostics, not reconstruction
accuracy or clinical/scientific validation.
"""

from __future__ import annotations

import argparse

import torch

from ..baselines.fixed_gaussian import FixedGaussianHead, FixedGaussianHeadConfig, construct_fixed_gaussians
from ..baselines.fixed_support import FixedSupportConfig, sample_fixed_supports
from ..contracts.coordinates import PhysicalPlane
from ..features.encoder import EncoderConfig, EvidenceEncoder
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


def _synthetic_image(shape_hw: tuple[int, int], *, device: torch.device, dtype: torch.dtype, phase: float = 0.0) -> torch.Tensor:
    height, width = shape_hw
    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    blob_a = torch.exp(-((u - 0.35 * width) ** 2 + (v - 0.45 * height) ** 2) / (2.0 * (0.12 * width) ** 2))
    blob_b = 0.65 * torch.exp(-((u - 0.68 * width) ** 2 + (v - 0.58 * height) ** 2) / (2.0 * (0.09 * width) ** 2))
    ridge = 0.2 * torch.sin(u / 4.0 + phase) * torch.exp(-((v - 0.5 * height) ** 2) / (2.0 * (0.18 * height) ** 2))
    return (blob_a + blob_b + ridge).unsqueeze(0).unsqueeze(0)


def run_demo(*, variant: str, steps: int = 1, device: str = "cpu", dtype: torch.dtype = torch.float32) -> dict[str, object]:
    if variant not in ("e0", "e1", "e2"):
        raise ValueError("variant must be one of e0, e1, or e2")
    if steps <= 0:
        raise ValueError("steps must be positive")
    torch.manual_seed(19)
    torch_device = torch.device(device)
    shape_hw = (31, 29)
    context_plane = _plane(z_mm=0.0, observation_id="synthetic-context", shape_hw=shape_hw)
    target_plane = _plane(z_mm=1.0, observation_id="synthetic-target", shape_hw=shape_hw)
    image = _synthetic_image(shape_hw, device=torch_device, dtype=dtype, phase=0.0)
    target = torch.tanh(_synthetic_image(shape_hw, device=torch_device, dtype=dtype, phase=0.35)[0, 0]).detach()
    encoder = EvidenceEncoder(EncoderConfig(variant=variant, output_stride=1)).to(device=torch_device, dtype=dtype)
    head_config = FixedGaussianHeadConfig(
        input_dim=16 + 8 + 1,
        appearance_channels=1,
        hidden_dim=32,
        max_center_offset_mm=0.25,
        min_scale_mm=1.5,
        max_scale_mm=5.0,
        max_off_diagonal_mm=0.5,
    )
    head = FixedGaussianHead(head_config).to(device=torch_device, dtype=dtype)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=2e-3)
    render_config = RenderConfig(
        support_epsilon=1e-10,
        profile=SlabProfile.box(3),
        minimum_supported_psf_mass=1.0,
    )
    last_loss: torch.Tensor | None = None
    supports = None
    features = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        features = encoder(image, context_plane, "synthetic-mri")
        supports = sample_fixed_supports(
            features,
            context_plane,
            config=FixedSupportConfig(step_vu=(4, 4), border_vu=(1, 1)),
        )
        raw = head(supports.feature_vectors)
        gaussians = construct_fixed_gaussians(supports, raw, config=head_config)
        rendered = render_plane(gaussians, target_plane, config=render_config)
        valid = ~rendered.unsupported_mask
        if not bool(valid.any()):
            raise RuntimeError("synthetic T1-B demo produced no supported target pixels")
        last_loss = (rendered.intensity[valid] - target[valid]).square().mean()
        if not bool(torch.isfinite(last_loss)):
            raise RuntimeError("synthetic T1-B demo produced a non-finite loss")
        last_loss.backward()
        optimizer.step()
    assert features is not None and supports is not None and last_loss is not None

    def _grad_norm(parameters: object) -> float:
        total = 0.0
        for parameter in parameters:  # type: ignore[union-attr]
            if parameter.grad is not None:
                total += float(parameter.grad.detach().norm().cpu())
        return total

    encoder_gradient_norm = _grad_norm(encoder.parameters())
    head_gradient_norm = _grad_norm(head.parameters())
    report = encoder.parameter_report
    valid_values = features.reliability[features.valid_feature_mask]
    return {
        "variant": variant,
        "parameter_count": report.parameter_count + sum(parameter.numel() for parameter in head.parameters()),
        "encoder_parameter_count": report.parameter_count,
        "head_parameter_count": sum(parameter.numel() for parameter in head.parameters()),
        "adapter_operation_count": report.adapter_operation_count,
        "structural_shape": tuple(features.structural.shape),
        "appearance_shape": tuple(features.appearance.shape),
        "reliability_min": float(valid_values.min().detach().cpu()),
        "reliability_max": float(valid_values.max().detach().cpu()),
        "support_count": supports.count,
        "supported_fraction": float((~rendered.unsupported_mask).to(dtype=torch.float64).mean().detach().cpu()),
        "loss": float(last_loss.detach().cpu()),
        "encoder_gradient_norm": encoder_gradient_norm,
        "head_gradient_norm": head_gradient_norm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the teacher-free T1-B synthetic encoder reference")
    parser.add_argument("--variant", choices=("e0", "e1", "e2"), default="e2")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--float64", action="store_true")
    args = parser.parse_args()
    metrics = run_demo(
        variant=args.variant,
        steps=args.steps,
        device=args.device,
        dtype=torch.float64 if args.float64 else torch.float32,
    )
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.8f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
