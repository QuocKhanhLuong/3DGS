"""Coverage-aware reconstruction metrics on serialized predictions only."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..contracts.outputs import VolumeReconstruction


@dataclass(frozen=True)
class ReconstructionMetrics:
    patient_id: str
    modality_id: str
    evaluable_voxels: int
    supported_voxels: int
    unsupported_voxels: int
    supported_fraction: float
    mae: float
    rmse: float
    nmse: float
    psnr: float
    ssim: float
    ncc: float
    gradient_mae: float
    frequency_error: float
    failure_reason: str | None = None


def _ncc(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = prediction - prediction.mean(); t = target - target.mean()
    denominator = torch.linalg.vector_norm(p) * torch.linalg.vector_norm(t)
    if float(denominator) <= torch.finfo(prediction.dtype).eps:
        return prediction.new_tensor(1.0 if torch.allclose(prediction, target) else 0.0)
    return (p * t).sum() / denominator


def _global_ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dynamic = (target.max() - target.min()).clamp_min(torch.finfo(target.dtype).eps)
    c1 = (0.01 * dynamic).square(); c2 = (0.03 * dynamic).square()
    mean_p, mean_t = prediction.mean(), target.mean()
    var_p = prediction.var(unbiased=False); var_t = target.var(unbiased=False)
    covariance = ((prediction - mean_p) * (target - mean_t)).mean()
    return ((2 * mean_p * mean_t + c1) * (2 * covariance + c2)) / ((mean_p.square() + mean_t.square() + c1) * (var_p + var_t + c2))


def compute_reconstruction_metrics(
    prediction: VolumeReconstruction, target: torch.Tensor, target_valid_mask: torch.Tensor,
) -> ReconstructionMetrics:
    if target.shape != prediction.intensity.shape or target_valid_mask.shape != target.shape or target_valid_mask.dtype is not torch.bool:
        raise ValueError("target and validity must match serialized prediction shape")
    if not bool(torch.isfinite(target[target_valid_mask]).all()):
        raise ValueError("valid audit targets must be finite")
    legal = target_valid_mask & ~prediction.unsupported_mask
    evaluable = int(target_valid_mask.sum()); supported = int(legal.sum()); unsupported = evaluable - supported
    if evaluable == 0:
        raise ValueError("evaluation target has no declared valid voxels")
    if supported == 0:
        nan = float("nan")
        return ReconstructionMetrics(prediction.patient_id, prediction.modality_id, evaluable, 0, unsupported, 0.0, nan, nan, nan, nan, nan, nan, nan, nan, "NO_SUPPORTED_EVALUABLE_VOXELS")
    p = prediction.intensity[legal]; t = target[legal]
    error = p - t; mse = error.square().mean(); rmse = mse.sqrt()
    target_energy = t.square().sum().clamp_min(torch.finfo(t.dtype).eps)
    nmse = error.square().sum() / target_energy
    peak = (t.max() - t.min()).clamp_min(torch.tensor(1.0, dtype=t.dtype, device=t.device))
    psnr = 10.0 * torch.log10(peak.square() / mse.clamp_min(torch.finfo(t.dtype).eps))
    gradients = []
    for axis in range(target.ndim):
        left = [slice(None)] * target.ndim; right = [slice(None)] * target.ndim
        left[axis] = slice(None, -1); right[axis] = slice(1, None)
        pair = legal[tuple(left)] & legal[tuple(right)]
        if bool(pair.any()):
            gradients.append(((prediction.intensity[tuple(right)] - prediction.intensity[tuple(left)]) - (target[tuple(right)] - target[tuple(left)]))[pair].abs())
    gradient_mae = torch.cat(gradients).mean() if gradients else p.new_tensor(float("nan"))
    frequency_error = (torch.fft.rfft(p) - torch.fft.rfft(t)).abs().mean()
    return ReconstructionMetrics(
        prediction.patient_id, prediction.modality_id, evaluable, supported, unsupported, supported / evaluable,
        float(error.abs().mean().detach()), float(rmse.detach()), float(nmse.detach()), float(psnr.detach()), float(_global_ssim(p, t).detach()),
        float(_ncc(p, t).detach()), float(gradient_mae.detach()), float(frequency_error.detach()), None,
    )
