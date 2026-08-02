"""Coverage-aware reconstruction metrics on serialized predictions only."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..contracts.coordinates import PhysicalPlane, TargetGrid
from ..contracts.outputs import VolumeReconstruction
from .medical_fidelity import ROIFidelity, evaluate_roi_fidelity


@dataclass(frozen=True)
class ReconstructionMetricConfig:
    """Evaluator-only metric policy; values are never fed back to training."""

    data_range: float | None = None
    ssim_window_policy: str = "global"
    edge_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.data_range is not None and (not math.isfinite(self.data_range) or self.data_range <= 0.0):
            raise ValueError("metric data_range must be positive and finite when declared")
        if self.ssim_window_policy != "global":
            raise ValueError("the maintained evaluator supports the declared global SSIM window policy")
        if not math.isfinite(self.edge_threshold) or self.edge_threshold < 0.0:
            raise ValueError("edge_threshold must be finite and non-negative")


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
    data_range: float | None = None
    data_range_source: str = "legacy_target_range"
    ssim_window_policy: str = "global"
    gradient_rmse: float = float("nan")
    edge_f1: float = float("nan")
    local_contrast_error: float = float("nan")
    unsupported_fraction: float = float("nan")
    metric_scope: str = "support_conditioned"
    complete_metric_status: str = "NOT_COMPUTED_UNSUPPORTED_PIXELS"
    complete_mae: float = float("nan")
    complete_rmse: float = float("nan")
    complete_psnr: float = float("nan")
    complete_ssim: float = float("nan")
    complete_ncc: float = float("nan")
    complete_gradient_mae: float = float("nan")
    complete_gradient_rmse: float = float("nan")
    complete_edge_f1: float = float("nan")
    complete_local_contrast_error: float = float("nan")
    complete_frequency_error: float = float("nan")
    distance_to_context_plane_status: str = "NOT_PROVIDED"
    distance_to_context_plane_mean_mm: float = float("nan")
    distance_to_context_plane_max_mm: float = float("nan")
    distance_to_context_plane_strata: tuple[tuple[str, float], ...] = ()
    context_gap_status: str = "NOT_PROVIDED"
    context_gap_mm: float = float("nan")
    error_vs_context_gap_mae: float = float("nan")
    local_observability_status: str = "NOT_PROVIDED"
    local_observability_mean: float = float("nan")
    error_vs_local_observability_strata: tuple[tuple[str, float], ...] = ()
    roi_status: str = "NOT_PROVIDED"
    roi_voxels: int = 0
    supported_roi_fraction: float = float("nan")
    roi_mae: float = float("nan")
    boundary_band_voxels: int = 0
    supported_boundary_band_fraction: float = float("nan")
    boundary_band_mae: float = float("nan")
    tumor_mae: float = float("nan")
    non_tumor_mae: float = float("nan")
    roi_contrast_error: float = float("nan")


def _ncc(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = prediction - prediction.mean(); t = target - target.mean()
    denominator = torch.linalg.vector_norm(p) * torch.linalg.vector_norm(t)
    if float(denominator) <= torch.finfo(prediction.dtype).eps:
        return prediction.new_tensor(1.0 if torch.allclose(prediction, target) else 0.0)
    return (p * t).sum() / denominator


def _global_ssim(prediction: torch.Tensor, target: torch.Tensor, data_range: torch.Tensor) -> torch.Tensor:
    c1 = (0.01 * data_range).square(); c2 = (0.03 * data_range).square()
    mean_p, mean_t = prediction.mean(), target.mean()
    var_p = prediction.var(unbiased=False); var_t = target.var(unbiased=False)
    covariance = ((prediction - mean_p) * (target - mean_t)).mean()
    return ((2 * mean_p * mean_t + c1) * (2 * covariance + c2)) / ((mean_p.square() + mean_t.square() + c1) * (var_p + var_t + c2))


def _edge_f1(prediction: torch.Tensor, target: torch.Tensor, legal: torch.Tensor, threshold: float) -> float:
    pred_edges: list[torch.Tensor] = []
    target_edges: list[torch.Tensor] = []
    for axis in range(target.ndim):
        left = [slice(None)] * target.ndim; right = [slice(None)] * target.ndim
        left[axis] = slice(None, -1); right[axis] = slice(1, None)
        pair = legal[tuple(left)] & legal[tuple(right)]
        if bool(pair.any()):
            pred_edges.append((prediction[tuple(right)] - prediction[tuple(left)]).abs()[pair] > threshold)
            target_edges.append((target[tuple(right)] - target[tuple(left)]).abs()[pair] > threshold)
    if not pred_edges:
        return float("nan")
    predicted = torch.cat(pred_edges); observed = torch.cat(target_edges)
    true_positive = (predicted & observed).sum().to(torch.float64)
    precision_denominator = predicted.sum().to(torch.float64)
    recall_denominator = observed.sum().to(torch.float64)
    precision = true_positive / precision_denominator.clamp_min(1.0)
    recall = true_positive / recall_denominator.clamp_min(1.0)
    if float(precision + recall) == 0.0:
        return 1.0 if bool(torch.equal(predicted, observed)) else 0.0
    return float((2.0 * precision * recall / (precision + recall)).detach())


def _world_coordinates(grid: TargetGrid, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    depth, height, width = grid.shape_dhw
    d, h, w = torch.meshgrid(
        torch.arange(depth, dtype=dtype, device=device),
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    homogeneous = torch.stack((w, h, d, torch.ones_like(d)), dim=0).reshape(4, -1)
    matrix = torch.as_tensor(grid.index_to_ras_mm, dtype=dtype, device=device)
    return (matrix @ homogeneous)[:3].transpose(0, 1).reshape(depth, height, width, 3)


def _distance_to_context_planes(grid: TargetGrid, planes: tuple[PhysicalPlane, ...], *, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
    if not planes:
        return None
    coordinates = _world_coordinates(grid, dtype=dtype, device=device)
    distances = []
    for plane in planes:
        origin = torch.as_tensor(plane.pixel_center_origin_ras_mm, dtype=dtype, device=device)
        normal = torch.as_tensor(plane.signed_normal_ras, dtype=dtype, device=device)
        distances.append(((coordinates - origin) * normal).sum(dim=-1).abs())
    return torch.stack(distances, dim=0).amin(dim=0)


def _error_strata(error: torch.Tensor, values: torch.Tensor, legal: torch.Tensor, *, low_name: str, high_name: str) -> tuple[tuple[str, float], ...]:
    usable = legal & torch.isfinite(values)
    if not bool(usable.any()):
        return ()
    selected_values = values[usable]
    selected_error = error[usable]
    midpoint = torch.quantile(selected_values, selected_values.new_tensor(0.5))
    low = selected_values <= midpoint
    high = selected_values > midpoint
    if not bool(high.any()):
        return (("all", float(selected_error.mean().detach())),)
    return (
        (low_name, float(selected_error[low].mean().detach())),
        (high_name, float(selected_error[high].mean().detach())),
    )


def compute_reconstruction_metrics(
    prediction: VolumeReconstruction, target: torch.Tensor, target_valid_mask: torch.Tensor,
    *, metric_config: ReconstructionMetricConfig | None = None,
    context_planes: tuple[PhysicalPlane, ...] = (),
    context_gap_mm: float | None = None,
    local_observability: torch.Tensor | None = None,
    segmentation: torch.Tensor | None = None,
) -> ReconstructionMetrics:
    if target.shape != prediction.intensity.shape or target_valid_mask.shape != target.shape or target_valid_mask.dtype is not torch.bool:
        raise ValueError("target and validity must match serialized prediction shape")
    if not bool(torch.isfinite(target[target_valid_mask]).all()):
        raise ValueError("valid audit targets must be finite")
    metric_config = metric_config or ReconstructionMetricConfig()
    roi: ROIFidelity | None = None
    if segmentation is not None:
        if segmentation.shape != target.shape or segmentation.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("evaluator segmentation must be an integer label map matching the target shape")
        roi = evaluate_roi_fidelity(
            prediction,
            target,
            segmentation > 0,
            segmentation == 0,
            valid_mask=target_valid_mask,
        )
    legal = target_valid_mask & ~prediction.unsupported_mask
    evaluable = int(target_valid_mask.sum()); supported = int(legal.sum()); unsupported = evaluable - supported
    if evaluable == 0:
        raise ValueError("evaluation target has no declared valid voxels")
    if supported == 0:
        nan = float("nan")
        return ReconstructionMetrics(
            prediction.patient_id, prediction.modality_id, evaluable, 0, unsupported, 0.0,
            nan, nan, nan, nan, nan, nan, nan, nan, "NO_SUPPORTED_EVALUABLE_VOXELS",
            metric_config.data_range, "declared" if metric_config.data_range is not None else "legacy_target_range",
            metric_config.ssim_window_policy,
            unsupported_fraction=unsupported / evaluable,
            complete_metric_status="NOT_COMPUTED_NO_SUPPORTED_EVALUABLE_VOXELS",
            roi_status="NOT_PROVIDED" if roi is None else roi.status,
            roi_voxels=0 if roi is None else roi.roi_voxels,
            supported_roi_fraction=float("nan") if roi is None else roi.supported_roi_fraction,
            roi_mae=float("nan") if roi is None else roi.roi_mae,
            boundary_band_voxels=0 if roi is None else roi.boundary_band_voxels,
            supported_boundary_band_fraction=float("nan") if roi is None else roi.supported_boundary_band_fraction,
            boundary_band_mae=float("nan") if roi is None else roi.boundary_band_mae,
            tumor_mae=float("nan") if roi is None else roi.tumor_mae,
            non_tumor_mae=float("nan") if roi is None else roi.non_tumor_mae,
            roi_contrast_error=float("nan") if roi is None else roi.contrast_error,
        )
    p = prediction.intensity[legal]; t = target[legal]
    error = p - t; mse = error.square().mean(); rmse = mse.sqrt()
    target_energy = t.square().sum().clamp_min(torch.finfo(t.dtype).eps)
    nmse = error.square().sum() / target_energy
    if metric_config.data_range is None:
        peak = (t.max() - t.min()).clamp_min(torch.tensor(1.0, dtype=t.dtype, device=t.device))
        data_range_source = "legacy_target_range"
    else:
        peak = t.new_tensor(metric_config.data_range)
        data_range_source = "declared"
    psnr = 10.0 * torch.log10(peak.square() / mse.clamp_min(torch.finfo(t.dtype).eps))
    gradients: list[torch.Tensor] = []
    local_contrast: list[torch.Tensor] = []
    gradient_squares: list[torch.Tensor] = []
    for axis in range(target.ndim):
        left = [slice(None)] * target.ndim; right = [slice(None)] * target.ndim
        left[axis] = slice(None, -1); right[axis] = slice(1, None)
        pair = legal[tuple(left)] & legal[tuple(right)]
        if bool(pair.any()):
            prediction_delta = prediction.intensity[tuple(right)] - prediction.intensity[tuple(left)]
            target_delta = target[tuple(right)] - target[tuple(left)]
            delta_error = (prediction_delta - target_delta)[pair]
            gradients.append(delta_error.abs())
            gradient_squares.append(delta_error.square())
            local_contrast.append((prediction_delta.abs() - target_delta.abs())[pair].abs())
    gradient_mae = torch.cat(gradients).mean() if gradients else p.new_tensor(float("nan"))
    gradient_rmse = torch.cat(gradient_squares).mean().sqrt() if gradient_squares else p.new_tensor(float("nan"))
    local_contrast_error = torch.cat(local_contrast).mean() if local_contrast else p.new_tensor(float("nan"))
    frequency_error = (torch.fft.rfft(p) - torch.fft.rfft(t)).abs().mean()
    edge_f1 = _edge_f1(prediction.intensity, target, legal, metric_config.edge_threshold * peak)
    support_values = {
        "mae": float(error.abs().mean().detach()),
        "rmse": float(rmse.detach()),
        "psnr": float(psnr.detach()),
        "ssim": float(_global_ssim(p, t, peak).detach()),
        "ncc": float(_ncc(p, t).detach()),
        "gradient_mae": float(gradient_mae.detach()),
        "gradient_rmse": float(gradient_rmse.detach()),
        "edge_f1": edge_f1,
        "local_contrast_error": float(local_contrast_error.detach()),
        "frequency_error": float(frequency_error.detach()),
    }
    complete_status = "COMPUTED_ALL_DECLARED_TARGET_VOXELS" if unsupported == 0 else "NOT_COMPUTED_UNSUPPORTED_PIXELS"
    complete_values = support_values if unsupported == 0 else {name: float("nan") for name in support_values}
    distance_map = _distance_to_context_planes(
        prediction.grid,
        context_planes,
        dtype=target.dtype,
        device=target.device,
    )
    if distance_map is None:
        distance_status = "NOT_PROVIDED"
        distance_mean = float("nan")
        distance_max = float("nan")
        distance_strata: tuple[tuple[str, float], ...] = ()
    else:
        distance_status = "COMPUTED_SUPPORT_CONDITIONED"
        distance_values = distance_map[legal]
        distance_mean = float(distance_values.mean().detach())
        distance_max = float(distance_values.max().detach())
        distance_strata = _error_strata((prediction.intensity - target).abs(), distance_map, legal, low_name="near", high_name="far")
    if context_gap_mm is None:
        gap_status = "NOT_PROVIDED"
        gap_value = float("nan")
        gap_error = float("nan")
    else:
        if not math.isfinite(float(context_gap_mm)) or float(context_gap_mm) <= 0.0:
            raise ValueError("context_gap_mm must be positive and finite when provided")
        gap_status = "COMPUTED_SUPPORT_CONDITIONED"
        gap_value = float(context_gap_mm)
        gap_error = support_values["mae"]
    if local_observability is None:
        observability_status = "NOT_PROVIDED"
        observability_mean = float("nan")
        observability_strata: tuple[tuple[str, float], ...] = ()
    else:
        if local_observability.shape != target.shape or local_observability.dtype not in (torch.float32, torch.float64):
            raise ValueError("local_observability must be a floating map matching the target shape")
        if not bool(torch.isfinite(local_observability).all()) or bool((local_observability < 0).any()) or bool((local_observability > 1).any()):
            raise ValueError("local_observability must be finite and bounded to [0,1]")
        observability_map = local_observability.to(device=target.device, dtype=target.dtype)
        observability_status = "COMPUTED_SUPPORT_CONDITIONED"
        observability_mean = float(observability_map[legal].mean().detach())
        observability_strata = _error_strata(
            (prediction.intensity - target).abs(), observability_map, legal,
            low_name="low_observability", high_name="high_observability",
        )
    return ReconstructionMetrics(
        prediction.patient_id, prediction.modality_id, evaluable, supported, unsupported, supported / evaluable,
        support_values["mae"], support_values["rmse"], float(nmse.detach()), support_values["psnr"],
        support_values["ssim"], support_values["ncc"], support_values["gradient_mae"],
        float(frequency_error.detach()), None, float(peak.detach()), data_range_source,
        metric_config.ssim_window_policy, support_values["gradient_rmse"], edge_f1,
        support_values["local_contrast_error"], unsupported / evaluable, "support_conditioned", complete_status,
        complete_values["mae"], complete_values["rmse"], complete_values["psnr"], complete_values["ssim"],
        complete_values["ncc"], complete_values["gradient_mae"], complete_values["gradient_rmse"],
        complete_values["edge_f1"], complete_values["local_contrast_error"], complete_values["frequency_error"],
        distance_status, distance_mean, distance_max, distance_strata,
        gap_status, gap_value, gap_error,
        observability_status, observability_mean, observability_strata,
        roi_status="NOT_PROVIDED" if roi is None else roi.status,
        roi_voxels=0 if roi is None else roi.roi_voxels,
        supported_roi_fraction=float("nan") if roi is None else roi.supported_roi_fraction,
        roi_mae=float("nan") if roi is None else roi.roi_mae,
        boundary_band_voxels=0 if roi is None else roi.boundary_band_voxels,
        supported_boundary_band_fraction=float("nan") if roi is None else roi.supported_boundary_band_fraction,
        boundary_band_mae=float("nan") if roi is None else roi.boundary_band_mae,
        tumor_mae=float("nan") if roi is None else roi.tumor_mae,
        non_tumor_mae=float("nan") if roi is None else roi.non_tumor_mae,
        roi_contrast_error=float("nan") if roi is None else roi.contrast_error,
    )
