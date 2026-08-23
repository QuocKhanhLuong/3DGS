"""Geometry-aware, parameter-free queries of the static spectral anchor.

Phase 7 deliberately keeps feature-grid geometry separate from the input
volume geometry.  A feature index is first mapped to its receptive-field
centre in the input lattice by composing the live MedicalNet spatial
operators, then through the source ``[w, h, d] -> RAS`` affine.  This avoids
assuming either a diagonal affine or a hard-coded feature scale.

The query itself is pointwise: every plane uses one ``[B, N, 1, 2]`` 2-D
``grid_sample`` grid with bilinear interpolation, zero padding, and
``align_corners=False``.  It never constructs a point-by-plane-area tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Literal, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import VolumeGeometry
from .sampling import ras_mm_to_voxel_dhw, voxel_dhw_to_grid_sample_coordinates, voxel_dhw_to_ras_mm
from .spectral_anchor import SPECTRAL_ANCHOR_CHANNELS, SpectralAnchor

if TYPE_CHECKING:
    from .medicalnet_resnet10 import MedicalNetResNet10


SpectralTap = Literal["conv1_pre_maxpool", "layer1"]


__all__ = [
    "FeatureGridGeometry",
    "SpectralPointQuery",
    "SpectralPointSamples",
    "SpectralTap",
    "derive_feature_grid_geometry",
]


def _shape_dhw(name: str, value: Sequence[int]) -> tuple[int, int, int]:
    shape = tuple(value)
    if len(shape) != 3 or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in shape):
        raise ValueError(f"{name} must contain three positive integers in [D, H, W] order")
    return shape  # type: ignore[return-value]


def _finite_triplet(name: str, value: Sequence[float], *, positive: bool) -> tuple[float, float, float]:
    raw = tuple(float(item) for item in value)
    if len(raw) != 3 or not all(math.isfinite(item) for item in raw) or (positive and any(item <= 0.0 for item in raw)):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must contain three {qualifier} values")
    return raw  # type: ignore[return-value]


def _validate_float_tensor(name: str, value: Tensor, *, rank: int, final_dimension: int | None = None) -> None:
    if not isinstance(value, Tensor) or value.ndim != rank:
        raise ValueError(f"{name} must be a rank-{rank} torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if final_dimension is not None and value.shape[-1] != final_dimension:
        raise ValueError(f"{name} must have final dimension {final_dimension}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class FeatureGridGeometry:
    """Derived RAS geometry for one regular selected MedicalNet feature grid.

    ``feature_to_source_scale_dhw`` and ``feature_to_source_offset_dhw``
    encode the centre map, per tensor axis,

    ``source_dhw = scale_dhw * feature_dhw + offset_dhw``.

    ``feature_geometry`` composes that map with ``source_geometry``'s full
    affine, so it retains rotation, shear, anisotropic spacing, and
    translation without reducing physical geometry to scalar spacing.
    """

    source_geometry: VolumeGeometry
    feature_geometry: VolumeGeometry
    tap: SpectralTap
    feature_to_source_scale_dhw: Sequence[float]
    feature_to_source_offset_dhw: Sequence[float]
    operator_chain: Sequence[str]

    def __post_init__(self) -> None:
        if not isinstance(self.source_geometry, VolumeGeometry):
            raise TypeError("source_geometry must be a VolumeGeometry")
        if not isinstance(self.feature_geometry, VolumeGeometry):
            raise TypeError("feature_geometry must be a VolumeGeometry")
        if self.tap not in ("conv1_pre_maxpool", "layer1"):
            raise ValueError("tap must be 'conv1_pre_maxpool' or 'layer1'")
        object.__setattr__(
            self,
            "feature_to_source_scale_dhw",
            _finite_triplet("feature_to_source_scale_dhw", self.feature_to_source_scale_dhw, positive=True),
        )
        object.__setattr__(
            self,
            "feature_to_source_offset_dhw",
            _finite_triplet("feature_to_source_offset_dhw", self.feature_to_source_offset_dhw, positive=False),
        )
        chain = tuple(self.operator_chain)
        if not chain or any(not isinstance(item, str) or not item for item in chain):
            raise ValueError("operator_chain must contain at least one nonempty operation label")
        object.__setattr__(self, "operator_chain", chain)

    @property
    def shape_dhw(self) -> tuple[int, int, int]:
        """The selected feature lattice shape in tensor ``[D, H, W]`` order."""

        return self.feature_geometry.shape_dhw

    @property
    def feature_shape_dhw(self) -> tuple[int, int, int]:
        """Explicit alias for :attr:`shape_dhw`."""

        return self.feature_geometry.shape_dhw

    def ras_mm_to_feature_dhw(self, ras_mm: Tensor) -> Tensor:
        """Map continuous RAS ``XYZ`` points to feature ``[d, h, w]`` indices."""

        return ras_mm_to_voxel_dhw(ras_mm, self.feature_geometry)

    def feature_dhw_to_ras_mm(self, feature_dhw: Tensor) -> Tensor:
        """Map continuous feature ``[d, h, w]`` indices to RAS ``XYZ`` mm."""

        return voxel_dhw_to_ras_mm(feature_dhw, self.feature_geometry)

    def feature_dhw_to_grid_sample_coordinates(self, feature_dhw: Tensor) -> Tensor:
        """Return 5-D sampling coordinates for the feature lattice if needed."""

        return voxel_dhw_to_grid_sample_coordinates(feature_dhw, self.feature_geometry.shape_dhw)


@dataclass(frozen=True)
class SpectralPointSamples:
    """Raw pointwise Phase-6 anchor samples in permanent XY/XZ/YZ order."""

    xy: Tensor  # [B, N, 56]
    xz: Tensor  # [B, N, 56]
    yz: Tensor  # [B, N, 56]

    def __post_init__(self) -> None:
        named = (("xy", self.xy), ("xz", self.xz), ("yz", self.yz))
        for name, value in named:
            _validate_float_tensor(name, value, rank=3, final_dimension=SPECTRAL_ANCHOR_CHANNELS)
            if value.shape[0] <= 0 or value.shape[1] <= 0:
                raise ValueError(f"{name} must have positive batch and point dimensions")
        reference = self.xy
        for name, value in named[1:]:
            if value.shape != reference.shape:
                raise ValueError(f"{name} must match xy shape")
            if value.dtype != reference.dtype:
                raise TypeError(f"{name} must match xy dtype")
            if value.device != reference.device:
                raise ValueError(f"{name} must match xy device")

    @property
    def f_xy(self) -> Tensor:
        """Explicit raw-XY feature spelling."""

        return self.xy

    @property
    def f_xz(self) -> Tensor:
        """Explicit raw-XZ feature spelling."""

        return self.xz

    @property
    def f_yz(self) -> Tensor:
        """Explicit raw-YZ feature spelling."""

        return self.yz


@dataclass(frozen=True)
class _LatticeState:
    """Regular-grid input-centre map accumulated through spatial operators."""

    shape_dhw: tuple[int, int, int]
    scale_dhw: tuple[float, float, float]
    offset_dhw: tuple[float, float, float]
    operation_chain: tuple[str, ...]


def _triple(name: str, value: int | Sequence[int] | None, *, allow_none: bool = False) -> tuple[int, int, int]:
    if value is None:
        if allow_none:
            return (0, 0, 0)
        raise ValueError(f"{name} must be specified")
    if isinstance(value, int) and not isinstance(value, bool):
        result = (value, value, value)
    else:
        result = tuple(value)  # type: ignore[arg-type]
        if len(result) != 3:
            raise ValueError(f"{name} must have three spatial values")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in result):
        raise ValueError(f"{name} must contain integers")
    return result  # type: ignore[return-value]


def _spatial_metadata(module: nn.Module, *, label: str) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    """Read an actual Conv3d/MaxPool3d centre transform or fail closed."""

    if isinstance(module, nn.Conv3d):
        kernel = _triple(f"{label}.kernel_size", module.kernel_size)
        stride = _triple(f"{label}.stride", module.stride)
        padding = _triple(f"{label}.padding", module.padding)
        dilation = _triple(f"{label}.dilation", module.dilation)
    elif isinstance(module, nn.MaxPool3d):
        if module.ceil_mode:
            raise ValueError(f"cannot derive one regular feature lattice through {label}: ceil_mode=True")
        if module.return_indices:
            raise ValueError(f"cannot derive one regular feature lattice through {label}: return_indices=True")
        kernel = _triple(f"{label}.kernel_size", module.kernel_size)
        stride = _triple(f"{label}.stride", module.stride if module.stride is not None else module.kernel_size)
        padding = _triple(f"{label}.padding", module.padding)
        dilation = _triple(f"{label}.dilation", module.dilation)
    else:
        raise ValueError(f"cannot derive feature geometry through unsupported spatial module {label}: {type(module).__name__}")

    if any(item <= 0 for item in (*kernel, *stride, *dilation)) or any(item < 0 for item in padding):
        raise ValueError(f"{label} has invalid spatial metadata")
    return kernel, stride, padding, dilation


def _apply_spatial_module(state: _LatticeState, module: nn.Module, *, label: str) -> _LatticeState:
    kernel, stride, padding, dilation = _spatial_metadata(module, label=label)
    output_shape: list[int] = []
    local_offset: list[float] = []
    for input_length, kernel_length, stride_length, padding_length, dilation_length in zip(
        state.shape_dhw,
        kernel,
        stride,
        padding,
        dilation,
    ):
        output_length = (input_length + 2 * padding_length - dilation_length * (kernel_length - 1) - 1) // stride_length + 1
        if output_length <= 0:
            raise ValueError(f"{label} produces a nonpositive spatial dimension")
        output_shape.append(output_length)
        local_offset.append(dilation_length * (kernel_length - 1) / 2.0 - padding_length)

    return _LatticeState(
        shape_dhw=tuple(output_shape),  # type: ignore[arg-type]
        scale_dhw=tuple(old * float(step) for old, step in zip(state.scale_dhw, stride)),  # type: ignore[arg-type]
        offset_dhw=tuple(
            old_scale * operation_offset + old_offset
            for old_scale, operation_offset, old_offset in zip(state.scale_dhw, local_offset, state.offset_dhw)
        ),  # type: ignore[arg-type]
        operation_chain=(*state.operation_chain, label),
    )


def _same_lattice(left: _LatticeState, right: _LatticeState) -> bool:
    return (
        left.shape_dhw == right.shape_dhw
        and left.scale_dhw == right.scale_dhw
        and all(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12) for actual, expected in zip(left.offset_dhw, right.offset_dhw))
    )


def _apply_neutral_or_spatial_path(state: _LatticeState, module: nn.Module, *, label: str) -> _LatticeState:
    """Follow a verified residual shortcut while rejecting unknown geometry ops."""

    if isinstance(module, (nn.Conv3d, nn.MaxPool3d)):
        return _apply_spatial_module(state, module, label=label)
    if isinstance(module, nn.Sequential):
        current = state
        for index, child in enumerate(module):
            current = _apply_neutral_or_spatial_path(current, child, label=f"{label}[{index}]")
        return current
    if isinstance(module, (nn.BatchNorm3d, nn.ReLU, nn.Identity)):
        return state
    raise ValueError(f"cannot derive feature geometry through unsupported module {label}: {type(module).__name__}")


def _apply_residual_block(state: _LatticeState, block: nn.Module, *, label: str) -> _LatticeState:
    """Derive a BasicBlock lattice only when its main and shortcut paths agree."""

    conv1 = getattr(block, "conv1", None)
    conv2 = getattr(block, "conv2", None)
    downsample = getattr(block, "downsample", None)
    if not isinstance(conv1, nn.Conv3d) or not isinstance(conv2, nn.Conv3d):
        raise ValueError(f"cannot derive feature geometry through {label}: expected a Conv3d residual basic block")
    main = _apply_spatial_module(state, conv1, label=f"{label}.conv1")
    main = _apply_spatial_module(main, conv2, label=f"{label}.conv2")
    shortcut = state if downsample is None else _apply_neutral_or_spatial_path(state, downsample, label=f"{label}.downsample")
    if not _same_lattice(main, shortcut):
        raise ValueError(f"cannot derive one regular feature lattice through {label}: residual paths disagree")
    return main


def _feature_affine(
    source_geometry: VolumeGeometry,
    *,
    feature_shape_dhw: tuple[int, int, int],
    scale_dhw: tuple[float, float, float],
    offset_dhw: tuple[float, float, float],
) -> VolumeGeometry:
    """Compose feature ``[w,h,d]`` centre mapping with the source RAS affine."""

    # Tensor layout is DHW, while the physical affine consumes WHD.
    scale_w, scale_h, scale_d = scale_dhw[2], scale_dhw[1], scale_dhw[0]
    offset_w, offset_h, offset_d = offset_dhw[2], offset_dhw[1], offset_dhw[0]
    feature_to_source_whd = (
        (scale_w, 0.0, 0.0, offset_w),
        (0.0, scale_h, 0.0, offset_h),
        (0.0, 0.0, scale_d, offset_d),
        (0.0, 0.0, 0.0, 1.0),
    )
    source_affine = source_geometry.voxel_to_ras_mm
    composed = tuple(
        tuple(
            sum(source_affine[row][inner] * feature_to_source_whd[inner][column] for inner in range(4))
            for column in range(4)
        )
        for row in range(4)
    )
    return VolumeGeometry(feature_shape_dhw, composed)


def derive_feature_grid_geometry(
    backbone: "MedicalNetResNet10 | nn.Module",
    source_geometry: VolumeGeometry,
    *,
    tap: SpectralTap = "conv1_pre_maxpool",
    observed_shape_dhw: Sequence[int],
) -> FeatureGridGeometry:
    """Derive selected MedicalNet-grid RAS geometry from live spatial metadata.

    The only supported taps are the locked pre-MaxPool Conv1 feature and the
    Layer1 ablation.  The function reads ``kernel_size``, ``stride``,
    ``padding``, and ``dilation`` from the supplied live backbone rather than
    encoding a scale assumption.  It also calculates and validates the exact
    observed selected-feature shape, failing closed if the architecture and
    tensor disagree.
    """

    if not isinstance(source_geometry, VolumeGeometry):
        raise TypeError("source_geometry must be a VolumeGeometry")
    if not isinstance(backbone, nn.Module):
        raise TypeError("backbone must be an nn.Module with the locked MedicalNet tap modules")
    if tap not in ("conv1_pre_maxpool", "layer1"):
        raise ValueError("tap must be 'conv1_pre_maxpool' or 'layer1'")
    observed_shape = _shape_dhw("observed_shape_dhw", observed_shape_dhw)

    conv1 = getattr(backbone, "conv1", None)
    if not isinstance(conv1, nn.Conv3d):
        raise ValueError("cannot derive feature geometry: backbone.conv1 must be Conv3d")
    state = _LatticeState(
        shape_dhw=source_geometry.shape_dhw,
        scale_dhw=(1.0, 1.0, 1.0),
        offset_dhw=(0.0, 0.0, 0.0),
        operation_chain=(),
    )
    state = _apply_spatial_module(state, conv1, label="conv1")

    if tap == "layer1":
        maxpool = getattr(backbone, "maxpool", None)
        layer1 = getattr(backbone, "layer1", None)
        if not isinstance(maxpool, nn.MaxPool3d):
            raise ValueError("cannot derive layer1 feature geometry: backbone.maxpool must be MaxPool3d")
        if not isinstance(layer1, nn.Sequential) or len(layer1) == 0:
            raise ValueError("cannot derive layer1 feature geometry: backbone.layer1 must be a nonempty Sequential")
        state = _apply_spatial_module(state, maxpool, label="maxpool")
        for index, block in enumerate(layer1):
            state = _apply_residual_block(state, block, label=f"layer1[{index}]")

    if state.shape_dhw != observed_shape:
        raise ValueError(
            "derived selected-feature shape does not match observed feature: "
            f"derived={state.shape_dhw}, observed={observed_shape}, tap={tap!r}"
        )
    feature_geometry = _feature_affine(
        source_geometry,
        feature_shape_dhw=observed_shape,
        scale_dhw=state.scale_dhw,
        offset_dhw=state.offset_dhw,
    )
    return FeatureGridGeometry(
        source_geometry=source_geometry,
        feature_geometry=feature_geometry,
        tap=tap,
        feature_to_source_scale_dhw=state.scale_dhw,
        feature_to_source_offset_dhw=state.offset_dhw,
        operator_chain=state.operation_chain,
    )


def _plane_grid(row: Tensor, column: Tensor, *, rows: int, columns: int) -> Tensor:
    """Encode continuous row/column indices as a 2-D pointwise sampling grid."""

    x = (2.0 * column + 1.0) / float(columns) - 1.0
    y = (2.0 * row + 1.0) / float(rows) - 1.0
    return torch.stack((x, y), dim=-1).unsqueeze(2)  # [B, N, 1, x/y]


def _sample_plane(plane: Tensor, *, row: Tensor, column: Tensor) -> Tensor:
    """Bilinearly sample one ``[B,C,H,W]`` plane at ``[B,N]`` coordinates."""

    grid = _plane_grid(row, column, rows=plane.shape[-2], columns=plane.shape[-1])
    # ``grid`` derives from physical RAS-mm geometry and deliberately retains
    # its dtype.  The static neural anchor may be autocast to lower precision,
    # so align it to the physical query only at the differentiable sampling
    # boundary rather than reducing physical coordinates to anchor precision.
    with torch.autocast(device_type=grid.device.type, enabled=False):
        sampling_plane = plane.to(dtype=grid.dtype)
        sampled = F.grid_sample(
            sampling_plane,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
    return sampled[..., 0].transpose(1, 2)


class SpectralPointQuery(nn.Module):
    """Parameter-free geometry-aware bilinear query of one static anchor.

    Inputs are the fixed Phase-6 anchor, refined RAS-mm points, and the
    derived feature-grid geometry for the exact selected spectral tap.  No
    clipping is performed: out-of-domain points follow the existing
    ``grid_sample(..., padding_mode='zeros')`` convention.
    """

    def forward(
        self,
        anchor: SpectralAnchor,
        points_ras_mm: Tensor,
        feature_geometry: FeatureGridGeometry,
    ) -> SpectralPointSamples:
        if not isinstance(anchor, SpectralAnchor):
            raise TypeError("anchor must be a SpectralAnchor")
        if not isinstance(feature_geometry, FeatureGridGeometry):
            raise TypeError("feature_geometry must be a FeatureGridGeometry")
        _validate_float_tensor("points_ras_mm", points_ras_mm, rank=3, final_dimension=3)
        if points_ras_mm.shape[0] != anchor.xy.shape[0] or points_ras_mm.shape[1] <= 0:
            raise ValueError("points_ras_mm must have shape [B, N, 3] matching the anchor batch")
        if points_ras_mm.device != anchor.xy.device:
            raise ValueError("points_ras_mm and anchor must share one device")

        depth, height, width = feature_geometry.shape_dhw
        expected_shapes = (
            ("anchor.xy", anchor.xy, (height, width)),
            ("anchor.xz", anchor.xz, (depth, width)),
            ("anchor.yz", anchor.yz, (depth, height)),
        )
        for name, plane, spatial_shape in expected_shapes:
            if tuple(plane.shape[-2:]) != spatial_shape:
                raise ValueError(
                    f"{name} spatial shape {tuple(plane.shape[-2:])} does not match "
                    f"the derived {feature_geometry.tap!r} feature grid {feature_geometry.shape_dhw}"
                )

        feature_dhw = feature_geometry.ras_mm_to_feature_dhw(points_ras_mm)
        depth_coordinate, height_coordinate, width_coordinate = feature_dhw.unbind(dim=-1)
        return SpectralPointSamples(
            xy=_sample_plane(anchor.xy, row=height_coordinate, column=width_coordinate),
            xz=_sample_plane(anchor.xz, row=depth_coordinate, column=width_coordinate),
            yz=_sample_plane(anchor.yz, row=depth_coordinate, column=height_coordinate),
        )
