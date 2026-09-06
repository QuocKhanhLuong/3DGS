"""Live multi-scale centre geometry for PFGR static synthesis.

The static B1/B2 heads consume shallow, Layer1, and deep MedicalNet maps.
This module derives each map's cell-centre transform from the instantiated
Conv3d/MaxPool3d/residual metadata.  It never substitutes a hard-coded
``/2`` scale, diagonal spacing, or anatomical axis transform, and it rejects
residual branches whose lattices disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..contracts import VolumeGeometry
from ..sampling import ras_mm_to_voxel_dhw, voxel_dhw_to_grid_sample_coordinates, voxel_dhw_to_ras_mm
from ..spectral_query import FeatureGridGeometry

if TYPE_CHECKING:
    from ..medicalnet_resnet10 import MedicalNetFeatures


STATIC_GEOMETRY_VERSION = "pfgr-lite-static-geometry-v1"


def _shape(name: str, value: Sequence[int]) -> tuple[int, int, int]:
    result = tuple(value)
    if len(result) != 3 or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in result):
        raise ValueError(f"{name} must contain three positive DHW integers")
    return result  # type: ignore[return-value]


def _triple(name: str, value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        result = (value, value, value)
    else:
        result = tuple(value)  # type: ignore[arg-type]
    if len(result) != 3 or any(not isinstance(item, int) or isinstance(item, bool) for item in result):
        raise ValueError(f"{name} must contain three integers")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class FeatureLattice:
    """Feature map shape and source-cell centre transform."""

    name: str
    source_geometry: VolumeGeometry
    feature_geometry: VolumeGeometry
    feature_shape_dhw: tuple[int, int, int]
    scale_dhw: tuple[float, float, float]
    offset_dhw: tuple[float, float, float]
    operator_chain: tuple[str, ...]
    version: str = STATIC_GEOMETRY_VERSION

    def __post_init__(self) -> None:
        if self.version != STATIC_GEOMETRY_VERSION:
            raise ValueError("unknown PFGR static geometry version")
        if not isinstance(self.source_geometry, VolumeGeometry) or not isinstance(self.feature_geometry, VolumeGeometry):
            raise TypeError("source_geometry and feature_geometry must be VolumeGeometry")
        object.__setattr__(self, "feature_shape_dhw", _shape("feature_shape_dhw", self.feature_shape_dhw))
        scale = tuple(float(item) for item in self.scale_dhw)
        offset = tuple(float(item) for item in self.offset_dhw)
        if len(scale) != 3 or any(not math.isfinite(item) or item <= 0.0 for item in scale):
            raise ValueError("scale_dhw must contain positive finite values")
        if len(offset) != 3 or any(not math.isfinite(item) for item in offset):
            raise ValueError("offset_dhw must contain finite values")
        object.__setattr__(self, "scale_dhw", scale)
        object.__setattr__(self, "offset_dhw", offset)
        chain = tuple(self.operator_chain)
        if not chain or any(not isinstance(item, str) or not item for item in chain):
            raise ValueError("operator_chain must contain nonempty labels")
        object.__setattr__(self, "operator_chain", chain)
        if self.feature_geometry.shape_dhw != self.feature_shape_dhw:
            raise ValueError("feature_geometry shape must equal feature_shape_dhw")

    @property
    def shape_dhw(self) -> tuple[int, int, int]:
        return self.feature_shape_dhw

    def feature_dhw_to_ras_mm(self, feature_dhw: Tensor) -> Tensor:
        return voxel_dhw_to_ras_mm(feature_dhw, self.feature_geometry)

    def ras_mm_to_feature_dhw(self, ras_mm: Tensor) -> Tensor:
        return ras_mm_to_voxel_dhw(ras_mm, self.feature_geometry)

    def digest_payload(self) -> dict[str, object]:
        return {
            "version": self.version,
            "name": self.name,
            "source_shape_dhw": self.source_geometry.shape_dhw,
            "feature_shape_dhw": self.feature_shape_dhw,
            "source_affine": self.source_geometry.voxel_to_ras_mm,
            "feature_affine": self.feature_geometry.voxel_to_ras_mm,
            "scale_dhw": self.scale_dhw,
            "offset_dhw": self.offset_dhw,
            "operator_chain": self.operator_chain,
        }


@dataclass(frozen=True)
class MultiScaleFeatureGeometry:
    """Shallow/Layer1/deep lattices from one MedicalNet traversal."""

    shallow: FeatureLattice
    layer1: FeatureLattice
    deep: FeatureLattice
    version: str = STATIC_GEOMETRY_VERSION

    def __post_init__(self) -> None:
        if self.version != STATIC_GEOMETRY_VERSION:
            raise ValueError("unknown PFGR multiscale geometry version")
        names = (self.shallow.name, self.layer1.name, self.deep.name)
        if names != ("shallow", "layer1", "deep"):
            raise ValueError("multiscale lattices must be named shallow/layer1/deep")
        source = self.shallow.source_geometry
        if self.layer1.source_geometry != source or self.deep.source_geometry != source:
            raise ValueError("all multiscale lattices must share source geometry")

    @property
    def source_geometry(self) -> VolumeGeometry:
        return self.shallow.source_geometry

    def __getitem__(self, key: str) -> FeatureLattice:
        if key not in ("shallow", "layer1", "deep"):
            raise KeyError(key)
        return getattr(self, key)

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name).digest_payload() for name in ("shallow", "layer1", "deep")}


@dataclass(frozen=True)
class _State:
    shape_dhw: tuple[int, int, int]
    scale_dhw: tuple[float, float, float]
    offset_dhw: tuple[float, float, float]
    chain: tuple[str, ...]


def _module_meta(module: nn.Module, *, label: str) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    if isinstance(module, nn.Conv3d):
        kernel = _triple(f"{label}.kernel_size", module.kernel_size)
        stride = _triple(f"{label}.stride", module.stride)
        padding = _triple(f"{label}.padding", module.padding)
        dilation = _triple(f"{label}.dilation", module.dilation)
    elif isinstance(module, nn.MaxPool3d):
        if module.ceil_mode or module.return_indices:
            raise ValueError(f"{label} has unsupported ceil_mode/return_indices")
        kernel = _triple(f"{label}.kernel_size", module.kernel_size)
        stride = _triple(f"{label}.stride", module.stride if module.stride is not None else module.kernel_size)
        padding = _triple(f"{label}.padding", module.padding)
        dilation = _triple(f"{label}.dilation", module.dilation)
    else:
        raise ValueError(f"unsupported spatial module {label}: {type(module).__name__}")
    if any(item <= 0 for item in (*kernel, *stride, *dilation)) or any(item < 0 for item in padding):
        raise ValueError(f"{label} has invalid spatial metadata")
    return kernel, stride, padding, dilation


def _apply_spatial(state: _State, module: nn.Module, *, label: str) -> _State:
    kernel, stride, padding, dilation = _module_meta(module, label=label)
    shape: list[int] = []
    local_offset: list[float] = []
    for length, k, s, p, dil in zip(state.shape_dhw, kernel, stride, padding, dilation):
        output = (length + 2 * p - dil * (k - 1) - 1) // s + 1
        if output <= 0:
            raise ValueError(f"{label} produces nonpositive spatial shape")
        shape.append(output)
        local_offset.append(dil * (k - 1) / 2.0 - p)
    return _State(
        shape_dhw=tuple(shape),  # type: ignore[arg-type]
        scale_dhw=tuple(old * float(step) for old, step in zip(state.scale_dhw, stride)),  # type: ignore[arg-type]
        offset_dhw=tuple(old_s * off + old_o for old_s, off, old_o in zip(state.scale_dhw, local_offset, state.offset_dhw)),  # type: ignore[arg-type]
        chain=(*state.chain, label),
    )


def _same(left: _State, right: _State) -> bool:
    return left.shape_dhw == right.shape_dhw and left.scale_dhw == right.scale_dhw and all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(left.offset_dhw, right.offset_dhw)
    )


def _neutral(state: _State, module: nn.Module, *, label: str) -> _State:
    if isinstance(module, (nn.Conv3d, nn.MaxPool3d)):
        return _apply_spatial(state, module, label=label)
    if isinstance(module, nn.Sequential):
        current = state
        for index, child in enumerate(module):
            current = _neutral(current, child, label=f"{label}[{index}]")
        return current
    if isinstance(module, (nn.BatchNorm3d, nn.ReLU, nn.Identity)):
        return state
    raise ValueError(f"unsupported residual path module {label}: {type(module).__name__}")


def _residual(state: _State, block: nn.Module, *, label: str) -> _State:
    conv1 = getattr(block, "conv1", None)
    conv2 = getattr(block, "conv2", None)
    shortcut = getattr(block, "downsample", None)
    if not isinstance(conv1, nn.Conv3d) or not isinstance(conv2, nn.Conv3d):
        raise ValueError(f"{label} must expose Conv3d conv1/conv2")
    main = _apply_spatial(state, conv1, label=f"{label}.conv1")
    main = _apply_spatial(main, conv2, label=f"{label}.conv2")
    skip = state if shortcut is None else _neutral(state, shortcut, label=f"{label}.downsample")
    if not _same(main, skip):
        raise ValueError(f"residual main/shortcut centre lattices disagree at {label}")
    return main


def _feature_geometry(source: VolumeGeometry, state: _State) -> VolumeGeometry:
    # Feature tensor axes are DHW; VolumeGeometry affine consumes WHD.
    scale_w, scale_h, scale_d = state.scale_dhw[2], state.scale_dhw[1], state.scale_dhw[0]
    offset_w, offset_h, offset_d = state.offset_dhw[2], state.offset_dhw[1], state.offset_dhw[0]
    feature_to_source = (
        (scale_w, 0.0, 0.0, offset_w),
        (0.0, scale_h, 0.0, offset_h),
        (0.0, 0.0, scale_d, offset_d),
        (0.0, 0.0, 0.0, 1.0),
    )
    source_affine = source.voxel_to_ras_mm
    composed = tuple(
        tuple(sum(source_affine[row][inner] * feature_to_source[inner][column] for inner in range(4)) for column in range(4))
        for row in range(4)
    )
    return VolumeGeometry(state.shape_dhw, composed)


def _lattice(name: str, source: VolumeGeometry, state: _State, observed: Sequence[int]) -> FeatureLattice:
    observed_shape = _shape(f"{name} observed shape", observed)
    if observed_shape != state.shape_dhw:
        raise ValueError(f"derived {name} shape {state.shape_dhw} differs from observed {observed_shape}")
    geometry = _feature_geometry(source, state)
    return FeatureLattice(name, source, geometry, observed_shape, state.scale_dhw, state.offset_dhw, state.chain)


def derive_multiscale_feature_geometry(
    backbone: nn.Module,
    source_geometry: VolumeGeometry,
    features: "MedicalNetFeatures | Mapping[str, Tensor] | None" = None,
) -> MultiScaleFeatureGeometry:
    """Derive shallow, Layer1, and deep centre lattices from live modules."""

    if not isinstance(backbone, nn.Module):
        raise TypeError("backbone must be an nn.Module")
    if not isinstance(source_geometry, VolumeGeometry):
        raise TypeError("source_geometry must be a VolumeGeometry")
    state = _State(source_geometry.shape_dhw, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), ())
    conv1 = getattr(backbone, "conv1", None)
    if not isinstance(conv1, nn.Conv3d):
        raise ValueError("backbone.conv1 must be Conv3d")
    shallow_state = _apply_spatial(state, conv1, label="conv1")
    maxpool = getattr(backbone, "maxpool", None)
    layer1_module = getattr(backbone, "layer1", None)
    if not isinstance(maxpool, nn.MaxPool3d) or not isinstance(layer1_module, nn.Sequential) or not layer1_module:
        raise ValueError("backbone must expose maxpool and nonempty layer1")
    layer1_state = _apply_spatial(shallow_state, maxpool, label="maxpool")
    for index, block in enumerate(layer1_module):
        layer1_state = _residual(layer1_state, block, label=f"layer1[{index}]")
    deep_state = layer1_state
    for layer_name in ("layer2", "layer3", "layer4"):
        layer = getattr(backbone, layer_name, None)
        if not isinstance(layer, nn.Sequential) or not layer:
            raise ValueError(f"backbone must expose nonempty {layer_name}")
        for index, block in enumerate(layer):
            deep_state = _residual(deep_state, block, label=f"{layer_name}[{index}]")

    observed_shapes: dict[str, Sequence[int]] = {
        "shallow": shallow_state.shape_dhw,
        "layer1": layer1_state.shape_dhw,
        "deep": deep_state.shape_dhw,
    }
    if features is not None:
        if isinstance(features, Mapping):
            observed_shapes = {name: features[name].shape[-3:] for name in observed_shapes}
        else:
            observed_shapes = {name: getattr(features, name).shape[-3:] for name in observed_shapes}
    return MultiScaleFeatureGeometry(
        shallow=_lattice("shallow", source_geometry, shallow_state, observed_shapes["shallow"]),
        layer1=_lattice("layer1", source_geometry, layer1_state, observed_shapes["layer1"]),
        deep=_lattice("deep", source_geometry, deep_state, observed_shapes["deep"]),
    )


# Names used by early PFGR callers; all aliases resolve to the same geometry
# derivation and therefore cannot diverge in centre semantics.
derive_multiscale_feature_geometries = derive_multiscale_feature_geometry
derive_feature_lattices = derive_multiscale_feature_geometry
derive_static_feature_geometry = derive_multiscale_feature_geometry


def _mesh_indices(shape_dhw: Sequence[int], *, device: torch.device, dtype: torch.dtype) -> Tensor:
    d, h, w = _shape("shape_dhw", shape_dhw)
    dd, hh, ww = torch.meshgrid(
        torch.arange(d, device=device, dtype=dtype),
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((dd, hh, ww), dim=-1).reshape(1, d * h * w, 3)


def sample_source_to_lattice(source: Tensor, lattice: FeatureLattice, *, chunk_size: int | None = None) -> Tensor:
    """Trilinearly sample ordered source channels at feature cell centres."""

    if not isinstance(source, Tensor) or source.ndim != 5 or not source.is_floating_point():
        raise ValueError("source must be floating [B,C,D,H,W]")
    if tuple(source.shape[-3:]) != lattice.source_geometry.shape_dhw:
        raise ValueError("source spatial shape must match lattice source geometry")
    if not bool(torch.isfinite(source).all()):
        raise ValueError("source must be finite")
    batch = source.shape[0]
    indices = _mesh_indices(lattice.feature_shape_dhw, device=source.device, dtype=source.dtype)
    ras = lattice.feature_dhw_to_ras_mm(indices)
    source_dhw = lattice.source_geometry
    source_indices = ras_mm_to_voxel_dhw(ras, source_dhw)
    grid = voxel_dhw_to_grid_sample_coordinates(source_indices, source_dhw.shape_dhw).reshape(1, *lattice.feature_shape_dhw, 3)
    grid = grid.expand(batch, -1, -1, -1, -1)
    # The grid is [x,y,z] while source is [d,h,w], as required by grid_sample.
    with torch.autocast(device_type=source.device.type, enabled=False):
        sampled = F.grid_sample(source, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return sampled


def _plane_grid_from_lattices(
    target_lattice: FeatureLattice,
    source_lattice: FeatureLattice,
    *,
    plane: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    shape = target_lattice.feature_shape_dhw
    indices = _mesh_indices(shape, device=device, dtype=dtype).reshape(1, *shape, 3)
    ras = target_lattice.feature_dhw_to_ras_mm(indices.reshape(1, -1, 3)).reshape(1, *shape, 3)
    old = source_lattice.ras_mm_to_feature_dhw(ras.reshape(1, -1, 3)).reshape(1, *shape, 3)
    d, h, w = old.unbind(dim=-1)
    if plane == "xy":
        row, column = h, w
        rows, columns = source_lattice.feature_shape_dhw[1], source_lattice.feature_shape_dhw[2]
    elif plane == "xz":
        row, column = d, w
        rows, columns = source_lattice.feature_shape_dhw[0], source_lattice.feature_shape_dhw[2]
    elif plane == "yz":
        row, column = d, h
        rows, columns = source_lattice.feature_shape_dhw[0], source_lattice.feature_shape_dhw[1]
    else:
        raise ValueError("plane must be xy, xz, or yz")
    x = (2.0 * column + 1.0) / float(columns) - 1.0
    y = (2.0 * row + 1.0) / float(rows) - 1.0
    if plane == "xy":
        return torch.stack((x, y), dim=-1).reshape(1, shape[0], shape[1], shape[2], 2)[:, 0]
    if plane == "xz":
        return torch.stack((x, y), dim=-1).reshape(1, shape[0], shape[1], shape[2], 2)[:, :, 0]
    return torch.stack((x, y), dim=-1).reshape(1, shape[0], shape[1], shape[2], 2)[:, :, :, 0]


def resample_plane_between_lattices(
    plane: Tensor,
    source_lattice: FeatureLattice,
    target_lattice: FeatureLattice,
    *,
    plane_name: str,
) -> Tensor:
    """Geometry-aware bilinear resampling between scale-specific 2-D planes."""

    if not isinstance(plane, Tensor) or plane.ndim != 4 or not plane.is_floating_point():
        raise ValueError("plane must be floating [B,C,H,W]")
    expected_source = {
        "xy": (source_lattice.feature_shape_dhw[1], source_lattice.feature_shape_dhw[2]),
        "xz": (source_lattice.feature_shape_dhw[0], source_lattice.feature_shape_dhw[2]),
        "yz": (source_lattice.feature_shape_dhw[0], source_lattice.feature_shape_dhw[1]),
    }
    expected_target = {
        "xy": (target_lattice.feature_shape_dhw[1], target_lattice.feature_shape_dhw[2]),
        "xz": (target_lattice.feature_shape_dhw[0], target_lattice.feature_shape_dhw[2]),
        "yz": (target_lattice.feature_shape_dhw[0], target_lattice.feature_shape_dhw[1]),
    }
    if plane_name not in expected_source:
        raise ValueError("plane_name must be xy, xz, or yz")
    if tuple(plane.shape[-2:]) != expected_source[plane_name]:
        raise ValueError(f"plane shape does not match source {plane_name} lattice")
    target_shape = expected_target[plane_name]
    # Build target physical coordinates in a flattened plane order.
    td, th, tw = target_lattice.feature_shape_dhw
    if plane_name == "xy":
        row, col = torch.meshgrid(torch.arange(th, device=plane.device, dtype=plane.dtype), torch.arange(tw, device=plane.device, dtype=plane.dtype), indexing="ij")
        target_dhw = torch.stack((torch.zeros_like(row), row, col), dim=-1).reshape(1, -1, 3)
    elif plane_name == "xz":
        row, col = torch.meshgrid(torch.arange(td, device=plane.device, dtype=plane.dtype), torch.arange(tw, device=plane.device, dtype=plane.dtype), indexing="ij")
        target_dhw = torch.stack((row, torch.zeros_like(row), col), dim=-1).reshape(1, -1, 3)
    else:
        row, col = torch.meshgrid(torch.arange(td, device=plane.device, dtype=plane.dtype), torch.arange(th, device=plane.device, dtype=plane.dtype), indexing="ij")
        target_dhw = torch.stack((row, col, torch.zeros_like(row)), dim=-1).reshape(1, -1, 3)
    ras = target_lattice.feature_dhw_to_ras_mm(target_dhw)
    old_dhw = source_lattice.ras_mm_to_feature_dhw(ras)
    d, h, w = old_dhw.unbind(dim=-1)
    if plane_name == "xy":
        source_row, source_col = h, w
    elif plane_name == "xz":
        source_row, source_col = d, w
    else:
        source_row, source_col = d, h
    rows, cols = expected_source[plane_name]
    grid = torch.stack(((2.0 * source_col + 1.0) / float(cols) - 1.0, (2.0 * source_row + 1.0) / float(rows) - 1.0), dim=-1).reshape(1, *target_shape, 2)
    grid = grid.expand(plane.shape[0], -1, -1, -1)
    with torch.autocast(device_type=plane.device.type, enabled=False):
        return F.grid_sample(plane, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def source_plane_means(source_aligned: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Collapse an aligned source map into ordered 3-channel XY/XZ/YZ planes."""

    if not isinstance(source_aligned, Tensor) or source_aligned.ndim != 5 or source_aligned.shape[1] != 3:
        raise ValueError("source_aligned must have shape [B,3,D,H,W]")
    return source_aligned.mean(dim=2), source_aligned.mean(dim=3), source_aligned.mean(dim=4)


__all__ = [
    "FeatureLattice",
    "FeatureGridGeometry",
    "MultiScaleFeatureGeometry",
    "STATIC_GEOMETRY_VERSION",
    "derive_feature_lattices",
    "derive_multiscale_feature_geometries",
    "derive_multiscale_feature_geometry",
    "derive_static_feature_geometry",
    "resample_plane_between_lattices",
    "sample_source_to_lattice",
    "source_plane_means",
]
