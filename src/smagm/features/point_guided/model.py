"""Public composition of the fully locked point-guided frontend."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from .config import PointGuidedConfig
from .contracts import FrontendOutput, VolumeGeometry
from .points import DeterministicPointInitializer
from .pou import SparseSemanticPoU
from .refinement import PointRefiner
from .semantic_prior import SemanticPrior
from .triplane_projection import BaseTriPlaneProjector


class PointGuidedMRIModel(nn.Module):
    """Compose the locked frontend and intentionally stop before trajectory work.

    The model accepts normalized full MRI volumes in ``[B, 3, D, H, W]`` order
    ``(T1, T2, FLAIR)``.  ``spacing_mm`` always means ``(x, y, z)`` and is
    converted explicitly to the tensor's ``(w, h, d)`` physical axes.
    """

    def __init__(self, config: PointGuidedConfig) -> None:
        super().__init__()
        if not isinstance(config, PointGuidedConfig):
            raise TypeError("config must be a PointGuidedConfig")
        self.config = config
        self.semantic_prior = SemanticPrior(config)
        self.point_initializer = DeterministicPointInitializer(config)
        self.point_refiner = PointRefiner(config)
        self.sparse_pou = SparseSemanticPoU(config)
        # This persistent module consumes the Phase-2-selected shared map;
        # channel count is read from the instantiated backbone rather than
        # hard-coded from today's two 64-channel taps.
        self.base_plane_projector = BaseTriPlaneProjector(
            config,
            input_channels=self.semantic_prior.selected_spectral_feature_channels,
        )

    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor) or x.ndim != 5:
            raise ValueError("x must be a floating-point tensor with shape [B, 3, D, H, W]")
        if not x.is_floating_point() or x.shape[0] <= 0 or x.shape[1] != 3:
            raise ValueError("x must have positive batch size and exactly T1/T2/FLAIR channels")
        if any(length <= 0 for length in x.shape[-3:]) or not bool(torch.isfinite(x).all()):
            raise ValueError("x must have positive finite spatial values")

    @staticmethod
    def _geometry(
        spatial_shape_dhw: Sequence[int],
        spacing_mm: Sequence[float],
        voxel_to_ras_mm: Sequence[Sequence[float]] | None,
    ) -> VolumeGeometry:
        if voxel_to_ras_mm is not None:
            geometry = VolumeGeometry(spatial_shape_dhw, voxel_to_ras_mm)
            supplied = tuple(float(value) for value in spacing_mm)
            if len(supplied) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in supplied):
                raise ValueError("spacing_mm must contain three positive finite XYZ values")
            if any(abs(actual - requested) > 1e-5 for actual, requested in zip(geometry.spacing_xyz_mm, supplied)):
                raise ValueError("spacing_mm must agree with voxel_to_ras_mm column lengths")
            return geometry
        return VolumeGeometry.from_spacing(spatial_shape_dhw, spacing_mm)

    def forward_frontend(
        self,
        x: torch.Tensor,
        brain_mask: torch.Tensor | None = None,
        spacing_mm: Sequence[float] = (1.0, 1.0, 1.0),
        *,
        voxel_to_ras_mm: Sequence[Sequence[float]] | None = None,
    ) -> FrontendOutput:
        """Run the complete locked frontend without decoding a T1ce target.

        ``brain_mask`` is an optional binary tensor of shape ``[B, 1, D, H, W]``
        or ``[B, D, H, W]``. It controls legal point/PoU support but never
        selects from intensity values. A caller with affine geometry may pass
        ``voxel_to_ras_mm``; it maps homogeneous ``[w, h, d, 1]`` to RAS-mm.
        """

        self._validate_input(x)
        geometry = self._geometry(x.shape[-3:], spacing_mm, voxel_to_ras_mm)
        features = self.semantic_prior.extract_intermediate_features(x)
        s_coarse = self.semantic_prior.forward_from_intermediate_features(
            features,
            output_spatial_shape=x.shape[-3:],
        )
        selected_feature = self.semantic_prior.select_spectral_feature(features)
        base_planes = self.base_plane_projector(selected_feature)
        semantic_sum = s_coarse.sum(dim=1)
        if not bool(torch.allclose(semantic_sum, torch.ones_like(semantic_sum), atol=1e-5, rtol=1e-5)):
            raise RuntimeError("coarse semantic prior must sum to one per voxel")
        initial_points = self.point_initializer(
            geometry,
            x.shape[0],
            brain_mask,
            device=x.device,
            dtype=x.dtype,
        )
        point_field = self.point_refiner(x, s_coarse, initial_points, geometry)
        sparse_pou = self.sparse_pou(
            point_field,
            s_coarse,
            geometry,
            valid_brain_mask=brain_mask,
        )
        return FrontendOutput(
            s_coarse=s_coarse,
            initial_points_ras_mm=point_field.original_centers_ras_mm,
            refined_points_ras_mm=point_field.refined_centers_ras_mm,
            displacement_ras_mm=point_field.displacement_ras_mm,
            point_semantic=point_field.semantic_vectors,
            sparse_pou=sparse_pou,
            geometry=geometry,
            base_planes=base_planes,
        )

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        """Refuse the unresolved trajectory/decoder path explicitly."""

        del args, kwargs
        raise NotImplementedError(
            "Full T1ce synthesis is unresolved. Use forward_frontend() for the locked frontend only."
        )


__all__ = ["PointGuidedMRIModel"]
