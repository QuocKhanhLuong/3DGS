"""Public composition of the fully locked point-guided frontend."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from .baseline_inference import BaselineInferenceResult, GateGInferenceConfig, run_baseline_inference
from .config import PointGuidedConfig
from .contracts import FrontendOutput, PointSpectralEvidence, VolumeGeometry
from .cross_plane_consistency import CrossPlaneConsistency
from .decoder import ImplicitTriPlaneDecoder, ReconstructionOutput
from .points import DeterministicPointInitializer
from .pou import SparseSemanticPoU
from .refinement import PointRefiner
from .semantic_prior import SemanticPrior
from .spectral_anchor import StaticSpectralAnchor
from .spectral_query import FeatureGridGeometry, SpectralPointQuery, derive_feature_grid_geometry
from .reward import GateBDescriptorContext
from .trajectory import AdaptiveRewardCostTrajectory, FrontendTrajectoryOutput, RouteAvailabilityPolicy
from .trajectory_cost import TrajectoryConfig
from .training_objective import (
    GateESupervisionContext,
    SupervisionConfig,
    TrainingObjectiveResult,
    _compute_training_objective,
)
from .triplane_projection import BaseTriPlaneProjector


class PointGuidedMRIModel(nn.Module):
    """Compose the locked frontend plus an explicit, bounded Gate-C trajectory.

    The model accepts normalized full MRI volumes in ``[B, 3, D, H, W]`` order
    ``(T1, T2, FLAIR)``.  ``spacing_mm`` always means ``(x, y, z)`` and is
    converted explicitly to the tensor's ``(w, h, d)`` physical axes.
    ``forward_frontend`` remains the unchanged Phase-1--7 diagnostic API;
    ``forward_trajectory`` is available only when a typed ``TrajectoryConfig``
    is explicitly supplied.  Neither path decodes a T1ce volume.
    """

    def __init__(self, config: PointGuidedConfig, *, trajectory_config: TrajectoryConfig | None = None) -> None:
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
        # The Phase-6 builder is persistent so its single shared band
        # projector participates in optimizers and checkpoint state.  It is
        # deliberately constructed after existing Phase-1–5 modules so adding
        # this diagnostic branch does not perturb their initialization order.
        self.spectral_anchor_builder = StaticSpectralAnchor(
            config,
            input_channels=self.semantic_prior.selected_spectral_feature_channels,
        )
        # Gate B is deliberately parameter-free.  Persistent ownership keeps
        # the one-call frontend composition inspectable without adding model
        # state or a learned spectral fusion path.
        self.spectral_point_query = SpectralPointQuery()
        self.cross_plane_consistency = CrossPlaneConsistency()
        self.trajectory = (
            AdaptiveRewardCostTrajectory(trajectory_config) if trajectory_config is not None else None
        )
        # Gate D is opt-in with Gate C so the existing Phase-1--7 constructor
        # retains its historical checkpoint and state-dict boundary.
        self.decoder = ImplicitTriPlaneDecoder() if trajectory_config is not None else None

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

    def _forward_frontend_with_gate_b_context(
        self,
        x: torch.Tensor,
        brain_mask: torch.Tensor | None = None,
        spacing_mm: Sequence[float] = (1.0, 1.0, 1.0),
        *,
        voxel_to_ras_mm: Sequence[Sequence[float]] | None = None,
    ) -> tuple[FrontendOutput, GateBDescriptorContext, FeatureGridGeometry]:
        """Run Phase 1-7 once and retain only private descriptors for Gate C.

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
        feature_grid_geometry = derive_feature_grid_geometry(
            self.semantic_prior.backbone,
            geometry,
            tap=self.config.spectral_tap,
            observed_shape_dhw=selected_feature.shape[-3:],
        )
        base_planes = self.base_plane_projector(selected_feature)
        spectral_anchor = self.spectral_anchor_builder(base_planes)
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
        spectral_samples = self.spectral_point_query(
            spectral_anchor,
            point_field.refined_centers_ras_mm,
            feature_grid_geometry,
        )
        consistency = self.cross_plane_consistency(
            spectral_samples.xy,
            spectral_samples.xz,
            spectral_samples.yz,
        )
        spectral_evidence = PointSpectralEvidence(
            f_spec=consistency.f_spec,
            reliability=consistency.reliability,
        )
        output = FrontendOutput(
            s_coarse=s_coarse,
            initial_points_ras_mm=point_field.original_centers_ras_mm,
            refined_points_ras_mm=point_field.refined_centers_ras_mm,
            displacement_ras_mm=point_field.displacement_ras_mm,
            point_semantic=point_field.semantic_vectors,
            sparse_pou=sparse_pou,
            geometry=geometry,
            base_planes=base_planes,
            spectral_anchor=spectral_anchor,
            spectral_evidence=spectral_evidence,
        )
        return (
            output,
            GateBDescriptorContext(
                q_xy=consistency.q_xy,
                q_xz=consistency.q_xz,
                q_yz=consistency.q_yz,
            ),
            feature_grid_geometry,
        )

    def forward_frontend(
        self,
        x: torch.Tensor,
        brain_mask: torch.Tensor | None = None,
        spacing_mm: Sequence[float] = (1.0, 1.0, 1.0),
        *,
        voxel_to_ras_mm: Sequence[Sequence[float]] | None = None,
    ) -> FrontendOutput:
        """Run the Phase 1-7 frontend without decoding a T1ce target."""

        output, _, _ = self._forward_frontend_with_gate_b_context(
            x,
            brain_mask,
            spacing_mm,
            voxel_to_ras_mm=voxel_to_ras_mm,
        )
        return output

    def forward_trajectory(
        self,
        x: torch.Tensor,
        brain_mask: torch.Tensor | None = None,
        spacing_mm: Sequence[float] = (1.0, 1.0, 1.0),
        *,
        voxel_to_ras_mm: Sequence[Sequence[float]] | None = None,
    ) -> FrontendTrajectoryOutput:
        """Run C1-C7 once after the shared frontend; never decode a T1ce volume."""

        if self.trajectory is None:
            raise RuntimeError("forward_trajectory requires an explicit TrajectoryConfig at model construction")
        output, gate_b_descriptors, feature_grid_geometry = self._forward_frontend_with_gate_b_context(
            x,
            brain_mask,
            spacing_mm,
            voxel_to_ras_mm=voxel_to_ras_mm,
        )
        trajectory = self.trajectory(
            output.base_planes,
            output.refined_points_ras_mm,
            output.point_semantic,
            output.f_spec,
            output.reliability,
            gate_b_descriptors,
            feature_grid_geometry,
            output.geometry,
        )
        return FrontendTrajectoryOutput(frontend=output, trajectory=trajectory)

    def forward_reconstruction(
        self,
        x: torch.Tensor,
        brain_mask: torch.Tensor | None = None,
        spacing_mm: Sequence[float] = (1.0, 1.0, 1.0),
        *,
        voxel_to_ras_mm: Sequence[Sequence[float]] | None = None,
        chunk_size: int,
    ) -> ReconstructionOutput:
        """Run one frontend, one Gate-C trajectory, then decode final ``Z`` once.

        The decoder receives only the final dynamic state and typed geometry;
        all static observations remain upstream and are not decoder inputs.
        """

        if self.trajectory is None or self.decoder is None:
            raise RuntimeError("forward_reconstruction requires an explicit TrajectoryConfig at model construction")
        output, gate_b_descriptors, feature_grid_geometry = self._forward_frontend_with_gate_b_context(
            x,
            brain_mask,
            spacing_mm,
            voxel_to_ras_mm=voxel_to_ras_mm,
        )
        trajectory = self.trajectory(
            output.base_planes,
            output.refined_points_ras_mm,
            output.point_semantic,
            output.f_spec,
            output.reliability,
            gate_b_descriptors,
            feature_grid_geometry,
            output.geometry,
        )
        return ReconstructionOutput(
            prediction=self.decoder(
                trajectory.final_state,
                feature_grid_geometry,
                output.geometry,
                chunk_size=chunk_size,
            ),
            geometry=output.geometry,
        )

    def forward_baseline_inference(
        self,
        x: torch.Tensor,
        brain_mask: torch.Tensor | None = None,
        spacing_mm: Sequence[float] = (1.0, 1.0, 1.0),
        *,
        inference_config: GateGInferenceConfig | None = None,
        voxel_to_ras_mm: Sequence[Sequence[float]] | None = None,
    ) -> BaselineInferenceResult:
        """Run the completed target-free Gate-G G1--G4 policy once.

        This API enforces eval/no-grad execution, performs one shared frontend
        traversal, runs hard deterministic exact-no-revisit routing over the
        existing Gate-C modules, and performs one final-Z dense decode.  It
        accepts no target, loss, optimizer, or checkpoint selector.
        """

        if self.trajectory is None or self.decoder is None:
            raise RuntimeError("forward_baseline_inference requires an explicit TrajectoryConfig at model construction")
        config = GateGInferenceConfig() if inference_config is None else inference_config
        if not isinstance(config, GateGInferenceConfig):
            raise TypeError("inference_config must be a GateGInferenceConfig")
        was_training = self.training
        try:
            self.eval()
            with torch.no_grad():
                output, gate_b_descriptors, feature_grid_geometry = self._forward_frontend_with_gate_b_context(
                    x,
                    brain_mask,
                    spacing_mm,
                    voxel_to_ras_mm=voxel_to_ras_mm,
                )
                return run_baseline_inference(
                    self.trajectory,
                    self.decoder,
                    output.base_planes,
                    output.refined_points_ras_mm,
                    output.point_semantic,
                    output.f_spec,
                    output.reliability,
                    gate_b_descriptors,
                    feature_grid_geometry,
                    output.geometry,
                    config=config,
                )
        finally:
            self.train(was_training)

    def forward_training_context(
        self,
        x: torch.Tensor,
        brain_mask: torch.Tensor | None = None,
        spacing_mm: Sequence[float] = (1.0, 1.0, 1.0),
        *,
        voxel_to_ras_mm: Sequence[Sequence[float]] | None = None,
        chunk_size: int,
        availability_policy: RouteAvailabilityPolicy | None = None,
    ) -> GateESupervisionContext:
        """Build one target-free Gate-E supervision context.

        This performs the shared frontend, one Gate-C route with its bounded
        training trace, and one final-Z Gate-D decode.  It intentionally does
        not accept T1ce: callers pass any target only later to
        :meth:`compute_training_objective`.
        """

        if self.trajectory is None or self.decoder is None:
            raise RuntimeError("forward_training_context requires an explicit TrajectoryConfig at model construction")
        output, gate_b_descriptors, feature_grid_geometry = self._forward_frontend_with_gate_b_context(
            x,
            brain_mask,
            spacing_mm,
            voxel_to_ras_mm=voxel_to_ras_mm,
        )
        trace = self.trajectory._forward_with_training_trace(
            output.base_planes,
            output.refined_points_ras_mm,
            output.point_semantic,
            output.f_spec,
            output.reliability,
            gate_b_descriptors,
            feature_grid_geometry,
            output.geometry,
            availability_policy=availability_policy,
        )
        reconstruction = ReconstructionOutput(
            prediction=self.decoder(
                trace.result.final_state,
                feature_grid_geometry,
                output.geometry,
                chunk_size=chunk_size,
            ),
            geometry=output.geometry,
        )
        return GateESupervisionContext(
            frontend=output,
            _trace=trace,
            _trajectory=self.trajectory,
            _decoder=self.decoder,
            reconstruction=reconstruction,
            gate_b_descriptors=gate_b_descriptors,
            feature_geometry=feature_grid_geometry,
        )

    def forward_supervision_context(
        self,
        x: torch.Tensor,
        brain_mask: torch.Tensor | None = None,
        spacing_mm: Sequence[float] = (1.0, 1.0, 1.0),
        *,
        voxel_to_ras_mm: Sequence[Sequence[float]] | None = None,
        chunk_size: int,
        availability_policy: RouteAvailabilityPolicy | None = None,
    ) -> GateESupervisionContext:
        """Alias emphasizing that this target-free context is Gate-E-only."""

        return self.forward_training_context(
            x,
            brain_mask,
            spacing_mm,
            voxel_to_ras_mm=voxel_to_ras_mm,
            chunk_size=chunk_size,
            availability_policy=availability_policy,
        )

    def compute_training_objective(
        self,
        context: GateESupervisionContext,
        target_t1ce: torch.Tensor,
        *,
        config: SupervisionConfig | None = None,
        valid_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> TrainingObjectiveResult:
        """Apply E1--E8 only after a target-free context already exists."""

        if self.trajectory is None or self.decoder is None:
            raise RuntimeError("compute_training_objective requires an explicit TrajectoryConfig at model construction")
        return _compute_training_objective(
            context,
            target_t1ce,
            config=config,
            valid_mask=valid_mask,
            generator=generator,
        )

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        """Refuse the generic full-model API until later governance defines it."""

        del args, kwargs
        raise NotImplementedError(
            "Full T1ce synthesis is unresolved through the generic forward() API. "
            "Use forward_frontend(), forward_trajectory(), or the explicit Gate-D "
            "forward_reconstruction() API."
        )


__all__ = ["PointGuidedMRIModel"]
