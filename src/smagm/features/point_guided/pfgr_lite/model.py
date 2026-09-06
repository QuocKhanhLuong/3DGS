"""PFGR-Lite target-free static composition and state seam (W1).

This composition wraps the legacy Phase-1--7 frontend with no trajectory or
RewardNet allocation, consumes its private one-traversal feature seam, builds
one of the versioned static heads, and exposes graph-preserving Z0 state.  The
canonical PFGR query lattice is intentionally injected by W2; until that
dependency is wired, ``decode_final`` fails closed rather than silently using
the legacy query implementation.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from ..config import PointGuidedConfig
from ..model import PointGuidedMRIModel
from ..spectral_query import derive_feature_grid_geometry
from .config import PFGRLiteConfig
from .provenance import (
    ProducerCompatibility,
    SourceProvenance,
    batchnorm_state_digest,
    best_effort_git_head,
    canonical_digest,
    module_parameter_digest,
    module_state_digest,
    source_provenance_from_semantic_prior,
    tensor_digest,
)
from .static_geometry import MultiScaleFeatureGeometry, derive_multiscale_feature_geometry
from .static_synthesis import StaticSynthesisHead
from .types import (
    ObservationContext,
    PFGRState,
    ProducerDependencies,
    clone_dynamic_planes,
)


QueryLatticeFactory = Any
"""W2 injection seam: an object exposing ``build(...)->lattice``."""


def _geometry_as_volume(
    x: Tensor,
    geometry: Any,
) -> Any:
    from ..contracts import VolumeGeometry

    if isinstance(geometry, VolumeGeometry):
        if tuple(geometry.shape_dhw) != tuple(x.shape[-3:]):
            raise ValueError("geometry.shape_dhw must match observation spatial shape")
        return geometry
    # A spacing triplet remains a convenience for callers migrating from the
    # legacy frontend; the typed public PFGR contract itself uses
    # VolumeGeometry and retains the resulting affine.
    if isinstance(geometry, Sequence) and not isinstance(geometry, (str, bytes)):
        spacing = tuple(float(value) for value in geometry)
        if len(spacing) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in spacing):
            raise ValueError("geometry spacing must contain three positive finite XYZ values")
        return VolumeGeometry.from_spacing(x.shape[-3:], spacing)
    raise TypeError("geometry must be VolumeGeometry or a positive XYZ spacing triplet")


class PFGRLiteModel(nn.Module):
    """Separate PFGR composition with one legacy frontend traversal.

    ``frontend`` is constructed without ``TrajectoryConfig`` and therefore
    has no legacy trajectory/RewardNet/decoder state.  ``updater`` and
    ``decoder`` are the existing standalone Gate-C/D modules reused by the
    future W4/W2 orchestration.  The public generic ``forward`` is
    deliberately fail-closed.
    """

    def __init__(
        self,
        config: PFGRLiteConfig | None = None,
        *,
        frontend_config: PointGuidedConfig | None = None,
        query_lattice_factory: QueryLatticeFactory | None = None,
    ) -> None:
        super().__init__()
        self.config = PFGRLiteConfig() if config is None else config
        if not isinstance(self.config, PFGRLiteConfig):
            raise TypeError("config must be PFGRLiteConfig")
        if frontend_config is not None and not isinstance(frontend_config, PointGuidedConfig):
            raise TypeError("frontend_config must be PointGuidedConfig")
        if frontend_config is None:
            frontend_config = PointGuidedConfig(
                num_semantic_classes=3,
                num_points=self.config.num_points,
                point_candidate_multiplier=3,
                offset_hidden_channels=12,
            )
        self._validate_frontend_contract(frontend_config)
        self.frontend_config = frontend_config
        # No TrajectoryConfig: this legacy instance owns only the existing
        # observation frontend and static B/A branch.
        self.frontend = PointGuidedMRIModel(frontend_config, trajectory_config=None)
        self.static_head = StaticSynthesisHead(self.config.static)

        # Existing, standalone Gate-C/D modules.  They are the only updater
        # and decoder owned by this composition; W2 supplies query mechanics.
        from ..updater import UpdateNet
        from ..decoder import ImplicitTriPlaneDecoder

        self.updater = UpdateNet()
        self.decoder = ImplicitTriPlaneDecoder()
        self._query_lattice_factory = query_lattice_factory

    @property
    def legacy_frontend(self) -> PointGuidedMRIModel:
        """Explicit alias for source/provenance inspection."""

        return self.frontend

    @property
    def update_net(self):
        """Compatibility alias for the single shared Gate-C ``U`` module."""

        return self.updater

    @property
    def implicit_decoder(self):
        """Compatibility alias for the single shared Gate-D decoder ``D``."""

        return self.decoder

    @staticmethod
    def _validate_observations(x: Tensor) -> None:
        if not isinstance(x, Tensor) or x.ndim != 5 or x.shape[1] != 3:
            raise ValueError("observations must have shape [B,3,D,H,W] ordered T1/T2/FLAIR")
        if not x.is_floating_point() or x.shape[0] <= 0 or any(size <= 0 for size in x.shape[-3:]):
            raise ValueError("observations must be finite floating-point with positive dimensions")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("observations must be finite")

    def _validate_frontend_contract(self, frontend_config: PointGuidedConfig) -> None:
        """Reject a legacy frontend that would silently change PFGR semantics."""

        if not frontend_config.freeze_coarse_backbone:
            raise ValueError("PFGR requires a frozen MedicalNet backbone and BatchNorm state")
        if frontend_config.spectral_tap != "conv1_pre_maxpool":
            raise ValueError("PFGR static synthesis requires the conv1_pre_maxpool spectral tap")
        if frontend_config.num_semantic_classes != 3:
            raise ValueError("PFGR semantic head is locked to exactly three classes")
        if not math.isclose(float(frontend_config.support_radius_mm), 4.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("PFGR support radius must be exactly 4 mm")
        if not math.isclose(float(frontend_config.max_displacement_mm), 2.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("PFGR displacement bound must be exactly 2 mm")
        if frontend_config.num_points != self.config.num_points:
            raise ValueError("frontend_config.num_points must match PFGRLiteConfig.num_points")
        if self.config.num_points != 2048 and not self.config.engineering_only:
            raise ValueError("reduced N requires explicit PFGR engineering_only capability")

    @staticmethod
    def _validate_mask(mask: Tensor | None, x: Tensor) -> Tensor:
        if mask is None:
            return torch.ones(x.shape[0], *x.shape[-3:], dtype=torch.bool, device=x.device)
        if not isinstance(mask, Tensor) or mask.ndim not in (4, 5):
            raise ValueError("brain_mask must be [B,D,H,W] or [B,1,D,H,W]")
        if mask.ndim == 5 and mask.shape[1] != 1:
            raise ValueError("brain_mask rank-5 form must have one channel")
        if tuple(mask.shape[0:1] + mask.shape[-3:]) != tuple(x.shape[0:1] + x.shape[-3:]):
            raise ValueError("brain_mask must match observations")
        if mask.dtype == torch.bool:
            resolved = mask[:, 0] if mask.ndim == 5 else mask
            return resolved.clone()
        if not mask.is_floating_point() and mask.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("brain_mask must be bool or numeric binary mask")
        if not bool(torch.isfinite(mask).all()):
            raise ValueError("brain_mask must be finite")
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("brain_mask must contain only exact binary 0/1 values")
        resolved = mask.to(dtype=torch.bool)
        return (resolved[:, 0] if resolved.ndim == 5 else resolved).clone()

    def _producer_dependencies(self, *, geometries: MultiScaleFeatureGeometry, traversal_count: int) -> ProducerDependencies:
        prior = self.frontend.semantic_prior
        source = source_provenance_from_semantic_prior(
            prior,
            source_sha=best_effort_git_head(),
            config_sha=canonical_digest(self.frontend_config, prefix="pfgr-legacy-config-v1|"),
        )
        source = SourceProvenance(
            schema_version=source.schema_version,
            source_sha=source.source_sha,
            config_sha=source.config_sha,
            implementation_version=source.implementation_version,
            model_family=source.model_family,
            source_input_channels=source.source_input_channels,
            adapted_input_channels=source.adapted_input_channels,
            input_conv_adapted=source.input_conv_adapted,
            checkpoint_path=source.checkpoint_path,
            checkpoint_sha256=source.checkpoint_sha256,
            checkpoint_integrity_verified=source.checkpoint_integrity_verified,
            source_state_dict_key_count=source.source_state_dict_key_count,
            loaded_backbone_key_count=source.loaded_backbone_key_count,
            adaptation_digest=source.adaptation_digest,
            parameter_hash=source.parameter_hash,
            frozen_bn_hash=source.frozen_bn_hash,
            official_pretrained_verified=source.official_pretrained_verified,
            synthetic_untrained=source.synthetic_untrained,
            traversal_count=traversal_count,
            details=source.details,
        )
        compatibility = ProducerCompatibility(
            observation_normalization_hash=canonical_digest(self.config.observation_normalization, prefix="pfgr-lite-observation-normalization-v1|"),
            # Compatibility binds the query algorithm/version, not one
            # subject's affine/shape.  Per-subject geometry hashes live in the
            # context/lattice metadata so a bank can span registered
            # multi-subject geometries without pretending their affines match.
            geometry_query_version_hash=canonical_digest("pfgr-lite-query-lattice-v1", prefix="pfgr-lite-geometry-version|"),
            # Only the scoped checkpoint/adaptation/backbone/BN identity is a
            # producer dependency.  Source Git/config SHA and traversal count
            # remain in SourceProvenance and must not invalidate a shared bank.
            medicalnet_provenance_hash=canonical_digest(
                {
                    "checkpoint_sha256": source.checkpoint_sha256,
                    "checkpoint_integrity_verified": source.checkpoint_integrity_verified,
                    "adaptation_digest": source.adaptation_digest,
                    "source_input_channels": source.source_input_channels,
                    "adapted_input_channels": source.adapted_input_channels,
                    "input_conv_adapted": source.input_conv_adapted,
                    "parameter_hash": source.parameter_hash,
                },
                prefix="pfgr-lite-medicalnet-producer-v1|",
            ),
            frozen_bn_hash=batchnorm_state_digest(prior.backbone),
            static_head_hash=module_state_digest(self.static_head),
            semantic_head_hash=module_state_digest(prior.semantic_head),
            point_refiner_hash=module_state_digest(self.frontend.point_refiner),
            spectral_projector_hash=module_state_digest(self.frontend.spectral_anchor_builder),
            state_initializer_hash=module_state_digest(self.static_head.final_projection),
            updater_hash=module_state_digest(self.updater),
            decoder_hash=module_state_digest(self.decoder),
            writer_hash=canonical_digest("compact-writeback-4mm-v1"),
            candidate_geometry_hash=canonical_digest(
                {
                    "candidate_count": self.config.candidate_count,
                    "num_points": self.config.num_points,
                    "point_candidate_multiplier": self.frontend_config.point_candidate_multiplier,
                    "directional_offsets_mm": tuple(self.frontend_config.directional_offsets_mm),
                    "support_radius_mm": self.config.support_radius_mm,
                    "max_displacement_mm": self.config.max_displacement_mm,
                    "algorithm_version": "point-candidate-geometry-v1",
                },
                prefix="pfgr-lite-candidate-geometry-v1|",
            ),
            label_definition_hash=canonical_digest(
                {
                    "definition": self.config.teacher.label_definition,
                    "rho": self.config.teacher.rho,
                    "epsilon": self.config.teacher.epsilon,
                    "mask_definition": self.config.teacher.mask_definition,
                    "global_mask_denominator": "sum(mask)>0_fixed_subject_v1",
                },
                prefix="pfgr-lite-label-definition-v1|",
            ),
            source_version="pfgr-lite-v1",
            component_versions=(
                ("geometry", geometries.version),
                ("static", self.config.static.variant),
                ("semantic", "medicalnet-resnet10-semantic-1x1-v1"),
                ("updater", "update-net-270-128-96-v1"),
                ("decoder", "implicit-decoder-96-64-32-1-v1"),
            ),
        )
        return ProducerDependencies(
            compatibility=compatibility,
            source_provenance=source,
            static_architecture=self.config.static.variant,
            config_version=self.config.schema_version,
        )

    @staticmethod
    def _context_id(x: Tensor, mask: Tensor | None, geometry: Any, producer: ProducerDependencies) -> str:
        payload: dict[str, object] = {
            "observation": tensor_digest(x, name="observations"),
            "geometry": geometry.voxel_to_ras_mm,
            "shape": geometry.shape_dhw,
            "producer": producer.compatibility_hash,
        }
        if mask is not None:
            payload["mask"] = tensor_digest(mask, name="observation_mask")
        return canonical_digest(payload, prefix="pfgr-lite-context-v1|")

    def encode_observations(
        self,
        x: Tensor,
        brain_mask: Tensor | None,
        geometry: Any,
    ) -> ObservationContext:
        """Build one target-free context and graph-preserving static Z0."""

        self._validate_observations(x)
        if self.config.numeric_mode == "fp32" and x.dtype != torch.float32:
            raise TypeError("PFGR production observations must be FP32")
        if self.config.numeric_mode == "fp64_test" and x.dtype != torch.float64:
            raise TypeError("PFGR FP64 test mode requires FP64 observations")
        geometry = _geometry_as_volume(x, geometry)
        mask = self._validate_mask(brain_mask, x)
        if x.shape[0] != 1:
            raise ValueError("PFGR-Lite W1 supports subject batch B=1; process subjects serially")
        # The private legacy seam performs exactly one MedicalNet traversal and
        # returns the local feature bundle while all downstream frontend maps
        # are still graph-connected.  It never accepts a target.
        traversal_count = [0]
        backbone = self.frontend.semantic_prior.backbone
        conv1 = getattr(backbone, "conv1", None)
        if not isinstance(conv1, nn.Module):
            raise RuntimeError("PFGR requires an instrumentable MedicalNet conv1 traversal")

        def _count_traversal(_module: nn.Module, _inputs: tuple[Tensor, ...], _output: Tensor) -> None:
            traversal_count[0] += 1

        handle = conv1.register_forward_hook(_count_traversal)
        try:
            output, descriptors, selected_geometry, features = self.frontend._forward_frontend_with_gate_b_context_and_features(
                x,
                mask,
                geometry.spacing_xyz_mm,
                voxel_to_ras_mm=geometry.voxel_to_ras_mm,
            )
        finally:
            handle.remove()
        if traversal_count[0] != 1:
            raise RuntimeError(f"PFGR requires exactly one MedicalNet traversal; observed {traversal_count[0]}")
        geometries = derive_multiscale_feature_geometry(self.frontend.semantic_prior.backbone, geometry, features)
        static_feature_geometry = derive_feature_grid_geometry(
            self.frontend.semantic_prior.backbone,
            geometry,
            tap="conv1_pre_maxpool",
            observed_shape_dhw=features.shallow.shape[-3:],
        )
        selected_lattice = geometries.shallow if self.frontend_config.spectral_tap == "conv1_pre_maxpool" else geometries.layer1
        initial_planes = self.static_head(
            features,
            x,
            output.base_planes,
            geometries,
            selected_lattice=selected_lattice,
        )
        q_bar = descriptors.reliability_weighted_mean(output.reliability)
        producer = self._producer_dependencies(geometries=geometries, traversal_count=traversal_count[0])
        context_id = self._context_id(x, mask, geometry, producer)
        owned_mask = mask.clone()
        context = ObservationContext(
            context_id=context_id,
            frontend=output,
            q_bar=q_bar,
            feature_geometry=static_feature_geometry,
            initial_planes=initial_planes,
            producer=producer,
            observation_mask=owned_mask,
            mask_provenance="caller_observation_only" if brain_mask is not None else "all_voxels_observation_default",
        )
        # Explicitly release the transient 64/512-channel bundle.  Context and
        # state retain only frontend descriptors and the owned graph-connected
        # Z0 planes.
        del features, geometries, descriptors, selected_geometry
        return context

    def initialize_state(self, context: ObservationContext, *, role: str = "training_behavior") -> PFGRState:
        """Clone Z0 without detach so S0/S1 gradients remain connected."""

        if not isinstance(context, ObservationContext):
            raise TypeError("context must be ObservationContext")
        context.validate_integrity()
        self._validate_context_producer_current(context)
        planes = clone_dynamic_planes(context.initial_planes)
        return PFGRState(
            planes=planes,
            context_id=context.context_id,
            state_version=0,
            producer=context.producer.compatibility,
            role=role,  # type: ignore[arg-type]
        )

    def _validate_context_producer_current(self, context: ObservationContext) -> None:
        """Reject using a context after any producer module was optimized."""

        compatibility = context.producer.compatibility
        expected_modules = (
            ("static_head_hash", module_state_digest(self.static_head)),
            ("semantic_head_hash", module_state_digest(self.frontend.semantic_prior.semantic_head)),
            ("point_refiner_hash", module_state_digest(self.frontend.point_refiner)),
            ("spectral_projector_hash", module_state_digest(self.frontend.spectral_anchor_builder)),
            ("state_initializer_hash", module_state_digest(self.static_head.final_projection)),
            ("updater_hash", module_state_digest(self.updater)),
            ("decoder_hash", module_state_digest(self.decoder)),
            ("frozen_bn_hash", batchnorm_state_digest(self.frontend.semantic_prior.backbone)),
        )
        for name, actual in expected_modules:
            if getattr(compatibility, name) != actual:
                raise ValueError(f"context producer is stale after producer change: {name}")
        source_parameter_hash = context.producer.source_provenance.parameter_hash
        if source_parameter_hash is not None and source_parameter_hash != module_parameter_digest(self.frontend.semantic_prior.backbone):
            raise ValueError("context producer is stale after MedicalNet parameter change")
        expected_normalization = canonical_digest(
            self.config.observation_normalization,
            prefix="pfgr-lite-observation-normalization-v1|",
        )
        if compatibility.observation_normalization_hash != expected_normalization:
            raise ValueError("context producer normalization identity is stale")

    def set_query_lattice_factory(self, factory: QueryLatticeFactory | None) -> None:
        """Inject W2's canonical lattice builder (no legacy fallback)."""

        if factory is not None and not hasattr(factory, "build") and not callable(factory):
            raise TypeError("query_lattice_factory must expose build(...) or be callable")
        self._query_lattice_factory = factory

    # Alias used by the W2 handoff.
    set_query_lattice_builder = set_query_lattice_factory

    def decode_final(self, state: PFGRState, context: ObservationContext, *, chunk_size: int) -> Tensor:
        """Decode only final Z through the injected canonical W2 lattice.

        W2's required callable is ``factory.build(output_geometry,
        feature_geometry, query_dtype, build_chunk_size)`` and the returned
        lattice must implement ``query(state.planes, voxel_ids_dhw,
        chunk_size=...) -> [Q,96]`` (or a batched ``[B,Q,96]`` equivalent).
        This method intentionally refuses the legacy RAS/grid query.
        """

        if not isinstance(state, PFGRState) or not isinstance(context, ObservationContext):
            raise TypeError("state and context must be PFGRState/ObservationContext")
        if state.context_id != context.context_id:
            raise ValueError("state/context IDs do not match")
        context.validate_integrity()
        state.validate_integrity()
        self._validate_context_producer_current(context)
        if state.producer is None:
            raise ValueError("decode_final requires a producer-bound PFGRState")
        if state.producer.digest != context.producer.compatibility_hash:
            raise ValueError("state producer is incompatible with context")
        if state.planes.xy.dtype != context.initial_planes.xy.dtype or state.planes.xy.device != context.initial_planes.xy.device:
            raise ValueError("state and context planes must share dtype and device")
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        factory = self._query_lattice_factory
        if factory is None:
            raise RuntimeError("PFGR decode_final requires W2 canonical PFGRQueryLattice injection; legacy query is not PFGR final semantics")
        build_kwargs = {
            "output_geometry": context.geometry,
            "feature_geometry": context.feature_geometry,
            "query_dtype": state.planes.xy.dtype,
            "build_chunk_size": self.config.build_chunk_size,
        }
        lattice = factory.build(**build_kwargs) if hasattr(factory, "build") else factory(**build_kwargs)
        if not hasattr(lattice, "query"):
            raise TypeError("injected PFGR query lattice must expose query(state, voxel_ids_dhw, chunk_size=...)")
        depth, height, width = context.geometry.shape_dhw
        voxel_count = depth * height * width
        batch = state.planes.xy.shape[0]
        output_chunks: list[Tensor] = []
        # Configured decode_chunk_size is a hard resource bound; callers may
        # request a smaller operational chunk but cannot silently exceed it.
        effective_chunk = min(chunk_size, self.config.decode_chunk_size)
        for start in range(0, voxel_count, effective_chunk):
            stop = min(start + effective_chunk, voxel_count)
            flat = torch.arange(start, stop, dtype=torch.long, device=state.planes.xy.device)
            area = height * width
            d = torch.div(flat, area, rounding_mode="floor")
            hw = flat - d * area
            h = torch.div(hw, width, rounding_mode="floor")
            w = hw - h * width
            # The canonical lattice accepts integer DHW centre IDs.  Keep its
            # required integer contract even though the state is floating.
            voxel_ids = torch.stack((d, h, w), dim=-1).to(dtype=torch.long)
            queried = lattice.query(state.planes, voxel_ids, chunk_size=effective_chunk)
            if not isinstance(queried, Tensor) or queried.shape[-1] != 96:
                raise ValueError("canonical PFGR lattice query must return [...,96]")
            if queried.ndim == 2:
                queried = queried.unsqueeze(0).expand(batch, -1, -1)
            if queried.ndim != 3 or queried.shape[:2] != (batch, stop - start):
                raise ValueError("canonical PFGR lattice query batch/voxel shape mismatch")
            output_chunks.append(self.decoder.mlp(queried.reshape(-1, 96)).reshape(batch, stop - start, 1))
        return torch.cat(output_chunks, dim=1).reshape(batch, 1, depth, height, width)

    def forward(self, *args: object, **kwargs: object) -> Tensor:
        del args, kwargs
        raise NotImplementedError(
            "PFGR-Lite generic forward() is fail-closed; use encode_observations(), "
            "initialize_state(), and the explicit W2-wired decode_final() API."
        )


__all__ = ["PFGRLiteModel", "QueryLatticeFactory"]
