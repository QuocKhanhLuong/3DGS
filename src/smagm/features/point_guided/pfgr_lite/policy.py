"""One effective PFGR-Lite policy loader and deterministic selector."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping
from typing import Literal

import torch
from torch import Tensor, nn

from .action_proposal import ACTION_GENERATOR_VERSION
from .config import PFGRLiteConfig
from .calibration import calibration_evidence
from .provenance import ProducerCompatibility, ValueFitIdentity, canonical_digest, module_state_digest
from .types import ActionProposalBatch, Decision, GainCalibration, ProducerDependencies


POLICY_EFFECTIVE_SCHEMA = "pfgr-lite-effective-policy-v1"
POLICY_CAPABILITIES = ("static", "forced_diagnostic", "adaptive")
POLICY_MODES = (
    "adaptive",
    "forced_diagnostic",
    "random",
    "fixed_learned",
    "parallel_topk",
    "static",
    "noop",
)
_BUDGETS = (0, 1, 2, 4)


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.lower() in {"unknown", "unset", "none", "null"}:
        raise ValueError(f"{name} must be a complete non-sentinel string")
    return value


_GAIN_SCALE_KEYS = {
    "schema_version",
    "scale",
    "quantile",
    "method",
    "floor",
    "floor_applied",
    "training_role",
    "training_row_hash",
    "training_row_count",
}


def _normalise_gain_scale(value: Mapping[str, object], *, expected_hash: str, expected_scale: float) -> dict[str, object]:
    """Validate the complete W3 GainScale envelope, including its number."""

    if not isinstance(value, Mapping):
        raise TypeError("gain_scale_provenance must be a mapping")
    if set(value) != _GAIN_SCALE_KEYS | {"digest"}:
        raise ValueError("gain_scale_provenance keys are incomplete or unknown")
    payload = {key: value[key] for key in _GAIN_SCALE_KEYS}
    if payload["schema_version"] != "point-guided-pfgr-lite-gain-scale-v1":
        raise ValueError("unknown gain-scale provenance schema")
    digest = value["digest"]
    if not isinstance(digest, str) or not digest:
        raise ValueError("gain_scale_provenance digest must be complete")
    if canonical_digest(payload, prefix="pfgr-lite-gain-scale-v1|") != digest:
        raise ValueError("gain_scale_provenance digest mismatch")
    if digest != expected_hash:
        raise ValueError("gain_scale_provenance hash does not match calibration/value identity")
    scale = float(payload["scale"])
    if not math.isfinite(scale) or scale <= 0.0 or scale != float(expected_scale):
        raise ValueError("gain_scale_provenance numeric scale does not match policy gain_scale")
    return dict(value)


@dataclass(frozen=True)
class EffectivePolicy:
    """Resolved mode used identically by training, evaluation and inference."""

    mode: Literal[
        "adaptive",
        "forced_diagnostic",
        "random",
        "fixed_learned",
        "parallel_topk",
        "static",
        "noop",
    ]
    budget: int = 4
    revisit: Literal["allow"] = "allow"
    tie_break: Literal["lowest_point_id"] = "lowest_point_id"
    gain_units: Literal["raw_signed_loss"] = "raw_signed_loss"
    quality_margin: float = 0.0
    compute_cost: float = 0.0
    producer_compatibility_hash: str = ""
    calibration: GainCalibration | None = None
    calibration_hash: str = ""
    value_input_variant: int = 366
    value_fit_identity: ValueFitIdentity | None = None
    gain_scale: float = 1.0
    gain_scale_hash: str = ""
    gain_scale_provenance: Mapping[str, object] | None = None
    candidate_chunk_size: int = 1024
    random_seed: int = 0
    engineering_only: bool = False
    capability: Literal["static", "forced_diagnostic", "adaptive"] = "static"
    proposal_generator_version: str = "pfgr-lite-action-generator-v1"
    policy_hash: str = ""
    schema_version: str = POLICY_EFFECTIVE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_EFFECTIVE_SCHEMA:
            raise ValueError("unknown effective policy schema")
        if self.mode not in POLICY_MODES:
            raise ValueError(f"unknown PFGR policy mode: {self.mode!r}")
        if self.budget not in _BUDGETS:
            raise ValueError("PFGR budget must be one of 0, 1, 2, or 4")
        if self.revisit != "allow":
            raise ValueError("PFGR revisit policy is locked to allow")
        if self.tie_break != "lowest_point_id":
            raise ValueError("PFGR tie-break is locked to lowest_point_id")
        if self.gain_units != "raw_signed_loss":
            raise ValueError("PFGR gain units are locked to raw_signed_loss")
        if self.capability not in POLICY_CAPABILITIES:
            raise ValueError("unknown effective policy capability")
        if self.mode == "adaptive" and self.capability != "adaptive":
            raise ValueError("adaptive mode requires adaptive capability")
        if self.mode != "adaptive" and self.capability == "adaptive":
            raise ValueError("adaptive capability requires adaptive mode")
        for name in ("quality_margin", "compute_cost"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.value_input_variant not in (126, 222, 270, 366):
            raise ValueError("value_input_variant must be one of 126, 222, 270, 366")
        if self.value_fit_identity is not None:
            if not isinstance(self.value_fit_identity, ValueFitIdentity):
                raise TypeError("value_fit_identity must be ValueFitIdentity or None")
            if self.value_fit_identity.input_variant != self.value_input_variant:
                raise ValueError("ValueFitIdentity input variant does not match effective policy")
            if self.gain_scale_hash and self.value_fit_identity.gain_scale_hash != self.gain_scale_hash:
                raise ValueError("ValueFitIdentity gain-scale identity does not match effective policy")
        if not math.isfinite(float(self.gain_scale)) or self.gain_scale <= 0.0:
            raise ValueError("gain_scale must be finite and positive")
        if not isinstance(self.engineering_only, bool):
            raise TypeError("engineering_only must be bool")
        if self.gain_scale_provenance is not None:
            if not self.gain_scale_hash:
                raise ValueError("gain_scale_provenance requires a complete gain_scale_hash")
            object.__setattr__(self, "gain_scale_provenance", _normalise_gain_scale(self.gain_scale_provenance, expected_hash=self.gain_scale_hash, expected_scale=self.gain_scale))
        if not isinstance(self.candidate_chunk_size, int) or isinstance(self.candidate_chunk_size, bool) or self.candidate_chunk_size <= 0:
            raise ValueError("candidate_chunk_size must be a positive integer")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise ValueError("random_seed must be an integer")
        _nonempty(self.producer_compatibility_hash, "producer_compatibility_hash")
        if self.calibration is not None and not isinstance(self.calibration, GainCalibration):
            raise TypeError("calibration must be GainCalibration or None")
        if self.calibration is not None:
            expected = canonical_digest(self.calibration, prefix="pfgr-lite-calibration-v1|")
            if self.calibration_hash and self.calibration_hash != expected:
                raise ValueError("calibration_hash does not match calibration")
            object.__setattr__(self, "calibration_hash", expected)
        elif self.calibration_hash:
            _nonempty(self.calibration_hash, "calibration_hash")
        _nonempty(self.proposal_generator_version, "proposal_generator_version")
        if self.mode == "adaptive":
            if self.calibration is None or self.calibration.capability != "adaptive":
                raise ValueError("adaptive policy requires an adaptive calibration")
            evidence = calibration_evidence(self.calibration)
            if evidence is None or (not evidence.deployment_ready and not self.engineering_only):
                raise ValueError("adaptive policy requires complete calibration evidence")
            if evidence.producer_compatibility_hash != self.producer_compatibility_hash:
                raise ValueError("calibration evidence producer identity does not match policy")
            if evidence.value_input_variant != self.value_input_variant:
                raise ValueError("calibration V identity does not match the effective policy")
            if self.value_fit_identity is None or evidence.value_fit_identity_hash != self.value_fit_identity.digest:
                raise ValueError("adaptive policy requires the exact ValueFitIdentity bound by calibration")
            _nonempty(self.gain_scale_hash, "gain_scale_hash")
            if evidence.gain_scale_hash != self.gain_scale_hash:
                raise ValueError("calibration gain-scale identity does not match policy")
            if self.gain_scale_provenance is None and not self.engineering_only:
                raise ValueError("production adaptive policy requires complete GainScale provenance")
        expected_hash = canonical_digest(
            {
                "schema_version": self.schema_version,
                "mode": self.mode,
                "budget": self.budget,
                "revisit": self.revisit,
                "tie_break": self.tie_break,
                "gain_units": self.gain_units,
                "quality_margin": self.quality_margin,
                "compute_cost": self.compute_cost,
                "producer_compatibility_hash": self.producer_compatibility_hash,
                "calibration_hash": self.calibration_hash,
                "value_input_variant": self.value_input_variant,
                "value_fit_identity_hash": "" if self.value_fit_identity is None else self.value_fit_identity.digest,
                "gain_scale": self.gain_scale,
                "gain_scale_hash": self.gain_scale_hash,
                "gain_scale_provenance": None if self.gain_scale_provenance is None else dict(self.gain_scale_provenance),
                "candidate_chunk_size": self.candidate_chunk_size,
                "random_seed": self.random_seed,
                "engineering_only": self.engineering_only,
                "capability": self.capability,
                "proposal_generator_version": self.proposal_generator_version,
            },
            prefix="pfgr-lite-effective-policy-v1|",
        )
        if self.policy_hash and self.policy_hash != expected_hash:
            raise ValueError("policy_hash does not match effective policy fields")
        object.__setattr__(self, "policy_hash", expected_hash)

    def as_dict(self) -> dict[str, object]:
        """Canonical metadata for strict checkpoint/policy provenance."""

        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "budget": self.budget,
            "revisit": self.revisit,
            "tie_break": self.tie_break,
            "gain_units": self.gain_units,
            "quality_margin": self.quality_margin,
            "compute_cost": self.compute_cost,
            "producer_compatibility_hash": self.producer_compatibility_hash,
            "calibration_hash": self.calibration_hash,
            "value_input_variant": self.value_input_variant,
            "value_fit_identity": None if self.value_fit_identity is None else {
                "schema_version": self.value_fit_identity.schema_version,
                "input_variant": self.value_fit_identity.input_variant,
                "architecture_hash": self.value_fit_identity.architecture_hash,
                "weights_hash": self.value_fit_identity.weights_hash,
                "fit_config_hash": self.value_fit_identity.fit_config_hash,
                "bank_manifest_hash": self.value_fit_identity.bank_manifest_hash,
                "gain_scale_hash": self.value_fit_identity.gain_scale_hash,
            },
            "gain_scale": self.gain_scale,
            "gain_scale_hash": self.gain_scale_hash,
            "gain_scale_provenance": None if self.gain_scale_provenance is None else dict(self.gain_scale_provenance),
            "candidate_chunk_size": self.candidate_chunk_size,
            "random_seed": self.random_seed,
            "engineering_only": self.engineering_only,
            "capability": self.capability,
            "proposal_generator_version": self.proposal_generator_version,
            "policy_hash": self.policy_hash,
        }


def _dependency_hash(dependencies: ProducerDependencies | ProducerCompatibility) -> str:
    if isinstance(dependencies, ProducerDependencies):
        return dependencies.compatibility_hash
    if isinstance(dependencies, ProducerCompatibility):
        return dependencies.digest
    raise TypeError("dependencies must be ProducerDependencies or ProducerCompatibility")


def load_effective_policy(
    config: PFGRLiteConfig | Mapping[str, object],
    calibration: GainCalibration | None,
    *,
    dependencies: ProducerDependencies | ProducerCompatibility,
    capability: Literal["static", "forced_diagnostic", "adaptive"],
    budget: int = 4,
    candidate_chunk_size: int | None = None,
    random_seed: int = 0,
    value_input_variant: int = 366,
    value_fit_identity_hash: str | None = None,
    role_manifest_hash: str | None = None,
    value_fit_identity: ValueFitIdentity | None = None,
    gain_scale: float | None = None,
    gain_scale_hash: str | None = None,
    gain_scale_provenance: Mapping[str, object] | None = None,
) -> EffectivePolicy:
    """Resolve one policy instance; callers must not reconstruct defaults."""

    if isinstance(config, Mapping):
        config = PFGRLiteConfig.from_dict(config)
    if not isinstance(config, PFGRLiteConfig):
        raise TypeError("config must be PFGRLiteConfig or its strict mapping")
    if capability not in POLICY_CAPABILITIES:
        raise ValueError("unknown policy capability")
    producer_hash = _dependency_hash(dependencies)
    mode = config.policy.mode
    if mode == "adaptive":
        if capability != "adaptive":
            raise ValueError("adaptive policy can only be loaded with adaptive capability")
        if calibration is None or calibration.capability != "adaptive":
            raise ValueError("adaptive policy requires a complete adaptive calibration")
        if calibration.producer_compatibility_hash != producer_hash:
            raise ValueError("calibration producer identity does not match dependencies")
        evidence = calibration_evidence(calibration)
        if evidence is None or (not evidence.deployment_ready and not config.engineering_only):
            raise ValueError("adaptive policy requires complete calibration evidence")
        if evidence.producer_compatibility_hash != producer_hash:
            raise ValueError("calibration evidence producer identity does not match dependencies")
        if evidence.value_input_variant != value_input_variant:
            raise ValueError("calibration V identity does not match the requested descriptor variant")
        compatibility = dependencies.compatibility if isinstance(dependencies, ProducerDependencies) else dependencies
        if evidence.writer_hash != compatibility.writer_hash:
            raise ValueError("calibration writer identity does not match current producer")
        if evidence.query_hash != compatibility.geometry_query_version_hash:
            raise ValueError("calibration query identity does not match current producer")
        if evidence.proposal_generator_hash != ACTION_GENERATOR_VERSION:
            raise ValueError("calibration proposal-generator identity is stale")
        if value_fit_identity_hash is not None and evidence.value_fit_identity_hash != value_fit_identity_hash:
            raise ValueError("calibration value-fit identity does not match the requested V")
        if value_fit_identity is None:
            raise ValueError("adaptive policy requires the exact ValueFitIdentity")
        if not isinstance(value_fit_identity, ValueFitIdentity):
            raise TypeError("value_fit_identity must be ValueFitIdentity")
        if value_fit_identity.input_variant != value_input_variant or value_fit_identity.digest != evidence.value_fit_identity_hash:
            raise ValueError("calibration does not match the exact ValueFitIdentity")
        if gain_scale is None or not math.isfinite(float(gain_scale)) or float(gain_scale) <= 0.0:
            raise ValueError("adaptive policy requires a positive fixed gain_scale")
        if gain_scale_hash is None or gain_scale_hash != evidence.gain_scale_hash:
            raise ValueError("adaptive policy requires the exact fixed gain-scale identity")
        if gain_scale_hash is not None and evidence.gain_scale_hash != gain_scale_hash:
            raise ValueError("calibration gain-scale identity is stale")
        if gain_scale_provenance is None:
            if not config.engineering_only:
                raise ValueError("production adaptive policy requires complete GainScale provenance")
        else:
            if gain_scale is None:
                raise ValueError("gain_scale must accompany gain_scale_provenance")
            _normalise_gain_scale(gain_scale_provenance, expected_hash=evidence.gain_scale_hash, expected_scale=float(gain_scale))
        if role_manifest_hash is None:
            raise ValueError("adaptive policy requires the exact TrainingRoleManifest identity")
        if evidence.role_manifest.digest != role_manifest_hash:
            raise ValueError("calibration role-manifest identity is stale")
        if evidence.policy_hash != canonical_digest(config.policy, prefix="pfgr-lite-policy-config-v1|"):
            raise ValueError("calibration policy identity does not match the effective config")
    elif capability == "adaptive":
        raise ValueError("non-adaptive policy cannot be loaded with adaptive capability")
    if mode in ("fixed_learned", "parallel_topk") and not config.engineering_only:
        if value_fit_identity is None or gain_scale is None or gain_scale_hash is None or gain_scale_provenance is None:
            raise ValueError("production learned diagnostic modes require exact V and GainScale provenance")
    if mode in ("fixed_learned", "parallel_topk") and value_fit_identity is not None:
        if value_fit_identity.input_variant != value_input_variant:
            raise ValueError("learned policy V descriptor variant does not match ValueFitIdentity")
        if gain_scale_hash != value_fit_identity.gain_scale_hash:
            raise ValueError("learned policy gain scale does not match ValueFitIdentity")
        if gain_scale_provenance is not None:
            if gain_scale is None:
                raise ValueError("gain_scale must accompany gain_scale_provenance")
            _normalise_gain_scale(gain_scale_provenance, expected_hash=value_fit_identity.gain_scale_hash, expected_scale=float(gain_scale))
    chunk = config.build_chunk_size if candidate_chunk_size is None else candidate_chunk_size
    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk <= 0:
        raise ValueError("candidate_chunk_size must be a positive integer")
    if budget not in _BUDGETS:
        raise ValueError("budget must be one of 0, 1, 2, or 4")
    return EffectivePolicy(
        mode=mode,
        budget=budget,
        revisit=config.policy.revisit,
        tie_break=config.policy.tie_break,
        gain_units=config.policy.gain_units,
        quality_margin=config.policy.quality_margin,
        compute_cost=config.policy.compute_cost,
        producer_compatibility_hash=producer_hash,
        calibration=calibration,
        value_fit_identity=value_fit_identity,
        gain_scale=1.0 if gain_scale is None else float(gain_scale),
        gain_scale_hash="" if gain_scale_hash is None else gain_scale_hash,
        gain_scale_provenance=gain_scale_provenance,
        value_input_variant=value_input_variant,
        candidate_chunk_size=chunk,
        random_seed=random_seed,
        engineering_only=config.engineering_only,
        capability=capability,
    )


def _normalise_scores(scores: Tensor, proposals: ActionProposalBatch) -> Tensor:
    if not isinstance(scores, Tensor):
        raise TypeError("predicted_raw_gain must be a torch.Tensor")
    expected = proposals.point_ids.shape
    if scores.ndim == 1:
        scores = scores.unsqueeze(0)
    if scores.shape != expected:
        raise ValueError(f"predicted_raw_gain must have shape {tuple(expected)}")
    if scores.device != proposals.delta.device or not scores.is_floating_point():
        raise ValueError("predicted_raw_gain must be floating and share proposal device")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("nonfinite predicted raw gains are an explicit numerical failure")
    return scores


def _decision_stop(
    policy: EffectivePolicy,
    *,
    reason: Literal["budget", "low_gain", "no_legal_action"],
    proposal_digest: str = "",
    raw: float = 0.0,
    calibrated: float = 0.0,
    conservative: float = 0.0,
    step: int,
) -> Decision:
    return Decision(
        selected_point_id=-1,
        proposal_digest=proposal_digest,
        action_digest="",
        active=False,
        raw_value=raw,
        calibrated_value=calibrated,
        conservative_value=conservative,
        allowance=0.0 if policy.calibration is None else policy.calibration.allowance,
        quality_margin=policy.quality_margin,
        compute_cost=policy.compute_cost,
        policy_hash=policy.policy_hash,
        stop_code=reason,
        step=step,
    )


def select_or_stop(
    proposals: ActionProposalBatch | None,
    predicted_raw_gain: Tensor | None,
    effective_policy: EffectivePolicy,
    *,
    step: int,
) -> Decision:
    """Select a stored action or stop using exact signed-gain semantics."""

    if not isinstance(effective_policy, EffectivePolicy):
        raise TypeError("effective_policy must be EffectivePolicy")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if step >= effective_policy.budget or effective_policy.mode in ("noop", "static"):
        return _decision_stop(effective_policy, reason="budget", step=step)
    if proposals is None:
        raise ValueError("a nonzero-budget policy requires proposals")
    if not isinstance(proposals, ActionProposalBatch):
        raise TypeError("proposals must be ActionProposalBatch")
    proposals.validate_integrity()
    if proposals.point_ids.shape[0] != 1:
        raise ValueError("selector currently requires one subject; process batches serially")
    if proposals.producer_compatibility_hash != effective_policy.producer_compatibility_hash:
        raise ValueError("proposal producer does not match effective policy")
    if effective_policy.mode == "random" and predicted_raw_gain is None:
        scores = torch.zeros(
            proposals.point_ids.shape,
            dtype=proposals.delta.dtype,
            device=proposals.delta.device,
        )
    else:
        if predicted_raw_gain is None:
            raise ValueError("this policy mode requires explicit V scores")
        scores = _normalise_scores(predicted_raw_gain, proposals)
    legal = proposals.legal.to(dtype=torch.bool) & (proposals.delta.abs().amax(dim=-1) > 0.0)
    if not bool(legal.any()):
        return _decision_stop(
            effective_policy,
            reason="no_legal_action",
            proposal_digest=proposals.proposal_digest,
            step=step,
        )
    calibration = effective_policy.calibration
    # V emits a fixed-scale normalized score; the stored gain scale is part
    # of the V fit identity and is applied before the affine calibration.
    raw_scores = scores * float(effective_policy.gain_scale)
    if calibration is None:
        calibrated = raw_scores
        allowance = 0.0
    else:
        calibrated = float(calibration.a) * raw_scores + float(calibration.b)
        allowance = float(calibration.allowance)
    conservative = calibrated - allowance - effective_policy.quality_margin - effective_policy.compute_cost
    if not bool(torch.isfinite(calibrated).all()) or not bool(torch.isfinite(conservative).all()):
        raise ValueError("nonfinite calibrated or conservative scores are an explicit numerical failure")
    legal_indices = [index for index in range(proposals.point_ids.shape[1]) if bool(legal[0, index])]
    if effective_policy.mode == "random":
        generator = torch.Generator(device=proposals.point_ids.device)
        generator.manual_seed(effective_policy.random_seed + step)
        choice = int(torch.randint(len(legal_indices), (1,), generator=generator, device=proposals.point_ids.device).item())
        selected_index = legal_indices[choice]
    else:
        # Python's stable sort makes the lowest point ID the exact tie-break;
        # no tolerance is added to the locked zero threshold.
        selected_index = min(
            legal_indices,
            key=lambda index: (-float(conservative[0, index].item()), int(proposals.point_ids[0, index].item())),
        )
    selected_point_id = int(proposals.point_ids[0, selected_index].item())
    selected_raw = float(raw_scores[0, selected_index].item())
    selected_calibrated = float(calibrated[0, selected_index].item())
    selected_conservative = float(conservative[0, selected_index].item())
    if effective_policy.mode == "adaptive" and selected_conservative <= 0.0:
        return _decision_stop(
            effective_policy,
            reason="low_gain",
            proposal_digest=proposals.proposal_digest,
            raw=selected_raw,
            calibrated=selected_calibrated,
            conservative=selected_conservative,
            step=step,
        )
    action = proposals.row(0, selected_index)
    return Decision(
        selected_point_id=selected_point_id,
        proposal_digest=proposals.proposal_digest,
        action_digest=action.action_digest,
        active=True,
        raw_value=selected_raw,
        calibrated_value=selected_calibrated,
        conservative_value=selected_conservative,
        allowance=allowance,
        quality_margin=effective_policy.quality_margin,
        compute_cost=effective_policy.compute_cost,
        policy_hash=effective_policy.policy_hash,
        stop_code="continue",
        step=step,
    )


__all__ = [
    "EffectivePolicy",
    "POLICY_CAPABILITIES",
    "POLICY_EFFECTIVE_SCHEMA",
    "POLICY_MODES",
    "load_effective_policy",
    "select_or_stop",
]
