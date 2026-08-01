"""Fail-closed representation and propagation switches for static attribution.

The switch is deliberately declarative: callers construct only the modules in
``active_modules``.  This prevents an ablation from retaining an unused field,
anchor path, or patient-specific parameter opportunity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json


class RepresentationVariant(str, Enum):
    INTERPOLATION = "interpolation"
    FIXED_SUPPORT_GAUSSIAN = "fixed_support_gaussian"
    FREE_GAUSSIAN = "free_gaussian"
    DIRECT_ANCHOR_GAUSSIAN = "direct_anchor_gaussian"
    ANCHOR_FIELD = "anchor_field"
    GLOBAL_FIELD = "global_field"


_ALIASES = {
    "r0": RepresentationVariant.INTERPOLATION,
    "r1": RepresentationVariant.FIXED_SUPPORT_GAUSSIAN,
    "r2": RepresentationVariant.FREE_GAUSSIAN,
    "r3": RepresentationVariant.DIRECT_ANCHOR_GAUSSIAN,
    "r4": RepresentationVariant.ANCHOR_FIELD,
    "r5": RepresentationVariant.GLOBAL_FIELD,
}


@dataclass(frozen=True)
class RepresentationPlan:
    """Exact causal module inventory for one matched static representation."""

    variant: RepresentationVariant
    propagation_variant: str
    active_modules: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.propagation_variant not in ("p0", "p1"):
            raise ValueError("only authorized static propagation variants p0 and p1 are executable")
        if len(self.active_modules) != len(set(self.active_modules)):
            raise ValueError("representation module inventory must not contain duplicates")
        if "routing" in self.active_modules or "topology" in self.active_modules:
            raise ValueError("T4 routing and adaptive topology are outside the static pipeline")

    @property
    def plan_hash(self) -> str:
        payload = {
            "active_modules": self.active_modules,
            "propagation_variant": self.propagation_variant,
            "variant": self.variant.value,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_representation_plan(
    variant: str | RepresentationVariant,
    *,
    propagation_variant: str = "p0",
) -> RepresentationPlan:
    """Resolve R0--R5 and P0/P1 without silently keeping removed modules."""

    if isinstance(variant, str):
        normalized = variant.lower()
        try:
            resolved = _ALIASES[normalized] if normalized in _ALIASES else RepresentationVariant(normalized)
        except ValueError as error:
            raise ValueError("unknown representation variant; expected R0--R5 or its canonical name") from error
    elif isinstance(variant, RepresentationVariant):
        resolved = variant
    else:
        raise TypeError("representation variant must be a string or RepresentationVariant")
    propagation = propagation_variant.lower()
    if propagation not in ("p0", "p1"):
        raise ValueError("only P0 and bounded fixed P1 are authorized")

    modules = {
        RepresentationVariant.INTERPOLATION: ("context_payloads", "physical_kernel_interpolator"),
        RepresentationVariant.FIXED_SUPPORT_GAUSSIAN: ("encoder", "fixed_supports", "fixed_gaussian_head"),
        RepresentationVariant.FREE_GAUSSIAN: ("context_payloads", "patient_free_gaussians"),
        RepresentationVariant.DIRECT_ANCHOR_GAUSSIAN: ("encoder", "physical_anchors", "seed_gaussians"),
        RepresentationVariant.ANCHOR_FIELD: ("encoder", "physical_anchors", "shared_local_field", "seed_gaussians"),
        RepresentationVariant.GLOBAL_FIELD: ("encoder", "physical_anchors", "global_coordinate_field", "seed_gaussians"),
    }[resolved]
    if propagation == "p1":
        if resolved is not RepresentationVariant.ANCHOR_FIELD:
            raise ValueError("P1 is matched only with R4 anchor_field in the authorized FULL method")
        modules = modules + ("bounded_fixed_propagation",)
    return RepresentationPlan(resolved, propagation, modules)


__all__ = [
    "RepresentationPlan",
    "RepresentationVariant",
    "resolve_representation_plan",
]
