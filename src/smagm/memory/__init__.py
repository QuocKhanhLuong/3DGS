"""Dual-bank seed memory and bounded static propagation."""

from .appearance import validate_appearance_slots
from .contracts import GaussianMemory, GaussianMemoryBank, PrimitiveKind, PrimitiveObservability, gaussian_memory_hash
from .index import query_memory_radius
from .initialize import SeedMemoryConfig, initialize_seed_memory
from .observability import initial_observability, propagated_observability
from .propagation import PropagationConfig, PropagationTransaction, propagate_memory

__all__ = [
    "GaussianMemory", "GaussianMemoryBank", "PrimitiveKind", "PrimitiveObservability",
    "PropagationConfig", "PropagationTransaction", "SeedMemoryConfig",
    "gaussian_memory_hash", "initial_observability",
    "initialize_seed_memory", "propagated_observability", "query_memory_radius",
    "propagate_memory", "validate_appearance_slots",
]
