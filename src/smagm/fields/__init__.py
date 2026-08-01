"""Shared anchor-local structural field with explicit unsupported output."""

from .blend import blend_local_fields, compact_support_weights
from .contracts import FieldQueryBatch, StructuralFieldOutput
from .global_field import GlobalStructuralField, GlobalStructuralFieldConfig
from .local import SharedStructuralField, StructuralFieldConfig
from .query import build_field_queries, query_structural_field
from .regularization import field_gradient_diagnostics, overlap_consistency_loss

__all__ = [
    "FieldQueryBatch", "GlobalStructuralField", "GlobalStructuralFieldConfig",
    "SharedStructuralField", "StructuralFieldConfig", "StructuralFieldOutput",
    "blend_local_fields", "build_field_queries", "compact_support_weights",
    "field_gradient_diagnostics", "overlap_consistency_loss", "query_structural_field",
]
