"""Declared registration metadata validation; no hidden registration model."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Sequence


@dataclass(frozen=True)
class RegistrationRecord:
    record_id: str
    source_observation_id: str
    target_observation_id: str
    source_to_target_ras: Sequence[Sequence[float]]
    confidence: float
    method_id: str
    used_manifest_observation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (self.record_id, self.source_observation_id, self.target_observation_id, self.method_id)
        ):
            raise ValueError("registration identifiers must be non-empty")
        matrix = tuple(tuple(float(value) for value in row) for row in self.source_to_target_ras)
        if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
            raise ValueError("source_to_target_ras must be a 4x4 matrix")
        if any(not math.isfinite(value) for row in matrix for value in row):
            raise ValueError("registration transform must be finite")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("registration confidence must lie in [0, 1]")
        used = tuple(sorted(self.used_manifest_observation_ids))
        if not used or any(not value for value in used):
            raise ValueError("registration must declare manifest observations used")
        object.__setattr__(self, "source_to_target_ras", matrix)
        object.__setattr__(self, "used_manifest_observation_ids", used)

    @property
    def record_hash(self) -> str:
        payload = {
            "confidence": self.confidence,
            "method_id": self.method_id,
            "record_id": self.record_id,
            "source_observation_id": self.source_observation_id,
            "source_to_target_ras": self.source_to_target_ras,
            "target_observation_id": self.target_observation_id,
            "used_manifest_observation_ids": self.used_manifest_observation_ids,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
