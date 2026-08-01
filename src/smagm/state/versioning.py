"""Canonical immutable patient-state identity without target values."""

from __future__ import annotations

import hashlib
import json


def state_version_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
