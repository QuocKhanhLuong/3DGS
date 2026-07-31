"""Manifest-bound byte decoding for legal sparse observations.

This module never accepts filesystem paths.  Callers must obtain ``payload``
bytes from an ``EpisodeLedger`` so access ordering remains owned by the T0.5
legality contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
from typing import Literal

import numpy as np
import torch

from ..contracts.observation import AvailabilityObservationMeta


@dataclass(frozen=True)
class DecoderConfig:
    """Configuration for a bounded NumPy ``.npy`` plane decoder."""

    dtype: Literal["float32", "float64"] = "float32"
    nonfinite_policy: Literal["mask", "reject"] = "mask"

    def __post_init__(self) -> None:
        if self.dtype not in ("float32", "float64"):
            raise ValueError("decoder dtype must be float32 or float64")
        if self.nonfinite_policy not in ("mask", "reject"):
            raise ValueError("nonfinite_policy must be mask or reject")

    @property
    def config_hash(self) -> str:
        payload = json.dumps(
            {"dtype": self.dtype, "format": "npy-v1", "nonfinite_policy": self.nonfinite_policy},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DecodedObservation:
    """One decoded physical plane; image and mask use ``[1, H, W]``."""

    observation_id: str
    patient_id: str
    modality_id: str
    image: torch.Tensor
    valid_mask: torch.Tensor
    metadata: AvailabilityObservationMeta
    payload_sha256: str
    decoder_config_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, AvailabilityObservationMeta):
            raise TypeError("metadata must be AvailabilityObservationMeta")
        if (
            self.observation_id != self.metadata.observation_id
            or self.patient_id != self.metadata.patient_id
            or self.modality_id != self.metadata.modality_id
        ):
            raise ValueError("decoded identifiers must match manifest metadata")
        expected = (1, *self.metadata.plane.shape_hw)
        if self.image.shape != expected or self.image.dtype not in (torch.float32, torch.float64):
            raise ValueError("decoded image must be float with shape [1, H, W]")
        if self.valid_mask.shape != expected or self.valid_mask.dtype is not torch.bool:
            raise ValueError("decoded valid_mask must be bool with shape [1, H, W]")
        if self.image.device != self.valid_mask.device:
            raise ValueError("decoded image and mask must share device")
        if not bool(self.valid_mask.any()):
            raise ValueError("decoded observation requires at least one valid pixel")
        if not bool(torch.isfinite(self.image).all()):
            raise ValueError("decoded image stores invalid pixels as finite zeros")
        for name, value in (
            ("payload_sha256", self.payload_sha256),
            ("decoder_config_hash", self.decoder_config_hash),
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 hex digest")


def decode_observation(
    payload: bytes,
    metadata: AvailabilityObservationMeta,
    *,
    config: DecoderConfig | None = None,
) -> DecodedObservation:
    """Decode legal ledger-returned bytes without accepting arbitrary paths."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes returned by a legal ledger")
    if not isinstance(metadata, AvailabilityObservationMeta):
        raise TypeError("metadata must be AvailabilityObservationMeta")
    config = config or DecoderConfig()
    try:
        array = np.load(BytesIO(payload), allow_pickle=False)
    except Exception as exc:
        raise ValueError("payload is not a valid non-pickled NumPy plane") from exc
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError("decoded payload must contain one rank-2 plane")
    if tuple(array.shape) != tuple(metadata.plane.shape_hw):
        raise ValueError("decoded payload shape disagrees with manifest plane")
    if array.dtype.kind not in "fiu":
        raise TypeError("decoded payload must use a real numeric dtype")
    target_dtype = np.float32 if config.dtype == "float32" else np.float64
    numeric = np.asarray(array, dtype=target_dtype)
    finite = np.isfinite(numeric)
    if config.nonfinite_policy == "reject" and not bool(finite.all()):
        raise ValueError("non-finite payload values are forbidden by decoder policy")
    if not bool(finite.any()):
        raise ValueError("decoded payload contains no finite pixels")
    clean = np.where(finite, numeric, 0.0).copy()
    image = torch.from_numpy(clean).unsqueeze(0)
    valid_mask = torch.from_numpy(finite.copy()).unsqueeze(0)
    return DecodedObservation(
        observation_id=metadata.observation_id,
        patient_id=metadata.patient_id,
        modality_id=metadata.modality_id,
        image=image,
        valid_mask=valid_mask,
        metadata=metadata,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        decoder_config_hash=config.config_hash,
    )
