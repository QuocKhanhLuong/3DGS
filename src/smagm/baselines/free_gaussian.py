"""Patient-specific free-Gaussian R2 baseline without anchors or fields."""

from __future__ import annotations

import hashlib

import torch
from torch import nn

from ..gaussians import GaussianBatch, RawGaussianParameters, gaussian_batch_from_raw


class FreeGaussianState(nn.Module):
    """Promote a deterministic context-only seed into free Gaussian parameters."""

    def __init__(self, seed: GaussianBatch) -> None:
        super().__init__()
        seed.validate()
        factor = seed.covariance_factor.detach().clone()
        diagonal = factor.diagonal(dim1=-2, dim2=-1)
        if not bool((diagonal > 0).all()):
            raise ValueError("free-Gaussian seed requires positive Cholesky diagonal")
        self.centers_ras_mm = nn.Parameter(seed.centers_ras_mm.detach().clone())
        self.log_factor_diagonal = nn.Parameter(diagonal.log())
        self.factor_off_diagonal = nn.Parameter(factor[:, (1, 2, 2), (0, 0, 1)])
        self.log_support_amplitude = nn.Parameter(seed.log_support_amplitude.detach().clone())
        self.appearance = nn.Parameter(seed.appearance.detach().clone())
        self.register_buffer("appearance_valid", seed.appearance_valid.detach().clone())
        self.primitive_ids = tuple(f"free:{value}" for value in (seed.primitive_id or tuple(map(str, range(seed.count)))))
        self.covariance_epsilon = seed.covariance_epsilon

    def forward(self) -> GaussianBatch:
        count = self.centers_ras_mm.shape[0]
        factor = self.centers_ras_mm.new_zeros((count, 3, 3))
        factor[:, (0, 1, 2), (0, 1, 2)] = self.log_factor_diagonal.exp()
        factor[:, (1, 2, 2), (0, 0, 1)] = self.factor_off_diagonal
        return gaussian_batch_from_raw(RawGaussianParameters(
            centers_ras_mm=self.centers_ras_mm,
            covariance_factor=factor,
            raw_log_support_amplitude=self.log_support_amplitude,
            appearance=self.appearance,
            appearance_valid=self.appearance_valid,
            patient_state_index=torch.zeros(count, dtype=torch.int64, device=factor.device),
            covariance_epsilon=self.covariance_epsilon,
            primitive_kind=("free_gaussian",) * count,
            primitive_id=self.primitive_ids,
        ))

    @property
    def state_hash(self) -> str:
        digest = hashlib.sha256(b"free-gaussian-state-v1")
        for name, value in self.state_dict().items():
            item = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(f"{item.dtype}:{tuple(item.shape)}".encode("utf-8"))
            digest.update(item.numpy().tobytes())
        return digest.hexdigest()


__all__ = ["FreeGaussianState"]
