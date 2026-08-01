from __future__ import annotations

import torch

import smagm  # noqa: F401  # package import applies the eager compatibility guard


def test_adam_is_constructible_and_updates_after_smagm_import() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam((parameter,), lr=0.1)
    loss = parameter.square().sum()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(parameter).all()
    assert float(parameter) < 1.0
