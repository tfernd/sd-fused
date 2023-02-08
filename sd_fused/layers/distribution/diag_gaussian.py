from __future__ import annotations

import torch
from torch import Tensor

from ..base import Module


class DiagonalGaussianDistribution(Module):
    def __init__(
        self,
        mean: Tensor,
        logvar: Tensor,
    ) -> None:
        super().__init__()

        self.mean = mean
        self.std = torch.exp(logvar / 2)
        # ? std is always ultra small...

    def __call__(self) -> Tensor:
        with torch.set_grad_enabled(self.training):
            return self.mean

    def sample(self) -> Tensor:
        with torch.set_grad_enabled(self.training):
            # TODO use seeds
            noise = torch.randn(self.std.shape, device=self.device, dtype=self.dtype)

            return self.mean + self.std * noise
