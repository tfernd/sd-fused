from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor


class DiagonalGaussianDistribution:
    def __init__(
        self, mean: Tensor, logvar: Tensor, *, deterministic: bool = False,
    ) -> None:
        self.deterministic = deterministic

        self.device = mean.device
        self.dtype = mean.dtype

        self.mean = mean
        self.std = logvar.clamp_(-30, 20).div_(2).exp_()

    def sample(self, generator: Optional[torch.Generator] = None) -> Tensor:
        if self.deterministic:
            assert generator is None

            return self.mode

        noise = torch.randn(
            self.std.shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        return noise.mul_(self.std).add_(self.mean)

    @property
    def mode(self) -> Tensor:
        return self.mean
