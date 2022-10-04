from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor

from ..base import InPlace


class DiagonalGaussianDistribution(InPlace):
    def __init__(
        self, mean: Tensor, logvar: Tensor, *, deterministic: bool = False,
    ) -> None:
        super().__init__()

        self.deterministic = deterministic

        self.device = mean.device
        self.dtype = mean.dtype

        self.mean = mean

        limits = (-30, 20)
        if self.inplace:
            logvar = logvar.clamp_(*limits)
            self.std = logvar.div_(2).exp_()
        else:
            logvar = logvar.clamp(*limits)
            self.std = torch.exp(logvar / 2)

    def sample(self, generator: Optional[torch.Generator] = None) -> Tensor:
        if self.deterministic:
            assert generator is None

            return self.mean

        noise = torch.randn(
            self.std.shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        if self.inplace:
            return noise.mul_(self.std).add_(self.mean)
        return self.mean + self.std * noise
