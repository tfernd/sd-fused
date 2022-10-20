from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor


class DiagonalGaussianDistribution:
    def __init__(
        self, mean: Tensor, logvar: Tensor, *, deterministic: bool = False,
    ) -> None:
        super().__init__()

        self.deterministic = deterministic

        self.device = mean.device
        self.dtype = mean.dtype

        self.mean = mean

        logvar = logvar.clamp(-30, 20)
        self.std = torch.exp(logvar / 2)

    def sample(self, generator: Optional[torch.Generator] = None) -> Tensor:
        if self.deterministic:
            assert generator is None

            return self.mean

        # TODO use seeds?
        noise = torch.randn(
            self.std.shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        return self.mean + self.std * noise
