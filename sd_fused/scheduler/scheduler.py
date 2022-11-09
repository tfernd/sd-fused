from __future__ import annotations
from typing import Optional
from typing_extensions import Self
import abc

import math
from torch import Tensor


class Scheduler(abc.ABC):
    skip_step: int

    @property
    @abc.abstractmethod
    def info(self) -> dict[str, str | int]:
        """Scheduler information."""

    @abc.abstractmethod
    def step(
        self,
        pred_noise: Tensor,
        latents: Tensor,
        i: int,
        eta: float | Tensor = 0,
    ) -> Tensor:
        """Get the previous latents."""

    @abc.abstractmethod
    def add_noise(
        self,
        latents: Tensor,
        eps: Tensor,
        i: int,
    ) -> Tensor:
        """Add noise to latents according to the index i."""

    def set_skip_step(self, strength: Optional[float]) -> Self:
        """The index generation needs to start."""

        if strength is None:
            self.skip_step = 0
        else:
            assert 0 < strength <= 1

            self.skip_step = math.ceil(len(self) * (1 - strength))

        return self

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @abc.abstractmethod
    def __repr__(self) -> str:
        ...
