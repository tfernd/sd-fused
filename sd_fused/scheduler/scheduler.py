from __future__ import annotations
import abc
from typing import Optional

from torch import Tensor


class Scheduler(abc.ABC):
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

    @abc.abstractmethod
    def set_skip_step(self, strength: Optional[float]) -> None:
        """The index generation needs to start."""

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @abc.abstractmethod
    def __repr__(self) -> str:
        ...
