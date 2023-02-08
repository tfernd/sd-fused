from __future__ import annotations

from abc import ABC

from torch import Tensor

from ..layers.base import Base


class Scheduler(ABC, Base):
    """Base-class for all schedulers."""

    steps: int
    timesteps: Tensor  # time-embedding step

    noise_latents: Tensor
    noise_noise: Tensor

    step_latents: Tensor
    step_pred_noise: Tensor
    step_noise: Tensor

    # for repr
    _parameters: dict[str, int | float]

    def __init__(
        self,
        steps: int,
    ) -> None:
        assert steps >= 1

        self.steps = steps

        self._parameters = dict(steps=steps)

    def add_noise(
        self,
        index: Tensor,
        latents: Tensor,
        noise: Tensor,
    ) -> Tensor:
        index = index.view(-1, 1, 1, 1)

        out = self.noise_latents[index] * latents
        out += self.noise_noise[index] * noise

        return out

    def step(
        self,
        index: Tensor,
        latents: Tensor,
        pred_noise: Tensor,
        noise: Tensor,
    ) -> Tensor:
        index = index.view(-1, 1, 1, 1)

        out = self.step_latents[index] * latents
        out += self.step_pred_noise[index] * pred_noise
        out += self.step_noise[index] * noise

        return out

    @property
    def parameters(self) -> dict[str, int | float | str]:
        return dict(name=self.__class__.__qualname__, **self._parameters)

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        args = [f"{name}={value}" for (name, value) in self._parameters.items()]
        args = ", ".join(args)

        return f"{name}({args})"

    def plot(self) -> None:
        import matplotlib.pyplot as plt

        plt.plot(self.noise_latents.cpu(), label="noise_latents", linestyle="dashed")
        plt.plot(self.noise_noise.cpu(), label="noise_noise", linestyle="dashed")
        plt.plot(self.step_latents.cpu(), label="step_latents")
        plt.plot(self.step_pred_noise.cpu(), label="step_pred_noise")
        plt.plot(self.step_noise.cpu(), label="step_noise")
        plt.legend()
        plt.xlabel("steps")
