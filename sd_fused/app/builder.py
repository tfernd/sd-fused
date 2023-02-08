from __future__ import annotations
from typing_extensions import Self

from itertools import product
import random

from ..layers.base import Base
from ..scheduler import Scheduler, DDIM
from .properties import Properties
from .containers import Prompt, StepInfo


class Builder(Base, Properties):
    _scheduler: Scheduler

    _prompts: list[Prompt]
    _negative_prompts: list[Prompt]
    _scales: list[float]
    _seeds: list[int]
    _height: int
    _width: int
    _steps: int
    _batch_size: int
    _repeat: int

    _step_info: list[StepInfo]

    def add_ddim(
        self,
        *,
        steps: int = 20,
        eta: float = 0,
        approx: bool = True,
    ) -> Self:
        self._steps = steps

        self._scheduler = DDIM(steps, eta, approx)
        self._scheduler.to(self.device, self.dtype)

        return self

    def builder(self) -> Self:
        # initialize empty lists
        self._prompts = []
        self._negative_prompts = []
        self._scales = []
        self._seeds = []
        self._step_info = []

        # defaults
        self.add_ddim()
        self.add_size()
        self.add_batch_size()
        self.add_repeat()

        return self

    def add_prompt(self, prompt: str) -> Self:
        emb, weights = self.clip(prompt)
        self._prompts.append(Prompt(prompt, emb, weights))

        return self

    # TODO avoid duplication
    def add_negative_prompt(self, prompt: str) -> Self:
        emb, weights = self.clip(prompt)
        self._negative_prompts.append(Prompt(prompt, emb, weights))

        return self

    def add_size(
        self,
        *,
        height: int = 512,
        width: int = 512,
    ) -> Self:
        height -= height % 8
        width -= width % 8
        assert height > 0 and width > 0

        self._height = height
        self._width = width

        return self

    def add_scales(self, *scale: float) -> Self:
        assert len(scale) > 0
        self._scales.extend(scale)

        return self

    def add_seeds(self, *seed: int) -> Self:
        assert len(seed) > 0

        self._seeds.extend(seed)

        return self

    def add_batch_size(self, batch_size: int = 1) -> Self:
        self._batch_size = batch_size

        return self

    def add_repeat(self, repeat: int = 1) -> Self:
        self._repeat = repeat

        return self

    def build(self) -> Self:
        # default list
        if len(self._negative_prompts) == 0:
            self.add_negative_prompt("")
        if len(self._scales) == 0:
            self.add_scales(7.5)

        # all permutations
        pns = product(self._prompts, self._negative_prompts, self._scales)
        pns = list(pns)
        size = len(pns)

        # copy pns and distribute seeds
        if len(self._seeds) == 0:
            self._seeds = [random.randint(0, 2**32 - 1) for _ in range(self._repeat)]

        pns = [p for p in pns for seed in self._seeds]
        self._seeds = [seed for _ in range(size) for seed in self._seeds]

        current_step = 0
        for (prompt, negative_prompt, scale), seed in zip(pns, self._seeds):
            info = StepInfo(prompt, negative_prompt, current_step, scale, seed)
            info.to(self.device, self.dtype)
            info.init_latents(self._height, self._width, self.latent_size)
            self._step_info.append(info)

        return self
