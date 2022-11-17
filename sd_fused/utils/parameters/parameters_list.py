from __future__ import annotations
from typing import Iterator, Optional

from functools import lru_cache

import torch
from torch import Tensor

from ...layers.base.types import Device
from ..diverse import separate, single
from .parameters import Parameters


class ParametersList:
    """Hold information from a many image generations."""

    def __init__(self, parameters: list[Parameters]) -> None:
        self.parameters = parameters

    def __len__(self) -> int:
        return len(self.parameters)

    def __iter__(self) -> Iterator[Parameters]:
        return iter(self.parameters)

    @property
    def prompts(self) -> Optional[list[str]]:
        prompts = [p.prompt for p in self.parameters]
        prompts = separate(prompts)

        return prompts

    @property
    def negative_prompts(self) -> list[str]:
        return [p.negative_prompt for p in self.parameters]

    @property
    def unconditional(self) -> bool:
        return self.prompts is None

    @property
    def seeds(self) -> list[int]:
        return [p.seed for p in self.parameters]

    @property
    def sub_seeds(self) -> Optional[list[int]]:
        sub_seeds = [p.sub_seed for p in self.parameters]
        sub_seeds = separate(sub_seeds)

        return sub_seeds

    @property
    def seeds_interpolation(self) -> Optional[Tensor]:
        seed_interpolations = [p.seed_interpolation for p in self.parameters]
        seed_interpolations = separate(seed_interpolations)

        if seed_interpolations is None:
            return None

        return torch.tensor(seed_interpolations, device=self.device, dtype=self.dtype)

    @property
    def height(self) -> int:
        height = set(p.height for p in self.parameters)

        return single(height)

    @property
    def width(self) -> int:
        width = set(p.width for p in self.parameters)

        return single(width)

    @property
    def steps(self) -> int:
        steps = set(p.steps for p in self.parameters)

        return single(steps)

    @property
    def strength(self) -> Optional[float]:
        strength = set(p.strength for p in self.parameters)

        return single(strength)

    @property
    def device(self) -> Optional[Device]:
        devices = set(p.device for p in self.parameters)

        return single(devices)

    @property
    def dtype(self) -> Optional[torch.dtype]:
        dtypes = set(p.dtype for p in self.parameters)

        return single(dtypes)

    @property
    def scales(self) -> Optional[Tensor]:
        scales = [p.scale for p in self.parameters]
        scales = separate(scales)

        if scales is None:
            return None

        return torch.tensor(scales, device=self.device, dtype=self.dtype)

    @property
    def etas(self) -> Optional[Tensor]:
        etas = [p.eta for p in self.parameters]
        etas = separate(etas)

        if etas is None:
            return None

        return torch.tensor(etas, device=self.device, dtype=self.dtype)

    @property
    @lru_cache(None)
    def images_data(self) -> Optional[Tensor]:
        data = [p.image_data for p in self.parameters]
        data = separate(data)

        if data is None:
            return None

        return torch.cat(data, dim=0)

    @property
    @lru_cache(None)
    def masks_data(self) -> Optional[Tensor]:
        data = [p.mask_data for p in self.parameters]
        data = separate(data)

        if data is None:
            return None

        return torch.cat(data, dim=0)

    @property
    @lru_cache(None)
    def masked_images_data(self) -> Optional[Tensor]:
        if self.images_data is None or self.masks_data is None:
            return None

        return self.images_data * (~self.masks_data)
