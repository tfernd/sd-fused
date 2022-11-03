from __future__ import annotations
from typing import Iterator, Optional

from dataclasses import dataclass
from typing_extensions import Self
from PIL.PngImagePlugin import PngInfo

import torch
from torch import Tensor

from ..utils import ResizeModes, image2tensor, image_base64
from .utils import separate


@dataclass
class Parameters:
    """Hold information from a single image generation."""

    eta: float
    steps: int
    scale: float
    height: int
    width: int
    seed: int
    negative_prompt: str
    prompt: Optional[str] = None
    img: Optional[str] = None
    mask: Optional[str] = None
    strength: Optional[float] = None
    mode: Optional[ResizeModes] = None

    # TODO add field
    device: Optional[torch.device] = None

    # TODO add field
    __save_args__ = [
        "eta",
        "steps",
        "scale",
        "height",
        "width",
        "seed",
        "negative_prompt",
        "prompt",
        "strength",
        "mode",
        "image_base64",
        "mask_base64",
    ]

    def __post_init__(self) -> None:
        assert self.height % 64 == 0
        assert self.width % 64 == 0

    def can_share_batch(self, other: Self) -> bool:
        """Determine if two parameters can share a batch."""

        value = self.steps == other.steps
        value &= self.strength == other.strength
        value &= self.height == other.height
        value &= self.width == other.width
        # guided/inguided?
        value &= (self.prompt is None) == (other.prompt is None)

        return value

    @property
    # @lru_cache(None)
    def image_data(self) -> Optional[Tensor]:
        """Image data as a Tensor."""

        if self.img is None or self.mode is None:
            return

        return image2tensor(
            self.img, self.height, self.width, self.mode, self.device
        )

    @property
    # @lru_cache(None)
    def mask_data(self) -> Optional[Tensor]:
        """Mask data as a Tensor."""

        if self.mask is None or self.mode is None:
            return

        return image2tensor(
            self.mask, self.height, self.width, self.mode, self.device
        )

    @property
    # @lru_cache(None)
    def image_base64(self) -> Optional[str]:
        """Image data as a base64 string."""

        if self.img is None or self.mode is None:
            return

        return image_base64(self.img)

    @property
    # @lru_cache(None)
    def mask_base64(self) -> Optional[str]:
        """Mask data as a base64 string."""

        if self.mask is None or self.mode is None:
            return

        return image_base64(self.mask)

    @property
    def png_info(self) -> PngInfo:
        info = PngInfo()

        for key in self.__save_args__:
            value = getattr(self, key)

            if value is None:
                continue
            if isinstance(value, (int, float)):
                value = str(value)

            info.add_text(f"SD {key}", value)

        return info


class ParametersList:
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
    def size(self) -> tuple[int, int]:
        height = set(p.height for p in self.parameters)
        width = set(p.width for p in self.parameters)

        assert len(height) == len(width) == 1

        return height.pop(), width.pop()

    @property
    def steps(self) -> int:
        steps = set(p.steps for p in self.parameters)

        assert len(steps) == 1

        return steps.pop()

    @property
    def strength(self) -> Optional[float]:
        strength = set(p.strength for p in self.parameters)

        assert len(strength) == 1

        return strength.pop()

    @property
    def device(self) -> Optional[torch.device]:
        devices = set(p.device for p in self.parameters)

        assert len(devices) == 1

        return devices.pop()

    @property
    def scales(self) -> Tensor:
        scales = [p.scale for p in self.parameters]

        return torch.tensor(scales, device=self.device)

    @property
    def etas(self) -> Tensor:
        etas = [p.eta for p in self.parameters]

        return torch.tensor(etas, device=self.device)

    @property
    def images_data(self) -> Optional[Tensor]:
        data = [p.image_data for p in self.parameters]
        data = separate(data)

        if data is None:
            return None

        return torch.cat(data, dim=0)

    @property
    def masks_data(self) -> Optional[Tensor]:
        data = [p.mask_data for p in self.parameters]
        data = separate(data)

        if data is None:
            return None

        return torch.cat(data, dim=0)
