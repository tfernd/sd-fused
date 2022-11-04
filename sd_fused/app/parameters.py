from __future__ import annotations
from typing import Iterator, Optional
from typing_extensions import Self

from dataclasses import dataclass, field
from functools import lru_cache
from PIL.PngImagePlugin import PngInfo

import torch
from torch import Tensor

from ..utils import ResizeModes, image2tensor, image_base64
from .utils import separate, single

SAVE_ARGS = [
    "eta",
    "steps",
    "scale",
    "height",
    "width",
    "seed",
    "negative_prompt",
    "sub_seed",
    "interpolation",
    "prompt",
    "strength",
    "mode",
    "image_base64",
    "mask_base64",
]


@dataclass
class Parameters:
    """Hold information from a single image generation."""

    eta: float
    steps: int
    scale: float  # TODO optional?
    height: int
    width: int
    seed: int
    negative_prompt: str
    sub_seed: Optional[int] = None
    interpolation: Optional[float] = None
    prompt: Optional[str] = None
    img: Optional[str] = None
    mask: Optional[str] = None
    strength: Optional[float] = None
    mode: Optional[ResizeModes] = None

    device: Optional[torch.device] = field(default=None, repr=False)
    dtype: Optional[torch.dtype] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        assert self.height % 8 == 0
        assert self.width % 8 == 0

        if self.img is None:
            assert self.mode is None
            assert self.strength is None
            assert self.mask is None
        else:
            assert self.mode is not None
            assert self.strength is not None

        if self.sub_seed is None:
            assert self.interpolation is None
        else:
            assert self.interpolation is not None

    @property
    def unconditional(self) -> bool:
        return self.prompt is None

    def can_share_batch(self, other: Self) -> bool:
        """Determine if two parameters can share a batch."""

        value = self.steps == other.steps
        value &= self.strength == other.strength
        value &= self.height == other.height
        value &= self.width == other.width
        # guided/inguided?
        value &= self.unconditional == other.unconditional  # XNOR?

        return value

    @property
    # @lru_cache(None) # TODO cached_property?
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

        data = image2tensor(
            self.mask, self.height, self.width, self.mode, self.device
        )

        # single-channel
        data = data.float().mean(dim=1, keepdim=True)

        return data >= 255 / 2  # bool-Tensor

    @property
    # @lru_cache(None)
    def image_base64(self) -> Optional[str]:
        """Image data as a base64 string."""

        if self.img is None:
            return

        return image_base64(self.img)

    @property
    # @lru_cache(None)
    def mask_base64(self) -> Optional[str]:
        """Mask data as a base64 string."""

        if self.mask is None:
            return

        return image_base64(self.mask)

    @property
    def png_info(self) -> PngInfo:
        """PNG metadata."""

        info = PngInfo()

        for key in SAVE_ARGS:
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
    def sub_seeds(self) -> Optional[list[int]]:
        sub_seeds = [p.sub_seed for p in self.parameters]
        sub_seeds = separate(sub_seeds)

        return sub_seeds

    @property
    def interpolations(self) -> Optional[Tensor]:
        interpolations = [p.interpolation for p in self.parameters]
        interpolations = separate(interpolations)

        if interpolations is None:
            return None

        return torch.tensor(
            interpolations, device=self.device, dtype=self.dtype
        )

    @property
    def size(self) -> tuple[int, int]:
        height = set(p.height for p in self.parameters)
        width = set(p.width for p in self.parameters)

        return single(height), single(width)

    @property
    def steps(self) -> int:
        steps = set(p.steps for p in self.parameters)

        return single(steps)

    @property
    def strength(self) -> Optional[float]:
        strength = set(p.strength for p in self.parameters)

        return single(strength)

    @property
    def device(self) -> Optional[torch.device]:
        devices = set(p.device for p in self.parameters)

        return single(devices)

    @property
    def dtype(self) -> Optional[torch.dtype]:
        dtypes = set(p.dtype for p in self.parameters)

        return single(dtypes)

    @property
    def scales(self) -> Tensor:
        scales = [p.scale for p in self.parameters]

        return torch.tensor(scales, device=self.device, dtype=self.dtype)

    @property
    def etas(self) -> Tensor:
        etas = [p.eta for p in self.parameters]

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


def group_parameters(parameters: list[Parameters]) -> list[list[Parameters]]:
    groups: list[list[Parameters]] = []
    for parameter in parameters:
        if len(groups) == 0:
            groups.append([parameter])
            continue

        can_share = False
        for group in groups:
            can_share = True
            for other_parameter in group:
                can_share &= parameter.can_share_batch(other_parameter)

            if can_share:
                group.append(parameter)
                break

        if not can_share:
            groups.append([parameter])

    return groups


def batch_parameters(
    groups: list[list[Parameters]],
    batch_size: int,
) -> list[list[Parameters]]:
    batched_parameters: list[list[Parameters]] = []
    for group in groups:
        for i in range(0, len(group), batch_size):
            s = slice(i, min(i + batch_size, len(group)))

            batched_parameters.append(group[s])

    return batched_parameters
