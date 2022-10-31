from __future__ import annotations
from typing import Any, Optional

from dataclasses import dataclass
from PIL.PngImagePlugin import PngInfo

from ..utils import ResizeModes


def fix_batch_size(
    seed: Optional[int | list[int]], batch_size: int
) -> tuple[Optional[int | list[int]], int]:
    """Make sure the number of seeds match the batch-size
    and/or auto-increament singl seed.
    """

    if isinstance(seed, list):
        seed = list(set(seed))  # remove duplicates
        batch_size = len(seed)
    elif isinstance(seed, int) and batch_size != 1:
        # auto-increament seed
        seed = [seed + i for i in range(batch_size)]

    return seed, batch_size


def kwargs2ignore(
    kwargs: dict[str, Any], *, keys: list[str]
) -> dict[str, Any]:
    return {
        key: value
        for (key, value) in kwargs.items()
        if key not in keys and key != "self"
    }


@dataclass(repr=False)
class Metadata:
    version: str
    model: str
    eta: float
    steps: int
    scale: float
    height: int
    width: int
    seed: int
    negative_prompt: str
    prompt: Optional[str] = None
    strength: Optional[float] = None
    mode: Optional[ResizeModes] = None
    img_base64: Optional[str] = None

    @property
    def png_info(self) -> PngInfo:
        info = PngInfo()
        for key, value in self.__dict__.items():
            if value is None:
                continue

            if isinstance(value, (int, float)):
                value = str(value)

            info.add_text(f"SD {key}", value)

        return info

    def __repr__(self) -> str:
        args: list[str] = []
        for key, value in self.__dict__.items():
            if value is None:
                continue
            if isinstance(value, str):
                args.append(f'{key}="{value}"')
            else:
                args.append(f"{key}={value}")

        name = self.__class__.__qualname__
        out = ", ".join(args)

        return f"{name}({out})"
