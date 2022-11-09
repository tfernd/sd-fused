from __future__ import annotations

from .open_image import open_image
from .types import ImageType


def image_size(path: ImageType) -> tuple[int, int]:
    img = open_image(path)

    width, height = img.size

    return height, width
