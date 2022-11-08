from __future__ import annotations

from PIL import Image
import requests

from io import BytesIO
import validators

from .types import ImageType


def open_image(path: ImageType) -> Image.Image:
    """Open a path or url as an image."""

    if isinstance(path, Image.Image):
        img = path
    elif isinstance(path, str) and validators.url(path):  # type: ignore
        response = requests.get(path)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(path)
    img = img.convert("RGB")

    # TODO get alpha channel as mask?

    return img
