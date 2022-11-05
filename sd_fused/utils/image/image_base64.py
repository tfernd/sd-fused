from __future__ import annotations

from pathlib import Path
import base64
from io import BytesIO

from .open_image import open_image


def image_base64(path: str | Path) -> str:
    """Encodes an image as base64 string."""

    img = open_image(path)

    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    code = base64.b64encode(buffered.getvalue())
    code = code.decode("ascii")

    return f"data:image/jpg;base64,{code}"
