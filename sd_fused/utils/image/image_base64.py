from __future__ import annotations

from PIL import Image
from pathlib import Path
import base64
from io import BytesIO

from .open_image import open_image


def image_base64(path: str | Path | Image.Image) -> str:
    """Encodes an image as base64 (JPGE) string."""

    img = open_image(path)

    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    code = base64.b64encode(buffered.getvalue())
    code = code.decode("ascii")

    return f"data:image/jpg;base64,{code}"
