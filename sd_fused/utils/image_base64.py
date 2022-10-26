from __future__ import annotations

from pathlib import Path
from PIL import Image

import base64
from io import BytesIO


def image_base64(path: str | Path) -> str:
    """Encodes an image as base64 string."""

    buffered = BytesIO()

    img = Image.open(path)
    img.save(buffered, format="JPEG")
    code = base64.b64encode(buffered.getvalue())
    code = code.decode("ascii")

    return f"data:image/jpg;base64,{code}"
