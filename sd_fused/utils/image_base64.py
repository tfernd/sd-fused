from __future__ import annotations

from pathlib import Path
from PIL import Image
import requests

import base64
from io import BytesIO
import validators


def image_base64(path: str | Path) -> str:
    """Encodes an image as base64 string."""

    buffered = BytesIO()

    # TODO de-duplicate
    if isinstance(path, str) and validators.url(path):  # type: ignore
        response = requests.get(path)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(path)
    img = img.convert("RGB")

    img.save(buffered, format="JPEG")
    code = base64.b64encode(buffered.getvalue())
    code = code.decode("ascii")

    return f"data:image/jpg;base64,{code}"
