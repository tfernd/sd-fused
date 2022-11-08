from __future__ import annotations
from typing import Union

from PIL import Image
from pathlib import Path

from ..typing import Literal

ResizeModes = Literal["resize", "resize-crop", "resize-pad"]
ImageType = Union[str, Path, Image.Image]
