from __future__ import annotations
from typing import Union
from typing_extensions import Literal

from PIL import Image
from pathlib import Path

ResizeModes = Literal["resize", "resize-crop", "resize-pad"]
ImageType = Union[str, Path, Image.Image]
