from __future__ import annotations
from typing import TypeVar, Union, Iterable

from PIL import Image

try:
    from typing import Literal, Final, Protocol
except ImportError:
    from typing_extensions import Literal, Final, Protocol


T = TypeVar("T", int, float, str, Union[str, Image.Image])
MaybeIterable = Union[T, Iterable[T]]
