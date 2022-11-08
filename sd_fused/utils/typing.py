from __future__ import annotations
from typing import TypeVar, Union, Iterable

try:
    from typing import Literal, Final, Protocol
except ImportError:
    from typing_extensions import Literal, Final, Protocol

from .image import ImageType

T = TypeVar("T", int, float, str, ImageType)
MaybeIterable = Union[T, Iterable[T]]
