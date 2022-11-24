from __future__ import annotations
from typing import TypeVar, Union, Iterable
from typing_extensions import Unpack

import sys

if sys.version_info >= (3, 8):
    from typing import Literal, Final, Protocol
else:
    from typing_extensions import Literal, Final, Protocol

if sys.version_info >= (3, 11):
    from typing import TypeVarTuple
else:
    from typing_extensions import TypeVarTuple

from .image import ImageType

T = TypeVar("T", int, float, str, ImageType)
MaybeIterable = Union[T, Iterable[T]]
