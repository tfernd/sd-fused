# type: ignore
from __future__ import annotations
from typing import TypeVar, Union, Iterable
from typing_extensions import Unpack

try:
    from typing import Literal, Final, Protocol
except ImportError:
    from typing_extensions import Literal, Final, Protocol

try:
    from typing import TypeVarTuple
except ImportError:
    from typing_extensions import TypeVarTuple


# import sys
# if sys.version_info >= (3, 8):
# else:
# if sys.version_info >= (3, 11):
# else:

from .image import ImageType

T = TypeVar("T", int, float, str, ImageType)
MaybeIterable = Union[T, Iterable[T]]
