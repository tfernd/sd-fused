from __future__ import annotations
from typing import Iterable, Optional, TypeVar, overload

from PIL import Image

from ..typing import MaybeIterable, T


@overload
def to_list(x: MaybeIterable[T]) -> list[T]:
    ...


@overload
def to_list(x: Optional[MaybeIterable[T]]) -> tuple[None] | list[T]:
    ...


def to_list(x: Optional[MaybeIterable[T]]) -> tuple[None] | list[T]:
    if x is None:
        return (None,)

    if isinstance(x, (int, float, str, Image.Image)):
        return [x]  # type: ignore

    return list(x)
