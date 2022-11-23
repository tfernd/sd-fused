from __future__ import annotations
from typing import Optional, overload

from PIL import Image
from pathlib import Path

from ..typing import MaybeIterable, T


@overload
def to_list(x: MaybeIterable[T]) -> list[T]:
    ...


@overload
def to_list(x: Optional[MaybeIterable[T]]) -> tuple[None] | list[T]:
    ...


def to_list(x: Optional[MaybeIterable[T]]) -> tuple[None] | list[T]:
    """Convert a `MaybeIterable` into a list."""

    if x is None:
        return (None,)

    if isinstance(x, (int, float, str, Path, Image.Image)):
        return [x]  # type: ignore

    return list(x)  # type: ignore
