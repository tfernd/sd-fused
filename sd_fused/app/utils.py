from __future__ import annotations
from typing import Iterable, Optional, TypeVar, overload

from torch import Tensor

T = TypeVar("T", int, float, str)
K = TypeVar("K", Tensor, str)


@overload
def to_list(x: T | Iterable[T]) -> list[T]:
    ...


@overload
def to_list(x: Optional[T | Iterable[T]]) -> tuple[None] | list[T]:
    ...


def to_list(x: Optional[T | Iterable[T]]) -> tuple[None] | list[T]:
    if x is None:
        return (None,)

    if isinstance(x, (int, float, str)):
        return [x]  # type: ignore

    return list(x)


def separate(xs: list[K | None]) -> Optional[list[K]]:
    if xs[0] is None:
        for x in xs:
            assert x is None

        return None

    for x in xs:
        assert x is not None

    xs = [x for x in xs if x is not None]

    return xs  # type: ignore
