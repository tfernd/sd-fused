from __future__ import annotations
from typing import Iterable, Optional, TypeVar, overload

T = TypeVar("T", int, float, str)


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
