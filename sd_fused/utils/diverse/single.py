from __future__ import annotations
from typing import Optional, TypeVar, overload

import torch

T = TypeVar("T", int, float, torch.device, torch.dtype)


@overload
def single(x: set[T]) -> T:
    ...


@overload
def single(x: set[T | None]) -> Optional[T]:
    ...


def single(x: set[T | None] | set[T]) -> Optional[T]:
    assert len(x) == 1

    return x.pop()
