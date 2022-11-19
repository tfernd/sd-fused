from __future__ import annotations
from typing import Optional, overload

import torch

from ...utils.typing import TypeVar
from ...layers.base.types import Device

T = TypeVar("T", int, float, Device, torch.dtype)


@overload
def single(x: set[T]) -> T:
    ...


@overload
def single(x: set[Optional[T]]) -> Optional[T]:
    ...


def single(x: set[Optional[T]] | set[T]) -> Optional[T]:
    """Get the single element of a set."""

    assert len(x) == 1

    return x.pop()
