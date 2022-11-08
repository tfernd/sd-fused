from __future__ import annotations
from typing import Optional, TypeVar

from torch import Tensor

T = TypeVar("T", Tensor, int, float, str)


def separate(xs: list[T | None]) -> Optional[list[T]]:
    """Separate a list of that may containt `None` into a list
    that does not contain `None` or is `None` itself."""

    assert len(xs) >= 1

    if xs[0] is None:
        for x in xs:
            assert x is None

        return None

    for x in xs:
        assert x is not None

    return [x for x in xs if x is not None]
