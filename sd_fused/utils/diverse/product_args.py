from __future__ import annotations
from typing import Optional

from itertools import product

from ..typing import MaybeIterable
from .to_list import to_list

# TODO types!
def product_args(
    repeat: int = 1,
    **kwargs: Optional[MaybeIterable],
) -> list[dict]:
    """All possible combintations of kwargs"""

    args = list(kwargs.values())
    keys = list(kwargs.keys())

    args = tuple(map(to_list, args))
    perms = list(product(*args))

    return [dict(zip(keys, args)) for args in perms for _ in range(repeat)]
