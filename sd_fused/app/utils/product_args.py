from __future__ import annotations

from itertools import product

from .to_list import to_list


def product_args(repeat: int = 1, **kwargs):
    args = list(kwargs.values())
    keys = list(kwargs.keys())

    args = tuple(map(to_list, args))
    perms = list(product(*args))

    return [dict(zip(keys, args)) for args in perms for _ in range(repeat)]
