from __future__ import annotations
from typing import Any, Optional


def fix_batch_size(
    seed: Optional[int | list[int]], batch_size: int
) -> tuple[Optional[int | list[int]], int]:
    """Make sure the number of seeds match the batch-size 
    and/or auto-increament singl seed.
    """

    if isinstance(seed, list):
        seed = list(set(seed))  # remove duplicates
        batch_size = len(seed)
    elif isinstance(seed, int) and batch_size != 1:
        # auto-increament seed
        seed = [seed + i for i in range(batch_size)]

    return seed, batch_size


def kwargs2ignore(
    kwargs: dict[str, Any], *, keys: list[str]
) -> dict[str, Any]:
    return {
        key: value
        for (key, value) in kwargs.items()
        if key not in keys and key != "self"
    }
