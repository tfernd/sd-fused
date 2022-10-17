from __future__ import annotations
from typing import Any, Optional


def fix_batch_size(seed: Optional[int | list[int]], batch_size: int) -> int:
    """Make sure the number of seeds matches the batch-size."""

    if isinstance(seed, list):
        seed = list(set(seed))  # remove duplicates
        batch_size = len(seed)

    return batch_size


def kwargs2ignore(
    kwargs: dict[str, Any], *, keys: list[str]
) -> dict[str, Any]:
    return {
        key: value
        for (key, value) in kwargs.items()
        if key not in keys and key != "self"
    }
