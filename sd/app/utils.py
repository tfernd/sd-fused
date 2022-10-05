from __future__ import annotations
from typing import Optional


def fix_batch_size(seed: Optional[int | list[int]], batch_size: int) -> int:
    """Make sure the number of seeds matches the batch-size."""

    if isinstance(seed, list):
        seed = list(set(seed))  # remove duplicates
        batch_size = len(seed)

    return batch_size
