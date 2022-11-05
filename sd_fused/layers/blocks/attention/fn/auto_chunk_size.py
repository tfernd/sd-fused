from typing import Optional

import torch
import math

from .....utils.cuda import free_memory


def auto_chunk_size(
    B: int,
    T: int,
    Tl: int,
    C: int,
    dtype: torch.dtype,
) -> Optional[int]:
    """Determine the maximum chunk size accordint
    to the available free memory.
    """

    assert dtype in (torch.float32, torch.float16)

    numel = 2 * B * T * Tl + B * T * C
    num_bytes = 2 if dtype == torch.float16 else 4
    memory = numel * num_bytes

    free = free_memory()

    if free > memory:
        return None

    return math.floor(B * free / memory)
