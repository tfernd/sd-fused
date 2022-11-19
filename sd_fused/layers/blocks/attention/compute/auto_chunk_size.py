from typing import Optional

import torch

from .....utils.typing import Literal
from .....utils.cuda import free_memory


ChunkType = Literal["batch", "sequence"]


def auto_chunk_size(
    chunks: Optional[int | Literal["auto"]],
    B: int,
    heads: int,
    T: int,
    Tl: int,
    C: int,
    dtype: torch.dtype,
    chunk_type: Optional[ChunkType],
) -> Optional[int]:
    """Determine the maximum chunk size according to the available free memory."""

    if chunks != "auto":
        return chunks

    B *= heads # ! ugly but...

    assert chunk_type is not None
    assert dtype in (torch.float32, torch.float16)

    num_bytes = 2 if dtype == torch.float16 else 4
    free = free_memory()

    if chunk_type == "batch":
        # Memory used: (2*Bchunks*T*Tl + Bchunks*T*C) * num_bytes
        Bchunks = free // (num_bytes * T * (C + 2 * Tl))

        if Bchunks >= B:
            return None
        return Bchunks

    # Memory used: (2*B*Tchunk*Tl + B*Tchunk*C) * num_bytes
    Tchunks = free // (num_bytes * B * (C + 2 * Tl))

    if Tchunks >= T:
        return None
    return Tchunks
