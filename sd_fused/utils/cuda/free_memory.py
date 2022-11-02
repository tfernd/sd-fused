from __future__ import annotations

import torch


def free_memory() -> int:
    """Amount of free memory available."""

    stats = torch.cuda.memory_stats()

    reserved = stats["reserved_bytes.all.current"]
    active = stats["active_bytes.all.current"]
    free = torch.cuda.mem_get_info()[0]  # type: ignore

    free += reserved - active

    return free
