import gc
import torch


def clear_cuda() -> None:
    """Clear CUDA memory and garbage collection."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
