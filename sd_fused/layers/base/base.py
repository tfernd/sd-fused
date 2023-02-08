from __future__ import annotations
from typing import Optional
from typing_extensions import Self

import torch
from torch import Tensor


class Base:
    # Default dtype and device for the module and its parameters
    dtype: torch.dtype = torch.float16
    device: torch.device = torch.device("cuda")

    def to(
        self,
        device: Optional[str | torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Self:
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype

        for value in self.__dict__.values():
            if isinstance(value, Tensor):
                value.data = value.to(device=device, dtype=dtype)

        return self

    def float(self) -> Self:
        """Moves the module and its parameters to the float32 dtype."""

        return self.to(dtype=torch.float32)

    def half(self) -> Self:
        """Moves the module and its parameters to the float16 dtype."""

        return self.to(dtype=torch.float16)

    def cpu(self) -> Self:
        """Moves the module and its parameters to the CPU."""

        return self.to(device=f"cpu")

    def cuda(self, index: int = 0) -> Self:
        """Moves the module and its parameters to the specified GPU."""

        return self.to(device=f"cuda:{index}")
