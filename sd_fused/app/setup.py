from __future__ import annotations
from typing import Optional
from typing_extensions import Self, Literal

import torch

from ..layers.base import Base
from ..utils.cuda import clear_cuda
from ..functional.attention import ChunkType
from .properties import Properties


class Setup(Base, Properties):
    use_low_ram: bool

    def low_ram(self, use: bool = True) -> Self:
        """Split context into two passes to save memory."""

        self.use_low_ram = use

        return self

    def to(self, *, device: Optional[str | torch.device] = None, dtype: Optional[torch.dtype] = None,) -> Self:
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype

        self.unet.to(device=self.device, dtype=dtype)
        self.vae.to(device=self.device, dtype=dtype)

        return self

    def cuda(self, index: int = 0) -> Self:
        """Send unet and auto-encoder to cuda."""

        clear_cuda()

        return self.to(device=f"cuda:{index}")

    def cpu(self) -> Self:
        """Send unet and auto-encoder to cpu."""

        return self.to(device="cpu")

    def half(self) -> Self:
        """Use half-precision for unet and auto-encoder."""

        return self.to(dtype=torch.float16)

    def float(self) -> Self:
        """Use full-precision for unet and auto-encoder."""

        return self.to(dtype=torch.float32)

    def half_weights(self, use: bool = True) -> Self:
        """Store the weights in half-precision but
        compute forward pass in full precision.
        Useful for GPUs that gives NaN when used in half-precision.
        """

        self.unet.half_weights(use)
        self.vae.half_weights(use)

        return self

    def split_attention(
        self,
        *,
        chunks: Optional[int | Literal["auto"]] = "auto",
        chunk_types: Optional[ChunkType] = None,
    ) -> Self:
        """Split cross-attention computation into chunks."""

        self.unet.split_attention(chunks, chunk_types)
        self.vae.split_attention(chunks, chunk_types)

        return self

    def flash_attention(self, use: bool = True) -> Self:
        """Use xformers flash-attention."""

        self.unet.flash_attention(use)
        self.vae.flash_attention(use)

        return self

    def tome(self, r: Optional[int | float] = None) -> Self:
        """Merge similar tokens."""

        self.unet.tome(r)
        self.vae.tome(r)

        return self
