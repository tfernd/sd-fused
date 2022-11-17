from __future__ import annotations
from typing import Optional
from typing_extensions import Self

import torch

from ..utils.typing import Literal
from ..utils.cuda import clear_cuda
from ..models import AutoencoderKL, UNet2DConditional
from ..layers.blocks.attention.compute import ChunkType
from ..layers.base.types import Device
from ..layers.base.base import Base


class Setup(Base):
    use_low_ram: bool

    vae: AutoencoderKL
    unet: UNet2DConditional

    def low_ram(self, use: bool = True) -> Self:
        """Split context into two passes to save memory."""

        self.use_low_ram = use

        return self

    def to(self, *, device: Optional[Device] = None, dtype: Optional[torch.dtype] = None) -> Self:
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype

        self.unet.to(device=self.device, dtype=dtype)
        self.vae.to(device=self.device, dtype=dtype)

        return self

    def cuda(self) -> Self:
        """Send unet and auto-encoder to cuda."""

        clear_cuda()

        return self.to(device="cuda")

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
        chunks: Optional[int | Literal["auto"]] = "auto",
        chunk_types: Optional[ChunkType] = None,
    ) -> Self:
        """Split cross-attention computation into chunks."""

        # TODO this should not be here...
        # default to batch if not set
        if chunks is not None and chunk_types is None:
            chunk_types = "batch"

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
