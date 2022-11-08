from __future__ import annotations
from typing import Optional
from typing_extensions import Self

import torch

from ..utils.typing import Literal
from ..utils.cuda import clear_cuda
from ..models import AutoencoderKL, UNet2DConditional
from ..layers.blocks.attention import ChunkType


class Setup:
    low_ram: bool

    device: torch.device
    dtype: torch.dtype

    vae: AutoencoderKL
    unet: UNet2DConditional

    def set_low_ram(self, low_ram: bool = True) -> Self:
        """Split context into two passes to save memory."""

        self.low_ram = low_ram

        return self

    def to(self, device: Literal["cpu", "cuda"] | torch.device) -> Self:
        """Send unet and auto-encoder to device."""

        self.device = device = torch.device(device)

        self.unet.to(device=self.device, non_blocking=True)
        self.vae.to(device=self.device, non_blocking=True)

        return self

    def cuda(self) -> Self:
        """Send unet and auto-encoder to cuda."""

        clear_cuda()

        return self.to("cuda")

    def cpu(self) -> Self:
        """Send unet and auto-encoder to cpu."""

        return self.to("cpu")

    def half(self) -> Self:
        """Use half-precision for unet and auto-encoder."""

        self.unet.half()
        self.vae.half()

        self.dtype = torch.float16

        return self

    def float(self) -> Self:
        """Use full-precision for unet and auto-encoder."""

        self.unet.float()
        self.vae.float()

        self.dtype = torch.float32

        return self

    def half_weights(self, use_half_weights: bool = True) -> Self:
        """Store the weights in half-precision but
        compute forward pass in full precision.
        Useful for GPUs that gives NaN when used in half-precision.
        """

        self.unet.half_weights(use_half_weights)
        self.vae.half_weights(use_half_weights)

        return self

    def split_attention(
        self,
        chunks: Optional[int | Literal["auto"]] = "auto",
        chunk_types: Optional[ChunkType] = None,
    ) -> Self:
        """Split cross-attention computation into chunks."""

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

    def scheduler_renorm(self, renorm: bool = False) -> Self:
        # TODO this should not be here... ugly
        self.renorm = renorm

        return self
