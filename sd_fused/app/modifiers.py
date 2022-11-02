from __future__ import annotations
from typing import Optional
from typing_extensions import Self

import torch

from ..models import AutoencoderKL, UNet2DConditional
from ..utils import clear_cuda
from ..utils.typing import Literal


class Modifiers:
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
        attention_chunks: Optional[int | Literal["auto"]] = "auto",
    ) -> Self:
        """Split cross-attention computation into chunks."""

        self.unet.split_attention(attention_chunks)
        self.vae.split_attention(attention_chunks)

        return self

    def flash_attention(self, use_flash_attention: bool = True) -> Self:
        """Use xformers flash-attention."""

        self.unet.flash_attention(use_flash_attention)
        self.vae.flash_attention(use_flash_attention)

        return self
