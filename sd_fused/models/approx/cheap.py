from __future__ import annotations

from torch import Tensor

from ...functional import denormalize
from ...layers.base import Module, Sequential
from ...layers.basic import Conv2d, GLU

class DecoderApproximationSmall(Module):
    # https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/2

    def __init__(
        self,
        kernel_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()

        assert kernel_size % 3 == 1

        self.kernel_size = kernel_size
        self.hidden_size = hidden_size

        self.seq = Sequential(
            Conv2d(4, 2 * hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            GLU(dim=1),
            Conv2d(hidden_size, 3, kernel_size=1),
        )

    def __call__(self, x: Tensor) -> Tensor:
        x = self.seq(x)
        x = denormalize(x)

        return x
