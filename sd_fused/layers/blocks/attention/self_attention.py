from __future__ import annotations
from typing import Optional

import torch.nn as nn
from torch import Tensor

from einops.layers.torch import Rearrange
from einops import rearrange

from ....utils.typing import Literal
from ...base import GroupNorm, Linear
from ...fn import attention


class SelfAttention(nn.Module):
    attention_chunks: Optional[int | Literal["auto"]] = None
    use_flash_attention: bool = False

    def __init__(
        self,
        *,
        in_features: int,
        head_features: Optional[int],
        num_groups: int,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.head_features = head_features
        self.num_groups = num_groups

        head_features = head_features or in_features
        num_heads = in_features // head_features

        # TODO These 4+1 operations can be fused
        self.group_norm = GroupNorm(num_groups, in_features)
        self.query = Linear(in_features)
        self.key = Linear(in_features)
        self.value = Linear(in_features)

        self.proj_attn = Linear(in_features)

        self.channel_last_and_spatial_join = Rearrange("B C H W -> B (H W) C")
        self.heads_to_batch = Rearrange(
            "B HW (heads C) -> (B heads) HW C",
            heads=num_heads,
        )
        self.heads_to_channel = Rearrange(
            "(B heads) HW C -> B HW (heads C)", heads=num_heads
        )

    def __call__(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        xin = x

        # normalization
        x = self.group_norm(x)

        x = self.channel_last_and_spatial_join(x)

        # key, query, value projections
        q = self.heads_to_batch(self.query(x))
        k = self.heads_to_batch(self.key(x))
        v = self.heads_to_batch(self.value(x))
        del x

        x = attention(
            q,
            k,
            v,
            chunks=self.attention_chunks,
            use_flash_attention=self.use_flash_attention,
        )
        del q, k, v
        x = self.proj_attn(self.heads_to_channel(x))

        # output
        x = rearrange(x, "B (H W) C -> B C H W", H=H, W=W)

        return xin + x
