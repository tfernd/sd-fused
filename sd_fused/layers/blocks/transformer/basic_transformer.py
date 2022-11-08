from __future__ import annotations
from typing import Optional

import torch.nn as nn
from torch import Tensor

from ..attention import CrossAttention
from ..simple import LayerNormGEGLULinear


class BasicTransformer(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        num_heads: int,
        head_features: int,
        context_features: Optional[int],
    ):
        super().__init__()

        self.in_features = in_features
        self.num_heads = num_heads
        self.head_features = head_features
        self.context_features = context_features

        self.attn1 = CrossAttention(
            query_features=in_features,
            num_heads=num_heads,
            head_features=head_features,
            context_features=None,
        )
        self.attn2 = CrossAttention(
            query_features=in_features,
            num_heads=num_heads,
            head_features=head_features,
            context_features=context_features,
        )

        self.ff = LayerNormGEGLULinear(in_features, expand=4)

    def __call__(
        self,
        x: Tensor,
        *,
        context: Optional[Tensor] = None,
        context_weights: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.attn1(x)
        x = self.attn2(x, context=context, context_weights=context_weights)
        x = self.ff(x)

        return x
