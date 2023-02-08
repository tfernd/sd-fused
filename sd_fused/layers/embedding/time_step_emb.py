from __future__ import annotations

from ...layers.basic import Linear
from ...layers.activation import SiLU
from ..base import Sequential


class TimestepEmbedding(Sequential):
    def __init__(
        self,
        *,
        num_channels: int,
        time_embed_dim: int,
    ) -> None:

        self.num_channels = num_channels
        self.time_embed_dim = time_embed_dim

        layers = (
            Linear(num_channels, time_embed_dim),
            SiLU(),
            Linear(time_embed_dim, time_embed_dim),
        )

        super().__init__(*layers)
