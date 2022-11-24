from __future__ import annotations

from ..base import Sequential
from ..basic import Linear, Identity
from ..activation import SiLU


class TimestepEmbedding(Sequential):
    def __init__(
        self,
        *,
        channel: int,
        time_embed_dim: int,
        use_silu: bool = True,  # !always true
    ) -> None:

        self.channel = channel
        self.time_embed_dim = time_embed_dim
        self.use_silu = use_silu

        layers = (
            Linear(channel, time_embed_dim),
            SiLU() if use_silu else Identity(),  # ? silu always true?
            Linear(time_embed_dim, time_embed_dim),
        )

        super().__init__(*layers)
