from __future__ import annotations
from typing import Optional

from functools import lru_cache
from pathlib import Path
import re

import torch
from torch import Tensor

from transformers.models.clip.modeling_clip import CLIPTextModel
from transformers.models.clip.tokenization_clip import CLIPTokenizer


class ClipEmbedding:
    """Convert a text to embeddings using a CLIP model."""

    def __init__(
        self, tokenizer_path: str | Path, text_encoder_path: str | Path
    ) -> None:
        # no need for CUDA for simple embeddings...
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)

    @staticmethod
    def parse_text(text: str) -> str:
        """Clean-up spaces/return characters."""

        text = text.replace("\n", " ")
        text = re.sub(r"[ ]+", r" ", text)
        text = text.strip()

        return text

    @lru_cache(maxsize=None)
    @torch.no_grad()
    def create_embedding(self, text: str) -> Tensor:
        """Creates an embedding for a text and cache it."""

        kwargs = dict(
            padding="max_length", truncation=True, return_tensors="pt"
        )

        # TODO give error if truncated?

        ids = self.tokenizer(text, **kwargs)["input_ids"]
        emb = self.text_encoder(ids)[0]  # type: ignore

        return emb.cpu()

    def __call__(
        self,
        text: str | list[str] = "",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Creates embeddings for a text and send to the correct device/dtype."""

        if isinstance(text, str):
            out = self.create_embedding(text)
        else:
            out = torch.cat([self.create_embedding(t) for t in text], dim=0)

        return out.to(device=device, dtype=dtype, non_blocking=True)