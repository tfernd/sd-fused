from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor

from transformers.models.clip.modeling_clip import CLIPTextModel
from transformers.models.clip.tokenization_clip import CLIPTokenizer


class ClipEmbedding:
    def __init__(
        self, tokenizer_path: str | Path, text_encoder_path: str | Path
    ) -> None:
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)

    @lru_cache(maxsize=None)
    @torch.no_grad()
    def create_embedding(self, text: str) -> Tensor:
        kwargs = dict(
            padding="max_length", truncation=True, return_tensors="pt"
        )

        ids = self.tokenizer(text, **kwargs)["input_ids"]
        emb = self.text_encoder(ids)[0]  # type: ignore

        return emb.cpu()

    def __call__(self, text: str | list[str] = "") -> Tensor:
        if isinstance(text, str):
            return self.create_embedding(text)

        return torch.cat([self.create_embedding(t) for t in text], dim=0)
