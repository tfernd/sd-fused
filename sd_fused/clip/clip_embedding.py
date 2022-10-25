from __future__ import annotations
from typing import Optional

from functools import lru_cache
from pathlib import Path
import re

import torch

from transformers.models.clip.modeling_clip import CLIPTextModel
from transformers.models.clip.tokenization_clip import CLIPTokenizer

from .text_segment import TextSegment
from .container import TensorAndWeight, TensorAndMaybeWeight
from .parser import (
    add_delimiter4words,
    expand_delimiters,
    add_split_maker4emphasis,
)

MAX_TOKENS = 77


class ClipEmbedding:
    """Convert a text to embeddings using a CLIP model."""

    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel

    def __init__(
        self, tokenizer_path: str | Path, text_encoder_path: str | Path,
    ) -> None:
        # TODO check if valid paths?
        # no need for CUDA for simple embeddings...
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)  # type: ignore

    @staticmethod
    def clean_spaces(text: str) -> str:
        """Clean-up spaces/return characters."""

        text = text.replace("\n", " ")
        text = re.sub(r"[ ]+", r" ", text)
        text = text.strip()

        return text

    @staticmethod
    def parse_emphasis(text: str) -> str:
        """Parse emphasis notation."""

        text = add_delimiter4words(text)
        text = expand_delimiters(text)
        text = add_split_maker4emphasis(text)

        return text

    @lru_cache(maxsize=None)
    def get_ids_and_weights(self, text: str) -> TensorAndWeight:
        """Get the token id and weight for a given text."""

        text = self.clean_spaces(text)
        text = self.parse_emphasis(text)

        segments = [TextSegment(t) for t in text.split("⏎")]

        ids: list[int] = []
        weights: list[float] = []
        for n, seg in enumerate(segments):
            w = 1 if seg.weight is None else seg.weight
            seg_ids = self.tokenizer.encode(seg.text)

            # remove initial/final ids
            seg_ids = seg_ids[1:-1]

            ids.extend(seg_ids)
            weights.extend([w] * (len(seg_ids)))

        # add padding and initial/final ids
        n = MAX_TOKENS - len(ids) - 2
        assert n >= 0, "Text too big, it will result in truncation"
        assert self.tokenizer.bos_token_id is not None
        assert self.tokenizer.eos_token_id is not None
        assert self.tokenizer.pad_token_id is not None
        ids = [
            self.tokenizer.bos_token_id,
            *ids,
            *[self.tokenizer.pad_token_id] * n,
            self.tokenizer.eos_token_id,
        ]
        weights = [1, *weights, *[1] * n, 1]

        return TensorAndWeight(
            torch.tensor([ids]), torch.tensor([weights]).float(),
        )

    @lru_cache(maxsize=None)
    @torch.no_grad()
    def get_embedding(self, text: str) -> TensorAndWeight:
        """Creates an embedding/weights for a text and cache it."""

        ids, weight = self.get_ids_and_weights(text)
        emb = self.text_encoder(ids)[0]

        return TensorAndWeight(emb, weight)

    def __call__(
        self,
        text: str | list[str] = "",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> TensorAndMaybeWeight:
        """Creates embeddings/weights for a text and send to the correct device/dtype."""

        if isinstance(text, str):
            emb, weight = self.get_embedding(text)
        else:
            values = [self.get_embedding(t) for t in text]
            emb = torch.cat([v.tensor for v in values])
            weight = torch.cat([v.weight for v in values])

        emb = emb.to(device=device, dtype=dtype, non_blocking=True)

        # special case where the weights are all one
        if weight.diff(1).any():
            weight = weight.to(device=device, dtype=dtype, non_blocking=True)

            return TensorAndMaybeWeight(emb, weight)

        return TensorAndMaybeWeight(emb)
