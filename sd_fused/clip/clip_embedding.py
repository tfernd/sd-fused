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
    split_prompt_into_segments,
)

MAX_TOKENS = 77


class ClipEmbedding:
    """Convert a text to embeddings using a CLIP model."""

    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel

    def __init__(
        self,
        tokenizer_path: str | Path,
        text_encoder_path: str | Path,
    ) -> None:
        # no need for CUDA for simple embeddings...
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)  # type: ignore

    @staticmethod
    def clean_spaces(prompt: str) -> str:
        """Clean-up spaces/return characters."""

        prompt = prompt.replace("\n", " ")
        prompt = re.sub(r"[ ]+", r" ", prompt)
        prompt = prompt.strip()

        return prompt

    @staticmethod
    def parse_emphasis(prompt: str) -> str:
        """Parse emphasis notation."""

        prompt = add_delimiter4words(prompt)
        prompt = expand_delimiters(prompt)
        prompt = add_split_maker4emphasis(prompt)

        return prompt

    @lru_cache(maxsize=None)
    def get_ids_and_weights(self, prompt: str) -> TensorAndWeight:
        """Get the token id and weight for a given prompt."""

        prompt = self.clean_spaces(prompt)
        prompt = self.parse_emphasis(prompt)

        segments = [TextSegment(t) for t in split_prompt_into_segments(prompt)]

        ids: list[int] = []
        weights: list[float] = []
        for n, seg in enumerate(segments):
            seg_ids = self.tokenizer.encode(seg.prompt)

            # remove initial/final ids
            seg_ids = seg_ids[1:-1]

            ids.extend(seg_ids)
            weights.extend([seg.weight] * (len(seg_ids)))

        # add padding and initial/final ids
        pad_size = MAX_TOKENS - len(ids) - 2
        assert pad_size >= 0, "Text too big, it will result in truncation"
        assert self.tokenizer.bos_token_id is not None
        assert self.tokenizer.eos_token_id is not None
        assert self.tokenizer.pad_token_id is not None

        ids = [
            self.tokenizer.bos_token_id,
            *ids,
            *[self.tokenizer.pad_token_id] * pad_size,
            self.tokenizer.eos_token_id,
        ]
        weights = [1, *weights, *[1] * pad_size, 1]

        return TensorAndWeight(torch.tensor([ids]), torch.tensor([weights]).float())

    @lru_cache(maxsize=None)
    @torch.no_grad()
    def get_embedding(self, prompt: str) -> TensorAndWeight:
        """Creates an embedding/weights for a prompt and cache it."""

        ids, weight = self.get_ids_and_weights(prompt)
        emb = self.text_encoder(ids)[0]

        return TensorAndWeight(emb, weight)

    def __call__(
        self,
        prompt: str | list[str] = "",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> TensorAndMaybeWeight:
        """Creates embeddings/weights for a prompt and send to the correct device/dtype."""

        if isinstance(prompt, str):
            prompt = [prompt]
        values = [self.get_embedding(t) for t in prompt]
        emb = torch.cat([v.tensor for v in values])
        weight = torch.cat([v.weight for v in values])

        emb = emb.to(device=device, dtype=dtype, non_blocking=True)

        # special case where all weights are one
        if weight.eq(1).all():
            return TensorAndMaybeWeight(emb)

        weight = weight.to(device=device, dtype=dtype, non_blocking=True)

        return TensorAndMaybeWeight(emb, weight)
