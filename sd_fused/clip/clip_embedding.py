from __future__ import annotations
from typing_extensions import Literal

from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor

from transformers.models.clip.modeling_clip import CLIPTextModel
from transformers.models.clip.tokenization_clip import CLIPTokenizer

from .parse import (
    clean_spaces,
    add_parentheses_for_weights,
    emphasize_words_with_delimiters,
    add_split_maker_for_emphasis,
    split_prompt_into_individual_segments,
)
from .containers import TextSegment


class ClipEmbedding:
    """Convert a text to embeddings using a CLIP model."""

    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel

    # IDs
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int

    max_tolens: Literal[77] = 77

    def __init__(
        self,
        tokenizer_path: str | Path,
        text_encoder_path: str | Path,
    ) -> None:
        # no need for CUDA for simple embeddings...
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)  # type: ignore

        # token ids
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    @lru_cache(maxsize=None)
    def get_ids(self, prompt: str) -> tuple[Tensor, Tensor]:
        """Get the token id and weight for a given prompt."""

        prompt = clean_spaces(prompt)
        prompt = add_parentheses_for_weights(prompt)
        prompt = emphasize_words_with_delimiters(prompt)
        prompt = add_split_maker_for_emphasis(prompt)

        segments = [TextSegment(t) for t in split_prompt_into_individual_segments(prompt) if len(t) > 0]

        # TODO create function
        ids: list[int] = []
        weights: list[float] = []

        for seg in segments:
            seg_ids = self.tokenizer.encode(seg.prompt)

            # remove initial/final ids
            seg_ids = seg_ids[1:-1]

            ids.extend(seg_ids)
            weights.extend([seg.weight] * (len(seg_ids)))

        # add padding and initial/final ids
        pad_size = self.max_tolens - len(ids) - 2
        assert pad_size >= 0, "Text too big, it will result in truncation"

        ids = [
            self.bos_token_id,
            *ids,
            *[self.pad_token_id] * pad_size,
            self.eos_token_id,
        ]
        weights = [1, *weights, *[1] * pad_size, 1]

        _ids = torch.tensor(ids).long()
        _weights = torch.tensor(weights).float()

        return _ids, _weights

    @lru_cache(maxsize=None)
    @torch.no_grad()
    def __call__(self, prompt: str) -> tuple[Tensor, Tensor]:
        """Creates an embedding/weights for a prompt and cache it."""

        ids, weights = self.get_ids(prompt)
        emb = self.text_encoder(ids[None])[0]

        return emb, weights
