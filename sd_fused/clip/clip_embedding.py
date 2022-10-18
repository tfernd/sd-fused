from __future__ import annotations
from typing import NamedTuple, Optional

from functools import lru_cache
from pathlib import Path
import re

import torch
from torch import Tensor

from transformers.models.clip.modeling_clip import CLIPTextModel
from transformers.models.clip.tokenization_clip import CLIPTokenizer

FACTOR = 1.1
MAX_EMPHASIS = 8


class TextSegment:
    text: str
    weight: Optional[float] = None

    def __init__(self, text: str) -> None:
        self.text = text
        self.weight = None

        pattern = r"\((.+?)\):([+-]?\d+(?:.\d+)?)"
        match = re.match(pattern, text)
        if not match:
            self.text = text
        else:
            self.text = match.group(1)
            self.weight = float(match.group(2))

    def __repr__(self) -> str:
        return f'{self.__class__.__qualname__}(text="{self.text}"; weight={self.weight})'


class TensorAndWeight(NamedTuple):
    tensor: Tensor
    weight: Tensor


class ClipEmbedding:
    """Convert a text to embeddings using a CLIP model."""

    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel

    def __init__(
        self, tokenizer_path: str | Path, text_encoder_path: str | Path,
    ) -> None:
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

        text = expand_delimiters(text)
        text = add_split_maker4emphasis(text)

        return text

    @lru_cache(maxsize=None)
    def get_ids(self, text: str) -> TensorAndWeight:
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
        n = 77 - len(ids) - 2
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
        """Creates an embedding for a text and cache it."""

        ids, weight = self.get_ids(text)
        emb = self.text_encoder(ids)[0]

        return TensorAndWeight(emb, weight)

    def __call__(
        self,
        text: str | list[str] = "",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Creates embeddings for a text and send to the correct device/dtype."""

        if isinstance(text, str):
            emb, weight = self.get_embedding(text)
        else:
            values = [self.get_embedding(t) for t in text]
            emb = torch.cat([v.tensor for v in values])
            weight = torch.cat([v.weight for v in values])

        emb = emb.to(device=device, dtype=dtype, non_blocking=True)

        if weight.diff(1).any():
            weight = weight.to(device=device, dtype=dtype, non_blocking=True)

            return emb, weight

        return emb, None


def expand_delimiters(text: str) -> str:
    """
    replace (^n ... )^n with ( ... ):factor^n
    replace [^n ... ]^n with ( ... ):factor^-n
    """

    delimiters = [
        (l * k, r * k, k * sign)
        for k in range(MAX_EMPHASIS, 0, -1)
        for (l, r), sign in zip(((r"\(", r"\)"), (r"\[", r"\]")), (1, -1))
    ]
    avoid = r"\(\)\[\]\\"
    for l, r, n in delimiters:
        pattern = f"{l}([^{avoid}]+?){r}([^:])"
        repl = f"(\\1):{FACTOR**n:.4f}\\2"
        text = re.sub(pattern, repl, text)

    # recover back parantheses and brackets
    text = text.replace(r"\(", "(").replace(r"\)", ")")
    text = text.replace(r"\[", "[").replace(r"\]", "]")

    return text


def add_split_maker4emphasis(text: str) -> str:
    pattern = r"(\(.+?\):[+-]?\d+(?:.\d+)?)"
    text = re.sub(pattern, r"⏎\1⏎", text)

    return text
