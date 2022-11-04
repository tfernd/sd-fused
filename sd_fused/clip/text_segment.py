from __future__ import annotations
from typing import Optional

import re


class TextSegment:
    """Split `(words):weight` for parsing."""

    text: str
    weight: float = 1

    def __init__(self, text: str) -> None:
        self.text = text

        pattern = r"\((.+?)\):([+-]?\d+(?:.\d+)?)"
        match = re.match(pattern, text)
        if match:
            self.text = match.group(1)
            self.weight = float(match.group(2))

    def __repr__(self) -> str:
        return f'{self.__class__.__qualname__}(text="{self.text}"; weight={self.weight})'
