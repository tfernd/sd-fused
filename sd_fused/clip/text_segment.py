from __future__ import annotations
from typing import Optional

import re


class TextSegment:
    """Split a (words):factor for parsing."""

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
