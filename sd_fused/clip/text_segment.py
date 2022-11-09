from __future__ import annotations

import re


class TextSegment:
    """Split `(prompt):weight` for parsing."""

    prompt: str
    weight: float = 1

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt

        pattern = r"\((.+?)\):([+-]?\d+(?:.\d+)?)"
        match = re.match(pattern, prompt)
        if match:
            self.prompt = match.group(1)
            self.weight = float(match.group(2))

    def __repr__(self) -> str:
        return f'{self.__class__.__qualname__}(text="{self.prompt}"; weight={self.weight})'
