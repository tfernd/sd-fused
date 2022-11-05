from __future__ import annotations

import re


def add_delimiter4words(text: str) -> str:
    """Replaces `word:weight` -> `(word):weight`."""

    text = re.sub(r"(\w+):([+-]?\d+(?:.\d+)?)", r"(\1):\2", text)

    return text
