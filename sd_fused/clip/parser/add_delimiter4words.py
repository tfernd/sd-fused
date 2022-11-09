from __future__ import annotations

import re


def add_delimiter4words(prompt: str) -> str:
    """Replaces `word:weight` -> `(word):weight`."""

    prompt = re.sub(r"(\w+):([+-]?\d+(?:.\d+)?)", r"(\1):\2", prompt)

    return prompt
