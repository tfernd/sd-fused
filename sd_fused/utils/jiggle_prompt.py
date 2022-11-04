from __future__ import annotations

import re
import random

# TODO experimental
def jiggle_prompt(prompt: str, vmin: float = 0, vmax: float = 1) -> str:
    assert ":" not in prompt

    words: list[str] = []
    for word in re.split(r"(\W)", prompt):
        if re.match(r"\w", word):
            v = random.uniform(vmin, vmax)
            word = f"{word}:{v:.4}"
        words.append(word)

    return "".join(words)
