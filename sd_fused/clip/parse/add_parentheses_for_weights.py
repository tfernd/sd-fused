from __future__ import annotations

import re


def add_parentheses_for_weights(prompt: str) -> str:
    """
    Adds parentheses around the word in the `word:weight` pairs in the given string.

    Example:
        `"word1:1 word2:2.0" -> "(word1):1 (word2):2.0"`
    """

    return re.sub(r"(\b[\w-]+\b):([+-]?\d+(?:\.\d+)?)", r"(\1):\2", prompt)
