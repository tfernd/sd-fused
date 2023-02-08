from __future__ import annotations

import re


def add_split_maker_for_emphasis(prompt: str) -> str:
    """
    Adds the split marker `⏎` to the beginning and end of the emphasized `(...):weight` pairs in the given string.

    Example:
        `"(words1):1.0 (words2):2.0" -> "⏎(words1):1.0⏎ (words2):2.0⏎"`
    """

    pattern = r"(\(.+?\):[+-]?\d+(?:.\d+)?)"

    return re.sub(pattern, r"⏎\1⏎", prompt)


def split_prompt_into_individual_segments(prompt: str) -> list[str]:
    """
    Splits the given string at the split marker `⏎` to create a list of individual segments.

    Example:
        `"⏎(word1):1.0⏎ (word2):2.0⏎" -> ["(word1):1.0", "(word2):2.0"]`
    """

    return prompt.split("⏎")
