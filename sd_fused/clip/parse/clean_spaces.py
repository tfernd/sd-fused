from __future__ import annotations

import re


def clean_spaces(prompt: str) -> str:
    """Removes extra spaces and newline characters from the given string."""

    prompt = prompt.replace("\n", " ")
    prompt = re.sub(r"\s+", r" ", prompt)

    return prompt.strip()
