from __future__ import annotations

import re


def clean_spaces(prompt: str) -> str:
    """Clean-up spaces/return characters."""

    prompt = prompt.replace("\n", " ")
    prompt = re.sub(r"[ ]+", r" ", prompt)
    prompt = prompt.strip()

    return prompt
