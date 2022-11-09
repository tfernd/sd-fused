from __future__ import annotations

import re


def add_split_maker4emphasis(prompt: str) -> str:
    """Add ⏎ to the begginig and end of (..):value"""

    pattern = r"(\(.+?\):[+-]?\d+(?:.\d+)?)"
    prompt = re.sub(pattern, r"⏎\1⏎", prompt)

    return prompt


def split_prompt_into_segments(prompt: str) -> list[str]:
    """Split a prompt at ⏎ to give prompt-segments."""

    return prompt.split("⏎")
