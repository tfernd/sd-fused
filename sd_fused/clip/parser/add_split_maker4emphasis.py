from __future__ import annotations

import re


def add_split_maker4emphasis(text: str) -> str:
    """add ⏎ to the begginig and end of (..):value"""

    pattern = r"(\(.+?\):[+-]?\d+(?:.\d+)?)"
    text = re.sub(pattern, r"⏎\1⏎", text)

    return text


def split_text(text: str) -> list[str]:
    return text.split("⏎")
