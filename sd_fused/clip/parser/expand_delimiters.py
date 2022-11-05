from __future__ import annotations

import re

FACTOR = 1.1
MAX_EMPHASIS = 8


def expand_delimiters(text: str) -> str:
    """
    replace `(^n ... )^n` with `( ... ):factor^n`
    replace `[^n ... ]^n` with `( ... ):factor^-n`
    """

    delimiters = [
        (left * repeat, right * repeat, repeat * sign)
        for repeat in range(MAX_EMPHASIS, 0, -1)
        for left, right, sign in ((r"\(", r"\)", 1), (r"\[", r"\]", 1))
    ]

    avoid = r"\(\)\[\]\\"
    for left, right, signed_repeat in delimiters:
        pattern = f"{left}([^{avoid}]+?){right}([^:])"
        repl = f"(\\1):{FACTOR**signed_repeat:.4f}\\2"
        text = re.sub(pattern, repl, text)

    # recover back parantheses and brackets
    text = text.replace(r"\(", "(").replace(r"\)", ")")
    text = text.replace(r"\[", "[").replace(r"\]", "]")

    return text
