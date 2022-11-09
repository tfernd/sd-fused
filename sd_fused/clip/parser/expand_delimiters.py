from __future__ import annotations

import re

FACTOR = 1.1
MAX_EMPHASIS = 8


def expand_delimiters(prompt: str) -> str:
    """Replace `(^n ... )^n` with `( ... ):factor^n`"""

    delimiters = [(r"\(" * repeat, r"\)" * repeat, repeat) for repeat in range(MAX_EMPHASIS, 0, -1)]

    avoid = r"\(\)"
    for left, right, repeat in delimiters:
        pattern = f"{left}([^{avoid}]+?){right}([^:]|$)"
        repl = f"(\\1):{FACTOR**repeat:.4f}\\2"
        prompt = re.sub(pattern, repl, prompt)

    # recover back parantheses and brackets
    prompt = prompt.replace(r"\(", "(").replace(r"\)", ")")
    prompt = prompt.replace(r"\[", "[").replace(r"\]", "]")

    return prompt
