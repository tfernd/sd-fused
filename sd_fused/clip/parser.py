from __future__ import annotations

import re

FACTOR = 1.1
MAX_EMPHASIS = 8


def add_delimiter4words(text: str) -> str:
    """Replaces `word:weight` -> `(word):weight`."""

    text = re.sub(r"(\w+):([+-]?\d+(?:.\d+)?)", r"(\1):\2", text)

    return text


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


def add_split_maker4emphasis(text: str) -> str:
    """add ⏎ to the begginig and end of (..):value"""

    pattern = r"(\(.+?\):[+-]?\d+(?:.\d+)?)"
    text = re.sub(pattern, r"⏎\1⏎", text)

    return text


def prompt_choices(text: str) -> list[str]:
    """Create a set of prompt word-choices from
    `{word 1 | another word | yet another one}`"""

    pattern = re.compile(r"{([^{}]+)}")

    temp: list[str] = [text]
    texts: list[str] = []

    while len(temp) != 0:
        text = temp.pop()

        match = pattern.search(text)
        if match is not None:
            start, end = match.span()

            choices = match.group(1).split("|")
            assert len(choices) > 1

            for choice in choices:
                choice = choice.strip()
                new_text = "".join([text[:start], choice, text[end:]])
                temp.append(new_text.strip())
        else:
            texts.append(text)

    return texts[::-1]
