from __future__ import annotations

import re


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
