from __future__ import annotations

import re

from ...utils.diverse import to_list
from ...utils.typing import MaybeIterable


def prompt_choices(prompt: str) -> list[str]:
    """Create a set of prompt word-choices from
    `[word 1 | another word | yet another one]`
    """

    pattern = re.compile(r"\[([^\[\]]+)\]")

    temp: list[str] = [prompt]
    prompts: list[str] = []

    while len(temp) != 0:
        prompt = temp.pop()

        match = pattern.search(prompt)
        if match is not None:
            start, end = match.span()

            choices = match.group(1).split("|")
            assert len(choices) > 1

            for choice in choices:
                choice = choice.strip()
                new_text = "".join([prompt[:start], choice, prompt[end:]])
                temp.append(new_text.strip())
        else:
            prompts.append(prompt)

    return prompts[::-1]


def prompts_choices(prompts: MaybeIterable[str]) -> list[str]:
    """Create a set of prompt word-choices from
    `[word 1 | another word | yet another one]`"""

    return [choice for prompt in to_list(prompts) for choice in prompt_choices(prompt)]
