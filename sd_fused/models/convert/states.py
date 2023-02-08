from __future__ import annotations

import re

from torch import Tensor

from ...layers.base import Parameter


def replace_state(
    old_state: dict[str, Tensor],
    replacements: list[tuple[str, str]],
) -> dict[str, Tensor]:
    """Replace the state-dict with new keys"""

    state: dict[str, Tensor] = {}
    for key in old_state.keys():
        new_key = key
        for old, new in replacements:
            new_key = re.sub(old, new, new_key)

        state[new_key] = old_state[key]

    return state


def debug_state_replacements(
    state: dict[str, Tensor] | dict[str, Parameter],
    replaced_state: dict[str, Tensor],
) -> None:
    good_keys = set(state.keys())
    replaced_keys = set(replaced_state.keys())

    delta = good_keys - replaced_keys
    if len(delta) != 0:
        print("miss replacing some keys")
        print("=" * 32)
        for key in delta:
            print(key)

    delta = replaced_keys - good_keys
    if len(delta) != 0:
        print("wrongly replaced some keys")
        print("=" * 32)
        for key in replaced_keys - good_keys:
            print(key)
