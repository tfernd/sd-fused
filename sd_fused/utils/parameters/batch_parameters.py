from __future__ import annotations

from .parameters import Parameters


def batch_parameters(
    groups: list[list[Parameters]],
    batch_size: int,
) -> list[list[Parameters]]:
    batched_parameters: list[list[Parameters]] = []
    for group in groups:
        for i in range(0, len(group), batch_size):
            s = slice(i, min(i + batch_size, len(group)))

            batched_parameters.append(group[s])

    return batched_parameters
