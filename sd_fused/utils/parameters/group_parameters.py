from __future__ import annotations

from .parameters import Parameters


def group_parameters(parameters: list[Parameters]) -> list[list[Parameters]]:
    """Group parameters that can share a batch."""

    groups: list[list[Parameters]] = []
    for parameter in parameters:
        if len(groups) == 0:
            groups.append([parameter])
            continue

        can_share = False
        for group in groups:
            can_share = True
            for other_parameter in group:
                can_share &= parameter.can_share_batch(other_parameter)

            if can_share:
                group.append(parameter)
                break

        if not can_share:
            groups.append([parameter])

    return groups
