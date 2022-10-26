from __future__ import annotations

import re
from torch import Tensor

# TODO make a config from sd?


def sd2diffusers(old_state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert a Stable-Diffusion checkpoint into a diffusers-one for the AutoencoderKL."""

    VAE = "first_stage_model."

    state: dict[str, Tensor] = {}
    for key in old_state.keys():
        if not key.startswith(VAE):
            continue

        new_key = key.replace(VAE, "")

        for old, new in REPLACEMENTS:
            new_key = re.sub(old, new, new_key)

        state[new_key] = old_state[key]

        # reshape attentions weights
        # needed due to conv -> linear conversion
        if re.match(
            r"(encoder|decoder).mid_block.attentions.\d.(key|query|value|proj_attn).weight",
            new_key,
        ):
            state[new_key] = state[new_key].squeeze()

    return state


# Stable-diffusion to diffusers replacements
REPLACEMENTS: list[tuple[str, str]] = [
    # reverse up-block order
    (r"up\.0", r"UP.3"),
    (r"up\.1", r"UP.2"),
    (r"up\.2", r"UP.1"),
    (r"up\.3", r"UP.0"),
    (r"UP", "up"),  # recover lower-case
    # short-cut connection
    (r"nin_shortcut", r"conv_shortcut"),
    # encoder/decoder up/down-blocks
    (
        r"(encoder|decoder)\.(down|up)\.(\d)\.block\.(\d)\.(norm\d|conv\d|conv_shortcut)\.(weight|bias)",
        r"\1.\2_blocks.\3.resnets.\4.\5.\6",
    ),
    (
        r"(encoder|decoder)\.(down|up)\.(\d)\.(downsample|upsample)\.conv\.(weight|bias)",
        r"\1.\2_blocks.\3.\4rs.0.conv.\5",
    ),
    # mid-blocks
    (r"block_1", r"block_0"),
    (r"block_2", r"block_1"),
    (
        r"(encoder|decoder)\.mid\.block_(\d)\.(norm\d|conv\d)\.(weight|bias)",
        r"\1.mid_block.resnets.\2.\3.\4",
    ),
    # mid-block attention
    (r"\.q\.", r".query."),
    (r"\.k\.", r".key."),
    (r"\.v\.", r".value."),
    (r"attn_1\.proj_out", r"attn_1.proj_attn"),
    (r"attn_1\.norm", r"attn_1.group_norm"),
    (
        r"(encoder|decoder)\.mid\.attn_1\.(group_norm|query|key|value|proj_attn)\.(weight|bias)",
        r"\1.mid_block.attentions.0.\2.\3",
    ),
    # in-out
    (r"(encoder|decoder)\.(norm_out)\.(weight|bias)", r"\1.conv_\2.\3"),
]
