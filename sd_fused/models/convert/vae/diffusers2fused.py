from __future__ import annotations

from torch import Tensor

from ..states import replace_state

# ! Tensor/Parameter?
def diffusers2fused(old_state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert a diffusers checkpoint into a sd-fused one for the AutoencoderKL."""

    return replace_state(old_state, REPLACEMENTS)


# fmt: off
# diffusers to sd-fused replacements
REPLACEMENTS: list[tuple[str, str]] = [
    # up/down samplers
    (r"(up|down)samplers.0", r"\1sampler"),
    # post_process
    (r"(decoder|encoder).conv_norm_out.(bias|weight)", r"\1.post_process.0.\2"),
    (r"(decoder|encoder).conv_out.(bias|weight)", r"\1.post_process.2.\2"),
    # resnet-blocks pre/post-process
    (r"resnets.(\d).norm1.(bias|weight)", r"resnets.\1.pre_process.0.\2"),
    (r"resnets.(\d).conv1.(bias|weight)", r"resnets.\1.pre_process.2.\2"),
    (r"resnets.(\d).norm2.(bias|weight)", r"resnets.\1.post_process.0.\2"),
    (r"resnets.(\d).conv2.(bias|weight)", r"resnets.\1.post_process.2.\2"),
    # pre-processing
    (r'(encoder|decoder).conv_in.(bias|weight)', r'\1.pre_process.\2')
]
