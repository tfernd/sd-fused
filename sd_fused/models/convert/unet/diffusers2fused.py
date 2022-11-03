from __future__ import annotations

from torch import Tensor

from ..states import replace_state


def diffusers2fused(old_state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert a diffusers checkpoint into a sd-fused one for the unet."""

    return replace_state(old_state, REPLACEMENTS)


# fmt: off
# diffusers to sd-fused replacements
REPLACEMENTS: list[tuple[str, str]] = [
    # Cross-attention
    (r"transformer_blocks.(\d).norm([12]).(weight|bias)", r"transformer_blocks.\1.attn\2.norm.\3"),
    ## FeedForward (norm)
    (r"transformer_blocks.(\d).norm3.(weight|bias)", r"transformer_blocks.\1.ff.0.\2"),
    ## FeedForward (geglu)
    (r"ff.net.0.proj.(weight|bias)", r"ff.1.proj.\1"),
    ## FeedForward-Linear
    (r"ff.net.2.(weight|bias)", r"ff.2.\1"),
    # up/down samplers
    (r"(up|down)samplers.0", r"\1sampler"),
    # CrossAttention projection
    (r"to_out.0.", r"to_out."),
    # TimeEmbedding
    (r"time_embedding.linear_1.(weight|bias)", r"time_embedding.0.\1"),
    (r"time_embedding.linear_2.(weight|bias)", r"time_embedding.2.\1"),
    # resnet-blocks pre/post-process
    (r"resnets.(\d).norm1.(bias|weight)", r"resnets.\1.pre_process.0.\2"),
    (r"resnets.(\d).conv1.(bias|weight)", r"resnets.\1.pre_process.2.\2"),
    (r"resnets.(\d).norm2.(bias|weight)", r"resnets.\1.post_process.0.\2"),
    (r"resnets.(\d).conv2.(bias|weight)", r"resnets.\1.post_process.2.\2"),
    # resnet-time-embedding
    (r"time_emb_proj.(bias|weight)", r"time_emb_proj.1.\1"),
    # spatial transformer fused
    (r"attentions.(\d).norm.(bias|weight)", r"attentions.\1.proj_in.0.\2"),
    (r"attentions.(\d).proj_in.(bias|weight)", r"attentions.\1.proj_in.1.\2"),
    # post-processing
    (r"conv_norm_out.(bias|weight)", r"post_process.0.\1"),
    (r"conv_out.(bias|weight)", r"post_process.2.\1"),
    # pre-processing
    (r'conv_in.(bias|weight)', r'pre_process.\1')
]
