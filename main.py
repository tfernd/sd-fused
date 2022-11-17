#%%
# %load_ext autoreload
# %autoreload 2

import torch
import numpy as np

from sd_fused.app import StableDiffusion

from pathlib import Path

import torch

# pipeline = StableDiffusion(".pretrained/stable-diffusion-v1.4")
pipeline = StableDiffusion(".pretrained/stable-diffusion-v1.5")
# # pipeline = StableDiffusion(".pretrained/stable-diffusion-inpainting")

pipeline.low_ram().half_weights()
pipeline.split_attention("auto", "batch")
pipeline.tome(0.5)

# %%
out = pipeline.generate(
    prompt="portrait of a woman, cyberpunk, digital art, detailed, epic, beautiful",
    steps=32,
    height=384,
    width=384,
    # seed=2133329843,
    # eta=[-1, 1],
    show=True,
    # img='img.png',
    # mask='mask.png',
    # mode='resize-crop',
    # strength=1,
    # repeat=8,
    # share_seed=False,
)
