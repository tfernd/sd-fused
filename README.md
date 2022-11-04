# Stable-Diffusion + Fused CUDA kernels = FUN!

## For the inpatients:

### Emphasis

Using the notation `(a few words):weight` you can give emphasis (high number), take out emphasis (small number), or even avoid the subject (negative number).
The words (tokens) inside the parentheses are given a weight that is passed down to the attention calculation, enhancing, attenuating, or negative the attention to the given token.

Below is a small test where the word `cyberpunk` is given a different emphasis.

```python
prompts = [
    f"portrait, woman, cyberpunk:{t}, digital art, detailed, epic, beautiful"
    for t in np.linspace(-1.2, 4.2, 32)
]
out = pipeline.generate(
    prompt=prompts,
    steps=24,
    scale=11,
    height=512,
    width=512,
    seed=1658926406,
    eta=0.6,
    show=True,
    batch_size=8,
)
```

![portrait, woman, cyberpunk:{t}, digital art, detailed, epic, beautiful](assets/animations/animation-1/animation.gif)

<!-- <video src="./assets/animations/animation-1/animation.avi" controls="controls" muted="muted" class="d-block rounded-bottom-2 width-fit"> -->


### Batched sweep

Any input parameter can be passed as a list for sweeping, where any multiple combinations of sweeps are allowed.
For example:

```python
out = pipeline.generate(
    prompt="portrait, woman, cyberpunk, digital art, detailed, epic, beautiful",
    steps=26,
    height=512,
    width=512,
    seed=1331366415,
    eta=np.linspace(-1, 1, 64),
    show=True,
    batch_size=8,
)
```

![portrait, woman, cyberpunk, digital art, detailed, epic, beautiful](assets/animations/animation-2/animation.gif)


## Introduction

This is a re-written implementation of Stable-Diffusion (SD) based on the original [diffusers](https://github.com/huggingface/diffusers) and [stable-diffusion](https://github.com/CompVis/stable-diffusion) repositories (all kudos for the original programmers).

The goal of this reimplementation is to make it clearer, more readable, and more upgradable code that is easy to read and modify.
Unfortunately, the original code is very difficult to read due to the lack of proper typing, variable naming, and other factors.

## Kernel fusion

This is an ongoing project to fuse as many layers as possible to make it more memory friendly and faster.

## Installation

```bash
pip install -U git+https://github.com/tfernd/sd-fused
```

## Text2Image generation

Base code for text-to-image generation.

```python
from IPython.display import display
from sd_fused.app import StableDiffusion

# Assuming you downloaded SD and put it in the folder below
pipeline = StableDiffusion('.pretrained/stable-diffusion')

# If you have a GPU with 3-4 Gb, use the line below
# pipeline.set_low_ram().half_weights().cuda()
pipeline.half().cuda()
pipeline.split_attention(cross_attention_chunks=1)
# if you have xformers installed, use the line below
# pipeline.flash_attention()

out = pipeline.generate(
    prompt='portrait of zombie, digital art, detailed, artistic',
    negative_prompt='old man',
    steps=28,
    scale=11,
    height=512,
    width=512,
    seed=42,
    show=True
)
```

![portrait of zombie, digital art, detailed, artistic](assets/text2img.png)


<!-- ## TODO list -->