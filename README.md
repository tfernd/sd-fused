# Stable-Diffusion + Fused CUDA kernels = FUN!

## Introduction

So you want to learn more about stable-diffusion (SD), how it works, to dive deep in its implementation but the original source its too complicated?
Or you just want a nice API to use SD?
Or you wrongly clicked this repo when you accually intented to click on the something else?

No fear, you came to the right place!

## How and why?

The original codebase [diffusers](https://github.com/huggingface/diffusers) is a bit too overly-complicated.
Most of us are only interested in SD and we do not care too much about other implementations.
So there is a need for a repository that is focused on only SD.

Additionally, the original codebase is too long, the classes and functions are not separated into their files, it has many bad practices in terms of typing (or the lack of typing at all in some parts), and also legacy python 2 practices that are not modern anymore.
This makes reading the code to understand how SD works very hard and the fact that VS-code screams at me with red errors everywhere does not help any bit (due to lack of typing).
As an example, they use a function `get_down_block` that has an input a string and it creases a block used in SD.
Good luck getting any typing information on that.

To fix these issues and the many bugs that come with such implementation, I decided to re-code SD from the ground up with [diffusers](https://github.com/huggingface/diffusers) in mind for HEAVY "inspiration".
Now everything has its own file, things are pretty, well-coded (wink), and might serve as a good guide for anyone trying to implement such a system.
Try reading the implementation of [DDIM scheduler](https://github.com/tfernd/sd/blob/master/sd/scheduler/ddim.py) and compare it with the [original](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py) one (85 lines of code instead of 305).

These new rewritting comes with many good features, such as renaming parameters and layers to be more uniform.
Better handling of internal states passed by up/down-blocks.
Creating new layers that will be fused latter on and better organization of layers in general.

## Kernel fusion
I'm currently writing some CUDA kernels for some layers that will be fused.
Such as Normalization+Activation+Linear/Convolution layers, and, of course, attention!
I plan on using [FlashAttention](https://github.com/HazyResearch/flash-attention) (also implemented by [xformers](https://github.com/facebookresearch/xformers)), but I'm doing my own implementation.

## News

Model-mixing is great art of mixing different SD models trained by DreamBooth (DB) to get a new model that can do multiple things at the same time (in theory).
Or perhaps you overtrainned on DB and want to dial back a bit (to be tested)?
The idea is to combine multiple models in the attempt to still retain a bit of each model.

```python
pipeline.unet_scale('.pretrained/other-model', scale=0.9)
```

## Installation
```python
!pip install -U git+https://github.com/tfernd/sd
```
