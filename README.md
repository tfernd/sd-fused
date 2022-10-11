# Stable-Diffusion + Fused CUDA kernels = fun

## Introduction

So you want to learn more about stable-diffusion (SD), how it works, to dive deep in its implementation but the original source its too complicated?
Or you just want a nice API to use SD?
Or you wrongly clicked this repo when you accually intented to click on the something else?

No fear, you came to the right place!

## News

Model-mixing is great art of mixing different SD models trained by DreamBooth (DB) to get a new model that can do multiple things at the same time (in theory).
Or perhaps you overtrainned on DB and want to dial back a bit (to be tested).?

```python
pipeline.unet_scale('.pretrained/other-model', scale=0.9)
```

## How and why?

The original codebase [diffusers](https://github.com/huggingface/diffusers) is a bit too overly-complicated. The code is too long, the classes and functions are not separated into their files, it has many bad practices in terms of typing (or the lack of typing at all in some parts), and also legacy python 2 practices that are not modern anymore. This makes reading the code to understand how SD works very hard and the fact that VS-code screams at me with red errors everywhere does not help any bit (due to lack of typing).

To fix these issues and the many bugs that come with such implementation, I decided to re-code SD from the ground up with [diffusers](https://github.com/huggingface/diffusers) in mind for HEAVY "inspiration". Now everything has its own file, things are pretty, well-coded (wink), and might serve as a good guide for anyone trying to implement such a system. Try reading the implementation of [DDIM scheduler](https://github.com/tfernd/sd/blob/master/sd/scheduler/ddim.py) and compare it with the [original](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py) one (85 lines of code instead of 305).

This new re-factoring allow us to improve upen SD in a easier manner without ned to hijack the original code, and doing other cumbersome tricks.

Additionally, a huge speed-up can be done by manually fusing some CUDA kernels (work in progress.)

## To come
DreamBooth training of SD and memory efficient attention.
Google Colab notebook and more.