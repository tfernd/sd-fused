# Stable-Diffusion + DreamBooth + Better CODE!

## Introduction

So you want to learn more about stable-diffusion, how it works, to dive deep in its implementation but the original source its too complicated?
Or you just want a nice API to use SD?
Or you wrongly clicked this repo when you accually intented to clip on the something else?

No fear, you came to the right place!

## How and why?

The original codebase [diffusers](https://github.com/huggingface/diffusers) is a bit too overly-complicated.
The code is too long, the classes and functions are not separated into their files, it has many bad practices in terms of typing (or the lack of typing at all in some parts), and also legacy python 2 practices that are not modern anymore.
This makes reading the code to understand how SD works very hard and the fact that VS-code screams at me with red errors everywhere does not help any bit.

To fix these issues and the many bugs that come with such implementation, I decided to re-code SD from the ground up with [diffusers](https://github.com/huggingface/diffusers) in mind for HEAVY "inspiration".
Now everything has its own file, things are pretty, well-coded (wink), and might serve as a good guide for anyone trying to implement such a system. Try reading the implementation of DDIM scheduler and compare it with the original one (85 lines of code instead of 305)

## To come
DreamBooth training of SD and memory efficient attention. Google Colab notebook and more.