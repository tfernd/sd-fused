# Stable-Diffusion with very low memory (3.3 Gb)

Yes. You read it correctly. 512x512 image generation with about 3 Gb generation without CPU/GPU swapping!

Generation on a Quadro T2000 (very very bad graphics card) takes about 49s (with 28 steps) and 8s in a Testa T4.

<!-- TODO compare speeds! -->

## How and why?

The original codebase [diffusers](https://github.com/huggingface/diffusers) is a bit too overly-complicated done.
Most users are only interested in SD, so it makes sense to have a repository that is only focused on it.
Additionally, VS-code linting screams at the code due to the absenscen of proper typing with the additionaly python 2 bad practices in coding.
Some files have thousands of lines with multiple different things. Some functions are very big and do multiple things.

To fix these issues and the many bugs that comes with such implementation, I decided to re-code SD from the ground up with [diffusers](https://github.com/huggingface/diffusers) in mind for HEAVY "inspiration".
Now everything has its own file, things are pretty, well-coded (wink), and might serve for a good guide for anyone trying to implement such a system.

## To come

DreamBooth training of SD and 