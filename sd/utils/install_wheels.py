from __future__ import annotations

import torch
import pip

# from https://github.com/TheLastBen/fast-stable-diffusion
WHEELS = {
    gpu: f"https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/{gpu}/xformers-0.0.13.dev0-py3-none-any.whl"
    for gpu in ["T4", "P100", "V100", "A100"]
}
FLASH_ATTENTION = False


def install_wheels() -> None:
    global FLASH_ATTENTION

    gpu = torch.cuda.get_device_name()

    if gpu in WHEELS:
        url = WHEELS[gpu]
        pip.main(["install", url])
        FLASH_ATTENTION = True
    else:
        print("GPU not supported yet.")


def has_flash_attention() -> bool:
    global FLASH_ATTENTION

    return FLASH_ATTENTION
