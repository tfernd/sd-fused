from __future__ import annotations

import torch
import pip

# from https://github.com/TheLastBen/fast-stable-diffusion
# TODO get full names as seen by `get_device_name`
WHEELS = {
    "A100": "https://github.com/TheLastBen/fast-stable-diffusion/blob/main/precompiled/A100/xformers-0.0.13.dev0-py3-none-any.whl",
    "V100": "https://github.com/TheLastBen/fast-stable-diffusion/blob/main/precompiled/V100/xformers-0.0.13.dev0-py3-none-any.whl",
    "P100": "https://github.com/TheLastBen/fast-stable-diffusion/blob/main/precompiled/P100/xformers-0.0.13.dev0-py3-none-any.whl",
    "Tesla T4": "https://github.com/TheLastBen/fast-stable-diffusion/blob/main/precompiled/T4/xformers-0.0.13.dev0-py3-none-any.whl",
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
        print(f"GPU ({gpu}) not supported yet.")


def has_flash_attention() -> bool:
    global FLASH_ATTENTION

    return FLASH_ATTENTION
