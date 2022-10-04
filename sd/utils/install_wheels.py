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

    device_name = torch.cuda.get_device_name()
    for gpu, url in WHEELS.items():
        if gpu == device_name:
            pip.main(["install", url])
            FLASH_ATTENTION = True
            break
    else:
        print("GPU not supported yet.")
