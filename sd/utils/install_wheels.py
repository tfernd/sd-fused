from __future__ import annotations

from subprocess import getoutput
import pip

from ..layers.attention import FLASH_ATTENTION

# from https://github.com/TheLastBen/fast-stable-diffusion
WHEELS = {
    gpu: f"https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/{gpu}/xformers-0.0.13.dev0-py3-none-any.whl"
    for gpu in ["T4", "P100", "V100", "A100"]
}


def install_wheels() -> None:
    global FLASH_ATTENTION

    s = getoutput("nvidia-smi")
    for gpu, url in WHEELS.items():
        if s in gpu:  # TODO not the best way to check...
            # %pip install $url
            pip.main(["install", url])
            FLASH_ATTENTION = True
            break
    else:
        FLASH_ATTENTION = False
        print("GPU not supported yet.")
