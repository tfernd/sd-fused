# type: ignore
from distutils.core import setup

from pathlib import Path

from sd_fused import StableDiffusion

packages = list(set(str(p.parent) for p in Path("sd_fused").rglob("*.py")))

with open("./requirements.txt") as handle:
    requirements = [l.strip() for l in handle.read().split()]


setup(
    name="sd-fused",
    version=StableDiffusion.version,
    description="Stable-Diffusion + Fused CUDA kernels",
    author="Thales Fernandes",
    author_email="thalesfdfernandes@gmail.com",
    url="https://github.com/tfernd/sd-fused",
    python_requires=">=3.7",  # TODO Less?
    # packages=find_packages(exclude=["/cuda"]),
    packages=packages,
    install_requires=requirements,
)
