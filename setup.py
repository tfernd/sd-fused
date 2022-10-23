from distutils.core import setup
from setuptools import find_packages

from pathlib import Path

packages = list(set(str(p.parent) for p in Path('sd_fused').rglob('*.py')))

with open("./requirements.txt") as handle:
    requirements = handle.read().split()

setup(
    name="sd-fused",
    version="0.3.2",
    description="Stable-Diffusion + Fused CUDA kernels",
    author="Thales Fernandes",
    author_email="thalesfdfernandes@gmail.com",
    url="https://github.com/tfernd/sd-fused",
    python_requires=">=3.7",  # TODO Less?
    # packages=find_packages(exclude=["/cuda"]),
    packages=packages,
    install_requires=requirements,
)

