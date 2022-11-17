from __future__ import annotations
from typing import Generator
from typing_extensions import Self

import torch.nn as nn

from .module import Module


class ModuleList(Module):
    def __init__(self, *layers: Module) -> None:  # ! type!
        self.layers = layers

    def append(self, layer: Module) -> Self:
        self.layers = (*self.layers, layer)

        return self

    def __call__(self):
        raise ValueError("ModuleList cannot be called directly")

    def __iter__(self) -> Generator[Module, None, None]:
        for layer in self.layers:
            yield layer

    def state_dict(self) -> dict[str, nn.Parameter]:
        params: dict[str, nn.Parameter] = {}
        for index, layer in enumerate(self.layers):
            for key, value in layer.state_dict().items():
                params[f"{index}.{key}"] = value

        return params

    def named_modules(self) -> dict[str, Module]:
        modules: dict[str, Module] = {}
        for index, layer in enumerate(self.layers):
            for key, value in layer.named_modules().items():
                if key != "":
                    modules[f"{index}.{key}"] = value
                else:
                    modules[f"{index}"] = value

        return modules
