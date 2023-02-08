from __future__ import annotations
from typing import NoReturn, Generator

from .module import Module
from .sequential import Sequential


class ModuleList(Sequential):
    def __init__(self, *layers: Module) -> None:
        super().__init__()

        self.layers = list(layers)

    def __call__(self) -> NoReturn:
        raise RuntimeError("ModuleList is not callable!")

    def append(self, layer: Module) -> None:
        self.layers.append(layer)

    def __iter__(self) -> Generator[Module, None, None]:
        for layer in self.layers:
            yield layer
