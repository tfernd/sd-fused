from __future__ import annotations
from typing import Generator, Generic, NoReturn

import torch.nn as nn

from ...utils.typing import TypeVar, TypeVarTuple, Unpack
from .module import Module


T = TypeVar("T", bound=Module)
Ts = TypeVarTuple("Ts")  # ? bound


class _ModuleSequence(Module):
    layers: tuple[Module, ...] | list[Module]

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
                name = str(index) if key == "" else f"{index}.{key}"
                modules[name] = value

        return modules

    def __call__(self) -> NoReturn:
        raise ValueError(f"{self.__class__.__qualname__} is not callable.")


class ModuleList(_ModuleSequence, Generic[T]):
    layers: list[T]

    def __init__(self, *layers: T) -> None:
        self.layers = list(layers)

    def append(self, layer: T) -> None:
        self.layers.append(layer)

    def __iter__(self) -> Generator[T, None, None]:
        for layer in self.layers:
            yield layer


# # ! only for debug
# class ModuleTuple(Module, Generic[Unpack[Ts]]):
#     layers: tuple[Unpack[Ts]]

#     def __init__(self, *layers: Unpack[Ts]) -> None:
#         self.layers = layers

#     def __iter__(self):  # ? type?
#         for layer in self.layers:
#             yield layer
