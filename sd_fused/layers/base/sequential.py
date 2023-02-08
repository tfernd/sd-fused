from __future__ import annotations

from torch import Tensor

from .module import Module
from .parameter import Parameter


class Sequential(Module):
    layers: list[Module]

    def __init__(self, *layers: Module) -> None:
        super().__init__()

        self.layers = list(layers)

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)

        return x

    def state_dict(self, *, skip_buffer: bool = True) -> dict[str, Parameter]:
        params: dict[str, Parameter] = {}
        for index, layer in enumerate(self.layers):
            for name, parameter in layer.state_dict(skip_buffer=skip_buffer).items():
                params[f"{index}.{name}"] = parameter

        return params

    def named_modules(self) -> dict[str, Module]:
        modules: dict[str, Module] = {}
        for index, layer in enumerate(self.layers):
            for name, module in layer.named_modules().items():
                name = str(index) if name == "" else f"{index}.{name}"
                modules[name] = module

        return modules
