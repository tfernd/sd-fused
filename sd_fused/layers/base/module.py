from __future__ import annotations
from typing import Optional
from typing_extensions import Self

from abc import ABC, abstractmethod

from pathlib import Path
import json

import torch
from torch import Tensor

from .base import Base
from .parameter import Buffer, Parameter


class Module(Base, ABC):
    training: bool = False

    def __init__(self) -> None:
        self.eval()

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Tensor:
        ...

    def named_modules(self) -> dict[str, Module]:
        """Returns a dictionary that maps names to submodules in the module."""

        modules: dict[str, Module] = {}

        # export itself
        modules[""] = self

        for name, module in self.__dict__.items():
            if isinstance(module, Module):
                modules[name] = module

                # Recursively call named_modules() on the submodule
                for sub_name, sub_module in module.named_modules().items():
                    if sub_name != "":  # Don't re-export the submodule again
                        modules[f"{name}.{sub_name}"] = sub_module

        return modules

    def state_dict(self, *, skip_buffer: bool = True) -> dict[str, Parameter]:
        """Returns a dictionary that maps names to parameters in the module."""

        params: dict[str, Parameter] = {}
        for name, parameter_or_module in self.__dict__.items():
            # skip buffers
            if skip_buffer and isinstance(parameter_or_module, Buffer):
                continue

            # If an attribute is a parameter, add it to the dictionary
            if isinstance(parameter_or_module, Parameter):
                params[name] = parameter_or_module

            # If an attribute is a module, recursively call state_dict() on the submodule
            if isinstance(parameter_or_module, Module):
                for sub_name, sub_parameter in parameter_or_module.state_dict(
                    skip_buffer=skip_buffer
                ).items():
                    params[f"{name}.{sub_name}"] = sub_parameter

        return params

    def load_state_dict(
        self,
        state: dict[str, Parameter] | dict[str, Tensor] | str | Path,
        *,
        strict: bool = False,
    ) -> Self:
        """Loads a state dictionary into the module."""

        if isinstance(state, (str, Path)):
            state = Path(state)
            assert state.suffix == ".pt"

            w: dict[str, Tensor] = torch.load(state, map_location="cpu")
            state = w  # ! to make linter happy...

        current_state = self.state_dict()
        assert len(current_state) == len(state)

        for name, parameter in current_state.items():
            # Ensure that the names in the state dictionaries match
            assert name in state

            new_value = state[name].data
            if strict:
                assert new_value.shape == parameter.shape

            parameter.data = new_value

        # Transfer the data to the correct device and dtype
        self.to(device=self.device, dtype=self.dtype)

        return self

    @classmethod
    def from_config(cls, path: str | Path) -> Self:
        path = Path(path)
        assert path.suffix == ".json"

        with open(path, "r", encoding="UTF-8") as handle:
            kwargs = json.load(handle)

        return cls(**kwargs)

    def to(
        self,
        *,
        device: Optional[str | torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Moves the module and its parameters to the specified device and dtype."""

        super().to(device, dtype)

        for name, module in self.named_modules().items():
            if name == "":
                continue

            module.to(device=device, dtype=dtype)

        for name, parameter in self.state_dict(skip_buffer=False).items():
            if "." in name:  # sub-module
                continue

            data = parameter.data
            parameter.data = data.to(device=device, dtype=dtype, non_blocking=True)

        return self

    def eval(self) -> Self:
        self.training = False

        return self

    def train(self) -> Self:
        self.training = True

        return self
