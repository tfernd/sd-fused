from __future__ import annotations
from typing import Optional
from typing_extensions import Self

from abc import abstractmethod, ABC

import torch
import torch.nn as nn
from torch import Tensor

from .base import Base
from .types import Device


class Module(Base, ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> Tensor:  # ! type!
        ...

    def named_modules(self) -> dict[str, Module]:
        modules: dict[str, Module] = {}

        modules[""] = self
        for key, value in self.__dict__.items():
            if isinstance(value, Module):
                modules[key] = value

                # recursive call
                for sub_key, sub_value in value.named_modules().items():
                    if sub_key != "":
                        modules[f"{key}.{sub_key}"] = sub_value

        return modules

    def state_dict(self) -> dict[str, nn.Parameter]:
        params: dict[str, nn.Parameter] = {}
        for key, value in self.__dict__.items():
            # single parameter
            if isinstance(value, nn.Parameter):
                params[key] = value

            # recursive call
            elif isinstance(value, Module):
                for sub_key, sub_value in value.state_dict().items():
                    params[f"{key}.{sub_key}"] = sub_value

        return params

    def load_state_dict(self, state: dict[str, nn.Parameter] | dict[str, Tensor], strict: bool = False) -> Self:
        current_state = self.state_dict()
        assert len(current_state) == len(state)

        for key, value in current_state.items():
            assert key in state

            new_value = state[key].data
            if strict:
                assert new_value.shape == value.shape

            new_value = new_value.to(device=self.device, dtype=self.dtype, non_blocking=True)
            value.data = new_value

        return self

    def float(self) -> Self:
        return self.to(dtype=torch.float32)

    def half(self) -> Self:
        return self.to(dtype=torch.float16)

    def cpu(self) -> Self:
        return self.to(device=f"cpu")

    def cuda(self, index: int = 0) -> Self:
        return self.to(device=f"cuda:{index}")

    def to(self, *, device: Optional[Device] = None, dtype: Optional[torch.dtype] = None):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype

        for key, value in self.state_dict().items():
            data = value.data
            value.data = data.to(device, dtype, non_blocking=True)

        return self
