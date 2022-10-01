from __future__ import annotations

import torch
import torch.nn as nn

# TODO rename
class Ops(nn.Module):
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
