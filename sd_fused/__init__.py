# type: ignore
import torch

from .app import StableDiffusion

# ? faster?
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
