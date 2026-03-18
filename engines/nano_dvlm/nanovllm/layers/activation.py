import torch
from torch import nn
import torch.nn.functional as F
from liger_kernel.ops.swiglu import LigerSiLUMulFunction


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return LigerSiLUMulFunction.apply(x, y)
