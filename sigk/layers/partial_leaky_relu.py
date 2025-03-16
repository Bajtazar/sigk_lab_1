from torch.nn import LeakyReLU
from torch import Tensor


class PartialLeakyReLU(LeakyReLU):
    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        return super().forward(tensor * mask), mask
