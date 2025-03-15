from torch.nn import GELU

from torch import Tensor


class PartialGELU(GELU):
    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        return super().forward(tensor * mask), mask
