from torch.nn import (
    Module,
    Sequential,
    AdaptiveAvgPool2d,
    Linear,
    ReLU,
    Sigmoid,
    Flatten,
)
from torch import Tensor, device as tensor_device, dtype as tensor_dtype

from typing import Optional


class SqueezeAndExcitationBlock(Module):
    def __init__(
        self,
        channels: int,
        bottleneck: int = 16,
        device: Optional[tensor_device | str] = None,
        dtype: Optional[tensor_dtype] = None,
    ) -> None:
        super().__init__()
        self.__channels = channels
        self.__excitation = Sequential(
            AdaptiveAvgPool2d(output_size=1),
            Flatten(start_dim=1),
            Linear(
                self.channels,
                self.channels // bottleneck,
                bias=False,
                device=device,
                dtype=dtype,
            ),
            ReLU(),
            Linear(
                self.channels // bottleneck,
                self.channels,
                bias=False,
                device=device,
                dtype=dtype,
            ),
            Sigmoid(),
        )

    @property
    def channels(self) -> int:
        return self.__channels

    def forward(self, tensor: Tensor) -> Tensor:
        return tensor * self.__excitation(tensor).view(*tensor.shape[:-2], 1, 1)
