from torch.nn import (
    Module,
    Linear,
    Sigmoid,
    AdaptiveAvgPool2d,
    AdaptiveMaxPool2d,
    ReLU,
    Sequential,
)
from torch import device as tensor_device, dtype as tensor_dtype, Tensor

from typing import Optional


class ChannelAttention(Module):
    class Mlp(Sequential):
        def __init__(
            self,
            channels: int,
            bottleneck: int = 16,
            device: Optional[tensor_device | str] = None,
            dtype: Optional[tensor_dtype] = None,
        ) -> None:
            super().__init__(
                Linear(
                    in_features=channels,
                    out_features=channels // bottleneck,
                    device=device,
                    dtype=dtype,
                ),
                ReLU(),
                Linear(
                    in_features=channels // bottleneck,
                    out_features=channels,
                    device=device,
                    dtype=dtype,
                ),
            )

    def __init__(
        self,
        channels: int,
        bottleneck: int = 16,
        device: Optional[tensor_device | str] = None,
        dtype: Optional[tensor_dtype] = None,
    ) -> None:
        super().__init__()
        self.__average_pool = AdaptiveAvgPool2d(output_size=1)
        self.__max_pool = AdaptiveMaxPool2d(output_size=1)
        self.__mlp = self.Mlp(
            channels=channels, bottleneck=bottleneck, device=device, dtype=dtype
        )
        self.__activation = Sigmoid()

    def __calculate_attention(self, tensor: Tensor) -> Tensor:
        # max pool [B, C, H, W] -> [B, C, 1, 1] -|flatten|-> [B, C]
        max_attn = self.__mlp(self.__max_pool(tensor).flatten(1))
        avg_attn = self.__mlp(self.__average_pool(tensor).flatten(1))
        return self.__activation(max_attn + avg_attn)

    def forward(self, tensor: Tensor) -> Tensor:
        return tensor * self.__calculate_attention(tensor).view(
            *tensor.shape[:-2], 1, 1
        )
