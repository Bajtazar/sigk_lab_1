from torch.nn import Module, Conv2d, AdaptiveMaxPool1d
from torch import device as tensor_device, dtype as tensor_dtype, Tensor, cat

from typing import Optional


class SpatialAttention(Module):
    def __init__(
        self,
        kernel_size: int = 7,
        padding: int = 3,
        device: Optional[tensor_device | str] = None,
        dtype: Optional[tensor_dtype] = None,
    ) -> None:
        super().__init__()
        self.__avg_pool = AdaptiveMaxPool1d(output_size=1)
        self.__max_pool = AdaptiveMaxPool1d(output_size=1)
        self.__conv = Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            device=device,
            dtype=dtype,
        )

    def __calculate_attention(self, tensor: Tensor) -> Tensor:
        # [B, C, H, W] -> [B, H, W, C]
        b, c, h, w = tensor.shape
        tensor = tensor.movedim(1, -1).view(b, h * w, c)
        max_pool = self.__max_pool(tensor).movedim(-1, 1).view(b, 1, h, w)
        avg_pool = self.__avg_pool(tensor).movedim(-1, 1).view(b, 1, h, w)
        connected = cat([max_pool, avg_pool], dim=1)  # [B, 2, H, W]
        return self.__conv(connected)  # [B, 1, H, W]

    def forward(self, tensor: Tensor) -> Tensor:
        attention = self.__calculate_attention(tensor)
        return tensor * attention
