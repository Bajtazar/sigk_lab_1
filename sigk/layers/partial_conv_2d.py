from torch.nn import Conv2d
from torch import (
    ones_like,
    tensor,
    dtype as tensor_dtype,
    device as tensor_device,
    Tensor,
    no_grad,
)

from numpy import prod

from typing import Optional


class PartialConv2d(Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] | int,
        stride: tuple[int, int] | int = 1,
        padding: tuple[int, int] | int = 0,
        dilation: tuple[int, int] | int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Optional[tensor_device] = None,
        dtype: Optional[tensor_dtype] = None,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.register_buffer("mask_coeffs", ones_like(self.weight))
        self.register_buffer(
            "conv_window_size",
            tensor([prod(self.weight.shape[1:])], dtype=dtype, device=device),
        )
        self.__epsilon = epsilon
        self.bias.data = self.bias.data.view(1, self.bias.shape[0], 1, 1)

    @property
    def epsilon(self) -> float:
        return self.__epsilon

    @no_grad
    def __calculate_current_mask(self, mask: Tensor) -> tuple[Tensor, float]:
        current_mask = self._conv_forward(mask, self.mask_coeffs, bias=None)
        ratio = self.conv_window_size / (current_mask + self.epsilon)
        current_mask = current_mask.clamp(min=0, max=1)
        return current_mask, current_mask * ratio

    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        mask, ratio = self.__calculate_current_mask(mask)

        out = self._conv_forward(tensor * mask, self.weight, bias=None)
        return (out * ratio) + self.bias, mask
