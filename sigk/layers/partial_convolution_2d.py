from torch.nn import Conv2d
from torch import ones_like, tensor, dtype as tensor_dtype, device as tensor_device

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
            "mask_norm_coeff",
            tensor([prod(self.weight.shape[1:])], dtype=dtype, device=device),
        )
