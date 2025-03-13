from sigk.layers.spectral.spectral_conv2d import SpectralConv2d

from torch import device as tensor_device, dtype as tensor_dtype

from typing import Optional


def assert_valid_depthwise_channels(in_channels: int, out_channels: int) -> None:
    if out_channels % in_channels != 0:
        raise ValueError(
            f"Output channels ({out_channels}) has to be a multiple of the input"
            f" channels ({in_channels})"
        )


class SpectralDepthwiseConv2d(SpectralConv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: str | int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Optional[tensor_device] = None,
        dtype: Optional[tensor_dtype] = None,
    ) -> None:
        assert_valid_depthwise_channels(in_channels, out_channels)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
