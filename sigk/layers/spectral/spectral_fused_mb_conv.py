from sigk.layers.squeeze_and_excitaition_block import SqueezeAndExcitationBlock
from sigk.layers.spectral.spectral_conv2d import SpectralConv2d

from torch.nn import Module, Sequential
from torch import Tensor, device as tensor_device, dtype as tensor_dtype

from typing import Optional


class SpectralFusedMBConv(Module):
    def __init__(
        self,
        channels: int,
        bottleneck: int = 16,
        device: Optional[tensor_device | str] = None,
        dtype: Optional[tensor_dtype] = None,
    ) -> None:
        super().__init__()
        self.__channels = channels
        self.__fused_mb_conv = Sequential(
            SpectralConv2d(
                self.channels,
                4 * self.channels,
                kernel_size=3,
                padding=1,
                dtype=dtype,
                device=device,
            ),
            SqueezeAndExcitationBlock(
                4 * self.channels, bottleneck, dtype=dtype, device=device
            ),
            SpectralConv2d(
                4 * self.channels,
                self.channels,
                kernel_size=1,
                dtype=dtype,
                device=device,
            ),
        )

    @property
    def channels(self) -> int:
        return self.__channels

    def forward(self, tensor: Tensor) -> Tensor:
        return self.__fused_mb_conv(tensor)
