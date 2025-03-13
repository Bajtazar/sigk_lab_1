from sigk.layers.spectral.partial_spectral_conv_2d import PartialSpectralConv2d
from sigk.layers.partial_squeeze_and_excitation_block import (
    PartialSqueezeAndExcitationBlock,
)
from sigk.utils.unpacking_sequence import UnpackingSequential

from torch.nn import Module
from torch import Tensor, device as tensor_device, dtype as tensor_dtype

from typing import Optional


class PartialSpectralFusedMBConv(Module):
    def __init__(
        self,
        channels: int,
        bottleneck: int = 16,
        device: Optional[tensor_device | str] = None,
        dtype: Optional[tensor_dtype] = None,
    ) -> None:
        super().__init__()
        self.__channels = channels
        self.__fused_mb_conv = UnpackingSequential(
            PartialSpectralConv2d(
                self.channels,
                4 * self.channels,
                kernel_size=3,
                padding=1,
                dtype=dtype,
                device=device,
            ),
            PartialSqueezeAndExcitationBlock(
                4 * self.channels, bottleneck, dtype=dtype, device=device
            ),
            PartialSpectralConv2d(
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

    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        tensor = tensor * mask
        excitation, exc_mask = self.__fused_mb_conv(tensor, mask)
        return tensor + excitation, (mask + exc_mask).clamp(min=0, max=1)
