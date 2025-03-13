from sigk.layers.squeeze_and_excitaition_block import SqueezeAndExcitationBlock

from torch import Tensor


class PartialSqueezeAndExcitationBlock(SqueezeAndExcitationBlock):
    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        return super().forward(tensor * mask), mask
