from sigk.layers.multihead_attention import MultiheadAttention

from torch import Tensor, ones_like


class PartialMultiheadAttention(MultiheadAttention):
    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        # Perform a full attention so all of the resulting pixels are valid now
        return super().forward(tensor * mask), ones_like(mask)
