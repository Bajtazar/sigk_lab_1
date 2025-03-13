from torch.nn.modules.utils import _pair
from torch.nn.functional import softmax
from torch.nn.init import trunc_normal_
from torch.nn import Module, Linear, Parameter
from torch import zeros, stack, meshgrid, arange, Tensor

from sigk.layers.gdn import GDN

from typing import Optional


class MultiheadAttention(Module):
    def __init__(
        self,
        channels: int,
        heads: int,
        channels_per_head: int,
        latent_size: int | tuple[int, int],
        bias: bool = False,
        scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.__norm = GDN(channels)
        self.__embedding_channels = heads * channels_per_head
        self.__qkv_projection = Linear(
            in_features=channels, out_features=self.__embedding_channels * 3, bias=bias
        )
        self.__out_projection = Linear(
            in_features=self.__embedding_channels, out_features=channels
        )
        self.__heads = heads
        self.__latent_size = _pair(latent_size)
        self.__scale = scale or channels_per_head**-0.5
        self.__channels_per_head = channels_per_head
        self.__initialize_parameters()

    def __initialize_parameters(self) -> None:
        self.__relative_position_bias_table = Parameter(
            zeros(
                (2 * self.latent_size[0] - 1) * (2 * self.latent_size[1] - 1),
                self.heads,
            )
        )
        trunc_normal_(self.__relative_position_bias_table, std=0.02)
        self.__initialize_relative_position_indices()

    def __initialize_relative_position_indices(self) -> None:
        # Rethinking and Improving Relative Position Encoding for Vision Transformer
        # https://arxiv.org/pdf/2107.14222
        coords = stack(
            meshgrid(
                [arange(self.latent_size[0]), arange(self.latent_size[1])],
                indexing="ij",
            )
        ).flatten(1)
        relative_coords = coords[..., None] - coords[:, None, :]
        relative_coords = relative_coords.movedim(0, -1).contiguous()
        relative_coords[..., 0] = (
            relative_coords[..., 0] + self.latent_size[0] - 1
        ) * (2 * self.latent_size[1] - 1)
        relative_coords[..., 1] += self.latent_size[1] - 1
        self.register_buffer("_relative_position_indices", relative_coords.sum(-1))

    @property
    def embedding_channels(self) -> int:
        return self.__embedding_channels

    @property
    def latent_size(self) -> tuple[int, int]:
        return self.__latent_size

    @property
    def heads(self) -> int:
        return self.__heads

    def __add_relative_position_encoding(
        self,
        attention: Tensor,
    ) -> Tensor:
        latent_area = self.latent_size[0] * self.latent_size[1]
        relative_position_bias = (
            self.__relative_position_bias_table[
                self._relative_position_indices.view(-1)
            ]
            .view(latent_area, latent_area, -1)
            .permute(2, 0, 1)
            .contiguous()
            .unsqueeze(0)
        )
        return attention + relative_position_bias

    def __calculate_qkv(self, tensor: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, n, _ = tensor.shape
        return (
            self.__qkv_projection(tensor)
            .reshape(b, n, 3, self.heads, self.__channels_per_head)
            .permute(2, 0, 3, 1, 4)
            .chunk(3, dim=0)
        )

    def forward(self, tensor: Tensor) -> Tensor:
        tensor = self.__norm(tensor)
        original_image_shape = tensor.shape
        tensor = tensor.flatten(start_dim=-2).movedim(1, -1)

        querry, key, value = self.__calculate_qkv(tensor)

        attention = querry @ key.transpose(-2, -1) * self.__scale
        attention = softmax(self.__add_relative_position_encoding(attention), dim=-1)

        result = (attention @ value).reshape(tensor.shape)
        result = self.__out_projection(result)

        return result.movedim(-1, 1).reshape(original_image_shape)
