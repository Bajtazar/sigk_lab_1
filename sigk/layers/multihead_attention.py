from torch.nn.modules.utils import _pair
from torch.nn.init import trunc_normal_
from torch.nn import Module, Linear, LayerNorm, Parameter
from torch import zeros, stack, meshgrid, arange


class MultiheadAttention(Module):
    def __init__(
        self,
        channels: int,
        heads: int,
        heads_per_channel: int,
        latent_size: int | tuple[int, int],
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.__norm = LayerNorm(channels)
        self.__embedding_channels = heads * heads_per_channel
        self.__qkv_projection = Linear(
            in_features=channels, out_features=self.__embedding_channels * 3, bias=bias
        )
        self.__out_projection = Linear(
            in_features=self.__embedding_channels, out_features=channels
        )
        self.__latent_size = _pair(latent_size)
        self.__initialize_parameters()

    def __initialize_parameters(self) -> None:
        self.__relative_position_bias_table = Parameter(
            zeros(
                (2 * self.latent_size[0] - 1) * (2 * self.latent_size[1] - 1),
                self.number_of_heads,
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
