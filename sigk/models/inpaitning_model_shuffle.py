from sigk.layers.spectral.partial_spectral_conv_2d import PartialSpectralConv2d
from sigk.layers.spectral.partial_spectral_fused_mb_conv import (
    PartialSpectralFusedMBConv,
)
from sigk.layers.partial_leaky_relu import PartialLeakyReLU
from sigk.layers.partial_gdn import PartialGDN, PartialIGDN
from sigk.layers.partial_multihead_attention import PartialMultiheadAttention
from sigk.utils.unpacking_sequence import UnpackingSequential


from torch import Tensor
from torch.nn import Module, ParameterList, PixelShuffle, PixelUnshuffle


class AnalysisConvolutionBlock(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.__sequence = UnpackingSequential(
            PartialSpectralConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            PartialGDN(channels=out_channels),
            PartialLeakyReLU(),
            PartialSpectralConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            PartialGDN(channels=out_channels),
            PartialLeakyReLU(),
        )
        self.__unshuffle = PixelUnshuffle(downscale_factor=2)

    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        tensor, mask = self.__sequence(tensor, mask)
        return self.__unshuffle(tensor), self.__unshuffle(mask)


class SynthesisConvolutionBlock(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.__shuffle = PixelShuffle(upscale_factor=2)
        self.__sequence = UnpackingSequential(
            PartialSpectralConv2d(in_channels, in_channels, kernel_size=3, padding=1),
            PartialIGDN(channels=in_channels),
            PartialLeakyReLU(),
            PartialSpectralConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            PartialIGDN(channels=out_channels),
            PartialLeakyReLU(),
        )

    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        tensor = self.__shuffle(tensor)
        mask = self.__shuffle(mask)
        return self.__sequence(tensor, mask)


class AnalysisFusedBlock(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.__sequence = UnpackingSequential(
            PartialSpectralConv2d(
                in_channels, out_channels, kernel_size=3, padding=1, groups=out_channels
            ),
            PartialSpectralFusedMBConv(
                in_channels,
            ),
            PartialGDN(channels=in_channels),
            PartialLeakyReLU(),
            PartialSpectralFusedMBConv(
                in_channels,
            ),
            PartialGDN(channels=in_channels),
            PartialLeakyReLU(),
        )
        self.__unshuffle = PixelUnshuffle(downscale_factor=2)

    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        tensor, mask = self.__sequence(tensor, mask)
        return self.__unshuffle(tensor), self.__unshuffle(mask)


class SynthesisFusedBlock(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.__sequence = UnpackingSequential(
            PartialSpectralFusedMBConv(
                in_channels,
            ),
            PartialIGDN(channels=in_channels),
            PartialLeakyReLU(),
            PartialSpectralFusedMBConv(
                in_channels,
            ),
            PartialIGDN(channels=in_channels),
            PartialLeakyReLU(),
            PartialSpectralConv2d(
                in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels
            ),
        )
        self.__shuffle = PixelShuffle(upscale_factor=2)

    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        tensor = self.__shuffle(tensor)
        mask = self.__shuffle(mask)
        return self.__sequence(tensor, mask)


class InpaintingModel(Module):
    def __init__(
        self,
        embedding_features: int,
        attention_heads: int,
        latent_size: int | tuple[int, int],
    ) -> None:
        super().__init__()
        assert 16 * embedding_features % attention_heads == 0
        self.__analysis_blocks = ParameterList(
            [
                AnalysisConvolutionBlock(3, embedding_features),
                AnalysisConvolutionBlock(
                    4 * embedding_features, 2 * embedding_features
                ),
                AnalysisConvolutionBlock(
                    8 * embedding_features, 4 * embedding_features
                ),
                AnalysisFusedBlock(16 * embedding_features, 8 * embedding_features),
                AnalysisFusedBlock(32 * embedding_features, 16 * embedding_features),
            ]
        )
        self.__low_level_recon = UnpackingSequential(
            PartialGDN(channels=16 * embedding_features),
            PartialMultiheadAttention(
                channels=16 * embedding_features,
                heads=attention_heads,
                channels_per_head=16 * embedding_features // attention_heads,
                latent_size=latent_size,
            ),
            PartialIGDN(16 * embedding_features),
        )
        self.__synthesis_blocks = ParameterList(
            [
                SynthesisFusedBlock(16 * embedding_features, 32 * embedding_features),
                SynthesisFusedBlock(8 * embedding_features, 16 * embedding_features),
                SynthesisConvolutionBlock(
                    4 * embedding_features, 8 * embedding_features
                ),
                SynthesisConvolutionBlock(
                    2 * embedding_features, 4 * embedding_features
                ),
                SynthesisConvolutionBlock(embedding_features, 3),
            ]
        )

    def forward(self, tensor: Tensor, mask: Tensor) -> Tensor:
        residuals = []
        ll, ll_mask = tensor, mask
        for block in self.__analysis_blocks:
            (ll, ll_mask), residual = block(ll, ll_mask)
            residuals.append(residual)
        recon, recon_mask = self.__low_level_recon(ll, ll_mask)
        for block, residue in zip(self.__synthesis_blocks, reversed(residuals)):
            recon, recon_mask = block(recon, recon_mask, *residue)
        return recon.clamp(min=0, max=1)
