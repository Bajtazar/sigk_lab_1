from sigk.layers.spectral.partial_spectral_conv_2d import PartialSpectralConv2d
from sigk.layers.spectral.partial_spectral_fused_mb_conv import (
    PartialSpectralFusedMBConv,
)
from sigk.layers.partial_gdn import PartialGDN, PartialIGDN
from sigk.layers.partial_leaky_relu import PartialLeakyReLU
from sigk.layers.partial_multihead_attention import PartialMultiheadAttention
from sigk.utils.unpacking_sequence import UnpackingSequential

from torch import Tensor, chunk, cat
from torch.nn import Module, ParameterList, PixelShuffle, PixelUnshuffle


class AnalysisConvolutionBlock(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.__sequence = UnpackingSequential(
            PartialSpectralConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            PartialGDN(channels=out_channels),
            PartialSpectralConv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.__unshuffle = PixelUnshuffle(downscale_factor=2)

    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[
        tuple[Tensor, Tensor],
        tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]],
    ]:
        tensor, mask = self.__sequence(tensor, mask)
        tensor = self.__unshuffle(tensor)
        mask = self.__unshuffle(mask)
        (ll, *bands) = chunk(tensor, 4, dim=1)
        (ll_mask, *bands_masks) = chunk(mask, 4, dim=1)
        return (ll, ll_mask), (bands, bands_masks)


class SynthesisConvolutionBlock(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.__shuffle = PixelShuffle(upscale_factor=2)
        self.__sequence = UnpackingSequential(
            PartialSpectralConv2d(in_channels, in_channels, kernel_size=3, padding=1),
            PartialLeakyReLU(),
            PartialIGDN(channels=in_channels),
            PartialSpectralConv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        ll: Tensor,
        ll_mask: Tensor,
        bands: tuple[Tensor, Tensor, Tensor],
        bands_masks: tuple[Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        tensor = self.__shuffle(cat((ll, *bands), dim=1))
        mask = self.__shuffle(cat((ll_mask, *bands_masks), dim=1))
        return self.__sequence(tensor, mask)


class AnalysisFusedBlock(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.__sequence = UnpackingSequential(
            PartialSpectralConv2d(
                in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels
            ),
            PartialSpectralFusedMBConv(
                out_channels,
            ),
            PartialGDN(channels=out_channels),
            PartialSpectralFusedMBConv(
                out_channels,
            ),
        )
        self.__unshuffle = PixelUnshuffle(downscale_factor=2)

    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[
        tuple[Tensor, Tensor],
        tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]],
    ]:
        tensor, mask = self.__sequence(tensor, mask)
        tensor = self.__unshuffle(tensor)
        mask = self.__unshuffle(mask)
        (ll, *bands) = chunk(tensor, 4, dim=1)
        (ll_mask, *bands_masks) = chunk(mask, 4, dim=1)
        return (ll, ll_mask), (bands, bands_masks)


class SynthesisFusedBlock(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.__sequence = UnpackingSequential(
            PartialSpectralFusedMBConv(in_channels),
            PartialLeakyReLU(),
            PartialIGDN(channels=in_channels),
            PartialSpectralFusedMBConv(in_channels),
            PartialSpectralConv2d(
                in_channels, out_channels, kernel_size=3, padding=1, groups=out_channels
            ),
        )
        self.__shuffle = PixelShuffle(upscale_factor=2)

    def forward(
        self,
        ll: Tensor,
        ll_mask: Tensor,
        bands: tuple[Tensor, Tensor, Tensor],
        bands_masks: tuple[Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        tensor = self.__shuffle(cat((ll, *bands), dim=1))
        mask = self.__shuffle(cat((ll_mask, *bands_masks), dim=1))
        return self.__sequence(tensor, mask)


class Bottleneck(Module):
    def __init__(self, channels: int, heads: int, latent_size: int) -> None:
        super().__init__()
        self.__channels = channels
        self.__stem = UnpackingSequential(
            PartialSpectralConv2d(channels, channels * 2, kernel_size=1),
            PartialGDN(2 * channels),
        )
        self.__low_pass = UnpackingSequential(
            PartialMultiheadAttention(
                channels=self.__channels,
                heads=heads,
                channels_per_head=self.__channels // heads,
                latent_size=latent_size,
            ),
        )
        self.__high_pass = UnpackingSequential(
            PartialSpectralConv2d(2 * channels, 2 * channels, kernel_size=3, padding=1),
            PartialGDN(2 * channels),
            PartialLeakyReLU(),
            PartialSpectralConv2d(2 * channels, channels, kernel_size=3, padding=1),
            PartialLeakyReLU(),
            PartialIGDN(channels),
        )

    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        stem_tensor, stem_mask = self.__stem(tensor, mask)
        low_pass, low_pass_mask = self.__low_pass(
            stem_tensor[:, self.__channels :, ...],
            stem_mask[:, self.__channels :, ...],
        )
        tensor = cat((low_pass, stem_tensor[:, : self.__channels, ...]), dim=1)
        mask = cat((low_pass_mask, stem_mask[:, : self.__channels, ...]), dim=1)
        return self.__high_pass(tensor, mask)


class InpaintingModelShuffle(Module):
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
                AnalysisConvolutionBlock(embedding_features, 2 * embedding_features),
                AnalysisConvolutionBlock(
                    2 * embedding_features, 4 * embedding_features
                ),
                AnalysisFusedBlock(4 * embedding_features, 8 * embedding_features),
                AnalysisFusedBlock(8 * embedding_features, 16 * embedding_features),
            ]
        )
        self.__low_level_recon = Bottleneck(
            16 * embedding_features, attention_heads, latent_size
        )
        self.__synthesis_blocks = ParameterList(
            [
                SynthesisFusedBlock(16 * embedding_features, 8 * embedding_features),
                SynthesisFusedBlock(8 * embedding_features, 4 * embedding_features),
                SynthesisConvolutionBlock(
                    4 * embedding_features, 2 * embedding_features
                ),
                SynthesisConvolutionBlock(2 * embedding_features, embedding_features),
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
