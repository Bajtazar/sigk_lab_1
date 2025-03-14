from sigk.layers.spectral.partial_spectral_conv_2d import PartialSpectralConv2d
from sigk.layers.spectral.partial_spectral_fused_mb_conv import (
    PartialSpectralFusedMBConv,
)
from sigk.layers.partial_gdn import PartialGDN
from sigk.layers.dwt import PartialDwt2D, COHEN_DAUBECHIES_FEAUVEAU_9_7_WAVELET

from torch import Tensor
from torch.nn import Module, ParameterList


class AnalysisConvolutionBlock(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.__conv = PartialSpectralConv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.__norm = PartialGDN(channels=out_channels)
        self.__dwt = PartialDwt2D(
            channels=out_channels, wavelet=COHEN_DAUBECHIES_FEAUVEAU_9_7_WAVELET
        )

    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[
        tuple[Tensor, Tensor],
        tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]],
    ]:
        (ll, *bands), (ll_mask, *bands_masks) = self.__dwt(
            *self.__norm(*self.__conv(tensor, mask))
        )
        return (ll, ll_mask), (bands, bands_masks)


class AnalysisFusedBlocks(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.__conv = PartialSpectralFusedMBConv(
            in_channels, in_channels, kernel_size=3, padding=1
        )
        self.__cast = PartialSpectralConv2d(
            in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels
        )
        self.__norm = PartialGDN(channels=out_channels)
        self.__dwt = PartialDwt2D(
            channels=out_channels, wavelet=COHEN_DAUBECHIES_FEAUVEAU_9_7_WAVELET
        )

    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[
        tuple[Tensor, Tensor],
        tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]],
    ]:
        (ll, *bands), (ll_mask, *bands_masks) = self.__dwt(
            *self.__norm(*self.__cast(*self.__conv(tensor, mask)))
        )
        return (ll, ll_mask), (bands, bands_masks)


class InpaintingMode(Module):
    def __init__(self, embedding_features: int) -> None:
        super().__init__()
        self.__analysis_conv_blocks = ParameterList(
            self.AnalysisConvolutionBlock(3, embedding_features),
            self.AnalysisConvolutionBlock(embedding_features, 2 * embedding_features),
        )
        self.__analysis_fused_blocks = ParameterList(
            self.AnalysisFusedBlocks(2 * embedding_features, 4 * embedding_features),
            self.AnalysisFusedBlocks(4 * embedding_features, 8 * embedding_features),
        )
