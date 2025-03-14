from sigk.layers.spectral.partial_spectral_conv_2d import PartialSpectralConv2d
from sigk.layers.spectral.partial_spectral_fused_mb_conv import (
    PartialSpectralFusedMBConv,
)
from sigk.layers.partial_gdn import PartialGDN
from sigk.layers.partial_multihead_attention import PartialMultiheadAttention
from sigk.layers.dwt import (
    PartialDwt2D,
    PartialIDwt2D,
    COHEN_DAUBECHIES_FEAUVEAU_9_7_WAVELET,
)

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


class SynthesisConvolutionBlock(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.__idwt = PartialIDwt2D(
            channels=in_channels, wavelet=COHEN_DAUBECHIES_FEAUVEAU_9_7_WAVELET
        )
        self.__conv = PartialSpectralConv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.__norm = PartialGDN(channels=out_channels)

    def forward(
        self,
        ll: Tensor,
        ll_mask: Tensor,
        bands: tuple[Tensor, Tensor, Tensor],
        bands_masks: tuple[Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        pre_recon, pre_recon_mask = self.__idwt((ll, *bands), (ll_mask, *bands_masks))
        return self.__norm(*self.__conv(pre_recon, pre_recon_mask))


class AnalysisFusedBlock(Module):
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


class SynthesisFusedBlock(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.__idwt = PartialDwt2D(
            channels=in_channels, wavelet=COHEN_DAUBECHIES_FEAUVEAU_9_7_WAVELET
        )
        self.__cast = PartialSpectralConv2d(
            in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels
        )
        self.__conv = PartialSpectralFusedMBConv(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.__norm = PartialGDN(channels=out_channels)

    def forward(
        self,
        ll: Tensor,
        ll_mask: Tensor,
        bands: tuple[Tensor, Tensor, Tensor],
        bands_masks: tuple[Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        pre_recon, pre_recon_mask = self.__idwt((ll, *bands), (ll_mask, *bands_masks))
        return self.__norm(*self.__conv(*self.__cast(pre_recon, pre_recon_mask)))


class InpaintingMode(Module):
    def __init__(self, embedding_features: int) -> None:
        super().__init__()
        assert embedding_features % 3 == 0, "Embedding features have to be a power of 3"
        self.__analysis_conv_blocks = ParameterList(
            AnalysisConvolutionBlock(3, embedding_features),
            AnalysisConvolutionBlock(embedding_features, 2 * embedding_features),
        )
        self.__analysis_fused_blocks = ParameterList(
            AnalysisFusedBlock(2 * embedding_features, 4 * embedding_features),
            AnalysisFusedBlock(4 * embedding_features, 8 * embedding_features),
        )
        self.__low_level_recon = PartialMultiheadAttention(
            channels=8 * embedding_features,
            heads=24,
            channels_per_head=embedding_features // 3,
        )
        self.__synthesis_conv_blocks = ParameterList(
            SynthesisConvolutionBlock(2 * embedding_features, embedding_features),
            SynthesisConvolutionBlock(embedding_features, 3),
        )
        self.__synthesis_fused_blocks = ParameterList(
            SynthesisFusedBlock(8 * embedding_features, 4 * embedding_features),
            SynthesisFusedBlock(4 * embedding_features, 2 * embedding_features),
        )
