from sigk.layers.spectral.partial_spectral_conv_2d import PartialSpectralConv2d
from sigk.layers.spectral.partial_spectral_fused_mb_conv import (
    PartialSpectralFusedMBConv,
)
from sigk.layers.partial_gdn import PartialGDN, PartialIGDN
from sigk.layers.partial_multihead_attention import PartialMultiheadAttention
from sigk.layers.dwt import (
    PartialDwt2D,
    PartialIDwt2D,
    COHEN_DAUBECHIES_FEAUVEAU_9_7_WAVELET,
)
from sigk.utils.unpacking_sequence import UnpackingSequential

from torch import Tensor
from torch.nn import Module, ParameterList


class AnalysisConvolutionBlock(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.__sequence = UnpackingSequential(
            PartialSpectralConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            PartialGDN(channels=out_channels),
            PartialSpectralConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            PartialGDN(channels=out_channels),
        )
        self.__dwt = PartialDwt2D(
            channels=out_channels, wavelet=COHEN_DAUBECHIES_FEAUVEAU_9_7_WAVELET
        )

    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[
        tuple[Tensor, Tensor],
        tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]],
    ]:
        (ll, *bands), (ll_mask, *bands_masks) = self.__dwt(
            *self.__sequence(tensor, mask)
        )
        return (ll, ll_mask), (bands, bands_masks)


class SynthesisConvolutionBlock(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.__idwt = PartialIDwt2D(
            channels=in_channels, wavelet=COHEN_DAUBECHIES_FEAUVEAU_9_7_WAVELET
        )
        self.__sequence = UnpackingSequential(
            PartialSpectralConv2d(in_channels, in_channels, kernel_size=3, padding=1),
            PartialIGDN(channels=in_channels),
            PartialSpectralConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            PartialIGDN(channels=out_channels),
        )

    def forward(
        self,
        ll: Tensor,
        ll_mask: Tensor,
        bands: tuple[Tensor, Tensor, Tensor],
        bands_masks: tuple[Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        pre_recon, pre_recon_mask = self.__idwt((ll, *bands), (ll_mask, *bands_masks))
        return self.__sequence(pre_recon, pre_recon_mask)


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
            PartialGDN(channels=out_channels),
        )
        self.__dwt = PartialDwt2D(
            channels=out_channels, wavelet=COHEN_DAUBECHIES_FEAUVEAU_9_7_WAVELET
        )

    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[
        tuple[Tensor, Tensor],
        tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]],
    ]:
        (ll, *bands), (ll_mask, *bands_masks) = self.__dwt(
            *self.__sequence(tensor, mask)
        )
        return (ll, ll_mask), (bands, bands_masks)


class SynthesisFusedBlock(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.__sequence = UnpackingSequential(
            PartialSpectralFusedMBConv(
                in_channels,
                in_channels,
            ),
            PartialIGDN(channels=in_channels),
            PartialSpectralFusedMBConv(
                in_channels,
                in_channels,
            ),
            PartialSpectralConv2d(
                in_channels, out_channels, kernel_size=2, padding=1, groups=out_channels
            ),
            PartialIGDN(channels=out_channels),
        )
        self.__idwt = PartialIDwt2D(
            channels=in_channels, wavelet=COHEN_DAUBECHIES_FEAUVEAU_9_7_WAVELET
        )

    def forward(
        self,
        ll: Tensor,
        ll_mask: Tensor,
        bands: tuple[Tensor, Tensor, Tensor],
        bands_masks: tuple[Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        pre_recon, pre_recon_mask = self.__idwt((ll, *bands), (ll_mask, *bands_masks))
        return self.__sequence(pre_recon, pre_recon_mask)


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
                AnalysisConvolutionBlock(embedding_features, 2 * embedding_features),
                AnalysisConvolutionBlock(
                    2 * embedding_features, 4 * embedding_features
                ),
                AnalysisFusedBlock(4 * embedding_features, 8 * embedding_features),
                AnalysisFusedBlock(8 * embedding_features, 16 * embedding_features),
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
        return recon
