from lightning import LightningModule

from torch import Tensor
from torch.nn import Module, ParameterList
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sigk.layers.spectral.partial_spectral_conv_2d import PartialSpectralConv2d
from sigk.layers.partial_gdn import PartialGDN
from sigk.layers.dwt import PartialDwt2D, COHEN_DAUBECHIES_FEAUVEAU_9_7_WAVELET


class InpaintingMode(Module):
    class ConvolutionBlock(Module):
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

    def __init__(self, embedding_features: int) -> None:
        super().__init__()
        self.__analysis_conv_blocks = ParameterList(
            self.ConvolutionBlock(3, embedding_features),
            self.ConvolutionBlock(embedding_features, 2 * embedding_features),
        )


class Inpainting(LightningModule):
    def __init__(self) -> None:
        super().__init__()

    def configure_optimizers(self) -> list[Adam | ReduceLROnPlateau | str]:
        adam = Adam(self.parameters(), lr=self.__learning_rate)
        plateau = ReduceLROnPlateau(adam, **self.__scheduler_params)
        return {
            "optimizer": adam,
            "lr_scheduler": plateau,
            "monitor": "validation loss",
        }
