from torch.nn import Sequential, GELU, Sigmoid

from sigk.layers.spectral.spectral_conv2d import SpectralConv2d
from sigk.layers.gdn import GDN


class DebluringModel(Sequential):
    def __init__(
        self, input_channels: int, latent_channels: int, output_channels: int
    ) -> None:
        super().__init__(
            SpectralConv2d(
                input_channels, latent_channels, kernel_size=9, padding=4, stride=1
            ),
            GDN(latent_channels),
            GELU(),
            SpectralConv2d(
                latent_channels, latent_channels, kernel_size=9, padding=4, stride=1
            ),
            GDN(latent_channels),
            GELU(),
            SpectralConv2d(
                latent_channels, latent_channels, kernel_size=9, padding=4, stride=1
            ),
            GDN(latent_channels),
            GELU(),
            SpectralConv2d(
                latent_channels, output_channels, kernel_size=9, padding=4, stride=1
            ),
            Sigmoid(),
        )
