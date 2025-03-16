from torch.nn import Sequential
from torch import Tensor

from sigk.layers.spectral.spectral_conv2d import SpectralConv2d
from sigk.layers.gdn import GDN


class DebluringModel(Sequential):
    def __init__(
        self, input_channels: int, latent_channels: int, output_channels: int
    ) -> None:
        super().__init__(
            SpectralConv2d(
                input_channels, latent_channels, kernel_size=5, padding=2, stride=1
            ),
            GDN(latent_channels),
            SpectralConv2d(
                latent_channels, latent_channels, kernel_size=5, padding=2, stride=1
            ),
            GDN(latent_channels),
            SpectralConv2d(
                latent_channels, latent_channels, kernel_size=5, padding=2, stride=1
            ),
            GDN(latent_channels),
            SpectralConv2d(
                latent_channels, output_channels, kernel_size=5, padding=2, stride=1
            ),
        )

    def __call__(self, tensor: Tensor) -> Tensor:
        return super().forward(tensor).clamp(min=0, max=1)
