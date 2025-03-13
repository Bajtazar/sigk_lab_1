from sigk.layers.spectral.spectral import spectral

from torch.nn import Conv2d


@spectral
class SpectralConv2d(Conv2d):
    pass
