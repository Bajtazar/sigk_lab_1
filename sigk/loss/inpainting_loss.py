from torch import Tensor, empty, empty_like
from torch.nn.modules.utils import _pair
from torch.nn import Module

from numpy import prod


class InpaintingLoss(Module):
    def __init__(
        self, image_size: int | tuple[int, int], max_batch_size: int, channels: int = 3
    ) -> None:
        super().__init__()
        self.__image_size = _pair(image_size)
        self.__max_batch_size = max_batch_size
        self.__channels = channels
        self.__initialize_imagenet_mean_and_stddev()

    @property
    def image_size(self) -> tuple[int, int]:
        return self.__image_size

    @property
    def channels(self) -> int:
        return self.__channels

    @property
    def max_batch_size(self) -> int:
        return self.__max_batch_size

    def __initialize_imagenet_mean_and_stddev(self) -> None:
        self.register_buffer(
            "imagenet_mean",
            empty((self.max_batch_size, self.channels, *self.image_size)),
        )
        self.register_buffer("imagenet_stddev", empty_like(self.imagenet_mean))
        self.imagenet_mean[:, 0, ...] = 0.485
        self.imagenet_mean[:, 1, ...] = 0.456
        self.imagenet_mean[:, 2, ...] = 0.406
        self.imagenet_stddev[:, 0, ...] = 0.229
        self.imagenet_stddev[:, 1, ...] = 0.224
        self.imagenet_stddev[:, 2, ...] = 0.225

    def __normalize_to_imagenet(self, tensor: Tensor) -> Tensor:
        tensor = tensor - self.imagenet_mean
        return tensor / self.imagenet_stddev

    def __call__(self, x: Tensor, x_hat: Tensor) -> Tensor:
        x = self.__normalize_to_imagenet(x)
        x_hat = self.__normalize_to_imagenet(x_hat)

    def __calculate_gram_matrix(matrix: Tensor) -> Tensor:
        left = matrix.flatten(dim=-2)
        right = left.transpose(-2, -1)
        return left @ right / prod(left.shape[1:])
