from sigk.datasets.compressed_image_dataset import CompressedImageDataset

from torchvision.transforms import Resize, CenterCrop, Compose, GaussianBlur

from torch import Tensor, float32

from typing import Sequence
from random import choice


class DeblurImageDataset(CompressedImageDataset):
    class Blurer:
        def __init__(
            self, kernel_sizes: Sequence[int], sigma_min: float, sigma_max: float
        ) -> None:
            self.__transforms = [
                GaussianBlur(kernel_size, (sigma_min, sigma_max))
                for kernel_size in kernel_sizes
            ]

        def __call__(self, image: Tensor) -> tuple[Tensor, Tensor]:
            image = image.to(float32) / 255.0
            return choice(self.__kernel_sizes)(image)

    def __init__(
        self,
        root: str,
        image_size: int | tuple[int, int],
        kernel_sizes: Sequence[int],
        sigma_min: float,
        sigma_max: float,
    ) -> None:
        super().__init__(
            root=root,
            precache_transformation=Compose(
                [Resize(image_size), CenterCrop(image_size)]
            ),
            postcache_transformation=self.Blurer(
                kernel_sizes=kernel_sizes, sigma_min=sigma_min, sigma_max=sigma_max
            ),
        )
