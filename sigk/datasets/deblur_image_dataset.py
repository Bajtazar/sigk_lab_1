from sigk.datasets.compressed_image_dataset import CompressedImageDataset

from torchvision.transforms import Resize, CenterCrop, Compose, GaussianBlur
from torchvision.transforms._functional_tensor import _get_gaussian_kernel2d
from torchvision.transforms.functional import gaussian_blur

from torch import Tensor, float32

from typing import Sequence
from random import choice


class DeblurImageDataset(CompressedImageDataset):
    class Blurer:
        def __init__(
            self, kernel_sizes: int | Sequence[int], sigma_min: float, sigma_max: float
        ) -> None:
            if isinstance(kernel_sizes, int):
                self.__kernel_sizes = kernel_sizes
                self.__sigma = (sigma_min, sigma_max)
                self.__inference = True
            else:
                self.__transforms = [
                    GaussianBlur(kernel_size, (sigma_min, sigma_max))
                    for kernel_size in kernel_sizes
                ]
                self.__inference = False

        def __call__(
            self, image: Tensor
        ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
            image = image.to(float32) / 255.0
            if self.__inference:
                sigma = GaussianBlur.get_params(*self.__sigma)
                kernel_size = (self.__kernel_sizes, self.__kernel_sizes)
                sigma = (sigma, sigma)
                filter = _get_gaussian_kernel2d(
                    kernel_size=kernel_size,
                    sigma=sigma,
                    dtype=image.dtype,
                    device=image.device,
                )
                return image, gaussian_blur(image, kernel_size, sigma), filter
            return image, choice(self.__transforms)(image)

    def __init__(
        self,
        root: str,
        image_size: int | tuple[int, int],
        kernel_sizes: int | Sequence[int],
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
