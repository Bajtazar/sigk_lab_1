from sigk.datasets.compressed_image_dataset import CompressedImageDataset

from torchvision.transforms import Resize, CenterCrop, Compose

from torch.nn.modules.utils import _pair
from torch import Tensor, float32, ones_like

from typing import Sequence
from random import choice, randint


class InpaintingImageDataset(CompressedImageDataset):
    class Patcher:
        def __init__(
            self, patch_sizes: Sequence[int | tuple[int, int]], patch_count: int
        ) -> None:
            self.__patch_sizes = patch_sizes
            self.__patch_count = patch_count

        def __call__(self, image: Tensor) -> tuple[Tensor, Tensor]:
            image = image.to(float32) / 255.0
            mask = ones_like(image)
            height, width = image.shape[-2:]
            for _ in range(self.__patch_count):
                ph, pw = _pair(choice(self.__patch_sizes))
                h_start = randint(0, height - 1 - ph)
                w_start = randint(0, width - 1 - pw)
                mask[..., h_start : h_start + ph, w_start : w_start + pw] = 0
            return image, mask

    def __init__(
        self,
        root: str,
        image_size: int | tuple[int, int],
        patch_sizes: Sequence[int | tuple[int, int]],
        patch_count: int,
    ) -> None:
        super().__init__(
            root=root,
            precache_transformation=Compose(Resize(image_size), CenterCrop(image_size)),
            postcache_transformation=self.Patcher(
                patch_sizes=patch_sizes, patch_count=patch_count
            ),
        )
