from torch.utils.data import Dataset
from torch import Tensor

from torchvision.io import (
    ImageReadMode,
    read_image,
    encode_png,
    decode_png,
)

from typing import Generator, Callable, Optional
from os.path import isfile
from os import walk


TransformCb = Callable[[Tensor], Tensor]


class CompressedImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        precache_transformation: Optional[TransformCb] = None,
        postcache_transformation: Optional[TransformCb] = None,
    ) -> None:
        super().__init__()
        self.__root = root
        self.__cache = [
            (path, self.__transform(path, precache_transformation))
            for path in self.__get_dataset_images()
        ]
        self.__postcache_transformation = postcache_transformation

    def __transform(
        self, path: str, precache_transformation: Optional[TransformCb]
    ) -> bytes:
        image = read_image(path, ImageReadMode.RGB)
        if precache_transformation is not None:
            image = precache_transformation
        return encode_png(image)

    def __get_dataset_images(self) -> Generator[str, None, None]:
        for root, _, image_paths in walk(self.root_directory):
            for image_path in image_paths:
                image_path = f"{root}/{image_path}"
                if isfile(image_path):
                    yield image_path

    @property
    def root_directory(self) -> str:
        return self.__root

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        path, cached = self.__cache[index]
        image = decode_png(cached, ImageReadMode.RGB)
        if self.__postcache_transformation:
            image = self.__postcache_transformation(image)
        return image, path

    def __len__(self) -> None:
        return len(self.__cache)
