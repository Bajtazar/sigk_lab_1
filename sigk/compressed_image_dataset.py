from torch.utils.data import Dataset
from torch import frombuffer, Tensor

from torchvision.io import decode_image, ImageReadMode

from typing import Generator, Callable, Optional
from os.path import isfile
from os import walk


TransformCb = Callable[[Tensor], Tensor]


class CompressedImageDataset(Dataset):
    def __init__(self, root: str, transformation: Optional[TransformCb] = None) -> None:
        super().__init__()
        self.__root = root
        self.__cache = [
            (path, frombuffer(self.__read_file(path)))
            for path in self.__get_dataset_images()
        ]
        self.__transformation = transformation

    def __get_dataset_images(self) -> Generator[str, None, None]:
        for root, _, image_paths in walk(self.root_directory):
            for image_path in image_paths:
                image_path = f"{root}/{image_path}"
                if isfile(image_path):
                    yield image_path

    @property
    def root_directory(self) -> str:
        return self.__root

    @staticmethod
    def __read_file(path: str) -> bytes:
        with open(path, "rb") as handle:
            return handle.read()

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        path, cached = self.__cache[index]
        image = decode_image(cached, ImageReadMode.RGB)
        if self.__transformation:
            image = self.__transformation(image)
        return image, path

    def __len__(self) -> None:
        return len(self.__cache)
