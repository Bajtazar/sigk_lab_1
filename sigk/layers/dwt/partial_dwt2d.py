from sigk.layers.dwt.dwt2d import Dwt2D
from sigk.layers.dwt.wavelet import Wavelet, PyWavelet

from torch import (
    dtype as tensor_dtype,
    device as tensor_device,
    ones_like,
    tensor,
    Tensor,
)

from numpy import prod

from typing import Optional


class PartialDwt2d(Dwt2D):
    def __init__(
        self,
        channels: int,
        wavelet: Wavelet | PyWavelet,
        padding_mode: str = "zeros",
        dtype: Optional[tensor_dtype] = None,
        device: Optional[tensor_device] = None,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(
            channels=channels,
            wavelet=wavelet,
            padding_mode=padding_mode,
            dtype=dtype,
            device=device,
        )
        self.register_buffer(
            "first_pass_mask_coeffs", ones_like(self.first_pass_kernel)
        )
        self.register_buffer(
            "second_pass_mask_coeffs", ones_like(self.second_pass_kernel)
        )
        self.__register_size_buffer("first_pass", self.first_pass_kernel)
        self.__register_size_buffer("second_pass", self.second_pass_kernel)
        self.__epsilon = epsilon

    def __register_size_buffer(self, name: str, kernel: Tensor) -> None:
        self.register_buffer(
            f"{name}_window_size",
            tensor([prod(kernel.shape[1:])], dtype=kernel.dtype, device=kernel.device),
        )

    @property
    def epsilon(self) -> float:
        return self.__epsilon
