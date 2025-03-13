from sigk.layers.dwt.idwt2d import IDwt2D
from sigk.layers.dwt.wavelet import Wavelet, PyWavelet

from torch import (
    device as tensor_device,
    dtype as tensor_dtype,
    ones_like,
    Tensor,
    tensor,
    no_grad,
)

from numpy import prod

from typing import Optional


class DwtBase(IDwt2D):
    def __init__(
        self,
        channels: int,
        wavelet: Wavelet | PyWavelet,
        dtype: Optional[tensor_dtype] = None,
        device: Optional[tensor_device] = None,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(channels=channels, wavelet=wavelet, dtype=dtype, device=device)
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

    def __calculate_pass_mask(
        self,
        mask: Tensor,
        pass_mask: Tensor,
        pass_window_size: int,
        position: int,
        groups: int,
    ) -> tuple[Tensor, Tensor]:
        current_mask = self._perform_dwt_pass(
            mask, kernel=pass_mask, position=position, groups=groups
        )
        ratio = pass_window_size / (current_mask + self.epsilon)
        current_mask = current_mask.clamp(min=0, max=1)
        return current_mask, current_mask * ratio

    @no_grad
    def __calculate_pass_masks(self, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        second_pass_mask, second_pass_ratio = self.__calculate_pass_mask(
            mask=mask,
            pass_mask=self.second_pass_mask_coeffs,
            pass_window_size=self.second_pass_window_size,
            position=-1,
            groups=2 * self.channels,
        )
        first_pass_mask, first_pass_ratio = self.__calculate_pass_mask(
            mask=second_pass_mask,
            pass_mask=self.first_pass_mask_coeffs,
            pass_window_size=self.firstpass_window_size,
            position=-2,
            groups=self.channels,
        )
        return second_pass_ratio, first_pass_mask, first_pass_ratio

    @property
    def epsilon(self) -> float:
        return self.__epsilon
