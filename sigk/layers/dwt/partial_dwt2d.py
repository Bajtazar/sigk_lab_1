from sigk.layers.dwt.dwt2d import Dwt2D
from sigk.layers.dwt.wavelet import Wavelet, PyWavelet

from torch import (
    dtype as tensor_dtype,
    device as tensor_device,
    ones_like,
    tensor,
    Tensor,
    no_grad,
)

from numpy import prod

from typing import Optional


class PartialDwt2D(Dwt2D):
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
        first_pass_mask, first_pass_ratio = self.__calculate_pass_mask(
            mask=mask,
            pass_mask=self.first_pass_mask_coeffs,
            pass_window_size=self.first_pass_window_size,
            position=-2,
            groups=self.channels,
        )
        second_pass_mask, second_pass_ratio = self.__calculate_pass_mask(
            mask=first_pass_mask,
            pass_mask=self.second_pass_mask_coeffs,
            pass_window_size=self.second_pass_window_size,
            position=-1,
            groups=2 * self.channels,
        )
        return first_pass_ratio, second_pass_mask, second_pass_ratio

    def __perform_partial_dwt2d(
        self,
        tensor: Tensor,
        mask: Tensor,
        first_pass_ratio: Tensor,
        second_pass_ratio: Tensor,
    ) -> Tensor:
        first_pass = (
            self._perform_dwt_pass(
                tensor * mask,
                kernel=self.first_pass_kernel,
                position=-2,
                groups=self.channels,
            )
            * first_pass_ratio
        )
        return (
            self._perform_dwt_pass(
                first_pass,
                kernel=self.second_pass_kernel,
                position=-1,
                groups=2 * self.channels,
            )
            * second_pass_ratio
        )

    def forward(
        self, tensor: Tensor, mask: Tensor, splitting_mode: str = "separate"
    ) -> (
        tuple[Tensor, Tensor]
        | tuple[
            tuple[Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor, Tensor]
        ]
    ):
        self._check_input_tensor(tensor, dimension=4)
        if self.padding_mode != "zeros":
            tensor = self._apply_padding(tensor, mode=self.padding_mode)
            mask = self._apply_padding(mask, mode="constant", value=1.0)

        fp_ratio, sp_mask, sp_ratio = self.__calculate_pass_masks(mask)

        result = self.__perform_partial_dwt2d(
            tensor, mask=mask, first_pass_ratio=fp_ratio, second_pass_ratio=sp_ratio
        )

        return (
            self._apply_postprocessing(result, splitting_mode),
            self._apply_postprocessing(sp_mask, splitting_mode),
        )
