from sigk.layers.dwt.idwt2d import IDwt2D
from sigk.layers.dwt.wavelet import Wavelet, PyWavelet
from sigk.layers.dwt.partial_dwt_base import PartialDwtBase

from torch import (
    device as tensor_device,
    dtype as tensor_dtype,
    Tensor,
    no_grad,
)


from typing import Optional


class PartialIDwt2D(IDwt2D, PartialDwtBase):
    def __init__(
        self,
        channels: int,
        wavelet: Wavelet | PyWavelet,
        dtype: Optional[tensor_dtype] = None,
        device: Optional[tensor_device] = None,
        epsilon: float = 1e-8,
    ) -> None:
        IDwt2D.__init__(
            self, channels=channels, wavelet=wavelet, dtype=dtype, device=device
        )
        PartialDwtBase.__init__(self, epsilon=epsilon)

    def __calculate_pass_mask(
        self,
        mask: Tensor,
        pass_mask: Tensor,
        pass_window_size: int,
        position: int,
        groups: int,
    ) -> tuple[Tensor, Tensor]:
        current_mask = self._perform_idwt_pass(
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
            pass_window_size=self.first_pass_window_size,
            position=-2,
            groups=self.channels,
        )
        return second_pass_ratio, first_pass_mask, first_pass_ratio

    def __peform_partial_idwt2d(
        self,
        tensor: Tensor,
        mask: Tensor,
        first_pass_ratio: Tensor,
        second_pass_ratio: Tensor,
    ) -> Tensor:
        second_pass = (
            self._perform_idwt_pass(
                tensor * mask,
                kernel=self.second_pass_kernel,
                position=-1,
                groups=2 * self.channels,
            )
            * second_pass_ratio
        )
        return (
            self._perform_idwt_pass(
                second_pass,
                kernel=self.first_pass_kernel,
                position=-2,
                groups=self.channels,
            )
            * first_pass_ratio
        )

    def forward(
        self,
        tensor: tuple[Tensor, Tensor, Tensor, Tensor] | Tensor,
        mask: tuple[Tensor, Tensor, Tensor, Tensor] | Tensor,
        splitting_mode: str = "separate",
    ) -> tuple[Tensor, Tensor]:
        tensor = self._apply_preprocessing(tensor, splitting_mode)
        mask = self._apply_preprocessing(mask, splitting_mode)

        sp_ratio, fp_mask, fp_ratio = self.__calculate_pass_masks(mask)

        result = self.__peform_partial_idwt2d(
            tensor, mask=mask, first_pass_ratio=fp_ratio, second_pass_ratio=sp_ratio
        )

        return result, fp_mask
