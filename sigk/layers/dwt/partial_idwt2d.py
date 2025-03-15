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
    ) -> None:
        IDwt2D.__init__(
            self, channels=channels, wavelet=wavelet, dtype=dtype, device=device
        )
        PartialDwtBase.__init__(self)

    def __calculate_pass_mask(
        self,
        mask: Tensor,
        pass_mask: Tensor,
        position: int,
        groups: int,
    ) -> Tensor:
        return self._perform_idwt_pass(
            mask, kernel=pass_mask, position=position, groups=groups
        ).clamp(min=0, max=1)

    @no_grad
    def __calculate_pass_masks(self, mask: Tensor) -> Tensor:
        second_pass_mask = self.__calculate_pass_mask(
            mask=mask,
            pass_mask=self.second_pass_mask_coeffs,
            position=-1,
            groups=2 * self.channels,
        )
        first_pass_mask = self.__calculate_pass_mask(
            mask=second_pass_mask,
            pass_mask=self.first_pass_mask_coeffs,
            position=-2,
            groups=self.channels,
        )
        return first_pass_mask

    def __peform_partial_idwt2d(
        self,
        tensor: Tensor,
        mask: Tensor,
    ) -> Tensor:
        second_pass = self._perform_idwt_pass(
            tensor * mask,
            kernel=self.second_pass_kernel,
            position=-1,
            groups=2 * self.channels,
        )
        return self._perform_idwt_pass(
            second_pass,
            kernel=self.first_pass_kernel,
            position=-2,
            groups=self.channels,
        )

    def forward(
        self,
        tensor: tuple[Tensor, Tensor, Tensor, Tensor] | Tensor,
        mask: tuple[Tensor, Tensor, Tensor, Tensor] | Tensor,
        splitting_mode: str = "separate",
    ) -> tuple[Tensor, Tensor]:
        tensor = self._apply_preprocessing(tensor, splitting_mode)
        mask = self._apply_preprocessing(mask, splitting_mode)

        fp_mask = self.__calculate_pass_masks(mask)

        result = self.__peform_partial_idwt2d(tensor, mask=mask)

        return result, fp_mask
