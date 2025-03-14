from sigk.layers.gdn import GDNBase

from torch.nn.functional import conv2d
from torch import (
    Tensor,
    device as tensor_device,
    dtype as tensor_dtype,
    sqrt,
    rsqrt,
    no_grad,
    ones,
    tensor,
)

from typing import Optional


class PartialGDNBase(GDNBase):
    def __init__(
        self,
        channels: int,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
        reparam_offset: float = 2**-18,
        device: Optional[tensor_device] = None,
        dtype: Optional[tensor_dtype] = None,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(
            channels=channels,
            beta_min=beta_min,
            gamma_init=gamma_init,
            reparam_offset=reparam_offset,
            device=device,
            dtype=dtype,
        )
        self.register_buffer(
            "gamma_mask_coeffs",
            ones(
                [channels, channels, 1, 1],
                dtype=self.gamma.dtype,
                device=self.gamma.device,
            ),
        )
        self.register_buffer(
            "gamma_window_size",
            tensor([channels], dtype=dtype, device=device),
        )
        self.__epsilon = epsilon

    @property
    def epsilon(self) -> float:
        return self.__epsilon

    @no_grad
    def __calculate_current_mask(self, mask: Tensor) -> tuple[Tensor, float]:
        current_mask = conv2d(mask, self.gamma_mask_coeffs, bias=None)
        ratio = self.gamma_window_size / (current_mask + self.epsilon)
        current_mask = current_mask.clamp(min=0, max=1)
        return current_mask, current_mask * ratio

    def _calculate_norm(self, tensor: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        self._assert_input_size(tensor)
        self._assert_input_size(mask)
        gamma, beta = self._projected_weights
        current_mask, ratio = self.__calculate_current_mask(mask)
        result = conv2d((tensor * mask) ** 2, gamma, bias=None) * ratio + beta.view(
            1, -1, 1, 1
        )
        return result, current_mask


class PartialGDN(PartialGDNBase):
    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        norm, current_mask = self._calculate_norm(tensor, mask)
        return current_mask * tensor * rsqrt(norm), current_mask


class PartialIGDN(PartialGDNBase):
    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        norm, current_mask = self._calculate_norm(tensor, mask)
        return current_mask * tensor * sqrt(norm), current_mask
