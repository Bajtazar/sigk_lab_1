from typing import Any, Optional

from torch import (
    Tensor,
    ones,
    device as tensor_device,
    dtype as tensor_dtype,
    tensor,
    sqrt,
    eye,
    rsqrt,
)
from torch.nn import Module, Parameter
from torch.nn.functional import conv2d

from sigk.layers.lower_bound import LowerBound


class GDNBase(Module):
    def __init__(
        self,
        channels: int,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
        reparam_offset: float = 2**-18,
        device: Optional[tensor_device] = None,
        dtype: Optional[tensor_dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.__initialize_buffers(beta_min, reparam_offset, **factory_kwargs)
        self.__initialize_parameters(channels, gamma_init, **factory_kwargs)

    def __initialize_buffers(
        self, beta_min: float, reparam_offset: float, **kwargs: Any
    ) -> None:
        reparam_offset = tensor([reparam_offset], **kwargs)
        self.register_buffer("pedestal", reparam_offset**2)
        self.__beta_bound = LowerBound(sqrt(beta_min + self.pedestal))
        self.__gamma_bound = LowerBound(reparam_offset)

    def __initialize_parameters(
        self, channels: int, gamma_init: float, **kwargs: Any
    ) -> None:
        self.beta = Parameter(sqrt(ones(channels, **kwargs) + self.pedestal))
        self.gamma = Parameter(
            sqrt(eye(channels, **kwargs) * gamma_init + self.pedestal)
        )

    def _assert_input_size(self, input_tensor: Tensor) -> Tensor:
        if input_tensor.dim() != 4:
            raise ValueError(
                "Only (B,C,H,W) tensors are accepted, got tensor with"
                f" ({input_tensor.dim()}) dimensions"
            )
        if input_tensor.shape[-3] != self.beta.shape[0]:
            raise ValueError(
                "Invalid number of channels in the inserted tensor, got"
                f" ({input_tensor.shape[-3]}), expected ({self.beta.shape[0]})"
            )

    @property
    def _projected_weights(self) -> tuple[Tensor, Tensor]:
        channels = self.beta.shape[0]
        beta = self.__beta_bound(self.beta) ** 2 - self.pedestal
        gamma = self.__gamma_bound(self.gamma) ** 2 - self.pedestal
        gamma_view = gamma.view(channels, channels, 1, 1)
        return gamma_view, beta

    def _calculate_norm(self, input_tensor: Tensor) -> Tensor:
        self._assert_input_size(input_tensor)
        gamma, beta = self._projected_weights
        return conv2d(input_tensor**2, gamma, beta)


class GDN(GDNBase):
    def forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor * rsqrt(self._calculate_norm(input_tensor))


class IGDN(GDNBase):
    def forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor * sqrt(self._calculate_norm(input_tensor))
