from torch.nn import Module, Parameter
from torch import (
    Tensor,
    complex32,
    complex64,
    complex128,
    dtype as tensor_dtype,
    get_default_dtype,
)
from torch.fft import rfftn, irfftn, fftn, ifftn

from typing import Sequence


COMPLEX_TYPES: type = [complex32, complex64, complex128]


class __SpectralMeta(type):
    def __call__(cls, *args, **kwargs):
        if not issubclass(cls, Module):
            raise ValueError("Given class is not a derivative of torch.nn.Module")
        initialized = type.__call__(cls, *args, **kwargs)
        initialized.convert_weights_to_spectral()


def spectral(cls: object) -> object:
    class Spectral(cls, metaclass=__SpectralMeta):
        def convert_weights_to_spectral(self) -> None:
            original_weight = self.weight
            if not hasattr(self, "weight"):
                raise ValueError("weight tensor not found - convertion to spectral is impossible")
            del self._parameters["weight"]
            if not hasattr(self, "dtype"):
                self.dtype = original_weight.dtype
            self.register_parameter("spectral_weight",
                                    self._to_spectral(original_weight))
            weight_property = property(self.__weight_getter)
            weight_property.setter(self.__weight_setter)
            setattr(self, "weight", weight_property)

        def __weight_getter(self) -> Tensor:
            return self._from_spectral(self.spectral_weight)

        def __weight_setter(self, value: Tensor) -> Tensor:
            self.spectral_weight = self._to_spectral(value)

        def _to_spectral(self, weights: Tensor) -> Tensor:
            if self.is_spectral_weight_complex:
                return fftn(input=weights, s=self.__signal_size, norm="ortho")
            return rfftn(input=weights, s=self.__signal_size, norm="ortho")

        def _from_spectral(self, weights: Tensor) -> Tensor:
            if self.is_spectral_weight_complex:
                return ifftn(input=weights, s=self.__signal_size, norm="ortho")
            return irfftn(input=weights, s=self.__signal_size, norm="ortho")

        @property
        def is_spectral_weight_complex(self) -> bool:
            obj_type = self.dtype
            if obj_type is None:
                obj_type = get_default_dtype()
            return obj_type in COMPLEX_TYPES

        @property
        def __signal_size(self) -> Sequence[int] | int:
            if hasattr(self, "kernel_size"):  # Convolution
                return self.kernel_size
            elif hasattr(self, "in_features"):  # Linear layer
                return self.in_features
            raise ValueError(
                "Invalid base class detected. Please use with Convolution or Linear layers"
                " or overwrite 'signal_size' property to return valid weights length in the"
                " transformed dimension"
            )

    return Spectral
