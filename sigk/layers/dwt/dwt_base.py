from torch.nn import Module
from torch import Tensor, device as tensor_device, dtype as tensor_dtype, cat

from abc import ABC, abstractmethod
from typing import Any, Optional

from sigk.layers.dwt import Wavelet, PyWavelet
from sigk.utils.tensor_utils import tensor_from_array


class DwtBase(Module, ABC):
    def __init__(
        self,
        channels: int,
        wavelet: Wavelet | PyWavelet,
        dtype: Optional[tensor_dtype] = None,
        device: Optional[tensor_device] = None,
    ) -> None:
        super().__init__()
        self.__wavelet = (
            Wavelet.from_pywt(wavelet) if isinstance(wavelet, PyWavelet) else wavelet
        )
        self._register_wavelet(channels, {"dtype": dtype, "device": device})
        self.__channels = channels
        self._padding = self.__wavelet.dwt_padding

    @abstractmethod
    def _register_wavelet(
        self, channels: int, factory: dict[str, tensor_device | tensor_dtype]
    ) -> None:
        pass

    def __generate_low_pass_filter(
        self,
        dimension: int,
        main_dimension: int,
        factory: dict[str, tensor_device | tensor_dtype],
        analysis: bool = True,
    ) -> Tensor:
        return tensor_from_array(
            (
                self.wavelet.analysis_low_pass
                if analysis
                else self.wavelet.synthesis_low_pass
            ),
            dim=dimension,
            main_dim=main_dimension,
            **factory,
        )

    def __generate_high_pass_filter(
        self,
        dimension: int,
        main_dimension: int,
        factory: dict[str, tensor_device | tensor_dtype],
        analysis: bool = True,
    ) -> Tensor:
        return tensor_from_array(
            (
                self.wavelet.analysis_high_pass
                if analysis
                else self.wavelet.synthesis_high_pass
            ),
            dim=dimension,
            main_dim=main_dimension,
            **factory,
        )

    def _generate_low_pass_kernel(
        self,
        channels: int,
        dimension: int,
        main_dimension: int,
        factory: dict[str, tensor_device | tensor_dtype],
        analysis: bool = True,
    ) -> Tensor:
        low = self.__generate_low_pass_filter(
            dimension, main_dimension, factory, analysis
        )
        return cat([low] * channels, dim=0)

    def _generate_pass_kernel(
        self,
        channels: int,
        dimension: int,
        main_dimension: int,
        factory: dict[str, tensor_device | tensor_dtype],
        analysis: bool = True,
    ) -> Tensor:
        high = self.__generate_high_pass_filter(
            dimension, main_dimension, factory, analysis
        )
        low = self.__generate_low_pass_filter(
            dimension, main_dimension, factory, analysis
        )
        return cat([high, low] * channels, dim=0)

    @abstractmethod
    def forward(self, tensor: Tensor, splitting_mode: str) -> Any:
        pass

    @property
    def wavelet(self) -> Wavelet:
        return self.__wavelet

    @property
    def channels(self) -> int:
        return self.__channels

    def _check_input_tensor(
        self, tensor: Tensor, dimension: int, multiplier: int = 1
    ) -> None:
        if tensor.shape[1] != self.channels * multiplier:
            raise ValueError(
                "Tensor with invalid number of features has been given. Expected"
                f" ({self.__channels}), got ({tensor.shape[1]})"
            )
        if tensor.dim() != dimension:
            raise ValueError(
                f"Invalid tensor given, expected it to have {dimension} dimensions, got"
                f" ({tensor.dim()})"
            )
