from __future__ import annotations
from typing import overload

from pywt import Wavelet as PyWavelet


class Wavelet:
    @overload
    def __init__(
        self,
        name: str,
        analysis_low_pass: list[float],
        analysis_high_pass: list[float],
        synthesis_low_pass: list[float],
        synthesis_high_pass: list[float],
    ) -> None:
        ...

    @overload
    def __init__(self, wavelet: PyWavelet) -> None:
        ...

    @overload
    def __init__(self, name: str) -> None:
        ...

    def __init__(self, *args) -> None:
        if len(args) == 1:
            self.__wavelet = PyWavelet(args[0]) if isinstance(args[0], str) else args[0]
        elif len(args) == 5:
            self.__wavelet = PyWavelet(
                args[0],
                args[1:],
            )
        else:
            raise ValueError(
                f"Number of arguments is invalid, got ({len(args)}), expected 1 or 5"
            )

    @property
    def as_pywt(self) -> PyWavelet:
        return self.__wavelet

    @property
    def name(self) -> str:
        return self.__wavelet.name

    @property
    def analysis_low_pass(self) -> list[float]:
        return self.__wavelet.inverse_filter_bank[2]

    @property
    def analysis_high_pass(self) -> list[float]:
        return self.__wavelet.inverse_filter_bank[3]

    @property
    def synthesis_low_pass(self) -> list[float]:
        return self.__wavelet.filter_bank[2]

    @property
    def synthesis_high_pass(self) -> list[float]:
        return self.__wavelet.filter_bank[3]

    @property
    def dwt_padding(self) -> int:
        return len(self.synthesis_low_pass) - 2
