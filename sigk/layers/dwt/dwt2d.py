from sigk.layers.dwt.dwt_base import DwtBase
from sigk.layers.dwt.wavelet import Wavelet, PyWavelet

from torch.nn.functional import pad, conv2d
from torch import Tensor, device as tensor_device, dtype as tensor_dtype, stack

from sigk.utils.tensor_utils import pad_value_with_sequence


class Dwt2D(DwtBase):
    __VALID_PADDING_MODES: list[str] = ["zeros", "reflect", "replicate", "circular"]

    def __init__(
        self,
        channels: int,
        wavelet: Wavelet | PyWavelet,
        padding_mode: str = "zeros",
        dtype: tensor_dtype | None = None,
        device: tensor_device | None = None,
    ) -> None:
        super().__init__(channels=channels, wavelet=wavelet, dtype=dtype, device=device)
        if padding_mode not in self.__VALID_PADDING_MODES:
            raise ValueError(
                f"Invalid padding mode, got ({padding_mode}), expected one from"
                f" ({self.__VALID_PADDING_MODES})"
            )
        self.__padding_mode = padding_mode

    def _register_wavelet(
        self,
        channels: int,
        factory: dict[str, tensor_device | tensor_dtype],
    ) -> None:
        first_pass_kernel = self._generate_pass_kernel(
            channels, dimension=4, main_dimension=-2, factory=factory
        )
        second_pass_kernel = self._generate_pass_kernel(
            2 * channels, dimension=4, main_dimension=-1, factory=factory
        )
        self.register_buffer("first_pass_kernel", first_pass_kernel)
        self.register_buffer("second_pass_kernel", second_pass_kernel)

    @staticmethod
    def __split_tensor_into_bands(
        tensor: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ll_band = tensor[..., 3::4, :, :]
        lh_band = tensor[..., 1::4, :, :]
        hl_band = tensor[..., 2::4, :, :]
        hh_band = tensor[..., ::4, :, :]
        return ll_band, lh_band, hl_band, hh_band

    def _apply_postprocessing(
        self, tensor: Tensor, splitting_mode: str
    ) -> Tensor | tuple[Tensor, Tensor, Tensor, Tensor]:
        if splitting_mode == "none":
            return tensor
        elif splitting_mode == "separate":
            return self.__split_tensor_into_bands(tensor)
        elif splitting_mode == "dimension":
            return stack(self.__split_tensor_into_bands(tensor), dim=2)
        else:
            raise ValueError(
                f'Invalid splitting mode, got ({splitting_mode}), expected (["none",'
                ' "separate", "dimension"])'
            )

    def _apply_padding(self, tensor: Tensor, mode: str, value: int = 0) -> Tensor:
        return pad(tensor, [self._padding] * 4, mode=mode, value=value)

    def _perform_dwt_pass(
        self, tensor: Tensor, kernel: Tensor, position: int, groups: int
    ) -> Tensor:
        stride = pad_value_with_sequence(value=2, filler=1, size=2, value_pos=position)
        if self.padding_mode == "zeros":
            padding = pad_value_with_sequence(
                value=self._padding, filler=0, size=2, value_pos=position
            )
            return conv2d(tensor, kernel, stride=stride, groups=groups, padding=padding)
        return conv2d(tensor, kernel, stride=stride, groups=groups)

    def forward(
        self, tensor: Tensor, splitting_mode: str = "separate"
    ) -> Tensor | tuple[Tensor, Tensor, Tensor, Tensor]:
        self._check_input_tensor(tensor, dimension=4)
        if self.padding_mode != "zeros":
            tensor = self._apply_padding(tensor, mode=self.padding_mode)
        # Transform rows, then columns
        first_pass = self._perform_dwt_pass(
            tensor, kernel=self.first_pass_kernel, position=-2, groups=self.channels
        )
        result = self._perform_dwt_pass(
            first_pass,
            kernel=self.second_pass_kernel,
            position=-1,
            groups=2 * self.channels,
        )
        return self._apply_postprocessing(result, splitting_mode)

    @property
    def padding_mode(self) -> str:
        return self.__padding_mode
