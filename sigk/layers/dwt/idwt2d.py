from torch.nn.functional import conv_transpose2d
from torch import (
    Tensor,
    device as tensor_device,
    dtype as tensor_dtype,
    empty,
    chunk,
)

from sigk.layers.dwt.dwt_base import DwtBase
from sigk.utils.tensor_utils import pad_value_with_sequence


class IDwt2D(DwtBase):
    def _register_wavelet(
        self,
        channels: int,
        factory: dict[str, tensor_device | tensor_dtype],
    ) -> None:
        first_pass_kernel = self._generate_pass_kernel(
            channels, dimension=4, main_dimension=-2, factory=factory, analysis=False
        )
        second_pass_kernel = self._generate_pass_kernel(
            2 * channels,
            dimension=4,
            main_dimension=-1,
            factory=factory,
            analysis=False,
        )
        self.register_buffer("first_pass_kernel", first_pass_kernel)
        self.register_buffer("second_pass_kernel", second_pass_kernel)

    def __perform_idwt_pass(
        self, tensor: Tensor, kernel: Tensor, position: int, groups: int
    ) -> Tensor:
        stride = pad_value_with_sequence(value=2, filler=1, size=2, value_pos=position)
        padding = pad_value_with_sequence(
            value=self._padding, filler=0, size=2, value_pos=position
        )
        return conv_transpose2d(
            tensor, kernel, stride=stride, groups=groups, padding=padding
        )

    def __reconstruct_tensor(
        self, ll: Tensor, lh: Tensor, hl: Tensor, hh: Tensor
    ) -> Tensor:
        self._check_input_tensor(ll, dimension=4)
        self._check_input_tensor(lh, dimension=4)
        self._check_input_tensor(hl, dimension=4)
        self._check_input_tensor(hh, dimension=4)
        reconstruction = empty(
            *ll.shape[:-3],
            4 * ll.shape[-3],
            *ll.shape[-2:],
            dtype=ll.dtype,
            device=ll.device,
        )
        reconstruction[..., 3::4, :, :] = ll
        reconstruction[..., 1::4, :, :] = lh
        reconstruction[..., 2::4, :, :] = hl
        reconstruction[..., ::4, :, :] = hh
        return reconstruction

    def _apply_preprocessing(
        self,
        tensor: tuple[Tensor, Tensor, Tensor, Tensor] | Tensor,
        splitting_mode: str,
    ) -> Tensor:
        if splitting_mode == "none":
            self._check_input_tensor(tensor, dimension=4, multiplier=4)
            return tensor
        elif splitting_mode == "separate":
            return self.__reconstruct_tensor(*tensor)
        elif splitting_mode == "dimension":
            ll, lh, hl, hh = chunk(tensor, chunks=4, dim=2)
            return self.__reconstruct_tensor(
                ll.squeeze(2), lh.squeeze(2), hl.squeeze(2), hh.squeeze(2)
            )
        else:
            raise ValueError(
                f'Invalid splitting mode, got ({splitting_mode}), expected (["none",'
                ' "separate", "dimension"])'
            )

    def forward(
        self,
        tensor: tuple[Tensor, Tensor, Tensor, Tensor] | Tensor,
        splitting_mode: str = "separate",
    ) -> Tensor:
        tensor = self._apply_preprocessing(tensor, splitting_mode)
        second_pass = self.__perform_idwt_pass(
            tensor,
            kernel=self.second_pass_kernel,
            position=-1,
            groups=2 * self.channels,
        )
        return self.__perform_idwt_pass(
            second_pass,
            kernel=self.first_pass_kernel,
            position=-2,
            groups=self.channels,
        )
