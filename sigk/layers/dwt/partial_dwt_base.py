from torch import Tensor, ones_like, tensor

from numpy import prod


class PartialDwtBase:
    def __init__(self, epsilon: float = 1e-8) -> None:
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
