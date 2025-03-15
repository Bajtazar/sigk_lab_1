from torch import ones_like


class PartialDwtBase:
    def __init__(self) -> None:
        self.register_buffer(
            "first_pass_mask_coeffs", ones_like(self.first_pass_kernel)
        )
        self.register_buffer(
            "second_pass_mask_coeffs", ones_like(self.second_pass_kernel)
        )
