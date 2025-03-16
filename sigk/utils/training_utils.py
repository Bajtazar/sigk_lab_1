from torch import Tensor, isfinite, all as tensor_all


class InvalidModelStateException(Exception):
    def __init__(self, tag: str | None) -> None:
        msg = "Model has been observed in the invalid state"
        if tag:
            msg += f", additional info: {tag}"
        super().__init__(msg)


def tensor_value_force_assert(tensor: Tensor, tag: str | None = None) -> None:
    if not tensor_all(isfinite(tensor)):
        raise InvalidModelStateException(tag)
