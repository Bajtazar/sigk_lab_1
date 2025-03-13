from torch import (
    Tensor,
    device as tensor_device,
    dtype as tensor_dtype,
    tensor as tensor_from_array_ctor,
)

from typing import Optional


def pad_value_with_sequence(
    value: any, filler: any, size: int, value_pos: int
) -> list[any]:
    if (value_pos >= 0 and value_pos >= size) or (
        value_pos < 0 and abs(value_pos) > size
    ):
        raise ValueError(
            "Value position is larger than the sequence itself,"
            f" value_pos=({value_pos}) and size=({size})"
        )
    if value_pos < 0:
        value_pos += size
    head_size = value_pos
    tail_size = size - 1 - value_pos
    return [
        *[filler for _ in range(head_size)],
        value,
        *[filler for _ in range(tail_size)],
    ]


def tensor_from_array(
    tensor: list[float],
    dim: int = 1,
    main_dim: int = -1,
    dtype: Optional[tensor_dtype] = None,
    device: Optional[tensor_device] = None,
) -> Tensor:
    result = tensor_from_array_ctor(tensor, dtype=dtype, device=device)
    if dim < 1:
        raise ValueError(
            f"Dimension of resulting tensor has to be bigger than 1, got ({dim})"
        )
    if dim == 1:
        if main_dim not in [-1, 0]:
            raise ValueError(
                "Main dimension is not equal to -1 or 0 despite tensor having only one"
                " dimension"
            )
        return result
    return result.view(
        pad_value_with_sequence(
            value=len(tensor), filler=1, size=dim, value_pos=main_dim
        )
    )
