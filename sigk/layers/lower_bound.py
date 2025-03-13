from torch import Tensor, max as tensor_max, tensor
from torch.nn import Module
from torch.autograd import Function
from torch.autograd.function import FunctionCtx


class LowerBoundFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, input_tensor: Tensor, bound: Tensor) -> Tensor:
        ctx.save_for_backward(input_tensor, bound)
        return tensor_max(input_tensor, bound)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> tuple[Tensor, None]:
        input_tensor, bound = ctx.saved_tensors
        pass_through = (input_tensor >= bound) | (grad_output < 0)
        return pass_through * grad_output, None


class LowerBound(Module):
    def __init__(self, bound: Tensor | float | int) -> None:
        super().__init__()
        self.register_buffer(
            "bound",
            bound.clone() if isinstance(bound, Tensor) else tensor([float(bound)]),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return LowerBoundFunction.apply(input_tensor, self.bound)
