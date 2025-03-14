from torch import Tensor
from torch.nn import Module

from numpy import prod


class InpaintingLoss(Module):
    def __init__(
        self,
        valid_lambda: float = 1.0,
        hole_lambda: float = 6.0,
        smooth_lamda: float = 0.1,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.__valid_lambda = valid_lambda
        self.__hole_lambda = hole_lambda
        self.__smooth_lambda = smooth_lamda
        self.__epsilon = epsilon
        self.__embedding = None

    @property
    def valid_lambda(self) -> float:
        return self.__valid_lambda

    @property
    def hole_lambda(self) -> float:
        return self.__hole_lambda

    @property
    def smooth_lambda(self) -> float:
        return self.__smooth_lambda

    @property
    def epsilon(self) -> float:
        return self.__epsilon

    def __get_composition(
        self, x: Tensor, x_hat: Tensor, mask: Tensor, hole_mask: Tensor
    ) -> Tensor:
        return mask * x + hole_mask * x_hat

    def __valid_loss(self, x: Tensor, x_hat: Tensor, mask: Tensor) -> Tensor:
        return self.valid_lambda * abs(x * mask - x_hat * mask).mean()

    def __hole_loss(self, x: Tensor, x_hat: Tensor, hole_mask: Tensor) -> Tensor:
        return self.hole_lambda * abs(x * hole_mask - x_hat * hole_mask).mean()

    def __total_variation_loss(self, composition: Tensor) -> Tensor:
        horizontal_loss = abs(composition[..., :-1] - composition[..., 1:]).mean()
        vertical_loss = abs(composition[..., :-1, :] - composition[..., 1:, :]).mean()
        return self.smooth_lambda * (horizontal_loss + vertical_loss)

    def __call__(self, x: Tensor, x_hat: Tensor, mask: Tensor) -> Tensor:
        hole_mask = 1 - mask
        composition = self.__get_composition(x, x_hat, mask, hole_mask)
        valid_loss = self.__valid_loss(x, x_hat, mask)
        hole_loss = self.__hole_loss(x, x_hat, hole_mask)
        total_variation_loss = self.__total_variation_loss(composition)

    def __calculate_gram_matrix(matrix: Tensor) -> Tensor:
        left = matrix.flatten(dim=-2)
        right = left.transpose(-2, -1)
        return left @ right / prod(left.shape[1:])
