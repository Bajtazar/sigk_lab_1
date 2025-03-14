from torch import Tensor, tensor
from torch.nn import L1Loss, Module, Sequential, ParameterList

from torchvision.models import vgg16

from numpy import prod


class Vgg16Embedding(Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", tensor([0.485, 0.456, 0.406]).view(-1, 1, 1))
        self.register_buffer("stddev", tensor([0.229, 0.224, 0.225]).view(-1, 1, 1))
        model = vgg16(pretrained=True)
        self.__embeddings = ParameterList(
            [
                Sequential(*model.features[:5]),
                Sequential(*model.features[5:10]),
                Sequential(*model.features[10:17]),
            ]
        )

        for feature in self.__embeddings:
            for param in feature.parameters():
                param.requires_grad = False

    def __normalize(self, tensor: Tensor) -> Tensor:
        return (tensor - self.mean) / self.stddev

    def forward(self, tensor: Tensor) -> list[Tensor]:
        tensor = self.__normalize(tensor)
        features = []
        for embedding in self.__embeddings:
            tensor = embedding(tensor)
            features.append(tensor)
        return features


class InpaintingLoss:
    def __init__(
        self,
        valid_lambda: float = 1.0,
        hole_lambda: float = 6.0,
        smooth_lamda: float = 0.1,
        perceptual_lambda: float = 0.05,
        style_lambda: float = 120,
    ) -> None:
        super().__init__()
        self.__valid_lambda = valid_lambda
        self.__hole_lambda = hole_lambda
        self.__smooth_lambda = smooth_lamda
        self.__perceptual_lambda = perceptual_lambda
        self.__style_lambda = style_lambda
        self.__l1_loss = L1Loss()
        self.__embedding = Vgg16Embedding()

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
    def perceptual_lambda(self) -> float:
        return self.__perceptual_lambda

    @property
    def style_lambda(self) -> float:
        return self.__style_lambda

    def __get_composition(
        self, x: Tensor, x_hat: Tensor, mask: Tensor, hole_mask: Tensor
    ) -> Tensor:
        return mask * x + hole_mask * x_hat

    def __valid_loss(self, x: Tensor, x_hat: Tensor, mask: Tensor) -> Tensor:
        return self.valid_lambda * self.__l1_loss(x * mask, x_hat * mask)

    def __hole_loss(self, x: Tensor, x_hat: Tensor, hole_mask: Tensor) -> Tensor:
        return self.hole_lambda * self.__l1_loss(x * hole_mask, x_hat * hole_mask)

    def __total_variation_loss(self, composition: Tensor) -> Tensor:
        horizontal_loss = self.__l1_loss(composition[..., :-1], composition[..., 1:])
        vertical_loss = self.__l1_loss(
            composition[..., :-1, :], composition[..., 1:, :]
        )
        return self.smooth_lambda * (horizontal_loss + vertical_loss)

    def __perceptual_loss(
        self, embedded_x: Tensor, embedded_x_hat: Tensor, embedded_comp: Tensor
    ) -> Tensor:
        loss = 0
        for i in range(3):
            loss += self.__l1_loss(embedded_x[i], embedded_x_hat[i])
            loss += self.__l1_loss(embedded_x[i], embedded_comp[i])
        return self.__perceptual_lambda * loss

    def __style_loss(
        self, embedded_x: Tensor, embedded_x_hat: Tensor, embedded_comp: Tensor
    ) -> Tensor:
        loss = 0
        for i in range(3):
            x_gram = self.__calculate_gram_matrix(embedded_x[i])
            x_hat_gram = self.__calculate_gram_matrix(embedded_x_hat[i])
            comp_gram = self.__calculate_gram_matrix(embedded_comp[i])
            loss += self.__l1_loss(x_hat_gram, x_gram)
            loss += self.__l1_loss(comp_gram, x_gram)
        return self.style_lambda * loss

    def __call__(
        self, x: Tensor, x_hat: Tensor, mask: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        hole_mask = 1 - mask
        composition = self.__get_composition(x, x_hat, mask, hole_mask)
        embedded_x = self.__embedding(x)
        embedded_x_hat = self.__embedding(x_hat)
        embedded_comp = self.__embedding(composition)
        losses = {
            "valid_loss": self.__valid_loss(x, x_hat, mask),
            "hole_loss": self.__hole_loss(x, x_hat, hole_mask),
            "total_variation_loss": self.__total_variation_loss(composition),
            "perceptual_loss": self.__perceptual_loss(
                embedded_x, embedded_x_hat, embedded_comp
            ),
            "style_loss": self.__style_loss(embedded_x, embedded_x_hat, embedded_comp),
        }
        return sum(losses.values()), losses

    @staticmethod
    def __calculate_gram_matrix(matrix: Tensor) -> Tensor:
        left = matrix.flatten(dim=-2)
        right = left.transpose(-2, -1)
        return left @ right / prod(left.shape[1:])
