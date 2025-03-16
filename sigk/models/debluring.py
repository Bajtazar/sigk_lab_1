from lightning import LightningModule

from sigk.models.debluring_model import DebluringModel

from torch import Tensor, isfinite, all as tensor_all
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss


class InvalidModelStateException(Exception):
    def __init__(self, tag: str | None) -> None:
        msg = "Model has been observed in the invalid state"
        if tag:
            msg += f", additional info: {tag}"
        super().__init__(msg)


def tensor_value_force_assert(tensor: Tensor, tag: str | None = None) -> None:
    if not tensor_all(isfinite(tensor)):
        raise InvalidModelStateException(tag)


class Debluring(LightningModule):
    def __init__(
        self,
        epochs_per_test: int,
        test_on_first_epoch: int,
        input_channels: int,
        latent_channels: int,
        output_channels: int,
        learning_rate: int,
        scheduler_params: dict[str, float | int | str],
    ) -> None:
        super().__init__()
        self.__epochs_per_test = epochs_per_test
        self.__test_on_first_epoch = test_on_first_epoch
        self.__model = DebluringModel(
            input_channels=input_channels,
            latent_channels=latent_channels,
            output_channels=output_channels,
        )
        self.__learning_rate = learning_rate
        self.__scheduler_params = scheduler_params
        self.__loss = MSELoss()

    def training_step(self, batch: tuple[Tensor, str], batch_idx: int) -> Tensor:
        x, _ = batch
        x_hat = self.__model(x)
        tensor_value_force_assert(x_hat)
        loss = self.__loss(x, x_hat)
        self.log("train loss", loss, batch_size=x.shape[0])
        tensor_value_force_assert(loss)
        return loss

    def __validation_step(self, x: Tensor) -> None:
        x_hat = self.__model(x)
        tensor_value_force_assert(x_hat)
        loss = self.__loss(x, x_hat)
        self.log(
            "validation loss", loss, add_dataloader_idx=False, batch_size=x.shape[0]
        )
        tensor_value_force_assert(loss)

    def __test_step(self, x: Tensor, image_path: str) -> None:
        x_hat = self.__model(x)
        self.logger.experiment.add_image(
            f"test_images/{image_path.split('/')[-1]}_recon",
            x_hat.squeeze(0),
            self.current_epoch,
        )
        self.logger.experiment.add_image(
            f"test_images/{image_path.split('/')[-1]}_orig",
            (x).squeeze(0),
            self.current_epoch,
        )

    def validation_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if dataloader_idx == 0:
            self.__validation_step(*batch[0])
        elif dataloader_idx == 1:
            x, (path,) = batch
            if (
                self.current_epoch == 0
                and self.__test_on_first_epoch
                or ((self.current_epoch + 1) % self.__epochs_per_test == 0)
            ):
                self.__test_step(x, path)
        else:
            raise ValueError(f"({dataloader_idx}) is not a valid dataloader index")

    def configure_optimizers(self) -> list[Adam | ReduceLROnPlateau | str]:
        adam = Adam(self.parameters(), lr=self.__learning_rate)
        plateau = ReduceLROnPlateau(adam, **self.__scheduler_params)
        return {
            "optimizer": adam,
            "lr_scheduler": plateau,
            "monitor": "validation loss",
        }
