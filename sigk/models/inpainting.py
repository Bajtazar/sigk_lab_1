from lightning import LightningModule

from sigk.models.inpainting_model import InpaintingModel
from sigk.loss.inpainting_loss import InpaintingLoss

from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from typing import Optional


class Inpainting(LightningModule):
    def __init__(
        self,
        epochs_per_test: int,
        test_on_first_epoch: int,
        embedding_features: int,
        attention_heads: int,
        latent_size: int | tuple[int, int],
        loss_args: Optional[dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.__epochs_per_test = epochs_per_test
        self.__test_on_first_epoch = test_on_first_epoch
        self.__model = InpaintingModel(
            embedding_features=embedding_features,
            attention_heads=attention_heads,
            latent_size=latent_size,
        )
        self.__loss = InpaintingLoss() if not loss_args else InpaintingLoss(**loss_args)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, mask = batch
        x_hat = self.__model(x * mask, mask)
        loss, stats = self.__loss(x, x_hat, mask)
        self.log("train loss", loss)
        for stat, value in stats.items():
            self.log(f"train {stat}", value)
        return loss

    def __validation_step(self, x: Tensor, mask: Tensor) -> None:
        x_hat = self.__model(x * mask, mask)
        loss, stats = self.__loss(x, x_hat, mask)
        self.log("validation loss", loss, add_dataloader_idx=False)
        for stat, value in stats.items():
            self.log(f"validation {stat}", value, add_dataloader_idx=False)

    def __test_step(self, x: Tensor, mask: Tensor, image_path: str) -> None:
        x_hat = self.__model(x * mask, mask)
        self.logger.experiment.add_image(
            f"test_images/{image_path.split('/')[-1]}_recon",
            x_hat,
            self.current_epoch,
        )
        self.logger.experiment.add_image(
            f"test_images/{image_path.split('/')[-1]}_masked",
            x * mask,
            self.current_epoch,
        )
        self.logger.experiment.add_image(
            f"test_images/{image_path.split('/')[-1]}_orig",
            x,
            self.current_epoch,
        )

    def validation_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if dataloader_idx == self.__VALIDATION_LOADER:
            self.__validation_step(*batch)
        elif dataloader_idx == self.__TEST_LOADER:
            x, path, mask = batch
            if (
                self.current_epoch == 0
                and self.__test_on_first_epoch
                or (self.current_epoch + 1 % self.__epochs_per_test == 0)
            ):
                self.__test_step(x, mask, path)
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
