from lightning import LightningModule

from sigk.models.debluring_model import DebluringModel
from sigk.utils.training_utils import tensor_value_force_assert

from torch import Tensor, uint8, float32
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

from numpy import mean


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
        (x, x_blurred), _ = batch
        x_hat = self.__model(x_blurred)
        tensor_value_force_assert(x_hat)
        loss = self.__loss(x, x_hat)
        self.log("train loss", loss, batch_size=x.shape[0])
        tensor_value_force_assert(loss)
        return loss

    def __validation_step(self, x: Tensor, x_blurred: Tensor) -> None:
        x_hat = self.__model(x_blurred)
        tensor_value_force_assert(x_hat)
        loss = self.__loss(x, x_hat)
        self.log(
            "validation loss", loss, add_dataloader_idx=False, batch_size=x.shape[0]
        )
        tensor_value_force_assert(loss)

    def __test_step(self, x: Tensor, x_blurred: Tensor, image_path: str) -> None:
        x_hat = self.__model(x_blurred)
        self.logger.experiment.add_image(
            f"test_images/{image_path.split('/')[-1]}_recon",
            x_hat.squeeze(0),
            self.current_epoch,
        )
        self.logger.experiment.add_image(
            f"test_images/{image_path.split('/')[-1]}_blur",
            x_blurred.squeeze(0),
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
            (x, x_blurred), (path,) = batch
            if (
                self.current_epoch == 0
                and self.__test_on_first_epoch
                or ((self.current_epoch + 1) % self.__epochs_per_test == 0)
            ):
                self.__test_step(x, x_blurred, path)
        else:
            raise ValueError(f"({dataloader_idx}) is not a valid dataloader index")

    def on_test_start(self) -> None:
        self.__stats = {"psnr": [], "ssim": [], "sse": [], "lpips": []}
        self.__metrics = {
            "psnr": PeakSignalNoiseRatio(data_range=1),
            "lpips": LearnedPerceptualImagePatchSimilarity(),
            "sse": MSELoss(reduction="sum"),
            "ssim": StructuralSimilarityIndexMeasure(),
        }
        return super().on_test_start()

    def test_step(self, batch: tuple[tuple[Tensor, Tensor], str]) -> None:
        (x, blurred_x), (path,) = batch
        x_hat = (self.__model(blurred_x) * 255).to(uint8).to(float32) / 255.0
        for stat, metric in self.__metrics:
            self.__stats[stat].append(metric(x, x_hat))
        self.logger.experiment.add_image(
            f"inference_images/{path.split('/')[-1]}_orig",
            x.squeeze(0),
            0,
        )
        self.logger.experiment.add_image(
            f"inference_images/{path.split('/')[-1]}_recon",
            x_hat.squeeze(0),
            0,
        )

    def on_test_end(self) -> None:
        super().on_test_end()
        for stat, values in self.__stats.items():
            print(f"{stat} - {mean(values)}")

    def configure_optimizers(self) -> list[Adam | ReduceLROnPlateau | str]:
        adam = Adam(self.parameters(), lr=self.__learning_rate)
        plateau = ReduceLROnPlateau(adam, **self.__scheduler_params)
        return {
            "optimizer": adam,
            "lr_scheduler": plateau,
            "monitor": "validation loss",
        }
