from lightning import LightningModule

from sigk.models.inpainting_model_shuffle import InpaintingModelShuffle
from sigk.loss.inpainting_loss import InpaintingLoss
from sigk.utils.training_utils import tensor_value_force_assert

from torch import Tensor, uint8, float32
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss

from torchvision.transforms.functional import to_tensor

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

from cv2 import cvtColor, COLOR_BGR2RGB, Mat, inpaint, INPAINT_TELEA


from numpy import mean

from typing import Optional


class Inpainting(LightningModule):
    def __init__(
        self,
        epochs_per_test: int,
        test_on_first_epoch: int,
        learning_rate: int,
        scheduler_params: dict[str, float | int | str],
        embedding_features: int,
        attention_heads: int,
        latent_size: int | tuple[int, int],
        loss_args: Optional[dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.__epochs_per_test = epochs_per_test
        self.__test_on_first_epoch = test_on_first_epoch
        self.__model = InpaintingModelShuffle(
            embedding_features=embedding_features,
            attention_heads=attention_heads,
            latent_size=latent_size,
        )
        self.__learning_rate = learning_rate
        self.__scheduler_params = scheduler_params
        self.__loss = InpaintingLoss() if not loss_args else InpaintingLoss(**loss_args)

    def training_step(
        self, batch: tuple[tuple[Tensor, Tensor], str], batch_idx: int
    ) -> Tensor:
        (x, mask), _ = batch
        x_hat = self.__model(x * mask, mask)
        tensor_value_force_assert(x_hat)
        loss, stats = self.__loss(x, x_hat, mask)
        self.log("train loss", loss, batch_size=x.shape[0])
        for stat, value in stats.items():
            self.log(f"train {stat}", value, batch_size=x.shape[0])
        tensor_value_force_assert(loss)
        return loss

    def __validation_step(self, x: Tensor, mask: Tensor) -> None:
        x_hat = self.__model(x * mask, mask)
        tensor_value_force_assert(x_hat)
        loss, stats = self.__loss(x, x_hat, mask)
        self.log(
            "validation loss", loss, add_dataloader_idx=False, batch_size=x.shape[0]
        )
        tensor_value_force_assert(loss)
        for stat, value in stats.items():
            self.log(
                f"validation {stat}",
                value,
                add_dataloader_idx=False,
                batch_size=x.shape[0],
            )

    def __test_step(self, x: Tensor, mask: Tensor, image_path: str) -> None:
        x_hat = self.__model(x * mask, mask)
        self.logger.experiment.add_image(
            f"test_images/{image_path.split('/')[-1]}_recon",
            x_hat.squeeze(0),
            self.current_epoch,
        )
        self.logger.experiment.add_image(
            f"test_images/{image_path.split('/')[-1]}_masked",
            (x * mask).squeeze(0),
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
            (x, mask), (path,) = batch
            if (
                self.current_epoch == 0
                and self.__test_on_first_epoch
                or ((self.current_epoch + 1) % self.__epochs_per_test == 0)
            ):
                self.__test_step(x, mask, path)
        else:
            raise ValueError(f"({dataloader_idx}) is not a valid dataloader index")

    @staticmethod
    def __torch_to_opencv(tensor: Tensor) -> Mat:
        transformed = (tensor * 255).to(uint8)
        if transformed.dim() != 3:
            transformed = transformed.squeeze(0)
        array = transformed.cpu().permute(1, 2, 0).numpy()
        if tensor.dim() == 3:
            return array
        return cvtColor(array, COLOR_BGR2RGB)

    @staticmethod
    def __opencv_to_torch(image: Mat) -> Tensor:
        return to_tensor(cvtColor(image, COLOR_BGR2RGB)).unsqueeze(0).to(float32)

    def on_test_start(self) -> None:
        self.__stats = {"psnr": [], "ssim": [], "sse": [], "lpips": []}
        self.__base_stats = {"psnr": [], "ssim": [], "sse": [], "lpips": []}
        self.__metrics = {
            "psnr": PeakSignalNoiseRatio(data_range=1),
            "lpips": LearnedPerceptualImagePatchSimilarity(),
            "sse": MSELoss(reduction="sum"),
            "ssim": StructuralSimilarityIndexMeasure(),
        }
        return super().on_test_start()

    def __get_baseline(self, mask: Tensor, masked: Tensor) -> Tensor:
        return self.__opencv_to_torch(
            inpaint(
                self.__torch_to_opencv(masked),
                self.__torch_to_opencv(1.0 - mask[:, 0, ...]),
                32,
                INPAINT_TELEA,
            )
        ).to(masked.device)

    def __calculate_stats(self, x: Tensor, x_hat: Tensor, baseline: Tensor) -> None:
        for stat, metric in self.__metrics.items():
            self.__stats[stat].append(metric.to(self.device)(x, x_hat).cpu().item())
            self.__base_stats[stat].append(
                metric.to(self.device)(x, baseline).cpu().item()
            )

    def test_step(self, batch: tuple[tuple[Tensor, Tensor], str]) -> None:
        (x, mask), (path,) = batch
        masked_x = x * mask
        x_hat = (self.__model(masked_x, mask) * 255).to(uint8).to(float32) / 255.0
        baseline = self.__get_baseline(mask, masked_x)
        self.__calculate_stats(x, x_hat, baseline)
        self.logger.experiment.add_image(
            f"inference_images/{path.split('/')[-1]}_orig",
            x.squeeze(0),
            0,
        )
        self.logger.experiment.add_image(
            f"inference_images/{path.split('/')[-1]}_masked",
            masked_x.squeeze(0),
            0,
        )
        self.logger.experiment.add_image(
            f"inference_images/{path.split('/')[-1]}_recon", x_hat.squeeze(0), 0
        )
        self.logger.experiment.add_image(
            f"inference_images/{path.split('/')[-1]}_telea", baseline.squeeze(0), 0
        )

    def on_test_end(self) -> None:
        super().on_test_end()
        print("Model")
        for stat, values in self.__stats.items():
            print(f"{stat} - {mean(values)}")
        print("INPAINT_TELEA")
        for stat, values in self.__base_stats.items():
            print(f"{stat} - {mean(values)}")

    def configure_optimizers(self) -> list[Adam | ReduceLROnPlateau | str]:
        adam = Adam(self.parameters(), lr=self.__learning_rate)
        plateau = ReduceLROnPlateau(adam, **self.__scheduler_params)
        return {
            "optimizer": adam,
            "lr_scheduler": plateau,
            "monitor": "validation loss",
        }
