from lightning import LightningModule

from sigk.models.inpainting_model import InpaintingModel
from sigk.loss.inpainting_loss import InpaintingLoss

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from typing import Optional


class Inpainting(LightningModule):
    def __init__(
        self,
        embedding_features: int,
        attention_heads: int,
        latent_size: int | tuple[int, int],
        loss_args: Optional[dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.__model = InpaintingModel(
            embedding_features=embedding_features,
            attention_heads=attention_heads,
            latent_size=latent_size,
        )
        self.__loss = InpaintingLoss() if not loss_args else InpaintingLoss(**loss_args)

    def configure_optimizers(self) -> list[Adam | ReduceLROnPlateau | str]:
        adam = Adam(self.parameters(), lr=self.__learning_rate)
        plateau = ReduceLROnPlateau(adam, **self.__scheduler_params)
        return {
            "optimizer": adam,
            "lr_scheduler": plateau,
            "monitor": "validation loss",
        }
