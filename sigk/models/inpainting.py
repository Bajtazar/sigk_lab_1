from lightning import LightningModule

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Inpainting(LightningModule):
    def __init__(self) -> None:
        super().__init__()

    def configure_optimizers(self) -> list[Adam | ReduceLROnPlateau | str]:
        adam = Adam(self.parameters(), lr=self.__learning_rate)
        plateau = ReduceLROnPlateau(adam, **self.__scheduler_params)
        return {
            "optimizer": adam,
            "lr_scheduler": plateau,
            "monitor": "validation loss",
        }
