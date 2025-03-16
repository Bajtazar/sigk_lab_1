#!/usr/bin/env python3
from sigk.models.debluring import Debluring
from sigk.datasets.deblur_image_dataset import DeblurImageDataset

from click import command, argument, Path, option

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    RichProgressBar,
    ModelCheckpoint,
    LearningRateMonitor,
    Callback,
    EarlyStopping,
)
from lightning import Trainer, seed_everything

from torch.utils.data import DataLoader
from torch import use_deterministic_algorithms

from os.path import exists
from os import listdir


IMAGE_SIZE: int = 256
TRAIN_VALID_SLIT_COEFF: float = 0.8
KERNEL_SIZES: list[int] = [3, 5]
SIGMA_MIN: float = 0.1
SIGMA_MAX: float = 2


def get_data_loaders(
    test_dataset_path: str, workers_num: int, kernel_size: int
) -> tuple[DataLoader, DataLoader]:
    return DataLoader(
        DeblurImageDataset(
            root=test_dataset_path,
            image_size=IMAGE_SIZE,
            kernel_sizes=kernel_size,
            sigma_max=SIGMA_MAX,
            sigma_min=SIGMA_MIN,
        ),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=workers_num,
    )


def build_callbacks(run_name: str, period: int, epochs: int) -> list[Callback]:
    return [
        RichProgressBar(),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor="lr-Adam",
            mode="min",
            stopping_threshold=1.2e-6,
            check_on_train_epoch_end=True,
            patience=epochs,  # Disable early stopping due to the learning rate not changing
        ),
        ModelCheckpoint(
            dirpath=f"models/{run_name}/checkpoints", every_n_epochs=period
        ),
    ]


def checkpoint_path(run_name: str) -> str | None:
    checkpoints_dir = f"models/{run_name}/checkpoints"
    if not exists(checkpoints_dir):
        return None
    last = f"{checkpoints_dir}/last.ckpt"
    if exists(last):
        return last
    files = list(listdir(checkpoints_dir))
    if not files:
        return None
    mapped = {file: int(file.split(".")[0].split("=")[-1]) for file in files}
    return f"{checkpoints_dir}/{max(mapped, key=mapped.get)}"


@command()
@argument("run_name", type=str)
@argument("test_dataset_path", type=Path(exists=True, readable=True))
@option("--workers_num", type=int, default=0)
@option("--epochs", type=int, default=2000)
@option("--devices", type=int, default=1)
@option("--accelerator", type=str, default="gpu")
@option("--learning_rate", type=float, default=1e-4)
@option("--epochs_per_test", type=int, default=30)
@option("--checkpoint_period", type=int, default=5)
@option("--seed", type=int, default=0xBAAD)
@option(
    "-an",
    "--allow-nondeterministic",
    type=bool,
    is_flag=True,
    default=False,
)
def main(
    run_name: str,
    test_dataset_path: str,
    workers_num: int,
    epochs: int,
    devices: int,
    accelerator: str,
    allow_nondeterministic: bool,
    learning_rate: float,
    epochs_per_test: int,
    checkpoint_period: int,
    seed: int,
) -> None:
    seed_everything(seed)
    model = Debluring(
        learning_rate=learning_rate,
        scheduler_params=dict(mode="min", factor=0.5, patience=100, eps=1.2e-6),
        epochs_per_test=epochs_per_test,
        test_on_first_epoch=True,
        input_channels=3,
        output_channels=3,
        latent_channels=64,
    )

    logger = TensorBoardLogger(
        save_dir=".",
        name="logs",
        version=run_name,
    )
    trainer = Trainer(
        max_epochs=epochs,
        devices=devices,
        accelerator=accelerator,
        deterministic=True,
        logger=logger,
        log_every_n_steps=1,
        callbacks=build_callbacks(run_name, checkpoint_period, epochs),
    )
    use_deterministic_algorithms(True, warn_only=allow_nondeterministic)
    for kernel_size in KERNEL_SIZES:
        print(f"For kernel_size={kernel_size}")
        trainer.test(
            model=model,
            ckpt_path=checkpoint_path(run_name),
            dataloaders=get_data_loaders(
                test_dataset_path=test_dataset_path,
                workers_num=workers_num,
                kernel_size=kernel_size,
            ),
        )


if __name__ == "__main__":
    main()
