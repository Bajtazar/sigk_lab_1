#!/usr/bin/env python3
from sigk.models.inpainting import Inpainting
from sigk.datasets.inpating_image_dataset import InpaintingImageDataset

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

from torch.utils.data import DataLoader, random_split
from torch import use_deterministic_algorithms

from os.path import exists


IMAGE_SIZE: int = 256
PATCHES: list[int] = [3, 32]
PATCHES_COUNT: int = 16
TRAIN_VALID_SLIT_COEFF: float = 0.8


def get_data_loaders(
    train_dataset_path: str, test_dataset_path: str, batch_size: int, workers_num: int
) -> tuple[DataLoader, DataLoader]:
    dataset = InpaintingImageDataset(
        root=train_dataset_path,
        image_size=IMAGE_SIZE,
        patch_count=PATCHES_COUNT,
        patch_sizes=PATCHES,
    )
    train_dataset_length = int(TRAIN_VALID_SLIT_COEFF * len(dataset))
    train_dataset, validation_dataset = random_split(
        dataset, [train_dataset_length, len(dataset) - train_dataset_length]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=True,
        num_workers=workers_num,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=workers_num,
    )
    test_loader = DataLoader(
        InpaintingImageDataset(
            root=test_dataset_path,
            image_size=IMAGE_SIZE,
            patch_count=PATCHES_COUNT,
            patch_sizes=PATCHES,
        ),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=workers_num,
    )
    return train_loader, [validation_loader, test_loader]


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
    checkpoint_path = f"models/{run_name}/checkpoints/last.ckpt"
    if not exists(checkpoint_path):
        return None
    return checkpoint_path


@command()
@argument("run_name", type=str)
@argument("train_dataset_path", type=Path(exists=True, readable=True))
@argument("test_dataset_path", type=Path(exists=True, readable=True))
@option("--workers_num", type=int, default=0)
@option("--batch_size", type=int, default=16)
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
    train_dataset_path: str,
    test_dataset_path: str,
    workers_num: int,
    batch_size: int,
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
    model = Inpainting(
        learning_rate=learning_rate,
        scheduler_params=dict(mode="min", factor=0.5, patience=20, eps=1.2e-6),
        epochs_per_test=epochs_per_test,
        test_on_first_epoch=True,
        embedding_features=64,
        attention_heads=64,
        latent_size=15,
    )

    train_loader, validation_loader = get_data_loaders(
        train_dataset_path=train_dataset_path,
        test_dataset_path=test_dataset_path,
        batch_size=batch_size,
        workers_num=workers_num,
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
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
        ckpt_path=checkpoint_path(run_name),
    )


if __name__ == "__main__":
    main()
