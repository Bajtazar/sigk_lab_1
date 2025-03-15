#!/usr/bin/env python3
from shutil import rmtree, move as move_file
from os import makedirs, listdir, remove
from subprocess import Popen
from os.path import exists
from typing import Callable

from wget import download as wget_download

from click import command, argument, Path, option


DIV2K_TRAIN_URL: str = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
DIV2K_TRAIN_DIR: str = "div2k_train"
DIV2K_TRAIN_TMP_ZIP: str = "div2k_train_tmp.zip"
DIV2K_TRAIN_TMP_DIR: str = "DIV2K_train_HR"

DIV2K_VALIDATION_URL: str = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
DIV2K_VALIDATION_DIR: str = "div2k_validation"
DIV2K_VALIDATION_TMP_ZIP: str = "div2k_validation_tmp.zip"
DIV2K_VALIDATION_TMP_DIR: str = "DIV2K_valid_HR"


def resolve_raw_dataset_download(
    dataset_path: str,
    dataset_dirs: str | list[str],
    dataset_name: str,
    always_download: bool,
    download_cb: Callable[[str], None],
) -> None:
    does_exist = False
    if isinstance(dataset_dirs, str):
        does_exist = exists(f"{dataset_path}/{dataset_dirs}")
    else:
        does_exist = any(
            exists(f"{dataset_path}/{dataset_dir}") for dataset_dir in dataset_dirs
        )
    if not does_exist:
        if always_download:
            return download_cb(dataset_path)
        print(
            f"{dataset_name} dataset has not been detected - would you like to download"
            " it automatically or rather do it manually?"
        )
        command = input(
            "[Press D to download automatically otherwise press any other key]:"
        )
        if command == "D":
            download_cb(dataset_path)
    else:
        print(f"{dataset_name} dataset has already been downloaded")


def download_div2k_train_dataset(dataset_path: str) -> None:
    print("Downloading DIV2K Train dataset")
    clic_dir = f"{dataset_path}/{DIV2K_TRAIN_DIR}"
    makedirs(clic_dir, exist_ok=True)
    wget_download(DIV2K_TRAIN_URL, out=f"{clic_dir}/{DIV2K_TRAIN_TMP_ZIP}")
    print("DIV2K Train dataset has been successfully downloaded")


def download_div2k_validation_dataset(dataset_path: str) -> None:
    print("Downloading DIV2K Validation dataset")
    clic_dir = f"{dataset_path}/{DIV2K_VALIDATION_DIR}"
    makedirs(clic_dir, exist_ok=True)
    wget_download(DIV2K_VALIDATION_URL, out=f"{clic_dir}/{DIV2K_VALIDATION_TMP_ZIP}")
    print("DIV2K Validation dataset has been successfully downloaded")


def execute_command(*command_args: str) -> None:
    Popen(["bash", "-c", " ".join(command_args)]).wait()


def move_directory_files(source: str, target: str) -> None:
    for path in list(listdir(source)):
        print(f"Moving {path} from {source} to {target}")
        move_file(f"{source}/{path}", f"{target}/")


def unzip_div2k_dataset(
    dataset_path: str, tmp_zip_file: str, tmp_dir: str, name: str
) -> None:
    zip_file = f"{dataset_path}/{tmp_zip_file}"
    if not exists(zip_file):
        print(f'"{zip_file}" does not exist - aborting decompression')
        return None
    print(f"Preparing to uzip {name} dataset")
    execute_command("unzip", zip_file, "-d", dataset_path)
    print(f"Unzipped {name} dataset")
    remove(zip_file)
    print(f"Removed temp {name} zip file")
    move_directory_files(f"{dataset_path}/{tmp_dir}", dataset_path)
    print(f"Moved {name} images to {dataset_path}")
    rmtree(f"{dataset_path}/{tmp_dir}")
    print(f"Finished preparing {name} dataset")
    return None


@command()
@argument("raw_datasets_path", type=Path())
@option("-y", "--always_download", is_flag=True, default=False)
def main(raw_datasets_path: str, always_download: bool) -> None:
    resolve_raw_dataset_download(
        raw_datasets_path,
        [DIV2K_TRAIN_DIR, f"{DIV2K_TRAIN_DIR}/{DIV2K_TRAIN_TMP_ZIP}"],
        "DIV2K train",
        always_download,
        download_div2k_train_dataset,
    )
    unzip_div2k_dataset(
        f"{raw_datasets_path}/{DIV2K_TRAIN_DIR}",
        DIV2K_TRAIN_TMP_ZIP,
        DIV2K_TRAIN_TMP_DIR,
        "DIV2K Train",
    )
    resolve_raw_dataset_download(
        raw_datasets_path,
        [DIV2K_VALIDATION_DIR, f"{DIV2K_VALIDATION_DIR}/{DIV2K_VALIDATION_TMP_ZIP}"],
        "DIV2K validation",
        always_download,
        download_div2k_validation_dataset,
    )
    unzip_div2k_dataset(
        f"{raw_datasets_path}/{DIV2K_VALIDATION_DIR}",
        DIV2K_VALIDATION_TMP_ZIP,
        DIV2K_VALIDATION_TMP_DIR,
        "DIV2K Validation",
    )


if __name__ == "__main__":
    main()
