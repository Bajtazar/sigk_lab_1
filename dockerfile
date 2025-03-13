FROM nvcr.io/nvidia/pytorch:24.12-py3

RUN apt update && apt upgrade -y
RUN apt-get update && apt-get install -y \
    nano \
    build-essential \
    wget \
    gdb
RUN pip install torch torchvision torchaudio --upgrade --force-reinstall
