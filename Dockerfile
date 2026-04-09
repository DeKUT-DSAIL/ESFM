# Copyright (c) 2026 ETH Zurich
# Authors: see CONTRIBUTORS.md
# Licensed under the MIT License. See the LICENSE file in the repository root.

FROM nvcr.io/nvidia/physicsnemo/physicsnemo:25.03
# FROM nvcr.io/nvidia/modulus/modulus:24.04 # not supported 
# modulus:24.12, 24.09, 24.07 gives nans after 2 steps of ESFM training
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ="Europe/Zurich"

# Disable noble-updates/security/backports: arm64 package versions lag amd64
# causing 404 on both new installs and upgrades of existing base image packages
RUN echo "deb http://ports.ubuntu.com/ubuntu-ports noble main restricted universe multiverse" > /etc/apt/sources.list && \
    rm -f /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget libeccodes0 libopenjp2-7 ncdu htop screen zip unzip vim \
    ffmpeg libjpeg-dev libpng-dev tree nvtop

# Install nodejs via NodeSource to avoid Ubuntu arm64 repo issues
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs

RUN pip install --upgrade pip setuptools
RUN pip install --no-deps wandb[media] DeepSpeed flash-attn nvidia-dali-cuda120 torchmetrics huggingface-hub memory_profiler  natsort scores==1.3.0 mmnpz cartopy pysteps torchdata==0.11.0 pyproj==3.7.1 pyshp==2.3.1 shapely==2.1.0 pynvml==11.5.0 nvitop==1.3.2 zarr==3.1.5
RUN pip install --quiet "lightning==2.5.1"

# Configure glymur
RUN mkdir -p /root/.config/glymur && printf "[library]\nopenjp2: /usr/lib/aarch64-linux-gnu/libopenjp2.so.7" > /root/.config/glymur/glymurrc
ENV XDG_CONFIG_HOME="/root/.config"

# Set up workspace
RUN mkdir -p /workspace
WORKDIR /workspace

# Set LD_LIBRARY_PATH for CUDA and HPC-X
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/hpcx/ucc/lib/:/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH

CMD [ "/bin/bash" ]
