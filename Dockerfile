# CUDA 11.8 base image
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Dependencies habitat-sim needs
RUN apt-get update && apt-get install -y \
    wget \
    git \
    bzip2 \
    ca-certificates \
    libx11-6 \
    libxext6 \
    libglib2.0-0 \
    libegl1 \
    libglvnd0 \
    # libgl1-mesa-glx \
    libgl1 \
    libglx-mesa0 \
    libgles2-mesa-dev \
    && rm -rf /var/lib/apt/lists/*


# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

# Create the conda environment with override-channels
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda create --override-channels -c conda-forge -n DCON python=3.9 cmake=3.14.0 -y && \
    echo "conda activate DCON" >> ~/.bashrc

SHELL ["bash", "-c"]
ENV CONDA_DEFAULT_ENV=DCON
ENV PATH=/opt/conda/envs/DCON/bin:$PATH

# Activate environment and install conda packages
RUN source activate DCON && conda install -c conda-forge -y \
    ipykernel \
    ipython \
    jupyter_client \
    jupyter_core \
    matplotlib-inline \
    && conda clean -afy

# Torch 11.8
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# tiny-cuda-nn
ENV TCNN_CUDA_ARCHITECTURES="89"
RUN pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Habitat-Sim (headless)
RUN git clone --branch stable https://github.com/facebookresearch/habitat-sim.git /workspace/habitat-sim && \
    cd /workspace/habitat-sim && \
    pip install -r requirements.txt && \
    python setup.py install --headless --bullet

# Habitat-Lab
RUN git clone --branch stable https://github.com/facebookresearch/habitat-lab.git /workspace/habitat-lab && \
    pip install -e /workspace/habitat-lab/habitat-lab && \
    pip install -e /workspace/habitat-lab/habitat-baselines

# Gsplat
RUN pip install gsplat || \
    pip install git+https://github.com/nerfstudio-project/gsplat.git


# X11 forwarding fixes
RUN apt-get update && apt-get install -y \
    libsm6 \
    libice6 \
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libx11-xcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-dri3-0 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# EGL / GL environment
ENV __GLX_VENDOR_LIBRARY_NAME=nvidia
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV EGL_PLATFORM=surfaceless

CMD ["/bin/bash"]
