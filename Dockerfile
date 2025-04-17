FROM dmml/conda:py39

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        vim \
        cmake \
        build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy some scripts into docker container
COPY setup.py .
COPY pyproject.toml .
COPY requirements.txt .

# Install packages in `pyproject.toml` via `pip`
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -e .

# Install PyTorch (with CUDA support for 12.1) using `conda`
# CUDA 12.1 is the latest version supported by PyTorch, 
# should also work for CUDA 12.2
RUN conda install -y -c nvidia cuda-nvcc
RUN conda install -y pytorch=2.1.* torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

WORKDIR /workspace
RUN chmod -R a+w /workspace && git config --global --add safe.directory /workspace

