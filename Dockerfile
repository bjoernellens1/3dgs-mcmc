FROM rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0

# Strix Halo (gfx1151) / ROCm 7.2 recommended env vars
# gfx1151 is natively supported in ROCm 7.2.2 — no HSA_OVERRIDE_GFX_VERSION needed
ENV HSA_XNACK=1
ENV HSA_ENABLE_SDMA=0
ENV PYTORCH_ROCM_ARCH=gfx1151

# Install system build deps and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    libglm-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /workspace/3dgs-mcmc

# Create a virtual environment with system-site-packages so torch from the base image is visible
RUN python3 -m venv /opt/venv --system-site-packages
ENV PATH=/opt/venv/bin:$PATH
ENV VIRTUAL_ENV=/opt/venv

# Install Python dependencies (torch/torchvision are pre-installed in the base image)
COPY pyproject.toml ./
RUN uv pip install numpy plyfile tqdm opencv-python rich jaxtyping

# Build amd-gsplat from source with submodules (glm is a submodule)
# NOTE: ROCm branch has a bug where glm include path is missing from include_dirs.
# We patch setup.py to add it before building.
RUN git clone --branch release/1.5.3b2 --depth 1 https://github.com/ROCm/gsplat.git /tmp/gsplat \
    && cd /tmp/gsplat \
    && git submodule update --init --recursive

COPY docker/patch_gsplat_setup.py /tmp/patch_gsplat_setup.py
COPY docker/patch_gsplat_warp32.py /tmp/patch_gsplat_warp32.py
RUN cd /tmp/gsplat && python3 /tmp/patch_gsplat_setup.py && python3 /tmp/patch_gsplat_warp32.py && python setup.py build_ext --inplace && pip install --no-deps --no-build-isolation . && rm -rf /tmp/gsplat

# Copy source code
COPY . .

CMD ["bash"]
