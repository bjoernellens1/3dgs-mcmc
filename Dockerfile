# syntax=docker/dockerfile:1.7

ARG ROCM_VERSION=7.2.2
ARG PYTORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
ARG TORCHAUDIO_VERSION=2.10.0
ARG TRITON_VERSION=3.6.0
ARG BASE_ROCM=rocm/pytorch:rocm${ROCM_VERSION}_ubuntu24.04_py3.12_pytorch_release_${PYTORCH_VERSION}

# =============================================================================
# STAGE 1 — Builder: compile gsplat inside the full ROCm dev image
# =============================================================================
FROM ${BASE_ROCM} AS builder

ENV HSA_XNACK=1
ENV HSA_ENABLE_SDMA=0
ENV PYTORCH_ROCM_ARCH=gfx1151

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    ninja-build \
    libglm-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

COPY docker/patch_gsplat_setup.py /tmp/patch_gsplat_setup.py
COPY docker/patch_gsplat_warp32.py /tmp/patch_gsplat_warp32.py

RUN git clone --branch release/1.5.3b2 --depth 1 https://github.com/ROCm/gsplat.git /tmp/gsplat \
    && cd /tmp/gsplat \
    && git submodule update --init --recursive \
    && python3 /tmp/patch_gsplat_setup.py \
    && python3 /tmp/patch_gsplat_warp32.py \
    && python3 setup.py bdist_wheel

# =============================================================================
# STAGE 2 — Runtime: minimal ROCm runtime + PyTorch wheels + app code
# =============================================================================
FROM ubuntu:24.04 AS runtime

ARG ROCM_VERSION
ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG TRITON_VERSION

ENV DEBIAN_FRONTEND=noninteractive
ENV HSA_XNACK=1
ENV HSA_ENABLE_SDMA=0
ENV PYTORCH_ROCM_ARCH=gfx1151
ENV ROCM_PATH=/opt/rocm
ENV HIP_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=/opt/rocm/lib
ENV PATH=/opt/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# --- system dependencies and Python ----------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    software-properties-common \
    python3.12 \
    python3.12-venv \
    python3-pip \
    ffmpeg \
    libglib2.0-0 \
    libnuma1 \
    libelf1 \
    libzstd1 \
    zlib1g \
    liblzma5 \
    libdrm-amdgpu1 \
    && rm -rf /var/lib/apt/lists/*

# --- ROCm apt repository ----------------------------------------------------
RUN mkdir -p --mode=0755 /etc/apt/keyrings \
    && curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg \
    && printf 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/%s noble main\n' "${ROCM_VERSION}" > /etc/apt/sources.list.d/rocm.list \
    && printf '%s\n' 'Package: *' 'Pin: release o=repo.radeon.com' 'Pin-Priority: 600' > /etc/apt/preferences.d/rocm-pin-600 \
    && apt-get update

# --- minimal ROCm runtime libraries (no -dev, no rocm meta-package) ---------
# Every library listed is pulled in by torch/lib*.so at runtime (verified via ldd).
RUN apt-get install -y --no-install-recommends \
    rocm-core \
    rocminfo \
    hip-runtime-amd \
    hsa-rocr \
    comgr \
    rocprofiler-register \
    roctracer \
    rocm-smi-lib \
    amd-smi-lib \
    hipblas \
    hipblaslt \
    rocblas \
    rocfft \
    rocrand \
    hiprand \
    rocsolver \
    rocsparse \
    rccl \
    rocalution \
    hipfft \
    hipsolver \
    hipsparse \
    miopen-hip \
    hipsparselt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/*

# --- Python virtual environment ---------------------------------------------
RUN python3.12 -m venv /opt/venv --system-site-packages
ENV PATH=/opt/venv/bin:$PATH
ENV VIRTUAL_ENV=/opt/venv

RUN pip install --no-cache-dir uv

# --- PyTorch ROCm wheels from the official AMD index -----------------------
# Use pip here (not uv) because uv doesn't support --prefer-binary.
RUN pip install --no-cache-dir \
    --find-links "https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VERSION}/" \
    --prefer-binary \
    "torch==${PYTORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    "triton==${TRITON_VERSION}"

# --- Python application dependencies ----------------------------------------
RUN uv pip install --no-cache \
    numpy \
    plyfile \
    tqdm \
    opencv-python-headless \
    rich \
    jaxtyping \
    tensorboard \
    open3d \
    h5py

# --- gsplat wheel from builder ----------------------------------------------
RUN --mount=from=builder,source=/tmp/gsplat/dist,target=/gsplat-dist \
    pip install --no-cache-dir /gsplat-dist/*.whl

# --- source code ------------------------------------------------------------
WORKDIR /workspace/3dgs-mcmc
COPY arguments ./arguments
COPY gaussian_renderer ./gaussian_renderer
COPY scene ./scene
COPY utils ./utils
COPY lpipsPyTorch ./lpipsPyTorch
COPY configs ./configs
COPY train.py render.py convert.py metrics.py ./

CMD ["bash"]
