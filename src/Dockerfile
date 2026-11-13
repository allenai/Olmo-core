ARG UBUNTU_VERSION=22.04
ARG TARGET_PLATFORM=x86_64
ARG CUDA_VERSION=12.8.1
ARG CUDA_VERSION_PATH=cu128
ARG PYTHON_VERSION=3.12
ARG BASE_IMAGE=ubuntu:${UBUNTU_VERSION}
ARG DEVEL_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

#########################################################################
# Build image
#########################################################################

FROM ${DEVEL_BASE_IMAGE} AS build

WORKDIR /app/build

# Install system dependencies.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        wget \
        libxml2-dev \
        libjpeg-dev \
        libpng-dev \
        gcc \
        git && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda, Python, and Python build dependencies.
ARG TARGET_PLATFORM
ARG PYTHON_VERSION
ENV PATH=/opt/conda/bin:$PATH
RUN curl -fsSL -v -o ~/miniconda.sh -O  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${TARGET_PLATFORM}.sh"
# NOTE: Manually invoke bash on miniconda script per https://github.com/conda/conda/issues/10431
RUN chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake conda-build pyyaml numpy ipython && \
    /opt/conda/bin/python -m pip install --upgrade --no-cache-dir pip wheel packaging "setuptools<70.0.0" ninja && \
    /opt/conda/bin/conda clean -ya

# Install PyTorch core ecosystem.
ARG CUDA_VERSION_PATH
ARG TORCH_VERSION=2.10.0
ARG INSTALL_CHANNEL=whl
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_VERSION_PATH}/ \
    torch==${TORCH_VERSION} torchvision

# Install ptxas for Triton and expose it at a stable path that is copied into the release image.
ARG CUDA_NVCC_VERSION=12.8.93
RUN pip install --no-cache-dir nvidia-cuda-nvcc-cu12==${CUDA_NVCC_VERSION} && \
    ln -s /opt/conda/lib/python${PYTHON_VERSION}/site-packages/nvidia/cuda_nvcc/bin/ptxas \
        /opt/conda/bin/ptxas && \
    /opt/conda/bin/ptxas --version

# Optionally install NVSHMEM. The symm_mem_vdev2d rowwise-EP extension links NVSHMEM's host + device
# libs and headers; the RMA image targets install it so `python -m
# olmo_core.kernels.build_symm_mem_vdev2d_ext` can compile the extension at runtime (the package
# source is cloned at launch, so the ext can't be prebuilt into the image). '_find_nvshmem_paths'
# discovers it under conda site-packages. '--no-deps' keeps pip off torch's pinned CUDA stack.
# Empty by default. NOTE: needs a CUDA 'devel' release base (see BASE_IMAGE) so the runtime has nvcc.
ARG NVSHMEM_PIP_SPEC=""
RUN if [ -n "${NVSHMEM_PIP_SPEC}" ]; then \
        pip install --no-cache-dir --no-deps "${NVSHMEM_PIP_SPEC}"; fi

# Compute capabilities the from-source CUDA extensions (grouped-gemm, transformer-engine) are built
# for. Include "10.3" (B300 / Blackwell Ultra, sm_103) only when building with CUDA >= 12.9 (see the
# 'beaker-image-b300' target). Passed inline per-tool rather than as a global ENV so it doesn't reach
# the flash-attn 3 (hopper) build.
ARG TORCH_CUDA_ARCH_LIST="9.0 10.0"

# Install grouped-gemm.
# NOTE: right now we need to build with CUTLASS so we can pass batch sizes on GPU.
# See https://github.com/tgale96/grouped_gemm/pull/21
ARG GROUPED_GEMM_SHA="f1429a3c44c98f7912aa4b00125144cdf4e7fdb2"
RUN TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" GROUPED_GEMM_CUTLASS="1" pip install --no-build-isolation --no-cache-dir "grouped_gemm @ git+https://git@github.com/tgale96/grouped_gemm.git@${GROUPED_GEMM_SHA}"

# Install flash-attn 2
# NOTE: flash-attn 2 picks build archs from FLASH_ATTN_CUDA_ARCHS (it does NOT read
# TORCH_CUDA_ARCH_LIST). Add "103" for B300 (sm_103); requires CUDA >= 12.9.
ARG FLASH_ATTN_VERSION=2.8.2
ARG FLASH_ATTN_CUDA_ARCHS="90;100"
RUN FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS}" pip install --no-build-isolation --no-cache-dir flash-attn==${FLASH_ATTN_VERSION}

# Install flash-attn 3.
# Skipped for B300 (B300=1): FA3 only builds sm_90a/sm_100a kernels (no sm_103), so it can't run on
# B300, and its bundled CUTLASS doesn't compile against CUDA 13. The default image still builds it.
ARG B300=
ARG FLASH_ATTN_3_SHA="92ca9da8d66f7b34ff50dc080ec0fef9661260d6"
ARG FA3_MAX_JOBS=16
RUN if [ -z "${B300}" ]; then \
        git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/Dao-AILab/flash-attention.git && \
        cd flash-attention && \
        git fetch --depth 1 origin ${FLASH_ATTN_3_SHA} && \
        git checkout ${FLASH_ATTN_3_SHA} && \
        git submodule update --init --depth 1 && \
        cd hopper && \
        FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_SM80=TRUE MAX_JOBS=${FA3_MAX_JOBS} python setup.py install && \
        cd /app/build && \
        rm -rf flash-attention; \
    fi

# flash-attn 3 will be installed to '/opt/conda/lib/python3.12/site-packages/flash_attn_3-*.egg/'
# and will only be importable as 'flash_attn_interface'. However, TE expects to import this
# as 'flash_attn_3.flash_attn_interface'. So we modify the package directory structure to adhere
# to that. Skipped for B300, which doesn't build FA3 (see above).
RUN if [ -z "${B300}" ]; then \
        cd /opt/conda/lib/python${PYTHON_VERSION}/site-packages/ && \
        mkdir -p flash_attn_3 && \
        mv flash_attn_3-*.egg/flash_attn_3/* flash_attn_3/ && \
        mv flash_attn_3-*.egg/flash_attn_interface.py flash_attn_3/ && \
        touch flash_attn_3/__init__.py && \
        rm -rf flash_attn_3-*; \
    fi

# Install flash-attn 4 (the `flash_attn.cute` backend behind AttentionBackendName.flash_4).
# Optional: skipped unless FLASH_ATTN_4_VERSION is set. Distributed as a prebuilt wheel, so it
# installs for B300 too — use FLASH_ATTN_4_EXTRAS='[cu13]' for the CUDA-13 build. Constrained to
# the installed torch so pip can't pull a different torch.
#
# FLASH_ATTN_4_CUTLASS_DSL_VERSION pins nvidia-cutlass-dsl. Needed because the FA4 betas pin it
# loosely (e.g. 4.0.0b16 requires 'nvidia-cutlass-dsl>=4.5.2'), so pip floats to the newest release
# — and cutlass-dsl 4.6.x removed `cutlass.cute.core.ThrMma`, which the 4.5-era wheel calls
# (AttributeError at import). Pinning to a 4.5.x here keeps the wheel and its DSL API in sync. It's
# installed in the same command so the resolver satisfies both constraints at once.
ARG FLASH_ATTN_4_VERSION=
ARG FLASH_ATTN_4_EXTRAS=""
ARG FLASH_ATTN_4_CUTLASS_DSL_VERSION=
RUN if [ -n "${FLASH_ATTN_4_VERSION}" ]; then \
        python -c "import torch; print('torch=='+torch.__version__)" > /tmp/torch-constraint.txt && \
        cutlass_spec=""; \
        if [ -n "${FLASH_ATTN_4_CUTLASS_DSL_VERSION}" ]; then \
            cutlass_spec="nvidia-cutlass-dsl${FLASH_ATTN_4_EXTRAS}==${FLASH_ATTN_4_CUTLASS_DSL_VERSION}"; \
        fi && \
        python -m pip install --no-cache-dir -c /tmp/torch-constraint.txt \
            "flash-attn-4${FLASH_ATTN_4_EXTRAS}==${FLASH_ATTN_4_VERSION}" ${cutlass_spec} && \
        rm -f /tmp/torch-constraint.txt; \
    fi

# Install transformer-engine.
# NVTE_WITH_NCCL_EP=0 disables TE's own NCCL expert-parallel extension (csrc/extensions/ep.cpp).
# We don't use it (olmo-core has its own EP: symm_mem_vdev2d / nccl_rma_p2p), and on a devel/RMA
# base it otherwise compiles against the system NCCL headers — TE >= 2.18 needs a newer NCCL
# (ncclWindow_vidmem) than the image ships, breaking the build. Only TE's MXFP8 path is needed.
ARG TE_VERSION=2.18.0
RUN NVTE_WITH_NCCL_EP=0 TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" pip install --no-build-isolation --no-cache-dir transformer-engine[pytorch]==${TE_VERSION}

# Install ring-flash-attn.
ARG RING_FLASH_ATTN_VERSION=0.1.8
RUN pip install --no-build-isolation --no-cache-dir ring-flash-attn==${RING_FLASH_ATTN_VERSION}

# Install liger-kernel.
ARG LIGER_KERNEL_VERSION=0.6.4
RUN pip install --no-build-isolation --no-cache-dir liger-kernel==${LIGER_KERNEL_VERSION}

# Optionally install Quack kernels.
ARG QUACK_VERSION=""
RUN if [ -n "${QUACK_VERSION}" ]; then \
    pip install --no-build-isolation --no-cache-dir quack-kernels==${QUACK_VERSION}; fi

# Install direct dependencies, but not source code.
COPY pyproject.toml .
COPY src/olmo_core/__init__.py src/olmo_core/__init__.py
COPY src/olmo_core/version.py src/olmo_core/version.py
ARG DION_SHA="e64832041d8e01989abf609c9550f6307efbff2a"
RUN pip install --no-cache-dir "dion @ git+https://github.com/microsoft/dion.git@${DION_SHA}" && \
    pip install --no-cache-dir '.[all]' && \
    pip uninstall -y ai2-olmo-core && \
    rm -rf *

# Pin NCCL to an RMA-capable build (>= 2.29.7) exposing the one-sided window signal API
# (ncclPutSignal / ncclWaitSignal) that the nccl_rma_p2p transport compiles against.
# MUST be the last install: torch's wheels and NVSHMEM both depend on nvidia-nccl-cu*, so any
# earlier pin gets re-resolved back down (dropping the RMA symbols) by later `pip install` steps.
# '_find_nccl_paths' discovers it under conda site-packages. Empty by default, so standard images
# keep torch's bundled NCCL.
ARG NCCL_PIP_SPEC=""
RUN if [ -n "${NCCL_PIP_SPEC}" ]; then \
        pip install --no-cache-dir --no-deps --force-reinstall "${NCCL_PIP_SPEC}"; fi

#########################################################################
# Release image
#########################################################################

FROM ${BASE_IMAGE} AS release

# Install system dependencies.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        wget \
        unzip \
        libxml2-dev \
        libjpeg-dev \
        libpng-dev \
        gnupg \
        jq \
        gcc \
        git && \
    rm -rf /var/lib/apt/lists/* \
    # AWS CLI \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm awscliv2.zip \
    # Google Cloud SDK \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
        | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update \
    && apt-get install -y google-cloud-cli \
    && rm -rf /var/lib/apt/lists/* \
    # GitHub CLI \
    && curl -sS https://webi.sh/gh | sh \
    # uv \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Install MLNX OFED user-space drivers
# See https://docs.nvidia.com/networking/pages/releaseview.action?pageId=15049785#Howto:DeployRDMAacceleratedDockercontaineroverInfiniBandfabric.-Dockerfile
ARG UBUNTU_VERSION
ARG TARGET_PLATFORM
ENV MOFED_VER="24.01-0.3.3.1"
RUN wget --quiet https://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VER}/MLNX_OFED_LINUX-${MOFED_VER}-ubuntu${UBUNTU_VERSION}-${TARGET_PLATFORM}.tgz && \
    tar -xvf MLNX_OFED_LINUX-${MOFED_VER}-ubuntu${UBUNTU_VERSION}-${TARGET_PLATFORM}.tgz && \
    MLNX_OFED_LINUX-${MOFED_VER}-ubuntu${UBUNTU_VERSION}-${TARGET_PLATFORM}/mlnxofedinstall --basic --user-space-only --without-fw-update -q && \
    rm -rf MLNX_OFED_LINUX-${MOFED_VER}-ubuntu${UBUNTU_VERSION}-${TARGET_PLATFORM} && \
    rm MLNX_OFED_LINUX-${MOFED_VER}-ubuntu${UBUNTU_VERSION}-${TARGET_PLATFORM}.tgz

# Copy conda environment.
COPY --from=build /opt/conda /opt/conda

# B300 (sm_103) fixes, baked in when B300=1 (empty for the default image → no-op).
#  - transformer-engine locates libnvrtc by grepping `ldconfig -p`, but on CUDA 13 nvrtc ships as a
#    pip wheel under site-packages and isn't in the linker cache, so `import transformer_engine`
#    fails. Register the wheel's lib dir so it's found.
#  - Triton's bundled ptxas doesn't know sm_103a. Symlink a CUDA-13 ptxas (torch's bundled one if
#    present, else the nvidia-cuda-nvcc wheel) to a fixed path and point Triton at it via
#    TRITON_PTXAS_PATH below. The symlink keeps this working for both torch 2.10 and 2.11 without
#    hard-coding a version-specific path (torch 2.11 ships torch/bin/ptxas; 2.10 may not).
ARG B300=
RUN if [ -n "${B300}" ]; then \
        NVRTC_DIR=$(find /opt/conda -name "libnvrtc.so*" 2>/dev/null | head -n1 | xargs -r dirname || true); \
        if [ -n "$NVRTC_DIR" ]; then echo "$NVRTC_DIR" > /etc/ld.so.conf.d/zz-nvrtc.conf && ldconfig; fi; \
        PTXAS=$(find /opt/conda -path "*/torch/bin/ptxas" 2>/dev/null | head -n1 || true); \
        if [ -z "$PTXAS" ]; then PTXAS=$(find /opt/conda -path "*cuda_nvcc*/bin/ptxas" 2>/dev/null | head -n1 || true); fi; \
        if [ -z "$PTXAS" ]; then pip install --no-cache-dir nvidia-cuda-nvcc-cu13 && PTXAS=$(find /opt/conda -path "*cuda_nvcc*/bin/ptxas" 2>/dev/null | head -n1 || true); fi; \
        if [ -n "$PTXAS" ]; then ln -sf "$PTXAS" /usr/local/bin/triton-ptxas; fi; \
    fi

# Use the image's assembler rather than Triton's bundled version. Triton 3.7 selects a separate
# assembler for Blackwell, so both overrides must be set. CUDA-13 targets use the symlink above.
ARG TRITON_PTXAS_PATH=/opt/conda/bin/ptxas
ENV TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH}
ENV TRITON_PTXAS_BLACKWELL_PATH=${TRITON_PTXAS_PATH}

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/root/.local/bin:/root/.cargo/bin:/opt/conda/bin:$PATH

LABEL org.opencontainers.image.source=https://github.com/allenai/Olmo-core
WORKDIR /app/olmo-core
