# =============================================================================
# Build Args
# =============================================================================
ARG DOCKER_FROM=pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime
# For fast rebuilds, set SERVER_BASE to a pre-built server-prepare image
ARG SERVER_BASE=server-prepare

# Model selection args (set to "true" to include, anything else to skip)
ARG INCLUDE_MODEL_06B=true
ARG INCLUDE_MODEL_17B=true
ARG INCLUDE_SAGE_ATTENTION=true

# =============================================================================
# Stage 1: Dependencies - Install additional Python packages
# =============================================================================
# Uses PyTorch base image which already has torch + torchaudio pre-installed
# We use the base image's /opt/conda environment directly (no venv needed)
FROM ${DOCKER_FROM} AS deps

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies (build + runtime)
# python3-dev: Required for SageAttention's Triton backend JIT compilation (Python.h)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    sox \
    libsox-fmt-all \
    libsndfile1-dev \
    libmagic1 \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create CUDA stubs symlink for Triton compilation
# Triton needs libcuda.so for gcc linking during JIT compilation
RUN ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so

# Upgrade pip
RUN pip install --no-cache-dir --break-system-packages --upgrade pip wheel setuptools

# Install torchaudio if not already present (some PyTorch images include it, some don't)
# Uses --no-deps to avoid re-downloading torch
RUN python -c "import torchaudio" 2>/dev/null || \
    pip install --no-cache-dir --break-system-packages torchaudio --no-deps

# Install core Python dependencies (heavy packages for model loading)
# Filter out torch/torchaudio since they should already be installed
COPY requirements-core.txt /tmp/
RUN grep -vE "^torch(>=|==|$)|^torchaudio" /tmp/requirements-core.txt > /tmp/requirements_filtered.txt && \
    pip install --no-cache-dir --break-system-packages -r /tmp/requirements_filtered.txt

# Install flash-attention (optional, will skip if not compatible)
RUN pip install --no-cache-dir --break-system-packages https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.6.3+cu130torch2.9-cp311-cp311-linux_x86_64.whl --no-build-isolation || true

# =============================================================================
# Stage 2: Models - Download ML models (cached separately from deps)
# =============================================================================
FROM deps AS models

# Re-declare ARGs after FROM (they don't persist across stages)
ARG INCLUDE_MODEL_06B
ARG INCLUDE_MODEL_17B

# Tokenizer is ALWAYS required (used by all TTS models)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-Tokenizer-12Hz')"

# Conditionally download TTS models based on build args
RUN if [ "$INCLUDE_MODEL_06B" = "true" ]; then \
    echo "Downloading 0.6B model..." && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-0.6B-Base')"; \
    else \
    echo "Skipping 0.6B model (INCLUDE_MODEL_06B=$INCLUDE_MODEL_06B)"; \
    fi

RUN if [ "$INCLUDE_MODEL_17B" = "true" ]; then \
    echo "Downloading 1.7B model..." && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base')"; \
    else \
    echo "Skipping 1.7B model (INCLUDE_MODEL_17B=$INCLUDE_MODEL_17B)"; \
    fi

# =============================================================================
# Stage 3: Server Prepare - Final base with all dependencies (no server code)
# =============================================================================
# Extends from models stage (which extends from deps)
# This ensures sequential building and reuses all previous layers
FROM models AS server-prepare

LABEL maintainer="Stream Talker Server"
LABEL description="Multi-model TTS server with voice cloning and dynamic model management"
LABEL version="1.0.0"

ENV SHELL=/bin/bash
ENV PYTHONUNBUFFERED=1

# =============================================================================
# Performance Optimization Environment Variables
# =============================================================================

# Optimize CUDA memory allocation for better GPU memory management
# expandable_segments: Reduces memory fragmentation
# garbage_collection_threshold:0.8: Trigger GC when 80% memory used
# max_split_size_mb:512: Prevent memory fragmentation from large allocations
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512

# Enable parallel tokenization for faster text processing
ENV TOKENIZERS_PARALLELISM=true

# Disable debug/profiling features for production performance
ENV TORCH_SHOW_CPP_STACKTRACES=0
ENV CUDA_LAUNCH_BLOCKING=0

# Enable TensorFloat-32 (TF32) for faster matmul on Ampere+ GPUs
ENV NVIDIA_TF32_OVERRIDE=1

# Reduce Python overhead in production (removes assert statements)
ENV PYTHONOPTIMIZE=1

# Hugging Face optimizations - enable faster model downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Prevent .pyc bytecode files (minor disk/startup benefit)
ENV PYTHONDONTWRITEBYTECODE=1

# Cache Numba JIT compilation (helps librosa audio processing)
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Cache Triton JIT compilation (required for SageAttention)
ENV TRITON_CACHE_DIR=/tmp/triton_cache

# Create necessary directories
# /app - Application code (not persistent)
# /data - Persistent storage (mounted as volume)
RUN mkdir -p /app/outputs /data/voices /data/cache /tmp/triton_cache

# Install optional/optimization packages (separate layer for faster rebuilds)
# Changes to requirements-optional.txt won't invalidate deps or models cache
COPY requirements-optional.txt /tmp/

RUN pip install --no-cache-dir --break-system-packages --no-build-isolation -r /tmp/requirements-optional.txt

# Install SageAttention from local wheel (custom build with correct architectures)
COPY lib/sageattention-*.whl /tmp/
RUN pip install --no-cache-dir --break-system-packages /tmp/sageattention-*.whl && rm -f /tmp/sageattention-*.whl

# =============================================================================
# Cleanup: Remove build-only packages to reduce image size
# =============================================================================
# NOTE: Keep build-essential (gcc, g++, etc.) for Triton JIT compilation at runtime
# NOTE: Keep python3-dev (Python.h) - required by SageAttention's Triton backend for JIT
# Only remove development headers that are truly build-only dependencies
# NOTE: Use `docker build --squash` to realize actual size savings.
RUN apt-get update \
    && apt-get purge -y --auto-remove \
    libjpeg-dev \
    libjpeg8-dev \
    libjpeg-turbo8-dev \
    libpng-dev \
    libsndfile1-dev \
    # Reinstall runtime library (without -dev headers)
    && apt-get install -y --no-install-recommends libsndfile1 \
    && apt-get autoremove -y \
    && apt-get clean \
    # Remove unnecessary system files
    && rm -rf \
    /var/lib/apt/lists/* \
    /var/cache/apt/archives/* \
    /var/log/* \
    /usr/share/doc/* \
    /usr/share/man/* \
    /usr/share/info/* \
    /usr/share/fonts/* \
    /tmp/* \
    /root/.cache/pip \
    # Remove non-English locales (keep en* and C)
    && find /usr/share/locale -mindepth 1 -maxdepth 1 ! -name 'en*' ! -name 'C' -exec rm -rf {} + 2>/dev/null || true

# Copy startup script
COPY start.sh /app/

# Fix line endings and make executable
RUN sed -i 's/\r$//' /app/start.sh \
    && chmod +x /app/start.sh

WORKDIR /app

# Expose the server port
EXPOSE 7860

# Define volume for persistent data
VOLUME ["/data"]

# =============================================================================
# Stage 4: Runtime - Final image with server code (default build target)
# =============================================================================
# Uses SERVER_BASE arg - defaults to server-prepare, but can reference pre-built image
ARG SERVER_BASE
FROM ${SERVER_BASE} AS runtime

# Copy server package (changes most frequently)
COPY server/ /app/server/

# Set the entrypoint
ENTRYPOINT ["/bin/bash", "/app/start.sh"]
