"""Configuration constants for the Stream Talker Server."""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# Paths
# =============================================================================

# Base data directory (persistent storage)
DATA_DIR = Path("/data")

# Settings file path
SETTINGS_FILE = DATA_DIR / "settings.json"

# Voice cache directory
VOICES_DIR = DATA_DIR / "voices"

# General cache directory (for future use)
CACHE_DIR = DATA_DIR / "cache"

# Local directories (non-persistent)
OUTPUT_DIR = Path("outputs")

# =============================================================================
# Model Existence Checks
# =============================================================================


def get_huggingface_cache_dir() -> Path:
    """
    Get the HuggingFace cache directory.

    Checks in order:
    1. HF_HOME environment variable
    2. HUGGINGFACE_HUB_CACHE environment variable
    3. Default: ~/.cache/huggingface/hub
    """
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"

    hf_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hf_cache:
        return Path(hf_cache)

    return Path.home() / ".cache" / "huggingface" / "hub"


def check_huggingface_model_exists(model_id: str) -> bool:
    """
    Check if a HuggingFace model is cached locally.

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    Returns:
        True if model folder exists in local cache, False otherwise
    """
    folder_name = f"models--{model_id.replace('/', '--')}"
    model_path = get_huggingface_cache_dir() / folder_name
    return model_path.is_dir()


def get_available_tts_models() -> Dict[str, str]:
    """
    Get dictionary of available TTS models that are actually cached locally.

    Returns:
        Dictionary mapping model ID to HuggingFace model path
    """
    available = {}
    for model_id, hf_id in ALL_TTS_MODELS.items():
        if check_huggingface_model_exists(hf_id):
            available[model_id] = hf_id
    return available


# =============================================================================
# Language Configuration
# =============================================================================

# Supported languages for TTS synthesis (from Qwen3-TTS documentation)
SUPPORTED_LANGUAGES = [
    "Auto",       # Automatic language detection (default)
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

# Default language setting
DEFAULT_LANGUAGE = "Auto"

# =============================================================================
# Model Configuration
# =============================================================================

# All defined TTS models with their HuggingFace identifiers
ALL_TTS_MODELS: Dict[str, str] = {
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

# Available TTS models (filtered to only locally cached models)
# This is populated at startup
AVAILABLE_MODELS: Dict[str, str] = {}

# Default TTS model to use when not specified
DEFAULT_MODEL = "1.7B"

def initialize_available_models() -> None:
    """
    Initialize the AVAILABLE_MODELS dict based on what's actually cached.
    Should be called at server startup.

    Note: We use .clear() and .update() instead of reassignment to preserve
    references from other modules that imported AVAILABLE_MODELS.
    """
    global DEFAULT_MODEL

    # Check TTS models - update in-place to preserve import references
    AVAILABLE_MODELS.clear()
    AVAILABLE_MODELS.update(get_available_tts_models())

    if not AVAILABLE_MODELS:
        logger.error("No TTS models are available! Server functionality will be limited.")

    # Update default model if it's not available
    if DEFAULT_MODEL not in AVAILABLE_MODELS and AVAILABLE_MODELS:
        DEFAULT_MODEL = next(iter(AVAILABLE_MODELS.keys()))
        logger.warning(f"Default model not available, using {DEFAULT_MODEL} instead")

    logger.info(f"Available models initialized: TTS={list(AVAILABLE_MODELS.keys())}")

# =============================================================================
# Auto-unload Configuration
# =============================================================================

# Default auto-unload timeout in minutes (0 = disabled)
DEFAULT_AUTO_UNLOAD_MINUTES = 30

# =============================================================================
# Performance Optimization Configuration
# =============================================================================

# Supported quantization options for model loading
QUANTIZATION_OPTIONS = ["none", "int8", "float8"]

# Quantization cache configuration
# Cache location: /data/cache/quantized/{model_id}_{quantization}_v{version}.pt
ENABLE_QUANTIZATION_CACHE = True
QUANTIZATION_CACHE_VERSION = "v2"
QUANTIZATION_CACHE_DIR = CACHE_DIR / "quantized"

# =============================================================================
# GPU Compatibility Configuration
# =============================================================================

# SM (Streaming Multiprocessor) version thresholds
# Used for automatic feature detection and fallback behavior
GPU_SM_TURING = 75       # RTX 20xx, GTX 16xx - NO native bf16, sage, flex, float8
GPU_SM_AMPERE = 80       # A100, RTX 30xx - bf16, sage, flex supported
GPU_SM_ADA = 89          # RTX 40xx - all features including float8
GPU_SM_HOPPER = 90       # H100 - all features

# Minimum SM requirements for features
GPU_MIN_SM_BF16 = 80           # Native bfloat16 hardware support
GPU_MIN_SM_FLOAT8 = 89         # Float8 quantization (torchao requirement)
GPU_MIN_SM_SAGE_ATTENTION = 80 # SageAttention CUDA kernels
GPU_MIN_SM_FLEX_ATTENTION = 80 # FlexAttention (Triton requirement)

# =============================================================================
# Audio Configuration
# =============================================================================

# Maximum reference audio duration in milliseconds (None = no limit)
MAX_REF_AUDIO_DURATION_MS = None

# Target sample rate for reference audio
TARGET_SAMPLE_RATE = 24000

# Maximum upload file size in bytes (200MB)
MAX_UPLOAD_SIZE = 200 * 1024 * 1024

# Allowed audio file extensions
ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3", "flac", "ogg"}

# =============================================================================
# Server Configuration
# =============================================================================

# Auto-unload check interval in seconds
AUTO_UNLOAD_CHECK_INTERVAL = 60

# Server host and port
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 7860

# =============================================================================
# Inference Configuration
# =============================================================================

# Default inference timeout in seconds (can be changed via API)
# Note: GPU operation continues in background until completion
DEFAULT_INFERENCE_TIMEOUT = 120  # 2 minutes

# Maximum batch size for synthesize_speech endpoint
# Prevents OOM crashes from unbounded batch requests
MAX_BATCH_SIZE = 50

# Default maximum VRAM usage percentage (used on first launch)
# Inference will stop early if total VRAM usage exceeds this limit
DEFAULT_MAX_VRAM_PERCENT = 70  # 70% of total GPU VRAM

# =============================================================================
# Generation Limits Configuration (Fix for Qwen-TTS hangs)
# =============================================================================

# Maximum new tokens limit - hard cap to prevent infinite generation
# The qwen-tts library may ignore max_new_tokens passed to generate_voice_clone(),
# so we override the model's internal generation_config at load time.
MAX_NEW_TOKENS_LIMIT = 2048

# EOS token IDs for Qwen3-TTS (from GitHub Issue #118)
# ~0.5% of inferences fail to generate EOS token, causing infinite generation.
# Multiple EOS token IDs ensure generation stops properly.
# Ref: https://github.com/QwenLM/Qwen3-TTS/issues/118
# NOTE: Handled internally by rekuenkdr fork (commit 3640831)
# QWEN_TTS_EOS_TOKEN_IDS = [2150, 2157, 151670, 151673, 151645, 151643]

# Default repetition penalty to prevent token loops
# Values > 1.0 discourage repeating the same tokens, preventing degenerate outputs.
DEFAULT_REPETITION_PENALTY = 1.05

# =============================================================================
# Language Detection Configuration
# =============================================================================

# Languages that use full-width punctuation (affects text normalization)
CJK_LANGUAGES: set[str] = {"Chinese", "Japanese", "Korean"}


# =============================================================================
# Memory Management Configuration
# =============================================================================

# GPU memory threshold for cleanup warnings (MB)
GPU_MEMORY_CLEANUP_THRESHOLD_MB: int = 100


# =============================================================================
# API Validation Constraints
# =============================================================================

# Speed adjustment limits
MIN_SPEED: float = 0.5
MAX_SPEED: float = 2.0
DEFAULT_SPEED: float = 1.0

# Temperature limits
MIN_TEMPERATURE: float = 0.1
MAX_TEMPERATURE: float = 2.0
DEFAULT_TEMPERATURE: float = 0.7

# Token generation limits (per-request override of model config)
MIN_MAX_NEW_TOKENS: int = 256
MAX_MAX_NEW_TOKENS: int = 8192
DEFAULT_MAX_NEW_TOKENS: int = 2048

# Repetition penalty limits
MIN_REPETITION_PENALTY: float = 1.0
MAX_REPETITION_PENALTY: float = 2.0
