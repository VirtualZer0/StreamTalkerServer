"""GPU detection and compatibility utilities.

Centralizes SM version detection and feature availability checks
for consistent behavior across the codebase.

SM Version Reference:
- SM75: Turing (RTX 20xx, GTX 16xx) - NO native bf16, sage, flex, float8
- SM80: Ampere (A100)
- SM86: Ampere (RTX 30xx consumer)
- SM87: Ampere (Jetson)
- SM89: Ada (RTX 40xx) - adds float8 support
- SM90: Hopper (H100)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from server.config import (
    GPU_SM_TURING as SM_TURING,
    GPU_SM_AMPERE as SM_AMPERE,
    GPU_SM_ADA as SM_ADA,
    GPU_SM_HOPPER as SM_HOPPER,
    GPU_MIN_SM_BF16 as SM_MIN_BF16_NATIVE,
    GPU_MIN_SM_FLOAT8 as SM_MIN_FLOAT8,
    GPU_MIN_SM_SAGE_ATTENTION as SM_MIN_SAGE_ATTENTION,
    GPU_MIN_SM_FLEX_ATTENTION as SM_MIN_FLEX_ATTENTION,
)

logger = logging.getLogger(__name__)

# SM version thresholds for feature support (additional constants not in config)
SM_AMPERE_CONSUMER = 86  # RTX 30xx


@dataclass
class GPUCapabilities:
    """GPU capabilities and feature support."""
    sm_version: int
    architecture: str
    name: str
    bf16_supported: bool
    float8_supported: bool
    sage_attention_supported: bool
    flex_attention_supported: bool
    recommended_dtype: torch.dtype


def get_sm_version(device_index: int = 0) -> Optional[int]:
    """
    Get SM (Streaming Multiprocessor) version for GPU.

    Args:
        device_index: CUDA device index (default: 0)

    Returns:
        SM version as integer (e.g., 75, 80, 86, 89, 90) or None if no CUDA
    """
    if not torch.cuda.is_available():
        return None

    try:
        props = torch.cuda.get_device_properties(device_index)
        return props.major * 10 + props.minor
    except Exception:
        return None


def get_gpu_architecture(sm_version: Optional[int]) -> str:
    """
    Get GPU architecture name from SM version.

    Args:
        sm_version: SM version number

    Returns:
        Architecture name (e.g., "Turing", "Ampere", "Ada", "Hopper")
    """
    if sm_version is None:
        return "Unknown"

    if sm_version >= 100:
        return "Blackwell"
    elif sm_version >= 90:
        return "Hopper"
    elif sm_version >= 89:
        return "Ada"
    elif sm_version >= 80:
        return "Ampere"
    elif sm_version >= 75:
        return "Turing"
    elif sm_version >= 70:
        return "Volta"
    elif sm_version >= 60:
        return "Pascal"
    else:
        return "Legacy"


def is_bf16_native_supported(sm_version: Optional[int] = None) -> bool:
    """
    Check if GPU has native bfloat16 hardware support.

    Native bf16 requires Ampere (SM80) or newer. On Turing (SM75),
    bf16 operations are emulated with 4-5x performance penalty.

    Args:
        sm_version: SM version (auto-detected if None)

    Returns:
        True if native bf16 is supported
    """
    if sm_version is None:
        sm_version = get_sm_version()
    return sm_version is not None and sm_version >= SM_MIN_BF16_NATIVE


def is_float8_supported(sm_version: Optional[int] = None) -> bool:
    """
    Check if GPU supports float8 quantization.

    Float8 requires Ada (SM89) or newer. On older GPUs, float8
    quantization will fail or produce incorrect results.

    See: https://docs.pytorch.org/ao/stable/quantization_overview.html

    Args:
        sm_version: SM version (auto-detected if None)

    Returns:
        True if float8 is supported
    """
    if sm_version is None:
        sm_version = get_sm_version()
    return sm_version is not None and sm_version >= SM_MIN_FLOAT8


def is_sage_attention_supported(sm_version: Optional[int] = None) -> bool:
    """
    Check if GPU supports SageAttention.

    SageAttention requires Ampere (SM80) or newer. On Turing (SM75),
    the CUDA kernels are not available in the official package.

    See: https://github.com/thu-ml/SageAttention/issues/137
    For RTX 20xx support, see: https://github.com/woct0rdho/SageAttention

    Args:
        sm_version: SM version (auto-detected if None)

    Returns:
        True if SageAttention is supported
    """
    if sm_version is None:
        sm_version = get_sm_version()
    return sm_version is not None and sm_version >= SM_MIN_SAGE_ATTENTION


def is_flex_attention_supported(sm_version: Optional[int] = None) -> bool:
    """
    Check if GPU supports FlexAttention.

    FlexAttention requires Ampere (SM80) or newer due to Triton requirements.

    Args:
        sm_version: SM version (auto-detected if None)

    Returns:
        True if FlexAttention is supported
    """
    if sm_version is None:
        sm_version = get_sm_version()
    return sm_version is not None and sm_version >= SM_MIN_FLEX_ATTENTION


def get_recommended_dtype(sm_version: Optional[int] = None) -> torch.dtype:
    """
    Get recommended dtype for GPU (general purpose).

    - SM >= 80 (Ampere+): bfloat16 (native hardware support)
    - SM < 80 (Turing, etc.): float16 (avoids 4-5x emulation penalty)
    - No CUDA: float32

    Args:
        sm_version: SM version (auto-detected if None)

    Returns:
        Recommended torch dtype
    """
    if sm_version is None:
        sm_version = get_sm_version()

    if sm_version is None:
        return torch.float32
    elif sm_version >= SM_MIN_BF16_NATIVE:
        return torch.bfloat16
    else:
        return torch.float16


def get_model_dtype(model_id: str, sm_version: Optional[int] = None) -> torch.dtype:
    """
    Get recommended dtype for a specific TTS model.

    Model-specific considerations:
    - 0.6B: ALWAYS uses bfloat16 - float16 has numerical overflow issues
      causing "probability tensor contains inf/nan" crashes. On Turing (SM75),
      bfloat16 is emulated with ~4-5x slowdown but avoids crashes.
    - 1.7B: Uses hardware-optimal dtype (bfloat16 on SM80+, float16 on SM75)
      as it doesn't have the same numerical stability issues.

    Args:
        model_id: Model identifier ("0.6B" or "1.7B")
        sm_version: SM version (auto-detected if None)

    Returns:
        Appropriate torch dtype for the model
    """
    if sm_version is None:
        sm_version = get_sm_version()

    # No CUDA - use float32
    if sm_version is None:
        return torch.float32

    # 0.6B model ALWAYS uses bfloat16 for numerical stability
    # (float16 overflow causes inf/nan in probability tensor)
    if model_id == "0.6B":
        if sm_version < SM_MIN_BF16_NATIVE:
            logger.info(
                f"[{model_id}] Using bfloat16 (emulated on SM{sm_version}) "
                f"for numerical stability - float16 has overflow issues"
            )
        return torch.bfloat16

    # 1.7B and other models use hardware-optimal dtype
    return get_recommended_dtype(sm_version)


def get_gpu_capabilities(device_index: int = 0) -> Optional[GPUCapabilities]:
    """
    Get comprehensive GPU capabilities.

    Args:
        device_index: CUDA device index

    Returns:
        GPUCapabilities dataclass or None if no CUDA
    """
    if not torch.cuda.is_available():
        return None

    try:
        props = torch.cuda.get_device_properties(device_index)
        sm_version = props.major * 10 + props.minor

        return GPUCapabilities(
            sm_version=sm_version,
            architecture=get_gpu_architecture(sm_version),
            name=props.name,
            bf16_supported=is_bf16_native_supported(sm_version),
            float8_supported=is_float8_supported(sm_version),
            sage_attention_supported=is_sage_attention_supported(sm_version),
            flex_attention_supported=is_flex_attention_supported(sm_version),
            recommended_dtype=get_recommended_dtype(sm_version),
        )
    except Exception as e:
        logger.warning(f"Failed to get GPU capabilities: {e}")
        return None


def log_gpu_compatibility() -> None:
    """
    Log GPU compatibility information at startup.

    Call this during server initialization to inform users
    about their GPU's capabilities and any limitations.
    """
    caps = get_gpu_capabilities()

    if caps is None:
        logger.info("GPU: No CUDA available, running on CPU")
        return

    logger.info(
        f"GPU: {caps.name} ({caps.architecture}, SM{caps.sm_version})"
    )
    logger.info(
        f"  Recommended dtype: {caps.recommended_dtype}"
    )

    # Log feature availability
    features = []
    if caps.bf16_supported:
        features.append("bfloat16")
    else:
        logger.info("  Note: bfloat16 not natively supported, using float16")

    if caps.float8_supported:
        features.append("float8")
    if caps.sage_attention_supported:
        features.append("SageAttention")
    if caps.flex_attention_supported:
        features.append("FlexAttention")

    if features:
        logger.info(f"  Supported features: {', '.join(features)}")

    # Log limitations for Turing specifically
    if caps.sm_version < SM_MIN_BF16_NATIVE:
        limitations = []
        if not caps.float8_supported:
            limitations.append("float8 quantization")
        if not caps.sage_attention_supported:
            limitations.append("SageAttention")
        if not caps.flex_attention_supported:
            limitations.append("FlexAttention")

        if limitations:
            logger.warning(
                f"  RTX 20xx/Turing limitations: No {', '.join(limitations)}"
            )
