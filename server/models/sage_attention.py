"""SageAttention integration for Qwen3-TTS.

SageAttention provides INT8-quantized attention that can be faster on some GPUs.
Performance by GPU architecture:
- Ada/Hopper (RTX 4090, H100): 2-3x speedup with CUDA backend
- Ampere (RTX 3090): ~5x SLOWER than SDPA (only benefits seq_len > 2000)

Requirements:
- sageattention>=2.2.0 (snw35 wheel recommended for multi-arch support)
- CUDA 12.0+

Notes:
- Forces CUDA backend explicitly (sageattn_qk_int8_pv_fp16_cuda)
- Critical: Must match SDPA's is_causal logic for correct generation
- If wheel lacks CUDA kernels for your arch, will get import/runtime error
- See: https://github.com/thu-ml/SageAttention
"""

import logging
from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Registration state
_sage_registered = False

# Cached backend function (architecture-specific)
_sage_backend: Optional[Callable] = None


def _get_sage_backend() -> Callable:
    """
    Get appropriate SageAttention backend for current GPU architecture.

    Forces CUDA backend for consistent behavior across architectures.
    The Triton backend (auto-selected by sageattn on some GPUs) produces incorrect audio.

    Requires Ampere (SM80) or newer:
    - Ampere (sm80, sm86, sm87): CUDA backend with INT8 QK, FP16 PV
    - Ada/Hopper/Blackwell: CUDA backend with INT8 QK, FP16 PV

    Turing (SM75) is NOT supported.
    For RTX 20xx support, see: https://github.com/woct0rdho/SageAttention

    Returns:
        SageAttention CUDA function appropriate for the GPU.

    Raises:
        RuntimeError: If GPU is SM75 or older
    """
    global _sage_backend

    if _sage_backend is not None:
        return _sage_backend

    props = torch.cuda.get_device_properties(0)
    sm_version = props.major * 10 + props.minor
    arch = f"sm{sm_version}"

    # Defensive check - should not reach here on SM75 due to is_sage_available()
    if sm_version < 80:
        raise RuntimeError(
            f"SageAttention requires Ampere (SM80) or newer GPU, got {arch}. "
            f"For RTX 20xx support, see: https://github.com/woct0rdho/SageAttention"
        )

    # Force CUDA backend explicitly - Triton backend produces incorrect audio
    # The manylinux wheel includes CUDA kernels for SM80/86/89/90/120
    try:
        from sageattention import sageattn_qk_int8_pv_fp16_cuda
        _sage_backend = sageattn_qk_int8_pv_fp16_cuda
        logger.info(f"SageAttention: using CUDA backend (INT8 QK, FP16 PV) for {arch}")
    except ImportError:
        # Fallback to auto if explicit CUDA import fails
        from sageattention import sageattn
        _sage_backend = sageattn
        logger.warning(
            f"SageAttention: CUDA backend not available, using auto for {arch}. "
            "If you see audio artifacts, try attention=sdpa instead."
        )

    return _sage_backend


def _fallback_to_sdpa(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float],
    dropout: float,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Fallback to SDPA attention when SageAttention cannot be used."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    return sdpa_attention_forward(
        module, query, key, value, attention_mask,
        scaling=scaling, dropout=dropout, **kwargs
    )



@torch._dynamo.disable
def _call_sage_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool,
    scaling: Optional[float],
) -> torch.Tensor:
    """
    Call SageAttention with dynamo disabled.

    The @torch._dynamo.disable decorator prevents torch.compile from tracing
    this function, which fixes "Cannot access data pointer of Tensor that
    doesn't have storage" errors with SageAttention CUDA kernels.

    Note: SageAttention is ~5x slower than SDPA on Ampere GPUs (RTX 30xx).
    It only provides speedup on very long sequences (>2000 tokens).
    """
    sageattn_fn = _get_sage_backend()

    return sageattn_fn(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        tensor_layout="HND",
        is_causal=is_causal,
        sm_scale=scaling,
        pv_accum_dtype="fp16",  # fp16 is faster, fp32 if you see quality issues
        smooth_k=False,  # Slight speedup, negligible accuracy impact
    )

def sage_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    SageAttention forward function for transformers attention dispatch.

    Fallback conditions (uses SDPA instead):
    - dropout > 0.0 (SageAttention doesn't support dropout)
    - dtype not float16/bfloat16 (SageAttention requires half precision)
    - head_dim > 128 (SageAttention max supported)
    - Non-causal attention with mask (SageAttention only handles causal masks)

    Args:
        module: Attention module (required for is_causal detection)
        query: Query tensor [batch, heads, seq_len, head_dim]
        key: Key tensor [batch, heads, seq_len, head_dim]
        value: Value tensor [batch, heads, seq_len, head_dim]
        attention_mask: Optional attention mask
        scaling: Optional scale factor (default: 1/sqrt(head_dim))
        dropout: Dropout probability (not supported, triggers fallback)
        **kwargs: Additional args (is_causal, etc.)

    Returns:
        Tuple of (output tensor, None) - output is [batch, seq_len, heads, head_dim]
    """
    # Determine is_causal - MUST match SDPA's logic for correct generation
    # SDPA uses: is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)
    # Without this, decoder models generate garbage because causal masking is not applied.
    is_causal = kwargs.get('is_causal', None)
    if is_causal is None:
        # Only use causal when: seq_len > 1, no mask, and module says it's causal
        is_causal = (
            query.shape[2] > 1
            and attention_mask is None
            and getattr(module, "is_causal", True)
        )

    # Fallback: dropout not supported
    if dropout > 0.0:
        logger.debug("SageAttention: dropout > 0, falling back to SDPA")
        return _fallback_to_sdpa(
            module, query, key, value, attention_mask,
            scaling, dropout, **kwargs
        )

    # Fallback: dtype must be half precision
    if query.dtype not in (torch.float16, torch.bfloat16):
        logger.debug(f"SageAttention: unsupported dtype {query.dtype}, falling back to SDPA")
        return _fallback_to_sdpa(
            module, query, key, value, attention_mask,
            scaling, dropout, **kwargs
        )

    # Fallback: head_dim must be <= 128
    head_dim = query.shape[-1]
    if head_dim > 128:
        logger.warning(f"SageAttention: head_dim={head_dim} > 128, falling back to SDPA")
        return _fallback_to_sdpa(
            module, query, key, value, attention_mask,
            scaling, dropout, **kwargs
        )

    # Fallback: non-causal with mask not supported
    if attention_mask is not None and not is_causal:
        logger.debug("SageAttention: non-causal attention with mask, falling back to SDPA")
        return _fallback_to_sdpa(
            module, query, key, value, attention_mask,
            scaling, dropout, **kwargs
        )

    # Call the dynamo-disabled wrapper
    out = _call_sage_attention(query, key, value, is_causal, scaling)

    # Transpose HND -> NHD for model's reshape operation
    # Input: [batch, heads, seq_len, head_dim]
    # Output: [batch, seq_len, heads, head_dim]
    out = out.transpose(1, 2).contiguous()

    return out, None


def register_sage_attention() -> bool:
    """
    Register SageAttention with transformers attention dispatcher.

    Requires Ampere (SM80) or newer GPU. RTX 20xx (Turing, SM75) is not supported.

    Uses the modern AttentionInterface API when available (transformers 4.57+),
    with fallback to legacy ALL_ATTENTION_FUNCTIONS for older versions.

    When using AttentionInterface, also registers AttentionMaskInterface to
    ensure proper mask handling.

    Returns:
        True if registration succeeded, False otherwise.
    """
    global _sage_registered

    if _sage_registered:
        return True

    # Check GPU compatibility FIRST (before trying to import/register)
    from server.utils.gpu import is_sage_attention_supported, get_sm_version, get_gpu_architecture

    sm_version = get_sm_version()
    if not is_sage_attention_supported(sm_version):
        arch = get_gpu_architecture(sm_version)
        logger.warning(
            f"SageAttention requires Ampere (SM80) or newer GPU, got {arch} (SM{sm_version}). "
            f"For RTX 20xx support, see: https://github.com/woct0rdho/SageAttention"
        )
        return False

    try:
        # Verify sageattention is available
        import sageattention  # noqa: F401

        # Try modern AttentionInterface API (transformers 4.57+)
        try:
            from transformers import AttentionInterface, AttentionMaskInterface
            from transformers.masking_utils import sdpa_mask

            AttentionInterface.register("sage_attn", sage_attention_forward)
            AttentionMaskInterface.register("sage_attn", sdpa_mask)

            _sage_registered = True
            logger.info("SageAttention registered with transformers (AttentionInterface + mask support)")
            return True

        except (ImportError, AttributeError):
            # Fallback to legacy API for older transformers versions
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            ALL_ATTENTION_FUNCTIONS["sage_attn"] = sage_attention_forward

            _sage_registered = True
            logger.info("SageAttention registered with transformers (legacy API, no mask support)")
            return True

    except ImportError as e:
        logger.warning(f"SageAttention not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to register SageAttention: {e}")
        return False


def apply_sage_to_submodels(model) -> int:
    """
    Apply sage_attn to all sub-models in a Qwen3TTS model.

    Qwen3-TTS has multiple sub-models (talker, code_predictor) that each
    have their own attention configuration. This function updates all of
    them to use SageAttention.

    Args:
        model: Qwen3TTSModel instance

    Returns:
        Number of sub-model configs updated.
    """
    updated = 0
    inner = getattr(model, 'model', None)
    if inner is None:
        return 0

    for name, module in inner.named_modules():
        if hasattr(module, 'config') and hasattr(module.config, '_attn_implementation'):
            if module.config._attn_implementation != 'sage_attn':
                module.config._attn_implementation = 'sage_attn'
                updated += 1

    if updated > 0:
        logger.debug(f"Applied sage_attn to {updated} sub-model config(s)")

    return updated


def is_sage_available() -> bool:
    """
    Check if SageAttention is available.

    Returns:
        True if:
        - sageattention package is installed
        - CUDA is available
        - GPU is SM80+ (Ampere or newer)

    RTX 20xx (Turing, SM75) is not supported. CUDA kernels were removed due to bf16 errors.
    See: https://github.com/thu-ml/SageAttention/issues/137
    """
    try:
        import sageattention  # noqa: F401

        if not torch.cuda.is_available():
            return False

        # SageAttention requires SM80+ (Ampere or newer)
        # RTX 20xx (Turing, SM75) kernels were removed due to bf16 dispatch errors
        from server.utils.gpu import is_sage_attention_supported
        return is_sage_attention_supported()

    except ImportError:
        return False


def reset_sage_state() -> None:
    """
    Reset all SageAttention global state.

    Call this when unloading models to ensure:
    - Backend reference is released
    - Registration state is reset for clean re-registration

    This ensures complete cleanup after model unload.
    """
    global _sage_registered, _sage_backend

    _sage_backend = None
    _sage_registered = False
    logger.debug("SageAttention: all state reset")
