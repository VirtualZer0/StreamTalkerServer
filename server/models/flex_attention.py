"""FlexAttention integration for Qwen3-TTS.

FlexAttention is PyTorch's new compiled attention API that provides:
- 2x+ speedup at long contexts compared to standard SDPA
- Works best with torch.compile
- Supports custom attention patterns (though we use standard causal here)

Requirements:
- PyTorch 2.5+ (FlexAttention introduced in 2.5)
- CUDA SM80+ (Ampere, Ada, Hopper) - RTX 30xx, RTX 40xx, A100, H100
- NOT supported on RTX 20xx (Turing, SM75)

References:
- https://pytorch.org/blog/flexattention/
- https://pytorch.org/blog/flexattention-for-inference/
"""

import logging
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Registration state
_flex_registered = False

# Cached compiled flex_attention function
_compiled_flex_attention: Optional[Callable] = None

# Block mask cache to avoid expensive create_block_mask on every forward pass
# Key: (batch_size, num_heads, seq_len, device_str)
# Value: block_mask
_block_mask_cache: OrderedDict[Tuple[int, int, int, str], Any] = OrderedDict()
_BLOCK_MASK_CACHE_MAX_SIZE = 32  # Limit cache size to prevent memory issues


def _get_compiled_flex_attention() -> Callable:
    """
    Get compiled FlexAttention function (cached for performance).

    Returns:
        Compiled flex_attention function.
    """
    global _compiled_flex_attention

    if _compiled_flex_attention is not None:
        return _compiled_flex_attention

    from torch.nn.attention.flex_attention import flex_attention

    # Compile for best performance
    # Use default mode (not reduce-overhead) to avoid CUDA graph issues with dynamic shapes
    _compiled_flex_attention = torch.compile(flex_attention, dynamic=True)

    logger.info("FlexAttention: compiled flex_attention function")
    return _compiled_flex_attention


def _get_or_create_block_mask(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    device: torch.device,
) -> Any:
    """
    Get cached block mask or create a new one.

    Block mask creation is expensive, so we cache masks by their dimensions.
    Uses LRU eviction to limit memory usage.

    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        device: Device for the mask

    Returns:
        BlockMask for FlexAttention
    """
    from torch.nn.attention.flex_attention import create_block_mask

    cache_key = (batch_size, num_heads, seq_len, str(device))

    # Check cache
    if cache_key in _block_mask_cache:
        # Move to end (LRU)
        _block_mask_cache.move_to_end(cache_key)
        return _block_mask_cache[cache_key]

    # Create causal mask function
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    # Create new block mask
    block_mask = create_block_mask(
        causal_mask,
        B=batch_size,
        H=num_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
    )

    # Add to cache with LRU eviction
    _block_mask_cache[cache_key] = block_mask
    while len(_block_mask_cache) > _BLOCK_MASK_CACHE_MAX_SIZE:
        _block_mask_cache.popitem(last=False)  # Remove oldest

    logger.debug(f"FlexAttention: created block mask for shape ({batch_size}, {num_heads}, {seq_len})")
    return block_mask


def clear_block_mask_cache() -> None:
    """Clear the block mask cache. Call when unloading models."""
    _block_mask_cache.clear()
    logger.debug("FlexAttention: block mask cache cleared")


def reset_flex_state() -> None:
    """
    Reset all FlexAttention global state.

    Call this when unloading models to ensure:
    - Block mask cache is cleared (frees GPU memory)
    - Compiled function reference is released
    - Registration state is reset for clean re-registration

    This ensures complete GPU memory release after model unload.
    """
    global _flex_registered, _compiled_flex_attention

    clear_block_mask_cache()
    _compiled_flex_attention = None
    _flex_registered = False
    logger.debug("FlexAttention: all state reset")


def _check_sm80_or_higher() -> bool:
    """
    Check if current GPU supports FlexAttention (SM80+).

    FlexAttention is implemented in Triton and requires SM80 or higher:
    - Ampere: SM80 (A100), SM86 (RTX 30xx), SM87 (Jetson)
    - Ada: SM89 (RTX 40xx)
    - Hopper: SM90 (H100)

    NOT supported:
    - Turing: SM75 (RTX 20xx)
    - Volta: SM70 (V100)

    Returns:
        True if GPU supports FlexAttention, False otherwise.
    """
    if not torch.cuda.is_available():
        return False

    props = torch.cuda.get_device_properties(0)
    # SM80 = 8.0, so major >= 8 is required
    return props.major >= 8


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
    """Fallback to SDPA attention when FlexAttention cannot be used."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    return sdpa_attention_forward(
        module, query, key, value, attention_mask,
        scaling=scaling, dropout=dropout, **kwargs
    )


def flex_attention_forward(
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
    FlexAttention forward function for transformers attention dispatch.

    Fallback conditions (uses SDPA instead):
    - GPU < SM80 (FlexAttention requires Ampere or newer)
    - Non-causal attention (FlexAttention excels at causal, custom patterns need more setup)

    Args:
        module: Attention module (unused, required by interface)
        query: Query tensor [batch, heads, seq_len, head_dim]
        key: Key tensor [batch, heads, seq_len, head_dim]
        value: Value tensor [batch, heads, seq_len, head_dim]
        attention_mask: Optional attention mask (ignored for causal attention)
        scaling: Optional scale factor (default: 1/sqrt(head_dim))
        dropout: Dropout probability
        **kwargs: Additional args (is_causal, etc.)

    Returns:
        Tuple of (output tensor, None) - output is [batch, seq_len, heads, head_dim]
    """
    is_causal = kwargs.get('is_causal', False)

    # Fallback: GPU must support SM80+
    if not _check_sm80_or_higher():
        logger.debug("FlexAttention: GPU < SM80, falling back to SDPA")
        return _fallback_to_sdpa(
            module, query, key, value, attention_mask,
            scaling, dropout, **kwargs
        )

    # For non-causal attention, fall back to SDPA (simpler than setting up custom mask)
    if not is_causal:
        logger.debug("FlexAttention: non-causal attention, falling back to SDPA")
        return _fallback_to_sdpa(
            module, query, key, value, attention_mask,
            scaling, dropout, **kwargs
        )

    try:
        # Get compiled flex_attention
        flex_attn_fn = _get_compiled_flex_attention()

        # Get dimensions
        batch_size, num_heads, seq_len, head_dim = query.shape

        # Calculate scale if not provided
        if scaling is None:
            scaling = 1.0 / (head_dim ** 0.5)

        # Get cached causal block mask (avoids expensive create_block_mask on every call)
        block_mask = _get_or_create_block_mask(
            batch_size, num_heads, seq_len, query.device
        )

        # Call FlexAttention
        # Input shape: [batch, heads, seq_len, head_dim]
        out = flex_attn_fn(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            block_mask=block_mask,
            scale=scaling,
        )

        # Transpose to match expected output format
        # Input: [batch, heads, seq_len, head_dim]
        # Output: [batch, seq_len, heads, head_dim]
        out = out.transpose(1, 2).contiguous()

        return out, None

    except Exception as e:
        logger.warning(f"FlexAttention failed: {e}, falling back to SDPA")
        return _fallback_to_sdpa(
            module, query, key, value, attention_mask,
            scaling, dropout, **kwargs
        )


def register_flex_attention() -> bool:
    """
    Register FlexAttention with transformers attention dispatcher.

    Uses the modern AttentionInterface API (transformers 4.57+).

    Returns:
        True if registration succeeded, False otherwise.
    """
    global _flex_registered

    if _flex_registered:
        return True

    try:
        # Verify PyTorch version supports FlexAttention (2.5+)
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if torch_version < (2, 5):
            logger.warning(
                f"FlexAttention requires PyTorch 2.5+, got {torch.__version__}. "
                "FlexAttention will not be available."
            )
            return False

        # Verify GPU supports SM80+
        if not _check_sm80_or_higher():
            props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
            arch = f"sm{props.major}{props.minor}" if props else "no GPU"
            logger.warning(
                f"FlexAttention requires SM80+ GPU (Ampere or newer), got {arch}. "
                "FlexAttention will not be available."
            )
            return False

        # Verify flex_attention is available
        try:
            from torch.nn.attention.flex_attention import flex_attention  # noqa: F401
        except ImportError as e:
            logger.warning(f"FlexAttention not available in torch: {e}")
            return False

        # Register with transformers AttentionInterface
        try:
            from transformers import AttentionInterface, AttentionMaskInterface
            from transformers.masking_utils import sdpa_mask

            AttentionInterface.register("flex_attn", flex_attention_forward)
            AttentionMaskInterface.register("flex_attn", sdpa_mask)

            _flex_registered = True

            props = torch.cuda.get_device_properties(0)
            arch = f"sm{props.major}{props.minor}"
            logger.info(f"FlexAttention registered with transformers (GPU: {arch})")
            return True

        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to register FlexAttention: {e}")
            return False

    except Exception as e:
        logger.error(f"Failed to register FlexAttention: {e}")
        return False


def apply_flex_to_submodels(model) -> int:
    """
    Apply flex_attn to all sub-models in a Qwen3TTS model.

    Qwen3-TTS has multiple sub-models (talker, code_predictor) that each
    have their own attention configuration. This function updates all of
    them to use FlexAttention.

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
            if module.config._attn_implementation != 'flex_attn':
                module.config._attn_implementation = 'flex_attn'
                updated += 1

    if updated > 0:
        logger.debug(f"Applied flex_attn to {updated} sub-model config(s)")

    return updated


def is_flex_available() -> bool:
    """
    Check if FlexAttention is available.

    Returns:
        True if:
        - PyTorch 2.5+ is installed
        - CUDA is available with SM80+ GPU
        - flex_attention module is importable
    """
    try:
        # Check PyTorch version
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if torch_version < (2, 5):
            return False

        # Check CUDA and GPU architecture
        if not _check_sm80_or_higher():
            return False

        # Check flex_attention is importable
        from torch.nn.attention.flex_attention import flex_attention  # noqa: F401
        return True

    except (ImportError, ValueError):
        return False
