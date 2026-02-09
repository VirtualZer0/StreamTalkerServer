"""GPU memory cleanup utilities.

Provides centralized GPU cleanup for all model managers to avoid code duplication.
"""

import gc
import logging
from typing import Any, Dict

import torch

from server.config import GPU_MEMORY_CLEANUP_THRESHOLD_MB

logger = logging.getLogger(__name__)


def full_gpu_cleanup(
    context: str = "unknown",
    clear_attention_state: bool = False,
    clear_audio_cache: bool = True,
) -> Dict[str, Any]:
    """
    Perform thorough GPU memory cleanup.

    Args:
        context: Description of what triggered cleanup (for logging)
        clear_attention_state: Whether to clear FlexAttention/SageAttention state
        clear_audio_cache: Whether to clear audio resampler cache

    Returns:
        Dictionary with cleanup statistics
    """
    cleanup_stats: Dict[str, Any] = {
        "gc_passes": 0,
        "gc_collected": [],
        "cuda_available": torch.cuda.is_available(),
        "cuda_memory_before": 0,
        "cuda_memory_after": 0,
        "memory_freed": 0,
    }

    # Step 1: Multiple GC passes (catches circular references)
    for _ in range(3):
        collected = gc.collect()
        cleanup_stats["gc_passes"] += 1
        cleanup_stats["gc_collected"].append(collected)

    if not torch.cuda.is_available():
        total_collected = sum(cleanup_stats["gc_collected"])
        logger.debug(
            f"GPU cleanup ({context}): GC collected {total_collected} objects "
            f"in {cleanup_stats['gc_passes']} passes (CUDA not available)"
        )
        return cleanup_stats

    # Record memory before cleanup
    cleanup_stats["cuda_memory_before"] = torch.cuda.memory_allocated()

    # Step 2: Synchronize CUDA streams
    torch.cuda.synchronize()

    # Step 3: Clear CUDA cache
    torch.cuda.empty_cache()

    # Step 4: Reset peak memory stats (for cleaner monitoring)
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        logger.debug(f"Could not reset peak memory stats: {e}")

    # Step 5: Reset accumulated memory stats
    try:
        torch.cuda.reset_accumulated_memory_stats()
    except Exception as e:
        logger.debug(f"Could not reset accumulated memory stats: {e}")

    # Step 6: Clear torch.compile / dynamo cache (if used)
    try:
        torch._dynamo.reset()
    except (AttributeError, Exception) as e:
        logger.debug(f"Could not reset dynamo cache: {e}")

    # Step 7: Clear attention implementation state (only for TTS models)
    if clear_attention_state:
        try:
            from server.models.flex_attention import reset_flex_state
            reset_flex_state()
        except ImportError:
            pass  # FlexAttention not available

        try:
            from server.models.sage_attention import reset_sage_state
            reset_sage_state()
        except ImportError:
            pass  # SageAttention not available

    # Step 8: Clear audio resampler cache
    if clear_audio_cache:
        try:
            from server.utils.audio import clear_resampler_cache
            clear_resampler_cache()
        except ImportError:
            pass  # Should not happen, but be safe

    # Step 9: Final synchronize and empty cache
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Record memory after cleanup
    cleanup_stats["cuda_memory_after"] = torch.cuda.memory_allocated()
    cleanup_stats["memory_freed"] = (
        cleanup_stats["cuda_memory_before"] - cleanup_stats["cuda_memory_after"]
    )

    # Log cleanup results
    total_collected = sum(cleanup_stats["gc_collected"])
    memory_before_mb = cleanup_stats["cuda_memory_before"] / 1024 / 1024
    memory_after_mb = cleanup_stats["cuda_memory_after"] / 1024 / 1024
    memory_freed_mb = cleanup_stats["memory_freed"] / 1024 / 1024

    logger.info(
        f"GPU cleanup ({context}): GC collected {total_collected} objects in "
        f"{cleanup_stats['gc_passes']} passes, "
        f"CUDA memory: {memory_before_mb:.1f} MB -> {memory_after_mb:.1f} MB "
        f"(freed {memory_freed_mb:.1f} MB)"
    )

    # Warn if significant memory still allocated
    if cleanup_stats["cuda_memory_after"] > GPU_MEMORY_CLEANUP_THRESHOLD_MB * 1024 * 1024:
        logger.warning(
            f"GPU memory not fully released after unload: "
            f"{memory_after_mb:.1f} MB still allocated. "
            f"This may indicate memory leaks or external references."
        )

    return cleanup_stats
