"""Model manager for dynamic model loading and unloading."""

import asyncio
import gc
import logging
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch

from server.config import (
    ALL_TTS_MODELS,
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_REPETITION_PENALTY,
    MAX_NEW_TOKENS_LIMIT,
    QUANTIZATION_OPTIONS,
)

if TYPE_CHECKING:
    from qwen_tts import Qwen3TTSModel

# NOTE: 'qwen_tts' is the external library package name (rekuenkdr/Qwen3-TTS-streaming fork)
# This project is "Stream Talker Server" - the imports above refer to the dependency.

logger = logging.getLogger(__name__)


# =============================================================================
# Monkey-patch for qwen-tts to fix stopping_criteria not being forwarded
# =============================================================================

_ORIGINAL_GENERATE = None  # Store original method for reference
_PATCH_APPLIED = False
_TOKENIZER_PATCH_APPLIED = False
_DECODE_PATCH_APPLIED = False


def _patch_tokenizer_padding():
    """
    Monkey-patch Qwen3TTSTokenizerV2CausalTransConvNet to fix padding bug.

    Official fix from QwenLM/Qwen3-TTS commit 5f8581d (Feb 5, 2026):
    - Changed left_pad from math.ceil(pad) to 0
    - Changed right_pad assignment and added conditional check

    This fixes audio artifacts caused by incorrect ConvTranspose1d padding.
    """
    global _TOKENIZER_PATCH_APPLIED

    if _TOKENIZER_PATCH_APPLIED:
        return True

    try:
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2CausalTransConvNet
        )
    except ImportError:
        logger.warning("Could not import tokenizer class for padding patch")
        return False

    import torch.nn as nn

    original_init = Qwen3TTSTokenizerV2CausalTransConvNet.__init__

    def patched_init(self, in_channels, out_channels, kernel_size, stride=1):
        # Call nn.Module.__init__ directly (not original_init to avoid old padding logic)
        nn.Module.__init__(self)
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)
        pad = kernel_size - stride
        self.left_pad = 0  # Fixed: was math.ceil(pad)
        self.right_pad = int(pad)  # Fixed: was pad = self.left_pad

    original_forward = Qwen3TTSTokenizerV2CausalTransConvNet.forward

    def patched_forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        if self.right_pad > 0:  # Fixed: added conditional
            hidden_state = hidden_state[..., : hidden_state.shape[-1] - self.right_pad]
        return hidden_state.contiguous()

    Qwen3TTSTokenizerV2CausalTransConvNet.__init__ = patched_init
    Qwen3TTSTokenizerV2CausalTransConvNet.forward = patched_forward

    _TOKENIZER_PATCH_APPLIED = True
    logger.info("[PATCH] Applied tokenizer padding fix (upstream 5f8581d)")
    return True


def _patch_qwen_tts_generate():
    """
    Monkey-patch Qwen3TTSForConditionalGeneration.generate to forward kwargs to talker.

    Strategy: Patch talker.generate instead to intercept and log the call,
    and inject stopping_criteria from a thread-local storage set before calling.
    """
    global _ORIGINAL_GENERATE, _PATCH_APPLIED

    if _PATCH_APPLIED:
        logger.debug("qwen-tts patch already applied")
        return True

    try:
        from qwen_tts.core.models.modeling_qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
            Qwen3TTSTalkerForConditionalGeneration,
        )
    except ImportError:
        logger.warning("Could not import qwen-tts classes for patching")
        return False

    # Store original talker generate
    original_talker_generate = Qwen3TTSTalkerForConditionalGeneration.generate

    # Thread-local storage for passing stopping_criteria
    import threading
    _patch_context = threading.local()

    def patched_talker_generate(self, *args, **kwargs):
        """Patched talker.generate that logs and can inject stopping_criteria."""
        # Check if we have stopping_criteria to inject
        stopping_criteria = getattr(_patch_context, 'stopping_criteria', None)
        if stopping_criteria is not None and 'stopping_criteria' not in kwargs:
            kwargs['stopping_criteria'] = stopping_criteria
            logger.info(f"[PATCH] Injected stopping_criteria into talker.generate")

        # Log what we're receiving
        max_tokens = kwargs.get('max_new_tokens', 'default')
        eos_id = kwargs.get('eos_token_id', 'default')
        has_stopping = 'stopping_criteria' in kwargs
        logger.info(
            f"[PATCH] talker.generate called: max_new_tokens={max_tokens}, "
            f"eos_token_id={eos_id}, has_stopping_criteria={has_stopping}"
        )

        # Call original
        result = original_talker_generate(self, *args, **kwargs)

        # Log result
        if hasattr(result, 'sequences'):
            seq_len = result.sequences.shape[-1] if hasattr(result.sequences, 'shape') else 'unknown'
            logger.info(f"[PATCH] talker.generate completed: sequence_length={seq_len}")

        return result

    # Patch talker generate
    Qwen3TTSTalkerForConditionalGeneration.generate = patched_talker_generate

    # Also patch the main generate to capture stopping_criteria
    original_main_generate = Qwen3TTSForConditionalGeneration.generate

    def patched_main_generate(self, *args, **kwargs):
        """Patched main generate that captures stopping_criteria for the talker."""
        stopping_criteria = kwargs.pop('stopping_criteria', None)

        if stopping_criteria is not None:
            logger.info(f"[PATCH] Main generate received stopping_criteria, storing for talker")
            _patch_context.stopping_criteria = stopping_criteria
        else:
            _patch_context.stopping_criteria = None

        try:
            return original_main_generate(self, *args, **kwargs)
        finally:
            # Clean up
            _patch_context.stopping_criteria = None

    Qwen3TTSForConditionalGeneration.generate = patched_main_generate

    _PATCH_APPLIED = True
    logger.info("[PATCH] Successfully patched qwen-tts to forward stopping_criteria")
    return True


def _patch_decode_padding():
    """
    Monkey-patch tokenizer decode to fix padding value bug.

    Official fix from QwenLM/Qwen3-TTS commit 6cafe558 (Feb 6, 2026):
    - Changed padding_value from 0 to -1 (0 is a valid audio code token)
    - Changed audio_lengths calculation from > 0 to > -1
    - Added torch.clamp(audio_codes, min=0) before decode

    This fixes incorrect audio length calculation in batch generation where
    shorter sequences padded with 0 could have their audio truncated.
    """
    global _DECODE_PATCH_APPLIED

    if _DECODE_PATCH_APPLIED:
        return True

    try:
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2Model,
            Qwen3TTSTokenizerV2DecoderOutput,
        )
    except ImportError:
        logger.warning("Could not import tokenizer model class for decode padding patch")
        return False

    # Patch 1: Fix Qwen3TTSTokenizerV2Model.decode to handle -1 padding
    def patched_decode(self, audio_codes, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        # Fixed: compute lengths before clamping, use > -1 to count all valid tokens
        audio_lengths = (audio_codes[..., 0] > -1).sum(1) * self.decode_upsample_rate
        # Fixed: clamp to 0 so -1 padding doesn't cause index errors in decoder
        audio_codes = torch.clamp(audio_codes, min=0)
        audio_values = self.decoder.chunked_decode(audio_codes.transpose(1, 2)).squeeze(1)
        audio_values = [a[:l] for a, l in zip(audio_values, audio_lengths)]
        if not return_dict:
            return (audio_values,)
        return Qwen3TTSTokenizerV2DecoderOutput(audio_values)

    Qwen3TTSTokenizerV2Model.decode = patched_decode

    # Patch 2: Fix pad_sequence padding_value in inference tokenizer
    try:
        import qwen_tts.inference.qwen3_tts_tokenizer as tok_module
        original_pad = tok_module.pad_sequence

        def _fixed_pad_sequence(sequences, batch_first=False, padding_value=0.0):
            # Use -1 for integer (audio code) tensors, keep 0 for float (ref_mels)
            if padding_value == 0 and sequences and sequences[0].dtype == torch.long:
                padding_value = -1
            return original_pad(
                sequences, batch_first=batch_first, padding_value=padding_value,
            )

        tok_module.pad_sequence = _fixed_pad_sequence
    except (ImportError, AttributeError):
        logger.warning("Could not patch pad_sequence in inference tokenizer")
        return False

    _DECODE_PATCH_APPLIED = True
    logger.info("[PATCH] Applied decode padding fix (upstream 6cafe558)")
    return True


class CUDAOutOfMemoryError(Exception):
    """
    Raised when CUDA runs out of memory with actionable guidance.

    Provides user-friendly error messages with:
    - Current VRAM availability
    - Actionable suggestions for recovery
    """

    def __init__(self, original_error: Exception, context: str = "operation"):
        self.original_error = original_error
        self.context = context
        self.vram_info = self._get_vram_info()
        super().__init__(self._format_message())

    def _get_vram_info(self) -> str:
        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info()
                return f"{free / 1024**3:.1f}GB free of {total / 1024**3:.1f}GB"
            except Exception:
                return "unknown"
        return "CUDA not available"

    def _format_message(self) -> str:
        return (
            f"GPU out of memory during {self.context}. "
            f"VRAM: {self.vram_info}. "
            f"Suggestions: 1) Unload unused models, 2) Use smaller batch size, "
            f"3) Enable quantization (float8 recommended), 4) Restart server. "
            f"Original error: {self.original_error}"
        )



@dataclass
class LoadTimings:
    """Tracks timing for each loading phase."""
    load: float = 0.0
    quantize: float = 0.0
    fork_opts: float = 0.0
    attention_impl: str = ""


def _enable_infnan_removal(model: "Qwen3TTSModel", model_id: str) -> int:
    """
    Enable remove_invalid_values on ALL generation configs.

    CRITICAL: Both talker.generate() AND talker.code_predictor.generate() are called
    during inference (see modeling_qwen3_tts.py lines 1672 and 2267).
    We must set remove_invalid_values on BOTH to prevent NaN/Inf crashes.

    The InfNanRemoveLogitsProcessor (enabled by remove_invalid_values=True) clamps:
    - +inf → max float value for dtype
    - -inf → min float value for dtype
    - nan → 0.0

    This prevents "probability tensor contains inf/nan" errors in torch.multinomial().

    Args:
        model: The loaded Qwen3TTSModel
        model_id: Short model identifier for logging

    Returns:
        Number of generation configs updated.
    """
    inner = getattr(model, 'model', None)
    if inner is None:
        logger.warning(f"[{model_id}] Could not access inner model for infnan removal")
        return 0

    count = 0

    # Get talker (required for both targets)
    talker = getattr(inner, 'talker', None)
    if talker is None:
        logger.warning(f"[{model_id}] Could not access talker for infnan removal")
        return 0

    # Target 1: talker.generation_config (used by self.talker.generate() at line 2267)
    talker_gen_config = getattr(talker, 'generation_config', None)
    if talker_gen_config is not None:
        if not getattr(talker_gen_config, 'remove_invalid_values', False):
            talker_gen_config.remove_invalid_values = True
            logger.info(f"[{model_id}] Enabled remove_invalid_values on model.talker.generation_config")
            count += 1
    else:
        logger.warning(f"[{model_id}] No generation_config found on talker")

    # Target 2: talker.code_predictor.generation_config (used by self.code_predictor.generate() at line 1672)
    code_predictor = getattr(talker, 'code_predictor', None)
    if code_predictor is not None:
        cp_gen_config = getattr(code_predictor, 'generation_config', None)
        if cp_gen_config is not None:
            if not getattr(cp_gen_config, 'remove_invalid_values', False):
                cp_gen_config.remove_invalid_values = True
                logger.info(f"[{model_id}] Enabled remove_invalid_values on model.talker.code_predictor.generation_config")
                count += 1
        else:
            logger.warning(f"[{model_id}] No generation_config found on code_predictor")
    else:
        logger.warning(f"[{model_id}] Could not access code_predictor for infnan removal")

    # Summary
    if count == 0:
        logger.warning(f"[{model_id}] Could not enable infnan removal on ANY generation config!")
    else:
        logger.info(f"[{model_id}] Enabled infnan removal on {count} generation config(s)")

    return count


def _configure_generation_limits(
    model: "Qwen3TTSModel",
    model_id: str,
    max_tokens: int = MAX_NEW_TOKENS_LIMIT,
) -> int:
    """
    Force generation limits on model's internal configs.

    CRITICAL: The qwen-tts `generate_voice_clone()` does NOT pass `max_new_tokens`
    or `eos_token_id` to internal `.generate()` calls. We must override the model's
    generation_config directly to ensure limits are respected.

    The model has dual components:
    1. **Talker** (main transformer, 28 layers)
    2. **Code Predictor** (acoustic predictor, 5 layers)

    Each may have separate generation_config. We need to override BOTH.

    Args:
        model: The loaded Qwen3TTSModel
        model_id: Short model identifier for logging
        max_tokens: Maximum new tokens limit (default: from config)

    Returns:
        Number of generation configs updated.
    """
    count = 0

    # Access inner model
    inner = getattr(model, 'model', None)
    if inner is None:
        logger.warning(f"[{model_id}] Could not access inner model for generation limits")
        return 0

    # Configure main model generation_config if it exists
    main_gen_config = getattr(inner, 'generation_config', None)
    if main_gen_config is not None:
        main_gen_config.max_new_tokens = max_tokens
        # eos_token_id handled internally by rekuenkdr fork (commit 3640831)
        main_gen_config.pad_token_id = 2150  # Suppress warning spam
        main_gen_config.repetition_penalty = DEFAULT_REPETITION_PENALTY
        logger.info(
            f"[{model_id}] Configured model.generation_config: "
            f"max_new_tokens={max_tokens}, repetition_penalty={DEFAULT_REPETITION_PENALTY}"
        )
        count += 1

    # Get talker
    talker = getattr(inner, 'talker', None)
    if talker is None:
        logger.warning(f"[{model_id}] Could not access talker for generation limits")
        return count

    # Configure talker.generation_config (used by self.talker.generate() at line 2267)
    talker_gen_config = getattr(talker, 'generation_config', None)
    if talker_gen_config is not None:
        talker_gen_config.max_new_tokens = max_tokens
        # eos_token_id handled internally by rekuenkdr fork (commit 3640831)
        talker_gen_config.pad_token_id = 2150
        talker_gen_config.repetition_penalty = DEFAULT_REPETITION_PENALTY
        logger.info(
            f"[{model_id}] Configured talker.generation_config: "
            f"max_new_tokens={max_tokens}"
        )
        count += 1

    # Configure code_predictor.generation_config (used by self.code_predictor.generate() at line 1672)
    code_predictor = getattr(talker, 'code_predictor', None)
    if code_predictor is not None:
        cp_gen_config = getattr(code_predictor, 'generation_config', None)
        if cp_gen_config is not None:
            cp_gen_config.max_new_tokens = max_tokens
            # eos_token_id handled internally by rekuenkdr fork (commit 3640831)
            cp_gen_config.pad_token_id = 2150
            cp_gen_config.repetition_penalty = DEFAULT_REPETITION_PENALTY
            logger.info(
                f"[{model_id}] Configured code_predictor.generation_config: "
                f"max_new_tokens={max_tokens}"
            )
            count += 1

    # Summary
    if count == 0:
        logger.warning(f"[{model_id}] Could not configure generation limits on ANY config!")
    else:
        logger.info(f"[{model_id}] Configured generation limits on {count} config(s)")

    return count


def _load_model_sync(
    hf_model_id: str,
    device: str,
    model_id: str,
    attention: str = "auto",
) -> Tuple["Qwen3TTSModel", str]:
    """
    Synchronous model loading with configurable attention implementation.

    Args:
        hf_model_id: HuggingFace model identifier
        device: Device to load model on
        model_id: Short model identifier for logging
        attention: Attention implementation to use:
            - "auto": Try flash_attention_2 > sdpa > eager
            - "sage_attn": Use SageAttention (requires sageattention package)
            - "flex_attn": Use FlexAttention (requires PyTorch 2.5+, SM80+ GPU)
            - "flash2_attn": Use Flash Attention 2
            - "sdpa": Use PyTorch's scaled dot-product attention
            - "eager": Use eager attention (slowest, always available)

    Returns:
        Tuple of (loaded model, attention implementation used)
    """
    from qwen_tts import Qwen3TTSModel

    # Apply monkey-patches for qwen-tts fixes
    _patch_qwen_tts_generate()  # Fix stopping_criteria not being forwarded
    _patch_tokenizer_padding()  # Fix padding bug in 12Hz tokenizer (upstream 5f8581d)
    _patch_decode_padding()  # Fix decode padding value bug (upstream 6cafe558)

    # Map API attention names to HuggingFace internal names
    if attention == "flash2_attn":
        attention = "flash_attention_2"

    # Determine if loading on CPU
    is_cpu = device == "cpu"

    # CPU requires float32 (bfloat16 is not well-supported on CPU)
    # GPU dtype depends on architecture AND model:
    # - 0.6B: ALWAYS bfloat16 (float16 has overflow issues causing inf/nan crashes)
    # - 1.7B: SM >= 80 (Ampere+): bfloat16, SM < 80 (Turing): float16
    if is_cpu:
        dtype = torch.float32
    else:
        from server.utils.gpu import get_model_dtype, get_sm_version, get_gpu_architecture

        sm_version = get_sm_version()
        # Use model-specific dtype selection (0.6B always uses bfloat16)
        dtype = get_model_dtype(model_id, sm_version)

        # Log dtype selection (get_model_dtype already logs for special cases)
        arch = get_gpu_architecture(sm_version)
        if dtype == torch.float16:
            logger.info(
                f"[{model_id}] Using float16 (SM{sm_version}/{arch} lacks native bf16 support)"
            )
        elif model_id != "0.6B":  # 0.6B already logged in get_model_dtype
            logger.info(f"[{model_id}] Using {dtype} (SM{sm_version}/{arch})")

    # CPU only supports eager attention (flash_attention_2, sage_attn, sdpa are GPU-only)
    if is_cpu:
        if attention not in ("eager", "auto"):
            logger.warning(
                f"[{model_id}] Attention '{attention}' not supported on CPU, using 'eager'"
            )
        attention = "eager"

    # Handle sage_attn specially - requires registration (GPU only)
    if attention == "sage_attn":
        from server.models.sage_attention import register_sage_attention, apply_sage_to_submodels, is_sage_available

        if not is_sage_available():
            raise RuntimeError("SageAttention not available. Install sageattention>=2.2.0 and ensure CUDA is available.")

        if not register_sage_attention():
            raise RuntimeError("Failed to register SageAttention")

        logger.info(f"[{model_id}] Loading with SageAttention...")
        load_kwargs = {
            "attn_implementation": "sage_attn",
            "device_map": device,
            "dtype": dtype,
        }
        model = Qwen3TTSModel.from_pretrained(hf_model_id, **load_kwargs)

        # Apply to sub-models (talker, code_predictor)
        updated = apply_sage_to_submodels(model)
        logger.info(f"[{model_id}] SageAttention applied to {updated} sub-model(s)")
        return model, "sage_attn"

    # Handle flex_attn specially - requires registration (GPU only, SM80+)
    if attention == "flex_attn":
        from server.models.flex_attention import register_flex_attention, apply_flex_to_submodels, is_flex_available

        if not is_flex_available():
            raise RuntimeError(
                "FlexAttention not available. Requires PyTorch 2.5+ and SM80+ GPU "
                "(RTX 30xx or newer). RTX 20xx is not supported."
            )

        if not register_flex_attention():
            raise RuntimeError("Failed to register FlexAttention")

        logger.info(f"[{model_id}] Loading with FlexAttention...")
        load_kwargs = {
            "attn_implementation": "flex_attn",
            "device_map": device,
            "dtype": dtype,
        }
        model = Qwen3TTSModel.from_pretrained(hf_model_id, **load_kwargs)

        # Apply to sub-models (talker, code_predictor)
        updated = apply_flex_to_submodels(model)
        logger.info(f"[{model_id}] FlexAttention applied to {updated} sub-model(s)")
        return model, "flex_attn"

    # Determine implementations to try
    if attention == "auto":
        if is_cpu:
            attn_implementations = ["eager"]
        else:
            attn_implementations = ["flash_attention_2", "sdpa", "eager"]
    else:
        attn_implementations = [attention]

    for attn_impl in attn_implementations:
        try:
            logger.debug(f"[{model_id}] Attempting to load with {attn_impl} attention (dtype={dtype})...")

            load_kwargs = {
                "attn_implementation": attn_impl,
                "device_map": device,
                "dtype": dtype,
            }

            model = Qwen3TTSModel.from_pretrained(hf_model_id, **load_kwargs)
            return model, attn_impl

        except Exception as attn_error:
            if attn_impl == attn_implementations[-1]:
                raise
            logger.warning(
                f"[{model_id}] Failed to load with {attn_impl} attention: {attn_error}. "
                f"Trying next fallback..."
            )

    # Should never reach here, but satisfy type checker
    raise RuntimeError(f"Failed to load model {model_id} with any attention implementation")


def _apply_quantization_sync(
    model: "Qwen3TTSModel",
    quantization: str,
    model_id: str,
) -> bool:
    """
    Apply torchao quantization to model synchronously.

    Args:
        model: The loaded model
        quantization: "int8" or "float8"
        model_id: Short model identifier for logging

    Returns:
        True if quantization was applied, False otherwise
    """
    try:
        from torchao.quantization import quantize_, Int8WeightOnlyConfig

        if not hasattr(model, 'model') or model.model is None:
            logger.warning(
                f"[{model_id}] Model does not have inner .model attribute, "
                f"skipping torchao quantization"
            )
            return False

        quant_type = quantization.upper()

        # Measure memory before quantization
        mem_before = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()
            logger.info(
                f"[{model_id}] Memory before {quant_type} quantization: "
                f"{mem_before / 1024**3:.2f} GB"
            )

        logger.info(f"[{model_id}] Applying {quant_type} quantization...")

        # Use CUDA device for faster quantization if available
        quant_device = "cuda" if torch.cuda.is_available() else None

        # Wrap quantization in inference_mode to ensure tensors are created
        # in the same context they'll be used (fixes float8 compatibility)
        with torch.inference_mode():
            if quantization == "int8":
                quantize_(model.model, Int8WeightOnlyConfig(version=2), device=quant_device)
            elif quantization == "float8":
                from torchao.quantization import Float8WeightOnlyConfig
                from server.utils.gpu import get_sm_version

                # Select optimal FP8 format based on GPU architecture
                sm_version = get_sm_version()
                if sm_version >= 89:
                    # Ada Lovelace+ (SM89+): Use E4M3 for better precision
                    fp8_dtype = torch.float8_e4m3fn
                    fp8_name = "e4m3fn"
                else:
                    # Ampere (SM86-88): Use E5M2 for compatibility
                    fp8_dtype = torch.float8_e5m2
                    fp8_name = "e5m2"

                logger.info(f"[{model_id}] Using float8_{fp8_name} for SM{sm_version}")
                quantize_(
                    model.model,
                    Float8WeightOnlyConfig(weight_dtype=fp8_dtype),
                    device=quant_device,
                )
            else:
                logger.warning(f"[{model_id}] Unknown quantization type: {quantization}")
                return False

        # CRITICAL: Force memory cleanup after quantization
        # Without this, original bf16 tensors may not be freed immediately
        # See: https://github.com/pytorch/ao/issues/346
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Measure memory after quantization and cleanup
        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated()
            saved_bytes = mem_before - mem_after
            saved_mb = saved_bytes / 1024 / 1024
            saved_gb = saved_bytes / 1024**3

            logger.info(
                f"[{model_id}] Memory after {quant_type} quantization: "
                f"{mem_after / 1024**3:.2f} GB (saved {saved_gb:.2f} GB / {saved_mb:.0f} MB)"
            )

            # Check if quantization had any effect
            # Actual savings vary depending on model architecture and which layers get quantized
            # Threshold set low (15%) to only warn when quantization clearly failed
            min_expected_reduction_pct = 0.15
            if mem_before > 0:
                actual_reduction_pct = saved_bytes / mem_before
                if actual_reduction_pct < min_expected_reduction_pct:
                    logger.warning(
                        f"[{model_id}] Quantization may not have applied correctly: "
                        f"got only {actual_reduction_pct*100:.1f}% memory reduction ({saved_mb:.0f} MB saved). "
                        f"This could indicate torchao compatibility issues or "
                        f"the model has few nn.Linear layers."
                    )

        return True

    except ImportError as import_error:
        logger.warning(
            f"[{model_id}] torchao quantization not available: {import_error}. "
            f"Install torchao>=0.15.0 for quantization support."
        )
        return False
    except Exception as quant_error:
        logger.warning(
            f"[{model_id}] torchao quantization failed, "
            f"continuing without quantization: {quant_error}"
        )
        return False


def _apply_fork_optimizations_sync(
    model: "Qwen3TTSModel",
    model_id: str,
    use_compile: bool = True,
    use_cuda_graphs: bool = True,
    compile_codebook_predictor: bool = True,
    use_fast_codebook: bool = True,
) -> bool:
    """
    Apply streaming optimizations from rekuenkdr fork.

    The fork provides optimized streaming generation with:
    - torch.compile on decoder (use_compile)
    - CUDA graphs for fixed-size decode windows (use_cuda_graphs)
    - Compiled codebook predictor (~2x faster)
    - Fast codebook generation (~18% faster)

    Args:
        model: The loaded model
        model_id: Short model identifier for logging
        use_compile: Apply torch.compile to decoder
        use_cuda_graphs: Capture CUDA graphs for decode windows
        compile_codebook_predictor: Apply torch.compile to codebook predictor
        use_fast_codebook: Use fast codebook generation

    Returns:
        True if fork optimizations were applied, False otherwise
    """
    try:
        # Check if this is the optimized fork
        if not hasattr(model, 'enable_streaming_optimizations'):
            logger.debug(
                f"[{model_id}] Fork optimizations not available "
                f"(enable_streaming_optimizations method not found)"
            )
            return False

        logger.info(
            f"[{model_id}] Enabling fork optimizations: compile={use_compile}, "
            f"cuda_graphs={use_cuda_graphs}, compile_codebook={compile_codebook_predictor}, "
            f"fast_codebook={use_fast_codebook}"
        )
        model.enable_streaming_optimizations(
            decode_window_frames=80,
            use_compile=use_compile,
            compile_mode="reduce-overhead",
            use_cuda_graphs=use_cuda_graphs,
            use_fast_codebook=use_fast_codebook,
            compile_codebook_predictor=compile_codebook_predictor,
        )
        logger.info(f"[{model_id}] Fork streaming optimizations enabled")
        return True

    except Exception as fork_error:
        logger.warning(
            f"[{model_id}] Fork optimizations failed: {fork_error}"
        )
        return False


def _run_warmup_inference_sync(
    model: "Qwen3TTSModel",
    texts: List[str],
    languages: List[str],
    warmup_prompt: Any,
    model_id: str,
) -> None:
    """
    Run warmup inference synchronously (for use in executor).

    Args:
        model: The loaded model
        texts: List of texts to generate (batch mode)
        languages: List of languages (one per text)
        warmup_prompt: Voice clone prompt to use
        model_id: Model ID for logging
    """
    try:
        logger.debug(f"[{model_id}] Starting batch inference with {len(texts)} texts...")
        with torch.inference_mode():
            model.generate_voice_clone(
                text=texts,
                language=languages,
                voice_clone_prompt=warmup_prompt,
                do_sample=False,
                max_new_tokens=2048,
            )
        logger.debug(f"[{model_id}] Batch inference completed successfully")
    except Exception as e:
        logger.error(f"[{model_id}] Warmup inference failed: {e}\n{traceback.format_exc()}")
        raise


def _run_single_warmup_sync(
    model: "Qwen3TTSModel",
    text: str,
    language: str,
    warmup_prompt: Any,
    model_id: str,
) -> None:
    """
    Run single warmup inference synchronously (for use in executor).
    """
    try:
        logger.debug(f"[{model_id}] Starting single inference...")
        with torch.inference_mode():
            model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=warmup_prompt,
                do_sample=False,
                max_new_tokens=2048,
            )
        logger.debug(f"[{model_id}] Single inference completed successfully")
    except Exception as e:
        logger.error(f"[{model_id}] Single warmup inference failed: {e}\n{traceback.format_exc()}")
        raise


def _create_warmup_prompt_sync(
    model: "Qwen3TTSModel",
    ref_audio_tuple: Tuple,
    model_id: str,
) -> Any:
    """
    Create warmup voice prompt synchronously (for use in executor).
    """
    try:
        prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio_tuple,
            x_vector_only_mode=True,
        )
        logger.debug(f"[{model_id}] Created warmup prompt successfully")
        return prompt
    except Exception as e:
        logger.error(f"[{model_id}] Failed to create warmup prompt: {e}\n{traceback.format_exc()}")
        raise


class ModelManager:
    """
    Manages loading, unloading, and status of TTS models.

    Features:
    - Dynamic model loading/unloading
    - Thread-safe concurrent access
    - Usage tracking to prevent unloading during active requests
    - Last access tracking for auto-unload
    """

    _instance: Optional["ModelManager"] = None

    def __new__(cls) -> "ModelManager":
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the model manager."""
        if self._initialized:
            return

        # Model instances (None if not loaded)
        # Use ALL_TTS_MODELS for initialization (static dict), not AVAILABLE_MODELS
        # (which is populated dynamically at runtime)
        self._models: Dict[str, Optional["Qwen3TTSModel"]] = {
            model_id: None for model_id in ALL_TTS_MODELS
        }

        # Per-model loading locks (asyncio locks for async safety)
        self._loading_locks: Dict[str, asyncio.Lock] = {
            model_id: asyncio.Lock() for model_id in ALL_TTS_MODELS
        }

        # Loading state tracking
        self._loading_state: Dict[str, bool] = {
            model_id: False for model_id in ALL_TTS_MODELS
        }

        # Warmup state tracking
        self._warming_up_state: Dict[str, bool] = {
            model_id: False for model_id in ALL_TTS_MODELS
        }

        # Last access timestamp (None if never accessed or not loaded)
        self._last_access: Dict[str, Optional[datetime]] = {
            model_id: None for model_id in ALL_TTS_MODELS
        }

        # --- Inference concurrency state ---
        # Inference semaphores (1 concurrent inference per model)
        self._inference_semaphore: Dict[str, asyncio.Semaphore] = {
            model_id: asyncio.Semaphore(1) for model_id in ALL_TTS_MODELS
        }

        # Whether an unload has been requested during inference
        self._pending_unload: Dict[str, bool] = {
            model_id: False for model_id in ALL_TTS_MODELS
        }

        # Whether inference is currently running for each model
        self._inference_active: Dict[str, bool] = {
            model_id: False for model_id in ALL_TTS_MODELS
        }

        # Thread-safe lock for quick state reads/writes (very short hold time)
        self._state_lock: threading.Lock = threading.Lock()

        # Device configuration
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Enable cuDNN optimizations for better GPU performance
        if torch.cuda.is_available():
            # Enable cuDNN auto-tuner to find the best algorithm for hardware
            # Slight warmup cost on first inference, but faster subsequent ones
            torch.backends.cudnn.benchmark = True
            # Use TF32 for faster matmul on Ampere+ GPUs (minimal precision loss)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Set float32 matmul precision for optimal speed/accuracy trade-off
            # 'high' enables TF32 precision (recommended by dffdeeq fork for ~10-30% speedup)
            # Note: 'medium' is more aggressive but may affect audio quality
            torch.set_float32_matmul_precision('high')
            logger.info("CUDA optimizations enabled: cudnn.benchmark=True, tf32=True, matmul_precision=high")

        # Disable gradient computation globally (inference only server)
        # This reduces memory usage and speeds up operations
        torch.set_grad_enabled(False)
        logger.info("Gradient computation disabled for inference-only mode")

        self._initialized = True
        logger.info(f"ModelManager initialized with device: {self._device}")

    def _full_gpu_cleanup(self, context: str = "unknown") -> Dict[str, Any]:
        """
        Perform thorough GPU memory cleanup.

        Args:
            context: Description of what triggered cleanup (for logging)

        Returns:
            Dictionary with cleanup statistics
        """
        from server.utils.cleanup import full_gpu_cleanup
        return full_gpu_cleanup(
            context=context,
            clear_attention_state=True,  # TTS models use FlexAttention/SageAttention
            clear_audio_cache=True,
        )

    @property
    def device(self) -> str:
        """Get the device being used for models."""
        return self._device

    def get_status(self, model_id: str) -> Dict[str, Any]:
        """
        Get status information for a specific model.

        Args:
            model_id: Model identifier

        Returns:
            Dictionary with model status information
        """
        # Import here to avoid circular imports
        from server.settings import settings_manager

        if model_id not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model ID: {model_id}")

        # Thread-safe state access
        with self._state_lock:
            loaded = self._models[model_id] is not None
            loading = self._loading_state[model_id]
            warming_up = self._warming_up_state.get(model_id, False)
            last_access = self._last_access[model_id]
            inference_active = self._inference_active.get(model_id, False)
            pending_unload = self._pending_unload.get(model_id, False)

        return {
            "loaded": loaded,
            "loading": loading,
            "warming_up": warming_up,
            "auto_unload_minutes": settings_manager.get_auto_unload_minutes(model_id),
            "inactive_since": last_access.isoformat() if last_access else None,
            "in_use_count": 1 if inference_active else 0,
            "inference_active": inference_active,
            "pending_unload": pending_unload,
        }

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all models.

        Returns:
            Dictionary with all model statuses
        """
        return {
            model_id: self.get_status(model_id)
            for model_id in AVAILABLE_MODELS
        }

    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is loaded."""
        if model_id not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model ID: {model_id}")
        with self._state_lock:
            return self._models[model_id] is not None

    def is_loading(self, model_id: str) -> bool:
        """Check if a model is currently being loaded."""
        if model_id not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model ID: {model_id}")
        with self._state_lock:
            return self._loading_state[model_id]

    def get_in_use_count(self, model_id: str) -> int:
        """Get the number of active requests using a model."""
        if model_id not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model ID: {model_id}")
        # Return 1 if inference is active, 0 otherwise (new model)
        with self._state_lock:
            return 1 if self._inference_active.get(model_id, False) else 0

    # --- Thread-safe state accessors ---

    def _set_inference_active(self, model_id: str, active: bool) -> None:
        """Thread-safe setter for inference active state."""
        with self._state_lock:
            self._inference_active[model_id] = active

    def _is_inference_active(self, model_id: str) -> bool:
        """Thread-safe getter for inference active state."""
        with self._state_lock:
            return self._inference_active.get(model_id, False)

    def _set_pending_unload(self, model_id: str, pending: bool) -> None:
        """Thread-safe setter for pending unload state."""
        with self._state_lock:
            self._pending_unload[model_id] = pending

    def _is_pending_unload(self, model_id: str) -> bool:
        """Thread-safe getter for pending unload state."""
        with self._state_lock:
            return self._pending_unload.get(model_id, False)

    def _get_model_ref(self, model_id: str) -> Optional["Qwen3TTSModel"]:
        """Thread-safe getter for model reference."""
        with self._state_lock:
            return self._models.get(model_id)

    async def load_model(
        self,
        model_id: str,
        quantization: str = "none",
        enable_optimizations: bool = True,
        torch_compile: bool = True,
        cuda_graphs: bool = False,
        compile_codebook: bool = True,
        fast_codebook: bool = True,
        attention: str = "auto",
        force_cpu: bool = False,
    ) -> None:
        """
        Load a model into memory.

        If the model is already loaded, this is a no-op.
        If the model is currently loading, waits for loading to complete.

        This method uses run_in_executor() to run blocking operations in a thread pool,
        keeping the event loop responsive. The endpoint will still wait for all operations
        to complete before returning.

        Args:
            model_id: Model identifier ("0.6B" or "1.7B")
            quantization: Quantization mode ("none", "int8", "float8")
            enable_optimizations: Master toggle for fork streaming optimizations (requires GPU)
            torch_compile: Apply torch.compile to decoder (within fork optimizations)
            cuda_graphs: Capture CUDA graphs for decode windows (within fork optimizations)
            compile_codebook: Apply torch.compile to codebook predictor (within fork optimizations)
            fast_codebook: Use fast codebook generation (within fork optimizations)
            attention: Attention implementation ("auto", "sage_attn", "flex_attn", "flash2_attn", "sdpa", "eager")
            force_cpu: Force loading on CPU instead of GPU (default: False)

        Raises:
            ValueError: If model_id is unknown or quantization/attention is invalid
            RuntimeError: If model is not available (not cached locally)
        """
        from server.config import ALL_TTS_MODELS

        # Valid attention options
        valid_attentions = ["auto", "sage_attn", "flex_attn", "flash2_attn", "sdpa", "eager"]
        if attention not in valid_attentions:
            raise ValueError(f"Invalid attention: {attention}. Options: {valid_attentions}")

        # Validate quantization option
        if quantization not in QUANTIZATION_OPTIONS:
            raise ValueError(f"Invalid quantization: {quantization}. Options: {QUANTIZATION_OPTIONS}")

        # Check if it's a valid model ID at all
        if model_id not in ALL_TTS_MODELS:
            raise ValueError(f"Unknown model ID: {model_id}")

        # Check if model is available (cached locally)
        if model_id not in AVAILABLE_MODELS:
            raise RuntimeError(
                f"Model {model_id} is not available. "
                f"It may not be cached locally. "
                f"Available models: {list(AVAILABLE_MODELS.keys())}"
            )

        # If already loaded, return immediately
        if self._models[model_id] is not None:
            logger.debug(f"Model {model_id} is already loaded")
            return

        # Acquire lock for this specific model
        async with self._loading_locks[model_id]:
            # Double-check after acquiring lock
            if self._models[model_id] is not None:
                logger.debug(f"Model {model_id} was loaded while waiting for lock")
                return

            # Mark as loading
            self._loading_state[model_id] = True

            try:
                # Determine device to use
                device = "cpu" if force_cpu else self._device

                # Determine steps for logging
                apply_quant = quantization in ("int8", "float8")
                apply_opts = enable_optimizations and not force_cpu and torch.cuda.is_available()

                total_steps = 1  # Loading is always step 1
                if apply_opts:
                    total_steps += 1
                if apply_quant:
                    total_steps += 1

                logger.info(
                    f"=== Loading model {model_id} (device={device}, attention={attention}, "
                    f"quantization={quantization}, optimizations={enable_optimizations}) ==="
                )

                hf_model_id = AVAILABLE_MODELS[model_id]
                loop = asyncio.get_event_loop()
                timings = LoadTimings()
                current_step = 1

                # Step 1: Load model from HuggingFace (BLOCKING - run in executor)
                logger.info(f"[{model_id}] Step {current_step}/{total_steps}: Loading model on {device}...")
                t0 = time.time()

                model, attn_impl = await loop.run_in_executor(
                    None,
                    _load_model_sync,
                    hf_model_id,
                    device,
                    model_id,
                    attention,
                )

                timings.load = time.time() - t0
                timings.attention_impl = attn_impl
                logger.info(f"[{model_id}] Step {current_step}/{total_steps}: Model loaded in {timings.load:.2f}s "
                           f"(attention={attn_impl})")

                # Enable NaN/Inf removal for float16 stability (critical for RTX 20xx)
                infnan_count = _enable_infnan_removal(model, model_id)
                if infnan_count == 0:
                    logger.warning(
                        f"[{model_id}] Failed to enable infnan removal on any generation config - "
                        f"float16 inference may be unstable on Turing GPUs (SM75)"
                    )

                # Configure generation limits (CRITICAL for preventing infinite generation)
                # This forces max_new_tokens, eos_token_id, and repetition_penalty at MODEL level
                # because qwen-tts ignores these parameters when passed to generate_voice_clone()
                genlimit_count = _configure_generation_limits(model, model_id)
                if genlimit_count == 0:
                    logger.warning(
                        f"[{model_id}] Failed to configure generation limits - "
                        f"model may run indefinitely on some inputs"
                    )

                current_step += 1

                # Step 2: Apply quantization if requested (BLOCKING - run in executor)
                # Quantization must happen BEFORE optimizations so torch.compile sees
                # the actual quantized dtypes and produces correct optimized code
                if apply_quant:
                    quant_type = quantization.upper()
                    logger.info(f"[{model_id}] Step {current_step}/{total_steps}: Applying {quant_type} quantization...")
                    t0 = time.time()

                    success = await loop.run_in_executor(
                        None,
                        _apply_quantization_sync,
                        model,
                        quantization,
                        model_id,
                    )

                    timings.quantize = time.time() - t0
                    if success:
                        logger.info(f"[{model_id}] Step {current_step}/{total_steps}: "
                                   f"Quantization complete in {timings.quantize:.2f}s")
                    else:
                        logger.warning(f"[{model_id}] Step {current_step}/{total_steps}: "
                                      f"Quantization skipped/failed after {timings.quantize:.2f}s")
                    current_step += 1

                # Step 3: Apply fork streaming optimizations (compile, CUDA graphs, codebook)
                # Applied after quantization so torch.compile optimizes for the actual dtypes
                if apply_opts:
                    logger.info(f"[{model_id}] Step {current_step}/{total_steps}: Applying streaming optimizations...")
                    t0 = time.time()
                    fork_applied = await loop.run_in_executor(
                        None,
                        partial(
                            _apply_fork_optimizations_sync,
                            model,
                            model_id,
                            use_compile=torch_compile,
                            use_cuda_graphs=cuda_graphs,
                            compile_codebook_predictor=compile_codebook,
                            use_fast_codebook=fast_codebook,
                        ),
                    )
                    timings.fork_opts = time.time() - t0
                    if fork_applied:
                        logger.info(f"[{model_id}] Step {current_step}/{total_steps}: "
                                   f"Optimizations applied in {timings.fork_opts:.2f}s")
                    else:
                        logger.warning(f"[{model_id}] Step {current_step}/{total_steps}: "
                                      f"Optimizations failed after {timings.fork_opts:.2f}s")

                # Store model and update access time
                self._models[model_id] = model
                self._last_access[model_id] = datetime.now(timezone.utc)

                # Log summary
                total_time = timings.load + timings.quantize + timings.fork_opts
                breakdown_parts = [f"load={timings.load:.2f}s"]
                if apply_quant:
                    breakdown_parts.append(f"quantize={timings.quantize:.2f}s")
                if apply_opts:
                    breakdown_parts.append(f"optimizations={timings.fork_opts:.2f}s")

                logger.info(f"=== Model {model_id} loaded successfully in {total_time:.2f}s ===")
                logger.info(f"    Breakdown: {', '.join(breakdown_parts)}")

            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
                raise
            finally:
                self._loading_state[model_id] = False

    async def unload_model(self, model_id: str, force: bool = False) -> dict:
        """
        Request model unload.

        Non-blocking behavior:
        - If no inference is active: unloads immediately
        - If inference is active: sets pending flag, returns immediately
        - Actual unload happens when inference completes

        Args:
            model_id: Model identifier
            force: If True, wait for inference to complete then unload (blocking)

        Returns:
            dict with status: "unloaded", "not_loaded", or "unload_pending"

        Raises:
            ValueError: If model_id is invalid
        """
        if model_id not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model ID: {model_id}")

        # Check if model is loaded
        with self._state_lock:
            if self._models[model_id] is None:
                logger.debug(f"Model {model_id} is not loaded")
                return {"status": "not_loaded"}

            # Check if inference is active
            if self._inference_active[model_id]:
                # Abort running inference to speed up unload
                from server.api.synthesis import request_abort_inference
                request_abort_inference()
                logger.info(f"Aborting inference for model {model_id} unload")

                self._pending_unload[model_id] = True

                if force:
                    # Wait mode: will block until inference completes (now aborted)
                    logger.info(f"Model {model_id} in use, waiting for aborted inference to complete...")
                else:
                    # Non-blocking mode: return immediately, inference will abort and unload
                    logger.info(f"Model {model_id} unload scheduled (inference aborting)")
                    return {
                        "status": "unload_pending",
                        "message": "Inference aborted, will unload shortly"
                    }

        # If force=True and inference was active, wait for pending unload to complete
        if force and self._is_pending_unload(model_id):
            # Poll until unload completes (inference finishes and triggers unload)
            while self._is_pending_unload(model_id) and self._get_model_ref(model_id) is not None:
                await asyncio.sleep(0.5)
            return {"status": "unloaded"}

        # No inference - unload immediately
        await self._do_unload(model_id)
        return {"status": "unloaded"}

    async def _do_unload(self, model_id: str) -> bool:
        """
        Internal unload implementation.

        Args:
            model_id: Model identifier

        Returns:
            True if unloaded successfully
        """
        async with self._loading_locks[model_id]:
            # Double-check after acquiring lock
            with self._state_lock:
                if self._models[model_id] is None:
                    self._pending_unload[model_id] = False
                    return True

            logger.info(f"Unloading model {model_id}...")

            try:
                # Clear voice prompts for this model FIRST
                # (prompts may reference model tensors)
                from server.voices import voice_cache_manager
                voice_cache_manager.clear_prompt_cache_for_model(model_id)

                # Delete model reference
                with self._state_lock:
                    del self._models[model_id]
                    self._models[model_id] = None
                    self._last_access[model_id] = None
                    self._pending_unload[model_id] = False
                    self._warming_up_state[model_id] = False

                # Perform full GPU cleanup
                self._full_gpu_cleanup(context=f"unload {model_id}")

                logger.info(f"Model {model_id} unloaded successfully")
                return True

            except Exception as e:
                logger.error(f"Error unloading model {model_id}: {e}")
                with self._state_lock:
                    self._pending_unload[model_id] = False
                return False

    async def get_model(self, model_id: str = DEFAULT_MODEL) -> "Qwen3TTSModel":
        """
        Get a model instance, loading it if necessary.

        Args:
            model_id: Model identifier (default: "1.7B")

        Returns:
            The loaded model instance

        Raises:
            ValueError: If model_id is unknown
            RuntimeError: If model is not available (not cached locally)
        """
        from server.config import ALL_TTS_MODELS

        # Check if it's a valid model ID at all
        if model_id not in ALL_TTS_MODELS:
            raise ValueError(f"Unknown model ID: {model_id}")

        # Check if model is available (cached locally)
        if model_id not in AVAILABLE_MODELS:
            raise RuntimeError(
                f"Model {model_id} is not available. "
                f"It may not be cached locally. "
                f"Available models: {list(AVAILABLE_MODELS.keys())}"
            )

        # Load if not already loaded
        if self._models[model_id] is None:
            await self.load_model(model_id)

        return self._models[model_id]

    async def acquire_inference_slot(self, model_id: str) -> "Qwen3TTSModel":
        """
        Acquire inference slot and return model.

        This method:
        - Rejects with HTTP 503 if unload is pending
        - Waits for other inference to complete (semaphore)
        - Marks inference as active

        Caller MUST call release_inference() when GPU operation completes,
        even after timeout (to properly track GPU availability).

        Args:
            model_id: Model identifier

        Returns:
            The loaded model instance

        Raises:
            HTTPException: 503 if model is being unloaded
            ValueError: If model_id is unknown
        """
        from fastapi import HTTPException

        if model_id not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model ID: {model_id}")

        # Check if unload is pending BEFORE waiting for semaphore
        if self._is_pending_unload(model_id):
            raise HTTPException(
                status_code=503,
                detail=f"Model {model_id} is being unloaded"
            )

        # Ensure model is loaded
        model = await self.get_model(model_id)

        # Acquire semaphore (blocks other inference)
        await self._inference_semaphore[model_id].acquire()

        # Double-check after acquiring semaphore
        if self._is_pending_unload(model_id) or self._get_model_ref(model_id) is None:
            self._inference_semaphore[model_id].release()
            raise HTTPException(
                status_code=503,
                detail=f"Model {model_id} is being unloaded"
            )

        self._set_inference_active(model_id, True)
        return model

    def release_inference(self, model_id: str) -> None:
        """
        Release inference slot. Call when executor task completes (not on timeout).

        This properly releases the semaphore when GPU is actually free.
        If unload was pending, triggers the actual unload.

        Args:
            model_id: Model identifier
        """
        self._set_inference_active(model_id, False)
        self._inference_semaphore[model_id].release()

        with self._state_lock:
            self._last_access[model_id] = datetime.now(timezone.utc)
            pending = self._pending_unload.get(model_id, False)

        # If unload was requested during inference, unload now
        if pending:
            task = asyncio.create_task(self._do_unload(model_id))
            task.add_done_callback(self._handle_unload_task_result)

    def release_inference_threadsafe(self, model_id: str, loop: asyncio.AbstractEventLoop) -> None:
        """
        Thread-safe version of release_inference for use from executor threads.

        Uses loop.call_soon_threadsafe() to schedule semaphore release on event loop.
        Falls back to direct release if event loop is closed (during shutdown).

        Args:
            model_id: Model identifier
            loop: The asyncio event loop to schedule on
        """
        self._set_inference_active(model_id, False)

        # Schedule semaphore release on event loop (thread-safe)
        semaphore_released = False
        try:
            loop.call_soon_threadsafe(self._inference_semaphore[model_id].release)
            semaphore_released = True
        except RuntimeError as e:
            # Event loop may be closed during shutdown - try direct release as fallback
            # This is safe during shutdown since no new tasks can acquire the semaphore
            logger.warning(f"[{model_id}] Event loop closed, attempting direct semaphore release: {e}")
            try:
                # Direct release - only safe when loop is closed (no concurrent access)
                self._inference_semaphore[model_id].release()
                semaphore_released = True
                logger.debug(f"[{model_id}] Direct semaphore release succeeded")
            except Exception as release_error:
                logger.error(f"[{model_id}] Failed to release semaphore: {release_error}")

        # Update state even if loop is closed
        with self._state_lock:
            self._last_access[model_id] = datetime.now(timezone.utc)
            pending = self._pending_unload.get(model_id, False)

        # If unload was requested during inference, try to unload now
        if pending:
            def _schedule_unload():
                task = asyncio.create_task(self._do_unload(model_id))
                task.add_done_callback(self._handle_unload_task_result)

            try:
                loop.call_soon_threadsafe(_schedule_unload)
            except RuntimeError as e:
                # Can't schedule unload if loop is closed - clear pending flag
                logger.warning(f"[{model_id}] Cannot schedule unload (loop closed): {e}")
                with self._state_lock:
                    self._pending_unload[model_id] = False

    def _handle_unload_task_result(self, task: asyncio.Task) -> None:
        """
        Callback to handle exceptions from background unload tasks.

        Without this, exceptions from asyncio.create_task() are silently lost.
        """
        try:
            # This will raise if the task had an exception
            task.result()
        except asyncio.CancelledError:
            logger.debug("Background unload task was cancelled")
        except Exception as e:
            logger.error(f"Background unload task failed: {e}\n{traceback.format_exc()}")

    def get_last_access(self, model_id: str) -> Optional[datetime]:
        """Get the last access time for a model."""
        if model_id not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model ID: {model_id}")
        with self._state_lock:
            return self._last_access[model_id]

    async def check_auto_unload(self) -> None:
        """
        Check and perform auto-unload for inactive models.

        This should be called periodically by a background task.
        Non-blocking: if inference is active, unload_model() sets pending flag.
        Also runs periodic garbage collection to clean up circular references.
        """
        from server.settings import settings_manager

        now = datetime.now(timezone.utc)

        # Run garbage collection if no inference is active (lightweight, ~1ms)
        # This helps clean up circular references between caches
        any_inference_active = False
        with self._state_lock:
            any_inference_active = any(self._inference_active.values())

        if not any_inference_active:
            gc.collect()

        for model_id in AVAILABLE_MODELS:
            # Quick state check with threading lock (non-blocking)
            with self._state_lock:
                if self._models[model_id] is None:
                    continue
                if self._pending_unload[model_id]:
                    continue  # Already pending
                last_access = self._last_access[model_id]

            # Get auto-unload setting
            timeout_minutes = settings_manager.get_auto_unload_minutes(model_id)

            # Skip if auto-unload is disabled
            if timeout_minutes == 0:
                continue

            # Check if inactive long enough
            if last_access is None:
                continue

            inactive_seconds = (now - last_access).total_seconds()
            inactive_minutes = inactive_seconds / 60

            if inactive_minutes >= timeout_minutes:
                logger.info(
                    f"Auto-unloading model {model_id} after "
                    f"{inactive_minutes:.1f} minutes of inactivity"
                )
                # unload_model() handles active inference gracefully:
                # - If inference active: sets pending flag, returns immediately
                # - If no inference: unloads immediately
                result = await self.unload_model(model_id)
                logger.debug(f"Auto-unload result for {model_id}: {result}")

    async def warmup_model(
        self,
        model_id: str,
        language: str = None,
        voice: str = None,
        mode: str = "single",
        timeout_sec: int = 120,
    ) -> dict:
        """
        Run warmup inference to prime torch.compile optimization.

        Args:
            model_id: Model identifier ("0.6B" or "1.7B")
            language: Language for warmup phrases (default: from config)
            voice: Optional voice name to use for warmup (uses cached voice prompt).
                   If not provided, uses silence audio.
            mode: Warmup mode - "single" (longest phrase) or "batch" (all phrases)
            timeout_sec: Timeout in seconds for warmup inference (default: 120).
                         If exceeded, returns early with skipped=True to catch hangs.

        Returns:
            dict with warmup statistics

        Raises:
            ValueError: If model_id is unknown
            RuntimeError: If model is not loaded
        """
        if model_id not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model ID: {model_id}")

        if self._models[model_id] is None:
            raise RuntimeError(f"Model {model_id} is not loaded")

        with self._state_lock:
            self._warming_up_state[model_id] = True

        try:
            return await self._run_warmup(model_id, language, voice, mode, timeout_sec)
        finally:
            with self._state_lock:
                self._warming_up_state[model_id] = False

    async def _run_warmup(
        self,
        model_id: str,
        language: str = None,
        voice: str = None,
        mode: str = "single",
        timeout_sec: int = 120,
    ) -> dict:
        """Internal warmup implementation."""
        import numpy as np
        from server.utils.warmup import get_warmup_phrases

        # Get warmup texts for the specified language
        warmup_texts, resolved_language = get_warmup_phrases(language)
        if language and language != resolved_language:
            logger.warning(f"No warmup phrases for language {language}, using {resolved_language}")
        language = resolved_language

        logger.info(f"Starting warmup for model {model_id} with language {language}, "
                   f"voice={voice or 'silence'}, mode={mode}...")

        # CRITICAL: Acquire inference semaphore to prevent concurrent disk writes
        # and concurrent GPU operations during warmup. This ensures:
        # 1. Prompt cache files are not corrupted by concurrent writes
        # 2. GPU inference is serialized (only one operation at a time per model)
        model = await self.acquire_inference_slot(model_id)

        try:
            warmup_times = []
            loop = asyncio.get_event_loop()
            warmup_prompt = None

            # Try to get voice prompt from cache if voice is specified
            if voice:
                try:
                    from server.voices.cache import voice_cache_manager
                    if voice_cache_manager.voice_exists(voice):
                        warmup_prompt = await voice_cache_manager.get_or_create_prompt(
                            voice_name=voice,
                            model_id=model_id,
                            model=model,
                        )
                        logger.info(f"Using cached voice '{voice}' for warmup")
                    else:
                        logger.warning(f"Voice '{voice}' not found, falling back to silence")
                except Exception as e:
                    logger.warning(f"Failed to get voice '{voice}': {e}, falling back to silence")

            # Fall back to silence if no voice prompt available
            if warmup_prompt is None:
                sample_rate = 24000
                silence_audio = np.zeros(sample_rate * 5, dtype=np.float32)  # 5 seconds minimum
                ref_audio_tuple = (silence_audio, sample_rate)

                try:
                    warmup_prompt = await loop.run_in_executor(
                        None,
                        _create_warmup_prompt_sync,
                        model,
                        ref_audio_tuple,
                        model_id,
                    )
                    logger.debug("Created warmup voice prompt from silence")
                except Exception as e:
                    logger.warning(f"Failed to create warmup prompt: {e}\n{traceback.format_exc()}")
                    return {"skipped": True, "reason": str(e)}

            # Run warmup based on mode
            if mode == "batch":
                # Batch mode: run all phrases at once
                languages_batch = [language] * len(warmup_texts)

                try:
                    logger.info(f"[{model_id}] Running batch warmup with {len(warmup_texts)} phrases "
                               f"(timeout={timeout_sec}s)...")
                    t0 = time.time()

                    await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            _run_warmup_inference_sync,
                            model,
                            warmup_texts,
                            languages_batch,
                            warmup_prompt,
                            model_id,
                        ),
                        timeout=timeout_sec,
                    )

                    batch_time = time.time() - t0
                    warmup_times.append(batch_time)
                    logger.info(f"[{model_id}] Batch warmup completed in {batch_time:.2f}s")

                except asyncio.TimeoutError:
                    elapsed = time.time() - t0
                    logger.error(f"[{model_id}] Warmup TIMEOUT after {elapsed:.1f}s - possible hang detected! "
                                f"Check Triton/GPU compatibility (SM75 hang?)")
                    return {"skipped": True, "reason": f"timeout after {elapsed:.1f}s", "timeout_sec": timeout_sec}

                except Exception as e:
                    logger.error(f"[{model_id}] Batch warmup failed: {e}\n{traceback.format_exc()}")
                    warmup_times.append(-1)

            else:
                # Single mode: use longest phrase (last one, index -1)
                longest_phrase = warmup_texts[-1]

                try:
                    logger.info(f"[{model_id}] Running single warmup with longest phrase "
                               f"(timeout={timeout_sec}s)...")
                    t0 = time.time()

                    await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            _run_single_warmup_sync,
                            model,
                            longest_phrase,
                            language,
                            warmup_prompt,
                            model_id,
                        ),
                        timeout=timeout_sec,
                    )

                    elapsed = time.time() - t0
                    warmup_times.append(elapsed)
                    logger.info(f"[{model_id}] Single warmup completed in {elapsed:.2f}s")

                except asyncio.TimeoutError:
                    elapsed = time.time() - t0
                    logger.error(f"[{model_id}] Warmup TIMEOUT after {elapsed:.1f}s - possible hang detected! "
                                f"Check Triton/GPU compatibility (SM75 hang?)")
                    return {"skipped": True, "reason": f"timeout after {elapsed:.1f}s", "timeout_sec": timeout_sec}

                except Exception as e:
                    logger.error(f"[{model_id}] Single warmup failed: {e}\n{traceback.format_exc()}")
                    warmup_times.append(-1)

            # Update last access time
            self._last_access[model_id] = datetime.now(timezone.utc)

            total_time = sum(t for t in warmup_times if t > 0)
            logger.info(f"[{model_id}] Warmup completed in {total_time:.2f}s total")

            return {
                "model_id": model_id,
                "mode": mode,
                "warmup_count": len(warmup_texts) if mode == "batch" else 1,
                "warmup_times": warmup_times,
                "total_time": total_time,
            }
        finally:
            # Always release the inference semaphore when warmup completes or fails
            self.release_inference(model_id)


# Global singleton instance
model_manager = ModelManager()
