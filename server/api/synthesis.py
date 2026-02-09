"""Speech synthesis API endpoints."""

import asyncio
import contextvars
import gc
import io
import logging
import time
import traceback
import zipfile
from dataclasses import dataclass
from typing import List, Literal, Optional, Union
from uuid import uuid4

import numpy as np
import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from transformers import StoppingCriteria, StoppingCriteriaList

from server.config import (
    AVAILABLE_MODELS,
    ALL_TTS_MODELS,
    CJK_LANGUAGES,
    DEFAULT_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_SPEED,
    DEFAULT_TEMPERATURE,
    MAX_BATCH_SIZE,
    MAX_MAX_NEW_TOKENS,
    MAX_REPETITION_PENALTY,
    MAX_SPEED,
    MAX_TEMPERATURE,
    MIN_MAX_NEW_TOKENS,
    MIN_REPETITION_PENALTY,
    MIN_SPEED,
    MIN_TEMPERATURE,
    OUTPUT_DIR,
    SUPPORTED_LANGUAGES,
)
from server.models import model_manager
from server.settings import settings_manager
from server.models.manager import CUDAOutOfMemoryError
from server.utils.audio import (
    apply_speed,
    audio_to_wav_bytes,
    validate_audio_file,
)
from server.voices import voice_cache_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Speech Synthesis"])

# Global abort flag for inference cancellation
_abort_inference = False


@dataclass
class TruncationInfo:
    """Truncation info for current inference request."""
    truncated: bool = False
    reason: Optional[str] = None


_truncation_context: contextvars.ContextVar[TruncationInfo] = contextvars.ContextVar(
    'truncation_info'
)


def get_truncation_info() -> TruncationInfo:
    """Get truncation info for current request context, creating if needed."""
    try:
        return _truncation_context.get()
    except LookupError:
        info = TruncationInfo()
        _truncation_context.set(info)
        return info


def reset_truncation_info() -> None:
    """Reset truncation info for a new inference request."""
    _truncation_context.set(TruncationInfo())


def set_truncation_info(truncated: bool, reason: Optional[str] = None) -> None:
    """Set truncation info for current request context."""
    info = get_truncation_info()
    info.truncated = truncated
    info.reason = reason


def request_abort_inference():
    """Request abortion of current inference."""
    global _abort_inference
    _abort_inference = True


def clear_abort_flag():
    """Clear the abort flag (called at start of new inference)."""
    global _abort_inference
    _abort_inference = False


def is_abort_requested() -> bool:
    """Check if abort was requested."""
    return _abort_inference


class InferenceStoppingCriteria(StoppingCriteria):
    """
    Stops generation if timeout or VRAM limit is exceeded.

    Checked between each token generation step, allowing graceful early stopping
    before the system runs out of memory or hangs indefinitely.

    VRAM checking uses a static maximum limit (configurable via API).
    """

    def __init__(self, model_id: str, timeout: float = None):
        self.model_id = model_id
        # Use settings-based timeout, fall back to default constant
        self.timeout = timeout if timeout is not None else settings_manager.get_inference_timeout()
        self.start_time = time.time()
        # Static VRAM limit from settings (configured via /environment/max-vram)
        self.max_vram_limit = settings_manager.get_max_vram_bytes()
        self.stopped_reason: Optional[str] = None
        self._check_interval = 5  # Check VRAM every N steps
        self._step_count = 0

        # Log initialization for debugging
        limit_mb = self.max_vram_limit / (1024 ** 2) if self.max_vram_limit > 0 else 0
        logger.info(
            f"[{self.model_id}] StoppingCriteria initialized: "
            f"timeout={self.timeout}s, max_vram={limit_mb:.0f}MB"
        )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self._step_count += 1

        # Log every 100 steps to track if stopping criteria is being called
        # DIAGNOSTIC: This helps verify if qwen-tts respects stopping_criteria
        if self._step_count % 100 == 0:
            elapsed = time.time() - self.start_time
            logger.debug(
                f"[{self.model_id}] StoppingCriteria step {self._step_count}, "
                f"elapsed={elapsed:.1f}s, tokens={input_ids.shape[-1]}"
            )

        # Check abort flag first (user requested skip)
        if is_abort_requested():
            self.stopped_reason = "user requested abort"
            logger.warning(f"[{self.model_id}] Stopping generation: {self.stopped_reason}")
            return True

        # Always check timeout
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout:
            self.stopped_reason = f"timeout ({elapsed:.1f}s > {self.timeout}s)"
            logger.warning(f"[{self.model_id}] Stopping generation: {self.stopped_reason}")
            return True

        # Check VRAM on first step and then periodically (not every step - expensive)
        # Uses static max limit from settings (0 = disabled)
        should_check_vram = (
            self._step_count == 1 or  # Always check on first step
            self._step_count % self._check_interval == 0
        )
        if should_check_vram and torch.cuda.is_available() and self.max_vram_limit > 0:
            # Use memory_reserved() to get full GPU memory claimed by PyTorch
            # (not just tensor allocations, but also cached memory)
            vram_current = torch.cuda.memory_reserved()
            current_mb = vram_current / (1024 ** 2)
            limit_mb = self.max_vram_limit / (1024 ** 2)

            if vram_current > self.max_vram_limit:
                self.stopped_reason = f"VRAM limit exceeded ({current_mb:.0f}MB > {limit_mb:.0f}MB)"
                logger.warning(f"[{self.model_id}] Stopping generation: {self.stopped_reason}")
                return True
            elif self._step_count == 1:
                # Log initial VRAM state for debugging
                logger.debug(
                    f"[{self.model_id}] VRAM check: {current_mb:.0f}MB / {limit_mb:.0f}MB limit"
                )

        return False


def ensure_ending_punctuation(text: str, language: str) -> str:
    """
    Ensure text ends with proper punctuation + runway for EOS token generation.

    This is a workaround for upstream Qwen3-TTS bug where missing ending punctuation
    can cause the model to fail generating proper end-of-sequence tokens, leading to
    cutoff audio. See: https://github.com/QwenLM/Qwen3-TTS/issues/55

    Strategy: Uses multiple trailing periods (ellipsis-style) instead of single period
    to give the model more "runway" tokens before needing to emit EOS. This matches
    natural trailing-off patterns in training data and produces minimal/no audible output.

    Args:
        text: Input text to check
        language: Target language (used to determine punctuation style)

    Returns:
        Text with proper ending punctuation and runway for reliable EOS generation
    """
    if not text or not text.strip():
        return text

    text = text.rstrip()  # Remove trailing whitespace
    if not text:
        return text

    # Strong endings that complete a sentence
    strong_endings = {'.', '!', '?', '。', '！', '？'}
    # Weak endings that may leave model uncertain about sentence completion
    weak_endings = {',', ';', ':', '~', '～', '，', '；', '：'}

    last_char = text[-1]
    is_cjk = language in CJK_LANGUAGES

    # Already ends with strong punctuation - add runway periods
    if last_char in strong_endings:
        return text + ("。" if is_cjk else "..")

    # Ends with weak punctuation - replace with strong ending + runway
    if last_char in weak_endings:
        text = text[:-1]  # Remove weak punctuation
        return text + ("。。" if is_cjk else "...")

    # Ends with ellipsis already - leave as-is (already has runway)
    if text.endswith('...') or text.endswith('…') or text.endswith('。。') or text.endswith('。。。'):
        return text

    # No punctuation at all - add full runway
    return text + ("。。" if is_cjk else "...")


# Pydantic models for POST request bodies
class SynthesizeRequest(BaseModel):
    text: Union[str, List[str]] = Field(
        ...,
        description="Text to synthesize. Can be a single string or array of strings for batch processing."
    )
    voice: str = Field(..., description="Voice name")
    speed: Optional[float] = Field(DEFAULT_SPEED, ge=MIN_SPEED, le=MAX_SPEED, description="Speed multiplier")
    model: Optional[Literal["0.6B", "1.7B"]] = Field(DEFAULT_MODEL, description="Model to use")
    language: Optional[Literal[
        "Auto", "Chinese", "English", "Japanese", "Korean",
        "German", "French", "Russian", "Portuguese", "Spanish", "Italian"
    ]] = Field(DEFAULT_LANGUAGE, description="Target language")
    # Inference optimization parameters
    do_sample: Optional[bool] = Field(
        True,
        description="Use sampling for generation. Set to false for deterministic output (faster)."
    )
    temperature: Optional[float] = Field(
        DEFAULT_TEMPERATURE,
        ge=MIN_TEMPERATURE,
        le=MAX_TEMPERATURE,
        description="Sampling temperature (only used if do_sample=true). Lower = more deterministic."
    )
    max_new_tokens: Optional[int] = Field(
        DEFAULT_MAX_NEW_TOKENS,
        ge=MIN_MAX_NEW_TOKENS,
        le=MAX_MAX_NEW_TOKENS,
        description="Maximum tokens to generate (~50 tokens ≈ 1 second of audio). Default 2000 allows ~40s audio."
    )
    repetition_penalty: Optional[float] = Field(
        DEFAULT_REPETITION_PENALTY,
        ge=MIN_REPETITION_PENALTY,
        le=MAX_REPETITION_PENALTY,
        description="Repetition penalty to prevent token loops (1.0 = disabled, 1.05 = default). "
                    "Higher values discourage repeating tokens, helping prevent infinite generation."
    )





def validate_model_param(model: str) -> str:
    """
    Validate and return the model parameter.

    Args:
        model: Model identifier to validate

    Returns:
        Validated model ID

    Raises:
        HTTPException: 400 if model is invalid, 503 if model is not available
    """
    # Check if it's a valid model ID at all
    if model not in ALL_TTS_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{model}'. Must be one of: {', '.join(ALL_TTS_MODELS.keys())}"
        )

    # Check if model is available (cached locally)
    if model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model}' is not available. It may not be cached locally. "
                   f"Available models: {', '.join(AVAILABLE_MODELS.keys()) if AVAILABLE_MODELS else 'none'}"
        )

    return model


def validate_language_param(language: str) -> str:
    """
    Validate and return the language parameter.

    Args:
        language: Language to validate

    Returns:
        Validated language string

    Raises:
        HTTPException: 400 if language is invalid
    """
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid language '{language}'. Must be one of: {', '.join(SUPPORTED_LANGUAGES)}"
        )
    return language


class InferenceAbortedError(Exception):
    """Raised when inference is aborted by user request (skip_inference endpoint)."""
    pass


@dataclass
class InferenceResult:
    """Result from inference, may contain partial audio if truncated."""

    wavs: List[np.ndarray]
    sample_rate: int
    truncated: bool = False
    truncation_reason: Optional[str] = None


def apply_fade_out(audio: np.ndarray, sample_rate: int, fade_ms: int = 50) -> np.ndarray:
    """
    Apply a short fade-out to the end of audio to soften abrupt truncation.

    Args:
        audio: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        fade_ms: Fade duration in milliseconds (default 50ms)

    Returns:
        Audio with fade-out applied
    """
    fade_samples = int(sample_rate * fade_ms / 1000)
    if fade_samples >= len(audio):
        # Audio too short for fade, just return as-is
        return audio

    # Create fade curve (linear fade)
    fade_curve = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

    # Apply fade to end of audio
    audio = audio.copy()  # Don't modify original
    audio[-fade_samples:] = audio[-fade_samples:] * fade_curve

    return audio


def _generate_voice_clone_sync(
    model,
    texts: List[str],
    languages: List[str],
    prompt,
    do_sample: bool,
    temperature: float,
    max_new_tokens: int,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    model_id: str = "unknown",
) -> InferenceResult:
    """
    Synchronous wrapper for model.generate_voice_clone.

    Runs in thread pool executor for proper async integration.
    Includes timeout and VRAM limit protection via StoppingCriteria.

    Returns:
        InferenceResult with audio and truncation info.
        - For user abort (skip_inference): raises InferenceAbortedError
        - For timeout/VRAM limit: returns partial audio with truncated=True

    Note: The eos_token_id and repetition_penalty are also passed here as a
    belt-and-suspenders approach. The model's generation_config is pre-configured
    at load time (see _configure_generation_limits in manager.py), but we pass
    these parameters explicitly in case qwen-tts respects them in some code paths.
    """
    stopping_criteria = InferenceStoppingCriteria(model_id)

    with torch.inference_mode():
        wavs, sr = model.generate_voice_clone(
            text=texts,
            language=languages,
            voice_clone_prompt=prompt,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            # repetition_penalty helps prevent token loops (Issue #118)
            # eos_token_id handled internally by rekuenkdr fork (commit 3640831)
            repetition_penalty=repetition_penalty,
            stopping_criteria=StoppingCriteriaList([stopping_criteria]),
        )

        # Check if we stopped early
        if stopping_criteria.stopped_reason:
            # User requested abort via /skip_inference - raise error, no partial audio
            if "user requested abort" in stopping_criteria.stopped_reason:
                raise InferenceAbortedError(
                    f"Inference aborted: {stopping_criteria.stopped_reason}"
                )

            # Timeout or VRAM limit - return partial audio with fade-out
            logger.warning(
                f"[{model_id}] Returning partial audio due to: {stopping_criteria.stopped_reason}"
            )

            # Apply fade-out to soften the abrupt truncation
            faded_wavs = [apply_fade_out(wav, sr, fade_ms=50) for wav in wavs]

            return InferenceResult(
                wavs=faded_wavs,
                sample_rate=sr,
                truncated=True,
                truncation_reason=stopping_criteria.stopped_reason,
            )

        return InferenceResult(wavs=list(wavs), sample_rate=sr)


async def generate_speech_with_voice(
    text: Union[str, List[str]],
    voice: str,
    model_id: str,
    speed: float = 1.0,
    language: str = "Auto",
    do_sample: bool = False,
    temperature: float = 0.9,
    max_new_tokens: int = 2048,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    pad_end_ms: int = 300,
) -> Union[tuple[np.ndarray, int], tuple[List[np.ndarray], int]]:
    """
    Generate speech using a voice (cached or legacy).

    Supports both single text and batch processing. When text is a list,
    uses Qwen3-TTS native batching for parallel inference.

    Concurrency model:
    - Acquires inference semaphore (blocks other inference)
    - Runs GPU operation in executor with timeout
    - On timeout: client gets HTTP 504, but semaphore held until GPU completes
    - On completion: semaphore released, pending unload triggered if needed

    Args:
        text: Text to synthesize (single string or list for batch)
        voice: Voice name
        model_id: Model identifier
        speed: Speed multiplier
        language: Target language for synthesis
        do_sample: Whether to use sampling (True) or greedy decoding (False)
        temperature: Sampling temperature (only used if do_sample=True)
        max_new_tokens: Maximum number of tokens to generate
        repetition_penalty: Penalty for repeating tokens (1.0 = disabled, 1.05 = default)
        pad_end_ms: Milliseconds of silence to add at the end (prevents cutoff)

    Returns:
        For single text: Tuple of (audio_data, sample_rate)
        For batch: Tuple of (list of audio_data, sample_rate)
    """
    total_start = time.time()
    timings = {}

    is_batch = isinstance(text, list)
    texts = text if is_batch else [text]
    languages = [language] * len(texts)

    logger.info(
        f"generate_speech_with_voice called: voice={voice}, model_id={model_id}, "
        f"language={language}, batch_size={len(texts)}, do_sample={do_sample}, "
        f"temperature={temperature}, max_new_tokens={max_new_tokens}"
    )

    # Mark voice in use (prevents deletion during synthesis)
    voice_cache_manager.mark_in_use(voice)

    try:
        # Acquire inference slot (semaphore + model)
        t0 = time.time()
        model = await model_manager.acquire_inference_slot(model_id)
        timings['model_acquire'] = time.time() - t0

        # Get voice prompt and prepare inference
        # CRITICAL: This entire block is protected by try/except to prevent semaphore leaks
        # If ANY exception occurs after acquiring the semaphore but before the executor
        # takes ownership, we must release it here to avoid permanent deadlock.
        t0 = time.time()
        logger.info(f"Getting or creating voice clone prompt for {voice}/{model_id}")
        try:
            prompt = await voice_cache_manager.get_or_create_prompt(
                voice_name=voice,
                model_id=model_id,
                model=model,
            )
            timings['prompt_get'] = time.time() - t0
            logger.info(f"Got voice clone prompt in {timings['prompt_get']:.3f}s")

            # Set fixed seed for reproducible output
            torch.manual_seed(42)

            # Run inference in executor (StoppingCriteria handles timeout/VRAM/abort)
            loop = asyncio.get_event_loop()
            t0 = time.time()
            clear_abort_flag()  # Reset abort flag for new inference
            reset_truncation_info()  # Reset truncation for new request
            logger.info(f"Starting inference with batch_size={len(texts)}, max_timeout={settings_manager.get_inference_timeout()}s")
        except (Exception, asyncio.CancelledError) as e:
            logger.error(f"Error preparing inference: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            # Release semaphore on error (executor hasn't taken ownership yet)
            model_manager.release_inference(model_id)
            raise

        # Create executor task that releases semaphore when GPU is actually done
        inference_result = None
        inference_error = None

        def _run_inference_and_release():
            """Run inference and release semaphore when GPU completes."""
            nonlocal inference_result, inference_error
            try:
                inference_result = _generate_voice_clone_sync(
                    model, texts, languages, prompt,
                    do_sample, temperature, max_new_tokens, repetition_penalty, model_id
                )
            except Exception as e:
                inference_error = e
                # Log the error immediately - important if timeout occurs and
                # this error would otherwise be lost
                logger.error(f"Inference failed in executor thread: {e}\n{traceback.format_exc()}")
            finally:
                # Release happens here - when GPU is truly free
                # Use threadsafe version since we're in executor thread
                try:
                    model_manager.release_inference_threadsafe(model_id, loop)
                except Exception as release_error:
                    logger.error(f"Failed to release inference slot: {release_error}")

        # Run inference (StoppingCriteria handles timeout internally)
        await loop.run_in_executor(None, _run_inference_and_release)

        # Check for errors from executor
        if inference_error is not None:
            # Handle inference aborted (timeout/VRAM limit via StoppingCriteria)
            if isinstance(inference_error, InferenceAbortedError):
                raise HTTPException(
                    status_code=503,  # Service Unavailable
                    detail=str(inference_error)
                )
            # Handle CUDA OOM specifically
            if isinstance(inference_error, (torch.cuda.OutOfMemoryError, RuntimeError)):
                error_str = str(inference_error).lower()
                if "out of memory" in error_str or "cuda" in error_str:
                    raise HTTPException(
                        status_code=507,  # Insufficient Storage
                        detail=str(CUDAOutOfMemoryError(inference_error, "inference"))
                    )
            raise inference_error

        # Extract results from InferenceResult
        wavs = inference_result.wavs
        sr = inference_result.sample_rate
        truncated = inference_result.truncated
        truncation_reason = inference_result.truncation_reason

        timings['inference'] = time.time() - t0

        if truncated:
            logger.warning(
                f"Inference truncated in {timings['inference']:.3f}s due to: {truncation_reason}, "
                f"outputs={len(wavs)}, sr={sr}"
            )
        else:
            logger.info(f"Inference completed in {timings['inference']:.3f}s, outputs={len(wavs)}, sr={sr}")

        # Log token usage for diagnostic purposes
        for i, audio in enumerate(wavs):
            audio_duration = len(audio) / sr
            estimated_tokens = int(audio_duration * 50)
            usage_pct = (estimated_tokens / max_new_tokens) * 100

            if usage_pct >= 95:
                logger.warning(
                    f"Audio {i}: {audio_duration:.2f}s (~{estimated_tokens} tokens, "
                    f"{usage_pct:.1f}% of limit) - APPROACHING TOKEN LIMIT!"
                )
            else:
                logger.info(
                    f"Audio {i}: {audio_duration:.2f}s (~{estimated_tokens} tokens, "
                    f"{usage_pct:.1f}% of {max_new_tokens} limit)"
                )

        # Post-processing
        pad_samples = int(sr * pad_end_ms / 1000)
        padding = np.zeros(pad_samples, dtype=np.float32)

        t0 = time.time()
        processed_audios = []
        for i, audio_data in enumerate(wavs):
            audio_data = np.concatenate([audio_data, padding])
            if speed != 1.0:
                audio_data = apply_speed(audio_data, sr, speed)
            processed_audios.append(audio_data)

            # Trigger garbage collection after each batch item to free intermediate arrays
            gc.collect()
        timings['post_process'] = time.time() - t0

        total_time = time.time() - total_start
        total_audio_duration = sum(len(a) / sr for a in processed_audios)

        logger.info(
            f"Generation completed in {total_time:.2f}s "
            f"(audio: {total_audio_duration:.2f}s, RTF: {total_time/total_audio_duration:.2f}x) | "
            f"TIMING: acquire={timings['model_acquire']:.3f}s, "
            f"prompt={timings['prompt_get']:.3f}s, "
            f"inference={timings['inference']:.3f}s, "
            f"post={timings['post_process']:.3f}s"
        )

        # Store truncation info in context-local for endpoint to read
        set_truncation_info(truncated, truncation_reason)

        if not is_batch:
            return processed_audios[0], sr
        return processed_audios, sr

    finally:
        voice_cache_manager.mark_not_in_use(voice)


async def ensure_voice_available(voice: str) -> None:
    """
    Ensure a voice is available in the cache.

    Raises:
        HTTPException: 404 if voice is not found
    """
    if not voice_cache_manager.is_cached_voice(voice):
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{voice}' not found. Please create it first using /voices/create endpoint."
        )


@router.post(
    "/synthesize_speech/",
    summary="Synthesize speech from text",
    description="Generate speech from text using a specified voice and model. "
                "Accepts either a single string or an array of strings for batch processing. "
                "Single text returns WAV audio, batch returns ZIP archive with numbered WAV files.",
    responses={
        200: {
            "content": {
                "audio/wav": {"schema": {"type": "string", "format": "binary"}},
                "application/zip": {"schema": {"type": "string", "format": "binary"}}
            },
            "description": "Generated audio file (WAV for single, ZIP for batch)"
        },
        400: {"description": "Invalid request parameters"},
        404: {"description": "Voice not found"},
        503: {"description": "Model not available"}
    }
)
async def synthesize_speech(request: SynthesizeRequest):
    """Synthesize speech from text using a specified voice."""
    start_time = time.time()

    model_id = validate_model_param(request.model)
    language = validate_language_param(request.language)

    # Normalize text to list for unified processing
    texts = request.text if isinstance(request.text, list) else [request.text]
    is_batch = isinstance(request.text, list)

    if not texts:
        raise HTTPException(status_code=400, detail="Text array cannot be empty")

    # Validate batch size limit
    if len(texts) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(texts)} exceeds maximum allowed ({MAX_BATCH_SIZE}). "
                   f"Please split into smaller batches."
        )

    # Filter out empty texts and track their indices
    valid_texts = []
    valid_indices = []
    for i, text in enumerate(texts):
        if text and text.strip():
            valid_texts.append(text)
            valid_indices.append(i)
        else:
            logger.warning(f"Skipping empty text at index {i}")

    if not valid_texts:
        raise HTTPException(status_code=400, detail="No valid text to synthesize")

    try:
        # Ensure voice is available in cache
        await ensure_voice_available(request.voice)

        # Apply cutoff fix: ensure texts end with punctuation to improve EOS token generation
        # This addresses upstream bug: https://github.com/QwenLM/Qwen3-TTS/issues/55
        valid_texts = [ensure_ending_punctuation(text, language) for text in valid_texts]
        logger.info(f"Applied ending punctuation fix to {len(valid_texts)} texts for language={language}")

        logger.info(f"Processing batch of {len(valid_texts)} texts using native batching")

        # Generate speech with native batch processing
        result = await generate_speech_with_voice(
            text=valid_texts if len(valid_texts) > 1 else valid_texts[0],
            voice=request.voice,
            model_id=model_id,
            speed=request.speed,
            language=language,
            do_sample=request.do_sample,
            temperature=request.temperature,
            max_new_tokens=request.max_new_tokens,
            repetition_penalty=request.repetition_penalty,
        )

        # Unpack result - handle both single and batch returns
        if len(valid_texts) == 1:
            audio_data, sr = result
            audio_results = [(audio_data, sr, valid_indices[0])]
        else:
            audio_list, sr = result
            audio_results = [(audio_list[i], sr, valid_indices[i]) for i in range(len(audio_list))]

        elapsed_time = time.time() - start_time

        # Single text: return WAV (backward compatible)
        if not is_batch:
            audio_data, sr, _ = audio_results[0]
            audio_buffer = audio_to_wav_bytes(audio_data, sr)
            audio_content = audio_buffer.getvalue()
            audio_buffer.close()  # Explicitly release buffer memory

            response_headers = {
                "X-Elapsed-Time": str(elapsed_time),
                "X-Device-Used": model_manager.device,
                "X-Model-Used": model_id,
                "Content-Disposition": "attachment; filename=synthesized_speech.wav",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Headers": (
                    "Origin, Content-Type, X-Amz-Date, Authorization, "
                    "X-Api-Key, X-Amz-Security-Token, locale"
                ),
                "Access-Control-Allow-Methods": "POST, OPTIONS",
            }

            # Add truncation header if audio was cut short
            truncation_info = get_truncation_info()
            if truncation_info.truncated:
                response_headers["X-Audio-Truncated"] = "true"
                response_headers["X-Truncation-Reason"] = truncation_info.reason or "unknown"

            return Response(
                content=audio_content,
                media_type="audio/wav",
                headers=response_headers
            )

        # Batch: return ZIP archive (no compression for speed)
        zip_buffer = io.BytesIO()
        try:
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_STORED) as zf:
                for audio_data, sr, idx in audio_results:
                    wav_buffer = audio_to_wav_bytes(audio_data, sr)
                    try:
                        filename = f"{idx:04d}.wav"
                        zf.writestr(filename, wav_buffer.getvalue())
                    finally:
                        # Explicitly close WAV buffer to free memory immediately
                        wav_buffer.close()

            zip_buffer.seek(0)
            zip_content = zip_buffer.getvalue()
        finally:
            # Close after getting content (Response will use the bytes copy)
            zip_buffer.close()

        # Trigger garbage collection to free ZIP buffer and temporary audio buffers
        gc.collect()

        logger.info(f"Batch synthesis completed: {len(audio_results)} files in {elapsed_time:.2f}s")

        response_headers = {
            "X-Elapsed-Time": str(elapsed_time),
            "X-Device-Used": model_manager.device,
            "X-Model-Used": model_id,
            "X-Batch-Count": str(len(audio_results)),
            "Content-Disposition": "attachment; filename=synthesized_batch.zip",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Headers": (
                "Origin, Content-Type, X-Amz-Date, Authorization, "
                "X-Api-Key, X-Amz-Security-Token, locale"
            ),
            "Access-Control-Allow-Methods": "POST, OPTIONS",
        }

        # Add truncation header if audio was cut short
        truncation_info = get_truncation_info()
        if truncation_info.truncated:
            response_headers["X-Audio-Truncated"] = "true"
            response_headers["X-Truncation-Reason"] = truncation_info.reason or "unknown"

        return Response(
            content=zip_content,
            media_type="application/zip",
            headers=response_headers
        )

    except HTTPException:
        raise
    except RuntimeError as e:
        # Model not available error
        logger.error(f"Model error in synthesize_speech: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=503, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError in synthesize_speech: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in synthesize_speech: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))




@router.post(
    "/change_voice/",
    summary="Change voice of audio",
    description="Convert text to speech using a different voice. "
                "Provide the transcription text and target voice to generate audio.",
    responses={
        200: {
            "content": {"audio/wav": {}},
            "description": "Converted audio file"
        },
        400: {"description": "Invalid request parameters"},
        404: {"description": "Voice not found"},
        503: {"description": "Model not available"}
    }
)
async def change_voice(
    reference_speaker: str = Form(..., description="Target voice name"),
    transcription: str = Form(..., description="Text to synthesize with the new voice"),
    model: Literal["0.6B", "1.7B"] = Form(DEFAULT_MODEL, description="Model to use"),
    language: Literal[
        "Auto", "Chinese", "English", "Japanese", "Korean",
        "German", "French", "Russian", "Portuguese", "Spanish", "Italian"
    ] = Form(DEFAULT_LANGUAGE, description="Target language for synthesis"),
):
    """Change the voice by synthesizing text with a different voice."""
    model_id = validate_model_param(model)
    language = validate_language_param(language)

    if not transcription or not transcription.strip():
        raise HTTPException(status_code=400, detail="Transcription text is required")

    try:
        logger.info(f'Synthesizing with voice {reference_speaker}...')

        # Ensure target voice is available
        await ensure_voice_available(reference_speaker)

        # Generate with the new voice
        audio_data, sr = await generate_speech_with_voice(
            text=transcription.strip(),
            voice=reference_speaker,
            model_id=model_id,
            language=language,
        )

        # Convert to WAV bytes (no file I/O needed)
        audio_buffer = audio_to_wav_bytes(audio_data, sr)
        audio_content = audio_buffer.getvalue()

        return Response(
            content=audio_content,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=converted_voice.wav"}
        )

    except HTTPException:
        raise
    except RuntimeError as e:
        # Model not available error
        logger.error(f"Model error in change_voice: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Error in change_voice: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/skip_inference")
async def skip_inference():
    """
    Request to skip/abort the current inference.

    This sets a flag that the StoppingCriteria checks between token generation steps.
    If inference is progressing (not completely hung), it will stop at the next check.

    NOTE: This won't help if the model is completely frozen/hung - only if it's
    still generating tokens but taking too long.

    Returns:
        Message confirming abort was requested
    """
    request_abort_inference()
    logger.warning("Inference abort requested via /skip_inference endpoint")
    return {"message": "Abort requested. Inference will stop at next token if still running."}
