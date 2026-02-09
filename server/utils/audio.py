"""Audio processing utilities."""

import io
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment, silence

from server.config import MAX_REF_AUDIO_DURATION_MS, OUTPUT_DIR, TARGET_SAMPLE_RATE

logger = logging.getLogger(__name__)

# Check if CUDA is available for GPU-accelerated audio processing
_CUDA_AVAILABLE = torch.cuda.is_available()
if _CUDA_AVAILABLE:
    logger.info("CUDA available for GPU-accelerated audio processing")
else:
    logger.info("CUDA not available, falling back to CPU audio processing")

# Cache for torchaudio Resample transforms to avoid recreating them
# Key: (orig_freq, new_freq, device) -> Resample transform
# Uses OrderedDict for LRU eviction - max 16 entries
_RESAMPLER_CACHE: OrderedDict[tuple[int, int, str], torchaudio.transforms.Resample] = OrderedDict()
_RESAMPLER_CACHE_MAX_ENTRIES = 16


def _get_resampler(orig_freq: int, new_freq: int, device: str) -> torchaudio.transforms.Resample:
    """
    Get a cached resampler or create a new one with LRU eviction.

    Caching resamplers avoids the overhead of creating new transform objects
    for each speed adjustment call (~50-100ms savings per call).

    Args:
        orig_freq: Original sample rate
        new_freq: Target sample rate
        device: Device string (e.g., "cuda:0" or "cpu")

    Returns:
        Cached or newly created Resample transform
    """
    key = (orig_freq, new_freq, device)
    if key in _RESAMPLER_CACHE:
        # Move to end for LRU tracking
        _RESAMPLER_CACHE.move_to_end(key)
        return _RESAMPLER_CACHE[key]

    # Evict oldest entries if cache is full
    while len(_RESAMPLER_CACHE) >= _RESAMPLER_CACHE_MAX_ENTRIES:
        oldest_key, _ = _RESAMPLER_CACHE.popitem(last=False)
        logger.debug(f"Evicted resampler {oldest_key[0]}Hz -> {oldest_key[1]}Hz from cache (LRU)")

    _RESAMPLER_CACHE[key] = torchaudio.transforms.Resample(
        orig_freq=orig_freq,
        new_freq=new_freq,
    ).to(device)
    logger.debug(f"Created new resampler: {orig_freq}Hz -> {new_freq}Hz on {device}")
    return _RESAMPLER_CACHE[key]


def clear_resampler_cache() -> int:
    """
    Clear the resampler cache to free GPU memory.

    Call this when unloading models to ensure GPU memory is fully released.
    Each cached resampler holds GPU tensors that won't be freed by
    torch.cuda.empty_cache() alone.

    Returns:
        Number of resamplers cleared from cache.
    """
    count = len(_RESAMPLER_CACHE)
    _RESAMPLER_CACHE.clear()
    if count > 0:
        logger.debug(f"Cleared {count} cached resampler(s)")
    return count


def convert_to_wav(input_path: str, output_path: str) -> None:
    """
    Convert any audio format to WAV using pydub.

    Args:
        input_path: Path to input audio file
        output_path: Path for output WAV file
    """
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)  # Convert to mono
    audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)  # Set sample rate
    audio.export(output_path, format='wav')


def detect_leading_silence(
    audio: AudioSegment,
    silence_threshold: int = -42,
    chunk_size: int = 10
) -> int:
    """
    Detect silence at the beginning of the audio.

    Args:
        audio: AudioSegment to analyze
        silence_threshold: dBFS threshold for silence
        chunk_size: Chunk size in milliseconds

    Returns:
        Duration of leading silence in milliseconds
    """
    trim_ms = 0
    while (
        audio[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold
        and trim_ms < len(audio)
    ):
        trim_ms += chunk_size
    return trim_ms


def remove_silence_edges(
    audio: AudioSegment,
    silence_threshold: int = -42
) -> AudioSegment:
    """
    Remove silence from the beginning and end of the audio.

    Args:
        audio: AudioSegment to process
        silence_threshold: dBFS threshold for silence

    Returns:
        Trimmed AudioSegment
    """
    start_trim = detect_leading_silence(audio, silence_threshold)
    end_trim = detect_leading_silence(audio.reverse(), silence_threshold)
    duration = len(audio)
    return audio[start_trim:duration - end_trim]


def process_reference_audio(
    reference_file: str,
    transcription: Optional[str] = None,
    skip_transcription: bool = False,
) -> Tuple[np.ndarray, int, str]:
    """
    Process reference audio: remove silence edges and optionally transcribe.

    If MAX_REF_AUDIO_DURATION_MS is set in config, audio will be clipped to that length.
    If MAX_REF_AUDIO_DURATION_MS is None, no clipping is performed.

    Args:
        reference_file: Path to reference audio file
        transcription: Optional pre-provided transcription
        skip_transcription: If True, skip transcription entirely (for x_vector_only_mode)

    Returns:
        Tuple of (audio_data, sample_rate, transcription)
        If skip_transcription=True, transcription will be empty string
    """
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    temp_short_ref = OUTPUT_DIR / 'temp_short_ref.wav'

    try:
        aseg = AudioSegment.from_file(reference_file)

        # Only apply clipping if MAX_REF_AUDIO_DURATION_MS is set
        if MAX_REF_AUDIO_DURATION_MS is not None:
            # Try different silence detection thresholds (coarse to fine)
            # Each config: (min_silence_len, silence_thresh, description)
            silence_configs = [
                (1000, -50, "long silence"),   # Try long silences first
                (100, -40, "short silence"),   # Fall back to shorter silences
            ]

            non_silent_wave = AudioSegment.silent(duration=0)
            for min_silence_len, silence_thresh, desc in silence_configs:
                non_silent_segs = silence.split_on_silence(
                    aseg,
                    min_silence_len=min_silence_len,
                    silence_thresh=silence_thresh,
                    keep_silence=1000,
                    seek_step=10
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    if (
                        len(non_silent_wave) > 6000
                        and len(non_silent_wave + non_silent_seg) > MAX_REF_AUDIO_DURATION_MS
                    ):
                        logger.info(f"Audio is over {MAX_REF_AUDIO_DURATION_MS/1000:.0f}s, clipping at {desc}.")
                        break
                    non_silent_wave += non_silent_seg

                # If we got a valid result, stop trying other configs
                if len(non_silent_wave) <= MAX_REF_AUDIO_DURATION_MS:
                    break

            aseg = non_silent_wave

            # If no proper silence found for clipping, hard clip
            if len(aseg) > MAX_REF_AUDIO_DURATION_MS:
                aseg = aseg[:MAX_REF_AUDIO_DURATION_MS]
                logger.info(f"Audio is over {MAX_REF_AUDIO_DURATION_MS/1000:.0f}s, hard clipping.")

        # Remove silence from edges and add small padding
        aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
        aseg.export(str(temp_short_ref), format='wav')

        # Handle transcription based on skip_transcription flag
        if skip_transcription:
            transcription = ""
            logger.info('Transcription skipped (x_vector_only_mode will be used)')
        elif transcription is None or transcription.strip() == "":
            # No transcription provided and auto-transcription not available
            transcription = ""
            logger.warning('No transcription provided - voice cloning quality may be reduced')
        else:
            logger.info(f'Using provided transcription: {transcription}')

        # Read processed audio as numpy array
        audio_data, sr = sf.read(str(temp_short_ref))

        return audio_data, sr, transcription

    finally:
        # Clean up temporary file
        if temp_short_ref.exists():
            try:
                temp_short_ref.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_short_ref}: {e}")


def apply_speed_gpu(audio_data: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """
    Apply speed adjustment to audio using GPU-accelerated torchaudio.

    Uses resampling for speed change, which is very fast on GPU but also
    changes the pitch. For pitch-preserving time stretch, use apply_speed_librosa.

    Args:
        audio_data: Audio data as numpy array
        sr: Sample rate
        speed: Speed multiplier (>1 = faster, <1 = slower)

    Returns:
        Speed-adjusted audio data
    """
    if speed == 1.0:
        return audio_data

    device = "cuda:0" if _CUDA_AVAILABLE else "cpu"

    # Convert to tensor and move to device
    audio_tensor = torch.from_numpy(audio_data).float().to(device)

    # Use resampling to change speed
    # To speed up by factor N, we resample to sr*N then interpret at sr
    # This is equivalent to playing at speed N (changes pitch)
    # Round to nearest 100Hz to reduce cache key fragmentation from variable speeds
    target_sr = round(sr * speed / 100) * 100
    if target_sr < 100:
        target_sr = 100  # Minimum safe value

    # Get cached resampler (avoids recreating transform each call)
    resampler = _get_resampler(sr, target_sr, device)

    # Add channel dimension if needed (resampler expects [batch, time] or [time])
    if audio_tensor.dim() == 1:
        resampled = resampler(audio_tensor.unsqueeze(0)).squeeze(0)
    else:
        resampled = resampler(audio_tensor)

    return resampled.cpu().numpy()


def apply_speed_librosa(audio_data: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """
    Apply speed adjustment to audio using librosa time stretching (CPU-based).

    This preserves pitch but is slower than GPU resampling.

    Args:
        audio_data: Audio data as numpy array
        sr: Sample rate
        speed: Speed multiplier (>1 = faster, <1 = slower)

    Returns:
        Speed-adjusted audio data
    """
    if speed == 1.0:
        return audio_data

    try:
        import librosa
        # Time stretch: speed > 1 = faster, speed < 1 = slower
        return librosa.effects.time_stretch(audio_data, rate=speed)
    except ImportError:
        logger.warning("librosa not installed, speed adjustment not available")
        return audio_data


def apply_speed(audio_data: np.ndarray, sr: int, speed: float, preserve_pitch: bool = False) -> np.ndarray:
    """
    Apply speed adjustment to audio.

    By default uses GPU-accelerated resampling for best performance.
    Set preserve_pitch=True to use librosa time stretching (slower but pitch-preserving).

    Args:
        audio_data: Audio data as numpy array
        sr: Sample rate
        speed: Speed multiplier (>1 = faster, <1 = slower)
        preserve_pitch: If True, use librosa to preserve pitch (slower)

    Returns:
        Speed-adjusted audio data
    """
    if speed == 1.0:
        return audio_data

    if preserve_pitch:
        return apply_speed_librosa(audio_data, sr, speed)
    else:
        return apply_speed_gpu(audio_data, sr, speed)


def audio_to_wav_bytes(audio_data: np.ndarray, sr: int) -> io.BytesIO:
    """
    Convert numpy audio array to WAV bytes with pre-allocated buffer.

    Args:
        audio_data: Audio data as numpy array
        sr: Sample rate

    Returns:
        BytesIO buffer containing WAV data
    """
    # Pre-allocate buffer with estimated size
    # WAV header is 44 bytes + audio data (converted to int16 = 2 bytes per sample)
    estimated_size = 44 + len(audio_data) * 2
    buffer = io.BytesIO(bytearray(estimated_size))
    buffer.seek(0)

    # Write WAV with PCM_16 subtype for better compatibility and smaller size
    sf.write(buffer, audio_data, sr, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    return buffer


def validate_audio_file(content: bytes, filename: str) -> Tuple[bool, str]:
    """
    Validate an uploaded audio file.

    Args:
        content: File content as bytes
        filename: Original filename

    Returns:
        Tuple of (is_valid, error_message)
    """
    import magic
    from server.config import ALLOWED_AUDIO_EXTENSIONS, MAX_UPLOAD_SIZE

    # Check file size
    if len(content) > MAX_UPLOAD_SIZE:
        return False, f"File size exceeds limit of {MAX_UPLOAD_SIZE // (1024*1024)}MB"

    # Check extension
    file_ext = filename.split('.')[-1].lower() if '.' in filename else ''
    if file_ext not in ALLOWED_AUDIO_EXTENSIONS:
        return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"

    # Check actual content type
    try:
        file_format = magic.from_buffer(content, mime=True)
        if 'audio' not in file_format:
            return False, "Invalid file content - not a valid audio file"
    except Exception as e:
        logger.warning(f"Could not verify file type with magic: {e}")
        # Fail closed: if we can't verify the file type, reject it for safety
        return False, "Could not verify file type - please ensure this is a valid audio file"

    return True, ""
