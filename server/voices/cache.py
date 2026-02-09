"""Voice cache manager for persistent voice storage."""

import gc
import json
import logging
import shutil
import time
import traceback
from datetime import datetime, timezone
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf
import torch

from server.config import AVAILABLE_MODELS, VOICES_DIR

logger = logging.getLogger(__name__)

# Maximum number of voices to keep in prompt cache (LRU eviction)
# Each prompt can be 50-200MB in RAM (pinned memory for GPU transfer)
_PROMPT_CACHE_MAX_VOICES = 10

# Maximum number of entries in voice info cache (LRU eviction)
# Each entry is ~1-5KB (metadata only)
_VOICE_INFO_CACHE_MAX_ENTRIES = 100


class VoiceCacheManager:
    """
    Manages persistent voice cloning cache.

    Voice data is stored in /data/voices/{voice_name}/:
    - metadata.json: Voice metadata (transcription, created_at, etc.)
    - reference.wav: Processed reference audio
    - prompt_{model_id}.pkl: Cached voice clone prompt for each model
    """

    _instance: Optional["VoiceCacheManager"] = None
    _lock = Lock()

    def __new__(cls) -> "VoiceCacheManager":
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the voice cache manager."""
        if self._initialized:
            return

        # In-memory prompt cache: {voice_name: {model_id: prompt}}
        self._prompt_cache: Dict[str, Dict[str, Any]] = {}

        # LRU access time tracking for prompt cache: {voice_name: timestamp}
        self._prompt_cache_access: Dict[str, float] = {}

        # In-memory voice info cache: OrderedDict for LRU eviction
        # {voice_name: (mtime, voice_info)} - Uses file mtime to invalidate cache when metadata changes
        self._voice_info_cache: OrderedDict[str, Tuple[float, Dict[str, Any]]] = OrderedDict()

        # Voices currently in use (for safe deletion)
        self._in_use: Set[str] = set()

        # Lock for thread safety
        self._cache_lock = Lock()

        self._initialized = True

        # Ensure voice directory exists
        try:
            VOICES_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Cannot create voices directory: {e}")

        logger.info("VoiceCacheManager initialized")

    def _get_voice_dir(self, voice_name: str) -> Path:
        """Get the directory path for a voice."""
        return VOICES_DIR / voice_name

    def _get_metadata_path(self, voice_name: str) -> Path:
        """Get the metadata file path for a voice."""
        return self._get_voice_dir(voice_name) / "metadata.json"

    def _get_reference_path(self, voice_name: str) -> Path:
        """Get the reference audio path for a voice."""
        return self._get_voice_dir(voice_name) / "reference.wav"

    def _get_prompt_path(self, voice_name: str, model_id: str) -> Path:
        """Get the prompt cache path for a voice and model."""
        return self._get_voice_dir(voice_name) / f"prompt_{model_id}.pkl"

    def _evict_oldest_prompt_if_needed(self) -> None:
        """
        Evict least recently used voice prompt when cache is full.

        Must be called with _cache_lock held.
        """
        if len(self._prompt_cache) >= _PROMPT_CACHE_MAX_VOICES:
            if not self._prompt_cache_access:
                # No access tracking, evict first entry
                oldest = next(iter(self._prompt_cache))
            else:
                # Find voice with oldest access time
                oldest = min(
                    self._prompt_cache.keys(),
                    key=lambda v: self._prompt_cache_access.get(v, 0)
                )

            # Evict the oldest voice
            del self._prompt_cache[oldest]
            self._prompt_cache_access.pop(oldest, None)
            logger.info(f"Evicted voice '{oldest}' from prompt cache (LRU, max={_PROMPT_CACHE_MAX_VOICES})")

    def voice_exists(self, voice_name: str) -> bool:
        """
        Check if a voice exists in the cache.
        """
        return self._get_metadata_path(voice_name).exists()

    def is_cached_voice(self, voice_name: str) -> bool:
        """Check if a voice is in the persistent cache (not legacy)."""
        return self._get_metadata_path(voice_name).exists()

    def list_voices(self) -> List[Dict[str, Any]]:
        """
        List all cached voices.

        Returns:
            List of voice metadata dictionaries
        """
        voices = []

        # List voices from persistent cache
        if VOICES_DIR.exists():
            for voice_dir in VOICES_DIR.iterdir():
                if voice_dir.is_dir():
                    metadata_path = voice_dir / "metadata.json"
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, "r", encoding="utf-8") as f:
                                metadata = json.load(f)
                            voices.append(metadata)
                        except Exception as e:
                            logger.warning(
                                f"Error reading metadata for {voice_dir.name}: {e}"
                            )

        return voices

    def get_voice_info(self, voice_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific voice.

        Uses in-memory cache with mtime-based invalidation to avoid
        repeated disk I/O (saves ~3-5ms per request).

        Args:
            voice_name: Name of the voice

        Returns:
            Voice metadata dictionary or None if not found
        """
        metadata_path = self._get_metadata_path(voice_name)

        if not metadata_path.exists():
            # Clear from cache if file was deleted
            with self._cache_lock:
                self._voice_info_cache.pop(voice_name, None)
            return None

        try:
            current_mtime = metadata_path.stat().st_mtime

            # Check cache with mtime validation
            with self._cache_lock:
                if voice_name in self._voice_info_cache:
                    cached_mtime, cached_info = self._voice_info_cache[voice_name]
                    if cached_mtime == current_mtime:
                        # Move to end for LRU tracking
                        self._voice_info_cache.move_to_end(voice_name)
                        return cached_info

            # Cache miss or stale - read from disk
            with open(metadata_path, "r", encoding="utf-8") as f:
                voice_info = json.load(f)

            # Update cache with LRU eviction
            with self._cache_lock:
                # Evict oldest entries if cache is full
                while len(self._voice_info_cache) >= _VOICE_INFO_CACHE_MAX_ENTRIES:
                    oldest = next(iter(self._voice_info_cache))
                    del self._voice_info_cache[oldest]
                    logger.debug(f"Evicted voice info '{oldest}' from cache (LRU)")

                self._voice_info_cache[voice_name] = (current_mtime, voice_info)

            return voice_info

        except Exception as e:
            logger.error(f"Error reading voice metadata: {e}")
            return None

    def get_cached_models(self, voice_name: str) -> List[str]:
        """
        Get list of model IDs that have cached prompts for a voice.

        Args:
            voice_name: Name of the voice

        Returns:
            List of model IDs with cached prompts
        """
        cached = []
        voice_dir = self._get_voice_dir(voice_name)

        if voice_dir.exists():
            for model_id in AVAILABLE_MODELS:
                prompt_path = self._get_prompt_path(voice_name, model_id)
                if prompt_path.exists():
                    cached.append(model_id)

        return cached

    async def create_voice(
        self,
        voice_name: str,
        audio_data: np.ndarray,
        sample_rate: int,
        transcription: str,
        transcription_type: str = "MANUAL",
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a new cached voice.

        Args:
            voice_name: Name for the voice
            audio_data: Processed reference audio as numpy array
            sample_rate: Audio sample rate
            transcription: Transcription of the reference audio
            transcription_type: Type of transcription (MANUAL or NONE)
            overwrite: Whether to overwrite existing voice

        Returns:
            Voice metadata dictionary

        Raises:
            FileExistsError: If voice exists and overwrite=False
        """
        voice_dir = self._get_voice_dir(voice_name)

        # Check if voice exists
        if voice_dir.exists() and not overwrite:
            raise FileExistsError(
                f"Voice '{voice_name}' already exists. Use overwrite=true to replace."
            )

        # Create voice directory
        voice_dir.mkdir(parents=True, exist_ok=True)

        # Save reference audio
        reference_path = self._get_reference_path(voice_name)
        sf.write(reference_path, audio_data, sample_rate)

        # Create metadata
        metadata = {
            "name": voice_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "transcription": transcription,
            "transcription_type": transcription_type,
        }

        # Save metadata
        metadata_path = self._get_metadata_path(voice_name)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Clear any cached data (prompts need to be regenerated, info changed)
        with self._cache_lock:
            self._prompt_cache.pop(voice_name, None)
            self._prompt_cache_access.pop(voice_name, None)
            self._voice_info_cache.pop(voice_name, None)

        # Delete any existing prompt files
        for model_id in AVAILABLE_MODELS:
            prompt_path = self._get_prompt_path(voice_name, model_id)
            if prompt_path.exists():
                prompt_path.unlink()

        logger.info(f"Created voice cache for: {voice_name}")
        return metadata

    def delete_voice(self, voice_name: str) -> bool:
        """
        Delete a cached voice.

        Args:
            voice_name: Name of the voice to delete

        Returns:
            True if deleted, False if not found

        Raises:
            RuntimeError: If voice is currently in use
        """
        # Check if voice is in use (thread-safe)
        with self._cache_lock:
            if voice_name in self._in_use:
                raise RuntimeError(f"Voice '{voice_name}' is currently in use")

        voice_dir = self._get_voice_dir(voice_name)

        if not voice_dir.exists():
            return False

        # Clear from memory caches
        with self._cache_lock:
            self._prompt_cache.pop(voice_name, None)
            self._prompt_cache_access.pop(voice_name, None)
            self._voice_info_cache.pop(voice_name, None)

        # Delete directory
        try:
            shutil.rmtree(voice_dir)
            logger.info(f"Deleted voice: {voice_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting voice {voice_name}: {e}")
            raise

    def rename_voice(self, old_name: str, new_name: str) -> Dict[str, Any]:
        """
        Rename a cached voice.

        Args:
            old_name: Current name of the voice
            new_name: New name for the voice

        Returns:
            Updated voice metadata dictionary

        Raises:
            FileNotFoundError: If voice doesn't exist
            FileExistsError: If new_name already exists
            RuntimeError: If voice is currently in use
            ValueError: If new_name is invalid
        """
        import re

        # Validate new_name format
        if not new_name or not new_name.strip():
            raise ValueError("New voice name cannot be empty")

        if not re.match(r'^[\w-]+$', new_name, re.UNICODE):
            raise ValueError(
                "Voice name can only contain letters, numbers, underscores, and hyphens"
            )

        # Check if voice is in use (early check with lock)
        with self._cache_lock:
            if old_name in self._in_use:
                raise RuntimeError(f"Voice '{old_name}' is currently in use")

        old_dir = self._get_voice_dir(old_name)
        new_dir = self._get_voice_dir(new_name)

        # Check if old voice exists
        if not old_dir.exists():
            raise FileNotFoundError(f"Voice '{old_name}' not found")

        # Check if new name already exists
        if new_dir.exists():
            raise FileExistsError(f"Voice '{new_name}' already exists")

        # Acquire lock for thread-safe cache updates
        with self._cache_lock:
            # Double-check in_use after acquiring lock
            if old_name in self._in_use:
                raise RuntimeError(f"Voice '{old_name}' is currently in use")

            try:
                # Rename directory on disk
                shutil.move(str(old_dir), str(new_dir))

                # Update metadata.json with new name
                metadata_path = self._get_metadata_path(new_name)
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                metadata["name"] = new_name

                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

                # Update in-memory caches
                # Transfer prompt cache entries
                if old_name in self._prompt_cache:
                    self._prompt_cache[new_name] = self._prompt_cache.pop(old_name)
                if old_name in self._prompt_cache_access:
                    self._prompt_cache_access[new_name] = self._prompt_cache_access.pop(old_name)

                # Transfer voice info cache entries
                if old_name in self._voice_info_cache:
                    mtime, info = self._voice_info_cache.pop(old_name)
                    info["name"] = new_name
                    self._voice_info_cache[new_name] = (mtime, info)

                logger.info(f"Renamed voice: {old_name} -> {new_name}")
                return metadata

            except Exception as e:
                logger.error(f"Error renaming voice {old_name} to {new_name}: {e}")
                # Attempt to rollback if directory was moved
                if new_dir.exists() and not old_dir.exists():
                    try:
                        shutil.move(str(new_dir), str(old_dir))
                        logger.info(f"Rolled back rename: {new_name} -> {old_name}")
                    except Exception as rollback_err:
                        logger.error(f"Failed to rollback rename: {rollback_err}")
                raise

    def get_reference_audio(
        self, voice_name: str
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Get reference audio for a voice.

        Args:
            voice_name: Name of the voice

        Returns:
            Tuple of (audio_data, sample_rate) or None if not found
        """
        reference_path = self._get_reference_path(voice_name)
        if reference_path.exists():
            try:
                audio_data, sr = sf.read(reference_path)
                return audio_data, sr
            except Exception as e:
                logger.error(f"Error reading reference audio: {e}")

        return None

    def get_transcription(self, voice_name: str) -> Optional[str]:
        """
        Get transcription for a voice.

        Args:
            voice_name: Name of the voice

        Returns:
            Transcription string or None if not found
        """
        info = self.get_voice_info(voice_name)
        if info:
            return info.get("transcription")
        return None

    async def get_or_create_prompt(
        self,
        voice_name: str,
        model_id: str,
        model: Any,
    ) -> Any:
        """
        Get cached voice clone prompt, creating if necessary.

        Args:
            voice_name: Name of the voice
            model_id: Model identifier
            model: The loaded model instance

        Returns:
            Voice clone prompt object

        Raises:
            FileNotFoundError: If voice doesn't exist
        """
        import asyncio

        # Check memory cache first
        with self._cache_lock:
            if voice_name in self._prompt_cache:
                if model_id in self._prompt_cache[voice_name]:
                    # Update access time for LRU tracking
                    self._prompt_cache_access[voice_name] = time.time()
                    logger.debug(f"Using memory-cached prompt for {voice_name}/{model_id}")
                    return self._prompt_cache[voice_name][model_id]

        # Check disk cache
        prompt_path = self._get_prompt_path(voice_name, model_id)
        if prompt_path.exists():
            try:
                # Use torch.load with map_location to handle GPU tensor context properly
                # This prevents CUDA context corruption when loading prompts saved
                # in a different session or after server restart
                device = self._get_device()
                prompt = torch.load(prompt_path, map_location=device, weights_only=False)

                # Ensure all tensors are on the correct device
                prompt = self._move_prompt_to_device(prompt, device)

                # Store in memory cache with LRU eviction
                with self._cache_lock:
                    if voice_name not in self._prompt_cache:
                        self._evict_oldest_prompt_if_needed()
                        self._prompt_cache[voice_name] = {}
                    self._prompt_cache[voice_name][model_id] = prompt
                    self._prompt_cache_access[voice_name] = time.time()

                # Trigger garbage collection to free old tensor copies and serialization intermediates
                gc.collect()

                logger.debug(f"Loaded prompt from disk for {voice_name}/{model_id}")
                return prompt
            except Exception as e:
                logger.warning(f"Error loading cached prompt, will recreate: {e}")
                # Delete corrupted cache file to prevent future errors
                try:
                    prompt_path.unlink(missing_ok=True)
                    logger.info(f"Deleted corrupted prompt cache: {prompt_path}")
                except Exception as del_err:
                    logger.warning(f"Failed to delete corrupted cache: {del_err}")

        # Need to create the prompt
        audio_result = self.get_reference_audio(voice_name)
        if audio_result is None:
            raise FileNotFoundError(f"Voice '{voice_name}' not found")

        audio_data, sr = audio_result

        # Pre-process audio for qwen-tts: convert stereo to mono and ensure float32
        # This prevents the library from trying to modify the tuple in-place
        if audio_data.ndim > 1 and audio_data.shape[-1] in (1, 2):
            # Convert stereo/mono to 1D array (mono)
            audio_data = np.mean(audio_data, axis=-1)
        audio_data = audio_data.astype(np.float32)

        # Get voice info to check transcription_type
        voice_info = self.get_voice_info(voice_name)
        transcription_type = voice_info.get("transcription_type", "MANUAL") if voice_info else "MANUAL"

        # Use x_vector_only_mode when transcription_type is NONE
        use_x_vector_only = (transcription_type == "NONE")

        if use_x_vector_only:
            logger.info(f"Creating voice clone prompt with x_vector_only_mode for {voice_name}/{model_id}")
            logger.info(f"  audio_data shape: {audio_data.shape}, dtype: {audio_data.dtype}, sr: {sr}")

            # Create prompt in thread pool with x_vector_only_mode
            loop = asyncio.get_event_loop()
            try:
                # Pass ref_audio as tuple (audio_data, sample_rate) as required by qwen-tts
                ref_audio_tuple = (audio_data, sr)
                prompt = await loop.run_in_executor(
                    None,
                    lambda: model.create_voice_clone_prompt(
                        ref_audio=ref_audio_tuple,
                        x_vector_only_mode=True,
                    )
                )
            except Exception as e:
                logger.error(f"Error in create_voice_clone_prompt (x_vector_only_mode): {e}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                raise
        else:
            transcription = self.get_transcription(voice_name)

            if transcription is None:
                raise ValueError(f"No transcription found for voice '{voice_name}'")

            logger.info(f"Creating voice clone prompt for {voice_name}/{model_id}")
            logger.info(f"  audio_data shape: {audio_data.shape}, dtype: {audio_data.dtype}, sr: {sr}")
            logger.info(f"  transcription: '{transcription[:100]}...' (len={len(transcription)})")

            # Create prompt in thread pool
            loop = asyncio.get_event_loop()
            try:
                # Pass ref_audio as tuple (audio_data, sample_rate) as required by qwen-tts
                ref_audio_tuple = (audio_data, sr)
                prompt = await loop.run_in_executor(
                    None,
                    lambda: model.create_voice_clone_prompt(
                        ref_audio=ref_audio_tuple,
                        ref_text=transcription,
                    )
                )
            except Exception as e:
                logger.error(f"Error in create_voice_clone_prompt: {e}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                raise

        # Save to disk cache (only for persistent voices)
        if self.is_cached_voice(voice_name):
            try:
                # Move tensors to CPU before saving for portability
                # This prevents CUDA context corruption when loading in different sessions
                prompt_cpu = self._move_prompt_to_cpu(prompt)
                torch.save(prompt_cpu, prompt_path)
                logger.debug(f"Saved prompt to disk for {voice_name}/{model_id}")
            except Exception as e:
                logger.warning(f"Failed to cache prompt to disk: {e}")

        # Store in memory cache with LRU eviction
        with self._cache_lock:
            if voice_name not in self._prompt_cache:
                self._evict_oldest_prompt_if_needed()
                self._prompt_cache[voice_name] = {}
            self._prompt_cache[voice_name][model_id] = prompt
            self._prompt_cache_access[voice_name] = time.time()

        return prompt

    def mark_in_use(self, voice_name: str) -> None:
        """Mark a voice as in use (prevents deletion)."""
        with self._cache_lock:
            self._in_use.add(voice_name)

    def mark_not_in_use(self, voice_name: str) -> None:
        """Mark a voice as no longer in use."""
        with self._cache_lock:
            self._in_use.discard(voice_name)

    def is_in_use(self, voice_name: str) -> bool:
        """Check if a voice is currently in use."""
        with self._cache_lock:
            return voice_name in self._in_use

    def clear_prompt_cache(self, voice_name: Optional[str] = None) -> None:
        """
        Clear prompt cache from memory.

        Args:
            voice_name: Specific voice to clear, or None for all
        """
        with self._cache_lock:
            if voice_name:
                self._prompt_cache.pop(voice_name, None)
                self._prompt_cache_access.pop(voice_name, None)
            else:
                self._prompt_cache.clear()
                self._prompt_cache_access.clear()

    def clear_all_caches(self, voice_name: Optional[str] = None) -> None:
        """
        Clear all in-memory caches (prompts and voice info).

        Args:
            voice_name: Specific voice to clear, or None for all
        """
        with self._cache_lock:
            if voice_name:
                self._prompt_cache.pop(voice_name, None)
                self._prompt_cache_access.pop(voice_name, None)
                self._voice_info_cache.pop(voice_name, None)
            else:
                self._prompt_cache.clear()
                self._prompt_cache_access.clear()
                self._voice_info_cache.clear()

    def clear_prompt_cache_for_model(self, model_id: str) -> int:
        """
        Clear all cached prompts for a specific model.

        This should be called when a model is unloaded to free GPU memory
        associated with that model's prompts.

        Args:
            model_id: The model identifier (e.g., "0.6B", "1.7B")

        Returns:
            Number of prompts cleared
        """
        cleared = 0
        with self._cache_lock:
            for voice_name in list(self._prompt_cache.keys()):
                if model_id in self._prompt_cache[voice_name]:
                    del self._prompt_cache[voice_name][model_id]
                    cleared += 1
                    # Clean up empty voice entries
                    if not self._prompt_cache[voice_name]:
                        del self._prompt_cache[voice_name]
                        self._prompt_cache_access.pop(voice_name, None)

        if cleared > 0:
            logger.info(f"Cleared {cleared} prompt(s) for model {model_id}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        return cleared

    def clear_disk_prompt_cache(self, voice_name: Optional[str] = None) -> int:
        """
        Delete prompt_*.pkl files from disk.

        Args:
            voice_name: Specific voice to clear, or None for all voices

        Returns:
            Number of prompt files deleted
        """
        deleted = 0

        if voice_name:
            voice_dir = self._get_voice_dir(voice_name)
            if voice_dir.exists():
                for pkl_file in voice_dir.glob("prompt_*.pkl"):
                    pkl_file.unlink()
                    deleted += 1
        else:
            if VOICES_DIR.exists():
                for voice_dir in VOICES_DIR.iterdir():
                    if voice_dir.is_dir():
                        for pkl_file in voice_dir.glob("prompt_*.pkl"):
                            pkl_file.unlink()
                            deleted += 1

        if deleted > 0:
            target = f"voice '{voice_name}'" if voice_name else "all voices"
            logger.info(f"Deleted {deleted} prompt file(s) from disk for {target}")

        return deleted

    def clear_full_prompt_cache(self, voice_name: Optional[str] = None) -> dict:
        """
        Clear both in-memory and on-disk prompt caches.

        Args:
            voice_name: Specific voice to clear, or None for all voices

        Returns:
            dict with clearing results
        """
        self.clear_prompt_cache(voice_name)
        disk_deleted = self.clear_disk_prompt_cache(voice_name)
        return {"memory_cleared": True, "disk_files_deleted": disk_deleted}

    def _get_device(self) -> str:
        """Get the current CUDA device or CPU."""
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _move_prompt_to_cpu(self, prompt: Any) -> Any:
        """
        Move all tensors in prompt to CPU for safe serialization.

        Args:
            prompt: Prompt object (may contain nested tensors)

        Returns:
            Prompt with all tensors moved to CPU
        """
        if isinstance(prompt, torch.Tensor):
            return prompt.cpu()
        elif isinstance(prompt, dict):
            return {k: self._move_prompt_to_cpu(v) for k, v in prompt.items()}
        elif isinstance(prompt, (list, tuple)):
            result = [self._move_prompt_to_cpu(v) for v in prompt]
            return type(prompt)(result) if isinstance(prompt, tuple) else result
        return prompt

    def _move_prompt_to_device(self, prompt: Any, device: str) -> Any:
        """
        Move all tensors in prompt to specified device.

        Args:
            prompt: Prompt object (may contain nested tensors)
            device: Target device (e.g., "cuda:0" or "cpu")

        Returns:
            Prompt with all tensors moved to device
        """
        if isinstance(prompt, torch.Tensor):
            return prompt.to(device)
        elif isinstance(prompt, dict):
            return {k: self._move_prompt_to_device(v, device) for k, v in prompt.items()}
        elif isinstance(prompt, (list, tuple)):
            result = [self._move_prompt_to_device(v, device) for v in prompt]
            return type(prompt)(result) if isinstance(prompt, tuple) else result
        return prompt


# Global singleton instance
voice_cache_manager = VoiceCacheManager()
