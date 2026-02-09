"""Settings manager for persistent configuration storage."""

import json
import logging
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Dict, Optional

from server.config import (
    AVAILABLE_MODELS,
    DATA_DIR,
    DEFAULT_AUTO_UNLOAD_MINUTES,
    DEFAULT_INFERENCE_TIMEOUT,
    DEFAULT_MAX_VRAM_PERCENT,
    SETTINGS_FILE,
    VOICES_DIR,
    CACHE_DIR,
)

logger = logging.getLogger(__name__)


def _get_default_max_vram_bytes() -> int:
    """
    Calculate default max VRAM bytes (70% of total GPU VRAM).

    Returns:
        Default max VRAM in bytes, or 0 if no GPU available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory
            return int(total_vram * DEFAULT_MAX_VRAM_PERCENT / 100)
    except Exception as e:
        logger.warning(f"Could not detect GPU VRAM: {e}")
    return 0


class SettingsManager:
    """
    Manages persistent settings stored in /data/settings.json.

    Thread-safe singleton that handles loading, saving, and accessing
    configuration settings that persist across container restarts.
    """

    _instance: Optional["SettingsManager"] = None
    _lock = Lock()

    def __new__(cls) -> "SettingsManager":
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the settings manager."""
        if self._initialized:
            return

        self._settings: Dict[str, Any] = {}
        self._file_lock = RLock()
        self._initialized = True

        # Ensure data directories exist
        self._ensure_directories()

        # Load settings from file
        self.load()

    def _ensure_directories(self) -> None:
        """Ensure all required data directories exist."""
        directories = [DATA_DIR, VOICES_DIR, CACHE_DIR]

        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {directory}")
            except PermissionError:
                logger.warning(
                    f"Cannot create directory {directory} - running in memory-only mode"
                )
            except Exception as e:
                logger.error(f"Error creating directory {directory}: {e}")

    def _get_defaults(self) -> Dict[str, Any]:
        """Get default settings structure."""
        return {
            "models": {
                model_id: {"auto_unload_minutes": DEFAULT_AUTO_UNLOAD_MINUTES}
                for model_id in AVAILABLE_MODELS
            },
            "inference": {
                "timeout_seconds": DEFAULT_INFERENCE_TIMEOUT
            }
        }

    def load(self) -> None:
        """Load settings from file, creating defaults if needed."""
        with self._file_lock:
            if SETTINGS_FILE.exists():
                try:
                    with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                        self._settings = json.load(f)
                    logger.info(f"Loaded settings from {SETTINGS_FILE}")

                    # Ensure all models have settings
                    self._merge_defaults()

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in settings file: {e}")
                    self._settings = self._get_defaults()
                    self.save()
                except Exception as e:
                    logger.error(f"Error loading settings: {e}")
                    self._settings = self._get_defaults()
            else:
                logger.info("No settings file found, creating defaults")
                self._settings = self._get_defaults()
                self.save()

    def _merge_defaults(self) -> None:
        """Merge default settings for any missing keys."""
        defaults = self._get_defaults()
        changed = False

        # Ensure models section exists
        if "models" not in self._settings:
            self._settings["models"] = {}
            changed = True

        # Ensure each model has settings
        for model_id in AVAILABLE_MODELS:
            if model_id not in self._settings["models"]:
                self._settings["models"][model_id] = defaults["models"][model_id]
                changed = True
            else:
                # Ensure auto_unload_minutes exists
                if "auto_unload_minutes" not in self._settings["models"][model_id]:
                    self._settings["models"][model_id]["auto_unload_minutes"] = (
                        DEFAULT_AUTO_UNLOAD_MINUTES
                    )
                    changed = True

        # Ensure inference section exists
        if "inference" not in self._settings:
            self._settings["inference"] = defaults["inference"]
            changed = True
        else:
            # Ensure timeout_seconds exists
            if "timeout_seconds" not in self._settings["inference"]:
                self._settings["inference"]["timeout_seconds"] = (
                    DEFAULT_INFERENCE_TIMEOUT
                )
                changed = True

        # Ensure max_vram_bytes exists (initialize to 70% of GPU VRAM on first launch)
        if "inference" not in self._settings:
            self._settings["inference"] = {}
        if "max_vram_bytes" not in self._settings["inference"]:
            default_vram = _get_default_max_vram_bytes()
            self._settings["inference"]["max_vram_bytes"] = default_vram
            if default_vram > 0:
                logger.info(
                    f"Initialized max VRAM limit to {default_vram / (1024**3):.2f} GB "
                    f"({DEFAULT_MAX_VRAM_PERCENT}% of total)"
                )
            changed = True

        if changed:
            self.save()

    def save(self) -> bool:
        """
        Save current settings to file.

        Returns:
            True if save was successful, False otherwise.
        """
        with self._file_lock:
            try:
                # Ensure parent directory exists
                SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

                with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                    json.dump(self._settings, f, indent=2)

                logger.debug(f"Saved settings to {SETTINGS_FILE}")
                return True

            except PermissionError:
                logger.warning(
                    f"Cannot write to {SETTINGS_FILE} - settings will not persist"
                )
                return False
            except Exception as e:
                logger.error(f"Error saving settings: {e}")
                return False

    def get_auto_unload_minutes(self, model_id: str) -> int:
        """
        Get auto-unload timeout for a model.

        Args:
            model_id: Model identifier ("0.6B" or "1.7B")

        Returns:
            Auto-unload timeout in minutes (0 = disabled)
        """
        if model_id not in AVAILABLE_MODELS:
            logger.warning(f"Unknown model ID: {model_id}")
            return DEFAULT_AUTO_UNLOAD_MINUTES

        try:
            return self._settings["models"][model_id]["auto_unload_minutes"]
        except KeyError:
            return DEFAULT_AUTO_UNLOAD_MINUTES

    def set_auto_unload_minutes(self, model_id: str, minutes: int) -> bool:
        """
        Set auto-unload timeout for a model.

        Args:
            model_id: Model identifier ("0.6B" or "1.7B")
            minutes: Timeout in minutes (0 = disabled)

        Returns:
            True if setting was saved successfully
        """
        if model_id not in AVAILABLE_MODELS:
            logger.warning(f"Unknown model ID: {model_id}")
            return False

        if minutes < 0:
            logger.warning(f"Invalid minutes value: {minutes}")
            return False

        # Update in memory
        if "models" not in self._settings:
            self._settings["models"] = {}
        if model_id not in self._settings["models"]:
            self._settings["models"][model_id] = {}

        self._settings["models"][model_id]["auto_unload_minutes"] = minutes

        # Persist to file
        return self.save()

    def get_all_settings(self) -> Dict[str, Any]:
        """Get a copy of all settings."""
        return self._settings.copy()

    def get_inference_timeout(self) -> int:
        """
        Get inference timeout in seconds.

        Returns:
            Inference timeout in seconds
        """
        try:
            return self._settings["inference"]["timeout_seconds"]
        except KeyError:
            return DEFAULT_INFERENCE_TIMEOUT

    def set_inference_timeout(self, seconds: int) -> bool:
        """
        Set inference timeout in seconds.

        Args:
            seconds: Timeout in seconds (minimum 10)

        Returns:
            True if setting was saved successfully
        """
        if seconds < 10:
            logger.warning(f"Invalid timeout value: {seconds} (minimum 10)")
            return False

        # Update in memory
        if "inference" not in self._settings:
            self._settings["inference"] = {}

        self._settings["inference"]["timeout_seconds"] = seconds

        # Persist to file
        return self.save()

    def get_max_vram_bytes(self) -> int:
        """
        Get maximum VRAM usage limit in bytes.

        Returns:
            Maximum VRAM in bytes (0 = no limit / no GPU)
        """
        try:
            return self._settings["inference"]["max_vram_bytes"]
        except KeyError:
            return _get_default_max_vram_bytes()

    def set_max_vram_bytes(self, bytes_limit: int) -> bool:
        """
        Set maximum VRAM usage limit in bytes.

        Args:
            bytes_limit: Maximum VRAM in bytes (0 = no limit)

        Returns:
            True if setting was saved successfully
        """
        if bytes_limit < 0:
            logger.warning(f"Invalid max VRAM value: {bytes_limit} (must be >= 0)")
            return False

        # Update in memory
        if "inference" not in self._settings:
            self._settings["inference"] = {}

        self._settings["inference"]["max_vram_bytes"] = bytes_limit

        # Persist to file
        return self.save()

    def get_total_vram_bytes(self) -> int:
        """
        Get total GPU VRAM in bytes.

        Returns:
            Total VRAM in bytes, or 0 if no GPU available.
        """
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory
        except Exception:
            pass
        return 0


# Global singleton instance
settings_manager = SettingsManager()
