"""
Stream Talker Server - Main Application Entry Point

A FastAPI-based text-to-speech server powered by Qwen3-TTS with support for:
- Multi-model architecture (0.6B and 1.7B TTS models)
- Dynamic model loading/unloading
- Voice cloning with persistent caching
- Automatic inactivity-based model unloading
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.api import api_router
from server.config import (
    AUTO_UNLOAD_CHECK_INTERVAL,
    OUTPUT_DIR,
    SERVER_HOST,
    SERVER_PORT,
    initialize_available_models,
    AVAILABLE_MODELS,
)
from server.models import model_manager
from server.settings import settings_manager
from server.voices import voice_cache_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Filter to exclude frequently-called monitoring endpoints from access logs
class EndpointFilter(logging.Filter):
    """Filter out health/status check endpoints from access logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Exclude monitoring endpoints that are called frequently
        excluded_paths = [
            "/health",
            "/models/status",
            "/environment/gpu-usage",
            "/environment/inference-timeout",
            "/environment/max-vram",
            "/environment",  # Full environment info
        ]

        # Check if this is an access log message
        if hasattr(record, "args") and len(record.args) >= 3:
            path = str(record.args[2])  # Path is typically the 3rd argument
            return not any(path.startswith(excluded) for excluded in excluded_paths)

        return True


# Apply the filter to uvicorn access logger
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())


async def auto_unload_task():
    """Background task that periodically checks for models to auto-unload."""
    import traceback

    logger.info(
        f"Auto-unload checker started "
        f"(interval: {AUTO_UNLOAD_CHECK_INTERVAL}s)"
    )

    while True:
        try:
            await asyncio.sleep(AUTO_UNLOAD_CHECK_INTERVAL)
            # Check TTS models for auto-unload
            await model_manager.check_auto_unload()
        except asyncio.CancelledError:
            logger.info("Auto-unload checker stopped")
            break
        except Exception as e:
            logger.error(f"Error in auto-unload checker: {e}\n{traceback.format_exc()}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # =========================================================================
    # Startup
    # =========================================================================
    logger.info("Starting Stream Talker Server...")

    # Log GPU compatibility information at startup
    from server.utils.gpu import log_gpu_compatibility
    log_gpu_compatibility()

    # Ensure local directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize available models (check which models are cached locally)
    initialize_available_models()
    logger.info(f"Available TTS models: {list(AVAILABLE_MODELS.keys())}")

    # Initialize settings (creates /data directories and settings.json)
    _ = settings_manager  # Trigger initialization
    logger.info("Settings manager initialized")

    # Initialize TTS model manager (but don't load any models)
    _ = model_manager  # Trigger initialization
    logger.info(f"TTS model manager initialized (device: {model_manager.device})")

    # Initialize voice cache manager
    _ = voice_cache_manager  # Trigger initialization
    logger.info("Voice cache manager initialized")

    # Start auto-unload background task
    auto_unload_task_handle = asyncio.create_task(auto_unload_task())

    logger.info("Stream Talker Server started successfully")

    yield

    # =========================================================================
    # Shutdown
    # =========================================================================
    logger.info("Shutting down Stream Talker Server...")

    # Cancel auto-unload task
    auto_unload_task_handle.cancel()
    try:
        await auto_unload_task_handle
    except asyncio.CancelledError:
        pass

    logger.info("Stream Talker Server shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Stream Talker Server",
    description="A powerful text-to-speech server with multi-model support, "
                "voice cloning, and dynamic model management.",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with server info."""
    from server.config import AVAILABLE_MODELS
    return {
        "name": "Stream Talker Server",
        "version": "1.0.0",
        "status": "running",
        "models_available": list(AVAILABLE_MODELS.keys()),
    }


@app.get("/health", tags=["Health"])
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


def main():
    """Run the server using uvicorn."""
    import uvicorn

    uvicorn.run(
        "server.main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
