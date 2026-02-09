"""Model management API endpoints."""

import logging
from typing import Literal, Union

from fastapi import APIRouter, HTTPException, Path, Query

from server.config import (
    AVAILABLE_MODELS,
    ALL_TTS_MODELS,
)
from server.models import model_manager
from server.models.schemas import (
    AutoUnloadRequest,
    AutoUnloadResponse,
    ModelLoadResponse,
    ModelsStatusResponse,
    ModelStatus,
    ModelUnloadResponse,
)
from server.settings import settings_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["Model Management"])


def validate_model_id(model_id: str) -> None:
    """
    Validate model_id and raise HTTPException if invalid.

    Raises:
        HTTPException: 400 if model_id is unknown
    """
    # Check all known TTS models (both available and unavailable)
    if model_id not in ALL_TTS_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_id '{model_id}'. Must be one of: {', '.join(ALL_TTS_MODELS.keys())}"
        )


def validate_model_available(model_id: str) -> None:
    """
    Validate that a model is available (cached locally).

    Args:
        model_id: Model identifier to check

    Raises:
        HTTPException: 503 if model is not available
    """
    if model_id not in AVAILABLE_MODELS:
        available = list(AVAILABLE_MODELS.keys()) if AVAILABLE_MODELS else []
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_id}' is not available. It may not be cached locally. "
                   f"Available models: {', '.join(available) if available else 'none'}"
        )


@router.get(
    "/status",
    response_model=ModelsStatusResponse,
    summary="Get status of all models",
    description="Returns the current status of all available TTS models (0.6B, 1.7B)."
)
async def get_models_status() -> ModelsStatusResponse:
    """Get status of all TTS models."""
    # Get TTS model statuses
    tts_status_dict = model_manager.get_all_status()

    # Convert TTS models to ModelStatus objects
    models = {}
    for model_id, status in tts_status_dict.items():
        if status["loading"]:
            model_status = "loading"
        elif status.get("warming_up", False):
            model_status = "warming_up"
        elif status["loaded"]:
            model_status = "ready"
        else:
            model_status = "unloaded"

        models[model_id] = ModelStatus(
            status=model_status,
            auto_unload_minutes=status["auto_unload_minutes"],
            inactive_since=status["inactive_since"],
        )

    return ModelsStatusResponse(models=models)


@router.post(
    "/{model_id}/load",
    response_model=ModelLoadResponse,
    summary="Load a model into memory",
    description="Loads the specified TTS model into GPU memory. Supports models "
                "0.6B and 1.7B. If the model is already loaded, returns success "
                "without reloading. Use warmup=true to run inference warmup after "
                "loading, which primes torch.compile optimization for faster first requests.",
    responses={
        400: {"description": "Invalid model ID"},
        503: {"description": "Model not available (not cached locally)"}
    }
)
async def load_model(
    model_id: str = Path(..., description="Model identifier (0.6B or 1.7B)"),
    warmup: Literal["none", "single", "batch"] = Query("none", description="Warmup mode: 'none' (skip), 'single' (one phrase), or 'batch' (multiple phrases)."),
    warmup_lang: str = Query(None, description="Language for warmup phrases (e.g., English, Russian, Chinese). If not specified, uses English."),
    warmup_voice: str = Query(None, description="Voice name to use for warmup (uses cached voice). If not specified, uses silence audio."),
    warmup_timeout: int = Query(120, description="Timeout in seconds for warmup inference. If exceeded, warmup is skipped to prevent hangs. Default: 120."),
    quantization: Literal["none", "int8", "float8"] = Query("none", description="Quantization mode: 'none' (BFloat16), 'int8', or 'float8'. Quantization reduces memory usage. Float8 may work better for autoregressive generation than INT8."),
    enable_optimizations: bool = Query(True, description="Enable streaming optimizations (compile, CUDA graphs, codebook). Requires GPU."),
    torch_compile: bool = Query(True, description="Apply torch.compile to decoder."),
    cuda_graphs: bool = Query(False, description="Capture CUDA graphs for fixed-size decode windows."),
    compile_codebook: bool = Query(True, description="Apply torch.compile to codebook predictor (~2x faster)."),
    fast_codebook: bool = Query(True, description="Use fast codebook generation."),
    attention: Literal["auto", "sage_attn", "flex_attn", "flash2_attn", "sdpa", "eager"] = Query("auto", description="Attention implementation: 'auto' (try flash2 > sdpa > eager), 'sage_attn', 'flex_attn', 'flash2_attn' (Flash Attention 2), 'sdpa', or 'eager'."),
    force_cpu: bool = Query(False, description="Force loading on CPU instead of GPU. WARNING: CPU inference is very slow (10-100x slower than GPU). Uses float32 instead of bfloat16 and eager attention only."),
) -> ModelLoadResponse:
    """Load a TTS model into memory."""
    validate_model_id(model_id)
    validate_model_available(model_id)

    try:
        already_loaded = model_manager.is_loaded(model_id)
        if not already_loaded:
            await model_manager.load_model(
                model_id,
                quantization=quantization,
                enable_optimizations=enable_optimizations,
                torch_compile=torch_compile,
                cuda_graphs=cuda_graphs,
                compile_codebook=compile_codebook,
                fast_codebook=fast_codebook,
                attention=attention,
                force_cpu=force_cpu,
            )

        # Run warmup if requested
        warmup_mode = warmup

        # Skip warmup on CPU (would be extremely slow)
        if force_cpu and warmup_mode != "none":
            logger.info(f"Skipping warmup for {model_id} on CPU (would be too slow)")
            warmup_mode = "none"

        if warmup_mode != "none" and not already_loaded:
            try:
                warmup_result = await model_manager.warmup_model(
                    model_id,
                    language=warmup_lang,
                    voice=warmup_voice,
                    mode=warmup_mode,
                    timeout_sec=warmup_timeout,
                )
                logger.info(f"Warmup result for {model_id}: {warmup_result}")

                # Handle warmup timeout/skip case
                if warmup_result.get("skipped"):
                    reason = warmup_result.get("reason", "unknown")
                    return ModelLoadResponse(
                        success=True,
                        model_id=model_id,
                        message=f"Model {model_id} loaded (warmup skipped: {reason})"
                    )

                return ModelLoadResponse(
                    success=True,
                    model_id=model_id,
                    message=f"Model {model_id} loaded and warmed up successfully "
                            f"({warmup_result.get('total_time', 0):.1f}s warmup)"
                )
            except Exception as warmup_error:
                logger.warning(f"Warmup failed for {model_id}: {warmup_error}")
                # Model is loaded, warmup just failed - still return success
                return ModelLoadResponse(
                    success=True,
                    model_id=model_id,
                    message=f"Model {model_id} loaded successfully (warmup failed: {warmup_error})"
                )
        elif already_loaded:
            return ModelLoadResponse(
                success=True,
                model_id=model_id,
                message=f"Model {model_id} is already loaded"
            )

        return ModelLoadResponse(
            success=True,
            model_id=model_id,
            message=f"Model {model_id} loaded successfully"
        )

    except RuntimeError as e:
        # Model not available error
        logger.error(f"Model {model_id} not available: {e}")
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )


@router.post(
    "/{model_id}/unload",
    response_model=ModelUnloadResponse,
    summary="Unload a model from memory",
    description="Unloads the specified TTS model from GPU memory. Supports models "
                "0.6B and 1.7B. If inference is in progress, the unload will be "
                "scheduled to happen after the current inference completes (non-blocking)."
)
async def unload_model(
    model_id: str = Path(..., description="Model identifier (0.6B or 1.7B)")
) -> ModelUnloadResponse:
    """Unload a TTS model from memory."""
    validate_model_id(model_id)

    try:
        result = await model_manager.unload_model(model_id)
        status = result.get("status", "unknown")

        if status == "not_loaded":
            return ModelUnloadResponse(
                success=True,
                model_id=model_id,
                message=f"Model {model_id} is not loaded"
            )
        elif status == "unload_pending":
            return ModelUnloadResponse(
                success=True,
                model_id=model_id,
                message=f"Model {model_id} will be unloaded when current inference completes"
            )
        else:  # status == "unloaded"
            return ModelUnloadResponse(
                success=True,
                model_id=model_id,
                message=f"Model {model_id} unloaded successfully"
            )

    except Exception as e:
        logger.error(f"Failed to unload model {model_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to unload model: {str(e)}"
        )


@router.post(
    "/{model_id}/auto-unload",
    response_model=AutoUnloadResponse,
    summary="Configure auto-unload timeout",
    description="Sets the auto-unload timeout for a TTS model. Supports models "
                "0.6B and 1.7B. The model will be automatically unloaded from "
                "memory after this many minutes of inactivity. Set to 0 to disable auto-unload."
)
async def set_auto_unload(
    request: AutoUnloadRequest,
    model_id: str = Path(..., description="Model identifier (0.6B or 1.7B)")
) -> AutoUnloadResponse:
    """Configure auto-unload timeout for a TTS model."""
    validate_model_id(model_id)

    try:
        success = settings_manager.set_auto_unload_minutes(model_id, request.minutes)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to save settings"
            )

        message = (
            f"Auto-unload disabled for model {model_id}"
            if request.minutes == 0
            else f"Auto-unload set to {request.minutes} minutes for model {model_id}"
        )

        return AutoUnloadResponse(
            success=True,
            model_id=model_id,
            auto_unload_minutes=request.minutes,
            message=message
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set auto-unload for {model_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to set auto-unload: {str(e)}"
        )
