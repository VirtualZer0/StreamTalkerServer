"""Environment information API endpoint."""

import logging
import platform
from datetime import datetime, timezone
from typing import List

import psutil
import torch
from fastapi import APIRouter

from server.config import AVAILABLE_MODELS
from server.models import model_manager
from server.settings import settings_manager

from .environment_schemas import (
    DeviceInfo,
    EnvironmentResponse,
    GPUInfo,
    GPUUsageResponse,
    InferenceTimeoutResponse,
    InferenceTimeoutSetResponse,
    MaxVramResponse,
    MaxVramSetResponse,
    ProcessInfo,
    RuntimeInfo,
    SystemInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/environment", tags=["Environment"])

# Store server start time for uptime calculation
_server_start_time = datetime.now(timezone.utc)


def get_gpu_info() -> List[GPUInfo]:
    """Get information about all available GPUs."""
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            # Get memory info
            mem_total = props.total_memory / (1024 ** 2)  # Convert to MB
            mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)

            gpus.append(GPUInfo(
                index=i,
                name=props.name,
                memory_total_mb=round(mem_total, 2),
                memory_used_mb=round(mem_reserved, 2),
                memory_free_mb=round(mem_total - mem_reserved, 2),
                utilization_percent=None,  # Requires pynvml for accurate reading
            ))
    return gpus


def get_device_info() -> DeviceInfo:
    """Get device and CUDA information."""
    cuda_available = torch.cuda.is_available()

    cuda_version = None
    cudnn_version = None

    if cuda_available:
        cuda_version = torch.version.cuda
        if torch.backends.cudnn.is_available():
            cudnn_version = str(torch.backends.cudnn.version())

    return DeviceInfo(
        type="cuda" if cuda_available else "cpu",
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        cudnn_version=cudnn_version,
        gpu_count=torch.cuda.device_count() if cuda_available else 0,
        gpus=get_gpu_info(),
        current_device=model_manager.device,
    )


def get_system_info() -> SystemInfo:
    """Get system resource information."""
    memory = psutil.virtual_memory()

    return SystemInfo(
        platform=platform.platform(),
        python_version=platform.python_version(),
        cpu_count=psutil.cpu_count(logical=False) or 1,
        cpu_count_logical=psutil.cpu_count(logical=True) or 1,
        cpu_percent=psutil.cpu_percent(interval=0.1),
        memory_total_mb=round(memory.total / (1024 ** 2), 2),
        memory_available_mb=round(memory.available / (1024 ** 2), 2),
        memory_used_mb=round(memory.used / (1024 ** 2), 2),
        memory_percent=memory.percent,
    )


def get_process_info() -> ProcessInfo:
    """Get current process information."""
    process = psutil.Process()
    mem_info = process.memory_info()

    # Calculate uptime
    now = datetime.now(timezone.utc)
    uptime = (now - _server_start_time).total_seconds()

    return ProcessInfo(
        pid=process.pid,
        memory_rss_mb=round(mem_info.rss / (1024 ** 2), 2),
        memory_vms_mb=round(mem_info.vms / (1024 ** 2), 2),
        memory_percent=round(process.memory_percent(), 2),
        cpu_percent=process.cpu_percent(interval=0.1),
        num_threads=process.num_threads(),
        uptime_seconds=round(uptime, 2),
    )


def get_runtime_info() -> RuntimeInfo:
    """Get runtime and application information."""
    # Import server version
    from server import __version__

    # Get list of loaded TTS models
    loaded_models = [
        model_id for model_id in AVAILABLE_MODELS
        if model_manager.is_loaded(model_id)
    ]

    return RuntimeInfo(
        pytorch_version=torch.__version__,
        server_version=__version__,
        models_loaded=loaded_models,
    )


@router.get(
    "",
    response_model=EnvironmentResponse,
    summary="Get environment information",
    description="Returns comprehensive information about the server environment "
                "including device/GPU info, system resources, process stats, "
                "and runtime information."
)
async def get_environment() -> EnvironmentResponse:
    """Get comprehensive environment information."""
    return EnvironmentResponse(
        device=get_device_info(),
        system=get_system_info(),
        process=get_process_info(),
        runtime=get_runtime_info(),
    )


@router.get(
    "/gpu-usage",
    response_model=GPUUsageResponse,
    summary="Get GPU memory usage (fast)",
    description="Returns total and reserved GPU memory. Uses memory_reserved() which "
                "reflects actual GPU memory claimed by PyTorch (same as max-vram limit check)."
)
async def get_gpu_usage() -> GPUUsageResponse:
    """Get GPU memory usage (fast, lightweight endpoint)."""
    if not torch.cuda.is_available():
        return GPUUsageResponse(total_mb=0, used_mb=0)

    props = torch.cuda.get_device_properties(0)
    total = props.total_memory
    # Use memory_reserved() to match max-vram limit checking behavior
    used = torch.cuda.memory_reserved(0)

    return GPUUsageResponse(
        total_mb=round(total / (1024 ** 2), 2),
        used_mb=round(used / (1024 ** 2), 2),
    )


@router.get(
    "/inference-timeout",
    response_model=InferenceTimeoutResponse,
    summary="Get inference timeout",
    description="Returns the current inference timeout in seconds. "
                "Generation will be stopped early if this timeout is exceeded."
)
async def get_inference_timeout() -> InferenceTimeoutResponse:
    """Get current inference timeout setting."""
    return InferenceTimeoutResponse(
        timeout_seconds=settings_manager.get_inference_timeout()
    )


@router.put(
    "/inference-timeout",
    response_model=InferenceTimeoutSetResponse,
    summary="Set inference timeout",
    description="Set the inference timeout in seconds. Minimum 10 seconds. "
                "Generation will be stopped early if this timeout is exceeded. "
                "This setting persists across server restarts."
)
async def set_inference_timeout(timeout_seconds: int) -> InferenceTimeoutSetResponse:
    """Set inference timeout setting."""
    if timeout_seconds < 10:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail="Timeout must be at least 10 seconds"
        )

    success = settings_manager.set_inference_timeout(timeout_seconds)
    if not success:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail="Failed to save inference timeout setting"
        )

    logger.info(f"Inference timeout set to {timeout_seconds} seconds")
    return InferenceTimeoutSetResponse(timeout_seconds=timeout_seconds, saved=True)


@router.get(
    "/max-vram",
    response_model=MaxVramResponse,
    summary="Get maximum VRAM limit",
    description="Returns the current maximum VRAM usage limit in megabytes. "
                "Inference will stop early if total VRAM usage exceeds this limit. "
                "Default is 70% of total GPU VRAM on first launch."
)
async def get_max_vram() -> MaxVramResponse:
    """Get current maximum VRAM limit setting."""
    max_bytes = settings_manager.get_max_vram_bytes()
    total_bytes = settings_manager.get_total_vram_bytes()

    return MaxVramResponse(
        max_vram_mb=round(max_bytes / (1024 ** 2), 2) if max_bytes > 0 else 0,
        total_vram_mb=round(total_bytes / (1024 ** 2), 2) if total_bytes > 0 else 0,
        usage_percent=round(max_bytes / total_bytes * 100, 1) if total_bytes > 0 else 0,
    )


@router.put(
    "/max-vram",
    response_model=MaxVramSetResponse,
    summary="Set maximum VRAM limit",
    description="Set the maximum VRAM usage limit in megabytes. "
                "Inference will stop early if total VRAM usage exceeds this limit. "
                "Set to 0 to disable the limit. "
                "This setting persists across server restarts."
)
async def set_max_vram(max_vram_mb: int) -> MaxVramSetResponse:
    """Set maximum VRAM limit setting."""
    if max_vram_mb < 0:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail="max_vram_mb must be >= 0"
        )

    total_bytes = settings_manager.get_total_vram_bytes()
    total_mb = total_bytes / (1024 ** 2) if total_bytes > 0 else 0

    if total_mb > 0 and max_vram_mb > total_mb:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"max_vram_mb ({max_vram_mb}) exceeds total GPU VRAM ({total_mb:.0f} MB)"
        )

    # Convert MB to bytes for storage
    max_bytes = max_vram_mb * (1024 ** 2)
    success = settings_manager.set_max_vram_bytes(max_bytes)
    if not success:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail="Failed to save max VRAM setting"
        )

    logger.info(f"Max VRAM limit set to {max_vram_mb} MB")
    return MaxVramSetResponse(max_vram_mb=max_vram_mb, saved=True)
