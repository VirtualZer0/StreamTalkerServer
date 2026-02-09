"""Pydantic schemas for environment information API."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class GPUInfo(BaseModel):
    """Information about a single GPU."""

    index: int = Field(description="GPU device index")
    name: str = Field(description="GPU model name")
    memory_total_mb: float = Field(description="Total GPU memory in MB")
    memory_used_mb: float = Field(description="Used GPU memory in MB")
    memory_free_mb: float = Field(description="Free GPU memory in MB")
    utilization_percent: Optional[float] = Field(
        default=None,
        description="GPU utilization percentage (if available)"
    )


class DeviceInfo(BaseModel):
    """Device and GPU information."""

    type: Literal["cuda", "cpu"] = Field(description="Device type")
    cuda_available: bool = Field(description="Whether CUDA is available")
    cuda_version: Optional[str] = Field(
        default=None,
        description="CUDA version (if available)"
    )
    cudnn_version: Optional[str] = Field(
        default=None,
        description="cuDNN version (if available)"
    )
    gpu_count: int = Field(description="Number of available GPUs")
    gpus: List[GPUInfo] = Field(
        default_factory=list,
        description="Information about each GPU"
    )
    current_device: str = Field(
        description="Device currently being used by the model manager"
    )


class SystemInfo(BaseModel):
    """System resource information."""

    platform: str = Field(description="Operating system platform")
    python_version: str = Field(description="Python version")
    cpu_count: int = Field(description="Number of physical CPU cores")
    cpu_count_logical: int = Field(description="Number of logical CPU cores")
    cpu_percent: float = Field(description="Current CPU usage percentage")
    memory_total_mb: float = Field(description="Total system RAM in MB")
    memory_available_mb: float = Field(description="Available system RAM in MB")
    memory_used_mb: float = Field(description="Used system RAM in MB")
    memory_percent: float = Field(description="RAM usage percentage")


class ProcessInfo(BaseModel):
    """Current process information."""

    pid: int = Field(description="Process ID")
    memory_rss_mb: float = Field(description="Resident Set Size memory in MB")
    memory_vms_mb: float = Field(description="Virtual Memory Size in MB")
    memory_percent: float = Field(description="Process memory as percentage of total")
    cpu_percent: float = Field(description="Process CPU usage percentage")
    num_threads: int = Field(description="Number of threads")
    uptime_seconds: float = Field(description="Process uptime in seconds")


class RuntimeInfo(BaseModel):
    """Runtime and application information."""

    pytorch_version: str = Field(description="PyTorch version")
    server_version: str = Field(description="Server application version")
    models_loaded: List[str] = Field(
        description="List of currently loaded TTS model IDs"
    )


class EnvironmentResponse(BaseModel):
    """Complete environment information response."""

    device: DeviceInfo = Field(description="Device and GPU information")
    system: SystemInfo = Field(description="System resource information")
    process: ProcessInfo = Field(description="Current process information")
    runtime: RuntimeInfo = Field(description="Runtime and application info")


class GPUUsageResponse(BaseModel):
    """Lightweight GPU usage response (fast endpoint)."""

    total_mb: float = Field(description="Total GPU memory in MB")
    used_mb: float = Field(description="Used GPU memory in MB")


class InferenceTimeoutResponse(BaseModel):
    """Inference timeout setting response."""

    timeout_seconds: int = Field(
        description="Inference timeout in seconds. Generation stops if exceeded."
    )


class InferenceTimeoutSetResponse(BaseModel):
    """Response after setting inference timeout."""

    timeout_seconds: int = Field(description="The new timeout value in seconds")
    saved: bool = Field(description="Whether the setting was persisted to disk")


class MaxVramResponse(BaseModel):
    """Maximum VRAM limit setting response."""

    max_vram_mb: float = Field(
        description="Maximum allowed VRAM usage in megabytes. "
                    "Inference stops early if this limit is exceeded. "
                    "Set to 0 to disable the limit."
    )
    total_vram_mb: float = Field(
        description="Total GPU VRAM available in megabytes"
    )
    usage_percent: float = Field(
        description="max_vram_mb as percentage of total_vram_mb"
    )


class MaxVramSetResponse(BaseModel):
    """Response after setting maximum VRAM limit."""

    max_vram_mb: int = Field(description="The new max VRAM limit in megabytes")
    saved: bool = Field(description="Whether the setting was persisted to disk")
