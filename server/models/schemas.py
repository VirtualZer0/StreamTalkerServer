"""Pydantic schemas for model management API."""

from datetime import datetime
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field


class ModelStatus(BaseModel):
    """Status information for a single model."""

    status: Literal["unloaded", "loading", "warming_up", "ready"] = Field(
        description="Current model state: unloaded, loading, warming_up, or ready"
    )
    auto_unload_minutes: int = Field(
        description="Auto-unload timeout in minutes (0 = disabled)"
    )
    inactive_since: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the model became inactive (last request completed)"
    )


class ModelInfo(BaseModel):
    """Detailed information about a model."""

    model_id: str = Field(description="Model identifier")
    huggingface_id: str = Field(description="HuggingFace model ID")
    status: ModelStatus = Field(description="Current status of the model")


class ModelsStatusResponse(BaseModel):
    """Response schema for GET /models/status."""

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "models": {
                    "0.6B": {
                        "status": "unloaded",
                        "auto_unload_minutes": 30,
                        "inactive_since": None,
                    },
                    "1.7B": {
                        "status": "ready",
                        "auto_unload_minutes": 30,
                        "inactive_since": "2026-02-07T12:00:00+00:00",
                    },
                }
            }]
        }
    }

    models: Dict[str, ModelStatus] = Field(
        description="Status of all available models keyed by model ID"
    )


class ModelLoadResponse(BaseModel):
    """Response schema for POST /models/{model_id}/load."""

    success: bool = Field(description="Whether the operation was successful")
    model_id: str = Field(description="Model identifier")
    message: str = Field(description="Status message")


class ModelUnloadResponse(BaseModel):
    """Response schema for POST /models/{model_id}/unload."""

    success: bool = Field(description="Whether the operation was successful")
    model_id: str = Field(description="Model identifier")
    message: str = Field(description="Status message")


class AutoUnloadRequest(BaseModel):
    """Request schema for POST /models/{model_id}/auto-unload."""

    minutes: int = Field(
        ge=0,
        description="Auto-unload timeout in minutes (0 = disabled)"
    )


class AutoUnloadResponse(BaseModel):
    """Response schema for POST /models/{model_id}/auto-unload."""

    success: bool = Field(description="Whether the operation was successful")
    model_id: str = Field(description="Model identifier")
    auto_unload_minutes: int = Field(description="New auto-unload timeout value")
    message: str = Field(description="Status message")
