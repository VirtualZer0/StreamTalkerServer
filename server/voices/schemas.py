"""Pydantic schemas for voice management API."""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class TranscriptionType(str, Enum):
    """Type of transcription used for voice cloning."""

    MANUAL = "MANUAL"    # User-provided transcription
    NONE = "NONE"        # No transcription - uses x_vector_only_mode


class VoiceInfo(BaseModel):
    """Information about a cached voice."""

    name: str = Field(description="Voice name/identifier")
    created_at: datetime = Field(description="When the voice was created")
    transcription: str = Field(description="Transcription of the reference audio")
    transcription_type: TranscriptionType = Field(
        default=TranscriptionType.MANUAL,
        description="How the transcription was obtained (MANUAL or NONE)"
    )
    cached_models: List[str] = Field(
        default_factory=list,
        description="List of model IDs that have cached prompts for this voice"
    )


class VoiceListResponse(BaseModel):
    """Response schema for GET /voices."""

    voices: List[VoiceInfo] = Field(description="List of all cached voices")


class VoiceDetailResponse(BaseModel):
    """Response schema for GET /voices/{voice_name}."""

    voice: VoiceInfo = Field(description="Voice information")
    cached_models: List[str] = Field(
        description="List of model IDs that have cached prompts for this voice"
    )


class VoiceCreateResponse(BaseModel):
    """Response schema for POST /voices/{voice_name}."""

    success: bool = Field(description="Whether the operation was successful")
    voice: VoiceInfo = Field(description="Created/updated voice information")
    message: str = Field(description="Status message")


class VoiceDeleteResponse(BaseModel):
    """Response schema for DELETE /voices/{voice_name}."""

    success: bool = Field(description="Whether the operation was successful")
    voice_name: str = Field(description="Name of the deleted voice")
    message: str = Field(description="Status message")


class VoiceUploadForm(BaseModel):
    """Form data for voice upload (for documentation purposes)."""

    transcription: Optional[str] = Field(
        default=None,
        description="Transcription of the audio. Recommended for best voice cloning quality."
    )
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite if voice already exists"
    )
    disable_transcription: bool = Field(
        default=False,
        description="Skip transcription entirely and use x_vector_only_mode for voice cloning (reduced quality)"
    )


class PromptCacheClearResponse(BaseModel):
    """Response schema for POST /voices/clear-prompt-cache."""

    success: bool = Field(description="Whether the operation was successful")
    voice_name: Optional[str] = Field(
        default=None,
        description="Voice name if clearing specific voice, null if clearing all"
    )
    disk_files_deleted: int = Field(
        description="Number of prompt files deleted from disk"
    )
    message: str = Field(description="Status message")


class VoiceRenameRequest(BaseModel):
    """Request schema for PATCH /voices/{voice_name}/rename."""

    new_name: str = Field(
        ...,
        description="New name for the voice",
        min_length=1,
        max_length=64,
        pattern=r'^[\w-]+$'
    )


class VoiceRenameResponse(BaseModel):
    """Response schema for PATCH /voices/{voice_name}/rename."""

    success: bool = Field(description="Whether the operation was successful")
    old_name: str = Field(description="Original voice name")
    new_name: str = Field(description="New voice name")
    message: str = Field(description="Status message")
