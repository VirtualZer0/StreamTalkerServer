"""Voice management API endpoints."""

import logging
import re
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Path, Query, UploadFile

from server.utils.audio import process_reference_audio, validate_audio_file
from server.voices import voice_cache_manager
from server.voices.schemas import (
    PromptCacheClearResponse,
    TranscriptionType,
    VoiceCreateResponse,
    VoiceDeleteResponse,
    VoiceDetailResponse,
    VoiceInfo,
    VoiceListResponse,
    VoiceRenameRequest,
    VoiceRenameResponse,
)
from server.config import OUTPUT_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voices", tags=["Voice Management"])


@router.get(
    "",
    response_model=VoiceListResponse,
    summary="List all cached voices",
    description="Returns a list of all voices that have been cached for voice cloning."
)
async def list_voices() -> VoiceListResponse:
    """List all cached voices."""
    voices_data = voice_cache_manager.list_voices()

    voices = [
        VoiceInfo(
            name=v["name"],
            created_at=datetime.fromisoformat(v["created_at"]),
            transcription=v.get("transcription", ""),
            transcription_type=TranscriptionType(v.get("transcription_type", "MANUAL")),
            cached_models=voice_cache_manager.get_cached_models(v["name"]),
        )
        for v in voices_data
    ]

    return VoiceListResponse(voices=voices)


@router.post(
    "/clear-prompt-cache",
    response_model=PromptCacheClearResponse,
    summary="Clear voice prompt cache",
    description="Clears cached voice prompts from both memory and disk. "
                "Prompt caches are .pkl files created when a voice is first used with a model. "
                "Use voice_name to clear a specific voice, or omit to clear all.",
    responses={
        404: {"description": "Voice not found"},
    }
)
async def clear_prompt_cache(
    voice_name: Optional[str] = Query(
        default=None,
        description="Specific voice to clear cache for. If omitted, clears all voice caches."
    ),
) -> PromptCacheClearResponse:
    """Clear voice prompt cache from memory and disk."""
    if voice_name and not voice_cache_manager.is_cached_voice(voice_name):
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{voice_name}' not found"
        )

    try:
        result = voice_cache_manager.clear_full_prompt_cache(voice_name)
        target = f"voice '{voice_name}'" if voice_name else "all voices"
        return PromptCacheClearResponse(
            success=True,
            voice_name=voice_name,
            disk_files_deleted=result["disk_files_deleted"],
            message=f"Prompt cache cleared for {target} ({result['disk_files_deleted']} file(s) deleted from disk)"
        )
    except Exception as e:
        logger.error(f"Failed to clear prompt cache: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear prompt cache: {str(e)}"
        )


@router.get(
    "/{voice_name}",
    response_model=VoiceDetailResponse,
    summary="Get voice information",
    description="Returns detailed information about a specific cached voice."
)
async def get_voice(
    voice_name: str = Path(..., description="Name of the voice")
) -> VoiceDetailResponse:
    """Get information about a specific voice."""
    voice_data = voice_cache_manager.get_voice_info(voice_name)

    if voice_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{voice_name}' not found"
        )

    cached_models = voice_cache_manager.get_cached_models(voice_name)

    voice = VoiceInfo(
        name=voice_data["name"],
        created_at=datetime.fromisoformat(voice_data["created_at"]),
        transcription=voice_data.get("transcription", ""),
        transcription_type=TranscriptionType(voice_data.get("transcription_type", "MANUAL")),
        cached_models=cached_models,
    )

    return VoiceDetailResponse(voice=voice, cached_models=cached_models)


@router.post(
    "/{voice_name}",
    response_model=VoiceCreateResponse,
    summary="Create or update a cached voice",
    description="Upload an audio file to create a new cached voice for voice cloning. "
                "Providing a transcription is recommended for best quality. "
                "Set disable_transcription=true to skip transcription entirely "
                "(uses x_vector_only_mode with reduced quality). "
                "Set overwrite=true to replace an existing voice."
)
async def create_voice(
    voice_name: str = Path(..., description="Name for the voice"),
    file: UploadFile = File(..., description="Audio file (wav, mp3, flac, ogg)"),
    transcription: Optional[str] = Form(
        default=None,
        description="Optional transcription of the audio"
    ),
    overwrite: bool = Form(
        default=False,
        description="Overwrite if voice already exists"
    ),
    disable_transcription: bool = Form(
        default=False,
        description="Skip transcription and use x_vector_only_mode (reduced quality)"
    ),
) -> VoiceCreateResponse:
    """Create or update a cached voice."""
    # Validate voice name
    if not voice_name or not voice_name.strip():
        raise HTTPException(
            status_code=400,
            detail="Voice name cannot be empty"
        )

    # Sanitize voice name (alphanumeric and underscores only)
    if not re.match(r'^[\w-]+$', voice_name, re.UNICODE):
        raise HTTPException(
            status_code=400,
            detail="Voice name can only contain letters, numbers, underscores, and hyphens"
        )

    # Validate that transcription and disable_transcription are not both set
    if disable_transcription and transcription:
        raise HTTPException(
            status_code=400,
            detail="Cannot provide both transcription and disable_transcription=true"
        )

    # Check if voice already exists (for determining create vs update message later)
    voice_existed = voice_cache_manager.is_cached_voice(voice_name)

    # Check if voice exists and overwrite is not set
    if voice_existed and not overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"Voice '{voice_name}' already exists. Use overwrite=true to replace."
        )

    # Read and validate file
    contents = await file.read()

    is_valid, error = validate_audio_file(contents, file.filename or "")
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save uploaded file temporarily
    temp_path = OUTPUT_DIR / f"temp_upload_{voice_name}.wav"
    with open(temp_path, 'wb') as f:
        f.write(contents)

    try:
        # Process reference audio
        audio_data, sample_rate, final_transcription = process_reference_audio(
            str(temp_path),
            transcription,
            skip_transcription=disable_transcription,
        )

        # Determine transcription_type based on inputs
        if disable_transcription:
            transcription_type = TranscriptionType.NONE
        elif transcription and transcription.strip():
            transcription_type = TranscriptionType.MANUAL
        else:
            # No transcription provided - will use empty string, quality may be reduced
            transcription_type = TranscriptionType.NONE

        # Create the cached voice
        metadata = await voice_cache_manager.create_voice(
            voice_name=voice_name,
            audio_data=audio_data,
            sample_rate=sample_rate,
            transcription=final_transcription,
            transcription_type=transcription_type.value,
            overwrite=overwrite,
        )

        voice = VoiceInfo(
            name=metadata["name"],
            created_at=datetime.fromisoformat(metadata["created_at"]),
            transcription=metadata["transcription"],
            transcription_type=TranscriptionType(metadata["transcription_type"]),
            cached_models=voice_cache_manager.get_cached_models(voice_name),
        )

        message = (
            f"Voice '{voice_name}' updated successfully"
            if overwrite and voice_existed
            else f"Voice '{voice_name}' created successfully"
        )

        # Add note about reduced quality if transcription was disabled
        if disable_transcription:
            message += " (using x_vector_only_mode - reduced quality)"

        return VoiceCreateResponse(
            success=True,
            voice=voice,
            message=message
        )

    except Exception as e:
        logger.error(f"Failed to create voice {voice_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create voice: {str(e)}"
        )
    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


@router.delete(
    "/{voice_name}",
    response_model=VoiceDeleteResponse,
    summary="Delete a cached voice",
    description="Deletes a cached voice and all its associated data. "
                "Returns 409 Conflict if the voice is currently being used."
)
async def delete_voice(
    voice_name: str = Path(..., description="Name of the voice to delete")
) -> VoiceDeleteResponse:
    """Delete a cached voice."""
    # Check if voice exists
    if not voice_cache_manager.is_cached_voice(voice_name):
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{voice_name}' not found"
        )

    # Check if voice is in use
    if voice_cache_manager.is_in_use(voice_name):
        raise HTTPException(
            status_code=409,
            detail=f"Voice '{voice_name}' is currently in use"
        )

    try:
        success = voice_cache_manager.delete_voice(voice_name)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Voice '{voice_name}' not found"
            )

        return VoiceDeleteResponse(
            success=True,
            voice_name=voice_name,
            message=f"Voice '{voice_name}' deleted successfully"
        )

    except RuntimeError as e:
        # Voice is in use
        raise HTTPException(
            status_code=409,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to delete voice {voice_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete voice: {str(e)}"
        )


@router.patch(
    "/{voice_name}/rename",
    response_model=VoiceRenameResponse,
    summary="Rename a cached voice",
    description="Renames a cached voice to a new name. The voice directory and "
                "all associated data will be renamed. "
                "Returns 409 Conflict if the voice is currently being used or "
                "if the new name already exists.",
    responses={
        400: {"description": "Invalid new name format"},
        404: {"description": "Voice not found"},
        409: {"description": "Voice in use or new name already exists"},
    }
)
async def rename_voice(
    voice_name: str = Path(..., description="Current name of the voice"),
    request: VoiceRenameRequest = ...,
) -> VoiceRenameResponse:
    """Rename a cached voice."""
    # Check if voice exists
    if not voice_cache_manager.is_cached_voice(voice_name):
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{voice_name}' not found"
        )

    # Check if trying to rename to the same name
    if voice_name == request.new_name:
        return VoiceRenameResponse(
            success=True,
            old_name=voice_name,
            new_name=request.new_name,
            message=f"Voice '{voice_name}' already has this name"
        )

    try:
        voice_cache_manager.rename_voice(voice_name, request.new_name)

        return VoiceRenameResponse(
            success=True,
            old_name=voice_name,
            new_name=request.new_name,
            message=f"Voice '{voice_name}' renamed to '{request.new_name}' successfully"
        )

    except ValueError as e:
        # Invalid name format
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except FileExistsError as e:
        raise HTTPException(
            status_code=409,
            detail=str(e)
        )
    except RuntimeError as e:
        # Voice is in use
        raise HTTPException(
            status_code=409,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to rename voice {voice_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to rename voice: {str(e)}"
        )
