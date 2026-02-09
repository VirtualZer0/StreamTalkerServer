"""API routes module."""

from fastapi import APIRouter

from .environment import router as environment_router
from .models import router as models_router
from .voices import router as voices_router
from .synthesis import router as synthesis_router

# Main API router that includes all sub-routers
api_router = APIRouter()

# Include sub-routers
api_router.include_router(environment_router)
api_router.include_router(models_router)
api_router.include_router(voices_router)
api_router.include_router(synthesis_router)

__all__ = [
    "api_router",
    "environment_router",
    "models_router",
    "voices_router",
    "synthesis_router",
]
