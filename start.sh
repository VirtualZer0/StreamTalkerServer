#!/bin/bash

# Stream Talker Server Startup Script
# ================================
# This script initializes and starts the TTS server with proper configuration.

set -e

# Change to the app directory
cd /app

# Add the current directory to PYTHONPATH for module imports
export PYTHONPATH="$PYTHONPATH:/app"

# Ensure /data directories exist (for non-volume mounts)
mkdir -p /data/voices /data/cache 2>/dev/null || true

echo "========================================"
echo "Stream Talker Server v1.0.0"
echo "========================================"
echo "Starting server..."
echo "- Host: 0.0.0.0"
echo "- Port: 7860"
echo "- Data directory: /data"
echo "- Models: 0.6B, 1.7B (lazy loading)"
echo "========================================"

# Start the FastAPI server with uvicorn in the foreground
# Using exec to replace shell process - ensures proper signal handling and log capture
exec python -m uvicorn server.main:app --host 0.0.0.0 --port 7860
