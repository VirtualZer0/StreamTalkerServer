# Changelog

All notable changes to Stream Talker Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-09

### Added
- Initial release of Stream Talker Server
- FastAPI-based TTS server with Qwen3-TTS models (0.6B, 1.7B)
- Dynamic model loading with automatic unloading
- Voice cloning with persistent caching
- Batch synthesis with ZIP archive output
- Multi-language support (10 languages)
- RESTful API for model, voice, and synthesis management
- Docker deployment with GPU support (CUDA 13.0)
- Health checks and environment introspection
- Configurable generation limits to prevent hangs

[1.0.0]: https://github.com/VirtualZer0/StreamTalkerServer/releases/tag/v1.0.0
