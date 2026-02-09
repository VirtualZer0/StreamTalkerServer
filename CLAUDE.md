# CLAUDE.md

## Project Overview

Stream Talker Server v1.0.0 - FastAPI TTS server with Qwen3-TTS models and voice cloning.

## Commands

```bash
# Development (runs in Docker)
docker compose up --build

# Benchmarks (runs on host)
python benchmarks/run_benchmark.py -d "description" -m 1.7B -r 5
python benchmarks/run_benchmark.py -d "no opts" --no-optimizations --attention sage_attn
python benchmarks/run_benchmark.py -d "partial" --no-cuda-graphs --no-fast-codebook
```

## Docker Build

| Changed | Command |
|---------|---------|
| `server/*.py` only | `docker compose up --build` |
| requirements/Dockerfile | `docker build --target server-prepare -t stream-talker-server-env:latest .` |

**Build Args:** `INCLUDE_MODEL_06B`, `INCLUDE_MODEL_17B` (all default `true`)

```bash
# Minimal: 1.7B only
docker build -t stream-talker-server:minimal --build-arg INCLUDE_MODEL_06B=false .
```

## Architecture

| Component | Purpose |
|-----------|---------|
| `server/models/manager.py` | TTS model management, optimizations |
| `server/api/synthesis.py` | `/synthesize_speech/`, `/change_voice/` |
| `server/api/voices.py` | Voice CRUD + `/clear-prompt-cache` |
| `server/voices/cache.py` | Voice storage at `/data/voices/{name}/` |
| `server/config.py` | Constants, model paths |

**Managers:** `ModelManager`, `SettingsManager`, `VoiceCacheManager` (singletons, lazy loading)

## Model Loading

**Load order:** Load model → Quantize → Apply optimizations (torch.compile, CUDA graphs, codebook)

Quantization is applied **before** optimizations so torch.compile sees the actual quantized dtypes.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `quantization` | `none` | none, int8, float8 (Ampere+) |
| `attention` | `auto` | auto, sage_attn, flex_attn, flash2_attn, sdpa, eager |
| `enable_optimizations` | `true` | Master toggle for all streaming optimizations |
| `torch_compile` | `true` | torch.compile on decoder |
| `cuda_graphs` | `false` | CUDA graphs for decode windows |
| `compile_codebook` | `true` | torch.compile on codebook predictor (~2x faster) |
| `fast_codebook` | `true` | Fast codebook generation (~18% faster) |

**Quantization:** `float8` recommended (best quality/memory).

## Model Status

`GET /models/status` returns a single `status` string per model (not booleans):

| Status | Meaning |
|--------|---------|
| `unloaded` | Not loaded in memory |
| `loading` | Currently being loaded |
| `warming_up` | Loaded, running warmup inference |
| `ready` | Loaded and ready for inference |

## GPU Compatibility

| GPU | dtype | Attention | Quantization |
|-----|-------|-----------|--------------|
| RTX 40xx (Ada) | bf16 | All | All |
| RTX 30xx (Ampere) | bf16 | sage/flex/flash/sdpa | int8, float8 |
| RTX 20xx (Turing) | fp16 (1.7B), bf16 emulated (0.6B) | flash/sdpa only | int8 |

**0.6B always uses bf16** (fp16 causes overflow). Server auto-detects and blocks incompatible features.

## Generation Limits (Hang Prevention)

Fixes for [Qwen-TTS Issue #118](https://github.com/QwenLM/Qwen3-TTS/issues/118) - infinite generation on ~0.5% of inputs.

**Applied at model load time:**

| Setting | Value |
|---------|-------|
| `max_new_tokens` | 2048 |
| `eos_token_id` | `[2150, 2157, 151670, 151673, 151645, 151643]` |
| `repetition_penalty` | 1.05 (adjustable per-request: 1.0-2.0) |

**Note:** `stopping_criteria` passed to `generate_voice_clone()` may be ignored by qwen-tts. The model-level config is the real fix.

## API Usage

```bash
# Single text -> WAV
curl -X POST "http://localhost:7860/synthesize_speech/" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "voice": "ponda"}'

# Batch -> ZIP
curl -X POST "http://localhost:7860/synthesize_speech/" \
  -d '{"text": ["Hello", "World"], "voice": "ponda"}' -o batch.zip

# With options
curl -X POST "http://localhost:7860/synthesize_speech/" \
  -d '{"text": "Hello", "voice": "ponda", "model": "1.7B", "repetition_penalty": 1.3}'
```

## Quick Reference

| Item | Value |
|------|-------|
| Sample rate | 24 kHz |
| Max upload | 200 MB |
| Audio formats | wav, mp3, flac, ogg |
| Auto-unload | 30 min (configurable) |
| Storage | `/data/voices/{name}/`, `/data/settings.json` |

**Voice names:** Support Unicode characters (Latin, Cyrillic, CJK, etc.), digits, underscores, hyphens.

**HTTP 409:** Delete voice in use, create duplicate voice without `overwrite=true`

## Model Unload Behavior

When unload is requested during active inference:
1. Running inference is **aborted immediately** (stops at next token)
2. Partial audio is returned with `X-Audio-Truncated: true` header
3. Model unloads after abort completes

This prevents waiting minutes for long inferences to complete.

## Dependency Notes

**qwen-tts** - Pinned to commit `dc7a8da2b1` of `rekuenkdr/Qwen3-TTS-streaming` fork. The server monkey-patches internal qwen-tts methods:
- `Qwen3TTSForConditionalGeneration.generate()` - captures stopping_criteria
- `Qwen3TTSTalkerForConditionalGeneration.generate()` - injects stopping_criteria
- `Qwen3TTSTokenizerV2CausalTransConvNet` - padding fix (upstream 5f8581d)
- `Qwen3TTSTokenizerV2Model.decode()` + `pad_sequence` - decode padding value fix (upstream 6cafe558)

Breaking changes to these classes will break the server's hang prevention. Streaming optimizations are delegated to the fork's `enable_streaming_optimizations()` API.

**Note:** References to "qwen-tts" in this section refer to the external Python package dependency,
not the project name (which is "Stream Talker Server").

## Release Process

Automated GitHub Actions workflow for releases. See **[docs/release-process.md](docs/release-process.md)** for full details.

**Quick steps:**
1. Update version in `server/__init__.py` and `CHANGELOG.md`
2. Create branch `release/X.Y.Z` and push
3. GitHub Actions builds Docker image and creates release
4. Merge to `main` after success

**Image tags:** `virtualzer0/stream-talker-server:vX.Y.Z` and `:latest`
