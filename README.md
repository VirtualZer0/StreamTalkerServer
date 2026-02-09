# StreamTalkerServer

🌐 **Language:** English | [Русский](docs/ru/README.md)

**AI text-to-speech server powered by Qwen3-TTS**

---

## 📖 Description

StreamTalkerServer is an AI-powered text-to-speech server built on Qwen3-TTS models. It receives text via HTTP API, synthesizes natural-sounding speech using AI, and returns WAV audio files. The server supports voice cloning, multiple languages, and batch processing.

Designed to work with [StreamTalkerClient](https://github.com/VirtualZer0/StreamTalkerClient) for reading stream chat aloud, but can also be used as a standalone TTS API.

**Hardware requirements:**
- NVIDIA GPU with **8 GB+ VRAM** (16 GB+ recommended for running both models)
- CUDA-compatible drivers

---

## 🚀 Installation

This guide will walk you through installing StreamTalkerServer using Docker — the recommended and easiest method.

### Step 1: Check your GPU

StreamTalkerServer requires an NVIDIA GPU with at least 8 GB of video memory (VRAM).

**Windows — how to check:**
1. Press `Win + X` → click **Device Manager**
2. Expand **Display adapters** — you should see an NVIDIA GPU (e.g., "NVIDIA GeForce RTX 3060")
3. To check VRAM: right-click the desktop → **Display settings** → **Advanced display** → **Display adapter properties** → look at "Dedicated Video Memory"

**Linux — how to check:**
```bash
nvidia-smi
```
This will show your GPU model and available memory. If the command is not found, NVIDIA drivers are not installed.

### Step 2: Install NVIDIA drivers

If you already have recent NVIDIA drivers, you can skip this step.

1. Go to the [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx) page
2. Select your GPU model and operating system
3. Download and install the driver
4. Restart your computer

### Step 3: Install Docker

**Windows:**
1. Download [Docker Desktop](https://www.docker.com/products/docker-desktop/) and install it
2. During installation, make sure **"Use WSL 2"** is checked
3. If prompted, install WSL 2 by running in PowerShell (as Administrator):
   ```powershell
   wsl --install
   ```
4. Restart your computer
5. Launch Docker Desktop and wait for it to start (whale icon in the system tray should stop animating)
6. Verify installation — open a terminal and run:
   ```bash
   docker --version
   ```

**Linux (Ubuntu/Debian):**
```bash
# Install Docker Engine
sudo apt update
sudo apt install docker.io docker-compose-v2
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
# Log out and back in for group changes to take effect
```

**Linux (Fedora):**
```bash
sudo dnf install docker docker-compose
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
# Log out and back in for group changes to take effect
```

### Step 4: Install NVIDIA Container Toolkit

This allows Docker to access your GPU.

**Windows:**
Docker Desktop with WSL 2 backend supports GPU passthrough automatically if your NVIDIA drivers are up to date. No extra steps needed.

**Linux:**
```bash
# Add NVIDIA Container Toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt update
sudo apt install nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Step 5: Create docker-compose.yml

Create a file named `docker-compose.yml` in any folder with the following content:

```yaml
name: stream-talker-server

services:
  stream-talker-server:
    image: virtualzer0/stream-talker-server:latest
    ports:
      - "7860:7860"
    volumes:
      - stream-talker-data:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

volumes:
  stream-talker-data:
```

### Step 6: Start the server

Open a terminal in the folder where you created `docker-compose.yml` and run:

```bash
docker compose up -d
```

> **First launch will take a while** — Docker will download the server image including AI models (~10-15 GB). This only happens once. Subsequent starts will be fast.

To see logs (useful to check progress on first launch):
```bash
docker compose logs -f
```

Press `Ctrl+C` to stop viewing logs (the server keeps running).

### Step 7: Verify it's running

Open your browser and go to:

```
http://localhost:7860/health
```

You should see a JSON response indicating the server is healthy. The server is now ready to accept TTS requests from StreamTalkerClient.

---

## ❓ FAQ

<details>
<summary><b>Docker won't start / WSL 2 errors on Windows</b></summary>

- Make sure **virtualization** is enabled in your BIOS (usually called "Intel VT-x" or "AMD-V")
- Install WSL 2 by running in PowerShell (as Administrator): `wsl --install`
- Restart your computer after enabling virtualization or installing WSL 2
- Make sure Docker Desktop is set to use the WSL 2 backend (Settings → General → "Use WSL 2 based engine")
</details>

<details>
<summary><b>GPU not detected in Docker</b></summary>

- Make sure your NVIDIA drivers are up to date
- On Linux: verify NVIDIA Container Toolkit is installed (`nvidia-ctk --version`)
- Run `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` to test GPU access in Docker
- On Windows: make sure Docker Desktop is using WSL 2 backend, not Hyper-V
</details>

<details>
<summary><b>Out of memory / CUDA OOM errors</b></summary>

- Close other applications using the GPU (games, other AI tools, browser hardware acceleration)
- Use the smaller 0.6B model instead of 1.7B — it requires less VRAM
- Try loading the model with `int8` quantization to reduce memory usage
- Check available VRAM with `nvidia-smi`
</details>

<details>
<summary><b>Server takes a long time to start the first time</b></summary>

This is normal. On first launch, Docker downloads the server image including AI models (~10-15 GB). Monitor progress with `docker compose logs -f`. Subsequent starts will be much faster.
</details>

<details>
<summary><b>Port 7860 already in use</b></summary>

Another application is using port 7860. Change the port in `docker-compose.yml`:
```yaml
ports:
  - "7861:7860"  # Use port 7861 instead
```
Then update the server URL in StreamTalkerClient to `http://localhost:7861`.
</details>

<details>
<summary><b>Model loading fails / insufficient VRAM</b></summary>

- The 1.7B model requires ~4-6 GB VRAM, the 0.6B model requires ~2-3 GB
- Try using quantization (`int8` or `float8`) to reduce memory usage
- Make sure no other GPU-heavy applications are running
- If you have less than 8 GB VRAM, use only the 0.6B model
</details>

<details>
<summary><b>How to update the server</b></summary>

```bash
docker compose pull
docker compose up -d
```
This will download the latest version and restart the server.
</details>

<details>
<summary><b>Voice quality issues</b></summary>

- When cloning a voice, always provide a transcription of the reference audio — this significantly improves quality
- Use the 1.7B model for better quality (0.6B is faster but lower quality)
- Make sure the reference audio is clear, without background noise, and 5-15 seconds long
</details>

---

## 🔗 Related Projects

### [StreamTalkerClient](https://github.com/VirtualZer0/StreamTalkerClient)

Desktop application for reading Twitch and VK Play chat messages aloud. Connects to StreamTalkerServer for AI voice synthesis. Available for Windows and Linux.

---

## Features

- **Multi-model support** — Both 0.6B and 1.7B TTS models available
- **Batch synthesis** — Process multiple texts in one request, returns ZIP archive
- **Dynamic model loading** — TTS models loaded on-demand
- **Automatic unloading** — Inactive models auto-unload to free GPU memory
- **Voice cloning** — Clone any voice from a short reference audio clip
- **Persistent voice cache** — Voice prompts cached to disk for faster reuse
- **Voice conversion** — Transform text to speech with different voices
- **Multi-language support** — 10 languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

---

## Quick Start (Development)

### Using Docker Compose

```bash
# Clone and build
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Using Docker

```bash
# Build full image (all models)
docker build -t stream-talker-server .

# Build minimal image (only 1.7B) - faster build, smaller image
docker build -t stream-talker-server:minimal \
  --build-arg INCLUDE_MODEL_06B=false .

# Run with GPU and persistent storage
docker run --gpus all -p 7860:7860 -v ./data:/data stream-talker-server
```

**Build Args:**
| Arg | Default | Description |
|-----|---------|-------------|
| `INCLUDE_MODEL_06B` | `true` | Include 0.6B TTS model |
| `INCLUDE_MODEL_17B` | `true` | Include 1.7B TTS model |

Note: Tokenizer is always included (required for TTS).

### Local Development

```bash
# Create virtual environment (Python 3.10+)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt

# Optional: Install FlashAttention 2
pip install flash-attn --no-build-isolation

# Run the server
python -m uvicorn server.main:app --host 0.0.0.0 --port 7860
```

---

## API Reference

### Health & Status

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Server info |
| `/health` | GET | Health check |
| `/environment` | GET | Environment and resource information |

### Model Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/models/status` | GET | Get status of all TTS models |
| `/models/{model_id}/load` | POST | Load a model into memory (`0.6B` or `1.7B`) |
| `/models/{model_id}/unload` | POST | Unload a model from memory |
| `/models/{model_id}/auto-unload` | POST | Configure auto-unload timeout |

### Voice Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/voices` | GET | List all cached voices (with cache status) |
| `/voices/{voice_name}` | GET | Get voice information |
| `/voices/{voice_name}` | POST | Create/update a cached voice |
| `/voices/{voice_name}` | DELETE | Delete a cached voice |
| `/voices/{voice_name}/rename` | PATCH | Rename a cached voice |
| `/voices/clear-prompt-cache` | POST | Clear voice prompt cache (memory + disk) |

### Speech Synthesis

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/synthesize_speech/` | POST | TTS with specified voice (single or batch) |
| `/change_voice/` | POST | Synthesize text with a different voice |

---

## Endpoint Details

### POST /models/{model_id}/load

Load a TTS model into GPU memory.

**Path Parameters:**
- `model_id`: `"0.6B"` or `"1.7B"`

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `warmup` | string | `none` | Warmup mode: `none`, `single`, or `batch` |
| `warmup_lang` | string | - | Language for warmup phrases |
| `warmup_voice` | string | - | Voice name for warmup |
| `warmup_timeout` | int | 120 | Timeout in seconds for warmup |
| `quantization` | string | `none` | Quantization: `none`, `int8`, or `float8` |
| `attention` | string | `auto` | Attention: `auto`, `sage_attn`, `flex_attn`, `flash2_attn`, `sdpa`, `eager` |
| `enable_optimizations` | bool | `true` | Master toggle for streaming optimizations |
| `torch_compile` | bool | `true` | torch.compile on decoder |
| `cuda_graphs` | bool | `true` | CUDA graphs for decode windows |
| `compile_codebook` | bool | `true` | torch.compile on codebook predictor |
| `fast_codebook` | bool | `true` | Fast codebook generation |
| `force_cpu` | bool | `false` | Force CPU loading (very slow) |

**Response:**
```json
{
  "success": true,
  "model_id": "1.7B",
  "message": "Model 1.7B loaded successfully"
}
```

---

### POST /synthesize_speech/

Generate speech from text. Supports both single text and batch processing.

**Request Body (JSON):**
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string or array | Yes | - | Single text or array of texts for batch |
| `voice` | string | Yes | - | Voice name |
| `speed` | float | No | 1.0 | Speed multiplier (0.5-2.0) |
| `model` | string | No | `"1.7B"` | Model to use: `0.6B` or `1.7B` |
| `language` | string | No | `"Auto"` | Target language for synthesis |
| `do_sample` | bool | No | `true` | Use sampling for varied output |
| `temperature` | float | No | 0.7 | Sampling temperature (if do_sample=true) |
| `max_new_tokens` | int | No | 2000 | Max tokens (~50 tokens = 1s audio) |
| `repetition_penalty` | float | No | 1.05 | Repetition penalty (1.0-2.0) |

**Supported Languages:**
`Auto`, `Chinese`, `English`, `Japanese`, `Korean`, `German`, `French`, `Russian`, `Portuguese`, `Spanish`, `Italian`

**Single Text Request:**
```json
{
  "text": "Hello, this is a test.",
  "voice": "demo_speaker0",
  "language": "English"
}
```
**Response:** `audio/wav` file

**Batch Request:**
```json
{
  "text": ["Hello world", "How are you?", "Goodbye"],
  "voice": "demo_speaker0"
}
```
**Response:** `application/zip` file containing `0000.wav`, `0001.wav`, `0002.wav`

**Response Headers:**
- `X-Elapsed-Time`: Processing time in seconds
- `X-Device-Used`: GPU/CPU device
- `X-Model-Used`: Model used for synthesis
- `X-Batch-Count`: Number of files in batch (batch only)

---

### POST /voices/{voice_name}

Create or update a cached voice.

**Form Parameters:**
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | file | Yes | - | Audio file (wav, mp3, flac, ogg) |
| `transcription` | string | No | - | Transcription (recommended for best quality) |
| `overwrite` | boolean | No | false | Overwrite existing voice |
| `disable_transcription` | boolean | No | false | Skip transcription (reduced quality) |

**Transcription Types:**
| Type | Description |
|------|-------------|
| `MANUAL` | User provided transcription text manually |
| `NONE` | No transcription - uses x_vector_only_mode (reduced quality) |

---

## Usage Examples

### Basic TTS

```bash
curl -X POST "http://localhost:7860/synthesize_speech/" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "demo_speaker0"}' \
  --output output.wav
```

### TTS with Specific Voice

```bash
curl -X POST "http://localhost:7860/synthesize_speech/" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "voice": "demo_speaker0", "speed": 1.0}' \
  --output output.wav
```

### Batch TTS

```bash
# Returns ZIP file with numbered WAV files
curl -X POST "http://localhost:7860/synthesize_speech/" \
  -H "Content-Type: application/json" \
  -d '{"text": ["Hello world", "How are you?", "Goodbye!"], "voice": "my_voice"}' \
  --output batch.zip

# Extract the files
unzip batch.zip
# Creates: 0000.wav, 0001.wav, 0002.wav
```

### Create Cached Voice

```bash
# With transcription (recommended for best quality)
curl -X POST "http://localhost:7860/voices/my_voice" \
  -F "file=@/path/to/voice_sample.mp3" \
  -F "transcription=Hello, this is my voice sample."

# Without transcription (faster, reduced quality)
curl -X POST "http://localhost:7860/voices/my_voice" \
  -F "file=@/path/to/voice_sample.mp3" \
  -F "disable_transcription=true"
```

### Rename Voice

```bash
curl -X PATCH "http://localhost:7860/voices/my_voice/rename" \
  -H "Content-Type: application/json" \
  -d '{"new_name": "my_new_voice"}'
```

### Clear Prompt Cache

```bash
# Clear cache for a specific voice
curl -X POST "http://localhost:7860/voices/clear-prompt-cache?voice_name=my_voice"

# Clear all voice caches
curl -X POST "http://localhost:7860/voices/clear-prompt-cache"
```

### Load Model with Options

```bash
# Basic load (optimizations enabled by default)
curl -X POST "http://localhost:7860/models/1.7B/load"

# Load without optimizations
curl -X POST "http://localhost:7860/models/1.7B/load?enable_optimizations=false"

# Load with warmup and quantization
curl -X POST "http://localhost:7860/models/1.7B/load?warmup=batch&quantization=int8"

# Load with selective optimizations
curl -X POST "http://localhost:7860/models/1.7B/load?cuda_graphs=false&fast_codebook=false"
```

### Configure Auto-Unload

```bash
# Set 60-minute timeout
curl -X POST "http://localhost:7860/models/1.7B/auto-unload" \
  -H "Content-Type: application/json" \
  -d '{"minutes": 60}'

# Disable auto-unload
curl -X POST "http://localhost:7860/models/1.7B/auto-unload" \
  -H "Content-Type: application/json" \
  -d '{"minutes": 0}'
```

---

## Directory Structure

```
.
├── server/                 # Server package
│   ├── main.py             # FastAPI app entry point
│   ├── config.py           # Configuration constants
│   ├── api/                # API routes
│   │   ├── environment.py  # Environment info endpoint
│   │   ├── models.py       # Model management endpoints
│   │   ├── voices.py       # Voice management endpoints
│   │   └── synthesis.py    # TTS & transcription endpoints
│   ├── models/             # Model management
│   │   └── manager.py      # TTS ModelManager class
│   ├── voices/             # Voice caching
│   │   └── cache.py        # VoiceCacheManager class
│   └── utils/              # Utilities
│       └── audio.py        # Audio processing
├── benchmarks/             # Performance benchmarks
│   └── run_benchmark.py    # Benchmark runner
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Docker Compose config
└── requirements.txt        # Python dependencies
```

## Storage

### Persistent Data (`/data`)

```
/data/
├── settings.json      # Auto-unload settings
└── voices/            # Cached voice data
    └── {voice_name}/
        ├── metadata.json
        ├── reference.wav
        ├── prompt_0.6B.pkl
        └── prompt_1.7B.pkl
```

---

## Models

| Model | HuggingFace ID | Size | VRAM |
|-------|----------------|------|------|
| 0.6B | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | 0.6B params | ~2-3 GB |
| 1.7B | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 1.7B params | ~4-6 GB |

Models are pre-downloaded during Docker build and stored in `/root/.cache/`.

---

## Requirements

- **GPU**: CUDA-compatible GPU with 8GB+ VRAM (16GB+ for both models)
- **CUDA**: 12.4+ (compatible with most modern NVIDIA drivers)
- **Python**: 3.10+

---

## Configuration

### Default Settings

| Setting | Default |
|---------|---------|
| Default model | `1.7B` |
| Auto-unload timeout | 30 minutes |
| Max upload size | 200 MB |
| Sample rate | 24 kHz |
| Inference timeout | 120 seconds |

### Model Loading Defaults

Streaming optimizations are enabled by default (delegated to fork):

| Parameter | Default |
|-----------|---------|
| warmup | `none` |
| quantization | `none` |
| attention | `auto` |
| enable_optimizations | `true` |
| torch_compile | `true` |
| cuda_graphs | `true` |
| compile_codebook | `true` |
| fast_codebook | `true` |

---

## Edge Cases

| Situation | Behavior |
|-----------|----------|
| Request to unloaded model | Auto-load, wait, execute |
| Model is loading, new request | Wait for loading to complete |
| Unload model while in use | Inference aborted, partial audio returned, then unload |
| Delete voice while in use | Error 409 |
| Upload voice with existing name | Error 409 if `overwrite=false` |
| Empty text in batch | Skipped (not included in output) |
| `auto_unload_minutes = 0` | Auto-unload disabled |

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** for the AI voice synthesis engine
- **[Qwen3-TTS-streaming](https://github.com/rekuenkdr/Qwen3-TTS-streaming)** for the streaming optimizations fork
