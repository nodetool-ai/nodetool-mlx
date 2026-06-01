# nodetool-mlx

High-performance MLX-native nodes for [Nodetool](https://github.com/nodetool-ai/nodetool) on Apple Silicon. This package wraps the community MLX implementations of Whisper, Kokoro/Sesame TTS, MFlux FLUX.1 image generation, and Stability AI's Stable Audio 3 so you can run state-of-the-art audio and vision workflows locally on macOS.

## Why nodetool-mlx?

- **Local-first** – keep data on-device by running speech, TTS, and image models without cloud calls
- **Optimised for Apple Silicon** – uses MLX kernels and quantized checkpoints to achieve strong throughput on M-series chips
- **Drop-in nodes** – integrates seamlessly with the Nodetool graph editor and `nodetool-core` runtime

## Provided Nodes

All nodes live under `src/nodetool/nodes/mlx`:

- `mlx.whisper.MLXWhisper` – streaming speech-to-text using MLX Whisper checkpoints
- `mlx.tts.TTS` – Kokoro and Sesame text-to-speech with optional chunked audio streaming
- `mlx.mflux.ImageGeneration` – FLUX.1 image generation via the MFlux project (supports quantized models)
- `mlx.text_to_audio.StableAudio3` – text-to-audio music & sound effects with [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) (44.1 kHz stereo)
- `mlx.text_to_audio.StableAudio3AudioToAudio` – prompt-guided variations of an input clip
- `mlx.text_to_audio.StableAudio3Inpaint` – regenerate a time range inside an audio clip

### Stable Audio 3

The Stable Audio 3 nodes run Stability AI's optimized MLX implementation (no PyTorch at
runtime), vendored under `nodetool.mlx.stable_audio_3` (MIT licensed — see that folder's
`LICENSE` and `NOTICE.md`). Three DiT variants are available via the **model** field:

- `sm-music` (50M) – fast music generation
- `sm-sfx` (50M) – sound effects
- `medium` (1.4B) – higher-fidelity music

Weights are pulled on demand from the Hugging Face repo
[`stabilityai/stable-audio-3-optimized`](https://huggingface.co/stabilityai/stable-audio-3-optimized)
(only the `MLX/*` files) and cached locally; you can also pre-download them from the
Models Manager.


## Requirements

- macOS 14+ on Apple Silicon (MLX currently supports Apple hardware only)
- Python 3.11
- [nodetool-core](https://github.com/nodetool-ai/nodetool-core) v0.6.0+
- Required MLX checkpoints managed via the Nodetool Models Manager (see [_Managing Models_](#managing-models))

## Installation

### From the Nodetool UI

1. Open Nodetool → **Tools ▸ Packages**
2. Install the `nodetool-mlx` pack from the package registry
3. Nodetool will handle dependencies and expose the MLX nodes in the graph editor once installed

### From source (development)

```bash
git clone https://github.com/nodetool-ai/nodetool-mlx.git
cd nodetool-mlx
uv pip install -e .
uv pip install -r requirements-dev.txt
```

If you prefer Poetry or pip, install the project the same way—just ensure dependencies are resolved against Python 3.11.

## Managing Models

All MLX nodes rely on locally cached checkpoints. The recommended way to download and update them is through the **Models Manager** built into Nodetool:

1. Open Nodetool → **Menu ▸ Models**
2. Select the `mlx` tab to view the recommended checkpoints for each node
3. Click **Download** for the models you plan to use; Nodetool stores them in the Hugging Face cache automatically
4. The UI will keep track of model availability and prompt you when updates are available

Advanced users can still seed the Hugging Face cache manually, but using the UI integration ensures consistent paths and avoids missing-model errors in workflows.

## Usage

1. Install `nodetool-core` and this package in the same environment
2. Run `nodetool package scan` to generate package metadata
3. Build workflows in the Nodetool UI using the `mlx` nodes

## Development

Run tests and lint checks before submitting PRs:

```bash
pytest -q
ruff check .
black --check .
```

Please open issues or pull requests for bug fixes, new MLX models, or performance improvements. Contributions are welcome!
