# nodetool-mlx

High-performance MLX-native nodes for [Nodetool](https://github.com/nodetool-ai/nodetool) on Apple Silicon. This package wraps the community MLX implementations of Whisper, Kokoro/Sesame TTS, and MFlux FLUX.1 image generation so you can run state-of-the-art audio and vision workflows locally on macOS.

## Why nodetool-mlx?

- **Local-first** – keep data on-device by running speech, TTS, and image models without cloud calls
- **Optimised for Apple Silicon** – uses MLX kernels and quantized checkpoints to achieve strong throughput on M-series chips
- **Drop-in nodes** – integrates seamlessly with the Nodetool graph editor and `nodetool-core` runtime

## Provided Nodes

All nodes live under `src/nodetool/nodes/mlx`:

- `mlx.whisper.MLXWhisper` – streaming speech-to-text using MLX Whisper checkpoints
- `mlx.tts.TTS` – Kokoro and Sesame text-to-speech with optional chunked audio streaming
- `mlx.mflux.ImageGeneration` – FLUX.1 image generation via the MFlux project (supports quantized models)
- `mlx.text_to_music.ACEStepMusicGeneration` – local text-to-music generation with [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) (prompt + lyrics → full songs)
- `mlx.text_to_music.ACEStepSongPlanner` – use the ACE-Step 5Hz language model to turn an idea into a caption, lyrics and musical metadata

### ACE-Step 1.5 (music generation)

The ACE-Step nodes wrap the official [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5)
project, which uses MLX for the diffusion transformer, VAE and language model on
Apple Silicon. ACE-Step 1.5 is **not** distributed on PyPI (the `ace-step` package
there is the older 1.0 release), so it must be installed separately:

```bash
git clone https://github.com/ace-step/ACE-Step-1.5.git
cd ACE-Step-1.5
uv sync   # installs the macOS/MLX dependencies
```

Install it into the same environment as `nodetool-mlx`. If the package is missing,
the nodes raise a clear error pointing back to the repository.

Checkpoints are downloaded automatically from HuggingFace on first use into
`~/.cache/nodetool/acestep/checkpoints` (override with the `ACESTEP_CHECKPOINTS_DIR`
environment variable). The main bundle `ACE-Step/Ace-Step1.5` includes the VAE, the
turbo DiT and the 1.7B planner LM; additional DiT and LM checkpoints are listed as
recommended models on each node.


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
