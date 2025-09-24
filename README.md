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

Their DSL wrappers are available under `src/nodetool/dsl/mlx` for use in generated workflows.

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
2. Run `nodetool package scan` to generate metadata and DSL bindings
3. (Optional) `nodetool codegen` to refresh typed DSL wrappers
4. Build workflows either in the Nodetool UI or through Python DSL scripts using the `mlx` namespace

Example (Python DSL):

```python
from nodetool.dsl.mlx import ImageGeneration

node = ImageGeneration(prompt="A retrofuturistic skyline at dusk", steps=6)
```

## Development

Run tests and lint checks before submitting PRs:

```bash
pytest -q
ruff check .
black --check .
```

Please open issues or pull requests for bug fixes, new MLX models, or performance improvements. Contributions are welcome!
