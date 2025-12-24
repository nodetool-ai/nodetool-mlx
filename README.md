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

### Memory Safety and Low-Memory Mode

All MFlux image generation nodes include built-in memory safety features to prevent macOS system freezes from unified memory exhaustion:

#### Low-Memory Mode (VAE Tiling)

Enable VAE tiling to reduce peak memory usage by approximately 4x at the cost of potential seams in the output:

```python
node = ImageGeneration(
    prompt="A vivid concept art piece",
    width=2048,
    height=2048,
    low_memory=True,           # Enable VAE tiling
    vae_tiling_split="horizontal"  # or "vertical"
)
```

**When to use low-memory mode:**
- Generating high-resolution images (>1536x1536) on 8-16GB RAM Macs
- Running multiple models simultaneously
- When you encounter memory warnings or system slowdowns

#### Automatic Memory Preflight Checks

Before each generation, the nodes perform a conservative memory check:
- Estimates memory needed based on resolution, steps, and quantization
- Requires at least 10% system memory headroom
- Fails fast with detailed error messages if insufficient memory

**Error messages include:**
- Current memory availability
- Estimated job memory usage
- Specific suggestions (enable low-memory mode, reduce resolution, etc.)

#### Memory Best Practices

1. **Start with quantized 4-bit models** – they use ~70% less memory than fp16
2. **Enable low-memory mode for large resolutions** – 2048x2048 or higher
3. **Monitor system memory** – keep Activity Monitor open during first runs
4. **Close other applications** – especially browsers and IDEs during generation
5. **Prefer schnell models** – they require fewer steps and less memory than dev models

## Development

Run tests and lint checks before submitting PRs:

```bash
pytest -q
ruff check .
black --check .
```

Please open issues or pull requests for bug fixes, new MLX models, or performance improvements. Contributions are welcome!
