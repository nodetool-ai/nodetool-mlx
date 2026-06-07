# nodetool-mlx

High-performance MLX-native nodes for [Nodetool](https://github.com/nodetool-ai/nodetool) on Apple Silicon. This package wraps the community MLX implementations of Whisper, Kokoro/Sesame TTS, and MFlux FLUX.1 image generation so you can run state-of-the-art audio and vision workflows locally on macOS.

## Why nodetool-mlx?

- **Local-first** – keep data on-device by running speech, TTS, and image models without cloud calls
- **Optimised for Apple Silicon** – uses MLX kernels and quantized checkpoints to achieve strong throughput on M-series chips
- **Drop-in nodes** – integrates seamlessly with the Nodetool graph editor and `nodetool-core` runtime

## Provided Nodes

All nodes live under `src/nodetool/nodes/mlx`. Audio nodes wrap the
[`mlx-audio`](https://github.com/Blaizzy/mlx-audio) library.

### Text-to-Speech (`mlx.text_to_speech`)

- `KokoroTTS` – fast multilingual TTS with 54 voice presets
- `SesameTTS` – CSM voice cloning from a reference clip
- `SparkTTS` – controllable speed / pitch / gender presets
- `Qwen3TTS` – multilingual TTS with speaker voices and voice design
- `KittenTTS` – compact, edge-friendly English voices
- `DiaTTS` – dialogue TTS with `[S1]` / `[S2]` speaker tags
- `OuteTTS` – efficient multilingual TTS with optional cloning
- `OmniVoiceTTS` – zero-shot multilingual (646+ languages)
- `MeloTTS` – lightweight VITS2 English accents
- `VoxtralTTS` – Mistral's multilingual TTS with voice presets
- `ChatterboxTTS` – expressive TTS with exaggeration control and cloning
- `HiggsAudioTTS` – conversational TTS with zero-shot cloning
- `LongCatAudioTTS` – diffusion TTS with zero-shot cloning
- `MLXTextToSpeech` – generic node that runs any `mlx-audio` TTS repo id

### Speech-to-Text (`mlx.automatic_speech_recognition`, `mlx.speech_to_text`)

- `Whisper` – MLX Whisper transcription with optional word timestamps
- `Parakeet` – NVIDIA Parakeet high-accuracy multilingual ASR
- `Qwen3ASR` – Alibaba's multilingual ASR with long-form chunking
- `Qwen3ForcedAligner` – word-level timestamp alignment for a known transcript
- `MLXSpeechToText` – generic node that runs any `mlx-audio` STT repo id

### Speech Enhancement (`mlx.speech_enhancement`)

- `DeepFilterNet` – real-time noise suppression (v1/v2/v3) at 48 kHz
- `MossFormer2` – high-quality 48 kHz speech enhancement

### Image (`mlx.text_to_image`, `mlx.image_to_image`)

- `MFlux` and the `MFlux*` family – FLUX.1 / Qwen-Image / Z-Image / FIBO generation
  and editing via the MFlux project (supports quantized models)

### Text (`mlx.text_generation`)

- `TextGeneration` – local LLM text generation via `mlx-lm`


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
