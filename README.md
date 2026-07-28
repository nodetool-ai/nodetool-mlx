# nodetool-mlx

[![CI](https://github.com/nodetool-ai/nodetool-mlx/actions/workflows/ci.yml/badge.svg)](https://github.com/nodetool-ai/nodetool-mlx/actions/workflows/ci.yml)

High-performance MLX-native nodes for [Nodetool](https://github.com/nodetool-ai/nodetool) on Apple Silicon. This package wraps the community MLX implementations of Whisper, Kokoro/Sesame TTS, MFlux FLUX.1 image generation, and Stability AI's Stable Audio 3 so you can run state-of-the-art audio and vision workflows locally on macOS.

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

### Image-to-Text (`mlx.image_to_text`)

- `MLXVisionLanguage` – image captioning, visual Q&A, and OCR via MLX vision-language models (Qwen3-VL, Gemma 4)

### Text-to-Audio (`mlx.text_to_audio`)

- `StableAudio3` – text-to-audio music & sound effects with [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) (44.1 kHz stereo)
- `StableAudio3AudioToAudio` – prompt-guided variations of an input clip
- `StableAudio3Inpaint` – regenerate a time range inside an audio clip

### Text-to-Music (`mlx.text_to_music`)

- `ACEStepMusicGeneration` – local text-to-music generation with [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) (prompt + lyrics → full songs)
- `ACEStepSongPlanner` – use the ACE-Step 5Hz language model to turn an idea into a caption, lyrics and musical metadata

### Text (`mlx.text_generation`)

- `TextGeneration` – local LLM text generation via `mlx-lm`

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
2. Run `nodetool-pkg scan --write --enrich` to generate package metadata
3. Build workflows in the Nodetool UI using the `mlx` nodes

## Development

Run tests and lint checks before submitting PRs:

```bash
pytest -q
ruff check .
black --check .
```

Most of the suite is written to run on any platform: the MLX runtimes are
imported lazily and the node logic is exercised against mocks. The handful of
tests that need `mflux` or `mlx-audio` skip themselves when those Apple Silicon
packages are unavailable, so a Linux run reports skips rather than failures.

If you change a node's `get_recommended_models()`, regenerate the committed
package metadata:

```bash
nodetool-pkg scan --write --enrich
```

`tests/test_package_metadata.py` fails when the two drift apart — the Nodetool
UI reads that file to populate the model picker, so a stale entry becomes a
model users cannot select or a download offer for a repository that no longer
exists.

### Continuous integration

`.github/workflows/ci.yml` runs on every pull request:

| Job | Runner | What it covers |
| --- | --- | --- |
| Lint and format | ubuntu | `ruff check .` and `black --check .` |
| Test (linux) | ubuntu, Python 3.11 + 3.12 | the platform-independent suite, installed with `--no-deps` |
| Test (macOS) | macos-14 (Apple Silicon) | the full suite against the real MLX stack |
| Build wheel | ubuntu | `python -m build` plus `twine check` |

Please open issues or pull requests for bug fixes, new MLX models, or performance improvements. Contributions are welcome!
