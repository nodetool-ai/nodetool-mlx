"""MLX unified provider implementation.

This provider integrates multiple MLX libraries to provide a complete multi-modal
AI provider for Apple Silicon:

- mlx-lm: Language models for chat and text generation
- mlx-vlm: Vision-language models for multimodal understanding
- mlx-audio: Text-to-speech audio generation
- mflux: FLUX models for image generation

The implementation keeps lazy references to the MLX runtimes because importing these
libraries requires a Metal-capable environment. When a capability is first used, we
import the required library, load the configured model, and execute the task.

The provider supports:
- Chat with tool calling conventions
- Vision and audio understanding
- Text-to-speech generation
- Text-to-image and image-to-image generation
"""

import ast
import asyncio
import base64
import json
import logging
import inspect
from pathlib import Path
import threading
from dataclasses import dataclass
import time
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Iterable,
    List,
    Sequence,
)
from io import BytesIO
from urllib.parse import urlparse, unquote
import os
import sys
import tempfile
from nodetool.media.audio.audio_helpers import convert_audio_to_standard_format
import numpy as np
from mlx import nn
import mlx_lm
from mlx_lm.generate import stream_generate
from mlx_lm.tokenizer_utils import TokenizerWrapper
import mlx_vlm
import mlx_audio
import mlx_lm.sample_utils
import mlx_vlm.prompt_utils
import mlx_vlm.utils
import mlx_audio.tts.utils
import mlx_audio.tts.generate

# mflux imports are lazy - only imported when image generation is used
# This allows the MLX provider to be registered even if mflux isn't installed

from nodetool.providers.base import BaseProvider, register_provider
from nodetool.metadata.tool_types import Tool
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.ml.core.model_manager import ModelManager
from nodetool.metadata.types import (
    ASRModel,
    Message,
    Provider,
    ToolCall,
    TTSModel,
    ASRModel,
    MessageContent,
    MessageTextContent,
    MessageImageContent,
    MessageAudioContent,
    ImageRef,
    AudioRef,
    LanguageModel,
    ImageModel,
)
from nodetool.workflows.types import Chunk
from nodetool.io.uri_utils import fetch_uri_bytes_and_mime
from nodetool.providers.types import ImageBytes, TextToImageParams, ImageToImageParams
from nodetool.workflows.processing_context import ProcessingContext

import PIL.Image
from pydub import AudioSegment  # type: ignore
from nodetool.integrations.huggingface.huggingface_models import (
    HF_FAST_CACHE,
    get_mlx_image_models_from_hf_cache,
    get_mlx_language_models_from_hf_cache,
)
from nodetool.mlx.flux_model_loader import (
    load_flux_model,
)

log = get_logger(__name__)
log.setLevel(logging.DEBUG)


DEFAULT_MLX_MODEL = "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"


# Simple in-memory TTL cache for loaded MLX models: 5 minutes
_CACHE_TTL_SECONDS: int = 300
_MODEL_CACHE: dict[str, tuple[nn.Module, TokenizerWrapper, float]] = {}
_MODEL_CACHE_LOCK = threading.Lock()

# Separate cache for VLM models (mlx-vlm)
_VLM_MODEL_CACHE: dict[str, tuple[nn.Module, Any, Any, float]] = {}
_VLM_MODEL_CACHE_LOCK = threading.Lock()

# Separate cache for TTS models (mlx-audio)
_TTS_MODEL_CACHE: dict[str, tuple[Any, float]] = {}
_TTS_MODEL_CACHE_LOCK = threading.Lock()


@register_provider(Provider.MLX)
class MLXProvider(BaseProvider):
    """Unified multi-modal provider for Apple Silicon using MLX.

    This provider exposes a complete AI interface leveraging multiple MLX libraries:
    - Language models (mlx-lm) for chat and text generation
    - Vision models (mlx-vlm) for multimodal understanding
    - Audio models (mlx-audio) for text-to-speech
    - Image models (mflux) for FLUX-based image generation

    All libraries are lazily imported to allow running in environments where they
    are not available at import time. The provider supports streaming, tool calling,
    and all standard Nodetool provider capabilities.
    """

    provider: Provider = Provider.MLX

    def __init__(
        self,
        secrets: dict[str, str] = {},
        adapter_path: str | None = None,
        tokenizer_config: dict[str, Any] | None = None,
        sampler_defaults: dict[str, Any] | None = None,
        lazy_load: bool | None = None,
    ) -> None:
        super().__init__(secrets=secrets)

        env = Environment.get_environment()
        self.adapter_path = adapter_path or env.get("MLX_ADAPTER_PATH")

        self._load_lock = asyncio.Lock()
        self._generation_lock = asyncio.Lock()  # Serialize generation calls

        self._vlm_load_lock = asyncio.Lock()
        self._vlm_generation_lock = asyncio.Lock()  # Serialize VLM generation calls

        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_requests": 0,
            "total_images": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def generate_message(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Tool] = (),
        max_tokens: int = 8192,
        context_window: int = 4096,
        response_format: dict | None = None,
        **kwargs: Any,
    ) -> Message:
        """Stream a single assistant message, collecting content and tool calls."""
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        async for item in self._stream_chat(
            messages,
            model,
            tools,
            max_tokens=max_tokens,
            context_window=context_window,
            response_format=response_format,
            **kwargs,
        ):
            if isinstance(item, ToolCall):
                tool_calls.append(item)
            elif isinstance(item, Chunk):
                if item.content:
                    content_parts.append(item.content)
                if item.done:
                    break

        message = Message(
            role="assistant",
            content="".join(content_parts) if content_parts else None,
            tool_calls=tool_calls or None,
            provider=self.provider,
            model=model,
        )
        return message

    async def generate_messages(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Tool] = (),
        max_tokens: int = 8192,
        context_window: int = 4096,
        response_format: dict | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Chunk | ToolCall]:
        async for item in self._stream_chat(
            messages,
            model,
            tools,
            max_tokens=max_tokens,
            context_window=context_window,
            response_format=response_format,
            **kwargs,
        ):
            yield item

    def get_context_length(self, model: str) -> int:
        # Most MLX converted instruct models default to 8k context unless
        # otherwise specified.
        return 8192

    async def get_available_language_models(self) -> List[LanguageModel]:
        """
        Get available MLX models.

        Returns MLX-converted models available in the local HuggingFace cache.
        Always returns models (doesn't check if MLX is available).

        Returns:
            List of LanguageModel instances for MLX
        """
        models = await get_mlx_language_models_from_hf_cache()
        fibo_vlm_repo = "briaai/FIBO-vlm"
        if await self._resolve_cached_repo_path(fibo_vlm_repo) is not None and not any(
            model.id == fibo_vlm_repo for model in models
        ):
            models.append(
                LanguageModel(
                    provider=Provider.MLX,
                    id=fibo_vlm_repo,
                    name="FIBO VLM",
                    supported_tasks=["text-generation", "vision"],
                )
            )
        log.debug(f"Found {len(models)} MLX models in HF cache")
        return models

    async def get_available_image_models(self) -> List[ImageModel]:
        """
        Get available MLX image models.

        Reads the local HuggingFace cache and returns only the MLX-compatible
        (mflux) checkpoints that are actually installed.

        Returns:
            List of ImageModel instances for MLX
        """
        try:
            models = await get_mlx_image_models_from_hf_cache()
            log.debug("Found %d installed MLX image models", len(models))
            return models
        except Exception as e:  # pragma: no cover
            log.debug("Could not read installed MLX image models: %s", e)
            return []

    async def get_available_asr_models(self) -> List[ASRModel]:
        """
        Get available MLX ASR models.
        """
        models = [
            ASRModel(
                id="mlx-community/whisper-tiny-mlx",
                name="Whisper Tiny",
                provider=Provider.MLX,
            ),
            ASRModel(
                id="mlx-community/whisper-base-mlx",
                name="Whisper Base",
                provider=Provider.MLX,
            ),
            ASRModel(
                id="mlx-community/whisper-base.en-mlx",
                name="Whisper Base EN",
                provider=Provider.MLX,
            ),
            ASRModel(
                id="mlx-community/whisper-small-mlx",
                name="Whisper Small",
                provider=Provider.MLX,
            ),
            ASRModel(
                id="mlx-community/whisper-small.en-mlx",
                name="Whisper Small EN",
                provider=Provider.MLX,
            ),
            ASRModel(
                id="mlx-community/whisper-medium-mlx",
                name="Whisper Medium",
                provider=Provider.MLX,
            ),
            ASRModel(
                id="mlx-community/whisper-medium.en-mlx",
                name="Whisper Medium EN",
                provider=Provider.MLX,
            ),
            ASRModel(
                id="mlx-community/whisper-large-v3-mlx",
                name="Whisper Large V3",
                provider=Provider.MLX,
            ),
            ASRModel(
                id="mlx-community/whisper-large-v3-turbo",
                name="Whisper Large V3 Turbo",
                provider=Provider.MLX,
            ),
            # mlx-audio speech-to-text models
            ASRModel(
                id="mlx-community/parakeet-tdt-0.6b-v2",
                name="Parakeet TDT 0.6B v2",
                provider=Provider.MLX,
            ),
            ASRModel(
                id="mlx-community/parakeet-tdt-0.6b-v3",
                name="Parakeet TDT 0.6B v3",
                provider=Provider.MLX,
            ),
            ASRModel(
                id="mlx-community/Qwen3-ASR-0.6B-8bit",
                name="Qwen3 ASR 0.6B (8-bit)",
                provider=Provider.MLX,
            ),
            ASRModel(
                id="mlx-community/Qwen3-ASR-1.7B-8bit",
                name="Qwen3 ASR 1.7B (8-bit)",
                provider=Provider.MLX,
            ),
        ]
        return models

    async def automatic_speech_recognition(
        self,
        audio: bytes,
        model: str,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float = 0.0,
        timeout_s: int | None = None,
        context: Any = None,
        word_timestamps: bool = False,
        **kwargs: Any,
    ) -> str | dict:
        """Transcribe audio to text using MLX Whisper checkpoints."""
        _ = context  # Context not currently used but kept for interface compatibility.
        if sys.platform != "darwin":
            raise RuntimeError("MLX automatic speech recognition requires macOS")

        if not audio:
            raise ValueError("audio must not be empty")

        # Whisper expects 16 kHz mono audio normalized to [-1, 1]
        samples = convert_audio_to_standard_format(audio, target_sample_rate=16_000)
        if samples.size == 0:
            return ""
        audio_float = (samples.astype(np.float32) / 32768.0).flatten()
        audio_float = np.ascontiguousarray(audio_float, dtype=np.float32)

        # Ensure the model is present in the local Hugging Face cache
        if await self._resolve_cached_repo_path(model) is None:
            raise ValueError(
                f"Model {model} must be downloaded first (not found in cache)"
            )

        # Non-Whisper models are served through the mlx-audio speech-to-text runtime.
        if "whisper" not in model.lower():
            return await self._transcribe_with_mlx_audio(
                audio=audio,
                model=model,
                language=language,
                temperature=temperature,
                timeout_s=timeout_s,
                word_timestamps=word_timestamps,
                **kwargs,
            )

        log.debug(
            "Transcribing audio with MLX Whisper model=%s language=%s temperature=%s",
            model,
            language,
            temperature,
        )

        def _run_transcription() -> dict[str, Any] | str | None:
            import mlx_whisper

            transcribe_fn = mlx_whisper.transcribe
            sig_params = set(inspect.signature(transcribe_fn).parameters)

            call_kwargs: dict[str, Any] = {"path_or_hf_repo": model}

            # Standard Whisper options with sensible defaults
            default_options: dict[str, Any] = {
                "compression_ratio_threshold": kwargs.get(
                    "compression_ratio_threshold", 2.4
                ),
                "logprob_threshold": kwargs.get("logprob_threshold", -1.0),
                "no_speech_threshold": kwargs.get("no_speech_threshold", 0.6),
                "condition_on_previous_text": kwargs.get(
                    "condition_on_previous_text", True
                ),
                "word_timestamps": kwargs.get("word_timestamps", word_timestamps),
                "temperature": kwargs.get("temperature", temperature),
            }

            optional_options: dict[str, Any] = {
                "language": kwargs.get("language", language),
                "initial_prompt": kwargs.get("initial_prompt", prompt),
            }

            # Include any additional keyword arguments provided by the caller
            extra_options = {
                key: value
                for key, value in kwargs.items()
                if key not in default_options and key not in optional_options
            }

            for options in (default_options, optional_options, extra_options):
                for key, value in options.items():
                    if value is not None and key in sig_params:
                        call_kwargs[key] = value

            return transcribe_fn(audio_float, **call_kwargs)

        transcribe_task = asyncio.to_thread(_run_transcription)
        try:
            if timeout_s is not None:
                result = await asyncio.wait_for(transcribe_task, timeout=timeout_s)
            else:
                result = await transcribe_task
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"MLX automatic speech recognition timed out after {timeout_s} seconds"
            ) from exc
        except Exception as exc:
            log.error("MLX Whisper transcription failed: %s", exc)
            raise RuntimeError(f"MLX Whisper transcription failed: {exc}") from exc

        text: str
        if isinstance(result, dict):
            text = str(result.get("text", "") or "")
        elif isinstance(result, str):
            text = result
        else:
            text = "" if result is None else str(result)

        log.debug("MLX Whisper transcription complete", extra={"length": len(text)})

        if not word_timestamps:
            return text

        # Extract chunks from mlx_whisper segments
        chunks = []
        if isinstance(result, dict):
            for segment in result.get("segments", []):
                # If word_timestamps was set, segments contain a "words" list
                words = segment.get("words", [])
                if words:
                    for w in words:
                        chunks.append(
                            {
                                "timestamp": [w.get("start", 0), w.get("end", 0)],
                                "text": w.get("word", ""),
                            }
                        )
                else:
                    chunks.append(
                        {
                            "timestamp": [
                                segment.get("start", 0),
                                segment.get("end", 0),
                            ],
                            "text": segment.get("text", ""),
                        }
                    )

        return {"text": text, "chunks": chunks}

    async def _load_stt_model(self, model: str) -> Any:
        """Load an mlx-audio speech-to-text model from cache or filesystem."""
        cache_key = f"{model}_stt_model"

        async with ModelManager.lock_model(cache_key):
            stt_model = ModelManager.get_model(cache_key)
            if stt_model is None:
                repo_path = await self._resolve_cached_repo_path(model)
                if repo_path is None:
                    raise ValueError(
                        f"Model {model} must be downloaded first (not found in cache)"
                    )

                def _load():
                    from mlx_audio.stt.utils import load_model as load_stt

                    log.info("Loading MLX STT model %s", model)
                    return load_stt(repo_path)

                stt_model = await asyncio.to_thread(_load)
                node_id = f"mlx_stt_provider_{model}"
                ModelManager.set_model(node_id, cache_key, stt_model)
            return stt_model

    async def _transcribe_with_mlx_audio(
        self,
        audio: bytes,
        model: str,
        language: str | None = None,
        temperature: float = 0.0,
        timeout_s: int | None = None,
        word_timestamps: bool = False,
        **kwargs: Any,
    ) -> str | dict:
        """Transcribe audio using non-Whisper mlx-audio STT models (Parakeet, Qwen3-ASR, ...)."""
        stt_model = await self._load_stt_model(model)

        # Persist the audio as a temporary 16 kHz mono WAV that mlx-audio can read.
        segment = AudioSegment.from_file(BytesIO(audio))
        segment = segment.set_frame_rate(16_000).set_channels(1).set_sample_width(2)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name
        segment.export(audio_path, format="wav")

        def _run() -> Any:
            generate = stt_model.generate
            sig = inspect.signature(generate)
            accepts_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            candidate = dict(kwargs)
            if language is not None:
                candidate.setdefault("language", language)
            if temperature is not None:
                candidate.setdefault("temperature", temperature)
            if accepts_var_kw:
                call_kwargs = candidate
            else:
                call_kwargs = {
                    k: v for k, v in candidate.items() if k in sig.parameters
                }
            return generate(audio_path, **call_kwargs)

        try:
            task = asyncio.to_thread(_run)
            if timeout_s is not None:
                result = await asyncio.wait_for(task, timeout=timeout_s)
            else:
                result = await task
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"MLX speech recognition timed out after {timeout_s} seconds"
            ) from exc
        except Exception as exc:
            log.error("MLX STT transcription failed: %s", exc)
            raise RuntimeError(f"MLX STT transcription failed: {exc}") from exc
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass

        text = str(getattr(result, "text", "") or "")
        if isinstance(result, str):
            text = result

        if not word_timestamps:
            return text

        # Normalize segments / sentences into timestamped chunks
        raw_segments = getattr(result, "segments", None)
        if raw_segments is None:
            raw_segments = getattr(result, "sentences", None)
        chunks = []
        for seg in raw_segments or []:
            if isinstance(seg, dict):
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                seg_text = seg.get("text", "")
            else:
                start = getattr(seg, "start", 0)
                end = getattr(seg, "end", 0)
                seg_text = getattr(seg, "text", "")
            chunks.append({"timestamp": [start, end], "text": seg_text})

        return {"text": text, "chunks": chunks}

    async def get_available_tts_models(self) -> List[TTSModel]:
        """
        Get available MLX TTS models.

        Returns TTS models from mlx-audio nodes that can be used for
        local text-to-speech generation on Apple Silicon.

        Returns:
            List of TTSModel instances for MLX TTS
        """
        # Kokoro TTS models (from KokoroTTS.Model enum)
        kokoro_models = [
            TTSModel(
                id="prince-canuma/Kokoro-82M",
                name="Kokoro TTS 82M",
                provider=Provider.MLX,
                voices=[
                    "af_alloy",
                    "af_aoede",
                    "af_bella",
                    "af_heart",
                    "af_jessica",
                    "af_kore",
                    "af_nicole",
                    "af_nova",
                    "af_river",
                    "af_sarah",
                    "af_sky",
                    "am_adam",
                    "am_echo",
                    "am_eric",
                    "am_fenrir",
                    "am_liam",
                    "am_michael",
                    "am_onyx",
                    "am_puck",
                    "am_santa",
                    "bf_alice",
                    "bf_emma",
                    "bf_isabella",
                    "bf_lily",
                    "bm_daniel",
                    "bm_fable",
                    "bm_george",
                    "bm_lewis",
                    "ef_dora",
                    "em_alex",
                    "em_santa",
                    "ff_siwis",
                    "hf_alpha",
                    "hf_beta",
                    "hm_omega",
                    "hm_psi",
                    "if_sara",
                    "im_nicola",
                    "jf_alpha",
                    "jf_gongitsune",
                    "jf_nezumi",
                    "jf_tebukuro",
                    "jm_kumo",
                    "pf_dora",
                    "pm_alex",
                    "pm_santa",
                    "zf_xiaobei",
                    "zf_xiaoni",
                    "zf_xiaoxiao",
                    "zf_xiaoyi",
                ],
            ),
            TTSModel(
                id="mlx-community/Kokoro-82M-bf16",
                name="Kokoro TTS 82M BF16",
                provider=Provider.MLX,
                voices=[
                    "af_alloy",
                    "af_aoede",
                    "af_bella",
                    "af_heart",
                    "af_jessica",
                    "af_kore",
                    "af_nicole",
                    "af_nova",
                    "af_river",
                    "af_sarah",
                    "af_sky",
                    "am_adam",
                    "am_echo",
                    "am_eric",
                    "am_fenrir",
                    "am_liam",
                    "am_michael",
                    "am_onyx",
                    "am_puck",
                    "am_santa",
                    "bf_alice",
                    "bf_emma",
                    "bf_isabella",
                    "bf_lily",
                    "bm_daniel",
                    "bm_fable",
                    "bm_george",
                    "bm_lewis",
                    "ef_dora",
                    "em_alex",
                    "em_santa",
                    "ff_siwis",
                    "hf_alpha",
                    "hf_beta",
                    "hm_omega",
                    "hm_psi",
                    "if_sara",
                    "im_nicola",
                    "jf_alpha",
                    "jf_gongitsune",
                    "jf_nezumi",
                    "jf_tebukuro",
                    "jm_kumo",
                    "pf_dora",
                    "pm_alex",
                    "pm_santa",
                    "zf_xiaobei",
                    "zf_xiaoni",
                    "zf_xiaoxiao",
                    "zf_xiaoyi",
                ],
            ),
            TTSModel(
                id="mlx-community/Kokoro-82M-4bit",
                name="Kokoro TTS 82M 4-bit",
                provider=Provider.MLX,
                voices=[
                    "af_alloy",
                    "af_aoede",
                    "af_bella",
                    "af_heart",
                    "af_jessica",
                    "af_kore",
                    "af_nicole",
                    "af_nova",
                    "af_river",
                    "af_sarah",
                    "af_sky",
                    "am_adam",
                    "am_echo",
                    "am_eric",
                    "am_fenrir",
                    "am_liam",
                    "am_michael",
                    "am_onyx",
                    "am_puck",
                    "am_santa",
                    "bf_alice",
                    "bf_emma",
                    "bf_isabella",
                    "bf_lily",
                    "bm_daniel",
                    "bm_fable",
                    "bm_george",
                    "bm_lewis",
                    "ef_dora",
                    "em_alex",
                    "em_santa",
                    "ff_siwis",
                    "hf_alpha",
                    "hf_beta",
                    "hm_omega",
                    "hm_psi",
                    "if_sara",
                    "im_nicola",
                    "jf_alpha",
                    "jf_gongitsune",
                    "jf_nezumi",
                    "jf_tebukuro",
                    "jm_kumo",
                    "pf_dora",
                    "pm_alex",
                    "pm_santa",
                    "zf_xiaobei",
                    "zf_xiaoni",
                    "zf_xiaoxiao",
                    "zf_xiaoyi",
                ],
            ),
            TTSModel(
                id="mlx-community/Kokoro-82M-6bit",
                name="Kokoro TTS 82M 6-bit",
                provider=Provider.MLX,
                voices=[
                    "af_alloy",
                    "af_aoede",
                    "af_bella",
                    "af_heart",
                    "af_jessica",
                    "af_kore",
                    "af_nicole",
                    "af_nova",
                    "af_river",
                    "af_sarah",
                    "af_sky",
                    "am_adam",
                    "am_echo",
                    "am_eric",
                    "am_fenrir",
                    "am_liam",
                    "am_michael",
                    "am_onyx",
                    "am_puck",
                    "am_santa",
                    "bf_alice",
                    "bf_emma",
                    "bf_isabella",
                    "bf_lily",
                    "bm_daniel",
                    "bm_fable",
                    "bm_george",
                    "bm_lewis",
                    "ef_dora",
                    "em_alex",
                    "em_santa",
                    "ff_siwis",
                    "hf_alpha",
                    "hf_beta",
                    "hm_omega",
                    "hm_psi",
                    "if_sara",
                    "im_nicola",
                    "jf_alpha",
                    "jf_gongitsune",
                    "jf_nezumi",
                    "jf_tebukuro",
                    "jm_kumo",
                    "pf_dora",
                    "pm_alex",
                    "pm_santa",
                    "zf_xiaobei",
                    "zf_xiaoni",
                    "zf_xiaoxiao",
                    "zf_xiaoyi",
                ],
            ),
            TTSModel(
                id="mlx-community/Kokoro-82M-8bit",
                name="Kokoro TTS 82M 8-bit",
                provider=Provider.MLX,
                voices=[
                    "af_alloy",
                    "af_aoede",
                    "af_bella",
                    "af_heart",
                    "af_jessica",
                    "af_kore",
                    "af_nicole",
                    "af_nova",
                    "af_river",
                    "af_sarah",
                    "af_sky",
                    "am_adam",
                    "am_echo",
                    "am_eric",
                    "am_fenrir",
                    "am_liam",
                    "am_michael",
                    "am_onyx",
                    "am_puck",
                    "am_santa",
                    "bf_alice",
                    "bf_emma",
                    "bf_isabella",
                    "bf_lily",
                    "bm_daniel",
                    "bm_fable",
                    "bm_george",
                    "bm_lewis",
                    "ef_dora",
                    "em_alex",
                    "em_santa",
                    "ff_siwis",
                    "hf_alpha",
                    "hf_beta",
                    "hm_omega",
                    "hm_psi",
                    "if_sara",
                    "im_nicola",
                    "jf_alpha",
                    "jf_gongitsune",
                    "jf_nezumi",
                    "jf_tebukuro",
                    "jm_kumo",
                    "pf_dora",
                    "pm_alex",
                    "pm_santa",
                    "zf_xiaobei",
                    "zf_xiaoni",
                    "zf_xiaoxiao",
                    "zf_xiaoyi",
                ],
            ),
        ]

        # Sesame TTS models (from SesameTTS.Model enum) - voice cloning models
        sesame_models = [
            TTSModel(
                id="mlx-community/csm-1b",
                name="Sesame TTS 1B",
                provider=Provider.MLX,
                voices=[],  # Voice cloning - uses reference audio
            ),
            TTSModel(
                id="mlx-community/csm-1b-8bit",
                name="Sesame TTS 1B 8-bit",
                provider=Provider.MLX,
                voices=[],  # Voice cloning - uses reference audio
            ),
        ]

        # Spark TTS models (from SparkTTS.Model enum)
        spark_models = [
            TTSModel(
                id="mlx-community/Spark-TTS-0.5B-bf16",
                name="Spark TTS 0.5B BF16",
                provider=Provider.MLX,
                voices=[],  # Spark uses speed/pitch/gender presets instead of voices
            ),
            TTSModel(
                id="mlx-community/Spark-TTS-0.5B-8bit",
                name="Spark TTS 0.5B 8-bit",
                provider=Provider.MLX,
                voices=[],  # Spark uses speed/pitch/gender presets instead of voices
            ),
        ]

        # Qwen3-TTS multilingual models with speaker voices
        qwen3_voices = ["Chelsie", "Ethan", "Vivian"]
        qwen3_models = [
            TTSModel(id=repo, name=name, provider=Provider.MLX, voices=qwen3_voices)
            for repo, name in [
                ("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16", "Qwen3-TTS 0.6B Base"),
                ("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16", "Qwen3-TTS 1.7B Base"),
                (
                    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
                    "Qwen3-TTS 1.7B Base (8-bit)",
                ),
                (
                    "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
                    "Qwen3-TTS 1.7B Custom Voice",
                ),
                (
                    "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
                    "Qwen3-TTS 1.7B Voice Design",
                ),
            ]
        ]

        # KittenTTS compact English models
        kitten_voices = [
            "expr-voice-2-m",
            "expr-voice-2-f",
            "expr-voice-3-m",
            "expr-voice-3-f",
            "expr-voice-4-m",
            "expr-voice-4-f",
            "expr-voice-5-m",
            "expr-voice-5-f",
        ]
        kitten_models = [
            TTSModel(id=repo, name=name, provider=Provider.MLX, voices=kitten_voices)
            for repo, name in [
                ("mlx-community/kitten-tts-nano-0.8", "Kitten TTS Nano"),
                ("mlx-community/kitten-tts-micro-0.8", "Kitten TTS Micro"),
                ("mlx-community/kitten-tts-mini-0.8", "Kitten TTS Mini"),
            ]
        ]

        # Voxtral TTS multilingual models with voice presets
        voxtral_voices = [
            "neutral_female",
            "neutral_male",
            "casual_female",
            "casual_male",
            "cheerful_female",
            "ar_male",
            "de_female",
            "de_male",
            "es_female",
            "es_male",
            "fr_female",
            "fr_male",
            "hi_female",
            "hi_male",
            "it_female",
            "it_male",
            "nl_female",
            "nl_male",
            "pt_female",
            "pt_male",
        ]
        voxtral_models = [
            TTSModel(
                id="mlx-community/Voxtral-4B-TTS-2603-mlx-bf16",
                name="Voxtral 4B TTS",
                provider=Provider.MLX,
                voices=voxtral_voices,
            )
        ]

        # MeloTTS lightweight English model (accent via language code)
        melotts_models = [
            TTSModel(
                id="mlx-community/MeloTTS-English-MLX",
                name="Melo TTS English",
                provider=Provider.MLX,
                voices=["EN-US", "EN-BR", "EN_INDIA", "EN-AU", "EN-Default"],
            )
        ]

        # Voice-cloning / zero-shot models (no fixed voice presets)
        cloning_models = [
            TTSModel(id=repo, name=name, provider=Provider.MLX, voices=[])
            for repo, name in [
                ("mlx-community/Dia-1.6B", "Dia 1.6B"),
                ("mlx-community/Dia-1.6B-fp16", "Dia 1.6B (fp16)"),
                ("mlx-community/OuteTTS-1.0-0.6B-fp16", "OuteTTS 1.0 0.6B"),
                ("mlx-community/outetts-0.3-500M-bf16", "OuteTTS 0.3 500M"),
                ("mlx-community/OmniVoice-bf16", "OmniVoice"),
                ("mlx-community/chatterbox-fp16", "Chatterbox TTS"),
                ("mlx-community/Chatterbox-TTS-fp16", "Chatterbox TTS (fp16)"),
                ("mlx-community/higgs-audio-v2-3B-mlx-bf16", "Higgs Audio v2 3B"),
                ("mlx-community/higgs-audio-v2-3B-mlx-q8", "Higgs Audio v2 3B (q8)"),
                (
                    "mlx-community/LongCat-AudioDiT-1B-4bit",
                    "LongCat-AudioDiT 1B (4-bit)",
                ),
                (
                    "mlx-community/LongCat-AudioDiT-3.5B-4bit",
                    "LongCat-AudioDiT 3.5B (4-bit)",
                ),
            ]
        ]

        return (
            kokoro_models
            + sesame_models
            + spark_models
            + qwen3_models
            + kitten_models
            + voxtral_models
            + melotts_models
            + cloning_models
        )

    # ------------------------------------------------------------------
    # Tool emulation helpers
    # ------------------------------------------------------------------
    def _format_tools_as_python(self, tools: Sequence[Tool]) -> str:
        """Format tools as Python function definitions for emulation.

        Args:
            tools: Sequence of tools to format.

        Returns:
            String containing Python function definitions.
        """
        log.debug(f"Formatting {len(tools)} tools as Python functions for emulation")

        function_defs = []
        for tool in tools:
            tool_param = tool.tool_param()
            func = tool_param.get("function", {})
            name = func.get("name", "unknown")
            description = func.get("description", "")
            parameters = func.get("parameters", {})

            # Build simplified function signature
            params = []
            if "properties" in parameters:
                for param_name in parameters["properties"].keys():
                    params.append(param_name)

            params_str = ", ".join(params) if params else ""
            func_def = f"# {name}({params_str})\n# {description}"
            if params:
                func_def += f"\n# Example: {name}({params[0]}='...')"
            function_defs.append(func_def)

        return "\n\n".join(function_defs)

    def _parse_function_calls(
        self, text: str, tools: Sequence[Tool] | None = None
    ) -> list[ToolCall]:
        """Parse Python function calls from text using AST parsing.

        Args:
            text: Text containing potential function calls.
            tools: Optional list of tools to map positional args to named parameters.

        Returns:
            List of parsed ToolCall objects.
        """
        if not text or not text.strip():
            return []

        tool_calls: list[ToolCall] = []
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Try to parse as a function call
            try:
                tree = ast.parse(line, mode="eval")
                if isinstance(tree.body, ast.Call):
                    call_node = tree.body
                    func_name = self._extract_function_name(call_node.func)

                    if func_name:
                        args = self._extract_call_arguments(call_node, func_name, tools)
                        tool_call = ToolCall(
                            id=f"call_{len(tool_calls)}",
                            name=func_name,
                            args=args,
                        )
                        tool_calls.append(tool_call)
                        log.debug(f"Parsed tool call: {func_name} with args {args}")
            except (SyntaxError, ValueError):
                # Not a valid function call, skip
                continue

        return tool_calls

    def _extract_function_name(self, node: ast.expr) -> str | None:
        """Extract function name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _extract_call_arguments(
        self, call_node: ast.Call, func_name: str, tools: Sequence[Tool] | None
    ) -> dict[str, Any]:
        """Extract arguments from a function call AST node."""
        args: dict[str, Any] = {}

        # Extract keyword arguments
        for keyword in call_node.keywords:
            if keyword.arg:
                value = self._ast_to_value(keyword.value)
                args[keyword.arg] = value

        # Handle positional arguments by mapping to parameter names
        if call_node.args and tools:
            matching_tool = next((t for t in tools if t.name == func_name), None)
            if matching_tool:
                tool_param = matching_tool.tool_param()
                func_schema = tool_param.get("function", {})
                parameters = func_schema.get("parameters", {})
                properties = parameters.get("properties", {})
                param_names = list(properties.keys())

                for i, arg in enumerate(call_node.args):
                    if i < len(param_names):
                        param_name = param_names[i]
                        value = self._ast_to_value(arg)
                        args[param_name] = value

        return args

    def _ast_to_value(self, node: ast.expr) -> Any:
        """Convert AST node to Python value."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Str):  # Python 3.7 compatibility
            return node.s
        elif isinstance(node, ast.Num):  # Python 3.7 compatibility
            return node.n
        elif isinstance(node, ast.List):
            return [self._ast_to_value(el) for el in node.elts]
        elif isinstance(node, ast.Dict):
            result = {}
            for k, v in zip(node.keys, node.values):
                if k is not None:  # Skip None keys
                    key = self._ast_to_value(k)
                    value = self._ast_to_value(v)
                    if key is not None:
                        result[key] = value
            return result
        elif isinstance(node, ast.NameConstant):  # Python 3.7 compatibility
            return node.value
        elif isinstance(node, (ast.Name, ast.Attribute)):
            # For variable references, try to return the name as a string
            return ast.unparse(node) if hasattr(ast, "unparse") else str(node)
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _stream_chat(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Tool],
        max_tokens: int,
        context_window: int,
        response_format: dict | None,
        **kwargs: Any,
    ) -> AsyncIterator[Chunk | ToolCall]:
        """Stream chat responses for a given message sequence.

        Internal helper that orchestrates prompting, token streaming, and
        extraction of tool calls from the MLX runtime. Yields `Chunk` and
        `ToolCall` items to the caller.
        """
        # Route FIBO VLM through its native runtime before generic mlx-vlm handling.
        image_parts = self._extract_image_parts(messages)
        audio_parts = self._extract_audio_parts(messages)
        if self._is_fibo_vlm_model(model):
            async for item in self._stream_fibo_vlm_chat(
                messages,
                model,
                image_parts,
                max_tokens=max_tokens,
                **kwargs,
            ):
                yield item
            return

        # Route to mlx-vlm if the model appears vision-capable and images are present
        if (image_parts and self._is_vision_model(model)) or (
            audio_parts and self._is_audio_model(model)
        ):
            async for item in self._stream_vlm_chat(
                messages,
                model,
                image_parts,
                audio_parts,
                max_tokens=max_tokens,
                **kwargs,
            ):
                yield item
            return

        log.debug(
            "MLX _stream_chat start | model=%s messages=%d tools=%d kwargs_keys=%s",
            model,
            len(messages),
            len(tools),
            sorted(list(kwargs.keys())),
        )

        mdl, tokenizer = await self._load_model(model)

        # Determine if we need tool emulation
        use_tool_emulation = len(tools) > 0 and not tokenizer.has_tool_calling
        if use_tool_emulation:
            log.info(f"Using tool emulation for model {model}")

        # Prepare messages with tool emulation if needed
        if use_tool_emulation and len(tools) > 0:
            tool_definitions = self._format_tools_as_python(tools)
            tool_instruction = (
                "\n\n=== AVAILABLE FUNCTIONS ===\n"
                "You can call these functions by writing a function call on a single line.\n"
                "DO NOT write function definitions - only write function CALLS.\n\n"
                f"{tool_definitions}\n\n"
                "=== INSTRUCTIONS ===\n"
                "When you need to use a function:\n"
                "1. Write ONLY the function call, nothing else\n"
                "2. Use this exact format: function_name(param='value')\n"
                "3. Do NOT write 'def', 'return', or any other Python keywords\n"
                "4. After calling a function, wait for the result\n"
                "5. Once you receive a function result, use it in your final answer\n"
                "6. Do NOT call the same function twice"
            )

            # Inject tool instructions into system message or prepend to first user message
            messages_list = list(messages)
            if messages_list and messages_list[0].role == "system":
                # Append to existing system message
                existing_content = messages_list[0].content or ""
                messages_list[0] = Message(
                    role="system", content=f"{existing_content}{tool_instruction}"
                )
            else:
                # Prepend new system message
                messages_list.insert(
                    0, Message(role="system", content=tool_instruction)
                )

            converted_messages = [
                self._convert_message(msg, index)
                for index, msg in enumerate(messages_list)
            ]
            # Don't pass tool_defs when using emulation
            tool_defs = None
        else:
            converted_messages = [
                self._convert_message(msg, index) for index, msg in enumerate(messages)
            ]
            tool_defs = self._convert_tools(tools)

        prompt = tokenizer.apply_chat_template(  # pyright: ignore[reportCallIssue]
            converted_messages,
            tool_defs or None,
            add_generation_prompt=True,
        )

        runtime_overrides = dict(kwargs)
        sampler = self._build_sampler(runtime_overrides)
        stream_kwargs = self._build_stream_kwargs(runtime_overrides)
        if sampler is not None:
            stream_kwargs["sampler"] = sampler

        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        # Acquire generation lock to serialize MLX model usage across concurrent requests
        await self._generation_lock.acquire()

        try:

            def _run_stream() -> None:
                log.debug(
                    "MLX _run_stream thread start | model=%s max_tokens=%s stream_kwargs=%s",
                    model,
                    max_tokens,
                    {
                        k: ("<callable>" if callable(v) else v)
                        for k, v in stream_kwargs.items()
                    },
                )
                try:
                    for response in stream_generate(
                        mdl,
                        tokenizer,
                        prompt,
                        max_tokens=max_tokens,
                        **stream_kwargs,
                    ):
                        asyncio.run_coroutine_threadsafe(
                            queue.put(("response", response)), loop
                        ).result()
                except Exception as exc:  # pragma: no cover - defensive
                    log.exception("MLX _run_stream thread error: %s", exc)
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("error", exc)), loop
                    ).result()
                finally:
                    log.debug("MLX _run_stream thread done")
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("done", None)), loop
                    ).result()

            threading.Thread(target=_run_stream, daemon=True).start()

            tool_state = {
                "buffer": "",
                "in_tool_call": False,
                "counter": 0,
            }
            done_emitted = False
            accumulated_content = ""  # For emulation parsing

            # Watchdog to log if no items arrive for too long
            last_activity = time.monotonic()
            watchdog_seconds = int(os.getenv("MLX_WATCHDOG_SECS", "60"))

            while True:
                try:
                    kind, payload = await asyncio.wait_for(queue.get(), timeout=5.0)
                    last_activity = time.monotonic()
                except asyncio.TimeoutError:
                    if (
                        watchdog_seconds > 0
                        and (time.monotonic() - last_activity) > watchdog_seconds
                    ):
                        log.warning(
                            "MLX _stream_chat watchdog tripped | no-activity-for=%ss model=%s in_tool_call=%s",
                            int(time.monotonic() - last_activity),
                            model,
                            tool_state["in_tool_call"],
                        )
                        # Continue waiting; we only log to surface the hang
                    continue
                if kind == "response":
                    response = payload
                    is_final = getattr(response, "finish_reason", None) is not None
                    if is_final:
                        self._update_usage(response)

                    clean_text = self._strip_terminal_special_tokens(response.text)
                    if not clean_text and not is_final:
                        clean_text = self._decode_stream_token(
                            tokenizer, getattr(response, "token", None)
                        )

                    # Process native tool calls if supported
                    segments, parsed_calls = self._process_response_text(
                        clean_text, tool_state, tokenizer
                    )

                    # Accumulate content for emulation
                    if use_tool_emulation:
                        accumulated_content += clean_text

                    # Yield native tool calls
                    for tool_call in parsed_calls:
                        log.debug(
                            "MLX parsed tool_call name=%s id=%s",
                            tool_call.name,
                            tool_call.id,
                        )
                        yield tool_call

                    # Parse emulated tool calls on final response
                    if is_final and use_tool_emulation and accumulated_content:
                        log.debug(
                            "Parsing emulated tool calls from accumulated content"
                        )
                        emulated_calls = self._parse_function_calls(
                            accumulated_content, tools
                        )
                        for tool_call in emulated_calls:
                            log.debug(f"Yielding emulated tool call: {tool_call.name}")
                            yield tool_call
                        # Don't emit segments if we found emulated tool calls
                        if emulated_calls:
                            segments = []

                    for i, segment in enumerate(segments):
                        if not segment:
                            continue
                        done_flag = (
                            is_final
                            and i == len(segments) - 1
                            and not tool_state["in_tool_call"]
                        )
                        if done_flag:
                            log.debug("MLX emitting final chunk (done=True)")
                        yield Chunk(content=segment, done=done_flag)
                        done_emitted = done_emitted or done_flag

                    if is_final:
                        if not done_emitted:
                            log.debug("MLX emitting trailing done chunk")
                            yield Chunk(content="", done=True)
                        break
                elif kind == "error":
                    log.error(
                        "MLX _stream_chat received error from thread: %s", payload
                    )
                    raise payload
                elif kind == "done":
                    if not done_emitted:
                        log.debug(
                            "MLX done without explicit final token; emitting done chunk"
                        )
                        yield Chunk(content="", done=True)
                    break
        finally:
            # Always release generation lock, even on error
            self._generation_lock.release()
            log.debug("MLX _stream_chat end | model=%s", model)

    async def _load_model(self, model: str) -> tuple[nn.Module, TokenizerWrapper]:
        # Construct cache key for ModelManager
        cache_key = f"{model}_language_model_{self.adapter_path}"

        # Use ModelManager lock for thread-safe model loading
        async with ModelManager.lock_model(cache_key):
            async with self._load_lock:
                # Try to get cached model from ModelManager
                cached = ModelManager.get_model(cache_key)
                if cached is not None:
                    try:
                        _mdl, cached_tokenizer = cached
                        self._configure_tokenizer(cached_tokenizer)
                    except Exception:
                        pass
                    return cached

                def _load() -> tuple[nn.Module, TokenizerWrapper]:
                    return mlx_lm.utils.load(  # pyright: ignore[reportReturnType]
                        model,
                        adapter_path=self.adapter_path,
                    )

                mdl, tokenizer = await asyncio.to_thread(_load)

                self._configure_tokenizer(tokenizer)

                # Cache in ModelManager (requires a node_id, use a synthetic one for provider-level caching)
                node_id = f"mlx_provider_{model}_{self.adapter_path or 'default'}"
                ModelManager.set_model(node_id, cache_key, (mdl, tokenizer))

                log.info("Loaded MLX model %s", model)
                return mdl, tokenizer

    # ------------------------------------------------------------------
    # mlx-vlm runtime and flow (vision models)
    # ------------------------------------------------------------------
    def _cache_key_vlm(self, model: str) -> str:
        adapter = self.adapter_path or ""
        return f"vlm|{model}|{adapter}"

    def _get_cached_vlm_model(self, model: str) -> tuple[nn.Module, Any, Any] | None:
        key = self._cache_key_vlm(model)
        now = time.monotonic()
        with _VLM_MODEL_CACHE_LOCK:
            entry = _VLM_MODEL_CACHE.get(key)
            if not entry:
                return None
            mdl, proc, cfg, expires_at = entry
            if expires_at < now:
                _VLM_MODEL_CACHE.pop(key, None)
                return None
            return mdl, proc, cfg

    def _set_cached_vlm_model(
        self, model: str, mdl: nn.Module, proc: Any, cfg: Any
    ) -> None:
        key = self._cache_key_vlm(model)
        expires_at = time.monotonic() + _CACHE_TTL_SECONDS
        with _VLM_MODEL_CACHE_LOCK:
            _VLM_MODEL_CACHE[key] = (mdl, proc, cfg, expires_at)

    async def _load_fibo_vlm_model(self, model: str) -> Any:
        cache_key = f"{model}_fibo_vlm_model"

        async with ModelManager.lock_model(cache_key):
            cached = ModelManager.get_model(cache_key)
            if cached is not None:
                return cached

            def _load() -> Any:
                from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM

                return FiboVLM(model_id=model, quantize=4)

            fibo_vlm = await asyncio.to_thread(_load)
            node_id = f"mlx_fibo_vlm_provider_{model}"
            ModelManager.set_model(node_id, cache_key, fibo_vlm)
            log.info("Loaded FIBO VLM model %s", model)
            return fibo_vlm

    async def _load_vlm_model(self, model: str) -> tuple[nn.Module, Any, Any]:
        # Construct cache key for ModelManager
        cache_key = f"{model}_vision_model_{self.adapter_path}"

        # Use ModelManager lock for thread-safe model loading
        async with ModelManager.lock_model(cache_key):
            async with self._vlm_load_lock:
                # Try to get cached model from ModelManager
                cached = ModelManager.get_model(cache_key)
                if cached is not None:
                    return cached

                def _load() -> tuple[nn.Module, Any, Any]:
                    mdl, proc = mlx_vlm.load(model)
                    cfg = getattr(mdl, "config", None)
                    if cfg is None and mlx_vlm.utils.load_config is not None:
                        cfg = mlx_vlm.utils.load_config(model)
                    return mdl, proc, cfg

                mdl, proc, cfg = await asyncio.to_thread(_load)

                # Cache in ModelManager
                node_id = f"mlx_vlm_provider_{model}_{self.adapter_path or 'default'}"
                ModelManager.set_model(node_id, cache_key, (mdl, proc, cfg))

                log.info("Loaded MLX-VLM model %s", model)
                return mdl, proc, cfg

    def _is_fibo_vlm_model(self, model: str) -> bool:
        name = (model or "").lower()
        return "fibo-vlm" in name or name == "briaai/fibo-vlm"

    def _is_vision_model(self, model: str) -> bool:
        name = (model or "").lower()
        keywords = (
            "qwen2-vl",
            "qwen2.5-vl",
            "qwen-vl",
            "llava",
            "idefics",
            "vl-",
            "gemma-3n",
        )
        return any(k in name for k in keywords)

    def _is_audio_model(self, model: str) -> bool:
        name = (model or "").lower()
        keywords = (
            "gemma-3n-e2b",
            "e2b-it",
            "audio",
        )
        # be permissive if both image/audio present and model is vision
        return any(k in name for k in keywords) or self._is_vision_model(model)

    def _ensure_vlm_processor_ready(self, proc: Any, cfg: Any) -> None:
        """Ensure mlx-vlm processor has necessary attributes set.

        Some processors (e.g., LLaVA) expect a non-None patch_size. If missing,
        attempt to infer from model config or fall back to a sane default (14).
        """
        if proc is None:
            return
        image_proc = getattr(proc, "image_processor", None)
        if image_proc is None:
            return
        patch_size = getattr(image_proc, "patch_size", None)
        if patch_size is None:
            inferred = None
            if cfg is not None:
                vision_cfg = getattr(cfg, "vision_config", None)
                inferred = getattr(vision_cfg, "patch_size", None)
            try:
                if inferred is None:
                    inferred = 14
                setattr(image_proc, "patch_size", int(inferred))
            except Exception:
                # Last resort default
                try:
                    setattr(image_proc, "patch_size", 14)
                except Exception:
                    pass

    def _extract_text_from_content(
        self, content: str | list[MessageContent] | None
    ) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        texts: list[str] = []
        for part in content:
            if isinstance(part, MessageTextContent):
                if part.text:
                    texts.append(part.text)
        return "".join(texts)

    def _extract_last_user_prompt(self, messages: Sequence[Message]) -> str:
        for msg in reversed(messages):
            if msg.role == "user":
                text = self._extract_text_from_content(msg.content)
                if text:
                    return text
        # Fallback: concatenate all text parts
        parts: list[str] = []
        for msg in messages:
            t = self._extract_text_from_content(msg.content)
            if t:
                parts.append(t)
        return "\n".join(parts)

    def _extract_image_parts(
        self, messages: Sequence[Message]
    ) -> list[MessageImageContent]:
        images: list[MessageImageContent] = []
        for msg in messages:
            if isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, MessageImageContent):
                        images.append(part)
        return images

    def _extract_audio_parts(
        self, messages: Sequence[Message]
    ) -> list[MessageAudioContent]:
        audios: list[MessageAudioContent] = []
        for msg in messages:
            if isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, MessageAudioContent):
                        audios.append(part)
        return audios

    async def _prepare_vlm_images(self, parts: list[MessageImageContent]) -> list[str]:
        """Load images as PIL.Image objects in memory (no temp files)."""
        prepared_images: list[str] = []
        for part in parts:
            image_ref: ImageRef = part.image
            uri = image_ref.uri or ""
            data: bytes | None = None
            if uri:
                try:
                    _mime, fetched = await fetch_uri_bytes_and_mime(uri)
                    data = fetched
                except Exception:
                    try:
                        parsed = urlparse(uri)
                        local_path = (
                            unquote(parsed.path) if parsed.scheme == "file" else uri
                        )
                        with open(local_path, "rb") as f:
                            data = f.read()
                    except Exception:
                        data = None
            if data is None and image_ref.data:
                data = image_ref.data

            if data is None:
                continue

            try:
                bytes_io = BytesIO()
                img = PIL.Image.open(BytesIO(data))
                img = img.convert("RGB")
                img = img.resize((224, 224))
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                img.save(tmp, format="PNG")
                prepared_images.append(tmp.name)
            except Exception:
                pass

        return prepared_images

    async def _prepare_vlm_audio(self, parts: list[MessageAudioContent]) -> list[str]:
        """Download/convert audios and persist as temporary WAV files.

        Returns (audio_paths, temp_files) where both lists contain filesystem paths.
        All returned paths in audio_paths are WAV files when possible.
        """
        prepared_paths: list[str] = []
        for part in parts:
            audio_ref: AudioRef = part.audio
            uri = audio_ref.uri or ""
            data: bytes | None = None
            if uri:
                try:
                    _mime, fetched = await fetch_uri_bytes_and_mime(uri)
                    data = fetched
                except Exception:
                    try:
                        parsed = urlparse(uri)
                        local_path = (
                            unquote(parsed.path) if parsed.scheme == "file" else uri
                        )
                        with open(local_path, "rb") as f:
                            data = f.read()
                    except Exception:
                        data = None
            if data is None and audio_ref.data:
                data = audio_ref.data

            if data is None:
                continue

            audio_seg = AudioSegment.from_file(BytesIO(data))
            audio_seg = (
                audio_seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            )
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            audio_seg.export(tmp, format="wav")
            prepared_paths.append(tmp.name)

        return prepared_paths

    async def _stream_fibo_vlm_chat(
        self,
        messages: Sequence[Message],
        model: str,
        image_parts: list[MessageImageContent],
        max_tokens: int,
        **kwargs: Any,
    ) -> AsyncIterator[Chunk | ToolCall]:
        _ = max_tokens, kwargs
        fibo_vlm = await self._load_fibo_vlm_model(model)
        prompt_text = self._extract_last_user_prompt(messages)
        images = await self._prepare_vlm_images(image_parts)

        def _run_generate() -> str:
            import PIL.Image

            if images:
                image = PIL.Image.open(images[-1]).convert("RGB")
                try:
                    return fibo_vlm.inspire(image=image, prompt=prompt_text or None)
                finally:
                    image.close()
            return fibo_vlm.generate(prompt=prompt_text)

        try:
            async with self._vlm_generation_lock:
                output: str = await asyncio.to_thread(_run_generate)
            yield Chunk(content=output, done=True)
        finally:
            for image_path in images:
                try:
                    os.remove(image_path)
                except Exception:
                    pass

    async def _stream_vlm_chat(
        self,
        messages: Sequence[Message],
        model: str,
        image_parts: list[MessageImageContent],
        audio_parts: list[MessageAudioContent],
        max_tokens: int,
        **kwargs: Any,
    ) -> AsyncIterator[Chunk | ToolCall]:
        log.debug(
            "MLX-VLM _stream_vlm_chat start | model=%s images=%d audios=%d",
            model,
            len(image_parts),
            len(audio_parts),
        )
        mdl, proc, cfg = await self._load_vlm_model(model)

        prompt_text = self._extract_last_user_prompt(messages)
        images = await self._prepare_vlm_images(image_parts)
        audios = await self._prepare_vlm_audio(audio_parts)
        log.debug(
            "MLX-VLM prepared assets | images=%d audios=%d", len(images), len(audios)
        )

        # Defensive processor normalization
        self._ensure_vlm_processor_ready(proc, cfg)

        formatted_prompt = mlx_vlm.prompt_utils.apply_chat_template(
            proc,
            cfg,
            prompt_text,
            num_images=len(images),
            num_audios=len(audios),
        )
        log.debug(
            "MLX-VLM prompt prepared | prompt_len=%d",
            len(formatted_prompt) if isinstance(formatted_prompt, str) else -1,
        )

        def _run_generate() -> str:
            # Keep params conservative; mlx-vlm's generate may accept more kwargs
            result = mlx_vlm.generate.generate(
                mdl,
                proc,
                formatted_prompt,
                image=images if images else None,
                audio=audios if audios else None,
                verbose=False,
                max_tokens=max_tokens,
            )
            # Some versions may return objects; coerce to string
            return result.text

        # Serialize VLM generation to avoid concurrent Metal access
        async with self._vlm_generation_lock:
            try:
                output: str = await asyncio.to_thread(_run_generate)
            except Exception as exc:
                log.exception("MLX-VLM generation error: %s", exc)
                raise RuntimeError(f"mlx-vlm generation failed: {exc}")

        log.debug(
            "MLX-VLM generation ok | output_len=%d",
            len(output) if isinstance(output, str) else -1,
        )
        yield Chunk(content=output, done=True)

        for audio_path in audios:
            try:
                os.remove(audio_path)
            except Exception:
                pass

        for image_path in images:
            try:
                os.remove(image_path)
            except Exception:
                pass
        log.debug("MLX-VLM _stream_vlm_chat end | model=%s", model)

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------
    def _cache_key(self, model: str) -> str:
        adapter = self.adapter_path or ""
        return f"{model}|{adapter}"

    def _get_cached_model(
        self, model: str
    ) -> tuple[nn.Module, TokenizerWrapper] | None:
        key = self._cache_key(model)
        now = time.monotonic()
        with _MODEL_CACHE_LOCK:
            entry = _MODEL_CACHE.get(key)
            if not entry:
                return None
            mdl, tok, expires_at = entry
            if expires_at < now:
                # Expired; evict
                _MODEL_CACHE.pop(key, None)
                return None
            return mdl, tok

    def _set_cached_model(
        self, model: str, mdl: nn.Module, tok: TokenizerWrapper
    ) -> None:
        key = self._cache_key(model)
        expires_at = time.monotonic() + _CACHE_TTL_SECONDS
        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE[key] = (mdl, tok, expires_at)

    # ------------------------------------------------------------------
    # TTS model caching helpers
    # ------------------------------------------------------------------
    def _get_cached_tts_model(self, model: str) -> Any | None:
        """Get cached TTS model if available and not expired."""
        now = time.monotonic()
        with _TTS_MODEL_CACHE_LOCK:
            entry = _TTS_MODEL_CACHE.get(model)
            if not entry:
                return None
            tts_model, expires_at = entry
            if expires_at < now:
                # Expired; evict
                _TTS_MODEL_CACHE.pop(model, None)
                return None
            return tts_model

    def _set_cached_tts_model(self, model: str, tts_model: Any) -> None:
        """Cache a TTS model with TTL expiration."""
        expires_at = time.monotonic() + _CACHE_TTL_SECONDS
        with _TTS_MODEL_CACHE_LOCK:
            _TTS_MODEL_CACHE[model] = (tts_model, expires_at)

    async def _load_tts_model(self, model: str) -> Any:
        """Load TTS model from cache or filesystem.

        Args:
            model: Model repository ID (e.g., "mlx-community/Kokoro-82M-bf16")

        Returns:
            Loaded TTS model

        Raises:
            ValueError: If model is not found in HuggingFace cache
        """
        # Construct cache key for ModelManager
        cache_key = f"{model}_tts_model"

        # Use ModelManager lock for thread-safe model loading
        async with ModelManager.lock_model(cache_key):
            # Try to get cached model from ModelManager
            tts_model = ModelManager.get_model(cache_key)

            if tts_model is None:
                # Model not in cache, load it
                repo_path = await self._resolve_cached_repo_path(model)
                if repo_path is None:
                    raise ValueError(
                        f"Model {model} must be downloaded first (not found in cache)"
                    )

                load_target = repo_path

                # Load the TTS model
                def _load():
                    log.info(f"Loading MLX TTS model {model}")
                    return mlx_audio.tts.utils.load_model(Path(load_target))

                tts_model = await asyncio.to_thread(_load)

                # Cache in ModelManager
                node_id = f"mlx_tts_provider_{model}"
                ModelManager.set_model(node_id, cache_key, tts_model)
                log.debug(f"Cached TTS model {model}")
            else:
                log.debug(f"Using cached TTS model {model}")

            return tts_model

    async def _resolve_cached_repo_path(self, repo_id: str) -> Path | None:
        """
        Locate the snapshot directory for a cached Hugging Face repo.

        Returns:
            Path to the cached repo snapshot, or None if not found.
        """
        repo_root = await HF_FAST_CACHE.resolve(repo_id, ".", repo_type="model")
        if not repo_root:
            return None
        path = Path(repo_root)
        return path if path.exists() else None

    def _build_stream_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        stream_kwargs = dict(kwargs)
        for key in (
            "temperature",
            "temp",
            "top_p",
            "top_k",
            "min_p",
            "min_tokens_to_keep",
            "xtc_probability",
            "xtc_threshold",
        ):
            stream_kwargs.pop(key, None)
        return stream_kwargs

    def _configure_tokenizer(self, tokenizer: TokenizerWrapper) -> None:
        for token in (
            "<|im_end|>",
            "<|eot_id|>",
            "<|end|>",
            "</s>",
        ):
            try:
                tokenizer.add_eos_token(token)
            except Exception:
                pass

    def _strip_terminal_special_tokens(self, text: str) -> str:
        if not text:
            return text
        for token in (
            "<|im_end|>",
            "<|eot_id|>",
            "<|end|>",
        ):
            text = text.replace(token, "")
        return text

    def _decode_stream_token(
        self, tokenizer: TokenizerWrapper, token: Any | None
    ) -> str:
        if token is None:
            return ""
        try:
            token_id = int(token)
        except (TypeError, ValueError):
            return ""

        try:
            text = tokenizer.decode([token_id], skip_special_tokens=False)
        except TypeError:
            try:
                text = tokenizer.decode([token_id])
            except Exception:
                return ""
        except Exception:
            return ""

        if not text:
            return ""

        text = self._strip_terminal_special_tokens(text)
        special_tokens = {
            getattr(tokenizer, "bos_token", None),
            getattr(tokenizer, "eos_token", None),
            "<s>",
            "</s>",
        }
        if text in special_tokens:
            return ""
        return text

    def _build_sampler(self, kwargs: dict[str, Any]) -> Any | None:
        if mlx_lm.sample_utils.make_sampler is None:
            return None

        sampler_params = {
            "temp": kwargs.pop("temperature", 0.5),
            "top_p": kwargs.pop("top_p", 0.95),
            "top_k": kwargs.pop("top_k", 50),
            "min_p": kwargs.pop("min_p", 0.0),
            "min_tokens_to_keep": kwargs.pop("min_tokens_to_keep", 1),
            "xtc_probability": kwargs.pop("xtc_probability", 0.0),
            "xtc_threshold": kwargs.pop("xtc_threshold", 0.0),
        }

        # When temperature is explicitly zero we avoid allocating a sampler.
        if (
            sampler_params.get("temp", 0.0) == 0
            and sampler_params.get("top_p", 0.0) == 0.0
        ):
            return None

        return mlx_lm.sample_utils.make_sampler(**sampler_params)

    def _ensure_usage_counters(self) -> None:
        if not isinstance(getattr(self, "usage", None), dict):
            self.usage = {}
        self.usage.setdefault("prompt_tokens", 0)
        self.usage.setdefault("completion_tokens", 0)
        self.usage.setdefault("total_tokens", 0)
        self.usage.setdefault("total_requests", 0)
        self.usage.setdefault("total_images", 0)

    def _update_usage(self, response: Any) -> None:
        self._ensure_usage_counters()
        prompt_tokens = int(getattr(response, "prompt_tokens", 0))
        completion_tokens = int(getattr(response, "generation_tokens", 0))
        self.usage["prompt_tokens"] += prompt_tokens
        self.usage["completion_tokens"] += completion_tokens
        self.usage["total_tokens"] += prompt_tokens + completion_tokens
        self.track_usage(
            model=getattr(response, "model", ""),
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )

    def _convert_message(self, message: Message, index: int) -> dict[str, Any]:
        content = self._normalize_content(message.content)
        payload: dict[str, Any] = {
            "role": message.role or "user",
            "content": content,
        }

        if message.name:
            payload["name"] = message.name
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id

        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tool_call.id or f"call_{index}_{i}",
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.args or {}),
                    },
                }
                for i, tool_call in enumerate(message.tool_calls)
            ]

        return payload

    def _convert_tools(self, tools: Sequence[Tool]) -> list[dict[str, Any]]:
        tool_defs: list[dict[str, Any]] = []
        for tool in tools:
            try:
                tool_defs.append(tool.tool_param())
            except Exception as exc:
                log.warning(
                    "Failed to convert tool %s: %s", getattr(tool, "name", tool), exc
                )
        return tool_defs

    def _normalize_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                text = getattr(part, "text", None)
                if text:
                    parts.append(text)
            return "".join(parts)
        return str(content)

    def _process_response_text(
        self,
        text: str,
        tool_state: dict[str, Any],
        tokenizer: TokenizerWrapper,
    ) -> tuple[list[str], list[ToolCall]]:
        if not text:
            return [], []

        if not tokenizer.has_tool_calling:
            return [text], []

        start_token = getattr(tokenizer, "tool_call_start", None)
        end_token = getattr(tokenizer, "tool_call_end", None)
        if not start_token or not end_token:
            return [text], []
        segments: list[str] = []
        tool_calls: list[ToolCall] = []
        remaining = text

        while remaining:
            if tool_state["in_tool_call"]:
                end_index = remaining.find(end_token)
                if end_index == -1:
                    tool_state["buffer"] += remaining
                    remaining = ""
                else:
                    tool_state["buffer"] += remaining[:end_index]
                    call = self._parse_tool_call(tool_state)
                    if call is not None:
                        tool_calls.append(call)
                    tool_state["in_tool_call"] = False
                    tool_state["buffer"] = ""
                    remaining = remaining[end_index + len(end_token) :]
            else:
                start_index = remaining.find(start_token)
                if start_index == -1:
                    segments.append(remaining)
                    remaining = ""
                else:
                    prefix = remaining[:start_index]
                    if prefix:
                        segments.append(prefix)
                    tool_state["in_tool_call"] = True
                    remaining = remaining[start_index + len(start_token) :]

        return segments, tool_calls

    def _parse_tool_call(self, tool_state: dict[str, Any]) -> ToolCall | None:
        payload = tool_state["buffer"].strip()
        if not payload:
            return None
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            log.warning("Failed to decode tool call payload: %s", payload)
            return None

        function = parsed.get("function") or {}
        name = parsed.get("name") or function.get("name") or ""
        if not name:
            log.warning("Tool call missing name: %s", payload)
            return None

        args = parsed.get("arguments") or function.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}

        call_id = parsed.get("id") or parsed.get("tool_call_id")
        if not call_id:
            call_id = f"mlx_tool_{tool_state['counter']}"
        tool_state["counter"] += 1

        return ToolCall(
            id=call_id,
            name=name,
            args=args if isinstance(args, dict) else {"value": args},
        )

    async def text_to_speech(
        self,
        text: str,
        model: str,
        voice: str | None = None,
        speed: float = 1.0,
        timeout_s: int | None = None,
        context: Any = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:  # Returns np.ndarray[np.int16]
        """Generate speech audio from text using MLX text-to-speech models with streaming.

        MLX TTS models support true streaming, yielding audio chunks as they're generated.

        Args:
            text: Input text to convert to speech
            model: Model repository ID (e.g., "mlx-community/Kokoro-82M-bf16")
            voice: Voice preset (e.g., "af_heart", "am_adam") - defaults to "af_heart"
            speed: Speech speed multiplier (0.5 to 2.0)
            timeout_s: Optional timeout in seconds
            context: Optional processing context
            **kwargs: Additional MLX TTS parameters (e.g., temperature, language)

        Yields:
            numpy.ndarray: Int16 audio chunks at 24kHz mono

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If generation fails or platform is unsupported
        """
        log.debug(
            f"Generating streaming speech with MLX model: {model}, voice: {voice}, speed: {speed}"
        )

        if sys.platform != "darwin":
            raise RuntimeError("MLX TTS requires macOS (Apple Silicon / MLX)")

        if not text:
            raise ValueError("text must not be empty")

        # Default voice and parameters
        voice = voice or "af_heart"
        speed = max(0.5, min(2.0, speed))

        try:
            # Load TTS model (from cache or filesystem)
            tts_model = await self._load_tts_model(model)

            # Prepare generation parameters - enable streaming
            gen_params = {
                "text": text,
                "speed": speed,
                "stream": True,  # Enable streaming
                "verbose": False,
            }

            # Add voice if the model supports it (Kokoro-style models)
            if voice:
                gen_params["voice"] = voice

            # Add additional kwargs
            if "temperature" in kwargs:
                gen_params["temperature"] = kwargs["temperature"]
            if "lang_code" in kwargs:
                gen_params["lang_code"] = kwargs["lang_code"]
            if "language" in kwargs:
                gen_params["lang_code"] = kwargs["language"]

            # Stream audio chunks
            queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def _stream_audio():
                """Stream audio chunks in background thread."""
                try:
                    assert tts_model.generate is not None
                    for result in tts_model.generate(**gen_params):
                        audio = result.audio
                        if audio is None:
                            continue

                        # Convert MLX array to numpy float32
                        if hasattr(audio, "astype") and hasattr(audio, "numpy"):
                            chunk_float = audio.astype(np.float32).numpy()
                        else:
                            chunk_float = np.asarray(audio, dtype=np.float32)

                        chunk_int16 = np.clip(chunk_float, -1.0, 1.0)
                        chunk_int16 = (chunk_int16 * 32767.0).astype(np.int16)

                        if chunk_float.size > 0:
                            # Put float chunk in queue
                            asyncio.run_coroutine_threadsafe(
                                queue.put(("chunk", chunk_int16)), loop
                            ).result()

                    # Signal completion
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("done", None)), loop
                    ).result()

                except Exception as exc:
                    log.exception(f"MLX TTS streaming error: {exc}")
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("error", exc)), loop
                    ).result()

            # Start streaming thread
            threading.Thread(target=_stream_audio, daemon=True).start()

            while True:
                kind, payload = await queue.get()

                if kind == "chunk":
                    yield payload
                elif kind == "error":
                    raise payload
                elif kind == "done":
                    break

        except Exception as e:
            log.error(f"MLX TTS generation failed: {e}")
            raise RuntimeError(f"MLX TTS generation failed: {str(e)}")

    async def text_to_image(
        self,
        params: TextToImageParams,
        timeout_s: int | None = None,
        context: ProcessingContext | None = None,
        node_id: str | None = None,
    ) -> ImageBytes:
        """Generate an image from a text prompt using MLX.

        Args:
            params: Text-to-image generation parameters
            timeout_s: Optional timeout in seconds
            context: Optional processing context

        Returns:
            Raw image bytes as PNG

        Raises:
            RuntimeError: If MLX is not available or generation fails
        """
        if not params.prompt:
            raise ValueError("Prompt cannot be empty for image generation.")

        try:
            model_id = params.model.id or ""
            model_id_lower = model_id.lower()
            is_z_image = "z-image" in model_id_lower
            is_flux2 = "flux.2" in model_id_lower or "flux2" in model_id_lower
            is_qwen = "qwen-image" in model_id_lower
            is_fibo = "fibo" in model_id_lower

            # Use ModelManager lock for thread-safe model loading
            cache_key = f"{model_id}_{'z-image' if is_z_image else 'flux2' if is_flux2 else 'qwen' if is_qwen else 'fibo' if is_fibo else 'flux'}"
            async with ModelManager.lock_model(cache_key):
                if is_z_image:
                    from mflux.models.common.config import ModelConfig
                    from mflux.models.z_image.variants import ZImage

                    model = ModelManager.get_model(cache_key)
                    if model is None:
                        quantize = 4
                        model_config = (
                            ModelConfig.z_image_turbo()
                            if "turbo" in model_id_lower
                            else ModelConfig.z_image()
                        )
                        model = ZImage(quantize=quantize, model_config=model_config)
                        ModelManager.set_model(cache_key, cache_key, model)
                elif is_flux2:
                    from mflux.models.common.config import ModelConfig
                    from mflux.models.flux2.variants import Flux2Klein

                    model = ModelManager.get_model(cache_key)
                    if model is None:
                        model = Flux2Klein(
                            quantize=4, model_config=ModelConfig.from_name(model_id)
                        )
                        ModelManager.set_model(cache_key, cache_key, model)
                elif is_qwen:
                    from mflux.models.common.config import ModelConfig
                    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

                    model = ModelManager.get_model(cache_key)
                    if model is None:
                        model = QwenImage(
                            quantize=8, model_config=ModelConfig.from_name(model_id)
                        )
                        ModelManager.set_model(cache_key, cache_key, model)
                elif is_fibo:
                    from mflux.models.common.config import ModelConfig
                    from mflux.models.fibo.variants.txt2img.fibo import FIBO

                    model = ModelManager.get_model(cache_key)
                    if model is None:
                        model = FIBO(
                            quantize=4, model_config=ModelConfig.from_name(model_id)
                        )
                        ModelManager.set_model(cache_key, cache_key, model)
                else:
                    model = await load_flux_model(
                        model_id=model_id,
                        quantize=4,
                        task="flux",
                    )

                loop = asyncio.get_running_loop()

                def _generate() -> PIL.Image.Image:
                    if is_fibo:
                        import gc
                        import mlx.core as mx
                        from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM

                        try:
                            json.loads(params.prompt)
                            prompt_value = params.prompt
                        except json.JSONDecodeError:
                            vlm = FiboVLM(quantize=4)
                            try:
                                prompt_value = vlm.generate(
                                    prompt=params.prompt,
                                    seed=params.seed if params.seed is not None else 42,
                                )
                            finally:
                                del vlm
                                gc.collect()
                                mx.clear_cache()
                    else:
                        prompt_value = params.prompt

                    default_steps = (
                        9
                        if "z-image-turbo" in model_id_lower
                        else (
                            50
                            if is_z_image
                            else (
                                4
                                if is_flux2
                                else (
                                    30
                                    if is_qwen
                                    else (
                                        8
                                        if "fibo-lite" in model_id_lower
                                        else 20 if is_fibo else 4
                                    )
                                )
                            )
                        )
                    )
                    generated = model.generate_image(
                        seed=params.seed if params.seed is not None else 0,
                        prompt=prompt_value,
                        num_inference_steps=params.num_inference_steps or default_steps,
                        height=params.height or 1024,
                        width=params.width or 1024,
                        guidance=params.guidance_scale,
                        negative_prompt=(
                            params.negative_prompt if (is_qwen or is_fibo) else None
                        ),
                    )
                    return generated.image if hasattr(generated, "image") else generated

                # Run generation in executor
                pil_image = await loop.run_in_executor(None, _generate)

            # Convert to bytes
            img_buffer = BytesIO()
            pil_image.save(img_buffer, format="PNG")
            image_bytes = img_buffer.getvalue()

            return image_bytes

        except Exception as e:
            log.error(f"MLX text-to-image generation failed: {e}")
            raise RuntimeError(f"MLX text-to-image generation failed: {e}")

    async def image_to_image(
        self,
        image: ImageBytes,
        params: ImageToImageParams,
        timeout_s: int | None = None,
        context: ProcessingContext | None = None,
        node_id: str | None = None,
    ) -> ImageBytes:
        """Transform an image based on a text prompt using MLX.

        Args:
            image: Input image as bytes
            params: Image-to-image generation parameters
            timeout_s: Optional timeout in seconds
            context: Optional processing context

        Returns:
            Raw image bytes as PNG

        Raises:
            RuntimeError: If MLX is not available or generation fails
        """

        model_id = params.model.id or ""
        model_id_lower = model_id.lower()
        is_z_image = "z-image" in model_id_lower
        is_flux2 = "flux.2" in model_id_lower or "flux2" in model_id_lower
        is_seedvr2 = "seedvr2" in model_id_lower
        is_qwen = "qwen-image" in model_id_lower
        is_fibo = "fibo" in model_id_lower

        if not params.prompt and not is_seedvr2:
            raise ValueError("Prompt cannot be empty for image-to-image generation.")

        if is_z_image:
            from mflux.models.common.config import ModelConfig
            from mflux.models.z_image.variants import ZImage

            cache_key = f"{model_id}_z-image"
            model = ModelManager.get_model(cache_key)
            if model is None:
                model_config = (
                    ModelConfig.z_image_turbo()
                    if "turbo" in model_id_lower
                    else ModelConfig.z_image()
                )
                model = ZImage(quantize=4, model_config=model_config)
                ModelManager.set_model(cache_key, cache_key, model)
        elif is_flux2:
            from mflux.models.common.config import ModelConfig
            from mflux.models.flux2.variants import Flux2Klein

            cache_key = f"{model_id}_flux2"
            model = ModelManager.get_model(cache_key)
            if model is None:
                model = Flux2Klein(
                    quantize=4, model_config=ModelConfig.from_name(model_id)
                )
                ModelManager.set_model(cache_key, cache_key, model)
        elif is_seedvr2:
            from mflux.models.common.config import ModelConfig
            from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2

            cache_key = f"{model_id}_seedvr2"
            model = ModelManager.get_model(cache_key)
            if model is None:
                model = SeedVR2(
                    quantize=4, model_config=ModelConfig.from_name(model_id)
                )
                ModelManager.set_model(cache_key, cache_key, model)
        elif is_qwen:
            from mflux.models.common.config import ModelConfig
            from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit

            cache_key = f"{model_id}_qwen-image-edit"
            model = ModelManager.get_model(cache_key)
            if model is None:
                model = QwenImageEdit(
                    quantize=8, model_config=ModelConfig.from_name(model_id)
                )
                ModelManager.set_model(cache_key, cache_key, model)
        elif is_fibo:
            from mflux.models.common.config import ModelConfig
            from mflux.models.fibo.variants.edit.fibo_edit import FIBOEdit

            cache_key = f"{model_id}_fibo-edit"
            model = ModelManager.get_model(cache_key)
            if model is None:
                model = FIBOEdit(
                    quantize=4, model_config=ModelConfig.from_name(model_id)
                )
                ModelManager.set_model(cache_key, cache_key, model)
        else:
            model = await load_flux_model(
                model_id=model_id,
                quantize=4,
                task="flux",
            )

        # Load input image
        base_image = PIL.Image.open(BytesIO(image))

        loop = asyncio.get_running_loop()

        def _generate() -> PIL.Image.Image:
            # Prepare image
            working_image = base_image.convert("RGB")
            target_width = 16 * ((params.target_width or 1024) // 16)
            target_height = 16 * ((params.target_height or 1024) // 16)

            if working_image.size != (target_width, target_height) and not is_seedvr2:
                working_image = working_image.resize(
                    (target_width, target_height), PIL.Image.Resampling.LANCZOS
                )

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image_path = Path(tmp.name)
                working_image.save(image_path)

            try:
                if is_seedvr2:
                    from mflux.utils.scale_factor import ScaleFactor

                    resolution_value = getattr(params, "resolution", None)
                    softness = float(getattr(params, "softness", 0.0) or 0.0)
                    if resolution_value is None:
                        if params.target_width and params.target_height:
                            resolution = min(params.target_width, params.target_height)
                        elif params.target_width:
                            resolution = params.target_width
                        elif params.target_height:
                            resolution = params.target_height
                        else:
                            resolution = 1800
                    elif isinstance(
                        resolution_value, str
                    ) and resolution_value.strip().lower().endswith("x"):
                        resolution = ScaleFactor.parse(resolution_value)
                    else:
                        resolution = int(resolution_value)

                    generated = model.generate_image(
                        seed=params.seed if params.seed is not None else 0,
                        image_path=image_path,
                        resolution=resolution,
                        softness=softness,
                    )
                    return generated.image if hasattr(generated, "image") else generated

                if is_qwen:
                    generated = model.generate_image(
                        seed=params.seed if params.seed is not None else 0,
                        prompt=params.prompt,
                        image_paths=[str(image_path)],
                        num_inference_steps=params.num_inference_steps or 30,
                        height=params.target_height,
                        width=params.target_width,
                        guidance=params.guidance_scale or 2.5,
                        negative_prompt=params.negative_prompt,
                    )
                    return generated.image if hasattr(generated, "image") else generated

                if is_fibo:
                    import gc
                    import mlx.core as mx
                    from PIL import Image as PILImage
                    from mflux.models.fibo.variants.edit.util import FiboEditUtil
                    from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM

                    try:
                        prompt_value = FiboEditUtil.ensure_edit_instruction(
                            params.prompt
                        )
                    except (TypeError, ValueError):
                        vlm = FiboVLM(quantize=4)
                        try:
                            prompt_value = vlm.edit(
                                image=PILImage.open(image_path).convert("RGB"),
                                edit_instruction=params.prompt,
                                use_mask=False,
                                seed=params.seed if params.seed is not None else 42,
                            )
                        finally:
                            del vlm
                            gc.collect()
                            mx.clear_cache()

                    generated = model.generate_image(
                        seed=params.seed if params.seed is not None else 0,
                        prompt=prompt_value,
                        image_path=image_path,
                        num_inference_steps=params.num_inference_steps or 20,
                        height=params.target_height or target_height,
                        width=params.target_width or target_width,
                        guidance=params.guidance_scale
                        or (1.0 if "rmbg" in model_id_lower else 4.0),
                        negative_prompt=params.negative_prompt,
                    )
                    return generated.image if hasattr(generated, "image") else generated

                model_name_lower = model_id_lower
                if is_z_image:
                    default_steps = 9 if "turbo" in model_name_lower else 50
                elif is_flux2:
                    default_steps = 4 if "base" not in model_name_lower else 50
                else:
                    default_steps = 4 if "schnell" in model_name_lower else 8

                generated = model.generate_image(
                    seed=params.seed if params.seed is not None else 0,
                    prompt=params.prompt,
                    num_inference_steps=params.num_inference_steps or default_steps,
                    height=target_height,
                    width=target_width,
                    guidance=params.guidance_scale,
                    image_strength=params.strength or 0.4,
                    image_path=image_path,
                )
                return generated.image if hasattr(generated, "image") else generated
            finally:
                # Clean up temp file
                try:
                    image_path.unlink()
                except FileNotFoundError:
                    pass

        # Run generation in executor
        pil_image = await loop.run_in_executor(None, _generate)

        # Convert to bytes
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format="PNG")
        image_bytes = img_buffer.getvalue()

        self._ensure_usage_counters()
        self.usage["total_requests"] += 1
        self.usage["total_images"] += 1
        self.track_usage(model=model, image_count=1)

        return image_bytes


async def main() -> None:
    """Run a comprehensive demo of the MLX provider.

    - Basic generation: ask the model to say hello
    - Multiple concurrent generations to test serialization lock
    - Optional tool-call round trip with a calculator tool
    """
    from nodetool.workflows.processing_context import ProcessingContext

    class CalculatorTool(Tool):
        name = "calculator"
        description = (
            "Perform basic arithmetic operations (add, subtract, multiply, divide)."
        )
        input_schema = {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform",
                },
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["operation", "a", "b"],
        }

        async def process(  # type: ignore[override]
            self, context: ProcessingContext, params: dict[str, Any]
        ) -> Any:
            operation = params.get("operation")
            a = float(params.get("a", 0))
            b = float(params.get("b", 0))

            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                if b == 0:
                    return {"error": "Cannot divide by zero"}
                result = a / b
            else:
                return {"error": f"Unknown operation: {operation}"}

            return {
                "operation": operation,
                "a": a,
                "b": b,
                "result": result,
            }

    provider = MLXProvider()
    model_name = "Qwen/Qwen3-4B-MLX-4bit"

    print("=" * 70)
    print("TEST 1: Basic single generation")
    print("=" * 70)

    content_parts = []
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Say 'Hello from MLX' in one short sentence."),
    ]

    async for item in provider.generate_messages(
        messages=messages, model=model_name, max_tokens=4096
    ):
        if isinstance(item, Chunk):
            if item.content:
                content_parts.append(item.content)
            if item.done:
                break
    print("Response:", "".join(content_parts))
    print()

    print("=" * 70)
    print("TEST 2: Multiple concurrent generations (serialization test)")
    print("=" * 70)

    async def generate_response(prompt: str, index: int) -> str:
        """Helper to generate a single response."""
        messages = [Message(role="user", content=prompt)]
        parts = []
        print(f"[{index}] Starting generation...")
        async for item in provider.generate_messages(
            messages=messages, model=model_name, max_tokens=100
        ):
            if isinstance(item, Chunk):
                if item.content:
                    parts.append(item.content)
                if item.done:
                    break
        result = "".join(parts)
        print(f"[{index}] Completed: {result[:50]}...")
        return result

    # Launch multiple concurrent requests
    prompts = [
        "What is 2+2? Answer in one sentence.",
        "Name a color. Just one word.",
        "Count from 1 to 3. Just the numbers.",
        "What is the capital of France? One word answer.",
        "Is water wet? Answer yes or no.",
    ]

    print(f"\nLaunching {len(prompts)} concurrent requests...")
    tasks = [generate_response(prompt, i) for i, prompt in enumerate(prompts)]
    results = await asyncio.gather(*tasks)

    print("\n✅ All concurrent requests completed successfully!")
    for i, result in enumerate(results):
        print(f"  [{i}]: {result[:60]}...")
    print()

    print("=" * 70)
    print("TEST 3: Tool calling with multiple rounds")
    print("=" * 70)

    tools: list[Tool] = [CalculatorTool()]
    context = ProcessingContext()
    messages = [
        Message(
            role="user",
            content="Use the calculator to multiply 23 by 17, then tell me the result.",
        ),
    ]

    try:
        tool_calls: list[ToolCall] = []
        content_parts: list[str] = []
        async for item in provider.generate_messages(
            messages=messages, model=model_name, tools=tools, max_tokens=4096
        ):
            if isinstance(item, ToolCall):
                tool_calls.append(item)
            elif isinstance(item, Chunk):
                if item.content:
                    content_parts.append(item.content)
                if item.done:
                    break
        if tool_calls:
            for tc in tool_calls:
                tool = next((t for t in tools if t.name == tc.name), None)
                if tool is None:
                    continue
                print(f"Processing tool call: {tc.name}")
                print(f"  Args: {tc.args}")
                result = await tool.process(context, tc.args or {})
                print(f"  Result: {result}")
                messages.append(
                    Message(
                        role="tool",
                        name=tool.name,
                        tool_call_id=tc.id,
                        content=json.dumps(result),
                    )
                )

            print("\nFinal response:")
            async for item in provider.generate_messages(
                messages=messages, model=model_name, tools=tools, max_tokens=4096
            ):
                if isinstance(item, Chunk):
                    if item.content:
                        print(item.content, end="", flush=True)
                    if item.done:
                        print("\n")
                        break
        else:
            print("No tool calls returned by the model.")
    except Exception as e:  # pragma: no cover - demo convenience
        print(f"Tool call demo failed: {e}")

    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)


async def available_models():
    models = await get_mlx_language_models_from_hf_cache()
    for model in models:
        print(model.id)
    return models


if __name__ == "__main__":
    # asyncio.run(available_models())

    asyncio.run(main())
